"""Native binder-only vdM pairs, optionally against a fixed ``anchor``.

Anchor positions follow 1-based polymer order. Repacking samples natural
same-identity sidechains for a local preview; only sequence edits are queued.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import logging
import os
import uuid
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.SeqUtils import seq1

from astrolabe.design.proposal_rejection import ProposalInfeasible
from astrolabe.core import global_config
from astrolabe.core.binding_site import parse_binding_site, parse_chain_positions
from astrolabe.core.protomer_config_parser import ProtomerTypes
from astrolabe.mutators.base import Mutator
from astrolabe.mutators.capabilities import PROTEIN_ONLY, MutatorCapabilities
from astrolabe.mutators.proton_pottsmpnn import (
    CANONICAL_AA, _mutable_positions, _segment_site, canonical_edits,
    tokens_for_excluded_aas, TOKEN_TO_CANONICAL_AA,
)
from astrolabe.mutators.utils import prepare_exclude_sites
from astrolabe.utils.mmcif_ids import int_id_to_str_id

logger = logging.getLogger(__name__)


def _integer(value, label, *, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"vdm_pair {label} must be a nonnegative integer.")
    if maximum is not None and value > maximum:
        raise ValueError(f"vdm_pair {label} must be <= {maximum} for a pair proposal.")
    return int(value)


def _sequence(value, label):
    if not isinstance(value, str) or not value or set(value) - CANONICAL_AA:
        raise ValueError(f"vdm_pair {label} must be a nonempty canonical uppercase sequence.")
    return value


def _load_proposer():
    from astrolabe.mutators.vdm_geometry import propose_pair_including_existing
    return propose_pair_including_existing


def _site_map_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"vdm_pair site_map duplicate key: {key!r}.")
        result[key] = value
    return result


def _read_site_map(path):
    """Read and hash the same bytes; reject duplicate keys at every level."""
    content = path.read_bytes()
    data = json.loads(content, object_pairs_hook=_site_map_object)
    if not isinstance(data, dict) or not data:
        raise ValueError("vdm_pair site_map must be a nonempty SHA256-keyed JSON object.")
    for digest, entry in data.items():
        if (len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("vdm_pair site_map keys must be lowercase sequence SHA256 digests.")
        if not isinstance(entry, dict) or set(entry) != {"site_1", "site_2"}:
            raise ValueError("vdm_pair site_map entry requires exactly site_1 and site_2.")
        positions = []
        for site in entry.values():
            if not isinstance(site, dict) or set(site) != {"position", "residue_type"}:
                raise ValueError("vdm_pair site_map site requires exactly position and residue_type.")
            pos = site["position"]
            if isinstance(pos, bool) or not isinstance(pos, int) or pos < 1:
                raise ValueError("vdm_pair site_map position must be a positive 1-based integer.")
            if site["residue_type"] != "H":
                raise ValueError("vdm_pair site_map currently requires H/H endpoints.")
            positions.append(pos)
        if positions[0] == positions[1]:
            raise ValueError("vdm_pair site_map endpoints must be distinct.")
    return data, hashlib.sha256(content).hexdigest()


def _sha256(value):
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _literal_lineage(proto):
    """Derive identity from native startup seqs, never from a visited branch.

    Restrict mapped runs to explicit literal segments. Resolving aliases,
    random segments or files against the *current* global segments would turn
    a dynamic sequence back into a purported immutable parent identity.
    """
    config = proto.config
    if not isinstance(config, str) or not config.split():
        raise ValueError("vdm_pair site_map requires literal startup seqs for lineage.")
    tokens = config.split()
    root, fixed, layout = [], [], []
    offset = 0
    if (len(tokens) != len(proto.components) or len(tokens) != len(proto.components_slices) or
            len(set(proto.components)) != len(tokens) or
            [name for name in global_config.segments if name in proto.components] != list(proto.components)):
        raise ValueError("vdm_pair site_map lineage segment layout mismatch.")
    for token, component, span in zip(tokens, proto.components, proto.components_slices):
        is_fixed = token.startswith("+")
        sequence = token[1:] if is_fixed else token
        if not sequence or set(sequence) - CANONICAL_AA:
            raise ValueError("NEED_DECISION: vdM site_map lineage supports only literal startup seqs, with native + protection.")
        segment = global_config.segments[component]
        stop = offset + len(sequence)
        if (span.start != offset or span.stop != stop or span.step is not None or
                len(segment.sequence) != len(sequence) or segment.is_fixed != is_fixed or
                global_config.segment_manager.segments.get(component) is not segment):
            raise ValueError("vdm_pair site_map lineage segment layout/protection mismatch.")
        root.append(sequence)
        fixed.extend(range(offset, stop) if is_fixed else [])
        layout.append([component, offset, stop, is_fixed])
        offset = stop
    root = "".join(root)
    current = _sequence(str(proto.sequence), "lineage current sequence")
    mask = np.asarray(proto.mutable_mask, dtype=bool)
    if len(current) != len(root) or mask.shape != (len(root),):
        raise ValueError("vdm_pair site_map lineage length mismatch.")
    if any(mask[p] or current[p] != root[p] for p in fixed):
        raise ValueError("vdm_pair site_map lineage fixed seqs+ identity/protection mismatch.")
    return root, {
        "root_sequence_sha256": _sha256(root),
        "seqs_sha256": _sha256(config),
        "length": len(root), "fixed_positions": fixed, "segment_layout": layout,
        "native_mutable_positions": np.flatnonzero(mask).tolist(),
        "current_sequence_sha256": _sha256(current),
        "root_edits": [{"position_one_based": p + 1, "from": a, "to": b}
                       for p, (a, b) in enumerate(zip(root, current)) if a != b],
    }


def _structure_sequences(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"vdm_pair structure does not exist: {path}")
    author_ids = None
    if path.suffix.lower() == ".pdb":
        parser = PDBParser(QUIET=True, PERMISSIVE=False)
    elif path.suffix.lower() in {".cif", ".mmcif"}:
        columns = MMCIF2Dict(str(path))
        author_ids = {}
        for label, author in zip(columns["_atom_site.label_asym_id"],
                                 columns["_atom_site.auth_asym_id"]):
            author_ids.setdefault(label, set()).add(author)
        # Biopython 1.79 in the existing runtime predates auth_chains=False.
        # Scope the column remap to this parser instance, never the source file.
        class LabelMMCIFParser(MMCIFParser):
            def _build_structure(self, structure_id):
                self._mmcif_dict["_atom_site.auth_asym_id"] = self._mmcif_dict["_atom_site.label_asym_id"]
                return super()._build_structure(structure_id)

        parser = LabelMMCIFParser(QUIET=True)
    else:
        raise ValueError("vdm_pair requires a PDB or mmCIF evaluator structure.")
    structure = parser.get_structure("vdm", str(path))
    models = list(structure)
    if len(models) != 1:
        raise ValueError("vdm_pair requires exactly one structure model.")
    sequences, numbering = {}, {}
    for chain in models[0]:
        residues, ids = [], []
        for residue in chain:
            aa = seq1(residue.resname, undef_code="X")
            if residue.id[0] == "W":
                continue
            if aa not in CANONICAL_AA:
                if residue.id[0] == " " or "CA" in residue:
                    raise ValueError("vdm_pair structure contains a noncanonical protein residue.")
                continue
            if residue.is_disordered() == 2:
                raise ValueError("vdm_pair cannot map disordered residue identities.")
            residues.append(aa)
            ids.append([int(residue.id[1]), residue.id[2].strip()])
        if residues:
            sequences[str(chain.id)] = "".join(residues)
            numbering[str(chain.id)] = ids
    chain_map = {chain: chain for chain in sequences}
    if author_ids is not None:
        for chain in sequences:
            if len(author_ids.get(chain, ())) != 1:
                raise ValueError("NEED_DECISION: ambiguous mmCIF label-to-author chain mapping.")
            chain_map[chain] = next(iter(author_ids[chain]))
        if len(set(chain_map.values())) != len(chain_map):
            raise ValueError("NEED_DECISION: merged mmCIF author chains cannot identify one binder.")
    return sequences, numbering, chain_map


def _no_op_status(status):
    status = status.lower().replace("-", "_").replace(" ", "_")
    return status.startswith(("no_op", "noop", "no_match", "no_candidate", "no_pair"))


def _resolve_anchor_chain(value, chains, chain_map):
    """Resolve native chains/unique aliases without guessing across namespaces."""
    if value.startswith("chain:"):
        matches = {value.split(":", 1)[1]} & set(chain_map)
    elif value.startswith("protomer:"):
        matches = {c for c, p in chains if p == value.split(":", 1)[1]}
    else:
        matches = {c for c, p in chains if value in (c, p, chain_map[c])}
    if len(matches) != 1:
        raise ValueError(f"vdm_pair anchor chain {value!r} is missing or ambiguous.")
    return next(iter(matches))


class VdmPairMutator(Mutator):
    name = "vdm_pair"
    capabilities = MutatorCapabilities(
        supported_types=PROTEIN_ONLY, needs_structure=True,
        outputs_probabilities=False, outputs_scores=False,
        supports_transition_blending=False, supports_shared_selection_policy=False,
        supports_initialization_from_structure=False,
    )

    def mutate(self, protomer, evaluation_scores=None, runtime_config=None):
        # Never leave a prior step's pair set usable after no-match or failure.
        if self.site_map_path is not None:
            self.last_mutation_telemetry = {"mutator": self.name, "oligos": {}}
        if runtime_config is not None:
            runtime_config["vdm_pair_roles"] = None
            runtime_config["vdm_pair_preview"] = None
        return super().mutate(protomer, evaluation_scores, runtime_config)

    def __init__(self, *, library_path, binder_chain=None, position_selector=None,
                 reference_sequence=None, max_total_mutations=None, max_edits=None,
                 preserve_disulfides=False, protect_gly_pro=False, parameters=None,
                 anchor=None, anchor_residue=None, repack_anchor=False,
                 site_1=None, site_2=None, save_repack_preview=True, aa_bias=None,
                 partner_types=None, site_map=None):
        if not isinstance(library_path, (str, Path)) or not str(library_path).strip():
            raise ValueError("vdm_pair requires library_path.")
        self.library_path = str(Path(library_path).expanduser().resolve())
        self.binder_chain = None if binder_chain is None else str(binder_chain)
        self.position_selector = position_selector
        self.max_edits = (None if max_edits is None else
                          _integer(max_edits, "max_edits"))
        self.reference_sequence = (None if reference_sequence is None else
                                   _sequence(reference_sequence, "reference_sequence"))
        self.max_total_mutations = (None if max_total_mutations is None else
                                   _integer(max_total_mutations, "max_total_mutations"))
        if (self.reference_sequence is None) != (self.max_total_mutations is None):
            raise ValueError("vdm_pair reference_sequence and max_total_mutations must be provided together.")
        if aa_bias not in (None, {}, []):
            raise ValueError("vdm_pair does not support nonempty aa_bias.")
        for label, value in (("preserve_disulfides", preserve_disulfides),
                             ("protect_gly_pro", protect_gly_pro),
                             ("repack_anchor", repack_anchor)):
            if not isinstance(value, bool):
                raise ValueError(f"vdm_pair {label} must be boolean.")
        self.preserve_disulfides = preserve_disulfides
        self.protect_gly_pro = protect_gly_pro
        if parameters is not None and not isinstance(parameters, Mapping):
            raise ValueError("vdm_pair parameters must be a mapping.")
        self.parameters = copy.deepcopy(dict(parameters or {}))
        if partner_types is not None:
            if 'partner_types' in self.parameters:
                raise ValueError('vdm_pair partner_types specified twice.')
            values = [partner_types] if isinstance(partner_types, str) else list(partner_types)
            values = [str(a).upper() for a in values]
            values = [seq1(a.title(), undef_code='X') if len(a) == 3 else a for a in values]
            if not values or any(a not in {'H', 'K', 'R', 'D', 'E'} for a in values):
                raise ValueError('vdm_pair partner_types must contain H, K, R, D or E.')
            self.parameters['partner_types'] = values
        self.anchor = None
        if anchor is not None:
            if not isinstance(anchor, str):
                raise ValueError("vdm_pair anchor must be a chain or chain:position string.")
            self.anchor = "binding_site" if anchor == "binding_site" else parse_chain_positions(anchor)
            if self.anchor != "binding_site" and len(self.anchor.positions) > 1:
                raise ValueError("vdm_pair anchor accepts one position or a whole chain.")
            if site_1 is not None or site_2 is not None:
                raise ValueError("vdm_pair anchor cannot be combined with legacy site fields.")
        elif anchor_residue is not None or repack_anchor:
            raise ValueError("vdm_pair anchor_residue/repack_anchor requires anchor.")
        self.anchor_residue = None
        if anchor_residue is not None:
            aa = str(anchor_residue).upper()
            aa = seq1(aa.title(), undef_code="X") if len(aa) == 3 else aa
            if len(aa) != 1 or aa not in CANONICAL_AA:
                raise ValueError("vdm_pair anchor_residue must be a canonical one- or three-letter identity.")
            self.anchor_residue = aa
        self.repack_anchor = repack_anchor
        self.sites = [copy.deepcopy(site_1), copy.deepcopy(site_2)]
        self.site_map_path = None
        if site_map is not None:
            if not isinstance(site_map, (str, Path)) or not str(site_map).strip():
                raise ValueError("vdm_pair site_map must be a nonempty JSON file path.")
            if anchor is not None or any(s is not None for s in self.sites):
                raise ValueError("vdm_pair site_map cannot be combined with anchor or site_1/site_2.")
            self.site_map_path = Path(site_map).expanduser().absolute()
            _, self.site_map_sha256 = _read_site_map(self.site_map_path)
        if not isinstance(save_repack_preview, bool):
            raise ValueError("save_repack_preview must be boolean.")
        self.save_repack_preview = save_repack_preview
        self.preview_dir = None
        self.last_mutation_telemetry = {"mutator": self.name, "oligos": {}}

    def _mapped_sites(self, sequence, proto):
        data, digest = _read_site_map(self.site_map_path)
        if digest != self.site_map_sha256:
            raise ValueError("vdm_pair site_map file changed after initialization.")
        root, lineage = _literal_lineage(proto)
        if str(proto.sequence) != sequence:
            raise ValueError("vdm_pair site_map lineage current sequence mismatch.")
        key = lineage["root_sequence_sha256"]
        if key not in data:
            raise ValueError(f"vdm_pair site_map has no exact startup parent sequence SHA256 key: {key}.")
        sites = [copy.deepcopy(data[key][s]) for s in ("site_1", "site_2")]
        if any(s["position"] > len(sequence) for s in sites):
            raise ValueError("vdm_pair site_map position outside current binder sequence.")
        if any(root[s["position"] - 1] == "H" and sequence[s["position"] - 1] != "H" for s in sites):
            raise ValueError("vdm_pair site_map lineage lost an accepted H/H endpoint identity.")
        source = {"kind": "site_map", "path": str(self.site_map_path),
                  "file_sha256": digest, "sequence_sha256": _sha256(sequence),
                  "lookup_sequence_sha256": key, "binder_protomer": proto.name,
                  "lineage": lineage,
                  "position_indexing": "1-based binder polymer order", "sites": sites}
        return sites, source

    def _site_inputs(self, name, features):
        # Pin current AF3 coordinates as well as sequence. A branch can have
        # the same sequence but a different pose; previews are never reused.
        targets = {}
        for p in dict.fromkeys(name):
            if p == features["binder_protomer"]:
                continue
            proto = global_config.protomers[p]
            root, identity = _literal_lineage(proto)
            if (str(proto.sequence) != root or np.asarray(proto.mutable_mask).any() or
                    str(proto.mutated_sequence) != root):
                raise ValueError("vdm_pair site_map target lineage/protection mismatch.")
            targets[p] = identity
        library = Path(self.library_path)
        return {"complex": name, "native_chain": features["native_chain"],
                "binder_chain": features["binder_chain"], "residue_ids": features["residue_ids"],
                "structure_path": features["structure_path"],
                "structure_sha256": hashlib.sha256(Path(features["structure_path"]).read_bytes()).hexdigest(),
                "library_path": self.library_path,
                "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest() if library.is_file() else None,
                "targets": targets}

    def _check_site_source(self, features):
        if self.site_map_path is not None:
            proto = global_config.protomers[features["binder_protomer"]]
            sites, source = self._mapped_sites(features["sequence"], proto)
            if features.get("site_source") != source:
                raise ValueError("vdm_pair site_map source changed after preparation.")
            expected_sites = [{"chain": features["binder_chain"], "position": s["position"],
                               "residue_type": ["H"], "sequence": "mutable", "sidechain": "template"}
                              for s in sites]
            if features["sites"] != expected_sites:
                raise ValueError("vdm_pair site_map prepared sites mismatch.")
            inputs = features["site_inputs"]
            if self._site_inputs(inputs["complex"], features) != inputs:
                raise ValueError("vdm_pair site_map structure/library/target input changed after preparation.")

    def prepare_mutation_input(self, protomers, evaluation_scores, runtime_config):
        runtime = runtime_config or {}
        ignored = set(runtime.get("ignored_complexes") or []) | set(runtime.get("negative_design_complexes") or [])
        names = [name for name in evaluation_scores.names if name not in ignored]
        if len(names) != 1:
            raise ValueError(f"vdm_pair requires exactly one active complex; got {names!r}.")
        name = names[0]
        active = evaluation_scores.filter_by_oligomer(names)
        all_protomers = global_config.protomers
        if any(p not in all_protomers for p in name):
            raise ValueError("vdm_pair active complex references an unknown protomer.")
        chains = [(int_id_to_str_id(i + 1), p) for i, p in enumerate(name)]
        if any(all_protomers[p].type != ProtomerTypes.PROTEIN for _, p in chains):
            raise ValueError("vdm_pair supports protein-only complexes.")
        excluded = prepare_exclude_sites(self.position_selector, active, protomers,
            runtime_config=runtime_config, mutator_alias=getattr(self, "alias", self.name),
            fallback_to_all_mutable_if_selector_empty=False)
        masks = {p: _mutable_positions(all_protomers[p], excluded.get(p, []))
                 for _, p in chains if p in protomers}
        candidates = [c for c, p in chains if p in masks and masks[p].any()]
        chain = self.binder_chain
        if chain is None:
            # An empty selector stays empty. Use only an unambiguous underlying
            # binder identity to return a no-op, never restore its mutable mask.
            if not candidates:
                candidates = [c for c, p in chains if p in masks and
                              np.asarray(all_protomers[p].mutable_mask).any()]
                if not candidates and len(chains) == 1 and chains[0][1] in masks:
                    candidates = [chains[0][0]]
            if len(candidates) != 1:
                raise ValueError("vdm_pair cannot infer a unique binder_chain from mutable masks.")
            chain = candidates[0]
        if chain not in dict(chains):
            raise ValueError("vdm_pair binder_chain must identify a native physical chain.")
        if any(c != chain for c, p in chains if p in masks and masks[p].any()):
            raise ValueError("vdm_pair only binder sequence may be mutable; fix the other chain masks.")
        binder_name = dict(chains)[chain]
        if binder_name not in protomers:
            raise ValueError("vdm_pair binder is not a mutable protomer.")
        if sum(p == binder_name for _, p in chains) != 1:
            raise ValueError("vdm_pair does not support tied physical binder chains.")
        proto = all_protomers[binder_name]
        if any(len(tuple(tied)) > 1 for tied in proto.tied_positions) or proto.ptm_dict:
            raise ValueError("vdm_pair does not support binder tied positions or PTMs.")
        mask = masks[binder_name]
        sequence = _sequence(str(proto.sequence), "binder sequence")
        if len(mask) != len(sequence):
            raise ValueError("vdm_pair mutable mask length mismatch.")
        if self.site_map_path is not None:
            _literal_lineage(proto)
        for i, aa in enumerate(sequence):
            if (self.preserve_disulfides and aa == "C") or (self.protect_gly_pro and aa in "GP"):
                mask[i] = False
        occurrences = Counter(id(global_config.segments[s]) for p in all_protomers.values() for s in p.components)
        for i in np.flatnonzero(mask):
            segment_name, _ = _segment_site(proto, int(i))
            if occurrences[id(global_config.segments[segment_name])] != 1:
                raise ValueError("vdm_pair cannot mutate shared or aliased segments.")
        structure_path = active.complex_paths.get(name)
        if not structure_path:
            raise FileNotFoundError("vdm_pair active complex has no structure.")
        structure_sequences, numbering, chain_map = _structure_sequences(structure_path)
        expected = {c: str(all_protomers[p].sequence) for c, p in chains}
        if structure_sequences != expected:
            raise ValueError(f"vdm_pair structure chain sequences do not match configured protomers: {list(expected)!r}")
        reference = self.reference_sequence if self.reference_sequence is not None else sequence
        total_budget = self.max_total_mutations if self.max_total_mutations is not None else len(sequence)
        if len(reference) != len(sequence):
            raise ValueError("vdm_pair reference_sequence length mismatch.")
        if sum(a != b for a, b in zip(sequence, reference)) > total_budget:
            raise ValueError("vdm_pair current binder exceeds max_total_mutations.")
        target_chains = [c for c, _ in chains if c != chain]
        sites = self.sites
        site_source = None
        if self.site_map_path is not None:
            sites, site_source = self._mapped_sites(sequence, proto)
            if any(not mask[s["position"] - 1] for s in sites):
                raise ProposalInfeasible(
                    "vdm_site_map_endpoint_not_mutable",
                    "vdm_pair site_map endpoint is fixed or excluded from the mutable mask.",
                )
        if self.anchor is not None:
            anchor = self.anchor
            if anchor == "binding_site":
                try:
                    binding_site = global_config.binding_site
                except RuntimeError:
                    binding_site = None
                binding_site = parse_binding_site(binding_site, oligo_name=name)
                if binding_site is None:
                    raise ValueError(f"vdm_pair anchor=binding_site has no binding_site for {name!r}.")
                anchor = binding_site.target
            if len(anchor.positions) > 1:
                raise ValueError("vdm_pair anchor supports one target position or a whole chain; multiple positions are unsupported.")
            anchor_chain = _resolve_anchor_chain(anchor.chain, chains, chain_map)
            sites = [None, {"chain": anchor_chain,
                "position": anchor.positions[0] if anchor.positions else None,
                "residue_type": self.anchor_residue, "sequence": "fixed",
                "sidechain": "repack" if self.repack_anchor else "fixed"}]
        normalized_sites = []
        for raw in sites:
            if raw is None:
                raw = {}
            if not isinstance(raw, Mapping):
                raise ValueError("site_1 and site_2 must be mappings.")
            raw = dict(raw)
            unknown = set(raw) - {"chain", "position", "pdb_residue", "residue_type", "sequence", "sidechain"}
            if unknown:
                raise ValueError(f"unknown vdM site fields: {sorted(unknown)!r}")
            site_chain = str(raw.get("chain") or chain)
            if site_chain == "binder":
                site_chain = chain
            elif site_chain == "target":
                if len(target_chains) != 1:
                    raise ValueError("chain=target requires exactly one target chain; use explicit chain ID.")
                site_chain = target_chains[0]
            if site_chain not in chain_map:
                raise ValueError(f"vdM site chain {site_chain!r} is absent.")
            seq_policy = str(raw.get("sequence") or ("mutable" if site_chain == chain else "fixed"))
            if seq_policy not in ("mutable", "fixed"):
                raise ValueError("site sequence must be mutable or fixed.")
            if site_chain != chain and seq_policy != "fixed":
                raise ValueError("target sequence must be fixed; only binder may be mutated.")
            sidechain = str(raw.get("sidechain") or ("template" if seq_policy == "mutable" else "fixed"))
            if sidechain not in ("fixed", "repack", "template"):
                raise ValueError("site sidechain must be fixed, repack, or template.")
            if seq_policy == "fixed" and sidechain == "template":
                raise ValueError("a fixed-sequence site must use sidechain fixed or repack.")
            if seq_policy == "mutable" and sidechain != "template":
                raise ValueError("a mutable-sequence site requires sidechain template.")
            pos = raw.get("position")
            pdb_pos = raw.get("pdb_residue")
            if pos is not None and pdb_pos is not None:
                raise ValueError("use position or pdb_residue for a site, not both.")
            if pdb_pos is not None:
                import re
                match = re.fullmatch(r"(-?\d+)([A-Za-z]?)", str(pdb_pos).strip())
                if not match:
                    raise ValueError("pdb_residue must be an integer optionally followed by an insertion code.")
                target_id = [int(match.group(1)), match.group(2)]
                matches = [i+1 for i, rid in enumerate(numbering[site_chain]) if rid == target_id]
                if len(matches) != 1:
                    raise ValueError(f"pdb_residue {pdb_pos!r} is missing or ambiguous on chain {site_chain!r}.")
                pos = matches[0]
            if pos is not None and (isinstance(pos, bool) or not isinstance(pos, (int, np.integer)) or not 1 <= int(pos) <= len(expected[site_chain])):
                raise ValueError("site position must be a 1-based in-range sequence position.")
            residue_type = raw.get("residue_type")
            if residue_type is not None:
                residue_type = [residue_type] if isinstance(residue_type, str) else list(residue_type)
                residue_type = [str(a).upper() for a in residue_type]
                residue_type = [seq1(a.title(), undef_code="X") if len(a) == 3 else a for a in residue_type]
                if not residue_type or any(len(a) != 1 or a not in CANONICAL_AA for a in residue_type):
                    raise ValueError("site residue_type must be a one-letter amino acid or list.")
                if pos is not None and seq_policy == "fixed" and expected[site_chain][int(pos)-1] not in residue_type:
                    raise ValueError("fixed target residue_type conflicts with the input structure sequence.")
            normalized_sites.append({"chain": chain_map[site_chain], "position": None if pos is None else int(pos),
                "residue_type": residue_type, "sequence": seq_policy, "sidechain": sidechain})
        excluded_aas = sorted({TOKEN_TO_CANONICAL_AA[t] for t in
                               tokens_for_excluded_aas(global_config.exclude_AA)
                               if t in TOKEN_TO_CANONICAL_AA})
        parameters = copy.deepcopy(self.parameters)
        parameters.setdefault("protect_gly_pro", self.protect_gly_pro)
        parameters.setdefault("preserve_disulfides", self.preserve_disulfides)
        native_budget = self.n_mutations()
        mutable_per_pair = sum(s["sequence"] == "mutable" for s in normalized_sites)
        if not mutable_per_pair:
            raise ValueError("vdm_pair requires at least one mutable site per pair.")
        pair_count, unused_budget = divmod(native_budget, mutable_per_pair)
        if not pair_count:
            raise ProposalInfeasible(
                "vdm_pair_budget_insufficient",
                f"vdm_pair resolved native mutation budget is insufficient for one pair (need {mutable_per_pair}).",
            )
        if pair_count > 1 and any(s["position"] is not None for s in normalized_sites):
            raise ValueError("vdm_pair budget requests multiple disjoint pairs but a site specifies one fixed position; supply a search over distinct positions or lower the native budget.")
        pair_budget = mutable_per_pair * pair_count
        if self.max_edits is not None and self.max_edits > pair_budget:
            raise ValueError("vdm_pair configured max_edits exceeds native mutation budget; refusing truncation.")
        edit_budget = pair_budget if self.max_edits is None else self.max_edits
        if site_source is not None:
            if pair_count != 1 or edit_budget == 0:
                raise ValueError("vdm_pair site_map requires one live pair with a nonzero edit budget.")
            if "H" in excluded_aas:
                raise ValueError("vdm_pair site_map H endpoints conflict with excluded amino acids.")
            logger.info("vdm_pair resolved site source: %s", site_source)
        features = {"structure_path": str(Path(structure_path).resolve()), "binder_chain": chain_map[chain],
            "native_chain": chain, "sequence": sequence, "binder_sequence": sequence,
            "mutable_positions": np.flatnonzero(mask).tolist(), "sites": normalized_sites,
            "library_path": self.library_path, "reference_sequence": reference,
            "max_total_mutations": total_budget,
            "max_edits": edit_budget, "pair_count": pair_count, "mutation_budget": native_budget,
            "unused_mutation_budget": unused_budget, "parameters": parameters,
            "binder_protomer": binder_name, "excluded_aas": excluded_aas, "residue_ids": numbering[chain],
            **({"site_source": site_source} if site_source is not None else {})}
        if site_source is not None:
            features["site_inputs"] = self._site_inputs(name, features)
        return {name: features}

    def compute_mutation(self, mutation_input_features):
        outputs = {}
        for name, features in mutation_input_features.items():
            # Keep an immutable-to-the-provider snapshot for output validation.
            features = copy.deepcopy(features)
            self._check_site_source(features)
            if not features["mutable_positions"] or features["max_edits"] == 0:
                result = {"designed_sequence": features["sequence"],
                          "status": "no_op_no_mutable_positions" if not features["mutable_positions"]
                          else "no_op_zero_budget", "metadata": {}}
            else:
                keys = ("structure_path", "binder_chain", "binder_sequence", "mutable_positions",
                        "library_path", "reference_sequence", "max_total_mutations",
                        "max_edits", "sites", "parameters", "excluded_aas")
                if features["pair_count"] == 1:
                    proposer = _load_proposer()
                else:
                    from astrolabe.mutators.vdm_geometry import propose_pair_set
                    proposer = propose_pair_set
                kwargs = copy.deepcopy({k: features[k] for k in keys})
                if features["pair_count"] != 1:
                    kwargs["pair_count"] = features["pair_count"]
                try:
                    inspect.signature(proposer).bind(**kwargs)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"NEED_DECISION: propose_pair signature cannot accept the agreed keyword API: {exc}") from exc
                result = proposer(**kwargs)
            self._check_site_source(features)
            if not isinstance(result, dict) or not {"designed_sequence", "status", "metadata"} <= result.keys():
                raise ValueError("NEED_DECISION: propose_pair must return dict with designed_sequence, status, metadata.")
            if self.site_map_path is not None and result["status"] != "proposed":
                logger.error("vdm_pair mapped search failed: %s", {
                    "status": result["status"], "site_source": features["site_source"],
                    "site_inputs": features["site_inputs"], "metadata": result["metadata"]})
                if result["status"] == "no_match":
                    raise ProposalInfeasible(
                        "vdm_site_map_no_match",
                        f"vdm_pair site_map live endpoint search failed: {result['status']}.",
                    )
                raise ValueError(f"vdm_pair site_map live endpoint search failed: {result['status']}.")
            if features["pair_count"] > 1 and result["status"] != "proposed":
                if result["status"] == "insufficient_pairs":
                    raise ProposalInfeasible(
                        "vdm_pair_set_insufficient_pairs",
                        f"vdm_pair insufficient complete disjoint pairs: {result['status']}; required {features['pair_count']}.",
                    )
                raise ValueError(f"vdm_pair insufficient complete disjoint pairs: {result['status']}; required {features['pair_count']}.")
            outputs[name] = {"result": copy.deepcopy(result), "features": features}
        return outputs

    def prepare_mutation_output(self, mutation_output, protomers, evaluation_scores, runtime_config):
        if len(mutation_output) != 1:
            raise ValueError("vdm_pair output must contain exactly one joint proposal.")
        name, output = next(iter(mutation_output.items()))
        features, result = output["features"], output["result"]
        self._check_site_source(features)
        proto = protomers[features["binder_protomer"]]
        current = features["sequence"]
        if str(proto.sequence) != current:
            raise ValueError("vdm_pair stale proposal: current binder changed after preparation.")
        designed = _sequence(result["designed_sequence"], "designed_sequence")
        if not isinstance(result["status"], str) or not result["status"].strip():
            raise ValueError("vdm_pair status must be a nonempty string.")
        if not isinstance(result["metadata"], Mapping):
            raise ValueError("vdm_pair metadata must be a mapping.")
        status = result["status"]
        if self.n_mutations() != features["mutation_budget"]:
            raise ValueError("vdm_pair resolved native mutation budget changed during proposal.")
        if features["pair_count"] > 1:
            metadata = result["metadata"]
            pair_list = metadata.get("pairs")
            if status != "proposed" or metadata.get("pair_count") != features["pair_count"] or not isinstance(pair_list, list) or len(pair_list) != features["pair_count"]:
                raise ValueError("vdm_pair incomplete pair set; refusing partial mutation.")
            endpoints = []
            for pair in pair_list:
                pair_sites = pair.get("sites", [])
                if len(pair_sites) != 2:
                    raise ValueError("vdm_pair pair must have exactly two sites.")
                endpoints.extend((s["chain"], s["position_zero_based"]) for s in pair_sites)
            if len(set(endpoints)) != len(endpoints):
                raise ValueError("vdm_pair pair endpoints overlap.")
        if status.lower().startswith(("error", "fail", "invalid", "reject")):
            raise ValueError(f"vdm_pair scientific engine returned failure status: {status}")
        mutable = set(features["mutable_positions"])
        edits = canonical_edits(current, designed,
                                fixed_positions=sorted(set(range(len(current))) - mutable),
                                label="vdm_pair")
        if self.site_map_path is not None:
            requested = features["site_source"]["sites"]
            endpoints = {s["position"] - 1 for s in requested}
            if any(not proto.mutable_mask[p] for p in endpoints):
                raise ValueError("vdm_pair site_map endpoint became fixed before output.")
            actual = result["metadata"].get("sites")
            expected_sites = [(features["binder_chain"], s["position"] - 1, "H") for s in requested]
            if (status != "proposed" or not isinstance(actual, list) or
                    any(not isinstance(s, Mapping) for s in actual) or
                    [(s.get("chain"), s.get("position_zero_based"), s.get("residue_type"))
                     for s in actual] != expected_sites or
                    any(designed[p] != "H" for p in endpoints) or
                    any(p not in endpoints for p, _, _ in edits)):
                raise ValueError("vdm_pair site_map live result does not match requested H/H endpoints.")
        if edits and _no_op_status(status):
            raise ValueError("vdm_pair no-op status returned a changed sequence.")
        if len(edits) > min(features["max_edits"], self.n_mutations()):
            raise ValueError("vdm_pair joint proposal exceeds max_edits/native mutation budget; refusing truncation.")
        if any(to_aa in features["excluded_aas"] for _, _, to_aa in edits):
            raise ValueError("vdm_pair joint proposal introduces an excluded amino acid.")
        manager = global_config.segment_manager
        pending, records, claimed = [], [], set()
        for pos, old, new in edits:
            segment_name, site = _segment_site(proto, pos)
            segment = global_config.segments[segment_name]
            if manager.segments.get(segment_name) is not segment:
                raise ValueError("vdm_pair segment_manager identity mismatch.")
            key = (id(segment), site)
            if key in claimed or any(int(m[0]) == site for m in segment.mutations or []):
                raise ValueError("vdm_pair proposed site already has a queued mutation.")
            if segment.is_fixed or site in (segment.ptm_dict or {}) or segment.sequence[site] != old:
                raise ValueError("vdm_pair protected/stale segment site.")
            claimed.add(key)
            pending.append((segment_name, (site, old, new, 0.0, 0.0)))
            records.append({"position": pos, "residue_id": features["residue_ids"][pos],
                            "segment": segment_name, "segment_site": site,
                            "from": old, "to": new})
        # Count physical queued sites once, including aliases and other mutators.
        queued = {(id(s), int(m[0])) for s in manager.unique_segments.values()
                  for m in s.mutations or []}
        if pending and len(queued) + len(pending) > int(global_config.mutation_num):
            raise ValueError("vdm_pair combined queue exceeds native global mutation budget.")
        projected = list(designed)
        for component, span in zip(proto.components, proto.components_slices):
            segment = global_config.segments[component]
            for site, old, new, *_ in segment.mutations or []:
                if not (0 <= site < len(segment.sequence)) or segment.sequence[site] != old:
                    raise ValueError("vdm_pair existing binder queue is stale or invalid.")
                projected[span.start + site] = new
        if features["pair_count"] > 1 or self.site_map_path is not None:
            if "".join(projected) != designed:
                raise ValueError("vdm_pair cannot mix pair set with unrelated queued binder edits.")
            if any(str(p.mutated_sequence) != str(p.sequence)
                   for key, p in protomers.items() if key != features["binder_protomer"]):
                raise ValueError("vdm_pair cannot mix pair set with queued target edits.")
        total = None
        if self.max_total_mutations is not None:
            total = sum(a != b for a, b in zip(projected, self.reference_sequence))
            if total > self.max_total_mutations:
                raise ValueError("vdm_pair combined proposal exceeds original-parent max_total_mutations.")
        result_metadata = copy.deepcopy(dict(result["metadata"]))
        if self.site_map_path is not None:
            result_metadata["site_source"] = copy.deepcopy(features["site_source"])
            result_metadata["site_inputs"] = copy.deepcopy(features["site_inputs"])
        result_metadata["pair_count"] = features["pair_count"]
        result_metadata["mutation_budget"] = features["mutation_budget"]
        result_metadata["unused_mutation_budget"] = features["unused_mutation_budget"]
        telemetry = {"binder_chain": features["native_chain"], "structure_chain": features["binder_chain"],
                     "status": status if status == "proposed" or edits or _no_op_status(status) else "no_op",
                     "engine_status": status, "n_edits": len(edits), "edits": records,
                     "designed_sequence": designed, "projected_sequence": "".join(projected),
                     "max_edits": features["max_edits"], "total_mutations": total,
                     "repack_preview_path": None, "metadata": result_metadata}
        roles = None
        if telemetry["status"] == "proposed":
            from astrolabe.mutators.vdm_pair_set_roles import build_vdm_pair_set_roles
            roles = build_vdm_pair_set_roles(oligo_name=name, features=features,
                                              telemetry=telemetry, protomers=global_config.protomers)
        # All validation precedes the one native write. Roll back only touched
        # queues if a manager implementation raises or fails to retain the pair.
        snapshots = {id(manager.segments[s]): (manager.segments[s],
                     copy.deepcopy(manager.segments[s].mutations)) for s, _ in pending}
        preview_path = None
        preview_sha = None
        temporary_paths = []
        published_paths = []
        try:
            needs_runtime_preview = roles is not None and runtime_config is not None
            needs_repack_preview = pending and self.save_repack_preview and any(
                p.get("sidechain_policy") == "repack"
                for p in result["metadata"].get("placed_sidechains", []))
            if needs_runtime_preview or needs_repack_preview:
                if self.preview_dir is None:
                    raise ValueError("vdm_pair repack preview has no configured raw output directory.")
                self.preview_dir.mkdir(parents=True, exist_ok=True)
                token = uuid.uuid4().hex
                preview = self.preview_dir / f"{token}.cif"
                audit = self.preview_dir / f"{token}.json"
                temporary_preview = self.preview_dir / f".{token}.tmp.cif"
                temporary_audit = self.preview_dir / f".{token}.tmp.json"
                temporary_paths = [temporary_preview, temporary_audit]
                from astrolabe.mutators import vdm_geometry as module
                if module is None or not callable(getattr(module, "write_preview", None)):
                    raise ValueError("vdm_pair geometry module has no coordinate preview exporter.")
                module.write_preview(features["structure_path"], result, temporary_preview)
                temporary_audit.write_text(json.dumps({"structure_path":features["structure_path"],
                    "result":result, "scope":"template-based repack preview; not automatically passed to AF3",
                    **({"site_source": features["site_source"], "site_inputs": features["site_inputs"]}
                       if self.site_map_path is not None else {})},
                    indent=2, allow_nan=False)+'\n')
            self._check_site_source(features)
            if pending:
                manager.add_mutations(pending, len(pending))
                if any(m not in (manager.segments[s].mutations or []) for s, m in pending):
                    raise ValueError("vdm_pair manager did not retain the complete joint proposal.")
            if temporary_paths:
                os.replace(temporary_preview, preview); published_paths.append(preview)
                os.replace(temporary_audit, audit); published_paths.append(audit)
                preview_path = str(preview)
                preview_sha = hashlib.sha256(preview.read_bytes()).hexdigest()
        except BaseException:
            for segment, mutations in snapshots.values():
                segment.mutations = mutations
            for path in temporary_paths + published_paths:
                path.unlink(missing_ok=True)
            raise
        telemetry["repack_preview_path"] = preview_path
        if runtime_config is not None and roles is not None:
            if preview_path is None:
                raise ValueError("vdm_pair matched pair set has no joint preview.")
            runtime_config["vdm_pair_roles"] = roles
            runtime_config["vdm_pair_preview"] = {
                "structure_path": preview_path,
                "structure_sha256": preview_sha,
            }
        self.last_mutation_telemetry = {"mutator": self.name, "oligos": {name: telemetry}}
        logger.info("vdm_pair[%s] status=%s engine_status=%s n_edits=%d",
                    name, telemetry["status"], status, len(edits))
        if edits and any(p.get("sidechain_policy") == "repack"
                         for p in result["metadata"].get("placed_sidechains", [])):
            logger.warning("vdm_pair[%s] repacked sidechain is a proposal preview, not a fixed input to AF3; preview=%s",
                           name, preview_path or "disabled")
        return {"vdm_pair": self.last_mutation_telemetry}
