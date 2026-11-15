"""One bounded frozen vdM packet followed by native Proton local redesign.

This adapter does not search a vdM library or graft onto a changed backbone.
It accepts a cached manifest/handoff, or a native per-step role contract in
runtime['vdm_pair_roles'] plus runtime['vdm_pair_preview'] containing
structure_path and structure_sha256 for the full complex. The latter must be
the output of build_vdm_pair_roles in the current step, not a JSON role export.
The preview is usable only in its validated source frame (or as that exact
installed pair). Fresh vdM stages can provide fresh frames; stale packets fail.
Coordinates agree to 0.001 Angstrom per component, the precision of a PDB
record. Only a coherent final genotype enters the native queue.

Pair identities, chain/numbering validation and masks belong exclusively to
``vdm_pair_roles.build_vdm_pair_roles``. That separately delivered helper must
be installed before use; unit tests with a stub are not integration evidence.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from numbers import Real
from pathlib import Path

import numpy as np

from astrolabe.core import global_config
from astrolabe.utils.mmcif_ids import int_id_to_str_id
from astrolabe.mutators.proton_pottsmpnn import (
    CANONICAL_AA,
    ProtonPottsMPNNMutator,
    _apply_excluded_amino_acids,
    _segment_site,
    canonical_edits,
)


def _pair_builder():
    try:
        from astrolabe.mutators.vdm_pair_roles import build_vdm_pair_roles
    except ModuleNotFoundError as exc:
        if exc.name != "astrolabe.mutators.vdm_pair_roles":
            raise
        raise RuntimeError(
            "vdm_proton unavailable: install the READY/hash-verified native "
            "astrolabe/mutators/vdm_pair_roles.py helper before proposing."
        ) from exc
    return build_vdm_pair_roles


def _pair_set_builder():
    try:
        from astrolabe.mutators.vdm_pair_set_roles import build_vdm_pair_set_roles
    except ModuleNotFoundError as exc:
        if exc.name != "astrolabe.mutators.vdm_pair_set_roles":
            raise
        raise RuntimeError("vdm_proton native v2 cached conversion awaits vdM owner READY/hash-verified helper.") from exc
    return build_vdm_pair_set_roles


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _checked(path, expected):
    path = Path(path).expanduser().resolve()
    if not expected or _sha(path) != expected:
        raise ValueError(f"vdm_proton SHA256 mismatch: {path}")
    return path


def _one(items, key, value):
    matches = [item for item in items if item.get(key) == value]
    if len(matches) != 1:
        raise ValueError(f"vdm_proton requires one {key}={value!r}; got {len(matches)}")
    return matches[0]


def _sequence(value):
    if not isinstance(value, str) or not value or set(value) - CANONICAL_AA:
        raise ValueError("vdm_proton requires canonical uppercase source/current sequences.")
    return value


def _frame(path):
    """Read coordinates only; scientific role/label mapping stays in the helper.

    CIF uses label chain IDs, like Astrolabe. The local Bio 1.79 parser needs a
    scoped column remap instead of the newer auth_chains=False constructor.
    """
    from Bio.PDB import MMCIFParser, PDBParser
    from Bio.SeqUtils import seq1

    if Path(path).suffix.lower() == ".pdb":
        parser = PDBParser(QUIET=True, PERMISSIVE=False)
    else:
        class LabelParser(MMCIFParser):
            def _build_structure(self, structure_id):
                self._mmcif_dict["_atom_site.auth_asym_id"] = self._mmcif_dict["_atom_site.label_asym_id"]
                return super()._build_structure(structure_id)
        parser = LabelParser(QUIET=True)
    models = list(parser.get_structure("frame", str(path)))
    if len(models) != 1:
        raise ValueError("vdm_proton frozen frame requires exactly one model.")
    result = {}
    for chain in models[0]:
        residues = []
        for residue in chain:
            aa = _sequence(seq1(residue.resname, undef_code="X"))
            if residue.is_disordered():
                raise ValueError("vdm_proton frozen frame does not support alternate conformers.")
            atoms = {atom.name: np.asarray(atom.coord, dtype=float)
                     for atom in residue if atom.element not in {"H", "D"}}
            if not {"N", "CA", "C", "O"} <= atoms.keys():
                raise ValueError("vdm_proton frozen frame requires complete N/CA/C/O backbone.")
            if any(not np.isfinite(x).all() for x in atoms.values()):
                raise ValueError("vdm_proton nonfinite frame coordinates.")
            residues.append((residue.id, aa, atoms))
        result[str(chain.id)] = residues
    return result


def _frame_sequences(frame):
    return {chain: "".join(row[1] for row in rows) for chain, rows in frame.items()}


def _same_frame(left, right, *, binder, roles=()):
    """Compare all heavy atoms except role sidechains, which may be installed."""
    if left.keys() != right.keys():
        raise ValueError("vdm_proton full-complex chain sets differ.")
    for chain in left:
        if len(left[chain]) != len(right[chain]):
            raise ValueError("vdm_proton frame residue counts differ.")
        for pos, (a, b) in enumerate(zip(left[chain], right[chain])):
            role = chain == binder and pos in roles
            if a[0] != b[0] or (not role and a[1] != b[1]):
                raise ValueError("vdm_proton frame residue identities/numbering differ.")
            names = {"N", "CA", "C", "O"} if role else set(a[2])
            if not role and names != set(b[2]):
                raise ValueError("vdm_proton frozen frame non-role atom sets differ.")
            if any(name not in b[2] or not np.allclose(a[2][name], b[2][name], rtol=0, atol=0.001)
                   for name in names):
                raise ValueError(
                    "vdm_proton unsupported stale context: coordinates differ from "
                    "the frozen packet frame (PDB tolerance 0.001 A); dynamic grafting is unavailable."
                )


def _pair_queue(protomer, before, pair):
    """Only the complete native vdM pair may already be pending, never edits
    from another mutator. Return its exact queue for the final atomic merge.
    """
    expected = {}
    for pos, old, new in canonical_edits(before, pair):
        segment, site = _segment_site(protomer, pos)
        expected[(id(global_config.segments[segment]), site)] = (old, new)
    observed, snapshot, seen = {}, {}, set()
    for name, segment in global_config.segments.items():
        if id(segment) in seen:
            continue
        seen.add(id(segment))
        if segment.mutations:
            snapshot[name] = copy.deepcopy(segment.mutations)
        for mutation in segment.mutations or []:
            key = (id(segment), int(mutation[0]))
            identities = tuple(mutation[1:3])
            if key in observed or expected.get(key) != identities:
                raise ValueError("vdm_proton queue contains unrelated or conflicting mutations.")
            observed[key] = identities
    if observed and observed != expected:
        raise ValueError("vdm_proton refuses a partial pending vdM pair.")
    return snapshot


class VdmProtonMutator(ProtonPottsMPNNMutator):
    name = "vdm_proton"
    requires_pre_update_stack = False

    def __init__(self, *, proposal_manifest=None, proposal_id=None, hbond_handoff=None,
                 pairset_handoff=None, pairset_handoff_sha256=None,
                 pairset_parent_id="p10_shard_134_rep33_1",
                 microenvironment_mutations=3, proposal_manifest_sha256=None,
                 hbond_handoff_sha256=None, criteria=None, **kwargs):
        if "pair_count" in kwargs:
            raise ValueError("vdm_proton pair_count is computed from the producer mutation budget, not configurable.")
        if type(microenvironment_mutations) is not int or microenvironment_mutations < 1:
            raise ValueError("vdm_proton microenvironment_mutations must be a positive integer.")
        self.microenvironment_mutations = microenvironment_mutations
        supplied = (proposal_manifest, proposal_id, hbond_handoff)
        if pairset_handoff is not None and any(x is not None for x in supplied):
            raise ValueError("vdm_proton pairset_handoff and legacy manifest/hbond inputs are mutually exclusive.")
        if pairset_handoff is not None and not pairset_handoff_sha256:
            raise ValueError("vdm_proton pairset_handoff requires pairset_handoff_sha256.")
        if any(x is not None for x in supplied) and not all(supplied):
            raise ValueError("vdm_proton cached mode requires proposal_manifest, proposal_id and hbond_handoff together.")
        criteria = copy.deepcopy(dict(criteria or {}))
        temperature = criteria.get("temperature", 0.0)
        if (isinstance(temperature, bool) or not isinstance(temperature, Real)
                or not math.isfinite(temperature) or temperature < 0):
            raise ValueError("vdm_proton criteria.temperature must be a non-negative finite number.")
        # Preserve legacy deterministic default; a positive value explicitly
        # opts into the official backend's sampling without changing its scale.
        criteria["temperature"] = float(temperature)
        required = dict(method="block_descent", backend="potts", seed_source="inverse",
                        infill_scope="neighbourhood", neighbour_k=16,
                        selective_source="potts", record_trajectory=True)
        for key, value in required.items():
            if key in criteria and criteria[key] != value:
                raise ValueError(f"vdm_proton requires criteria.{key}={value!r}.")
        if criteria.get("max_mutations") not in (None, microenvironment_mutations):
            raise ValueError("vdm_proton criteria.max_mutations must match microenvironment_mutations.")
        if criteria.get("explicit_centers") or criteria.get("center_types"):
            raise ValueError("vdm_proton owns explicit His centers; caller centers are unsupported.")
        dep_map = dict(criteria.get("dep_map") or {})
        if dep_map.get("HIS-S", ["HIS-P"]) != ["HIS-P"]:
            raise ValueError("vdm_proton requires HIS-S -> [HIS-P].")
        # Base preparation builds masks before the validated role set is known.
        # This one-centre preflight is replaced below, together with its cap;
        # reserving all pins here would double-count already installed His.
        criteria.update(required, max_mutations=microenvironment_mutations, center_count=1)
        criteria.setdefault("selective", True)
        criteria.setdefault("combined_lambda", 0.3)
        criteria["dep_map"] = {**dep_map, "HIS-S": ["HIS-P"]}
        if kwargs.get("network_only") or (kwargs.get("network_proposal") or {}).get("enabled"):
            raise ValueError("vdm_proton consumes a frozen packet, not a dynamic network search.")
        if kwargs.get("max_total_mutations") is not None:
            raise ValueError("vdm_proton uses a per-call budget, not max_total_mutations.")
        super().__init__(criteria=criteria, **kwargs)
        self.proposal_manifest = Path(proposal_manifest).expanduser().resolve() if proposal_manifest else None
        self.hbond_handoff = Path(hbond_handoff).expanduser().resolve() if hbond_handoff else None
        self.proposal_id = proposal_id
        self.manifest_sha = proposal_manifest_sha256
        self.handoff_sha = hbond_handoff_sha256
        self.pairset_handoff = Path(pairset_handoff).expanduser().resolve() if pairset_handoff else None
        self.pairset_handoff_sha = pairset_handoff_sha256
        self.pairset_parent_id = pairset_parent_id

    def _mutation_budget(self):
        """Bound the WHOLE pending genotype by the global step budget.

        Parent preparation/output must not reserve pins or reject the full
        genotype against this consumer's weighted local allocation. The local
        native cap is resolved separately from n_mutations(), below. Neither
        global mutation_num nor either alias's mutation_weight is changed.
        """
        budget = global_config.mutation_num
        if type(budget) is not int or budget < 1:
            raise ValueError("vdm_proton global mutation_num must be a positive integer.")
        return budget

    def _local_allocation(self, *, live, producer_allocated=None):
        if live:
            allocated = self.n_mutations()
        else:
            if type(producer_allocated) is not int or producer_allocated < 1:
                raise ValueError("vdm_proton cached source allocated mutation budget must be a positive integer.")
            allocated = self._mutation_budget() - producer_allocated
        if allocated < 1:
            raise ValueError(
                f"vdm_proton local allocation {allocated} must leave at least 1 local edit "
                f"(global={self._mutation_budget()}, source={producer_allocated}, live={live})."
            )
        return allocated

    def _local_designable_cap(self, pair_edits, *, live=False, producer_allocated=None):
        actual = len(pair_edits)
        total = self._mutation_budget()
        remaining = total - actual
        if remaining < 1:
            raise ValueError(
                f"vdm_proton complete pair set requires {actual} canonical edits; "
                f"total mutation budget {total} leaves {remaining} "
                "local edits, but at least 1 is required; refusing partial pair set."
            )
        allocated = self._local_allocation(live=live, producer_allocated=producer_allocated)
        return min(self.microenvironment_mutations, remaining, allocated)

    @staticmethod
    def _legacy_producer_budget(features, packet=None):
        # These are existing native producer fields, not a consumer setting.
        packet = packet or {}
        selected = [item["selected_mutation_budget"] for item in (packet, features)
                    if "selected_mutation_budget" in item]
        if selected and any(value != selected[0] for value in selected):
            raise ValueError("vdm_proton cached producer allocated budgets disagree.")
        budget = (selected[0] if selected else packet.get("max_edits", features.get("max_edits")))
        if type(budget) is not int or budget < 2:
            raise ValueError("vdm_proton cached packet requires producer selected_mutation_budget/max_edits >= 2.")
        count, unused = divmod(budget, 2)
        if count != 1:
            raise ValueError(
                f"vdm_proton legacy cached contract provides 1 pair; producer budget {budget} "
                f"requires {count}; refusing incomplete set. Native multi-pair schema awaits owner READY."
            )
        return dict(selected_mutation_budget=budget, requested_pair_count=count,
                    unused_odd_mutations=unused, source="cached_native_producer_input")

    @staticmethod
    def _pair_set_budget(contract):
        # The owner derives cardinality from budget AND site policy, and owns
        # physical endpoint uniqueness/mapping. Consume that complete result;
        # do not derive another count from edits, His centers or endpoint count.
        count = contract.get("pair_count")
        budget = contract.get("mutation_budget")
        unused = contract.get("unused_mutation_budget")
        if (type(count) is not int or count < 1 or type(budget) is not int or budget < 1
                or type(unused) is not int or not 0 <= unused < budget):
            raise ValueError("vdm_proton requires native computed pair_count, mutation_budget and unused_mutation_budget.")
        pairs = contract.get("pairs")
        available = len(pairs) if isinstance(pairs, list) else 0
        if available != count:
            raise ValueError(
                f"vdm_proton incomplete pair set: requested {count}, available {available}; no fallback."
            )
        return dict(selected_mutation_budget=budget, requested_pair_count=count,
                    unused_mutation_budget=unused, source="native_v2_role_contract")

    @staticmethod
    def _fixed_role_sidechains(contract, binder_chain):
        fixed = set()
        if contract["schema"] != "vdm_pair_set_roles_v2":
            return fixed
        for pair in contract["pairs"]:
            # Both lists are the validated owner's native endpoint order.
            for site, raw in zip(pair["sites"], pair["provenance"]["sites"]):
                policy = raw["sidechain_policy"]
                if site["chain_label"] != binder_chain:
                    if policy != "fixed":
                        raise ValueError("vdm_proton target sidechain repack is unsupported; target coordinates must stay fixed.")
                elif policy == "fixed":
                    fixed.add(site["position_0based"])
        return fixed

    def _packet(self):
        if self.manifest_sha:
            _checked(self.proposal_manifest, self.manifest_sha)
        if self.handoff_sha:
            _checked(self.hbond_handoff, self.handoff_sha)
        handoff = json.loads(self.hbond_handoff.read_text())
        if Path(handoff["scoring_input_draft"]).resolve() != self.proposal_manifest:
            raise ValueError("vdm_proton handoff refers to a different proposal manifest.")
        _checked(self.proposal_manifest, handoff["scoring_input_sha256"])
        for path_key, hash_key in (("proposal_packet", "proposal_packet_sha256"),
                                   ("hbond_summary", "hbond_summary_sha256"),
                                   ("raw_hbond_result", "raw_hbond_result_sha256")):
            _checked(handoff[path_key], handoff[hash_key])
        manifest = json.loads(self.proposal_manifest.read_text())
        candidate = _one(manifest["items"], "item_id", self.proposal_id)
        parents = [x for x in manifest["items"] if x.get("role") == "vdm_parent"]
        parent = _one(parents, "parent_id", candidate["parent_id"])
        packet_file = json.loads(Path(handoff["proposal_packet"]).read_text())
        packet = _one(packet_file["packets"], "candidate_id", self.proposal_id)
        support = _one(handoff["candidates"], "candidate_id", self.proposal_id)
        if candidate.get("role") != "vdm_candidate" or packet.get("route") != "vdm_pair":
            raise ValueError("vdm_proton requires a vdM candidate packet.")
        for item in (candidate, parent):
            _sequence(item["sequence"])
            _checked(item["structure_path"], item["structure_sha256"])
            _checked(item["source_structure"], item["source_structure_sha256"])
        if (packet["parent_structure_sha256"] != parent["source_structure_sha256"]
                or packet["binder_sequence"] != candidate["sequence"]
                or Path(packet["template_preview_path"]).resolve() != Path(candidate["source_structure"]).resolve()
                or packet["template_preview_sha256"] != candidate["source_structure_sha256"]
                or packet["pair_positions_1based"] != support["pair_positions_1based"]):
            raise ValueError("vdm_proton inconsistent packet/manifest/handoff identities.")
        original = _checked(packet["original_proposal_path"], packet["original_proposal_sha256"])
        return candidate, parent, packet, json.loads(original.read_text()), support

    def _pairset_packet(self, protomers):
        handoff_path = _checked(self.pairset_handoff, self.pairset_handoff_sha)
        handoff = json.loads(handoff_path.read_text())
        if handoff.get("schema") != "vdm_pairset_handoff" or handoff.get("status") != "READY":
            raise ValueError("vdm_proton pairset handoff is not READY.")
        if self.pairset_parent_id != "p10_shard_134_rep33_1":
            raise ValueError("vdm_proton frozen handoff contains only p10; requested parent unavailable, no fallback (p14 two pairs insufficient).")
        root = Path(__file__).resolve().parents[2]
        for relative, record in handoff["source_snapshot"].items():
            if relative.startswith("astrolabe/"):
                _checked(record["path"], record["sha256"])
                _checked(root / relative, record["sha256"])
        checked = {key: _checked(handoff[key]["path"], handoff[key]["sha256"])
                   for key in ("p10_proposal", "p10_joint_preview", "p10_roles",
                               "original_p10_overlay", "original_p10_effective_config")}
        original = json.loads(checked["p10_proposal"].read_text())
        if len(original["mutation_input"]) != 1:
            raise ValueError("vdm_proton frozen pairset requires one native complex.")
        name, source_features = next(iter(original["mutation_input"].items()))
        features = copy.deepcopy(source_features)
        budget = original["selected_mutation_budget"]
        if (type(budget) is not int or budget < 1
                or original["global_mutation_budget"] != self._mutation_budget()):
            raise ValueError("vdm_proton frozen source allocation/global mutation budget mismatch.")
        if "mutation_budget" in features and features["mutation_budget"] != budget:
            raise ValueError("vdm_proton frozen native and captured allocation disagree.")
        # Explicitly authorized metadata replay: only the captured allocation is
        # added in memory. Raw packet, original masks/policies and edits stay frozen.
        features["mutation_budget"] = budget
        _checked(original["input_structure"], original["input_sha256"])
        if (Path(features["structure_path"]).resolve() != Path(original["input_structure"]).resolve()
                or self.pairset_parent_id not in Path(original["input_structure"]).parts
                or Path(original["template_preview"]).resolve() != checked["p10_joint_preview"]
                or original["template_preview_sha256"] != handoff["p10_joint_preview"]["sha256"]):
            raise ValueError("vdm_proton frozen pairset source/preview/parent identity mismatch.")
        roles = _pair_set_builder()(oligo_name=name, features=features,
                                    telemetry=original["proposal"], protomers=protomers)
        if json.loads(json.dumps(roles)) != json.loads(checked["p10_roles"].read_text()):
            raise ValueError("vdm_proton actual native helper differs from frozen exported roles.")
        trace = dict(handoff=str(handoff_path), handoff_sha256=self.pairset_handoff_sha,
                     proposal=handoff["p10_proposal"], preview=handoff["p10_joint_preview"],
                     roles=handoff["p10_roles"], parent_id=self.pairset_parent_id,
                     budget_replay=dict(source_field="proposal.selected_mutation_budget",
                                        value=budget, added_in_memory="mutation_budget" not in source_features,
                                        raw_proposal_preserved=True))
        return roles, str(checked["p10_joint_preview"]), trace

    def _validate_input_contract(self, protomers, prepared=None):
        self._validate_segment_manager(protomers)
        for name, features in (prepared or {}).items():
            evidence = features["vdm_proton"]
            budget = evidence["budget"]
            if (self._mutation_budget() != budget["total"]
                    or self.n_mutations() != budget["native_consumer_n_mutations"]):
                raise ValueError("vdm_proton global/consumer mutation budget changed after preparation.")
            binder = features["binder_protomer"]
            before, pair = evidence["before_sequence"], evidence["pair_sequence"]
            if str(protomers[binder].sequence) != before:
                raise ValueError("vdm_proton binder changed after proposal preparation.")
            if _pair_queue(protomers[binder], before, pair) != evidence["queued_pair_mutations"]:
                raise ValueError("vdm_proton pending pair queue changed after preparation.")
            for index, proto in enumerate(name):
                chain = int_id_to_str_id(index + 1)
                expected = before if chain == features["binder_chain"] else features["chain_sequences"][chain]
                if str(protomers[proto].sequence) != expected:
                    raise ValueError("vdm_proton context changed after proposal preparation.")

    def prepare_mutation_input(self, protomers, evaluation_scores, runtime_config):
        # A rejected/malformed handoff must not expose the last sibling's roles.
        # Clear before any validation, including the pre-super preview checks.
        if runtime_config is not None:
            runtime_config.pop("network_pair_roles", None)
        self._mutation_budget()
        runtime = runtime_config or {}
        live = "vdm_pair_roles" in runtime
        pairset = not live and self.pairset_handoff is not None
        if live:
            # A native per-step helper result, never a fallback to a cached pair.
            live_roles = runtime["vdm_pair_roles"]
            if live_roles is None:
                raise ValueError("vdm_proton no-match role contract; independent Potts fallback is forbidden.")
            preview_info = runtime.get("vdm_pair_preview")
            if not isinstance(preview_info, dict) or not {"structure_path", "structure_sha256"} <= preview_info.keys():
                raise ValueError("vdm_proton runtime roles require vdm_pair_preview with full-complex structure_path and structure_sha256.")
            preview_path = str(_checked(preview_info["structure_path"], preview_info["structure_sha256"]))
        elif pairset:
            cached_roles, preview_path, pairset_trace = self._pairset_packet(protomers)
        else:
            if self.proposal_manifest is None:
                raise ValueError("vdm_proton requires a cached packet or native runtime vdm_pair_roles and vdm_pair_preview.")
            candidate, parent, packet, original, support = self._packet()
            preview_path = candidate["source_structure"]
        # This is only parent mask/frame preparation. The complete pair's
        # actual edits and weighted local allocation are not known yet. Do
        # not reject a configured local ceiling against an artificial single
        # pin reserve; resolve the real bounded native cap below instead.
        preflight = copy.copy(self)
        preflight.criteria = {**self.criteria, "max_mutations": None}
        prepared = super(VdmProtonMutator, preflight).prepare_mutation_input(
            protomers, evaluation_scores, runtime_config)
        for name, features in prepared.items():
            current = _sequence(features["binder_sequence"])
            chain = features["binder_chain"]
            # Cached mode passes the ACTUAL original packet directly. The role
            # helper owns all parsing and checks; do not reconstruct telemetry.
            if live:
                role_contract = copy.deepcopy(live_roles)
            elif pairset:
                role_contract = copy.deepcopy(cached_roles)
            else:
                native_features = original["mutation_input"][name]
                builder = _pair_set_builder() if "mutation_budget" in native_features else _pair_builder()
                role_contract = builder(oligo_name=name, features=native_features,
                                        telemetry=original["proposal"], protomers=protomers)
            if role_contract is None:
                raise ValueError("vdm_proton no-match role contract; independent Potts fallback is forbidden.")
            schema = role_contract.get("schema")
            if schema == "vdm_pair_set_roles_v2":
                producer_budget = self._pair_set_budget(role_contract)
            elif schema == "vdm_pair_roles_v1":
                producer_budget = (dict(requested_pair_count=1, source="legacy_single_pair_contract")
                                   if live else self._legacy_producer_budget(original["mutation_input"][name], packet))
            else:
                raise ValueError("vdm_proton unsupported role contract schema; expected native v1 or v2.")
            binder = role_contract["binder"]
            if (role_contract["oligo_name"] != name or binder["protomer"] != features["binder_protomer"]
                    or binder["chain_label"] != chain):
                raise ValueError("vdm_proton role contract belongs to a different active binder/complex.")
            roles = role_contract["network_pair_roles"][features["binder_protomer"]]
            if (not roles or any(type(p) is not int or p < 0 or p >= len(current)
                                                 or aa not in {"H", "K", "R"} for p, aa in roles.items())):
                raise ValueError("vdm_proton requires native integer role positions with H/K/R identities.")
            original_fixed = set(features["fixed_positions"])
            if any(current[pos] != aa for pos, aa in roles.items() if pos in original_fixed):
                raise ValueError("vdm_proton pair role changes an immutable/selector identity.")
            pair_sequence = _sequence(binder["designed_sequence"])
            parent_sequence = _sequence(binder["original_sequence"])
            if current not in {parent_sequence, pair_sequence}:
                raise ValueError("vdm_proton unsupported stale sequence outside source parent/pair genotype.")
            if not live and not pairset and (pair_sequence != candidate["sequence"] or parent_sequence != parent["sequence"]):
                raise ValueError("vdm_proton helper and cached candidate/parent identities disagree.")
            if not live and not pairset and sorted(p+1 for p in roles) != sorted(packet["pair_positions_1based"]):
                raise ValueError("vdm_proton helper roles disagree with packet/handoff 1-based positions.")
            pair_edits = canonical_edits(current, pair_sequence, fixed_positions=features["fixed_positions"])
            if any(pos not in roles for pos, _, _ in pair_edits):
                raise ValueError("vdm_proton vdM network edits outside the complete protected role set.")
            producer_allocated = producer_budget.get("selected_mutation_budget")
            if producer_allocated is not None and len(pair_edits) > producer_allocated:
                raise ValueError("vdm_proton pair edits exceed the captured producer allocation.")
            local_allocation = self._local_allocation(live=live, producer_allocated=producer_allocated)
            if live and producer_allocated is not None and producer_allocated + local_allocation > self._mutation_budget():
                raise ValueError("vdm_proton source and consumer allocations exceed the global mutation budget.")
            local_cap = self._local_designable_cap(pair_edits, live=live, producer_allocated=producer_allocated)
            source = role_contract["source"]
            source_path = str(_checked(source["structure_path"], source["structure_sha256"]))
            if not live and not pairset and source["structure_sha256"] != parent["source_structure_sha256"]:
                raise ValueError("vdm_proton helper source differs from the frozen parent frame.")
            preview = _frame(preview_path)
            parent_frame, current_frame = _frame(source_path), _frame(features["pdb_path"])
            expected_preview = {**features["chain_sequences"], chain: pair_sequence}
            if (_frame_sequences(preview) != expected_preview
                    or _frame_sequences(current_frame) != features["chain_sequences"]
                    or _frame_sequences(parent_frame) != {**expected_preview, chain: parent_sequence}):
                raise ValueError("vdm_proton full-complex source/current sequences differ.")
            # seqs+ fixes canonical identity. Binder role sidechains can use
            # the owner's explicit repack preview even at immutable positions;
            # coordinate clamps come from its sidechain policy, not seqs+.
            fixed_sidechains = self._fixed_role_sidechains(role_contract, chain)
            _same_frame(parent_frame, preview, binder=chain, roles=set(roles) - fixed_sidechains)
            _same_frame(current_frame, parent_frame if current == parent_sequence else preview, binder=chain)
            centers, his_positions = [], []
            for center in role_contract["his_centers"]:
                if center["chain_label"] != chain or center["requires_target_his_s_backend_support"]:
                    raise ValueError("vdm_proton native binder-only backend cannot pin a target His endpoint to HIS-S.")
                pos = center["position_0based"]
                if (center["requested_microstate"] != "HIS-S" or center["insertion_code"]
                        or preview[chain][pos][0][1] != center["pdb_number"]):
                    raise ValueError("vdm_proton native His center microstate/res_id is unsupported or mismatched.")
                his_positions.append(pos)
                centers.append(dict(res_id=center["pdb_number"], protonation_type="HIS-S"))
            if (not centers or len(his_positions) != len(set(his_positions))
                    or set(his_positions) != {p for p, aa in roles.items() if aa == "H"}):
                raise ValueError("vdm_proton requires every His endpoint as a native HIS-S center.")
            residue_numbers = [row[0][1] for row in preview[chain]]
            if len(set(residue_numbers)) != len(residue_numbers):
                raise ValueError("vdm_proton native res_id centers cannot represent duplicated PDB numbers/insertion codes.")
            # Consume the helper's Potts mask and narrow it by the CURRENT
            # selector. Native pins, rather than fixed_positions, protect His.
            fixed = sorted((set(binder["potts_fixed_positions_0based"])
                            | set(features["fixed_positions"])) - set(his_positions))
            criteria = copy.deepcopy(features["criteria"])
            criteria.update(explicit_centers=centers, center_count=len(centers),
                            center_types=[], max_mutations=local_cap)
            criteria = _apply_excluded_amino_acids(criteria, global_config.exclude_AA)
            evidence = dict(
                mode="native_runtime" if live else "cached_pairset_handoff" if pairset else "cached_packet",
                role_contract=role_contract, current_structure=features["pdb_path"],
                current_structure_sha256=_sha(features["pdb_path"]),
                source_structure=preview_path, source_structure_sha256=_sha(preview_path),
                before_sequence=current, pair_sequence=pair_sequence,
                queued_pair_mutations=_pair_queue(protomers[features["binder_protomer"]], current, pair_sequence),
                original_fixed_positions=features["fixed_positions"], pair_edits=pair_edits,
                budget=dict(total=self._mutation_budget(), vdm_actual_edits=len(pair_edits),
                            configured_local_max=self.microenvironment_mutations,
                            consumer_allocated=local_allocation, native_consumer_n_mutations=self.n_mutations(),
                            producer=producer_budget,
                            native_local_designable=local_cap),
                pair_count=producer_budget["requested_pair_count"],
                state_audit=dict(his_count=len(centers), max_supported_his=4,
                                 status="unsupported_more_than_4_his" if len(centers) > 4
                                 else "separate_audit_not_run"),
                native_seed_base=getattr(self.runner, "seed", None),
                limitation="validated source frame only; no dynamic grafting; preview is not an AF3-realized network",
            )
            if pairset:
                evidence["pairset_handoff"] = pairset_trace
            elif not live:
                evidence.update(proposal_id=self.proposal_id, proposal_manifest=str(self.proposal_manifest),
                                proposal_manifest_sha256=_sha(self.proposal_manifest),
                                hbond_handoff=str(self.hbond_handoff), hbond_handoff_sha256=_sha(self.hbond_handoff),
                                packet=packet, hbond_support=support)
            features.update(pdb_path=preview_path, binder_sequence=pair_sequence,
                            chain_sequences=expected_preview, fixed_positions=fixed,
                            criteria=criteria, pin_budget=len(centers), vdm_proton=evidence)
        return prepared

    def compute_mutation(self, mutation_input_features):
        # Existing runner ignores additive evidence keys; proposal_only captures
        # the complete feature_dict, including the budget and packet evidence.
        return super().compute_mutation(mutation_input_features)

    def prepare_mutation_output(self, mutation_output, protomers, evaluation_scores, runtime_config):
        if len(mutation_output or {}) != 1:
            raise ValueError("vdm_proton requires one complete joint output.")
        self._validate_input_contract(protomers, {
            name: output["feature_dict"] for name, output in mutation_output.items()
        })
        normalized = copy.deepcopy(mutation_output)
        audits = {}
        for name, output in normalized.items():
            features, selected = output["feature_dict"], output["selected"]
            evidence = features["vdm_proton"]
            before, pair = evidence["before_sequence"], evidence["pair_sequence"]
            after = _sequence(selected["canonical_sequence"])
            total = canonical_edits(before, after, fixed_positions=evidence["original_fixed_positions"])
            roles = evidence["role_contract"]["network_pair_roles"][features["binder_protomer"]]
            if any(after[pos] != aa for pos, aa in roles.items()):
                raise ValueError("vdm_proton selected genotype changed a protected pair role.")
            tokens = selected.get("extended_tokens")
            if isinstance(tokens, str):
                tokens = tokens.split()
            if tokens is None or len(tokens) != len(pair) or any(
                    tokens[pos] != "HIS-S" for pos, aa in roles.items() if aa == "H"):
                raise ValueError("vdm_proton selected tokens must retain HIS-S at every His endpoint.")
            micro = canonical_edits(pair, after, fixed_positions=features["fixed_positions"])
            if (len(micro) > evidence["budget"]["native_local_designable"]
                    or len(total) > min(evidence["budget"]["total"], self._mutation_budget())):
                raise ValueError("vdm_proton selected joint genotype exceeds local/total budget; refusing truncation.")
            audits[name] = dict(input=copy.deepcopy(features), before_sequence=before,
                                pair_sequence=pair, after_sequence=after,
                                pair_edits=evidence["pair_edits"], microenvironment_edits=micro,
                                native_output=copy.deepcopy(output))
            # Native queue sees the complete genotype relative to the accepted
            # sequence. The vdM intermediate is never precommitted.
            # Keep the complete producer queue intact and append only local
            # edits. Composite cached calls without a pending set queue the
            # whole genotype. Both paths are checked relative to the parent.
            pending = bool(evidence["queued_pair_mutations"])
            features["binder_sequence"] = pair if pending else before
            features["fixed_positions"] = (features["fixed_positions"] if pending
                                           else evidence["original_fixed_positions"])
        super().prepare_mutation_output(normalized, protomers, evaluation_scores, runtime_config)
        role_runtime = {}
        for name, audit in audits.items():
            self.last_mutation_telemetry["oligos"][name]["vdm_proton"] = audit
            role_runtime.update(audit["input"]["vdm_proton"]["role_contract"]["network_pair_roles"])
        self.last_mutation_telemetry["network_pair_roles"] = role_runtime
        if runtime_config is not None:
            runtime_config["network_pair_roles"] = role_runtime
        return {"vdm_proton": self.last_mutation_telemetry}
