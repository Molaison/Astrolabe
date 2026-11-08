"""Joint sequence/structure proposals from accepted complexes."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import tempfile

from astrolabe.core import global_config
from astrolabe.mutators.state_protocol import (
    ScientificState, digest, fields, natural, file_reference, validate_reference,
)
from astrolabe.core.protomer_config_parser import ProtomerTypes
from astrolabe.mutators.base import Mutator
from astrolabe.mutators.capabilities import MutatorCapabilities, PROTEIN_ONLY
from astrolabe.mutators.rfd4_types import (
    AtomRef, ChainSpec, ConditionPlan, GenerationRequest, GenerationResult,
    HBondCondition, ResidueRef,
)
from astrolabe.mutators.utils import prepare_exclude_sites
from astrolabe.per_residue_scores import combine_per_proto_mutator_scores
from astrolabe.utils.mmcif_ids import int_id_to_str_id


AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


def structure_sequences(path: str | Path, chains: tuple[ChainSpec, ...] = ()) -> dict[str, str]:
    """Read protein sequences and explicitly declared single-CCD ligand chains."""
    from Bio.PDB import PDBParser
    from Bio.SeqUtils import seq1
    path = Path(path)
    ligands = {c.chain: c.sequence for c in chains if c.kind == "ligand"}
    if path.suffix.lower() in {".cif", ".mmcif"}:
        from Bio.PDB.MMCIF2Dict import MMCIF2Dict
        record = MMCIF2Dict(str(path))
        if any(group == "HETATM" and residue not in {"HOH", "WAT"}
               and ligands.get(chain) != residue
               for group, residue, chain in zip(record.get("_atom_site.group_PDB", ()),
                                                record.get("_atom_site.label_comp_id", ()),
                                                record.get("_atom_site.label_asym_id", ()))):
            raise ValueError("RFD4 does not support nonprotein chemical components")
        if any(kind in {"covale", "disulf", "modres"}
               for kind in record.get("_struct_conn.conn_type_id", ())):
            raise ValueError("RFD4 does not support covalent or disulfide constraints")
        names = record["_atom_site.label_comp_id"]
        count = len(names)
        rows = zip(record["_atom_site.group_PDB"], record["_atom_site.label_asym_id"],
                   record["_atom_site.label_seq_id"], names,
                   record.get("_atom_site.label_alt_id", ["."] * count),
                   record.get("_atom_site.pdbx_PDB_model_num", ["1"] * count),
                   record.get("_atom_site.auth_seq_id", ["."] * count), strict=True)
        residues, models = {}, set()
        for group, chain, position, name, alternate, model, author_position in rows:
            if name in {"HOH", "WAT"}:
                continue
            if chain in ligands:
                if group != "HETATM" or name != ligands[chain]:
                    raise ValueError("RFD4 ligand chain must contain its declared CCD component only")
                position, token = author_position, name
            else:
                token = seq1(name, undef_code="?")
                if group != "ATOM" or token not in AMINO_ACIDS or position in {".", "?"}:
                    raise ValueError("RFD4 requires canonical unmodified protein residues")
            if alternate not in {".", "?"}:
                raise ValueError("RFD4 requires unambiguous residue/atom identities")
            models.add(model)
            key = (chain, position)
            if key in residues and residues[key] != token:
                raise ValueError("RFD4 has ambiguous physical residue identities")
            residues[key] = token
        if len(models) != 1:
            raise ValueError("RFD4 requires exactly one structural model")
        sequences = {}
        for (chain, _), token in residues.items():
            if chain in ligands and chain in sequences:
                raise ValueError("RFD4 supports exactly one ligand residue per physical chain")
            sequences[chain] = sequences.get(chain, "") + token
        if not set(ligands) <= set(sequences):
            raise ValueError("RFD4 structure is missing a declared ligand chain")
        return sequences
    else:
        if any(line.startswith(("LINK  ", "SSBOND")) for line in path.read_text().splitlines()):
            raise ValueError("RFD4 does not support covalent or disulfide constraints")
        parser = PDBParser(QUIET=True)
    structure = parser.get_structure("accepted", str(path))
    if len(structure) != 1:
        raise ValueError("RFD4 requires exactly one structural model")
    sequences = {}
    for chain in structure[0]:
        residues = [residue for residue in chain if residue.id[0] != "W"]
        if not residues:
            continue
        sequence = ""
        if chain.id in ligands:
            if (len(residues) != 1 or residues[0].resname != ligands[chain.id]
                    or residues[0].id[0] == " " or residues[0].is_disordered()):
                raise ValueError("RFD4 requires one unambiguous declared CCD ligand residue")
            sequences[str(chain.id)] = residues[0].resname
            continue
        for residue in residues:
            token = seq1(residue.resname, undef_code="?")
            if residue.id[0] != " " or token not in AMINO_ACIDS:
                raise ValueError("RFD4 requires canonical unmodified protein residues")
            if residue.is_disordered():
                raise ValueError("RFD4 requires unambiguous residue/atom identities")
            sequence += token
        sequences[str(chain.id)] = sequence
    if not set(ligands) <= set(sequences):
        raise ValueError("RFD4 structure is missing a declared ligand chain")
    return sequences


def _request_from_record(record: dict, structure_path: str) -> GenerationRequest:
    plan = record["plan"]
    return GenerationRequest(
        structure_path=structure_path, assembly=record["assembly"],
        chains=tuple(ChainSpec(**chain) for chain in record["chains"]),
        plan=ConditionPlan(
            released=tuple(ResidueRef(**site) for site in plan["released"]),
            fixed_coordinates=tuple(ResidueRef(**site) for site in plan["fixed_coordinates"]),
            hbonds=tuple(HBondCondition(AtomRef(**item["atom"]), item["role"])
                         for item in plan["hbonds"]),
            excluded_amino_acids=tuple(plan["excluded_amino_acids"]),
            feedback_json=plan["feedback_json"],
            reconstruction=plan.get("reconstruction"),
        ), seed=record["seed"],
    )


class RFD4Mutator(Mutator):
    name = "rfd4"
    requires_exclusive_mutation = True
    capabilities = MutatorCapabilities(supported_types=PROTEIN_ONLY, needs_structure=True)

    def __init__(self, *, assembly, binder_chains, target_chains, python_bin,
                 source_dir, model_dir, cache_dir, generation=None,
                 max_fixed_atom_rmsd=0.5, request_timeout=1800, feedback=None,
                 position_selector=None, mutation_weight=1.0, seed=0):
        from astrolabe.mutators.rfd4_backend import RFD4Backend
        if not isinstance(assembly, str) or not assembly:
            raise ValueError("RFD4 assembly must be explicit")
        self.assembly = assembly
        self.binder_chains = tuple(str(chain) for chain in binder_chains)
        self.target_chains = tuple(str(chain) for chain in target_chains)
        groups = self.binder_chains + self.target_chains
        if not self.binder_chains or not self.target_chains or len(set(groups)) != len(groups):
            raise ValueError("RFD4 binder_chains and target_chains must be nonempty and disjoint")
        if feedback is not None and not hasattr(feedback, "items"):
            raise TypeError("RFD4 feedback must be an optional mapping")
        self.feedback = dict(feedback or {})
        unknown = set(self.feedback) - {"enabled", "chemistry", "retention", "fixed_plan_path",
                                       "repair_unsatisfied_attempts", "reconstruction"}
        if unknown:
            raise ValueError(f"Unknown RFD4 feedback options: {sorted(unknown)}")
        proposal_feedback = self.feedback.get("repair_unsatisfied_attempts", False)
        if not isinstance(proposal_feedback, bool):
            raise ValueError("repair_unsatisfied_attempts must be boolean")
        if proposal_feedback and (not self.feedback.get("enabled")
                                  or not ((self.feedback.get("retention") or {}).get("local_repair")
                                          or self.feedback.get("reconstruction"))
                                  or self.feedback.get("fixed_plan_path")):
            raise ValueError("Proposal feedback requires enabled local repair without a fixed plan")
        self._failed_hbond_attempts = Counter()
        self._hbond_attempt_history = []
        self._observed_requests = {}
        # The ordinary CLI represents an absent selector with the literal
        # expression "None", while programmatic construction uses Python None.
        if isinstance(position_selector, str) and position_selector.strip() == "None":
            position_selector = None
        self.position_selector = position_selector
        self.mutation_weight = float(mutation_weight)
        if not math.isfinite(self.mutation_weight) or self.mutation_weight < 0:
            raise ValueError("RFD4 mutation_weight must be finite and nonnegative")
        self.seed = int(seed)
        self.backend = RFD4Backend(
            python_bin=python_bin, source_dir=source_dir, model_dir=model_dir,
            cache_dir=cache_dir, generation=generation,
            max_fixed_atom_rmsd=max_fixed_atom_rmsd, request_timeout=request_timeout,
        )
        if self.feedback.get("reconstruction") is not None:
            from .rfd4_reconstruction import reconstruction_bounds
            reconstruction_bounds(dict(self.feedback["reconstruction"]))
            if (not self.feedback.get("enabled") or self.feedback.get("fixed_plan_path")
                    or self.backend.generation.get("mode") != "inpaint" or position_selector is not None):
                raise ValueError("Variable reconstruction requires dynamic feedback, inpaint and no position selector")

    def declare_variable_topology(self):
        """Declare the bounded logical variable before the design System exists."""
        config = self.feedback.get("reconstruction")
        if config is None:
            return
        from .rfd4_reconstruction import reconstruction_bounds
        bounds = reconstruction_bounds(dict(config))
        if len(self.binder_chains) != 1 or len(set(self.assembly)) != len(self.assembly):
            raise ValueError("Variable reconstruction requires one nonrepeated protein binder")
        uses = Counter(id(global_config.segments[component])
                       for proto in global_config.protomers.values() for component in proto.components)
        binder = None
        for index, name in enumerate(self.assembly, 1):
            cid, proto = int_id_to_str_id(index), global_config.protomers[name]
            if cid in self.binder_chains:
                if (proto.type != ProtomerTypes.PROTEIN or len(proto.components) != 1
                        or proto.ptm_dict or proto.locked_positions or getattr(proto, "_homology", None)
                        or not all(proto.mutable_mask)):
                    raise ValueError("Variable binder must be one fully mutable unmodified segment without homology")
                segment = global_config.segments[proto.components[0]]
                if uses[id(segment)] != 1 or segment.is_fixed:
                    raise ValueError("Variable binder cannot share a segment or fix sequence positions")
                if not bounds[0] <= len(segment.sequence) <= bounds[1]:
                    raise ValueError("Accepted binder length is outside the declared reconstruction bounds")
                if getattr(segment, "length_bounds", None) not in (None, bounds):
                    raise ValueError("Conflicting variable-length declarations")
                binder = segment
            elif cid not in self.target_chains or proto.type != ProtomerTypes.CCD:
                raise ValueError("Variable reconstruction supports only fixed CCD ligand targets")
        if binder is None:
            raise ValueError("Variable reconstruction binder chain is absent")
        binder.length_bounds = bounds

    @staticmethod
    def _validate_boundary(runtime_config):
        config = runtime_config or {}
        if config.get("mutator_stack_pre_update") or config.get("updated_sites"):
            raise ValueError("RFD4 cannot run on preupdated or stacked mutations")
        if global_config.segment_manager._pre_update_snapshot is not None:
            raise ValueError("RFD4 cannot run on preupdated mutations")
        for segment in global_config.segments.values():
            if segment.mutations or getattr(segment, "pending_sequence", None) is not None:
                raise ValueError("RFD4 cannot run on pending mutation stacks")
        for key in ("aa_bias", "bias_AA", "AA_bias", "enforce_mutation_num",
                    "entropy_cutoff", "transition_consistency_gate_context"):
            if config.get(key):
                raise ValueError(f"RFD4 does not support probability-dependent option {key}")
        diagnostics = config.get("state_conflict_diagnostics")
        if diagnostics and (not hasattr(diagnostics, "get") or diagnostics.get("enabled", False)):
            raise ValueError("RFD4 does not provide probability-based state conflict diagnostics")
        for key in ("covalent_bonds", "cyclization"):
            if global_config.build_runtime_context().get_const(key, None):
                raise ValueError(f"RFD4 does not support {key}")

    def prepare_generation(self, protomers, evaluation_scores, runtime_config=None):
        from astrolabe.mutators.rfd4_feedback import analyze_interface, plan_conditions
        self._validate_boundary(runtime_config)
        self.declare_variable_topology()
        all_protomers = global_config.protomers
        chains = []
        for index, name in enumerate(self.assembly):
            if name not in all_protomers:
                raise ValueError(f"Unknown RFD4 protomer {name!r}")
            proto = all_protomers[name]
            physical_id = int_id_to_str_id(index + 1)
            if proto.type == ProtomerTypes.CCD:
                if (physical_id not in self.target_chains or not proto.sequence
                        or "," in proto.sequence or not proto.sequence.isalnum()):
                    raise ValueError("RFD4 supports one fixed CCD ligand per target chain")
                chains.append(ChainSpec(physical_id, name, proto.sequence, kind="ligand"))
            else:
                if proto.type != ProtomerTypes.PROTEIN or proto.ptm_dict or set(proto.sequence) - AMINO_ACIDS:
                    raise ValueError("RFD4 supports canonical proteins and fixed CCD ligand targets")
                chains.append(ChainSpec(physical_id, name, proto.sequence))
        chains = tuple(chains)
        physical = {chain.chain for chain in chains}
        if set(self.binder_chains + self.target_chains) != physical:
            raise ValueError("RFD4 chain groups must cover the complete physical assembly")
        paths = getattr(evaluation_scores, "complex_paths", {}) or {}
        if self.assembly not in paths:
            raise ValueError("RFD4 requires an accepted evaluator complex for its assembly")
        path = str(paths[self.assembly])
        if structure_sequences(path, chains) != {chain.chain: chain.sequence for chain in chains}:
            raise ValueError("RFD4 accepted structure does not match current source sequences")

        uses = Counter(id(global_config.segments[component])
                       for proto in all_protomers.values() for component in proto.components)
        copies = Counter(chain.protomer for chain in chains)
        for chain in chains:
            if chain.chain not in self.binder_chains or chain.protomer not in protomers:
                continue
            proto = all_protomers[chain.protomer]
            for component, region in zip(proto.components, proto.components_slices):
                segment = global_config.segments[component]
                if any(proto.mutable_mask[region]) and (uses[id(segment)] > 1 or copies[chain.protomer] > 1):
                    raise ValueError("RFD4 cannot represent mutable shared segments or repeated copies")
        selected = {name: proto for name, proto in protomers.items()
                    if name in copies and any(chain.protomer == name and chain.chain in self.binder_chains
                                              for chain in chains)}
        reconstruction = self.feedback.get("reconstruction")
        excluded = prepare_exclude_sites(
            self.position_selector, evaluation_scores, selected,
            runtime_config=runtime_config, mutator_alias=getattr(self, "alias", self.name),
            fallback_to_all_mutable_if_selector_empty=False,
        ) if selected and reconstruction is None else {}
        eligible = tuple(ResidueRef(chain.chain, position + 1)
                         for chain in chains if chain.chain in self.binder_chains and chain.protomer in selected
                         for position in range(len(chain.sequence))
                         if selected[chain.protomer].mutable_mask[position]
                         and position not in excluded.get(chain.protomer, ()))
        budget = self.n_mutations() if reconstruction is None else len(eligible)
        if not eligible or budget == 0:
            return None
        combined = (combine_per_proto_mutator_scores(evaluation_scores, runtime_config)
                    if reconstruction is None else None)
        site_scores = None
        if combined is not None:
            site_scores = {}
            for chain in chains:
                if chain.protomer not in selected:
                    continue
                values = combined.get(chain.protomer)
                if values is None or len(values) != len(chain.sequence):
                    raise ValueError("RFD4 named per-residue scores must cover each mutable protomer")
                for position, value in enumerate(values, 1):
                    if not math.isfinite(float(value)):
                        raise ValueError("RFD4 site scores must be finite")
                    site_scores[ResidueRef(chain.chain, position)] = float(value)
        feedback = None
        fixed_plan_path = self.feedback.get("fixed_plan_path")
        if self.feedback.get("enabled", False) and not fixed_plan_path:
            feedback = analyze_interface(path, self.binder_chains, self.target_chains,
                                         **dict(self.feedback.get("chemistry") or {}))
        config = runtime_config or {}
        if "seed" in config:
            base_seed = config["seed"]
        else:
            try:
                base_seed = getattr(global_config, "seed", None)
            except RuntimeError:
                base_seed = None
            if base_seed is None:
                base_seed = 0
        seed = (self.seed + int(base_seed) + int(config.get("step", 0))) % 2**32
        excluded_aa = tuple(sorted(set(global_config.build_runtime_context().get_const("exclude_AA", ()) or ())))
        if set(excluded_aa) - AMINO_ACIDS:
            raise ValueError("RFD4 exclude_AA must contain canonical amino acids")
        policy = dict(self.feedback.get("retention") or {})
        for key, group in (("binder_chains", self.binder_chains), ("target_chains", self.target_chains)):
            if key in policy and set(policy[key]) != set(group):
                raise ValueError("RFD4 retention policy cannot override physical chain groups")
            policy[key] = list(group)
        if fixed_plan_path:
            # Controlled static-condition ablation: identities stay fixed, while
            # coordinates always come from the current accepted parent.
            raw = Path(fixed_plan_path).read_bytes()
            record = json.loads(raw)
            plan = _request_from_record(
                {"plan": record["plan"], "chains": [asdict(c) for c in chains],
                 "assembly": self.assembly, "seed": seed}, path,
            ).plan
            universe = {ResidueRef(c.chain, i) for c in chains
                        for i in range(1, c.residue_count + 1)}
            released, fixed = set(plan.released), set(plan.fixed_coordinates)
            if (len(fixed) != len(plan.fixed_coordinates) or released & fixed
                    or released | fixed != universe
                    or plan.excluded_amino_acids != excluded_aa
                    or any(c.atom.chain not in self.target_chains
                           or ResidueRef(c.atom.chain, c.atom.position) not in fixed
                           or c.role not in {"DONOR", "ACCEPTOR", "BOTH"} for c in plan.hbonds)):
                raise ValueError("Fixed RFD4 plan disagrees with current assembly or conditions")
            report = json.loads(plan.feedback_json)
            report["condition_update"] = {
                "mode": "fixed_initial_identities",
                "plan_sha256": hashlib.sha256(raw).hexdigest(),
                "initial_structure_sha256": record["source_sha256"],
                "coordinates": "current accepted parent, not initial coordinates",
            }
            plan = ConditionPlan(plan.released, plan.fixed_coordinates, plan.hbonds,
                                 plan.excluded_amino_acids,
                                 json.dumps(report, sort_keys=True, allow_nan=False))
        else:
            proposal_feedback = self.feedback.get("repair_unsatisfied_attempts", False)
            if reconstruction is not None:
                from .rfd4_reconstruction import plan_reconstruction
                plan = plan_reconstruction(
                    path, chains, feedback, config=dict(reconstruction), policy=policy,
                    excluded_amino_acids=excluded_aa,
                    failed_hbond_attempts=dict(self._failed_hbond_attempts) if proposal_feedback else None,
                )
            else:
                plan = plan_conditions(chains, eligible, budget, feedback=feedback,
                                       policy=policy,
                                       site_scores=site_scores, seed=seed,
                                       excluded_amino_acids=excluded_aa,
                                       **({"failed_hbond_attempts": dict(self._failed_hbond_attempts)}
                                          if proposal_feedback else {}))
            if proposal_feedback:
                report = json.loads(plan.feedback_json)
                report["proposal_feedback"] = {
                    "rule": "deprioritize chemically unsatisfied attempted targets; not round-number rotation",
                    "failed_hbond_attempts": dict(self._failed_hbond_attempts),
                    "prior_outcomes": list(self._hbond_attempt_history),
                    "acceptance": "diagnose all generated proposals, including those later rejected",
                }
                plan = ConditionPlan(plan.released, plan.fixed_coordinates, plan.hbonds,
                                     plan.excluded_amino_acids,
                                     json.dumps(report, sort_keys=True, allow_nan=False),
                                     plan.reconstruction)
        if not plan.released:
            return None
        if len(set(plan.released)) != len(plan.released) or len(plan.released) > budget or not set(plan.released) <= set(eligible):
            raise ValueError("RFD4 condition plan exceeds the eligible mutation budget")
        return GenerationRequest(path, self.assembly, chains, plan, seed)

    def _observe_hbond_attempt(self, request, generated):
        """Carry measured failed-goal evidence, not rejected coordinates, forward."""
        if (not self.feedback.get("repair_unsatisfied_attempts", False)
                or not request.plan.hbonds):
            return
        observation = {
            "plan": asdict(request.plan),
            "assembly": getattr(request, "assembly", None),
            "chains": [asdict(c) for c in getattr(request, "chains", ())],
            "seed": getattr(request, "seed", None),
            "structure_digest": generated.structure_digest,
            "structure_path": generated.structure_path,
            "sequences": getattr(generated, "sequences", ()),
        }
        observation = digest(observation)
        prior = self._observed_requests.get(generated.request_digest)
        if prior is not None:
            if prior != observation:
                raise ValueError("Conflicting RFD4 observed request identity")
            return
        from astrolabe.mutators.rfd4_feedback import analyze_interface
        feedback = analyze_interface(generated.structure_path, self.binder_chains, self.target_chains,
                                     **dict(self.feedback.get("chemistry") or {}))
        atoms = {a.atom: a for a in feedback.polar_atoms}
        outcomes = []
        for condition in request.plan.hbonds:
            atom = atoms.get(condition.atom)
            if atom is None or not atom.source_present or atom.role != condition.role:
                raise ValueError("Generated proposal changed the attempted target's chemical identity")
            key = f"{atom.atom.chain}:{atom.atom.position}:{atom.atom.atom}"
            outcomes.append({
                "atom": asdict(atom.atom), "role": atom.role, "key": key,
                "satisfied": atom.satisfied,
                "qualifying_hbond_count": atom.qualifying_hbond_count,
                "sasa": atom.sasa,
                "nearest_binder_heavy_distance": atom.nearest_other_group_heavy_atom_distance,
            })
        # Build the complete observation off-owner. A later result or history
        # construction failure must not publish counters without its receipt.
        failures = self._failed_hbond_attempts.copy()
        for outcome in outcomes:
            if not outcome["satisfied"]:
                failures[outcome["key"]] += 1
            outcome["failed_attempts_after"] = failures[outcome["key"]]
        history = [*self._hbond_attempt_history, {
            "request_digest": generated.request_digest,
            "observation_identity": observation,
            "generated_structure_path": generated.structure_path,
            "generated_structure_sha256": generated.structure_digest, "outcomes": outcomes,
        }]
        observed = {**self._observed_requests, generated.request_digest: observation}
        self._failed_hbond_attempts, self._hbond_attempt_history, self._observed_requests = (
            failures, history, observed)

    def _state_identity(self, binding):
        # Asset digests/environment are supplied by the durable owner; calling
        # backend.fingerprint here would start a native CPU interpreter.
        return {"binding": binding, "assembly": self.assembly,
                "binder": self.binder_chains, "target": self.target_chains,
                "feedback": self.feedback, "selector": self.position_selector,
                "fixed_plan": file_reference(self.feedback["fixed_plan_path"])
                if self.feedback.get("fixed_plan_path") else None,
                "backend": {key: str(getattr(self.backend, key, None)) for key in
                            ("python_bin", "source_dir", "model_dir", "max_fixed_atom_rmsd")},
                "weight": self.mutation_weight, "seed": self.seed,
                "generation": self.backend.generation}

    def snapshot_state(self, *, binding):
        references = {h["generated_structure_path"]: file_reference(h["generated_structure_path"])
                      for h in self._hbond_attempt_history}
        for h in self._hbond_attempt_history:
            if references[h["generated_structure_path"]]["sha256"] != h["generated_structure_sha256"]:
                raise ValueError("RFD4 observed structure changed")
        return ScientificState.create("rfd4-feedback", self._state_identity(binding), {
            "failures": dict(self._failed_hbond_attempts), "history": self._hbond_attempt_history,
            "observed": self._observed_requests, "references": references})

    def export_state(self):
        return self.snapshot_state(binding={}).to_dict()

    def restore_state(self, state, *, binding=None):
        binding = {} if binding is None else binding
        payload = ScientificState.validate(state, "rfd4-feedback", self._state_identity(binding))
        fields(payload, ("failures", "history", "observed", "references"))
        failures, history, observed, references = (payload[k] for k in
                                                  ("failures", "history", "observed", "references"))
        if (not all(isinstance(v, dict) for v in (failures, observed, references))
                or not isinstance(history, list)):
            raise ValueError("Invalid RFD4 feedback state")
        counts, identities, paths = Counter(), {}, set()
        for row in history:
            fields(row, ("request_digest", "observation_identity", "generated_structure_path",
                         "generated_structure_sha256", "outcomes"))
            rid, path = row["request_digest"], row["generated_structure_path"]
            if (not isinstance(rid, str) or not rid or rid in identities
                    or not isinstance(path, str) or path not in references
                    or references[path]["sha256"] != row["generated_structure_sha256"]
                    or observed.get(rid) != row["observation_identity"]):
                raise ValueError("Invalid RFD4 observation/reference binding")
            identities[rid] = row["observation_identity"]
            paths.add(path)
            if not isinstance(row["outcomes"], list):
                raise ValueError("Invalid RFD4 feedback outcomes")
            for outcome in row["outcomes"]:
                atom = outcome["atom"]
                if (outcome["key"] != f"{atom['chain']}:{atom['position']}:{atom['atom']}"
                        or type(outcome["satisfied"]) is not bool):
                    raise ValueError("Invalid RFD4 failure identity")
                if not outcome["satisfied"]:
                    counts[outcome["key"]] += 1
                if natural(outcome["failed_attempts_after"]) != counts[outcome["key"]]:
                    raise ValueError("Invalid RFD4 failure history")
        if dict(counts) != failures or identities != observed or paths != set(references):
            raise ValueError("Incomplete RFD4 feedback state")
        for ref in references.values():
            validate_reference(ref)
        self._failed_hbond_attempts = Counter(failures)
        self._hbond_attempt_history, self._observed_requests = history, observed

    def prepare_native_mutation(self, protomers, evaluation_scores, runtime_config):
        request = self.prepare_generation(protomers, evaluation_scores, runtime_config)
        request_record = None
        if request is not None:
            request_record = json.loads(json.dumps(asdict(request), allow_nan=False))
            request_record.pop("structure_path")
        return {
            "request": request_record,
            "input_hex": Path(request.structure_path).read_bytes().hex() if request else None,
            "input_suffix": Path(request.structure_path).suffix.lower() if request else None,
            "backend": self.backend.fingerprint(),
            "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "feedback_sha256": hashlib.sha256(Path(__file__).with_name("rfd4_feedback.py").read_bytes()).hexdigest(),
            "reconstruction_sha256": hashlib.sha256(Path(__file__).with_name("rfd4_reconstruction.py").read_bytes()).hexdigest(),
            "artifact_sha256": hashlib.sha256(Path(__file__).parents[1].joinpath("design/generated_structure.py").read_bytes()).hexdigest(),
        }

    def mutate_prepared(self, prepared, protomers, evaluation_scores, runtime_config):
        self._validate_boundary(runtime_config)
        if prepared["request"] is None:
            return None
        if prepared["backend"] != self.backend.fingerprint():
            raise ValueError("RFD4 assets changed after proposal preparation")
        suffix = prepared["input_suffix"]
        with tempfile.TemporaryDirectory(prefix="astrolabe-rfd4-input-") as directory:
            path = Path(directory) / ("accepted" + suffix)
            path.write_bytes(bytes.fromhex(prepared["input_hex"]))
            request = _request_from_record(prepared["request"], str(path))
            for chain in request.chains:
                if global_config.protomers[chain.protomer].sequence != chain.sequence:
                    raise ValueError("RFD4 prepared source sequence no longer matches working state")
            generated = self.backend.generate(request)
        generated_sequences = dict(generated.sequences)
        if len(generated_sequences) != len(generated.sequences) or set(generated_sequences) != {chain.chain for chain in request.chains}:
            raise ValueError("RFD4 generated chain mapping is incomplete or duplicated")
        if structure_sequences(generated.structure_path, request.chains) != generated_sequences:
            raise ValueError("RFD4 generated coordinates and sequences disagree")
        if hashlib.sha256(Path(generated.structure_path).read_bytes()).hexdigest() != generated.structure_digest:
            raise ValueError("RFD4 generated structure digest mismatch")
        if request.plan.reconstruction is not None:
            from .rfd4_reconstruction import read_motif_lineage
            plan = request.plan.reconstruction
            staged = []
            for chain in request.chains:
                sequence = generated_sequences[chain.chain]
                if chain.chain == plan["chain"]:
                    if (set(sequence) - AMINO_ACIDS
                            or not plan["length_min"] <= len(sequence) <= plan["length_max"]):
                        raise ValueError("Generated binder violates canonical variable-length bounds")
                    proto = global_config.protomers[chain.protomer]
                    if len(proto.components) != 1:
                        raise ValueError("Variable binder topology changed after preparation")
                    staged.append((global_config.segments[proto.components[0]], sequence))
                elif sequence != chain.sequence:
                    raise ValueError("Variable reconstruction changed a fixed ligand")
            lineage = read_motif_lineage(generated.structure_path)
            if set(lineage.values()) != {m["id"] for m in plan["motifs"]}:
                raise ValueError("Generated motif lineage differs from requested protection")
            try:
                for segment, sequence in staged:
                    segment.clean_up()
                    segment.stage_sequence(sequence)
                self._observe_hbond_attempt(request, generated)
            except Exception:
                for segment, _ in staged:
                    segment.clear_mutations()
                raise
            return GenerationResult(request.assembly, request.chains, generated)
        staged = {}
        released = set(request.plan.released)
        for chain in request.chains:
            sequence = generated_sequences[chain.chain]
            if chain.kind == "ligand":
                if sequence != chain.sequence:
                    raise ValueError("RFD4 changed a fixed ligand CCD identity")
                continue
            if len(sequence) != len(chain.sequence) or set(sequence) - AMINO_ACIDS:
                raise ValueError("RFD4 output must preserve canonical protein lengths")
            proto = global_config.protomers[chain.protomer]
            for component, region in zip(proto.components, proto.components_slices):
                segment = global_config.segments[component]
                mutations = []
                for position in range(region.start, region.stop):
                    before, after = chain.sequence[position], sequence[position]
                    if before == after:
                        continue
                    if ResidueRef(chain.chain, position + 1) not in released or not proto.mutable_mask[position]:
                        raise ValueError("RFD4 changed a nonreleased or fixed sequence position")
                    if after in request.plan.excluded_amino_acids:
                        raise ValueError("RFD4 generated an excluded amino acid")
                    mutations.append((position - region.start, before, after, 0.0, 0.0))
                if id(segment) in staged and staged[id(segment)][1] != mutations:
                    raise ValueError("RFD4 output couples shared sequence variables inconsistently")
                staged[id(segment)] = (segment, mutations)
        for segment, mutations in staged.values():
            segment.clean_up()
            segment.mutations = mutations
        try:
            for chain in request.chains:
                if global_config.protomers[chain.protomer].mutated_sequence != generated_sequences[chain.chain]:
                    raise ValueError("RFD4 pending mutations do not exactly reproduce generated sequences")
            self._observe_hbond_attempt(request, generated)
        except Exception:
            for segment, _ in staged.values():
                segment.clear_mutations()
            raise
        return GenerationResult(request.assembly, request.chains, generated)

    def mutate(self, protomer, evaluation_scores=None, runtime_config=None):
        prepared = self.prepare_native_mutation(protomer, evaluation_scores, runtime_config)
        return self.mutate_prepared(prepared, protomer, evaluation_scores, runtime_config)

    def shutdown(self):
        self.backend.shutdown()

    def prepare_mutation_input(self, protomer, evaluation_scores, runtime_config):
        return self.prepare_generation(protomer, evaluation_scores, runtime_config)

    def compute_mutation(self, input_feat):
        raise NotImplementedError("RFD4 uses joint generation, not amino-acid probabilities")

    def prepare_mutation_output(self, mutation_output, protomer, evaluation_scores, runtime_config):
        raise NotImplementedError("RFD4 stages exact generated differences in mutate()")
