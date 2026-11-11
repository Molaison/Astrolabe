"""Protein-only Molence base/Constraint input and real confidence translation."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, replace
import json
from pathlib import Path
import re
import shutil

import numpy as np

from astrolabe.core import global_config
from astrolabe.core.chain_resolution import chain_to_protomer, resolve_selector_chain_ids
from astrolabe.core.folding_input import ProteinChain
from astrolabe.evaluators.base import Evaluator
from astrolabe.evaluators.mol1_contract import active_constraints, constraints_for_chains, explicit_seeds, output_samples
from astrolabe.evaluators.structure_output import project_structure, structure_directory
from astrolabe.evaluators.utils import StandardScores


@dataclass(frozen=True)
class Mol1PreparedInput:
    json_path: Path
    output_dir: Path
    job: dict


def _binding_constraints(oligo, lengths):
    from astrolabe.core.binding_site import parse_binding_site, resolve_pocket_constraint

    spec = None
    for field in ("binding_site", "dynamic_binding_site"):
        raw = getattr(global_config, field, None)
        if raw is not None:
            if not isinstance(raw, Mapping):
                raise ValueError(f"Mol-1 {field} must be a mapping")
            selected = raw if "binder" in raw or "target" in raw else raw.get(oligo)
            if selected is not None:
                keys = {"binder", "target", "mode", "max_distance", "min_distance",
                        "max_distance_angstrom", "min_distance_angstrom"}
                if not isinstance(selected, Mapping) or set(selected) - keys:
                    raise ValueError(f"Mol-1 {field} contains unsupported binding_site fields")
            spec = parse_binding_site(raw, oligo_name=oligo)
        if spec is not None:
            break
    if spec is None:
        return {}
    # Native conditioning uses fixed contact semantics, not arbitrary bounds.
    if spec.max_distance != 4.5 or spec.min_distance != 0:
        raise ValueError("Mol-1 binding_site requires max_distance=4.5 and min_distance=0; arbitrary distance bounds are unsupported")

    def chain(selector):
        matches = resolve_selector_chain_ids(oligo, selector)
        if len(matches) != 1 or matches[0] not in lengths:
            raise ValueError(f"Mol-1 binding_site selector must resolve to one present chain: {selector!r}")
        return matches[0]

    binder, target = chain(spec.binder.chain), chain(spec.target.chain)
    if binder == target:
        raise ValueError("Mol-1 binding_site binder and target must differ")
    result = {}
    if spec.mode in {"pocket", "both"}:
        pocket = resolve_pocket_constraint(spec)
        source, partner = chain(pocket.contact_chain), chain(pocket.binder_chain)
        result["site"] = {"source_chains": [source], "partner_chains": [partner],
                          "residue_ids": [[source, pos] for pos in pocket.contact_positions]}
    if spec.mode in {"contact", "both"}:
        if not spec.binder.positions or not spec.target.positions:
            raise ValueError("Mol-1 binding_site contact/both requires residue positions on both chains")
        result["contact"] = {"pairs": [[[binder, a], [target, b]] for a in spec.binder.positions for b in spec.target.positions]}
    return result


def _merge_constraints(*sources):
    result = {}
    for source in sources:
        source = active_constraints(source)
        if set(result) & set(source):
            raise ValueError("Mol-1 constraint primitive supplied by multiple sources; specify it once")
        result.update(source)
    return result


def collect_sample(job, summary_path, full_path, cif_path):
    """Map confidence through native atom/token IDs, and verify output CIF identities."""
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    from Bio.Data.PDBData import protein_letters_3to1

    summary = json.loads(summary_path.read_text())
    full = json.loads(full_path.read_text())
    identity = json.loads((summary_path.parent / "native_identity.json").read_text())
    sequences = {row["proteinChain"]["id"][0]: row["proteinChain"]["sequence"] for row in job["sequences"]}
    tokens = identity["tokens"]
    expected_residues = {(cid, pos) for cid, sequence in sequences.items() for pos in range(1, len(sequence) + 1)}
    if len(tokens) != len(expected_residues) or {(c, p) for c, p, _ in tokens} != expected_residues:
        raise ValueError("Mol-1 native tokens do not preserve all input chain/residue identities")
    atom_scores = np.asarray(full["atom_plddt"], dtype=float)
    atom_token = np.asarray(full["atom_to_token_idx"])
    token_asym = np.asarray(full["token_asym_id"])
    if (atom_scores.ndim != 1 or atom_token.shape != atom_scores.shape or token_asym.shape != (len(tokens),)
            or not np.issubdtype(atom_token.dtype, np.integer) or not np.issubdtype(token_asym.dtype, np.integer)
            or not np.isfinite(atom_scores).all() or np.any(atom_scores < 0) or np.any(atom_scores > 1)
            or np.any(atom_token < 0) or np.any(atom_token >= len(tokens))
            or atom_token.tolist() != identity["atom_to_token_idx"]
            or token_asym.tolist() != [row[2] for row in tokens]):
        raise ValueError("Mol-1 full confidence has invalid atom/token axes or pLDDT units (expected 0..1)")
    native_atoms = identity["atoms"]
    if len(native_atoms) != len(atom_token) or len({tuple(a) for a in native_atoms}) != len(native_atoms):
        raise ValueError("Mol-1 native atom identities are incomplete or duplicated")
    for index, token in enumerate(atom_token):
        if native_atoms[index][:2] != tokens[int(token)][:2]:
            raise ValueError("Mol-1 native atom-to-token residue identity mismatch")
    cif = MMCIF2Dict(str(cif_path))
    atoms = list(zip(cif["_atom_site.label_asym_id"], map(int, cif["_atom_site.label_seq_id"]),
                     cif["_atom_site.label_atom_id"]))
    if len(atoms) != len(native_atoms) or set(atoms) != {tuple(a) for a in native_atoms}:
        raise ValueError("Mol-1 CIF atoms differ from native atom identities")
    coordinates = np.asarray([cif[f"_atom_site.Cartn_{axis}"] for axis in "xyz"], dtype=float)
    if not np.isfinite(coordinates).all():
        raise ValueError("Mol-1 CIF contains nonfinite coordinates")
    for (cid, position, _), residue in zip(atoms, cif["_atom_site.label_comp_id"]):
        if protein_letters_3to1.get(residue, "X") != sequences[cid][position - 1]:
            raise ValueError("Mol-1 output sequence differs from the explicit input chain")

    def scalar(field, low, high=None):
        value = np.asarray(summary[field], dtype=float)
        if value.size != 1 or not np.isfinite(value).all():
            raise ValueError(f"Mol-1 {field} must contain one finite value per sample")
        number = float(value.reshape(-1)[0])
        if number < low or high is not None and number > high:
            raise ValueError(f"Mol-1 {field} has unexpected units/range")
        return number

    per_chain = {cid: np.empty(len(sequence), dtype=float) for cid, sequence in sequences.items()}
    asym_by_chain = {}
    for token, (cid, position, asym) in enumerate(tokens):
        scores = atom_scores[atom_token == token]
        if scores.size == 0:
            raise ValueError("Mol-1 token has no real atom confidence")
        per_chain[cid][position - 1] = scores.mean()
        if cid in asym_by_chain and asym_by_chain[cid] != asym:
            raise ValueError("Mol-1 chain spans multiple native asym IDs")
        asym_by_chain[cid] = asym
    if len(set(asym_by_chain.values())) != len(sequences):
        raise ValueError("Mol-1 native asym ID identifies multiple input chains")
    native_order = sorted(asym_by_chain.values())
    order = [native_order.index(asym_by_chain[cid]) for cid in sequences]
    pairs = {}
    for field in ("chain_pair_iptm", "chain_pair_iptm_global", "chain_pair_plddt", "chain_pair_gpde"):
        if field in summary:
            matrix = np.asarray(summary[field], dtype=float)
            if matrix.shape != (len(sequences), len(sequences)) or not np.isfinite(matrix).all():
                raise ValueError(f"Mol-1 {field} has invalid native chain axes")
            pairs[field] = matrix[np.ix_(order, order)]
    return {"plddt": scalar("plddt", 0, 100) / 100,
            "ptm": scalar("ptm", 0, 1), "iptm": scalar("iptm", 0, 1),
            "pde": scalar("gpde", 0), "per_chain": per_chain, "chain_pairs": pairs}


class Mol1Evaluator(Evaluator):
    _staged_evaluation = True

    def __init__(self, model_runner, output_dir):
        self.model_runner = model_runner
        self.output_dir = Path(output_dir)

    def prepare_input(self, inputs, runtime_config):
        result = {}
        step = int(getattr(global_config, "step", 0) or 0)
        for oligo, fold in inputs.items():
            if not isinstance(oligo, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", oligo):
                raise ValueError("Mol-1 oligomer name must be a safe path component")
            if any(getattr(fold, key, None) for key in
                   ("bonded_atom_pairs", "bonded_atom_pair_meta", "cyclic_chain_ids", "cyclic_relative_patches", "user_ccd")):
                raise ValueError("Mol-1 adapter does not support topology, covalent bonds, cyclization or user CCD")
            seeds = explicit_seeds(list(fold.rng_seeds))
            job_name = f"step_{step:03d}_{oligo}"
            directory = self.output_dir / "inputs" / job_name
            directory.mkdir(parents=True, exist_ok=True)
            rows, lengths = [], {}
            for chain in fold.chains:
                if not isinstance(chain, ProteinChain):
                    raise ValueError("Mol-1 adapter supports protein chains only")
                if chain.ptms or chain.templates:
                    raise ValueError("Mol-1 adapter does not support PTM/templates")
                cid, sequence = chain.id, chain.sequence
                if (not re.fullmatch(r"[A-Za-z0-9_-]+", cid) or cid in lengths
                        or not sequence or set(sequence) - set("ACDEFGHIKLMNPQRSTVWYX")):
                    raise ValueError("Mol-1 invalid/duplicated chain ID or unsupported protein sequence")
                lengths[cid] = len(sequence)
                entry = {"id": [cid], "sequence": sequence, "count": 1}
                supplied = False
                for attr, key in (("unpaired_msa", "unpairedMsaPath"), ("paired_msa", "pairedMsaPath")):
                    text = getattr(chain, attr)
                    if text:
                        if not self.model_runner.settings["use_msa"]:
                            raise ValueError("Mol-1 received MSA but use_msa=false; refusing to discard it")
                        path = directory / f"{cid}_{attr}.a3m"
                        path.write_text(text, encoding="utf-8")
                        entry[key] = str(path.resolve())
                        supplied = True
                if self.model_runner.settings["use_msa"] and not supplied:
                    raise ValueError("Mol-1 use_msa=true requires supplied MSA on every protein chain; no remote search")
                rows.append({"proteinChain": entry})
            if not rows or set(lengths) != set(chain_to_protomer(oligo)):
                raise ValueError("Mol-1 physical chain IDs must match Astrolabe oligomer identity")
            raw = _merge_constraints(self.model_runner.settings.get("constraints"),
                                     getattr(fold, "constraint_ir", None), _binding_constraints(oligo, lengths))
            constraints = constraints_for_chains(raw, lengths)
            job = {"name": job_name, "modelSeeds": seeds, "sequences": rows}
            if constraints:
                job["constraints"] = constraints
            path = directory / "input.json"
            path.write_text(json.dumps([job], allow_nan=False), encoding="utf-8")
            result[oligo] = Mol1PreparedInput(path, self.output_dir, job)
        return result

    def run_function(self, prepared):
        return {name: replace(value, output_dir=self.model_runner.run_inference(
            input_json=value.json_path, out_dir=value.output_dir)) for name, value in prepared.items()}

    def prepare_output(self, output, inputs, runtime_config):
        structures = structure_directory(global_config.out_dir, getattr(self, "alias", None))
        structures.mkdir(parents=True, exist_ok=True)
        fields = {field: {} for field in ("plddt", "ptm", "iptm", "pde")}
        per_proto, pairs, paths = defaultdict(list), {}, {}
        for name, prepared in output.items():
            samples = output_samples(prepared.output_dir, prepared.job, self.model_runner.settings["num_samples"])
            pair_values = defaultdict(list)
            for _, _, summary, full, cif in samples:
                values = collect_sample(prepared.job, summary, full, cif)
                for field in fields:
                    fields[field].setdefault(name, []).append(values[field])
                for cid, scores in values["per_chain"].items():
                    per_proto[chain_to_protomer(name)[cid]].append(scores)
                for field, value in values["chain_pairs"].items():
                    pair_values[field].append(value)
            paths[name] = structures / f"{prepared.job['name']}_model.cif"
            # Canonical output is the first explicit seed/sample, never best-of-N.
            shutil.copy2(samples[0][-1], paths[name])
            project_structure(paths[name], global_config.out_dir, getattr(self, "alias", None))
            if pair_values:
                pairs[name] = dict(pair_values)
        return StandardScores(names=list(output), **fields, pae=None, complex_paths=paths,
                              per_proto_scores=dict(per_proto), chain_pair_scores=pairs or None)
