"""Computational readouts from Ahn et al., bioRxiv 2025.09.29.678932v1.

The paper's ad-hoc pH score is not a binding-affinity or pKa prediction.
Unspecified structural conventions are documented in docs/ph_sensitive_design.md.
PyRosetta imports and initialization remain lazy.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import math

import numpy as np

from astrolabe.core.chain_resolution import resolve_selector_chain_ids
from astrolabe.losses.base import loss_registry


PH_SCORE_WEIGHTS = {
    "low_pH_surf_D": 1.0,
    "low_pH_surf_DD": 3.0,
    "low_pH_surf_D_bb": 2.0,
    "low_pH_surf_DD_bb": 6.0,
    "low_pH_core_D": 3.0,
    "low_pH_core_DD": 9.0,
    "low_pH_core_D_bb": 6.0,
    "low_pH_core_DD_bb": 18.0,
    "high_pH_surf_A": 1.0,
    "high_pH_surf_D": 1.0,
    "high_pH_surf_AD": 3.0,
    "high_pH_surf_A_bb": 2.0,
    "high_pH_surf_D_bb": 2.0,
    "high_pH_surf_AD_bb": 6.0,
    "high_pH_core_A": 3.0,
    "high_pH_core_D": 3.0,
    "high_pH_core_AD": 9.0,
    "high_pH_core_A_bb": 6.0,
    "high_pH_core_D_bb": 6.0,
    "high_pH_core_AD_bb": 18.0,
}


def ph_score_from_counts(counts: dict[str, int]) -> float:
    """Apply the published 20 coefficients to a complete histidine-count table."""
    if set(counts) != set(PH_SCORE_WEIGHTS):
        raise ValueError("pH score requires exactly the 20 published count terms.")
    for key, value in counts.items():
        if not math.isfinite(value) or value < 0 or int(value) != value:
            raise ValueError(f"Invalid histidine count for {key}: {value!r}")
    return float(sum(PH_SCORE_WEIGHTS[key] * value for key, value in counts.items()))


def _pyrosetta():
    from astrolabe.losses.design_screening import _ensure_pyrosetta

    _ensure_pyrosetta()
    import pyrosetta

    return pyrosetta


def _load_pose(path):
    from astrolabe.losses.design_screening import _to_pdb_path

    pr = _pyrosetta()
    original = Path(path)
    pdb = _to_pdb_path(original)
    try:
        return pr.pose_from_file(str(pdb))
    finally:
        if pdb != original:
            pdb.unlink()


def _chain_residues(pose, oligo_name, selector):
    chains = resolve_selector_chain_ids(oligo_name, selector)
    if not chains:
        raise ValueError(f"Chain selector {selector!r} does not match oligomer {oligo_name!r}.")
    found = {pose.pdb_info().chain(i) for i in range(1, pose.total_residue() + 1)}
    if not set(chains).issubset(found):
        raise ValueError(f"Selected chain(s) {chains!r} missing from structure; found {sorted(found)!r}.")
    return {
        i for i in range(1, pose.total_residue() + 1)
        if pose.pdb_info().chain(i) in chains
    }


def _interface_residues(pose, name, binder_chain, target_chain):
    binder = _chain_residues(pose, name, binder_chain)
    target = _chain_residues(pose, name, target_chain)
    if binder & target:
        raise ValueError("binder_chain and target_chain must select disjoint chains.")
    return binder, target


def _repack(pose):
    pr = _pyrosetta()
    task = pr.standard_packer_task(pose)
    task.restrict_to_repacking()
    task.or_include_current(True)
    for i in range(1, pose.total_residue() + 1):
        residue = pose.residue(i)
        if not residue.is_protein():
            continue
        base = residue.type().base_name()
        # Keep the specified microstate even when the caller enabled pH_mode.
        names = ["HIS", "HIS_D"] if base in {"HIS", "HIS_D"} else [base]
        allowed = pr.rosetta.utility.vector1_std_string()
        for name in names:
            allowed.append(name)
        task.nonconst_residue_task(i).restrict_restypes(allowed)
    pr.rosetta.protocols.minimization_packing.PackRotamersMover(
        pr.create_score_function("ref2015"), task
    ).apply(pose)


@lru_cache(maxsize=4)
def _ph_states_cached(path: str, mtime_ns: int, size: int):
    """Fixed His microstates, without changing process-global Rosetta pH options.

    The paper drives all His to the doubly protonated state using pH=0/e_pH=100
    to represent its low-pH pose, and to neutral states at pH=7. Loading HIS_P on
    these poses gives that state constraint even if Rosetta was initialized by
    another evaluator without pH_mode. Other titratable residues are unchanged.
    """
    pr = _pyrosetta()
    native = _load_pose(path)
    low, high = native.clone(), native.clone()
    histidines = [i for i in range(1, native.total_residue() + 1)
                  if native.residue(i).name3() == "HIS"]
    if histidines:
        params = pr.rosetta.basic.database.full_name(
            "chemical/residue_type_sets/fa_standard/residue_types/protonation_states/HIS_P.params"
        )
        if not low.conformation().residue_type_set_for_conf().has_name("HIS_P"):
            pr.generate_nonstandard_residue_set(low, [params])
        for i in histidines:
            pr.rosetta.protocols.simple_moves.MutateResidue(i, "HIS_P").apply(low)
            pr.rosetta.protocols.simple_moves.MutateResidue(i, "HIS").apply(high)
    _repack(low)
    _repack(high)
    # A process initialized in pH_mode must not silently sample other states.
    for pose, protonated in ((low, True), (high, False)):
        for i in histidines:
            if pose.residue(i).has_variant_type(pr.rosetta.core.chemical.PROTONATED) != protonated:
                raise RuntimeError("Histidine protonation changed during pH-state repacking.")
    return low, high


def _ph_states(path):
    path = Path(path).resolve()
    stat = path.stat()
    low, high = _ph_states_cached(str(path), stat.st_mtime_ns, stat.st_size)
    return low.clone(), high.clone()


def _hbonds(pose):
    pr = _pyrosetta()
    pr.create_score_function("ref2015")(pose)
    result = pr.rosetta.core.scoring.hbonds.HBondSet()
    pr.rosetta.core.scoring.hbonds.fill_hbond_set(pose, False, result)
    return [result.hbond(i) for i in range(1, result.nhbonds() + 1)]


def _histidine_pattern_counts(pose, binder, target, phase, core, max_energy):
    counts = {key: 0 for key in PH_SCORE_WEIGHTS}
    patterns = {}
    for hb in _hbonds(pose):
        if hb.energy() > max_energy:
            continue
        don, acc = hb.don_res(), hb.acc_res()
        if acc in binder and don in target:
            his, atom, role, partner, partner_atom = acc, hb.acc_atm(), "A", don, hb.don_hatm()
        elif don in binder and acc in target:
            his, atom, role, partner, partner_atom = don, hb.don_hatm(), "D", acc, hb.acc_atm()
        else:
            continue
        residue = pose.residue(his)
        ring_atom = residue.atom_base(atom) if role == "D" else atom
        if residue.name3() != "HIS" or residue.atom_name(ring_atom).strip() not in {"ND1", "NE2"}:
            continue
        record = patterns.setdefault(his, {"A": set(), "D": set(), "bb": False})
        record[role].add(ring_atom)
        record["bb"] |= pose.residue(partner).atom_is_backbone(partner_atom)
    for his, record in patterns.items():
        if phase == "low_pH":
            if not record["D"]:
                continue
            role = "DD" if len(record["D"]) >= 2 else "D"
        else:
            role = "AD" if record["A"] and record["D"] else ("A" if record["A"] else "D")
        layer = "core" if his in core else "surf"
        key = f"{phase}_{layer}_{role}" + ("_bb" if record["bb"] else "")
        counts[key] += 1
    return counts


@loss_registry.register(
    name="ph_score",
    description="Ahn et al. 20-term interface histidine H-bond pattern score (higher is favored)",
    required_params=["binder_chain", "target_chain"],
    optional_params=["hbond_max_energy"],
    supports_oligomer_specific=True,
)
def ph_score(eval_result, binder_chain: str, target_chain: str, hbond_max_energy: float = 0.0):
    if not math.isfinite(hbond_max_energy) or hbond_max_energy > 0:
        raise ValueError("hbond_max_energy must be finite and non-positive.")
    pr = _pyrosetta()
    result = {}
    for name in eval_result.names:
        low, high = _ph_states(eval_result.complex_paths[name])
        binder, target = _interface_residues(high, name, binder_chain, target_chain)
        layer = pr.rosetta.core.select.residue_selector.LayerSelector()
        layer.set_layers(True, False, False)
        counts = {key: 0 for key in PH_SCORE_WEIGHTS}
        for pose, phase in ((low, "low_pH"), (high, "high_pH")):
            selected = layer.apply(pose)
            core = {i for i in binder if selected[i]}
            for key, value in _histidine_pattern_counts(
                pose, binder, target, phase, core, hbond_max_energy
            ).items():
                counts[key] += value
        result[name] = ph_score_from_counts(counts)
    return result


@loss_registry.register(
    name="ph_histidine_cation_acceptors",
    description="Number of binder histidines accepting a sidechain H-bond from target His/Lys/Arg",
    required_params=["binder_chain", "target_chain"],
    optional_params=["hbond_max_energy"],
    supports_oligomer_specific=True,
)
def ph_histidine_cation_acceptors(
    eval_result, binder_chain: str, target_chain: str, hbond_max_energy: float = 0.0,
):
    if not math.isfinite(hbond_max_energy) or hbond_max_energy > 0:
        raise ValueError("hbond_max_energy must be finite and non-positive.")
    scores = {}
    for name in eval_result.names:
        _low, high = _ph_states(eval_result.complex_paths[name])
        binder, target = _interface_residues(high, name, binder_chain, target_chain)
        acceptors = set()
        for hb in _hbonds(high):
            if hb.acc_res() not in binder or hb.don_res() not in target or hb.energy() > hbond_max_energy:
                continue
            his, donor = high.residue(hb.acc_res()), high.residue(hb.don_res())
            if (his.name3() == "HIS" and his.atom_name(hb.acc_atm()).strip() in {"ND1", "NE2"}
                    and donor.name3() in {"HIS", "LYS", "ARG"}
                    and not donor.atom_is_backbone(donor.atom_base(hb.don_hatm()))):
                acceptors.add(hb.acc_res())
        scores[name] = float(len(acceptors))
    return scores


@loss_registry.register(
    name="ph_positive_charge_count",
    description="Positive residue count: Lys/Arg plus doubly protonated His on selected chain(s)",
    required_params=["chain"],
    optional_params=["phase"],
    supports_oligomer_specific=True,
)
def ph_positive_charge_count(eval_result, chain: str, phase: str = "high"):
    if phase not in {"low", "high"}:
        raise ValueError("phase must be 'low' or 'high'.")
    pr = _pyrosetta()
    scores = {}
    for name in eval_result.names:
        states = _ph_states(eval_result.complex_paths[name])
        pose = states[0 if phase == "low" else 1]
        indices = _chain_residues(pose, name, chain)
        scores[name] = float(sum(
            pose.residue(i).name3() in {"LYS", "ARG"}
            or (pose.residue(i).name3() == "HIS"
                and pose.residue(i).has_variant_type(pr.rosetta.core.chemical.PROTONATED))
            for i in indices
        ))
    return scores


def _electrostatic_binding_energy(pose, binder, target, pack_separated=False):
    pr = _pyrosetta()
    # Slice complete selected chains, retaining their original residue types.
    def subset(indices):
        out = pr.Pose()
        selected = pr.rosetta.utility.vector1_unsigned_long()
        for index in sorted(indices):
            selected.append(index)
        pr.rosetta.core.pose.pdbslice(out, pose, selected)
        return out

    bound = subset(binder | target)
    separated = [subset(binder), subset(target)]
    if pack_separated:
        for chain_pose in separated:
            _repack(chain_pose)
    sf = pr.ScoreFunction()
    sf.set_weight(pr.rosetta.core.scoring.fa_elec, 1.0)
    return float(sf(bound) - sum(sf(partner) for partner in separated))


@loss_registry.register(
    name="ph_dddg_elec",
    description="fa_elec binding energy(low pH) minus binding energy(high pH), Rosetta units",
    required_params=["binder_chain", "target_chain"],
    optional_params=["pack_separated"],
    supports_oligomer_specific=True,
)
def ph_dddg_elec(eval_result, binder_chain: str, target_chain: str, pack_separated: bool = False):
    scores = {}
    for name in eval_result.names:
        low, high = _ph_states(eval_result.complex_paths[name])
        binder, target = _interface_residues(high, name, binder_chain, target_chain)
        scores[name] = _electrostatic_binding_energy(low, binder, target, pack_separated) - \
            _electrostatic_binding_energy(high, binder, target, pack_separated)
    return scores


def _network_pair(pose, histidine_residue, partner_residue):
    for index in (histidine_residue, partner_residue):
        if isinstance(index, bool) or int(index) != index or not 1 <= index <= pose.total_residue():
            raise ValueError("Network residues must be distinct, valid 1-based pose residue indices.")
    if histidine_residue == partner_residue:
        raise ValueError("Network residues must be distinct.")
    histidine_residue, partner_residue = int(histidine_residue), int(partner_residue)
    if pose.residue(histidine_residue).name3() != "HIS":
        raise ValueError("histidine_residue must identify a histidine.")
    if pose.residue(partner_residue).name3() not in {"HIS", "LYS", "ARG"}:
        raise ValueError("partner_residue must identify HIS, LYS or ARG.")
    return int(histidine_residue), int(partner_residue)


def _primary_network_hbond(pose, histidine_residue, partner_residue):
    his, partner = _network_pair(pose, histidine_residue, partner_residue)
    bonds = []
    for hb in _hbonds(pose):
        if {hb.don_res(), hb.acc_res()} != {his, partner} or hb.energy() >= 0.0:
            continue
        don = pose.residue(hb.don_res())
        acc = pose.residue(hb.acc_res())
        don_atom = don.atom_base(hb.don_hatm())
        if don.atom_is_backbone(don_atom) or acc.atom_is_backbone(hb.acc_atm()):
            continue
        his_atom = hb.acc_atm() if hb.acc_res() == his else don_atom
        if pose.residue(his).atom_name(his_atom).strip() in {"ND1", "NE2"}:
            bonds.append(hb)
    return min(bonds, key=lambda hb: hb.energy()) if bonds else None


@loss_registry.register(
    name="ph_histidine_hbond_energy",
    description="Most favorable side-chain His-(His/Lys/Arg) H-bond energy; no bond returns zero",
    required_params=["histidine_residue", "partner_residue"],
    supports_oligomer_specific=True,
)
def ph_histidine_hbond_energy(eval_result, histidine_residue: int, partner_residue: int):
    scores = {}
    for name in eval_result.names:
        pose = _load_pose(eval_result.complex_paths[name])
        bond = _primary_network_hbond(pose, histidine_residue, partner_residue)
        scores[name] = float(bond.energy()) if bond else 0.0
    return scores


@loss_registry.register(
    name="ph_histidine_hbond_depth",
    description="Mean AtomicDepth of the primary network H-bond heavy atoms (angstrom); no bond=0",
    required_params=["histidine_residue", "partner_residue"],
    optional_params=["probe_radius", "resolution"],
    supports_oligomer_specific=True,
)
def ph_histidine_hbond_depth(
    eval_result, histidine_residue: int, partner_residue: int,
    probe_radius: float = 1.4, resolution: float = 0.25,
):
    pr = _pyrosetta()
    if not math.isfinite(probe_radius) or probe_radius <= 0 or not math.isfinite(resolution) or resolution <= 0:
        raise ValueError("AtomicDepth probe_radius and resolution must be positive and finite.")
    scores = {}
    for name in eval_result.names:
        pose = _load_pose(eval_result.complex_paths[name])
        hb = _primary_network_hbond(pose, histidine_residue, partner_residue)
        if hb is None:
            scores[name] = 0.0
            continue
        depth = pr.rosetta.core.scoring.atomic_depth.AtomicDepth(pose, probe_radius, False, resolution)
        don = pose.residue(hb.don_res())
        acc = pose.residue(hb.acc_res())
        scores[name] = float((
            depth.calcdepth(don.atom(don.atom_base(hb.don_hatm())), don.atom_type_set()) +
            depth.calcdepth(acc.atom(hb.acc_atm()), acc.atom_type_set())
        ) / 2.0)
    return scores


@loss_registry.register(
    name="ph_histidine_pair_energy",
    description="Network interaction E(H,P)-E(G,P)-E(H,G)+E(G,G), fixed-backbone ref2015 units",
    required_params=["histidine_residue", "partner_residue"],
    supports_oligomer_specific=True,
)
def ph_histidine_pair_energy(eval_result, histidine_residue: int, partner_residue: int):
    pr = _pyrosetta()
    sf = pr.create_score_function("ref2015")
    scores = {}
    for name in eval_result.names:
        pose = _load_pose(eval_result.complex_paths[name])
        his, partner = _network_pair(pose, histidine_residue, partner_residue)
        values = []
        for mutated in ((), (his,), (partner,), (his, partner)):
            baseline = pose.clone()
            for index in mutated:
                pr.rosetta.protocols.simple_moves.MutateResidue(index, "GLY").apply(baseline)
            values.append(sf(baseline))
        scores[name] = float(values[0] - values[1] - values[2] + values[3])
    return scores


def _rotamer_sidechain_coordinates(pose, positions, atom_mode):
    pr = _pyrosetta()
    sf = pr.create_score_function("ref2015")
    sf(pose)
    task = pr.standard_packer_task(pose)
    task.restrict_to_repacking()
    task.or_include_current(True)
    graph = pr.rosetta.core.pack.create_packer_graph(pose, sf, task)
    result = {}
    for index in positions:
        rotamers = pr.rosetta.core.pack.rotamer_set.RotamerSetFactory.create_rotamer_set(pose)
        rotamers.set_resid(index)
        rotamers.build_rotamers(pose, sf, task, graph)
        coordinates = []
        for n in range(1, rotamers.num_rotamers() + 1):
            residue = rotamers.rotamer(n)
            charged_atoms = {
                "HIS": {"ND1", "NE2"}, "ARG": {"NE", "NH1", "NH2"}, "LYS": {"NZ"},
            }[residue.name3()]
            coordinates.extend(
                list(residue.xyz(atom)) for atom in range(1, residue.nheavyatoms() + 1)
                if not residue.atom_is_backbone(atom)
                and (atom_mode == "sidechain_heavy" or residue.atom_name(atom).strip() in charged_atoms)
            )
        if not coordinates:
            raise ValueError(f"No sidechain rotamers for residue {index}.")
        result[index] = np.asarray(coordinates, dtype=float)
    return result


@loss_registry.register(
    name="ph_histidine_cation_contacts",
    description="Binder His-target His/Lys/Arg pairs within cutoff over all library rotamers",
    required_params=["binder_chain", "target_chain", "distance_cutoff"],
    optional_params=["atom_mode"],
    supports_oligomer_specific=True,
)
def ph_histidine_cation_contacts(
    eval_result, binder_chain: str, target_chain: str, distance_cutoff: float,
    atom_mode: str = "cation_nitrogens",
):
    from scipy.spatial import cKDTree

    if not math.isfinite(distance_cutoff) or distance_cutoff <= 0:
        raise ValueError("distance_cutoff must be positive and finite (angstrom).")
    if atom_mode not in {"cation_nitrogens", "sidechain_heavy"}:
        raise ValueError("atom_mode must be 'cation_nitrogens' (Fig. S6C) or 'sidechain_heavy'.")
    result = {}
    for name in eval_result.names:
        pose = _load_pose(eval_result.complex_paths[name])
        binder, target = _interface_residues(pose, name, binder_chain, target_chain)
        histidines = {i for i in binder if pose.residue(i).name3() == "HIS"}
        cations = {i for i in target if pose.residue(i).name3() in {"HIS", "LYS", "ARG"}}
        coordinates = _rotamer_sidechain_coordinates(pose, histidines | cations, atom_mode)
        trees = {i: cKDTree(coordinates[i]) for i in cations}
        result[name] = float(sum(
            trees[j].query(coordinates[i], k=1)[0].min() <= distance_cutoff
            for i in histidines for j in cations
        ))
    return result
