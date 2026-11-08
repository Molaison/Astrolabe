"""Dynamic, sequence-free geometric support for joint Proton proposals.

Virtual rotamers are a proposal heuristic, never an installed motif or a scored
AF3 state. Only sparse canonical pair preferences reach the Potts solver. The
parent structure and sequence are not rewritten, and actual networks are still
measured on the subsequent AF3 output by ph_core.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import time
from typing import Any

import numpy as np


@dataclass(frozen=True)
class NetworkProposalParameters:
    enabled: bool = False
    max_preferences: int = 16
    max_rotamers: int = 12
    candidate_pair_limit: int = 32
    min_cb_distance: float = 3.0
    max_cb_distance: float = 9.0
    min_sequence_separation: int = 3
    min_pair_depth: float = 3.0
    clash_distance: float = 2.0
    geometry_trials: int = 256

    @classmethod
    def from_mapping(cls, value):
        values = dict(value or {})
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown network_proposal parameters: {sorted(unknown)}")
        result = cls(**values)
        if not isinstance(result.enabled, bool):
            raise ValueError("network_proposal.enabled must be boolean")
        for key in ("max_preferences", "max_rotamers", "candidate_pair_limit",
                    "min_sequence_separation", "geometry_trials"):
            x = getattr(result, key)
            if isinstance(x, bool) or int(x) != x or x < 1:
                raise ValueError(f"network_proposal.{key} must be a positive integer")
        for key in ("min_cb_distance", "max_cb_distance", "min_pair_depth", "clash_distance"):
            x = getattr(result, key)
            if not math.isfinite(x) or x <= 0:
                raise ValueError(f"network_proposal.{key} must be positive and finite")
        if result.min_cb_distance >= result.max_cb_distance:
            raise ValueError("network_proposal requires min_cb_distance < max_cb_distance")
        if result.max_preferences > 16:
            raise ValueError("network_proposal.max_preferences cannot exceed 16")
        return result


_CACHE: dict[tuple, tuple[list[dict], dict]] = {}


def _xyz(residue, atom):
    return np.asarray(tuple(residue.xyz(atom)), dtype=float)


def _rotamers(pr, pose, index, name, scorefxn, maximum):
    virtual = pose.clone()
    pr.rosetta.protocols.simple_moves.MutateResidue(index, name).apply(virtual)
    task = pr.standard_packer_task(virtual)
    task.restrict_to_repacking()
    task.or_include_current(True)
    for i in range(1, virtual.total_residue() + 1):
        if i != index:
            task.nonconst_residue_task(i).prevent_repacking()
    scorefxn(virtual)
    graph = pr.rosetta.core.pack.create_packer_graph(virtual, scorefxn, task)
    rotset = pr.rosetta.core.pack.rotamer_set.RotamerSetFactory.create_rotamer_set(virtual)
    rotset.set_resid(index)
    rotset.build_rotamers(virtual, scorefxn, task, graph)
    count = int(rotset.num_rotamers())
    if not count:
        return [virtual.residue(index).clone()]
    # Even coverage of the deterministic library ordering; no identity/site pins.
    chosen = np.unique(np.linspace(1, count, min(count, maximum), dtype=int))
    return [rotset.rotamer(int(i)).clone() for i in chosen]


def _potential_hbonds(acceptor, donor):
    """Cheap geometric prefilter; Rosetta, depth and clashes validate witnesses."""
    acc = [i for i in range(1, acceptor.nheavyatoms() + 1)
           if acceptor.atom_name(i).strip() in {"ND1", "NE2"}
           and acceptor.atom_type(i).is_acceptor()]
    polar_h = list(donor.Hpos_polar())
    result = []
    for a in acc:
        ap = _xyz(acceptor, a)
        for h in polar_h:
            d = int(donor.atom_base(h))
            if donor.atom_is_backbone(d):
                continue
            hp, dp = _xyz(donor, h), _xyz(donor, d)
            ha = float(np.linalg.norm(ap - hp))
            da = float(np.linalg.norm(ap - dp))
            if ha > 2.6 or da > 3.5 or ha < 1.2:
                continue
            v, w = dp - hp, ap - hp
            angle = float(np.degrees(np.arccos(np.clip(np.dot(v, w) /
                max(np.linalg.norm(v) * np.linalg.norm(w), 1e-12), -1., 1.))))
            if angle >= 120.0:
                result.append((ha, da, angle))
    return min(result) if result else None


def _clash_free(pose, replacements, minimum):
    # Side chains against all other residues, including the other replacement.
    # Backbone covalent contacts are excluded by considering sidechain atoms only.
    all_atoms = []
    owners = []
    for i in range(1, pose.total_residue() + 1):
        residue = replacements.get(i, pose.residue(i))
        for a in range(1, residue.nheavyatoms() + 1):
            all_atoms.append(_xyz(residue, a)); owners.append(i)
    all_atoms, owners = np.asarray(all_atoms), np.asarray(owners)
    for i, residue in replacements.items():
        others = all_atoms[owners != i]
        sc = [_xyz(residue, a) for a in range(1, residue.nheavyatoms() + 1)
              if not residue.atom_is_backbone(a)]
        if sc and np.any(np.linalg.norm(np.asarray(sc)[:, None] - others[None], axis=-1) < minimum):
            return False
    return True


def _best_effort_pair_preferences(sequence, mutable_positions, anchors, depths, parameters):
    """Choose one legal pair without claiming a validated hydrogen bond.

    Reach and sequence separation rank first, then burial and proximity to the
    midpoint of the usual CB distance interval. These are proposal heuristics;
    neither a geometry-quality reward nor an observed network score is assigned.
    """
    free = set(mutable_positions)
    candidates = []
    midpoint = (parameters.min_cb_distance + parameters.max_cb_distance) / 2.0
    roles = (("HIS", "H"), ("LYS", "K"), ("ARG", "R"))
    positions = sorted(anchors)
    for offset, i in enumerate(positions):
        for j in positions[offset + 1:]:
            preferences = []
            for h, d in ((i, j), (j, i)):
                if h not in free and sequence[h] != "H":
                    continue
                for donor_name, donor_aa in roles:
                    if d not in free and sequence[d] != donor_aa:
                        continue
                    if not ((h in free and sequence[h] != "H") or
                            (d in free and sequence[d] != donor_aa)):
                        continue
                    preferences.append({"i": h, "j": d, "aa_i": "HIS",
                                        "aa_j": donor_name, "quality": 0.0})
            if not preferences:
                continue
            distance = float(np.linalg.norm(anchors[i] - anchors[j]))
            reach_gap = max(parameters.min_cb_distance - distance,
                            distance - parameters.max_cb_distance, 0.0)
            separation_gap = max(parameters.min_sequence_separation - (j - i), 0)
            mean_depth = (depths[i] + depths[j]) / 2.0
            rank = (reach_gap, separation_gap, -mean_depth, abs(distance - midpoint), i, j)
            candidates.append((rank, preferences, distance, mean_depth))
    if not candidates:
        return [], {"fallback_reason": "no_legal_changed_pair"}
    rank, preferences, distance, mean_depth = min(candidates, key=lambda item: item[0])
    return preferences, {"fallback_pair": list(rank[-2:]), "fallback_rank": list(rank),
                         "fallback_anchor_distance": distance,
                         "fallback_mean_anchor_depth": mean_depth,
                         "fallback_reason": "no_geometry_validated_pair"}


def build_network_pair_preferences(path, binder_chain, sequence, mutable_positions,
                                   parameters: NetworkProposalParameters, *,
                                   fallback_to_best=False):
    """Return dynamic pair rewards and an auditable virtual-geometry summary.

    i/j are zero-based BINDER SEQUENCE positions, not pose or complex indices.
    Existing fixed His/Arg/Lys may be a partner but can never be substituted.
    """
    source = Path(path).resolve()
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    mutable = tuple(sorted(set(int(i) for i in mutable_positions)))
    key = (sha, str(binder_chain), sequence, mutable, parameters, bool(fallback_to_best))
    if key in _CACHE:
        prefs, evidence = copy.deepcopy(_CACHE[key])
        evidence["cache_hit"] = True
        return prefs, evidence
    started = time.monotonic()
    from astrolabe.losses.ph_sensitive import _load_pose, _pyrosetta, _hbonds
    from astrolabe.losses.ph_core import _binder_apo_pose, _heavy_endpoint_depths
    pr = _pyrosetta()
    complex_pose = _load_pose(source)
    indices = [i for i in range(1, complex_pose.total_residue() + 1)
               if complex_pose.pdb_info().chain(i) == binder_chain]
    apo = _binder_apo_pose(complex_pose, indices)
    actual = "".join(apo.residue(i).name1() for i in range(1, apo.total_residue() + 1))
    if actual != sequence or any(i < 0 or i >= len(sequence) for i in mutable):
        raise ValueError("Network proposal structure/mutable positions do not match binder sequence")
    free = set(mutable)
    scorefxn = pr.create_score_function("ref2015")
    base_depth = pr.rosetta.core.scoring.atomic_depth.AtomicDepth(apo, 1.4, False, 0.25)
    cb = {}
    cb_depth = {}
    for i in range(len(sequence)):
        residue = apo.residue(i + 1)
        if not residue.has("CB"):
            continue
        cb[i] = _xyz(residue, "CB")
        cb_depth[i] = float(base_depth.calcdepth(residue.atom(residue.atom_index("CB")),
                                               residue.atom_type_set()))
    pairs = []
    for i in sorted(cb):
        for j in sorted(cb):
            if j <= i or j - i < parameters.min_sequence_separation:
                continue
            if not ((i in free and (j in free or sequence[j] in "HRK")) or
                    (j in free and sequence[i] in "HRK")):
                continue
            distance = float(np.linalg.norm(cb[i] - cb[j]))
            if parameters.min_cb_distance <= distance <= parameters.max_cb_distance:
                pairs.append((-(cb_depth[i] + cb_depth[j]), abs(distance - 6.), i, j, distance))
    pairs.sort()
    selected_pairs = pairs[:parameters.candidate_pair_limit]
    rotcache = {}
    def rots(position, name):
        k = (position, name)
        if k not in rotcache:
            # Fixed partners retain their current sidechain geometry as well as identity.
            rotcache[k] = (_rotamers(pr, apo, position + 1, name, scorefxn, parameters.max_rotamers)
                           if position in free else [apo.residue(position + 1).clone()])
        return rotcache[k]
    witnesses = []
    trials = 0
    geometric_hits = 0
    clash_rejected = 0
    for _, _, i, j, distance in selected_pairs:
        possibilities = []
        for h, d in ((i, j), (j, i)):
            if h not in free and sequence[h] != "H":
                continue
            for donor_name, donor_aa in (("HIS", "H"), ("LYS", "K"), ("ARG", "R")):
                if d not in free and sequence[d] != donor_aa:
                    continue
                for hi, hr in enumerate(rots(h, "HIS")):
                    for di, dr in enumerate(rots(d, donor_name)):
                        geometry = _potential_hbonds(hr, dr)
                        if geometry is not None:
                            possibilities.append((geometry, h, d, donor_name, hi, di))
        possibilities.sort(key=lambda v: (v[0], v[1:]))
        # Expensive full-pose checks are bounded, after distance/angle screening.
        for geometry, h, d, donor_name, hi, di in possibilities[:parameters.geometry_trials]:
            geometric_hits += 1
            hr, dr = rots(h, "HIS")[hi], rots(d, donor_name)[di]
            if not _clash_free(apo, {h + 1: hr, d + 1: dr}, parameters.clash_distance):
                clash_rejected += 1
                continue
            trials += 1
            virtual = apo.clone()
            virtual.replace_residue(h + 1, hr, False)
            virtual.replace_residue(d + 1, dr, False)
            bonds = [b for b in _hbonds(virtual) if b.acc_res() == h + 1 and b.don_res() == d + 1
                     and virtual.residue(h + 1).atom_name(b.acc_atm()).strip() in {"ND1", "NE2"}
                     and not virtual.residue(d + 1).atom_is_backbone(
                         virtual.residue(d + 1).atom_base(b.don_hatm()))
                     and float(b.energy()) < 0]
            if not bonds:
                continue
            bond = min(bonds, key=lambda b: float(b.energy()))
            depth_calc = pr.rosetta.core.scoring.atomic_depth.AtomicDepth(virtual, 1.4, False, 0.25)
            depth = float(np.mean(_heavy_endpoint_depths(virtual, bond, depth_calc)))
            if depth < parameters.min_pair_depth:
                continue
            quality = min(1., -float(bond.energy()) / .5, depth / 4.)
            witnesses.append({"i": h, "j": d, "aa_i": "HIS", "aa_j": donor_name,
                              "quality": quality, "hbond_energy": float(bond.energy()),
                              "mean_atomic_depth": depth, "cb_distance": distance,
                              "hydrogen_acceptor_distance": geometry[0],
                              "donor_acceptor_distance": geometry[1], "DHA_angle": geometry[2],
                              "his_chi": [float(hr.chi(k)) for k in range(1, hr.nchi() + 1)],
                              "donor_chi": [float(dr.chi(k)) for k in range(1, dr.nchi() + 1)]})
            # One validated geometry is enough for this role; more can be found in other pairs.
            break
    best = {}
    for w in witnesses:
        k = (w["i"], w["j"], w["aa_i"], w["aa_j"])
        if k not in best or w["quality"] > best[k]["quality"]:
            best[k] = w
    witnesses = sorted(best.values(), key=lambda w: (-w["quality"], w["i"], w["j"], w["aa_j"]))[:parameters.max_preferences]
    prefs = [{k: w[k] for k in ("i", "j", "aa_i", "aa_j", "quality")} for w in witnesses]
    strict_count = len(prefs)
    fallback_details = {}
    if not prefs and fallback_to_best:
        anchors, depths = dict(cb), dict(cb_depth)
        ca_anchors = []
        for i in range(len(sequence)):
            residue = apo.residue(i + 1)
            if i not in anchors and residue.has("CA"):
                anchors[i] = _xyz(residue, "CA")
                depths[i] = float(base_depth.calcdepth(residue.atom(residue.atom_index("CA")),
                                                      residue.atom_type_set()))
                ca_anchors.append(i)
        prefs, fallback_details = _best_effort_pair_preferences(
            sequence, mutable, anchors, depths, parameters)
        fallback_details["fallback_CA_anchor_positions"] = ca_anchors
    evidence: dict[str, Any] = {"schema": "proton_network_proposal_geometry_v1",
        "source_sha256": sha, "binder_chain": binder_chain, "binder_sequence": sequence,
        "position_basis": "zero_based_binder_sequence", "parameters": asdict(parameters),
        "mutable_count": len(mutable), "candidate_pair_count": len(pairs),
        "screened_pair_count": len(selected_pairs), "geometric_hits": geometric_hits,
        "clash_rejected": clash_rejected, "full_pose_trials": trials,
        "preference_count": len(prefs), "virtual_witnesses": witnesses,
        "geometry_validated_preference_count": strict_count,
        "fallback_enabled": bool(fallback_to_best),
        "fallback_used": bool(prefs and not strict_count), **fallback_details,
        "observed_network": False, "role": "proposal_only_virtual_rotamers_not_AF3_evidence",
        "cache_hit": False, "elapsed_seconds": time.monotonic() - started}
    _CACHE[key] = copy.deepcopy((prefs, evidence))
    return prefs, evidence
