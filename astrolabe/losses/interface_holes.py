"""Native Rosetta Holes score for the two-sided protein interface in a full pose."""
from __future__ import annotations

import math
import numpy as np
from scipy.spatial import cKDTree
from astrolabe.losses.base import loss_registry


def interface_residues(pose, chains_a, chains_b, distance=10.0):
    """Return pose indices with a cross-group heavy-atom neighbor (inclusive)."""
    if not math.isfinite(distance) or distance <= 0:
        raise ValueError("interface_distance must be positive and finite")
    a, b = set(chains_a), set(chains_b)
    present = {str(pose.pdb_info().chain(i)) for i in range(1, pose.total_residue()+1)}
    if not a or not b or a & b or not (a | b) <= present:
        raise ValueError("Interface groups must exist, be nonempty and disjoint")
    coords, owners = [[], []], [[], []]
    for i in range(1, pose.total_residue()+1):
        residue = pose.residue(i)
        chain = str(pose.pdb_info().chain(i))
        if not residue.is_protein() or chain not in a | b:
            continue
        side = 0 if chain in a else 1
        for j in range(1, residue.nheavyatoms()+1):
            p = residue.xyz(j)
            coords[side].append([p.x, p.y, p.z])
            owners[side].append(i)
    if not all(coords):
        raise ValueError("Interface groups must contain protein heavy atoms")
    if not all(np.isfinite(np.asarray(c)).all() for c in coords):
        raise ValueError("Nonfinite interface coordinates")
    neighbors = cKDTree(coords[0]).query_ball_tree(cKDTree(coords[1]), distance)
    selected = set()
    for i, indices in enumerate(neighbors):
        if indices:
            selected.add(owners[0][i])
            selected.update(owners[1][j] for j in indices)
    if not selected:
        raise ValueError("No cross-group protein interface at the requested distance")
    return sorted(selected)


def _selected_interface_residues(pose, chains_a, chains_b, distance, selected_residues=None):
    selected = interface_residues(pose, chains_a, chains_b, distance)
    if selected_residues is not None:
        selected = sorted(set(selected) & set(selected_residues))
        if not selected:
            raise ValueError("Selected scoring region contains no interface residues")
    return selected


def native_interface_holes(pose, chains_a, chains_b, distance=10.0, *, selected_residues=None):
    from pyrosetta.rosetta.core.select.residue_selector import ResidueIndexSelector
    from pyrosetta.rosetta.protocols.simple_filters import HolesFilter
    selected = _selected_interface_residues(pose, chains_a, chains_b, distance, selected_residues)
    filt = HolesFilter()
    filt.set_residue_selector(ResidueIndexSelector(",".join(map(str, selected))))
    # One call on the complete pose; the direct constructor sums selected atom scores.
    value = float(filt.report_sm(pose))
    if not math.isfinite(value):
        raise ValueError("Rosetta interface Holes returned a nonfinite value")
    return value, selected


def native_interface_holes_details(pose, chains_a, chains_b, distance=10.0, *, selected_residues=None):
    """One native packing result supplies both sum and the exact normalization.

    N counts entries in native atom_scores for selected residues, matching the
    HolesFilter loop. It is neither a heavy-atom nor a nonzero-score count.
    """
    from pyrosetta.rosetta.core.scoring.packing import HolesParams, compute_holes_score
    from pyrosetta.rosetta.basic.database import full_name
    from pyrosetta.rosetta.core.id import AtomID
    selected=_selected_interface_residues(pose,chains_a,chains_b,distance,selected_residues)
    result=compute_holes_score(pose,HolesParams(full_name('scoring/rosettaholes/decoy15.params')))
    scores=result.atom_scores
    count=sum(scores.n_atom(r) for r in selected)
    value=sum(sum(float(scores.get(AtomID(a,r))) for a in range(1,scores.n_atom(r)+1)) for r in selected)
    if count<1 or not math.isfinite(value):
        raise ValueError('Invalid native interface holes result/count')
    return dict(sum=value,atom_count=count,mean=value/count,residue_count=len(selected),selected_residues=selected)


@loss_registry.register(
    name="interface_holes",
    description="Native full-pose HolesFilter score over both sides of a protein interface; lower is better",
    required_params=["protomer_a", "protomer_b"],
    optional_params=["interface_distance", "backend", "protomer_a_residues", "protomer_b_residues",
                     "position_selector_a", "position_selector_b"], supports_oligomer_specific=True,
)
def interface_holes(eval_result, protomer_a, protomer_b, interface_distance=10.0, backend="auto",
                    protomer_a_residues=None, protomer_b_residues=None,
                    position_selector_a=None, position_selector_b=None):
    """Retain the calibrated RosettaHoles definition; no SASA proxy substitution."""
    from astrolabe.losses.metric_dispatch import score_interface_metric
    if not math.isfinite(interface_distance) or interface_distance <= 0:
        raise ValueError("interface_distance must be positive and finite")
    collected = []

    def native(pose, a, b, selection, *, interface_distance):
        selected = None if selection is None else [i + 1 for i in (*selection.residues_a, *selection.residues_b)]
        if getattr(eval_result, "collect_interface_holes_details", False):
            detail = native_interface_holes_details(pose, a, b, float(interface_distance), selected_residues=selected)
            collected.append(detail)
            return detail["sum"]
        return native_interface_holes(pose, a, b, float(interface_distance), selected_residues=selected)[0]

    values = score_interface_metric(
        eval_result, "interface_holes", native=native, backend=backend,
        protomer_a=protomer_a, protomer_b=protomer_b,
        protomer_a_residues=protomer_a_residues, protomer_b_residues=protomer_b_residues,
        position_selector_a=position_selector_a, position_selector_b=position_selector_b,
        force_native_reason="RosettaHoles decoy15 has no validated independent equivalent",
        native_definition="RosettaHoles_decoy15_selected_atom_sum",
        native_options={"interface_distance": interface_distance},
    )
    if collected:
        eval_result._interface_holes_details = dict(zip(values, collected))
    return values
