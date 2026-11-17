"""Native Rosetta Holes score for the two-sided protein interface in a full pose."""
from __future__ import annotations

import math
import numpy as np
from astrolabe.losses.base import loss_registry


def interface_residues(pose, chains_a, chains_b, distance=10.0):
    """Return pose indices with a cross-group heavy-atom neighbor (inclusive)."""
    from scipy.spatial import cKDTree

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


def native_interface_holes(pose, chains_a, chains_b, distance=10.0, selected_residues=None):
    from pyrosetta.rosetta.core.select.residue_selector import ResidueIndexSelector
    from pyrosetta.rosetta.protocols.simple_filters import HolesFilter
    selected = interface_residues(pose, chains_a, chains_b, distance)
    if selected_residues is not None:
        selected = sorted(set(selected) & set(selected_residues))
        if not selected:
            return 0.0, []
    filt = HolesFilter()
    filt.set_residue_selector(ResidueIndexSelector(",".join(map(str, selected))))
    # One call on the complete pose; the direct constructor sums selected atom scores.
    value = float(filt.report_sm(pose))
    if not math.isfinite(value):
        raise ValueError("Rosetta interface Holes returned a nonfinite value")
    return value, selected


def native_interface_holes_details(pose, chains_a, chains_b, distance=10.0, selected_residues=None):
    """One native packing result supplies both sum and the exact normalization.

    N counts entries in native atom_scores for selected residues, matching the
    HolesFilter loop. It is neither a heavy-atom nor a nonzero-score count.
    """
    from pyrosetta.rosetta.core.scoring.packing import HolesParams, compute_holes_score
    from pyrosetta.rosetta.basic.database import full_name
    from pyrosetta.rosetta.core.id import AtomID
    selected=interface_residues(pose,chains_a,chains_b,distance)
    if selected_residues is not None:
        selected = sorted(set(selected) & set(selected_residues))
        if not selected:
            return dict(sum=0.0,atom_count=0,mean=0.0,residue_count=0,selected_residues=[])
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
    optional_params=["interface_distance", "backend", "selector_a", "selector_b"], supports_oligomer_specific=True,
)
def interface_holes(eval_result, protomer_a, protomer_b, interface_distance=10.0,
                    backend="rosetta", selector_a=None, selector_b=None):
    if backend != "rosetta":
        raise ValueError("The trained RosettaHoles score requires backend='rosetta'; a geometric cavity score is not equivalent")

    from astrolabe.losses.interaction import _resolve_contact_chain_groups, validate_load_poses
    poses = getattr(getattr(eval_result, "complex_poses", None), "poses", None)
    poses = validate_load_poses(poses, getattr(eval_result, "complex_paths", None))
    values = {}
    for oligo, pose in poses.items():
        a, b = _resolve_contact_chain_groups(oligo, protomer_a, protomer_b)
        extra = {}
        if selector_a is not None or selector_b is not None:
            from astrolabe.losses.interaction import _loss_pose_selector
            selected = []
            for chains, expression in ((a, selector_a), (b, selector_b)):
                mask = _loss_pose_selector(eval_result, oligo, pose, chains, expression).apply(pose)
                selected.extend(i for i in range(1, pose.total_residue() + 1) if mask[i])
            extra["selected_residues"] = selected
        if getattr(eval_result,'collect_interface_holes_details',False):
            details=native_interface_holes_details(pose,a,b,float(interface_distance),**extra)
            if not hasattr(eval_result,'_interface_holes_details'):
                eval_result._interface_holes_details={}
            eval_result._interface_holes_details[oligo]=details
            values[oligo]=details['sum']
        else:
            values[oligo], _ = native_interface_holes(pose, a, b, float(interface_distance), **extra)
    if set(values) != set(eval_result.names):
        raise ValueError("Missing interface Holes result for an evaluated oligomer")
    return values
