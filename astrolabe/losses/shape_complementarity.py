"""Spatial searches over Rosetta's unchanged molecular surface dots.

The float arithmetic, last-neighbor tie rule, and histogram interpolation
follow Rosetta 5e498f1409c68ade56c8ce5842bf79e1b02e8db4, core/scoring/sc.
Surface generation remains in the installed, licensed Rosetta library.
"""
from __future__ import annotations

import math
import time
from collections import Counter
from functools import lru_cache

import numpy as np
from scipy.spatial import cKDTree


@lru_cache(maxsize=1)
def _supported_native_build():
    import pyrosetta
    return (
        "5e498f1409c68ade56c8ce5842bf79e1b02e8db4" in pyrosetta.version()
        and issubclass(pyrosetta.rosetta.core.scoring.sc.Atom,
                       pyrosetta.rosetta.numeric.xyzVector_float_t)
    )


def supports_spatial_sc(pose):
    """Keep other native builds and nonstandard fold trees on the existing path."""
    if not _supported_native_build() or pose.num_chains() != 2:
        return False
    from pyrosetta.rosetta.core.pose.symmetry import is_symmetric
    if is_symmetric(pose) or pose.fold_tree().num_jump() != 1:
        return False
    edge = pose.fold_tree().jump_edge(1)
    if pose.chain(edge.start()) == pose.chain(edge.stop()):
        return False
    return all(pose.residue(index).is_protein() for index in range(1, pose.total_residue() + 1))


def _squared_distance(a, b):
    delta = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    return (delta[..., 0] * delta[..., 0] + delta[..., 1] * delta[..., 1]
            + delta[..., 2] * delta[..., 2])


def _surface_calculator(pose, native=False):
    from pyrosetta.rosetta.core.scoring import sc

    class Surface(sc.MolecularSurfaceCalculator):
        def AssignAttentionNumbers(self, atoms):
            coords = np.asarray([[atom.x, atom.y, atom.z] for atom in atoms], dtype=np.float32)
            groups = np.asarray([atom.molecule for atom in atoms])
            for group in (0, 1):
                own = np.flatnonzero(groups == group)
                other = coords[groups != group]
                for start in range(0, len(own), 128):
                    indices = own[start:start + 128]
                    distances = np.sqrt(_squared_distance(coords[indices, None, :], other[None, :, :]))
                    attention = np.min(distances, axis=1) < np.float32(8)
                    for index, buried in zip(indices, attention):
                        atoms[int(index)].atten = (
                            sc.ATTEN_BURIED_FLAGGED if buried else sc.ATTEN_BLOCKER
                        )
            return 1

    calculator = sc.ShapeComplementarityCalculator() if native else Surface()
    if not calculator.Init():
        raise RuntimeError("Rosetta could not initialize the SC surface calculator")
    if pose.num_chains() != 2:
        raise ValueError("SC surface calculation requires an isolated two-chain pose")
    for residue_index in range(1, pose.total_residue() + 1):
        residue = pose.residue(residue_index)
        if residue.type().name() == "VRT" or residue.type().is_metal():
            continue
        for atom_index in range(1, residue.nheavyatoms() + 1):
            if residue.is_virtual(atom_index):
                continue
            xyz = residue.xyz(atom_index)
            atom = sc.Atom()
            atom.x, atom.y, atom.z = xyz.x, xyz.y, xyz.z
            atom.nresidue = residue.seqpos()
            atom.radius = residue.atom_type(atom_index).lj_radius()
            atom.atom_type_radius = atom.radius
            if not calculator.AddAtom(0 if pose.chain(residue_index) == 1 else 1, atom):
                raise RuntimeError("Invalid Rosetta atom radius for shape complementarity")
    return calculator


def _trim(dots):
    if not len(dots):
        raise RuntimeError("No molecular surface dots were generated")
    coordinates, directions, flags = [], [], []
    for dot in dots:
        xyz, normal = dot.coor, dot.outnml
        coordinates.append((xyz.x, xyz.y, xyz.z))
        directions.append((normal.x, normal.y, normal.z))
        flags.append(dot.buried != 0)
    coords = np.asarray(coordinates, dtype=np.float32)
    normals = np.asarray(directions, dtype=np.float32)
    buried = np.asarray(flags)
    indices = np.flatnonzero(buried)
    accessible = coords[~buried]
    if len(accessible):
        tree = cKDTree(accessible)
        distances, neighbors = tree.query(coords[indices], distance_upper_bound=1.500002)
        keep = np.ones(len(indices), dtype=bool)
        present = np.flatnonzero(np.isfinite(distances))
        d2 = _squared_distance(coords[indices[present]], accessible[neighbors[present]])
        keep[present] = d2 > np.float32(2.25)
        # A double-precision nearest point can differ from the nearest under
        # Rosetta's float arithmetic at a boundary. Recheck every such candidate.
        for pos in present[keep[present]]:
            candidates = tree.query_ball_point(coords[indices[pos]], 1.500002)
            if np.any(_squared_distance(coords[indices[pos]], accessible[candidates])
                      <= np.float32(2.25)):
                keep[pos] = False
        indices = indices[keep]
    if not len(indices):
        raise RuntimeError("No molecular surface dots remain after peripheral trimming")
    return coords[indices], normals[indices], indices


def _median_shape(coords, normals, other_coords, other_normals):
    tree = cKDTree(other_coords)
    distances, nearest = tree.query(coords, k=2)
    selected = nearest[:, 0].copy()
    minimum = _squared_distance(other_coords[selected], coords)
    radius = np.nextafter(
        np.maximum(distances[:, 0], math.sqrt(float(np.nextafter(np.float32(0), np.float32(1)))))
        * (1 + 8 * np.finfo(np.float32).eps), np.inf,
    )
    # The second double-distance neighbor bounds all candidates that can tie
    # or reverse order after native float rounding; most dots need no tie loop.
    for index in np.flatnonzero(distances[:, 1] <= radius):
        neighbors = tree.query_ball_point(coords[index], radius[index])
        neighbors = np.asarray(neighbors, dtype=np.intp)
        d2 = _squared_distance(other_coords[neighbors], coords[index])
        minimum[index] = np.min(d2)
        # Native search uses <= and therefore retains the last dot on a tie.
        selected[index] = np.max(neighbors[d2 == minimum[index]])
    if np.any(minimum > np.float32(999999)):
        raise RuntimeError("SC surface separation exceeds Rosetta's neighbor search limit")
    distance = np.sqrt(minimum)
    products = normals * other_normals[selected]
    normal = (products[:, 0] + products[:, 1]) + products[:, 2]
    attenuation = np.fromiter(
        (math.exp(-math.pow(float(value), 2) * 0.5) for value in distance),
        dtype=np.float64, count=len(distance),
    )
    weighted = (normal.astype(np.float64) * attenuation).astype(np.float32)
    weighted = np.clip(weighted.astype(np.float64), -0.999, 0.999).astype(np.float32)
    scaled = (weighted.astype(np.float64) / 0.02).astype(np.float32)
    bin_indices = scaled.astype(np.int32) - (scaled < 0)
    bins = Counter(int(value) for value in bin_indices)
    cumulative = np.float32(0)
    median = np.float32(0)
    for bin_index, count in sorted(bins.items()):
        percent = np.float32(np.float32(np.float32(count) * np.float32(100))
                             / np.float32(len(coords)))
        following = np.float32(cumulative + percent)
        if cumulative <= 50 and following >= 50:
            left = np.float32(bin_index * 0.02)
            median = np.float32(
                float(left) + float(np.float32(np.float32(50) - cumulative)) * 0.02
                / float(np.float32(following - cumulative))
            )
        cumulative = following
    return float(-median), selected, dict(bins)


def spatial_shape_complementarity(pose, *, diagnostic=False):
    """Use native float surface geometry and preserve its SC histogram semantics."""
    started = time.perf_counter()
    calculator = _surface_calculator(pose)
    loaded = time.perf_counter()
    if not calculator.Calc():
        raise RuntimeError("Rosetta molecular surface generation failed")
    generated = time.perf_counter()
    surfaces = [_trim(calculator.GetDots(group)) for group in (0, 1)]
    trimmed = time.perf_counter()
    results = [_median_shape(surfaces[group][0], surfaces[group][1],
                             surfaces[1-group][0], surfaces[1-group][1]) for group in (0, 1)]
    score = (results[0][0] + results[1][0]) / 2
    if diagnostic:
        return score, dict(load_s=loaded-started, surface_s=generated-loaded,
                           trim_s=trimmed-generated, neighbor_s=time.perf_counter()-trimmed,
                           dots=[len(calculator.GetDots(group)) for group in (0, 1)],
                           trimmed_dots=[len(surface[0]) for surface in surfaces],
                           directional_scores=[result[0] for result in results],
                           bins=[result[2] for result in results])
    return score
