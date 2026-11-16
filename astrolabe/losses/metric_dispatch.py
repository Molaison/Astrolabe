"""One explicit backend and residue-selection boundary for interface scores."""
from __future__ import annotations

import importlib.metadata
import math
import sys
import warnings

import numpy as np

from astrolabe.losses.selection import (
    resolve_chain_groups, select_atoms, select_pose, validate_region_arguments,
)


def _sources(eval_result, *, prefer_paths=False):
    poses = getattr(getattr(eval_result, "complex_poses", None), "poses", None) or {}
    paths = getattr(eval_result, "complex_paths", None) or {}
    names = list(getattr(eval_result, "names", None) or (paths if prefer_paths and paths else dict.fromkeys([*poses, *paths])))
    result = {}
    for name in names:
        # A supplied pose may contain a deliberate post-prediction intervention.
        # Do not silently score an older structure path in its place.
        if prefer_paths and name in paths and paths[name] is not None:
            result[name] = (paths[name], False)
        elif name in poses and poses[name] is not None:
            result[name] = (poses[name], True)
        elif name in paths and paths[name] is not None:
            result[name] = (paths[name], False)
        else:
            raise ValueError(f"No scored structure is available for oligomer {name!r}")
    return result


def _rosetta_version():
    module = sys.modules.get("pyrosetta")
    version = getattr(module, "version", None)
    if callable(version):
        return str(version())
    try:
        return importlib.metadata.version("pyrosetta")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def score_interface_metric(eval_result, metric, *, native, backend="auto", protomer_a=None,
                           protomer_b=None, included_protomers=None, protomer_a_residues=None,
                           protomer_b_residues=None, position_selector_a=None,
                           position_selector_b=None, independent_options=None,
                           native_options=None, native_definition=None, native_radii=None,
                           force_native_reason=None, allow_missing_pair=False,
                           prefer_paths=False, native_uses_source=False,
                           native_incompatible_options=()):
    """Fallback only for unavailable implementations, never for invalid scores.

    ``native`` receives (pose, chains_a, chains_b, selection, **options).
    Selection is None when no residue restriction was requested, retaining the
    existing full-interface native behavior without another walk over the pose.
    """
    if not isinstance(backend, str) or backend not in {"auto", "independent", "rosetta"}:
        raise ValueError("backend must be auto, independent, or rosetta")
    validate_region_arguments(protomer_a_residues, protomer_b_residues,
                              position_selector_a, position_selector_b)
    regions = dict(residues_a=protomer_a_residues, residues_b=protomer_b_residues,
                   selector_a=position_selector_a, selector_b=position_selector_b)
    restricted = any(v is not None for v in regions.values())
    independent_options = dict(independent_options or {})
    native_options = dict(native_options or {})
    values, records = {}, {}
    for oligo, (source, is_pose) in _sources(eval_result, prefer_paths=prefer_paths).items():
        a, b = resolve_chain_groups(oligo, protomer_a, protomer_b, included_protomers)
        if not a or not b:
            if allow_missing_pair and not restricted:
                values[oligo] = 0.0
                records[oligo] = dict(metric=metric, backend="none", method="no_chain_pair", units="dimensionless")
                continue
            raise ValueError(f"{metric} requires two nonempty disjoint chain groups for {oligo!r}")

        result = selected = None
        fallback_reason = force_native_reason
        if backend != "rosetta":
            from astrolabe.losses import independent
            try:
                if force_native_reason is not None:
                    raise independent.BackendUnavailable(force_native_reason)
                atoms = independent.load_structure(source)
                selected, mask_a, mask_b = select_atoms(eval_result, oligo, atoms, a, b, **regions)
                options = dict(independent_options)
                if metric == "interface_buns":
                    options.update(group_a=np.isin(atoms.chain_id, a), group_b=np.isin(atoms.chain_id, b))
                result = getattr(independent, metric)(atoms, mask_a, mask_b, **options)
            except independent.BackendUnavailable as exc:
                if backend == "independent":
                    raise
                fallback_reason = str(exc)

        if result is not None:
            value = float(result.value)
            record = dict(result.details)
            record.update(metric=metric, backend="independent", method=result.method,
                          implementation=result.details.get("backend"),
                          selection=selected.evidence(), options=independent_options)
            if metric in {"hbond_count", "interface_buns"}:
                warnings.warn(
                    f"{metric} uses {result.method}; geometric counts are not numerically equivalent "
                    "to Rosetta scores. Recalibrate thresholds or set backend='rosetta' for legacy reproduction.",
                    UserWarning, stacklevel=3,
                )
        else:
            from astrolabe.losses.interaction import load_pose
            if native_incompatible_options:
                raise ValueError("The Rosetta backend does not implement independent geometry options: "
                                 + ", ".join(native_incompatible_options))
            if backend == "auto" and fallback_reason is not None and force_native_reason is None and not is_pose:
                from astrolabe.losses import interaction
                from astrolabe.utils.pyrosetta_init import pyrosetta_was_initialized
                interaction._require_pyrosetta()
                if not pyrosetta_was_initialized(interaction.pyrosetta):
                    raise RuntimeError(
                        f"{metric}: independent backend is unavailable ({fallback_reason}), but PyRosetta "
                        "has not been initialized with this run's chemistry. Set backend='rosetta' before "
                        "startup so the existing initialization owner can prepare the required parameters."
                    )
            pose = source if is_pose or (native_uses_source and not restricted) else load_pose(source)
            selected = select_pose(eval_result, oligo, pose, a, b, **regions) if restricted else None
            value = float(native(source if native_uses_source else pose, a, b, selected, **native_options))
            record = dict(metric=metric, backend="rosetta", method=native_definition or metric,
                          version=_rosetta_version(), options=native_options,
                          selection=selected.evidence() if selected is not None else
                          {"numbering": "one_based_chain_relative", "a": {c: "all" for c in a},
                           "b": {c: "all" for c in b}})
            record["units"] = {"shape_complementarity": "dimensionless", "contact_molecular_surface": "angstrom^2",
                               "interface_holes": "RosettaHoles_decoy15_atom_score_sum"}.get(metric, "count")
            if metric in {"shape_complementarity", "contact_molecular_surface"}:
                record["radii"] = native_radii if native_radii is not None else (
                    "Rosetta LJ" if native_options.get("use_rosetta_radii", True) else "Rosetta classic SC"
                )
            if fallback_reason is not None:
                record["fallback_reason"] = fallback_reason
                warnings.warn(f"{metric}: independent backend unavailable; using Rosetta ({fallback_reason})",
                              UserWarning, stacklevel=3)
        if not math.isfinite(value):
            raise ValueError(f"{metric} returned a nonfinite score for {oligo!r}")
        values[oligo], records[oligo] = value, record

    # Do not publish partial backend evidence if another oligomer failed.
    existing = getattr(eval_result, "loss_backend_records", None)
    if existing is None:
        existing = {}
        eval_result.loss_backend_records = existing
    for oligo, record in records.items():
        existing.setdefault(oligo, []).append(record)
    return values
