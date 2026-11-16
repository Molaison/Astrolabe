"""Accepted RFD4 physical interface scores, separate from general losses."""
import math
from pathlib import Path

from astrolabe.losses.base import loss_registry


def _group_alias(value, legacy, name):
    if value is not None and legacy is not None:
        norm = lambda v: [p.strip() for p in v.split(",")] if isinstance(v, str) else list(v)
        if norm(value) != norm(legacy):
            raise ValueError(f"{name} disagrees with its binder/target chain alias")
    return value if value is not None else legacy


def _native_feedback(source, a, b, selection, *, metric, **options):
    from astrolabe.mutators.rfd4_feedback import analyze_interface, analyze_pose_interface
    analyzer = analyze_interface if isinstance(source, (str, Path)) else analyze_pose_interface
    feedback = analyzer(source, tuple(a), tuple(b), **options)
    if metric == "shape_complementarity":
        if selection is not None:
            from astrolabe.losses.interaction import load_pose, _native_shape_score
            pose = load_pose(source) if isinstance(source, (str, Path)) else source
            return _native_shape_score(pose, a, b, selection, strict=True)
        return feedback.shape_complementarity
    if selection is None:
        return feedback.interfacial_hbond_count
    selected_a = {(chain, pos) for chain, positions in selection.positions_a.items() for pos in positions}
    selected_b = {(chain, pos) for chain, positions in selection.positions_b.items() for pos in positions}
    return sum(
        ((hb.donor.chain, hb.donor.position) in selected_a and (hb.acceptor.chain, hb.acceptor.position) in selected_b)
        or ((hb.donor.chain, hb.donor.position) in selected_b and (hb.acceptor.chain, hb.acceptor.position) in selected_a)
        for hb in feedback.interfacial_hbonds
    )


def _interface_feedback_scores(eval_result, binder_chains, target_chains, metric, *, empty_interface_score=None, **options):
    from astrolabe.mutators.rfd4_feedback import analyze_interface, analyze_pose_interface

    sources = getattr(eval_result, "complex_paths", None)
    if sources:
        # Match next-step feedback's source normalization, including H placement.
        analyzer = analyze_interface
    else:
        from astrolabe.losses.interaction import validate_load_poses

        complex_poses = getattr(eval_result, "complex_poses", None)
        sources = validate_load_poses(getattr(complex_poses, "poses", None), None)
        analyzer = analyze_pose_interface
    scores = {}
    for name, source in sources.items():
        feedback = analyzer(source, binder_chains, target_chains, **options)
        value = getattr(feedback, metric)
        if value is None:
            if empty_interface_score is None:
                raise ValueError(f"{metric} is undefined for {name!r}: empty interface polar denominator")
            value = float(empty_interface_score)
            if not math.isfinite(value):
                raise ValueError("empty_interface_score must be finite or None for an error")
        scores[name] = value
    return scores


@loss_registry.register(
    name="interface_shape_complementarity",
    optional_params=["binder_chains", "target_chains", "protomer_a", "protomer_b", "backend",
                     "protomer_a_residues", "protomer_b_residues", "position_selector_a", "position_selector_b",
                     "hbond_energy_cutoff", "interface_distance", "buried_sasa_cutoff", "sasa_probe_radius"],
    description="Whole-group shape complementarity with explicit independent/Rosetta backend provenance",
    supports_oligomer_specific=True,
)
def interface_shape_complementarity(eval_result, binder_chains=None, target_chains=None, *,
                                    protomer_a=None, protomer_b=None, backend="auto",
                                    protomer_a_residues=None, protomer_b_residues=None,
                                    position_selector_a=None, position_selector_b=None, **options):
    from astrolabe.losses.metric_dispatch import score_interface_metric
    options = {key: value for key, value in options.items() if value is not None}
    restricted = any(value is not None for value in (
        protomer_a_residues, protomer_b_residues, position_selector_a, position_selector_b,
    ))
    return score_interface_metric(
        eval_result, "shape_complementarity", native=_native_feedback, backend=backend,
        protomer_a=_group_alias(protomer_a, binder_chains, "protomer_a"),
        protomer_b=_group_alias(protomer_b, target_chains, "protomer_b"),
        protomer_a_residues=protomer_a_residues, protomer_b_residues=protomer_b_residues,
        position_selector_a=position_selector_a, position_selector_b=position_selector_b,
        native_options={"metric": "shape_complementarity", **options},
        native_definition="RosettaShapeComplementarity", prefer_paths=True, native_uses_source=True,
        native_radii="Rosetta LJ" if restricted else "Rosetta ShapeComplementarityCalculator default (not overridden)",
        force_native_reason="Legacy physical feedback options require the Rosetta analyzer" if options else None,
    )


@loss_registry.register(
    name="interface_hbond_count",
    optional_params=["binder_chains", "target_chains", "protomer_a", "protomer_b", "backend",
                     "protomer_a_residues", "protomer_b_residues", "position_selector_a", "position_selector_b",
                     "hbond_energy_cutoff", "interface_distance", "buried_sasa_cutoff", "sasa_probe_radius",
                     "cutoff_distance", "cutoff_angle", "hydrogen_policy", "ph"],
    description="Count cross-group hydrogen bonds with an explicit geometric or Rosetta definition; no per-residue double counting",
    supports_oligomer_specific=True,
)
def interface_hbond_count(eval_result, binder_chains=None, target_chains=None, *,
                          protomer_a=None, protomer_b=None, backend="auto",
                          protomer_a_residues=None, protomer_b_residues=None,
                          position_selector_a=None, position_selector_b=None,
                          cutoff_distance=2.5, cutoff_angle=120.0, hydrogen_policy="add", ph=7.0, **options):
    from astrolabe.losses.metric_dispatch import score_interface_metric
    options = {key: value for key, value in options.items() if value is not None}
    geometric = {"cutoff_distance": cutoff_distance, "cutoff_angle": cutoff_angle,
                 "hydrogen_policy": hydrogen_policy, "ph": ph}
    defaults = {"cutoff_distance": 2.5, "cutoff_angle": 120.0, "hydrogen_policy": "add", "ph": 7.0}
    return score_interface_metric(
        eval_result, "hbond_count", native=_native_feedback, backend=backend,
        protomer_a=_group_alias(protomer_a, binder_chains, "protomer_a"),
        protomer_b=_group_alias(protomer_b, target_chains, "protomer_b"),
        protomer_a_residues=protomer_a_residues, protomer_b_residues=protomer_b_residues,
        position_selector_a=position_selector_a, position_selector_b=position_selector_b,
        independent_options=geometric,
        native_options={"metric": "hbond_count", **options}, native_definition="RosettaHBondSet",
        force_native_reason="Energy/physical-feedback parameters require the Rosetta H-bond definition" if options else None,
        prefer_paths=True, native_uses_source=True,
        native_incompatible_options=tuple(k for k, v in geometric.items() if v != defaults[k]),
    )


@loss_registry.register(
    name="interface_polar_satisfaction",
    required_params=["binder_chains", "target_chains"],
    optional_params=["hbond_energy_cutoff", "interface_distance", "buried_sasa_cutoff", "sasa_probe_radius", "empty_interface_score"],
    description="Interface polar heavy-atom fraction with any actual qualifying HB, including same-chain; not valence-complete saturation; empty denominator maps to configurable objective-only score, default 0",
    supports_oligomer_specific=True,
)
def interface_polar_satisfaction(eval_result, binder_chains, target_chains, empty_interface_score=0.0, **options):
    """Map an undefined raw fraction only for the objective. None requests an error."""
    return _interface_feedback_scores(
        eval_result, binder_chains, target_chains, "interface_polar_satisfaction",
        empty_interface_score=empty_interface_score, **options,
    )


@loss_registry.register(
    name="short_range_interface_pair_energy",
    required_params=["binder_chains", "target_chains"],
    optional_params=["hbond_energy_cutoff", "interface_distance", "buried_sasa_cutoff", "sasa_probe_radius"],
    description="Signed ref2015 crossgroup EnergyGraph pair-energy sum; excludes long-range/nonpair terms; not binding ddG",
    supports_oligomer_specific=True,
)
def short_range_interface_pair_energy(eval_result, binder_chains, target_chains, **options):
    return _interface_feedback_scores(
        eval_result, binder_chains, target_chains, "short_range_interface_pair_energy", **options,
    )


