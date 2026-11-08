"""Accepted RFD4 physical interface scores, separate from general losses."""
import math

from astrolabe.losses.base import loss_registry


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
    required_params=["binder_chains", "target_chains"],
    optional_params=["hbond_energy_cutoff", "interface_distance", "buried_sasa_cutoff", "sasa_probe_radius"],
    description="Rosetta whole-group shape complementarity scalar; measured empty interface is zero",
    supports_oligomer_specific=True,
)
def interface_shape_complementarity(eval_result, binder_chains, target_chains, **options):
    return _interface_feedback_scores(
        eval_result, binder_chains, target_chains, "shape_complementarity", **options,
    )


@loss_registry.register(
    name="interface_hbond_count",
    required_params=["binder_chains", "target_chains"],
    optional_params=["hbond_energy_cutoff", "interface_distance", "buried_sasa_cutoff", "sasa_probe_radius"],
    description="Count qualifying Rosetta HBondSet donor-H/acceptor observations across binder/target groups once each; no per-residue double counting",
    supports_oligomer_specific=True,
)
def interface_hbond_count(eval_result, binder_chains, target_chains, **options):
    return _interface_feedback_scores(
        eval_result, binder_chains, target_chains, "interfacial_hbond_count", **options,
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


