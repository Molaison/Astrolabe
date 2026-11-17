"""Accepted RFD4 physical interface scores, separate from general losses."""
import math

from astrolabe.losses.base import loss_registry


def _interface_feedback_scores(eval_result, binder_chains, target_chains, metric, *, empty_interface_score=None, selector_a=None, selector_b=None, **options):
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
        if selector_a is not None or selector_b is not None:
            if metric != "interfacial_hbond_count":
                raise ValueError("This native feedback metric does not expose positional selection")
            from astrolabe.losses.selection import selected_residue_ordinals
            lengths = {}
            for residue in feedback.residues:
                ref = residue.residue
                lengths[ref.chain] = max(lengths.get(ref.chain, 0), ref.position)
            first = selected_residue_ordinals(
                eval_result, name, feedback.binder_chains, chain_lengths=lengths, selector=selector_a,
            )
            second = selected_residue_ordinals(
                eval_result, name, feedback.target_chains, chain_lengths=lengths, selector=selector_b,
            )
            def selected(atom, group):
                return atom.position in group.get(atom.chain, ())
            value = sum(
                (selected(h.donor, first) and selected(h.acceptor, second))
                or (selected(h.donor, second) and selected(h.acceptor, first))
                for h in feedback.interfacial_hbonds
            )
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
    optional_params=["backend", "selector_a", "selector_b", "hbond_energy_cutoff",
                     "interface_distance", "buried_sasa_cutoff", "sasa_probe_radius"],
    description="Lawrence-Colman whole-group shape complementarity, sc_rs or explicit Rosetta",
    supports_oligomer_specific=True,
)
def interface_shape_complementarity(
    eval_result, binder_chains, target_chains, backend="sc_rs",
    selector_a=None, selector_b=None, **options,
):
    from astrolabe.losses.independent import validate_backend
    from astrolabe.losses.interaction import protomer_shape_complementarity

    validate_backend(backend, "sc_rs")
    if backend == "rosetta" and selector_a is None and selector_b is None:
        return _interface_feedback_scores(
            eval_result, binder_chains, target_chains, "shape_complementarity", **options,
        )
    if options:
        raise ValueError("RFD4 energy/SASA feedback options require backend='rosetta'")
    return protomer_shape_complementarity(
        eval_result, binder_chains, target_chains, backend=backend,
        selector_a=selector_a, selector_b=selector_b, strict=True,
    )


@loss_registry.register(
    name="interface_hbond_count",
    required_params=["binder_chains", "target_chains"],
    optional_params=["backend", "selector_a", "selector_b", "hydrogens", "hydrogen_ph",
                     "hbond_distance", "hbond_angle", "hbond_energy_cutoff",
                     "interface_distance", "buried_sasa_cutoff", "sasa_probe_radius"],
    description="Crossgroup D-H-A observation count; chemically typed Baker-Hubbard or explicit Rosetta energy definition",
    supports_oligomer_specific=True,
)
def interface_hbond_count(
    eval_result, binder_chains, target_chains, backend="biotite",
    selector_a=None, selector_b=None, hydrogens="add", hydrogen_ph=7.0,
    hbond_distance=2.5, hbond_angle=120.0, **options,
):
    from astrolabe.losses.independent import interface_scores, validate_backend

    validate_backend(backend, "biotite")
    if backend == "biotite":
        if options:
            raise ValueError("Rosetta H-bond energy/SASA options require backend='rosetta'")
        return interface_scores(
            eval_result, "hbonds", binder_chains, target_chains,
            selector_a=selector_a, selector_b=selector_b,
            hydrogens=hydrogens, hydrogen_ph=hydrogen_ph,
            hbond_distance=hbond_distance, hbond_angle=hbond_angle,
        )
    from astrolabe.losses.independent import validate_rosetta_polar_options
    validate_rosetta_polar_options(hydrogens, hydrogen_ph, hbond_distance, hbond_angle)
    return _interface_feedback_scores(
        eval_result, binder_chains, target_chains, "interfacial_hbond_count",
        selector_a=selector_a, selector_b=selector_b, **options,
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


