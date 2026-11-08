import json
from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import pytest

from astrolabe.mutators.rfd4_feedback import (
    HBondObservation,
    InterfaceFeedback,
    PolarAtomObservation,
    ResidueObservation,
    plan_conditions,
)
from astrolabe.mutators.rfd4_types import AtomRef, ChainSpec, ResidueRef


CHAINS = (ChainSpec("A", "binder", "SSS"), ChainSpec("B", "target", "ST"))
GROUPS = {"binder_chains": ("A",), "target_chains": ("B",)}
ELIGIBLE = tuple(ResidueRef("A", i) for i in (1, 2, 3))


def feedback():
    xyz = (0.0, 0.0, 0.0)
    residues = tuple(
        ResidueObservation(ResidueRef(chain, pos), "THR" if (chain, pos) == ("B", 2) else "SER",
                           i, 100 + i, "", energy, 0, unsat)
        for i, (chain, pos, energy, unsat) in enumerate((
            ("A", 1, -3.0, 0), ("A", 2, 2.0, 1), ("A", 3, 1.0, 3),
            ("B", 1, -1.0, 0), ("B", 2, 1.0, 1),
        ), 1)
    )
    hb = HBondObservation(
        AtomRef("A", 1, "OG"), AtomRef("A", 1, "HG"), AtomRef("B", 1, "OG"),
        xyz, xyz, xyz, -0.5, 1.0, True, True, True,
    )
    atoms = (
        PolarAtomObservation(AtomRef("B", 1, "OG"), "SER", "OH", "BOTH", xyz, 0.0, 3.0, True, True, True, 1),
        PolarAtomObservation(AtomRef("B", 2, "N"), "THR", "Nbb", "DONOR", xyz, 0.2, 3.0, True, True, False, 0),
        PolarAtomObservation(AtomRef("A", 2, "OG"), "SER", "OH", "BOTH", xyz, 0.0, 3.0, True, True, False, 0),
    )
    return InterfaceFeedback(("A",), ("B",), residues, (hb,), atoms, (), 0.5, ())


def test_retention_budget_and_selected_atoms_own_role():
    plan = plan_conditions(CHAINS, ELIGIBLE, 1, feedback=feedback())
    assert plan.released == (ResidueRef("A", 2),)
    assert ResidueRef("A", 1) in plan.fixed_coordinates
    assert set(plan.fixed_coordinates) | set(plan.released) == {
        *(ResidueRef("A", i) for i in (1, 2, 3)), ResidueRef("B", 1), ResidueRef("B", 2),
    }
    assert not set(plan.fixed_coordinates) & set(plan.released)
    assert len(plan.hbonds) == 1
    assert plan.hbonds[0].atom == AtomRef("B", 2, "N")
    assert plan.hbonds[0].role == "DONOR"
    report = json.loads(plan.feedback_json)
    assert len(report["feedback"]["polar_atoms"]) == 3
    assert report["feedback"]["hbonds"][0]["hydrogen"]["atom"] == "HG"


def test_favorable_retention_only_and_site_scores_override_ranking():
    fb = feedback()
    plan = plan_conditions(CHAINS, ELIGIBLE, 3, feedback=fb,
                           policy={"freeze_hbonds": False, "energy_top_k": 3})
    assert plan.released == (ResidueRef("A", 2), ResidueRef("A", 3))
    plan = plan_conditions(CHAINS, ELIGIBLE, 1, feedback=fb,
                           site_scores={r: float(r.position) for r in ELIGIBLE})
    assert plan.released == (ResidueRef("A", 3),)


def test_empty_masks_never_broaden_and_groups_cannot_be_inferred():
    plan = plan_conditions(CHAINS, (), 5, policy=GROUPS)
    assert not plan.released
    assert len(plan.fixed_coordinates) == 5
    with pytest.raises(ValueError, match="explicitly"):
        plan_conditions(CHAINS, (), 0)
    with pytest.raises(ValueError, match="Eligible"):
        plan_conditions(CHAINS, (ResidueRef("B", 1),), 1, policy=GROUPS)


def test_seeded_ties_and_hbond_condition_limit():
    assert plan_conditions(CHAINS, ELIGIBLE, 1, policy=GROUPS, seed=7) == plan_conditions(
        CHAINS, tuple(reversed(ELIGIBLE)), 1, policy=GROUPS, seed=7,
    )
    plan = plan_conditions(CHAINS, ELIGIBLE, 0, feedback=feedback(), policy={"max_hbond_conditions": 0})
    assert not plan.released and not plan.hbonds


def test_polar_denominator_and_any_hbond_satisfaction():
    fb = feedback()
    assert fb.interface_polar_count == 3
    assert fb.satisfied_interface_polar_count == 1
    assert fb.interface_polar_satisfaction == pytest.approx(1 / 3)
    assert replace(fb, polar_atoms=()).interface_polar_satisfaction is None
    intra = replace(fb.hbonds[0], interchain=False, crossgroup=False)
    assert not replace(fb, hbonds=(intra,)).interfacial_hbonds
    assert replace(fb, hbonds=(intra,)).satisfied_interface_polar_count == 1


def test_interfacial_hbond_count_counts_observations_not_residues_or_polar_atoms():
    fb = feedback()
    hb = fb.hbonds[0]
    second = replace(hb, hydrogen=AtomRef("A", 1, "H2"))
    same_group = replace(hb, acceptor=AtomRef("C", 1, "OG"), crossgroup=False)
    intra = replace(hb, acceptor=AtomRef("A", 2, "OG"), interchain=False, crossgroup=False)
    weak = replace(hb, energy=0.0, qualifying=False)
    fb = replace(fb, hbonds=(hb, second, same_group, intra, weak),
                 residues=tuple(replace(r, interface_hbond_count=2)
                                if r.residue in (ResidueRef("A", 1), ResidueRef("B", 1))
                                else r for r in fb.residues))
    assert fb.interfacial_hbonds == (hb, second)
    assert fb.interfacial_hbond_count == 2
    assert sum(r.interface_hbond_count for r in fb.residues) == 4
    assert fb.interface_polar_satisfaction == pytest.approx(1 / 3)
    assert fb.to_dict()["metrics"]["interfacial_hbond_count"] == 2
    assert replace(fb, hbonds=()).interfacial_hbond_count == 0


def test_interface_hbond_count_scalar_registration_and_adapter(monkeypatch):
    from types import SimpleNamespace
    from astrolabe.losses import interaction
    from astrolabe.mutators import rfd4_feedback

    fb = feedback()
    poses = {"bound": object(), "empty": object()}
    calls = []

    def analyze(pose, binder_chains, target_chains, **options):
        calls.append((binder_chains, target_chains, options))
        return fb if pose is poses["bound"] else replace(fb, hbonds=(), polar_atoms=())

    monkeypatch.setattr(rfd4_feedback, "analyze_pose_interface", analyze)
    loss = interaction.loss_registry.get_function("interface_hbond_count")
    assert interaction.loss_registry.get_metadata("interface_hbond_count").supports_oligomer_specific
    options = {"hbond_energy_cutoff": -0.25, "interface_distance": 6.0,
               "buried_sasa_cutoff": 0.5, "sasa_probe_radius": 1.2}
    result = SimpleNamespace(complex_poses=SimpleNamespace(poses=poses))
    assert loss(result, **GROUPS, **options) == {"bound": 1, "empty": 0}
    assert calls == [(GROUPS["binder_chains"], GROUPS["target_chains"], options)] * 2
    assert not interaction.loss_registry.validate_params(
        "interface_hbond_count", [*GROUPS, *options],
    )


@pytest.mark.parametrize("loss_name,expected", [
    ("interface_hbond_count", 1),
    ("interface_shape_complementarity", 0.5),
    ("interface_polar_satisfaction", 1 / 3),
    ("short_range_interface_pair_energy", 0.0),
])
def test_feedback_scalar_prefers_source_paths_over_cached_poses(monkeypatch, tmp_path, loss_name, expected):
    from types import SimpleNamespace
    from astrolabe.losses import interaction
    from astrolabe.mutators import rfd4_feedback

    paths = {"AB": tmp_path / "accepted.cif"}
    result = SimpleNamespace(complex_paths=paths,
                             complex_poses=SimpleNamespace(poses={"stale": object()}))
    calls = []
    options = {"hbond_energy_cutoff": -0.25, "interface_distance": 6.0,
               "buried_sasa_cutoff": 0.5, "sasa_probe_radius": 1.2}

    def analyze(path, binder_chains, target_chains, **kwargs):
        calls.append((path, binder_chains, target_chains, kwargs))
        return feedback()

    def reject_pose(*args, **kwargs):
        pytest.fail("Cached poses must not override source-path feedback")

    monkeypatch.setattr(rfd4_feedback, "analyze_interface", analyze)
    monkeypatch.setattr(rfd4_feedback, "analyze_pose_interface", reject_pose)
    loss = interaction.loss_registry.get_function(loss_name)
    assert loss(result, **GROUPS, **options) == {"AB": pytest.approx(expected)}
    assert calls == [(paths["AB"], GROUPS["binder_chains"], GROUPS["target_chains"], options)]

    def invalid_source(*args, **kwargs):
        raise ValueError("invalid source chemistry")

    monkeypatch.setattr(rfd4_feedback, "analyze_interface", invalid_source)
    with pytest.raises(ValueError, match="invalid source chemistry"):
        loss(result, **GROUPS, **options)


def test_plan_rejects_stale_chemistry_and_invalid_budget():
    with pytest.raises(ValueError, match="chemistry"):
        plan_conditions((ChainSpec("A", "binder", "AAA"), CHAINS[1]), ELIGIBLE, 1, feedback=feedback())
    with pytest.raises(ValueError, match="budget"):
        plan_conditions(CHAINS, ELIGIBLE, -1, policy=GROUPS)


def test_single_ccd_target_is_fixed_with_its_own_atom_chemistry():
    chains = (CHAINS[0], ChainSpec("B", "caffeine", "CFF", kind="ligand"))
    fb = feedback()
    ligand = replace(fb.residues[3], residue_name="CFF")
    acceptor = replace(
        fb.polar_atoms[0], atom=AtomRef("B", 1, "O2"), residue_name="CFF",
        role="ACCEPTOR", satisfied=False, qualifying_hbond_count=0,
    )
    fb = replace(fb, residues=fb.residues[:3] + (ligand,), hbonds=(), polar_atoms=(acceptor,))
    plan = plan_conditions(chains, ELIGIBLE, 1, feedback=fb)
    assert plan.released == (ResidueRef("A", 2),)
    assert set(plan.fixed_coordinates) | set(plan.released) == {
        *ELIGIBLE, ResidueRef("B", 1),
    }
    assert plan.hbonds[0].atom == AtomRef("B", 1, "O2")
    assert plan.hbonds[0].role == "ACCEPTOR"
    assert json.loads(plan.feedback_json)["feedback"]["residues"][-1]["residue_name"] == "CFF"
    with pytest.raises(ValueError, match="Eligible"):
        plan_conditions(chains, (ResidueRef("B", 1),), 1, feedback=fb)
    with pytest.raises(ValueError, match="chemistry"):
        plan_conditions((chains[0], replace(chains[1], sequence="GOL")), ELIGIBLE, 1, feedback=fb)
    with pytest.raises(ValueError, match="protein"):
        plan_conditions(chains, (), 0, policy={"binder_chains": ("B",), "target_chains": ("A",)})
    without_feedback = plan_conditions(chains, (), 0, policy=GROUPS)
    assert without_feedback.fixed_coordinates == (*ELIGIBLE, ResidueRef("B", 1))


@pytest.mark.parametrize("sequence", ["CFF,GOL", "[CFF]", "CFF GOL"])
def test_ligand_requires_one_explicit_ccd(sequence):
    with pytest.raises(ValueError, match="CCD"):
        plan_conditions((CHAINS[0], ChainSpec("B", "ligand", sequence, kind="ligand")),
                        (), 0, policy=GROUPS)


def test_explicit_params_copy_exact_identity_and_reject_wrong_targets(tmp_path):
    from types import SimpleNamespace
    from astrolabe.core.protomer_config_parser import ProtomerTypes
    from astrolabe.utils.extra_residues import _reuse_explicit_residue_params

    # Metadata-only fixture for staging, not a chemical definition.
    source = tmp_path / "CFF.params"
    source.write_text("NAME CFF\nIO_STRING CFF Z\nTYPE LIGAND\n")
    protomers = {"B": SimpleNamespace(type=ProtomerTypes.CCD, sequence="CFF", name="caffeine")}
    output = tmp_path / "raw_model_results"
    staged = _reuse_explicit_residue_params([source], protomers, {}, {}, output)
    assert staged == {"CFF": str(output / "CFF.params")}
    assert (output / "CFF.params").read_bytes() == source.read_bytes()
    assert _reuse_explicit_residue_params([source], protomers, {}, {}, output) == staged
    with pytest.raises(ValueError, match="configured target"):
        _reuse_explicit_residue_params([source], {}, {}, {}, output)
    source.write_text(source.read_text() + "# different identity\n")
    with pytest.raises(ValueError, match="Refusing to replace"):
        _reuse_explicit_residue_params([source], protomers, {}, {}, output)


def test_source_import_cannot_silently_drop_a_target_ligand(tmp_path, monkeypatch):
    pyrosetta = pytest.importorskip("pyrosetta")
    from astrolabe.mutators.rfd4_feedback import analyze_interface

    if not pyrosetta.rosetta.basic.was_init_called():
        pyrosetta.init("-ignore_zero_occupancy false -mute all -beta")
    binder = pyrosetta.pose_from_sequence("AA")
    for index in range(1, binder.total_residue() + 1):
        binder.pdb_info().chain(index, "A")
    source = tmp_path / "ligand.pdb"
    binder.dump_pdb(str(source))
    # Emulate an importer initialized with -ignore_unrecognized_res: it returns
    # a perfectly valid protein pose while silently dropping the target chain.
    source.write_text(source.read_text().replace("END\n", "") +
                      "HETATM  100  O2  CFF B   1      50.000  50.000  50.000  1.00 20.00           O  \nEND\n")
    monkeypatch.setattr(pyrosetta, "pose_from_file", lambda *args: binder.clone())
    with pytest.raises(ValueError, match="dropped.*chains"):
        analyze_interface(source, ("A",), ("B",), extra_residue_params=())


def test_analysis_only_reconstructed_atoms_cannot_be_conditioned():
    fb = feedback()
    fb = replace(fb, polar_atoms=tuple(replace(a, source_present=False) for a in fb.polar_atoms))
    plan = plan_conditions(CHAINS, ELIGIBLE, 1, feedback=fb)
    assert not plan.hbonds
    assert json.loads(plan.feedback_json)["feedback"]["polar_atoms"]

    fb = feedback()
    oxt = replace(fb.polar_atoms[1], atom=AtomRef("B", 2, "OXT"), role="ACCEPTOR",
                  satisfied=True, qualifying_hbond_count=1)
    fb = replace(fb, hbonds=(replace(fb.hbonds[0], acceptor=oxt.atom),),
                 polar_atoms=fb.polar_atoms + (oxt,))
    original = fb.to_json()
    plan = plan_conditions(CHAINS, ELIGIBLE, 3, feedback=fb)
    report = json.loads(plan.feedback_json)
    assert plan.released == ELIGIBLE
    assert len(report["planning"]["native_unrepresented_hbond_retention_omissions"]) == 1
    assert report["feedback"] == json.loads(original)
    assert fb.to_json() == original and oxt.source_present

    fb = replace(fb, hbonds=(), polar_atoms=fb.polar_atoms[:-1] + (
        replace(oxt, satisfied=False, qualifying_hbond_count=0),
    ))
    original = fb.to_json()
    plan = plan_conditions(CHAINS, ELIGIBLE, 1, feedback=fb)
    report = json.loads(plan.feedback_json)
    assert plan.hbonds and all(condition.atom.atom != "OXT" for condition in plan.hbonds)
    omissions = report["planning"]["native_unrepresented_atom_condition_omissions"]
    assert len(omissions) == 1 and omissions[0]["atom"]["atom"] == "OXT"
    assert "not_represented" in omissions[0]["reason"]
    assert report["planning"]["native_unrepresented_atoms"][0]["source_present"] is True
    assert fb.to_json() == original


def local_feedback():
    fb = feedback()
    coordinates = ((0., 0., 0.), (4., 0., 0.), (5., 0., 0.),
                   (0., 3., 0.), (4., 3., 0.))
    return replace(
        fb, residues=tuple(replace(r, heavy_atom_coordinates=(xyz,),
                                   nearest_other_group_heavy_atom_distance=3.)
                           for r, xyz in zip(fb.residues, coordinates)),
        polar_atoms=tuple(replace(a, xyz=coordinates[3 if a.atom.position == 1 else 4])
                          if a.atom.chain == "B" else a for a in fb.polar_atoms),
    )


def test_local_repair_defaults_reachable_goal_and_deterministic_distance_rank():
    fb = local_feedback()
    original = fb.to_json()
    plan = plan_conditions(CHAINS, ELIGIBLE, 1, feedback=fb, policy={"local_repair": True})
    assert plan.released == (ELIGIBLE[1],)  # Distance beats A3's larger unsatisfied count.
    assert plan.hbonds[0].atom == AtomRef("B", 2, "N")
    assert plan.hbonds[0].role == "DONOR"
    assert set(plan.fixed_coordinates) == {ELIGIBLE[0], ELIGIBLE[2], ResidueRef("B", 1), ResidueRef("B", 2)}
    planning = json.loads(plan.feedback_json)["planning"]
    assert planning["policy"] == {
        "local_repair": True, "freeze_hbonds": True, "repair_radius": 6.,
        "hbond_retention_cutoff": -.5, "energy_retention_cutoff": -1.,
        "energy_top_k": 2, "max_hbond_conditions": 1,
    }
    assert planning["selected_goal"]["atom"] == {"chain": "B", "position": 2, "atom": "N"}
    assert planning["local_residues"][0]["protected_reasons"] == ["favorable_short_range_pair_energy"]
    assert planning["local_residues"][1]["selected_goal_heavy_atom_distance"] == 3.
    assert planning["local_mask_reason"] and planning["rationale"]
    again = plan_conditions(CHAINS, tuple(reversed(ELIGIBLE)), 1, feedback=fb,
                            seed=932, policy={"local_repair": True},
                            site_scores={r: float(r.position) for r in ELIGIBLE})
    assert (again.released, again.fixed_coordinates, again.hbonds) == (plan.released, plan.fixed_coordinates, plan.hbonds)
    assert plan == plan_conditions(CHAINS, ELIGIBLE, 1, feedback=fb, policy={"local_repair": True})
    assert fb.to_json() == original
    assert plan_conditions(CHAINS, ELIGIBLE, 1, feedback=fb) == plan_conditions(
        CHAINS, ELIGIBLE, 1, feedback=fb, policy={"local_repair": False})
    # A3 remains an interface candidate, but only goal-local sites may consume budget.
    for goal_x, expected in ((-1., ELIGIBLE[1:]), (-1.01, (ELIGIBLE[1],))):
        goal = replace(fb.polar_atoms[1], xyz=(goal_x, 0., 0.),
                       nearest_other_group_heavy_atom_distance=abs(goal_x))
        bounded = plan_conditions(CHAINS, ELIGIBLE, 3,
                                  feedback=replace(fb, polar_atoms=(goal,)),
                                  policy={"local_repair": True})
        assert bounded.released == expected
        planning = json.loads(bounded.feedback_json)["planning"]
        assert planning["local_candidates"] == [dict(chain=r.chain, position=r.position) for r in ELIGIBLE[1:]]
        assert planning["selection_candidates"] == [dict(chain=r.chain, position=r.position) for r in expected]
        assert len(bounded.hbonds) == 1
        if goal_x < -1.:
            detail = planning["local_residues"][2]
            assert detail["candidate"] and not detail["selection_candidate"]
            assert detail["protected_reasons"] == ["outside_selected_goal_radius"]
            assert planning["chosen_reasons"][2]["reasons"] == ["outside_selected_goal_radius"]


def test_local_mask_radius_boundary_missing_geometry_and_zero_budget():
    fb = local_feedback()
    fb = replace(fb, residues=tuple(
        replace(r, nearest_other_group_heavy_atom_distance=6. if r.residue == ELIGIBLE[1] else 6.01)
        if r.residue.chain == "A" else r for r in fb.residues))
    policy = {"local_repair": True, "energy_top_k": 0, "freeze_hbonds": False}
    plan = plan_conditions(CHAINS, ELIGIBLE[1:], 3, feedback=fb, policy=policy)
    assert plan.released == (ELIGIBLE[1],)
    details = json.loads(plan.feedback_json)["planning"]["local_residues"]
    assert details[0]["protected_reasons"] == ["outside_eligible_mask"]
    assert details[2]["protected_reasons"] == ["outside_interface_repair_radius"]
    assert not plan_conditions(CHAINS, ELIGIBLE, 3, feedback=feedback(), policy=policy).released
    assert not plan_conditions(CHAINS, ELIGIBLE, 3, policy={**GROUPS, **policy}).released
    assert not plan_conditions(CHAINS, (), 3, feedback=fb, policy=policy).released
    zero = plan_conditions(CHAINS, ELIGIBLE, 0, feedback=fb, policy=policy)
    assert not zero.released and not zero.hbonds and len(zero.fixed_coordinates) == 5


@pytest.mark.parametrize("energy,source_endpoint,qualifying,retained", [
    (-.5, None, True, False), (-.51, None, True, True),
    (-.51, "donor", True, False), (-.51, "acceptor", True, False),
    (-.51, None, False, False),
])
def test_local_hbond_retention_strict_cutoff_and_source_presence(energy, source_endpoint, qualifying, retained):
    fb = local_feedback()
    donor = replace(fb.polar_atoms[-1], atom=fb.hbonds[0].donor, satisfied=True)
    hb = replace(fb.hbonds[0], energy=energy, qualifying=qualifying)
    atoms = fb.polar_atoms + (donor,)
    if source_endpoint:
        atoms = tuple(replace(a, source_present=False) if a.atom == getattr(hb, source_endpoint) else a for a in atoms)
    fb = replace(fb, hbonds=(hb,), polar_atoms=atoms)
    plan = plan_conditions(CHAINS, ELIGIBLE, 3, feedback=fb,
                           policy={"local_repair": True, "energy_top_k": 0, "max_hbond_conditions": 0})
    assert (ELIGIBLE[0] in plan.fixed_coordinates) == retained


def test_local_pair_energy_retention_is_strict_and_top_k_bounded():
    fb = local_feedback()
    fb = replace(fb, residues=tuple(replace(r, short_range_interface_pair_energy=energy)
                                   for r, energy in zip(fb.residues, (-3., -2., -1., 0., 0.))))
    policy = {"local_repair": True, "freeze_hbonds": False, "max_hbond_conditions": 0}
    assert plan_conditions(CHAINS, ELIGIBLE, 3, feedback=fb, policy=policy).released == (ELIGIBLE[2],)
    assert plan_conditions(CHAINS, ELIGIBLE, 3, feedback=fb, policy={**policy, "energy_top_k": 1}).released == ELIGIBLE[1:]
    assert plan_conditions(CHAINS, ELIGIBLE, 3, feedback=fb,
                           policy={**policy, "energy_retention_cutoff": -2.}).released == ELIGIBLE[1:]


def test_local_goal_ranks_buried_then_sasa_identity_and_caps_protocol_at_one():
    fb = local_feedback()
    buried = replace(fb.polar_atoms[0], satisfied=False, qualifying_hbond_count=0, sasa=.1)
    exposed = replace(buried, buried=False, sasa=0.)
    other = replace(buried, atom=AtomRef("B", 1, "O"), role="ACCEPTOR")
    fb = replace(fb, polar_atoms=(exposed, buried, other, fb.polar_atoms[1]))
    plan = plan_conditions(CHAINS, ELIGIBLE, 2, feedback=fb,
                           policy={"local_repair": True, "max_hbond_conditions": 5})
    assert plan.hbonds == (type(plan.hbonds[0])(other.atom, "ACCEPTOR"),)
    assert json.loads(plan.feedback_json)["planning"]["effective_max_hbond_conditions"] == 1


def test_measured_failed_goal_is_deprioritized_without_changing_geometry():
    fb = local_feedback()
    first = replace(fb.polar_atoms[0], satisfied=False, qualifying_hbond_count=0, sasa=.1)
    fb = replace(fb, polar_atoms=(first, fb.polar_atoms[1]))
    initial = plan_conditions(CHAINS, ELIGIBLE, 2, feedback=fb, policy={"local_repair": True})
    retry = plan_conditions(CHAINS, ELIGIBLE, 2, feedback=fb, policy={"local_repair": True},
                            failed_hbond_attempts={"B:1:OG": 1})
    assert initial.hbonds[0].atom == first.atom
    assert retry.hbonds[0].atom == AtomRef("B", 2, "N")
    assert json.loads(retry.feedback_json)["planning"]["failed_hbond_attempts"] == {"B:1:OG": 1}


@pytest.mark.parametrize("change", [
    {"source_present": False}, {"satisfied": True}, {"atom": AtomRef("B", 2, "OXT")},
    {"nearest_other_group_heavy_atom_distance": 6.01}, {"xyz": (100., 0., 0.)},
    {"atom": AtomRef("A", 2, "OG")},
])
def test_local_invalid_or_unreachable_goal_falls_back_to_interface_proxy_rank(change):
    fb = local_feedback()
    fb = replace(fb, polar_atoms=(replace(fb.polar_atoms[1], **change),), hbonds=())
    plan = plan_conditions(CHAINS, ELIGIBLE, 1, feedback=fb, policy={"local_repair": True})
    assert plan.released == (ELIGIBLE[2],)  # Unsatisfied count beats A2's higher pair energy.
    assert not plan.hbonds
    assert json.loads(plan.feedback_json)["planning"]["selected_goal"] is None


def test_local_goal_must_be_reachable_from_nonpreserved_allowed_residue():
    fb = local_feedback()
    goal = replace(fb.polar_atoms[1], xyz=(-4., 0., 0.))  # Only retained A1 is close.
    fb = replace(fb, polar_atoms=(goal,), hbonds=())
    plan = plan_conditions(CHAINS, ELIGIBLE, 1, feedback=fb, policy={"local_repair": True})
    assert plan.released == (ELIGIBLE[2],) and not plan.hbonds
    assert json.loads(plan.feedback_json)["planning"]["selected_goal"] is None
    tied = replace(fb, residues=tuple(replace(r, unsatisfied_interface_polar_count=1,
                                            short_range_interface_pair_energy=1.) for r in fb.residues))
    assert plan_conditions(CHAINS, tuple(reversed(ELIGIBLE)), 1, feedback=tied,
                           policy={"local_repair": True}).released == (ELIGIBLE[0],)


@pytest.mark.parametrize("policy", [
    {"local_repair": 1}, {"repair_radius": 0}, {"repair_radius": -1},
    {"repair_radius": float("nan")}, {"repair_radius": True},
    {"hbond_retention_cutoff": float("inf")}, {"hbond_retention_cutoff": .1},
    {"energy_retention_cutoff": "-1"}, {"energy_retention_cutoff": .1},
    {"energy_top_k": -1}, {"energy_top_k": True}, {"max_hbond_conditions": 1.5},
])
def test_local_policy_parameter_validation(policy):
    with pytest.raises(ValueError):
        plan_conditions(CHAINS, ELIGIBLE, 1, feedback=local_feedback(),
                        policy={"local_repair": True, **policy})


def test_contiguous_repair_keeps_protection_and_includes_connecting_flank():
    fb = local_feedback()
    fb = replace(fb, residues=tuple(
        replace(r, heavy_atom_coordinates=((15., 0., 0.),),
                nearest_other_group_heavy_atom_distance=11.)
        if r.residue == ELIGIBLE[2] else r for r in fb.residues))
    plan = plan_conditions(CHAINS, ELIGIBLE, 3, feedback=fb,
                           policy={"local_repair": True, "repair_segment_length": 2})
    assert plan.released == ELIGIBLE[1:]
    assert ELIGIBLE[0] in plan.fixed_coordinates
    assert plan.hbonds[0].atom == AtomRef("B", 2, "N")
    segment = json.loads(plan.feedback_json)["planning"]["repair_segment"]
    assert segment["status"] == "selected"
    assert segment["outside_radius_flanks"] == [{"chain": "A", "position": 3}]


def test_contiguous_repair_never_crosses_protected_or_ineligible_sites():
    fb = local_feedback()
    fb = replace(fb, residues=tuple(
        replace(r, short_range_interface_pair_energy=-10.)
        if r.residue == ELIGIBLE[1] else r for r in fb.residues))
    policy = {"local_repair": True, "repair_segment_length": 2, "energy_top_k": 1}
    for eligible, sample in ((ELIGIBLE, fb), ((ELIGIBLE[0], ELIGIBLE[2]), local_feedback())):
        plan = plan_conditions(CHAINS, eligible, 3, feedback=sample, policy=policy)
        assert not plan.released and not plan.hbonds
        assert len(plan.fixed_coordinates) == 5
        assert json.loads(plan.feedback_json)["planning"]["repair_segment"]["status"] == \
            "no_complete_eligible_segment"


def test_contiguous_repair_goal_requires_a_complete_reachable_window():
    fb = local_feedback()
    first = replace(fb.polar_atoms[0], xyz=(-5., 0., 0.), satisfied=False,
                    qualifying_hbond_count=0)
    fb = replace(fb, polar_atoms=(first, *fb.polar_atoms[1:]))
    plan = plan_conditions(CHAINS, ELIGIBLE, 2, feedback=fb,
                           failed_hbond_attempts={"B:2:N": 1},
                           policy={"local_repair": True, "repair_segment_length": 2})
    assert plan.released == ELIGIBLE[1:]
    assert plan.hbonds[0].atom == AtomRef("B", 2, "N")


@pytest.mark.parametrize("length,local,budget", [
    (None, True, 3), (True, True, 3), (1, True, 3), (2.5, True, 3),
    (2, False, 3), (3, True, 2),
])
def test_contiguous_repair_rejects_invalid_length_or_budget(length, local, budget):
    with pytest.raises(ValueError, match="repair_segment_length"):
        plan_conditions(CHAINS, ELIGIBLE, budget, feedback=local_feedback(),
                        policy={"local_repair": local, "repair_segment_length": length})


@pytest.mark.parametrize("sequence,position,expected_roles", [
    pytest.param("APA", 2, {"N": None}, id="internal-proline-no-donor"),
    pytest.param("PA", 1, {"N": "DONOR"}, id="nterminal-proline-donor"),
    pytest.param("AAA", 2, {}, id="backbone-n-donor-o-acceptor"),
    pytest.param("ANA", 2, {"ND2": "DONOR", "OD1": "ACCEPTOR"}, id="asparagine-amide"),
    pytest.param("AQA", 2, {"NE2": "DONOR", "OE1": "ACCEPTOR"}, id="glutamine-amide"),
    pytest.param("ADA", 2, {"OD1": "ACCEPTOR", "OD2": "ACCEPTOR"}, id="aspartate-carboxylate"),
    pytest.param("AEA", 2, {"OE1": "ACCEPTOR", "OE2": "ACCEPTOR"}, id="glutamate-carboxylate"),
    pytest.param("AKA", 2, {"NZ": "DONOR"}, id="lysine-ammonium"),
    pytest.param("ARA", 2, {"NE": "DONOR", "NH1": "DONOR", "NH2": "DONOR"}, id="arginine-guanidinium"),
    pytest.param("ASA", 2, {"OG": "BOTH"}, id="serine-hydroxyl"),
    pytest.param("ATA", 2, {"OG1": "BOTH"}, id="threonine-hydroxyl"),
    pytest.param("AYA", 2, {"OH": "BOTH"}, id="tyrosine-hydroxyl"),
    pytest.param("AHA", 2, {"ND1": "ACCEPTOR", "NE2": "DONOR"}, id="histidine-epsilon-proton"),
    pytest.param("AH[HIS_D]A", 2, {"ND1": "DONOR", "NE2": "ACCEPTOR"}, id="histidine-delta-proton"),
    pytest.param("AH[HIS_P]A", 2, {"ND1": "DONOR", "NE2": "DONOR"}, id="histidine-diprotonated"),
    # Rosetta's neutral SH1 thiol is not an HBondSet donor or acceptor.
    pytest.param("ACA", 2, {"SG": None}, id="cysteine-thiol"),
    pytest.param("ACACA", 2, {"SG": None}, id="cystine-disulfide"),
])
def test_analyze_pose_interface_real_residue_polar_roles(sequence, position, expected_roles):
    if sequence != "ACA":
        _assert_real_residue_polar_roles(sequence, position, expected_roles)
        return

    pytest.importorskip("pyrosetta")
    # SH1 donor typing differs between default and -beta initialization. Other
    # tests may already have initialized Rosetta; never reset that live runtime.
    # Keep this test's original -beta chemistry contract in a fresh interpreter.
    result = subprocess.run(
        [sys.executable, "-c", (
            "import json, runpy, sys; "
            "runpy.run_path(sys.argv[1])['_assert_real_residue_polar_roles']"
            "(*json.loads(sys.argv[2]))"
        ), str(Path(__file__).resolve()), json.dumps([sequence, position, expected_roles])],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _assert_real_residue_polar_roles(sequence, position, expected_roles):
    pyrosetta = pytest.importorskip("pyrosetta")
    from astrolabe.mutators.rfd4_feedback import analyze_pose_interface

    r = pyrosetta.rosetta
    if not r.basic.was_init_called():
        pyrosetta.init("-ignore_zero_occupancy false -mute all -beta")
    if "[HIS_P]" in sequence:
        # Use Rosetta's actual protonation-state params, registered only on
        # this pose: default fa_standard does not enable HIS_P without pH mode.
        pose = pyrosetta.Pose()
        params = r.basic.database.full_name(
            "chemical/residue_type_sets/fa_standard/residue_types/protonation_states/HIS_P.params"
        )
        residue_types = pyrosetta.generate_nonstandard_residue_set(pose, [params])
        r.core.pose.make_pose_from_sequence(pose, sequence, residue_types, True)
        assert pose.residue(position).name() == "HIS_P"
    else:
        pose = pyrosetta.pose_from_sequence(sequence)
    positions = (position,)
    if sequence == "ACACA":
        r.core.conformation.form_disulfide(pose.conformation(), 2, 4)
        positions = (2, 4)
        for index in positions:
            assert pose.residue(index).has_variant_type(r.core.chemical.DISULFIDE)
        assert pose.conformation().is_bonded(
            r.core.id.AtomID(pose.residue(2).atom_index("SG"), 2),
            r.core.id.AtomID(pose.residue(4).atom_index("SG"), 4),
        )

    # A distant real partner keeps these role checks independent of contact
    # geometry and the shape-complementarity backend, without mocking either.
    binder_size = pose.total_residue()
    partner = pyrosetta.pose_from_sequence("AA")
    for index in range(1, partner.total_residue() + 1):
        for atom in range(1, partner.residue(index).natoms() + 1):
            partner.set_xyz(r.core.id.AtomID(atom, index), partner.residue(index).xyz(atom)
                            + r.numeric.xyzVector_double_t(100, 0, 0))
    r.core.pose.append_pose_to_pose(pose, partner, True)
    pose.pdb_info(r.core.pose.PDBInfo(pose))
    for index in range(1, pose.total_residue() + 1):
        pose.pdb_info().chain(index, "A" if index <= binder_size else "B")

    observed = analyze_pose_interface(pose, ("A",), ("B",))
    for obs in observed.residues:
        residue = pose.residue(obs.pose_index)
        expected_xyz = tuple(tuple(residue.xyz(atom)) for atom in range(1, residue.nheavyatoms() + 1)
                             if not residue.atom_type(atom).is_virtual())
        assert obs.heavy_atom_coordinates == expected_xyz
        assert obs.nearest_other_group_heavy_atom_distance > 5.
    # Chemical expectations are explicit, not recomputed from Hpos_polar or
    # accpt_pos. N-terminal Pro's polar Hs must count for physical N even when
    # its residue type attaches them to the virtual shadow NV.
    expected = {"N": "DONOR", "O": "ACCEPTOR", **expected_roles}
    for index in positions:
        roles = {atom.atom.atom: atom.role for atom in observed.polar_atoms
                 if atom.atom.chain == "A" and atom.atom.position == index}
        assert roles == {name: role for name, role in expected.items() if role is not None}


def test_nterminal_proline_hbond_uses_physical_donor_and_satisfaction():
    pyrosetta = pytest.importorskip("pyrosetta")
    import numpy as np
    from astrolabe.mutators.rfd4_feedback import analyze_pose_interface

    r = pyrosetta.rosetta
    if not r.basic.was_init_called():
        pyrosetta.init("-ignore_zero_occupancy false -mute all -beta")
    pose = pyrosetta.pose_from_sequence("PA")
    partner = pyrosetta.pose_from_sequence("A")
    # Rigid placement from the real-residue CPU probe: carbonyl O accepts
    # Pro 1H at 1.8 A. Internal residue geometry and chemistry are unchanged.
    rotation = np.array([
        [-0.7745318928699444, 0.6100438130812754, -0.16717324262141398],
        [-0.5413338736723102, -0.7760003125281154, -0.32369916924700065],
        [-0.32719716401923715, -0.16021879127469701, 0.9312743713751834],
    ])
    translation = np.array([-0.21610578546746528, 5.613141597034069, 0.798103470639479])
    for atom in range(1, partner.residue(1).natoms() + 1):
        xyz = rotation @ np.array(tuple(partner.residue(1).xyz(atom))) + translation
        partner.set_xyz(r.core.id.AtomID(atom, 1), r.numeric.xyzVector_double_t(*xyz))
    r.core.pose.append_pose_to_pose(pose, partner, True)
    pose.pdb_info(r.core.pose.PDBInfo(pose))
    for index in range(1, pose.total_residue() + 1):
        pose.pdb_info().chain(index, "A" if index <= 2 else "B")

    # HB observations and any-HB satisfaction are independent of this small
    # interface-neighborhood cutoff, which avoids invoking the SC backend.
    observed = analyze_pose_interface(pose, ("A",), ("B",), interface_distance=0.1)
    pro_bonds = [hb for hb in observed.hbonds
                 if hb.donor.chain == "A" and hb.donor.position == 1 and hb.qualifying]
    assert len(pro_bonds) == 1
    hb = pro_bonds[0]
    physical_donor = AtomRef("A", 1, "N")
    assert hb.donor == physical_donor
    assert hb.hydrogen == AtomRef("A", 1, "1H")
    assert hb.acceptor == AtomRef("B", 1, "O")
    polar = next(atom for atom in observed.polar_atoms if atom.atom == physical_donor)
    assert polar.role == "DONOR"
    assert hb.donor_xyz == pytest.approx(tuple(pose.residue(1).xyz("N")))
    assert hb.donor_xyz == pytest.approx(polar.xyz)
    # One bond satisfies this atom despite its two donor hydrogens.
    assert len(pose.residue(1).Hpos_polar()) == 2
    assert polar.satisfied and polar.qualifying_hbond_count == 1
    assert all(atom.atom != AtomRef("A", 1, "NV") for atom in observed.polar_atoms)


@pytest.mark.parametrize("omit_bfactor", [False, True])
def test_mmcif_loader_ignores_empty_nonatom_metadata(tmp_path, omit_bfactor):
    pyrosetta = pytest.importorskip("pyrosetta")
    from Bio.PDB import MMCIFIO, PDBParser
    from astrolabe.mutators.rfd4_feedback import analyze_interface

    r = pyrosetta.rosetta
    if not r.basic.was_init_called():
        pyrosetta.init("-ignore_zero_occupancy false -mute all -beta")
    pose = pyrosetta.pose_from_sequence("AS")
    other = pose.clone()
    for i in range(1, other.total_residue() + 1):
        for j in range(1, other.residue(i).natoms() + 1):
            other.set_xyz(r.core.id.AtomID(j, i), other.residue(i).xyz(j)
                          + r.numeric.xyzVector_double_t(100, 0, 0))
    r.core.pose.append_pose_to_pose(pose, other, True)
    pose.pdb_info(r.core.pose.PDBInfo(pose))
    for i in range(1, 5):
        pose.pdb_info().chain(i, "X" if i <= 2 else "Y")
        pose.pdb_info().number(i, 100 + i)
    pdb = tmp_path / "source.pdb"
    pose.dump_pdb(str(pdb))
    writer = MMCIFIO()
    writer.set_structure(PDBParser(QUIET=True).get_structure("source", str(pdb)))
    cif = tmp_path / "accepted.cif"
    writer.save(str(cif))
    cif.write_text(cif.read_text() + "\n_dummy.title ''\n")
    original_feedback = analyze_interface(cif, ("A",), ("B",))
    if omit_bfactor:
        from Bio.PDB.MMCIF2Dict import MMCIF2Dict
        record = MMCIF2Dict(str(cif))
        del record["_atom_site.B_iso_or_equiv"]
        record.pop("_dummy.title")  # MMCIFIO itself cannot serialize an empty quoted value.
        writer.set_dict(record)
        writer.save(str(cif))
    copied = tmp_path / "copied.cif"
    copied.write_bytes(cif.read_bytes())

    feedback = analyze_interface(cif, ("A",), ("B",))
    assert replace(feedback, provenance=()) == replace(original_feedback, provenance=())
    assert feedback == analyze_interface(copied, ("A",), ("B",))
    assert json.loads(dict(feedback.provenance)["physical_chain_author_ids"]) == {"A": ["X"], "B": ["Y"]}
    assert [residue.author_number for residue in feedback.residues] == [101, 102, 103, 104]
    plan = plan_conditions(
        (ChainSpec("A", "binder", "AS"), ChainSpec("B", "target", "AS")),
        (ResidueRef("A", 1), ResidueRef("A", 2)), 1, feedback=feedback,
    )
    assert len(plan.released) == 1
    assert all(ref.chain == "A" for ref in plan.released)

    # Runtime output roots are not part of the scientific params/cache identity.
    with open(r.basic.database.full_name(
        "chemical/residue_type_sets/fa_standard/residue_types/protonation_states/HIS_P.params"
    ), "rb") as params_file:
        params_bytes = params_file.read()
    relocated = []
    for root in (tmp_path / "run_one", tmp_path / "run_two"):
        root.mkdir()
        params = root / "HIS_P.params"
        params.write_bytes(params_bytes)
        observed = analyze_interface(cif, ("A",), ("B",), extra_residue_params=[params])
        relocated.append(plan_conditions(
            (ChainSpec("A", "binder", "AS"), ChainSpec("B", "target", "AS")),
            (ResidueRef("A", 1), ResidueRef("A", 2)), 1, feedback=observed,
        ))
    assert relocated[0].feedback_json == relocated[1].feedback_json
    params_identity = json.loads(json.loads(relocated[0].feedback_json)["feedback"]["provenance"]["extra_residue_params"])
    assert params_identity[0]["name"] == "HIS_P.params"
    assert "path" not in params_identity[0]
