"""Score-region and backend behavior across real atom arrays and native seams."""
from types import SimpleNamespace

import numpy as np
import pytest

from astrolabe.core import global_config
from astrolabe.losses import independent, interaction
from astrolabe.losses.rfd4_feedback import interface_hbond_count
from astrolabe.losses.selection import (
    atom_residue_table, residue_positions, resolve_chain_groups, select_atoms,
)
from astrolabe.scoring_startup import validate_scoring_settings

struc = pytest.importorskip("biotite.structure")


def atoms(records, bonds=()):
    result = struc.AtomArray(len(records))
    for field, column in (("chain_id", 0), ("res_id", 1), ("ins_code", 2), ("res_name", 3),
                          ("atom_name", 4), ("element", 5), ("coord", 6)):
        setattr(result, field, np.asarray([r[column] for r in records]))
    result.bonds = struc.BondList(len(result), np.asarray(bonds, dtype=int).reshape(-1, 3))
    return result


def scores(structure, oligo="AB"):
    return SimpleNamespace(names=[oligo], complex_poses=SimpleNamespace(poses={oligo: structure}))


def simple_atoms():
    return atoms([
        ("A", 10, "", "ALA", "CA", "C", [0, 0, 0]),
        ("A", 10, "A", "ALA", "CA", "C", [0, 2, 0]),
        ("A", 40, "", "ALA", "CA", "C", [0, 4, 0]),
        ("B", 7, "", "ALA", "CA", "C", [4, 0, 0]),
        ("B", 9, "", "ALA", "CA", "C", [4, 4, 0]),
    ])


def hbond_atoms():
    return atoms([
        ("A", 10, "", "ASN", "ND2", "N", [0, 0, 0]),
        ("A", 10, "", "ASN", "HD21", "H", [1, 0, 0]),
        ("A", 10, "", "ASN", "CG", "C", [0, 1.3, 0]),
        ("B", 7, "", "ASP", "OD1", "O", [2.7, 0, 0]),
        ("B", 7, "", "ASP", "CG", "C", [3.9, 0, 0]),
        ("B", 99, "", "ALA", "CA", "C", [20, 0, 0]),
    ], [(0, 1, 1), (0, 2, 1), (3, 4, 2)])


def test_group_resolution_expands_copies_and_rejects_ambiguous_aliases():
    assert resolve_chain_groups("AAB", "protomer:A", "protomer:B") == (["A", "B"], ["C"])
    assert resolve_chain_groups("AAB", "A", "C") == (["A"], ["C"])
    assert resolve_chain_groups("A" * 28, "chain:AA", "chain:BA") == (["AA"], ["BA"])
    with pytest.raises(ValueError, match="one selector"):
        resolve_chain_groups("ABC", included_protomers="A,B")
    with pytest.raises(ValueError, match="disagree"):
        resolve_chain_groups("ABC", "A", included_protomers="B")
    with pytest.raises(ValueError, match="disjoint"):
        resolve_chain_groups("AAB", "protomer:A", "chain:B")


def test_regions_use_chain_ordinals_preserving_insertion_codes():
    source = simple_atoms()
    region, a, b = select_atoms(scores(source), "AB", source, ["A"], ["B"],
                                residues_a="2", residues_b="2")
    assert np.flatnonzero(a).tolist() == [1]
    assert np.flatnonzero(b).tolist() == [4]
    assert region.evidence()["a"] == {"A": [2]}
    assert len(source) == 5
    with pytest.raises(ValueError, match="outside"):
        select_atoms(scores(source), "AB", source, ["A"], ["B"], residues_a="10")


@pytest.mark.parametrize("value", ["", "0", "3-2", "1,", "1-4", [True], "1.5"])
def test_explicit_invalid_or_empty_ranges_fail(value):
    with pytest.raises(ValueError):
        residue_positions(value, 3)


def test_duplicate_discontiguous_residue_identity_is_rejected():
    source = simple_atoms()
    source.res_id[2] = 10
    with pytest.raises(ValueError, match="Discontiguous"):
        atom_residue_table(source)


def test_existing_hotspot_dsl_and_false_selector_are_used_for_scores():
    source = simple_atoms()
    global_config.reset()
    global_config.init_const("protomers", {
        "A": SimpleNamespace(length=3, select_residues_by_alias=lambda _: np.array([False, True, False])),
        "B": SimpleNamespace(length=2, select_residues_by_alias=lambda _: np.array([True, False])),
    })
    try:
        _, a, b = select_atoms(scores(source), "AB", source, ["A"], ["B"], selector_a="hotspot('chosen')")
        assert np.flatnonzero(a).tolist() == [1]
        assert np.flatnonzero(b).tolist() == [3, 4]
        with pytest.raises(ValueError, match="empty"):
            select_atoms(scores(source), "AB", source, ["A"], ["B"], selector_a="False")
        with pytest.raises(ValueError, match="only one"):
            select_atoms(scores(source), "AB", source, ["A"], ["B"], residues_a="1", selector_a="True")
    finally:
        global_config.reset()


def test_real_hbond_entrypoint_respects_regions_and_records_method(monkeypatch):
    source = hbond_atoms()
    monkeypatch.setattr(interaction, "_require_pyrosetta", lambda: pytest.fail("independent path imported PyRosetta"))
    result = scores(source)
    with pytest.warns(UserWarning, match="not numerically equivalent"):
        assert interface_hbond_count(result, protomer_a="A", protomer_b="B", backend="independent", hydrogen_policy="require") == {"AB": 1}
    record = result.loss_backend_records["AB"][0]
    assert record["backend"] == "independent"
    assert record["selection"]["a"] == {"A": [1]}
    assert record["version"] != "unknown"
    with pytest.warns(UserWarning):
        assert interface_hbond_count(scores(source), "A", "B", backend="independent",
                                     hydrogen_policy="require", protomer_b_residues="2") == {"AB": 0}


def test_null_native_feedback_options_are_unset_for_independent_backend():
    options = dict(hbond_energy_cutoff=None, interface_distance=None,
                   buried_sasa_cutoff=None, sasa_probe_radius=None)
    assert not validate_scoring_settings("loss", "interface_hbond_count",
                                         {"backend": "independent", **options}, path="loss.hb", check_paths=False)
    with pytest.warns(UserWarning, match="not numerically equivalent"):
        result = interface_hbond_count(scores(hbond_atoms()), backend="independent", hydrogen_policy="require",
                                       **options)
    assert result == {"AB": 1}


def test_auto_fallback_only_catches_backend_unavailable(monkeypatch):
    result = scores(simple_atoms())
    monkeypatch.setattr(independent, "shape_complementarity", lambda *a, **kw: (_ for _ in ()).throw(independent.BackendUnavailable("missing test dependency")))
    monkeypatch.setattr(interaction, "_native_shape_score", lambda *a, **kw: 0.61)
    with pytest.warns(UserWarning, match="using Rosetta"):
        assert interaction.protomer_shape_complementarity(result) == {"AB": 0.61}
    assert result.loss_backend_records["AB"][0]["backend"] == "rosetta"
    monkeypatch.setattr(independent, "shape_complementarity", lambda *a, **kw: (_ for _ in ()).throw(ValueError("invalid geometry")))
    with pytest.raises(ValueError, match="invalid geometry"):
        interaction.protomer_shape_complementarity(scores(simple_atoms()))


@pytest.mark.parametrize("surface_side", ["a", "b"])
def test_native_region_keeps_complete_pose_and_maps_one_based_indices(monkeypatch, surface_side):
    chains = ["A", "A", "A", "B", "B"]
    pose = SimpleNamespace(total_residue=lambda: 5,
                           pdb_info=lambda: SimpleNamespace(chain=lambda i: chains[i - 1]))
    captured = {}
    def calculate(actual_pose, a, b, **options):
        assert actual_pose is pose
        captured["chains"] = (a, b)
        captured.update(options)
        return 40.0
    monkeypatch.setattr(interaction, "_compute_contact_molecular_surface_between_chains", calculate)
    assert interaction.contact_molecular_surface(scores(pose), backend="rosetta",
                                                 protomer_a_residues="1,3", protomer_b_residues="2",
                                                 surface_side=surface_side) == {"AB": 40.0}
    expected_a, expected_b = ([1, 3], [5]) if surface_side == "a" else ([5], [1, 3])
    assert captured["selected_residues_a"] == expected_a
    assert captured["selected_residues_b"] == expected_b
    assert captured["chains"] == ((["A"], ["B"]) if surface_side == "a" else (["B"], ["A"]))


@pytest.mark.parametrize("restricted", [False, True])
def test_native_feedback_sc_records_the_radius_policy_of_the_executed_path(monkeypatch, restricted):
    from astrolabe.losses.rfd4_feedback import interface_shape_complementarity
    from astrolabe.mutators import rfd4_feedback

    chains = ["A", "A", "A", "B", "B"]
    pose = SimpleNamespace(total_residue=lambda: len(chains),
                           pdb_info=lambda: SimpleNamespace(chain=lambda i: chains[i - 1]))
    calls = []
    def analyze(actual_pose, a, b, **options):
        assert actual_pose is pose
        calls.append("default_calculator")
        return SimpleNamespace(shape_complementarity=0.6)
    def selected_shape(actual_pose, a, b, selection, *, strict):
        assert actual_pose is pose and strict
        assert selection.residues_a == (0, 2)
        calls.append("selected_lj_filter")
        return 0.7
    monkeypatch.setattr(rfd4_feedback, "analyze_pose_interface", analyze)
    monkeypatch.setattr(interaction, "_native_shape_score", selected_shape)
    result = scores(pose)
    options = {"protomer_a_residues": "1,3"} if restricted else {}
    measured = interface_shape_complementarity(result, backend="rosetta", **options)
    assert measured == {"AB": 0.7 if restricted else 0.6}
    assert calls == (["default_calculator", "selected_lj_filter"] if restricted else ["default_calculator"])
    expected = "Rosetta LJ" if restricted else "Rosetta ShapeComplementarityCalculator default (not overridden)"
    assert result.loss_backend_records["AB"][0]["radii"] == expected


def test_real_cms_entrypoint_reports_requested_side_and_norm_uses_it():
    source = simple_atoms()
    a, b = source.chain_id == "A", source.chain_id == "B"
    expected = {side: independent.contact_molecular_surface(source, a, b, surface_side=side).value
                for side in ("a", "b")}
    assert expected["a"] != pytest.approx(expected["b"])
    for side in ("a", "b"):
        result = scores(source)
        measured = interaction.contact_molecular_surface(result, backend="independent", surface_side=side)
        assert measured["AB"] == pytest.approx(expected[side])
        assert result.loss_backend_records["AB"][0]["options"]["surface_side"] == side
        normalized = interaction.contact_molecular_surface_norm(scores(source), backend="independent", surface_side=side)
        assert normalized["AB"] == pytest.approx(interaction.normalize_contact_molecular_surface(expected[side]))


def test_holes_does_not_invent_an_independent_proxy():
    from astrolabe.losses.interface_holes import interface_holes
    with pytest.raises(independent.BackendUnavailable, match="no validated independent"):
        interface_holes(scores(simple_atoms()), "A", "B", backend="independent")


def test_ligand_hbond_rejects_geometric_options_on_rosetta(monkeypatch):
    from astrolabe.core.protomer_config_parser import ProtomerTypes

    global_config.reset()
    global_config.init_const("protomers", {
        "A": SimpleNamespace(type=ProtomerTypes.CCD),
        "B": SimpleNamespace(type=ProtomerTypes.PROTEIN),
    })
    monkeypatch.setattr(interaction, "_native_ligand_hbond_score", lambda *args, **options: 4.0)
    try:
        with pytest.raises(ValueError, match="does not implement independent geometry options: cutoff_distance"):
            interaction.ligand_protein_hbond(scores(simple_atoms()), backend="rosetta", cutoff_distance=3.2)
    finally:
        global_config.reset()


@pytest.mark.parametrize("options,key", [
    ({"backend": "mystery"}, "backend"),
    ({"backend": "independent", "use_rosetta_radii": True}, "use_rosetta_radii"),
    ({"protomer_a_residues": "1", "position_selector_a": "True"}, "position_selector_a"),
    ({"protomer_b_residues": "0"}, "protomer_b_residues"),
    ({"surface_side": "both"}, "surface_side"),
])
def test_preflight_rejects_ambiguous_or_backend_incompatible_settings(options, key):
    errors = validate_scoring_settings("loss", "contact_molecular_surface", options,
                                       path="loss.loss_terms.score", check_paths=False)
    assert any(error.startswith(f"loss.loss_terms.score.{key}:") for error in errors)


def test_removed_shape_parameters_fail_before_runtime():
    for name in ("target_sc", "saturation_constant", "return_raw"):
        assert interaction.loss_registry.validate_params("protomer_shape_complementarity", [name])


def test_hotspot_normalization_argument_reaches_calculation(monkeypatch):
    monkeypatch.setattr(interaction, "_require_pyrosetta", lambda: None)
    monkeypatch.setattr(interaction, "compute_hotspot_delta_sasa", lambda result, expression, normalized_by: {"AB": 80 / normalized_by})
    assert interaction.hotspot_dsasa(object(), "B_1:1", normalize_factor=40) == {"AB": 2.0}
