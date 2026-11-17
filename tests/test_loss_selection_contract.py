from types import SimpleNamespace
import numpy as np
import pytest
from omegaconf import OmegaConf

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import Protomer, SegmentManager
from astrolabe.evaluators.utils import StandardScores
from astrolabe.loss_config import loss_backend, normalize_loss_options, validate_loss_selector
from astrolabe.losses.base import LossEngine, loss_registry
from astrolabe.losses.selection import (
    loss_atom_mask, pose_residue_indices, resolve_loss_chain_groups,
    selected_residue_ordinals,
)
from astrolabe.selectors.registry import SelectorRegistry
from astrolabe.selectors.utils import construct_position_selector
from astrolabe.startup_dependencies import dalphaball_required_by
from astrolabe.startup_validation import StartupValidationError, validate_startup_config

@pytest.fixture(autouse=True)
def reset_runtime():
    global_config.reset()
    yield
    global_config.reset()

@pytest.fixture
def protomers():
    for name, value in (
        ("segments", {}), ("named_files", {}), ("predefined_statement", {}),
        ("exclude_AA", []), ("segment_manager", SegmentManager()),
    ):
        global_config.init_const(name, value)
    values = {"A": Protomer("A", "AC@partA", "protein"),
              "B": Protomer("B", "DE@partB", "protein")}
    global_config.init_const("protomers", values)
    return values

def atoms():
    return SimpleNamespace(
        chain_id=np.repeat(list("ABC"), 4),
        res_id=np.tile([10, 10, 10, 10], 3),
        ins_code=np.tile(["", "", "A", "A"], 3),
    )

def setup_engine(options):
    engine = LossEngine({"loss_terms": {"metric": options}})
    engine.setup("metric", None, None)
    return engine

def config(options):
    return OmegaConf.create({
        "seed": 17, "steps": 2,
        "evaluator": {"evaluator_terms": {"first": {"_type": "sequence"}}},
        "mutator": {"mutator_terms": {"m": {"_type": "mcmc"}}},
        "stages": {"default": {"evaluator": "first", "mutators": ["m"],
                   "weighted_score_expression": "metric", "mutation_num": 1}},
        "loss": {"loss_terms": {"metric": options}},
    })

def test_repeated_protomer_masks_preserve_insertion_codes(protomers):
    result = StandardScores(names=["ABA"], plddt=None, complex_paths=None)
    group_a, group_b = resolve_loss_chain_groups("ABA", "protomer:A", "protomer:B")
    assert group_a == ["A", "C"] and group_b == ["B"]
    mask = loss_atom_mask(result, "ABA", atoms(), group_a, selector="hotspot('partA:2')")
    np.testing.assert_array_equal(mask, [False, False, True, True, False, False, False, False,
                                       False, False, True, True])
    assert result.complex_poses is None

def test_chain_selector_chooses_only_one_repeated_copy(protomers):
    result = StandardScores(names=["ABA"], plddt=None, complex_paths=None)
    a, b = resolve_loss_chain_groups("ABA", "chain:A", "chain:B")
    mask = loss_atom_mask(result, "ABA", atoms(), a, selector="hotspot('partA')")
    assert np.flatnonzero(mask).tolist() == [0, 1, 2, 3]
    assert b == ["B"]

@pytest.mark.parametrize("expression", ["False", "hotspot('partB')"])
def test_empty_selection_never_expands_to_whole_group(protomers, expression):
    mask = loss_atom_mask(StandardScores(names=["ABA"], plddt=None, complex_paths=None), "ABA", atoms(), ["A"], selector=expression)
    assert not np.any(mask)

def test_ranges_intersect_position_selector(protomers):
    selected = selected_residue_ordinals(StandardScores(names=["ABA"], plddt=None, complex_paths=None), "ABA", ["A", "C"],
        chain_lengths={"A": 2, "B": 2, "C": 2}, selector="hotspot('partA:2')", residues="1")
    assert selected == {"A": frozenset(), "C": frozenset()}

@pytest.mark.parametrize("expression", ["", "hotspoot('partA')", "[]", "42", "hotspot.__call__('partA')"])
def test_invalid_selector_is_rejected_before_evaluation(expression):
    with pytest.raises(ValueError):
        validate_loss_selector(expression)

@pytest.mark.parametrize("residues", ["", "0", "3", "2-1", "1,"])
def test_invalid_range_cannot_select_all(residues):
    with pytest.raises(ValueError):
        loss_atom_mask(StandardScores(names=["ABA"], plddt=None, complex_paths=None), "ABA", atoms(), ["A"], residues=residues)

def test_selector_length_must_match_current_structure(protomers):
    shortened = SimpleNamespace(chain_id=np.array(["A"]), res_id=np.array([10]),
                                ins_code=np.array([""]))
    with pytest.raises(ValueError, match="scored structure has 1 residues"):
        loss_atom_mask(StandardScores(names=["A"], plddt=None, complex_paths=None), "A", shortened, ["A"], selector="hotspot('partA')")

def test_missing_masks_are_empty_and_broadcast_masks_are_rejected(monkeypatch):
    protomers = {"A": SimpleNamespace(length=2), "B": SimpleNamespace(length=2)}
    sites = {"A": np.array([True, False])}
    monkeypatch.setattr(SelectorRegistry, "construct_environment",
                        lambda *args, **kwargs: {"hotspot": lambda *_: SimpleNamespace(sites=sites)})
    result = construct_position_selector("hotspot('unused')", SimpleNamespace(), protomers, strict=True)
    np.testing.assert_array_equal(result["B"], [False, False])
    sites["A"] = np.array([[True, False]])
    with pytest.raises(ValueError, match="shape"):
        construct_position_selector("hotspot('unused')", SimpleNamespace(), protomers, strict=True)

def test_pose_indices_follow_order_not_pdb_residue_numbers():
    chains = ["A", "A", "B", "B", "C", "C"]
    pose = SimpleNamespace(total_residue=lambda: len(chains),
        pdb_info=lambda: SimpleNamespace(chain=lambda index: chains[index - 1]))
    assert pose_residue_indices(pose, {"A": frozenset({2}), "C": frozenset({1})}) == [2, 5]

@pytest.mark.parametrize("options, fragment", [
    ({"_type": "protomer_shape_complementarity", "return_raw": True}, "return_raw"),
    ({"_type": "contact_molecular_surface", "distance_weight": float("nan")}, "distance_weight"),
    ({"_type": "contact_molecular_surface", "_for": "AB", "oligomer_specific": ["AC"]}, "conflicting"),
    ({"_type": "contact_molecular_surface", "protomer_a": "A", "included_protomers": "B"}, "conflicts"),
    ({"_type": "contact_molecular_surface", "protomer_a_residues": ""}, "nonempty"),
])
def test_runtime_and_preflight_reject_same_malformed_options(options, fragment):
    with pytest.raises(ValueError, match=fragment):
        setup_engine(options)
    with pytest.raises(StartupValidationError, match=fragment):
        validate_startup_config(config(options), check_dependencies=False)

def test_scope_normalization_preserves_equivalent_aliases():
    assert normalize_loss_options({"_for": " AB, AC ", "oligomer_specific": ["AC", "AB"]}) == {
        "oligomer_specific": ["AB", "AC"]}
    assert normalize_loss_options({"_for": None}) == {}

def test_backend_defaults_and_rosetta_dependencies_are_explicit():
    assert loss_backend("contact_molecular_surface") == "py_contact_ms"
    assert loss_backend("interface_buried_unsatisfied_hbonds") == "biotite"
    assert loss_backend("interface_holes") == "rosetta"
    for kind in ("interface_buried_unsatisfied_hbonds", "ligand_buried_unsat_hbonds"):
        assert not dalphaball_required_by({"loss": {"loss_terms": {"x": {"_type": kind}}}})
        assert dalphaball_required_by({"loss": {"loss_terms": {"x": {"_type": kind, "backend": "rosetta"}}}})
        assert not dalphaball_required_by({"stages": {"default": {"weighted_score_expression": kind}}})
    assert dalphaball_required_by({"stages": {"default": {"weighted_score_expression": "interface_holes"}}})
    with pytest.raises(ValueError):
        loss_backend("interface_holes", {"backend": "biotite"})

def test_loss_engine_resolves_current_masks_without_reusing_stale_score(protomers, monkeypatch):
    monkeypatch.setattr(loss_registry, "_loss_functions", dict(loss_registry._loss_functions))
    monkeypatch.setattr(loss_registry, "_metadata", dict(loss_registry._metadata))
    state = np.array([True, False])
    monkeypatch.setattr(protomers["A"], "select_residues_by_alias", lambda *_: state)

    @loss_registry.register(name="__selected_atom_count__", optional_params=["selector_a"])
    def selected_count(eval_result, selector_a=None):
        return {"ABA": float(np.sum(loss_atom_mask(eval_result, "ABA", atoms(), ["A"], selector=selector_a)))}

    engine = setup_engine({"_type": "__selected_atom_count__", "selector_a": "hotspot('partA')"})
    result = StandardScores(names=["ABA"], plddt=None, complex_paths=None)
    assert engine.eval_score(result, "metric") == 2.0
    state[:] = True
    assert engine.eval_score(result, "metric") == 4.0

@pytest.mark.parametrize("options, key", [
    ({"hydrogens": ["add"]}, "hydrogens"),
    ({"hydrogen_ph": 15}, "hydrogen_ph"),
    ({"hbond_distance": float("inf")}, "hbond_distance"),
    ({"hbond_angle": 180}, "hbond_angle"),
    ({"sasa_points": 19}, "sasa_points"),
    ({"sasa_points": True}, "sasa_points"),
    ({"sasa_probe_radius": 0}, "sasa_probe_radius"),
    ({"ignore_surface_res": True}, "ignore_surface_res"),
    ({"separation_distance": 10}, "separation_distance"),
])
def test_independent_controls_fail_at_shared_boundary(options, key):
    from astrolabe.scoring_startup import validate_scoring_settings
    errors = validate_scoring_settings("loss", "interface_buried_unsatisfied_hbonds",
        options, path="loss.metric", check_paths=False)
    assert any(key in error for error in errors)

def test_independent_control_valid_values_and_native_overrides():
    from astrolabe.scoring_startup import validate_scoring_settings
    validate = lambda kind, options: validate_scoring_settings(
        "loss", kind, options, path="loss.metric", check_paths=False)
    assert not validate("interface_buried_unsatisfied_hbonds", {
        "hydrogens": "existing", "hydrogen_ph": 0, "hbond_distance": 2.5,
        "hbond_angle": 120, "sasa_points": 1000, "sasa_probe_radius": 1.4,
        "selector_a": "hotspot('partA:2')", "selector_b": "True",
    })
    assert not validate("contact_molecular_surface", {"use_rosetta_radii": None})
    assert validate("contact_molecular_surface", {"use_rosetta_radii": True})
    assert not validate("contact_molecular_surface", {"backend": "rosetta", "use_rosetta_radii": True})
    assert validate("interface_hbond_count", {"hbond_energy_cutoff": -0.5})
    assert not validate("interface_hbond_count", {"backend": "rosetta", "hbond_energy_cutoff": -0.5})

def test_common_chain_parameters_normalize_for_legacy_feedback_wrappers():
    options = normalize_loss_options({"_type": "interface_hbond_count",
                                      "protomer_a": "A", "protomer_b": ["B", "C"]})
    assert options == {"_type": "interface_hbond_count", "binder_chains": "A",
                       "target_chains": ["B", "C"]}
    with pytest.raises(ValueError, match="conflicts"):
        normalize_loss_options({"_type": "interface_shape_complementarity",
                                "protomer_a": "A", "binder_chains": "B"})

def test_registered_selector_alias_and_boolean_composition_remain_supported(protomers, monkeypatch):
    monkeypatch.setattr(SelectorRegistry, "_registrations", dict(SelectorRegistry._registrations))
    SelectorRegistry.register("selected_patch", module="astrolabe.selectors.hotspot",
        class_name="HotspotAliasPositionSelector", context="complex_poses",
        pose_required=False)
    expression = "selected_patch('partA') and not hotspot('partA:1')"
    validate_loss_selector(expression)
    mask = loss_atom_mask(StandardScores(names=["ABA"], plddt=None, complex_paths=None),
                          "ABA", atoms(), ["A"], selector=expression)
    assert np.flatnonzero(mask).tolist() == [2, 3]

def test_scoped_loss_aliases_share_content_bound_structure_preparation(monkeypatch):
    pytest.importorskip("biotite")
    from pathlib import Path
    from astrolabe.losses.independent import structure_context
    monkeypatch.setattr(loss_registry, "_loss_functions", dict(loss_registry._loss_functions))
    monkeypatch.setattr(loss_registry, "_metadata", dict(loss_registry._metadata))
    preparations = []

    @loss_registry.register(name="__scoped_preparation__", optional_params=[])
    def atom_count(eval_result):
        context = structure_context(eval_result, "ABC")
        preparations.append(context)
        return {"ABC": float(len(context["atoms"]))}

    spec = {"_type": "__scoped_preparation__", "_for": "ABC"}
    engine = LossEngine({"loss_terms": {"first": spec, "second": spec}})
    engine.setup("first + second", None, None)
    path = Path(__file__).parents[1] / "examples/paper/inputs/assemblies/01_C3_seed2503_step024_reference.cif"
    scores = StandardScores(names=["ABC"], plddt=None, complex_paths={"ABC": path})
    value = engine.eval_score(scores, "first + second")
    assert value == 2 * len(preparations[0]["atoms"])
    assert len(preparations) == 2
    assert preparations[0] is preparations[1]


_ROSETTA_POLAR_KINDS = (
    "interface_hbond_count", "ligand_protein_hbond",
    "interface_buried_unsatisfied_hbonds", "ligand_buried_unsat_hbonds",
)

@pytest.mark.parametrize("kind", _ROSETTA_POLAR_KINDS)
@pytest.mark.parametrize("key, value", [
    ("hydrogens", "existing"), ("hydrogen_ph", 8),
    ("hbond_distance", 3.0), ("hbond_angle", 130),
])
def test_native_polar_geometry_requires_independent_backend_at_preflight(kind, key, value):
    from astrolabe.scoring_startup import validate_scoring_settings
    errors = validate_scoring_settings("loss", kind,
        {"backend": "rosetta", key: value}, path="loss.metric", check_paths=False)
    assert errors == [f"loss.metric.{key}: requires backend=biotite"]

@pytest.mark.parametrize("kind", _ROSETTA_POLAR_KINDS[2:])
@pytest.mark.parametrize("key, value", [
    ("sasa_probe_radius", 1.5), ("buried_sasa_cutoff", 2),
    ("sasa_points", 2000),
])
def test_native_burial_geometry_requires_independent_backend_at_preflight(kind, key, value):
    from astrolabe.scoring_startup import validate_scoring_settings
    errors = validate_scoring_settings("loss", kind,
        {"backend": "rosetta", key: value}, path="loss.metric", check_paths=False)
    assert errors == [f"loss.metric.{key}: requires backend=biotite"]

def test_native_polar_defaults_and_feedback_sasa_stay_valid():
    from astrolabe.scoring_startup import validate_scoring_settings
    for kind in _ROSETTA_POLAR_KINDS:
        options = {"backend": "rosetta", "hydrogens": "add", "hydrogen_ph": 7,
                   "hbond_distance": 2.5, "hbond_angle": 120}
        if kind in _ROSETTA_POLAR_KINDS[2:]:
            options.update(sasa_probe_radius=1.4, buried_sasa_cutoff=1, sasa_points=1000)
        assert not validate_scoring_settings(
            "loss", kind, options, path="loss.metric", check_paths=False)
    # These are native RFD4 feedback options, unlike the independent BUNS geometry.
    assert not validate_scoring_settings(
        "loss", "interface_hbond_count",
        {"backend": "rosetta", "sasa_probe_radius": 1.8, "buried_sasa_cutoff": 2},
        path="loss.metric", check_paths=False)
