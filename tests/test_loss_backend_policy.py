from types import SimpleNamespace

import pytest

from astrolabe import loss_backend_policy as policy
from astrolabe.runtime_requirements import task_requirements
from astrolabe.startup_dependencies import dalphaball_required_by


@pytest.fixture(autouse=True)
def available_packages(monkeypatch):
    monkeypatch.setattr(policy.importlib.util, "find_spec", lambda name: object())


@pytest.mark.parametrize("backend, expected", [("auto", False), ("independent", False), ("rosetta", True)])
def test_selected_backend_controls_pipeline_pose_acquisition(backend, expected):
    import run
    function = lambda result: {}
    engine = SimpleNamespace(loss_specs={"sc": (
        "protomer_shape_complementarity", function, {"backend": backend},
    )})
    assert run._loss_computer_needs_complex_poses(engine) is expected


def test_auto_missing_package_and_explicit_rosetta_settings_require_native(monkeypatch):
    monkeypatch.setattr(policy.importlib.util, "find_spec", lambda name: None)
    assert policy.loss_requires_rosetta("protomer_shape_complementarity") is True
    assert policy.loss_requires_rosetta("protomer_shape_complementarity", {"backend": "independent"}) is False
    monkeypatch.setattr(policy.importlib.util, "find_spec", lambda name: object())
    for name, option in (
        ("contact_molecular_surface", {"use_rosetta_radii": True}),
        ("interface_hbond_count", {"hbond_energy_cutoff": -0.25}),
        ("interface_buried_unsatisfied_hbonds", {"ignore_surface_res": True}),
    ):
        assert policy.loss_requires_rosetta(name, option) is True


@pytest.mark.parametrize("backend", ["independent", "auto"])
def test_geometric_buns_needs_neither_pyrosetta_nor_dalphaball(backend):
    cfg = {"loss": {"loss_terms": {"buns": {
        "_type": "interface_buried_unsatisfied_hbonds", "backend": backend,
    }}}}
    requirements = task_requirements(cfg)
    assert "pyrosetta" not in requirements["required_imports"]
    assert {"biotite", "hydride"} <= set(requirements["required_imports"])
    assert dalphaball_required_by(cfg) == ()
    cfg["loss"]["loss_terms"]["buns"]["backend"] = "rosetta"
    assert "pyrosetta" in task_requirements(cfg)["required_imports"]
    assert dalphaball_required_by(cfg)


def test_native_positional_selector_still_declares_its_own_dependency():
    cfg = {"loss": {"loss_terms": {"sc": {
        "_type": "protomer_shape_complementarity", "backend": "independent",
        "position_selector_a": "interface",
    }}}}
    assert "pyrosetta" in task_requirements(cfg)["required_imports"]


@pytest.mark.parametrize("expression", ["surface()", "buried_unsat()", "ligand_contact()"])
def test_native_loss_selector_controls_both_startup_and_pose_acquisition(expression):
    import run
    params = {"backend": "independent", "position_selector_a": expression}
    name = "protomer_shape_complementarity"
    assert run._loss_spec_needs_complex_poses(name, None, params)
    assert "pyrosetta" in task_requirements({"loss": {"loss_terms": {
        "sc": {"_type": name, **params},
    }}})["required_imports"]


@pytest.mark.parametrize("name, imports", [
    ("interface_buried_unsatisfied_hbonds", {"biotite", "hydride"}),
    ("contact_molecular_surface", {"biotite", "py_contact_ms", "scipy"}),
])
def test_implicit_loss_expression_declares_the_default_backend_dependencies(name, imports):
    cfg = {"loss": {"expression_terms": {"quality": f"-AB.{name}"}},
           "stages": {"default": {"weighted_score_expression": "quality"}}}
    assert imports <= set(task_requirements(cfg)["required_imports"])
    assert "pyrosetta" not in task_requirements(cfg)["required_imports"]


def test_explicit_hydrogens_do_not_require_hydride_or_force_rosetta(monkeypatch):
    monkeypatch.setattr(policy.importlib.util, "find_spec",
                        lambda name: None if name == "hydride" else object())
    params = {"hydrogen_policy": "require"}
    name = "interface_hbond_count"
    assert policy.loss_requires_rosetta(name, params) is False
    required = task_requirements({"loss": {"loss_terms": {
        "hb": {"_type": name, **params},
    }}})["required_imports"]
    assert "biotite" in required
    assert "hydride" not in required
    assert "pyrosetta" not in required


def test_true_holes_never_inherits_geometric_buns_exemption():
    cfg = {"loss": {"loss_terms": {"holes": {"_type": "interface_holes"}}}}
    assert dalphaball_required_by(cfg)
