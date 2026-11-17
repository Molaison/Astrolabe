from astrolabe.runtime_requirements import task_requirements


def requirements(kind, **options):
    return set(task_requirements({
        "enable_pyrosetta": False,
        "loss": {"loss_terms": {"score": {"_type": kind, **options}}},
    })["required_imports"])


def test_independent_surface_default_does_not_require_pyrosetta():
    required = requirements("contact_molecular_surface")
    assert "pyrosetta" not in required
    assert {"py_contact_ms", "biotite"} <= required


def test_explicit_rosetta_surface_requires_only_native_backend():
    required = requirements("contact_molecular_surface", backend="rosetta")
    assert "pyrosetta" in required
    assert "py_contact_ms" not in required


def test_loss_selector_dependencies_are_not_hidden_by_independent_scoring():
    required = requirements("contact_molecular_surface", selector_a="surface()")
    assert {"pyrosetta", "py_contact_ms"} <= required


def test_implicit_physical_loss_in_nested_expression_gets_backend_dependencies():
    required = set(task_requirements({
        "loss": {"expression_terms": {"surface_score": "contact_molecular_surface"}},
        "stages": {"default": {"weighted_score_expression": "surface_score"}},
    })["required_imports"])
    assert {"py_contact_ms", "biotite"} <= required
    assert "pyrosetta" not in required


def test_inactive_native_alias_does_not_force_native_dependency():
    required = set(task_requirements({
        "loss": {"loss_terms": {
            "unused_native": {"_type": "contact_molecular_surface", "backend": "rosetta"},
            "used_independent": {"_type": "protomer_shape_complementarity"},
        }},
        "stages": {"default": {
            "active_loss_terms": ["used_independent"],
            "weighted_score_expression": "used_independent",
        }},
    })["required_imports"])
    assert "protein_interface" in required
    assert "pyrosetta" not in required


def test_active_builtin_loss_is_prepared_without_expression_reference():
    from astrolabe.losses.base import LossEngine
    engine = LossEngine({})
    engine.setup("0", None, None, active_loss_terms=["contact_molecular_surface"])
    assert "contact_molecular_surface" in engine.loss_specs
    required = set(task_requirements({
        "loss": {},
        "stages": {"default": {
            "weighted_score_expression": "0",
            "active_loss_terms": ["contact_molecular_surface"],
        }},
    })["required_imports"])
    assert "py_contact_ms" in required


def test_entrance_uses_source_stage_not_destination_active_loss_whitelist():
    from astrolabe.losses.base import LossEngine
    engine = LossEngine({})
    engine.setup("0", None, None)
    expression = "contact_molecular_surface > 0"
    assert "contact_molecular_surface" in engine._build_loss_specs_for_expressions([expression])
    required = set(task_requirements({
        "loss": {},
        "stages": {
            "default": {"weighted_score_expression": "0"},
            "later": {"weighted_score_expression": "0", "active_loss_terms": [],
                      "_entrance": expression},
        },
    })["required_imports"])
    assert "py_contact_ms" in required
