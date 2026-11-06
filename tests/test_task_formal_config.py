"""Explicit malformed-YAML corpus for the common pre-runtime task contract."""
from __future__ import annotations

import importlib.abc
import json
from pathlib import Path
import sys

from omegaconf import OmegaConf
import pytest

from astrolabe.startup_validation import StartupValidationError, validate_startup_config


FIXTURES = Path(__file__).parent / "fixtures" / "formal_config" / "task"
CASES = json.loads((FIXTURES / "manifest.json").read_text())["cases"]
VALID_CASES = [case for case in CASES if case["expected_error"] is None]


@pytest.fixture(autouse=True)
def forbid_scientific_imports():
    class Boundary(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in {"torch", "jax", "pyrosetta", "tensorflow", "alphafold", "alphafold3", "run"}:
                pytest.fail(f"Scientific runtime imported during task preflight: {fullname}")
            return None

    boundary = Boundary()
    sys.meta_path.insert(0, boundary)
    try:
        yield
    finally:
        sys.meta_path.remove(boundary)


def minimal():
    return OmegaConf.load(FIXTURES / "valid_minimal.yaml")


def test_manifest_is_explicit_complete_and_has_valid_controls():
    assert len({case["id"] for case in CASES}) == len(CASES)
    by_file = {case["file"]: case for case in CASES}
    assert len(by_file) == len(CASES)
    assert set(by_file) == {path.name for path in FIXTURES.glob("*.yaml")}
    for case in CASES:
        required = {"id", "file", "expected_error", "parameters", "valid_control", "source"}
        assert required <= set(case) <= required | {"native_only", "dry_run"}
        assert all(type(case[key]) is bool for key in ("native_only", "dry_run") if key in case)
        assert not Path(case["file"]).is_absolute()
        assert case["expected_error"] is None or isinstance(case["expected_error"], str) and case["expected_error"]
        assert by_file[case["valid_control"]]["expected_error"] is None
        assert case["parameters"] and all(isinstance(field, str) and field for field in case["parameters"])
        assert case["source"] and all(isinstance(source, str) and source for source in case["source"])


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_explicit_common_task_corpus(case):
    cfg = OmegaConf.load(FIXTURES / case["file"])
    before = OmegaConf.to_container(cfg, resolve=False)
    if case["expected_error"] is None:
        assert validate_startup_config(cfg, check_dependencies=False,
                                       resolve_environment=not case.get("dry_run", False)) == {}
    else:
        if case.get("native_only", False):
            assert validate_startup_config(cfg, check_dependencies=False) == {}
        with pytest.raises(StartupValidationError) as caught:
            validate_startup_config(cfg, check_dependencies=False,
                                    require_inputs=True)
        assert case["expected_error"] in str(caught.value)
        if case["file"] != "invalid_execution_unbound_actual_run.yaml":
            assert "search.execution.socket_path: required non-empty" not in str(caught.value)
    assert OmegaConf.to_container(cfg, resolve=False) == before


def test_only_designated_execution_cases_have_an_unbound_socket():
    for case in CASES:
        cfg = OmegaConf.to_container(OmegaConf.load(FIXTURES / case["file"]), resolve=False)
        search = cfg.get("search")
        if not isinstance(search, dict):
            continue
        execution = search.get("execution")
        if isinstance(execution, dict) and "socket_path" in execution and execution["socket_path"] is None:
            assert case["file"] in {"valid_execution_unbound_composition.yaml", "invalid_execution_unbound_actual_run.yaml"}
            assert case.get("dry_run", False) or case.get("native_only", False)


@pytest.mark.parametrize("case", VALID_CASES, ids=lambda case: case["id"])
def test_valid_corpus_supports_unresolved_environment_dry_run(case):
    cfg = OmegaConf.load(FIXTURES / case["file"])
    cfg.evaluator.alphafold3 = {"model_dir": "${unselected.weights}"}
    cfg.initializer.rfd3 = {"repo_path": "${unselected.repository}"}
    validate_startup_config(cfg, check_dependencies=False, resolve_environment=False)


@pytest.mark.parametrize("field", ["seqs.A", "types.A", "homologies.A", "named_files.input", "enable_pyrosetta", "initial_temperature"])
def test_dry_run_defers_interpolated_common_leaves(field):
    cfg = minimal()
    OmegaConf.update(cfg, field, "${not_configured.value}", force_add=True)
    validate_startup_config(cfg, check_dependencies=False, resolve_environment=False)
    # The inactive temperature is intentionally not evaluated at runtime either.
    if field != "initial_temperature":
        with pytest.raises(StartupValidationError, match=field):
            validate_startup_config(cfg, check_dependencies=False)


def test_dry_run_still_rejects_literal_sibling_of_deferred_leaf():
    cfg = minimal()
    cfg.seqs.A = "${not_configured.sequence}"
    cfg.debug = "false"
    with pytest.raises(StartupValidationError, match="debug"):
        validate_startup_config(cfg, check_dependencies=False, resolve_environment=False)


@pytest.mark.parametrize("field", ["seqs.A", "types.A", "homologies.A"])
@pytest.mark.parametrize("resolve_environment", [True, False])
def test_mandatory_slots_are_not_optional(field, resolve_environment):
    cfg = minimal()
    OmegaConf.update(cfg, field, "???", force_add=True)
    with pytest.raises(StartupValidationError, match=field):
        validate_startup_config(cfg, check_dependencies=False, resolve_environment=resolve_environment)


@pytest.mark.parametrize("resolve_environment", [True, False])
def test_partial_api_config_need_not_supply_sequence_inputs(resolve_environment):
    cfg = minimal()
    del cfg.seqs
    del cfg.types
    validate_startup_config(cfg, check_dependencies=False, resolve_environment=resolve_environment)
    validate_startup_config(OmegaConf.create({}), check_dependencies=False, resolve_environment=resolve_environment)


@pytest.mark.parametrize("readonly,struct", [(True, False), (False, True), (True, True)])
def test_validation_does_not_mutate_flagged_input(readonly, struct):
    cfg = minimal()
    cfg.sequence = "ACDEFGHIKLMNPQRSTVWY"
    cfg.seqs.A = "${sequence}"
    OmegaConf.set_readonly(cfg, readonly)
    OmegaConf.set_struct(cfg, struct)
    before = OmegaConf.to_container(cfg, resolve=False)
    validate_startup_config(cfg, check_dependencies=False)
    assert OmegaConf.to_container(cfg, resolve=False) == before
    assert OmegaConf.is_readonly(cfg) == readonly
    assert OmegaConf.is_struct(cfg) == struct


@pytest.mark.parametrize("specification", ["query.a3m", "input.json:A", {"unpairedMsa": "query.a3m", "pairedMsa": None, "templates": []}])
def test_real_relative_input_assets(tmp_path, monkeypatch, specification):
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    (tmp_path / "query.a3m").write_text(f">query\n{sequence}\n")
    (tmp_path / "input.json").write_text(json.dumps({"dialect": "alphafold3", "version": 1, "name": "query", "modelSeeds": [17],
        "sequences": [{"protein": {"id": "A", "sequence": sequence, "unpairedMsa": f">query\n{sequence}\n", "pairedMsa": "", "templates": []}}]}))
    monkeypatch.chdir(tmp_path)
    cfg = minimal()
    cfg.named_files = {"input": "input.json"}
    cfg.homologies.A = specification
    before = sorted(path.name for path in tmp_path.iterdir())
    validate_startup_config(cfg, check_dependencies=False)
    assert sorted(path.name for path in tmp_path.iterdir()) == before


def test_unselected_defaults_and_input_homologies_are_not_resolved():
    cfg = minimal()
    cfg.evaluator.alphafold3 = {"model_dir": "${missing.weights}"}
    cfg.mutator.LigandMPNN = {"repo_path": "${missing.repository}"}
    cfg.homologies.unselected = {"unpairedMsa": "${missing.asset}"}
    validate_startup_config(cfg, check_dependencies=False)


def test_selected_nested_homology_resolution_error_names_the_input():
    cfg = minimal()
    cfg.homologies.A = {"unpairedMsa": "${missing.asset}"}
    with pytest.raises(StartupValidationError, match="homologies.A.unpairedMsa"):
        validate_startup_config(cfg, check_dependencies=False)
    validate_startup_config(cfg, check_dependencies=False, resolve_environment=False)


def test_interpolated_sequence_mapping_retains_leaf_validation():
    cfg = minimal()
    cfg.input_sequences = {"A": False}
    cfg.seqs = "${input_sequences}"
    with pytest.raises(StartupValidationError, match="seqs.A"):
        validate_startup_config(cfg, check_dependencies=False)


@pytest.fixture
def protomer_parser(monkeypatch):
    """Isolate the real lightweight parser's state, without resetting live globals."""
    from astrolabe.core import global_config
    from astrolabe.core import protomer_config_parser as parser

    for name, value in {"segments": {}, "named_files": {}, "predefined_statement": {},
                        "exclude_AA": [], "segment_manager": parser.SegmentManager(),
                        "nonstandard_aliases": {}, "user_ccd": None}.items():
        monkeypatch.setitem(global_config.__dict__, name, value)
    return parser, global_config


@pytest.mark.parametrize("first,second,expected", [
    ("ACD@legacy", "+legacy", "ACD"),
    ("ACD", "A_1", "ACD"),
    ("+ACD", "+A_1", "ACD"),
    ("ACD@legacy +EF", "legacy A_2", "ACDEF"),
])
def test_segment_and_protomer_aliases_match_real_parser(protomer_parser, first, second, expected):
    parser, _ = protomer_parser
    cfg = minimal()
    cfg.seqs = {"A": first, "B": second}
    cfg.types = {"A": "p", "B": "protein"}
    cfg.stages.default.oligo = ["AB"]
    validate_startup_config(cfg, check_dependencies=False)
    parser.Protomer("A", first, "p")
    parsed = parser.Protomer("B", second, "protein")
    assert parsed.sequence == expected


@pytest.mark.parametrize("statement,expected", [("+ACD EF", [False, False, False, True, True]), ("ACD +EF", [True, True, True, False, False])])
def test_fixed_segment_marker_matches_real_parser(protomer_parser, statement, expected):
    parser, _ = protomer_parser
    cfg = minimal()
    cfg.seqs.A = statement
    validate_startup_config(cfg, check_dependencies=False)
    parsed = parser.Protomer("A", statement, "p")
    assert parsed.sequence == "ACDEF"
    assert parsed.mutable_mask.tolist() == expected


@pytest.mark.parametrize("statement", ["input:A", "+input:A1-2"])
def test_named_structure_alias_matches_real_parser(protomer_parser, tmp_path, statement):
    parser, global_config = protomer_parser
    pdb = tmp_path / "input.pdb"
    pdb.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\n"
                   "ATOM      2  CA  CYS A   2       3.800   0.000   0.000  1.00 20.00           C\nTER\nEND\n")
    cfg = minimal()
    cfg.named_files = {"input": str(pdb)}
    cfg.seqs.A = statement
    validate_startup_config(cfg, check_dependencies=False)
    global_config.named_files["input"] = parser.ParsedFile(str(pdb))
    assert parser.Protomer("A", statement, "p").sequence == "AC"


@pytest.mark.parametrize("statement,expected,ccd_id", [
    ("A[SEP]G", "ASG", "SEP"),
    ("+A(SEP)G", "ASG", "SEP"),
    ("A[PHOS]G", "ASG", "SEP"),
    ("A[SEC]G", "AUG", "SEC"),
])
def test_ptm_and_nonstandard_alias_dispatch_matches_real_parser(protomer_parser, monkeypatch, statement, expected, ccd_id):
    parser, global_config = protomer_parser
    # Only chemical-dictionary lookup is isolated: real dispatch, token parsing,
    # alias resolution and one-letter extraction all execute. No AF3 import.
    global_config.nonstandard_aliases["PHOS"] = {"ccd_id": "SEP", "parent_aa": "S", "kind": "ptm"}
    monkeypatch.setattr(parser, "cached_ccd", lambda **kwargs: {
        "SEP": {"_chem_comp.one_letter_code": ["S"]}, "SEC": {"_chem_comp.one_letter_code": ["U"]}})
    cfg = minimal()
    cfg.seqs.A = statement
    validate_startup_config(cfg, check_dependencies=False)
    parsed = parser.Protomer("A", statement, "p")
    assert parsed.sequence == expected
    assert parsed.ptm_dict == {1: ccd_id}


@pytest.mark.parametrize("statement", ["AU", "AO"])
def test_unannotated_u_and_o_follow_actual_protein_literal_contract(protomer_parser, statement):
    parser, _ = protomer_parser
    # These raw letters are not accepted by the current dispatch alphabet;
    # [SEC] above can still legitimately produce U through PTM decoding.
    assert not parser.is_literal(parser.ProtomerTypes.PROTEIN, statement)
    with pytest.raises(ValueError, match="Invalid segment definition"):
        parser.Protomer("A", statement, "p")
    cfg = minimal()
    cfg.seqs.A = statement
    with pytest.raises(StartupValidationError, match="seqs.A"):
        validate_startup_config(cfg, check_dependencies=False)


@pytest.mark.parametrize("search", [None, {"algorithm": "branching", "owner": "isolated_native"},
    {"algorithm": "branching", "owner": "native_refinement"},
    {"algorithm": "branching", "owner": "archived_refinement"}])
def test_required_inputs_apply_to_builtin_sequence_workflows(search):
    cfg = minimal()
    del cfg.seqs
    cfg.search = search
    if search is not None and search["owner"] == "isolated_native":
        cfg.search.execution = {"socket_path": None, "workers": 2, "in_flight": 2,
                                "resume": False, "retry_interrupted": False}
    validate_startup_config(cfg, check_dependencies=False)
    with pytest.raises(StartupValidationError, match="seqs: required"):
        validate_startup_config(cfg, check_dependencies=False, require_inputs=True)


def test_registered_custom_search_owns_its_input_schema(monkeypatch):
    from astrolabe.configuration import SEARCH_FACTORY_CATALOG
    monkeypatch.setitem(SEARCH_FACTORY_CATALOG, ("custom", "independent_inputs"), lambda cfg, repeat: None)
    cfg = minimal()
    del cfg.seqs
    cfg.search = {"algorithm": "custom", "owner": "independent_inputs"}
    cfg.custom_input = {"document": "independent"}
    validate_startup_config(cfg, check_dependencies=False, require_inputs=True)


def test_cli_missing_sequence_fails_before_gpu_or_application(monkeypatch):
    from astrolabe import cli, app
    cfg = OmegaConf.load(FIXTURES / "invalid_missing_seqs.yaml")
    monkeypatch.setattr(cli, "_maybe_autoselect_cuda_visible_devices", lambda **kw: pytest.fail("GPU selection before input validation"))
    monkeypatch.setattr(cli, "_preinitialize_cuda_driver", lambda *a: pytest.fail("CUDA initialization before input validation"))
    monkeypatch.setattr(app, "run_from_cfg", lambda *a: pytest.fail("application before input validation"))
    with pytest.raises(StartupValidationError, match="seqs: required"):
        cli._run_from_config(cfg, autoselect_gpu=True)


def test_permissive_scope_does_not_hide_stronger_input_requirement(monkeypatch):
    from astrolabe import startup_dependencies
    from astrolabe.startup_validation import startup_validation_scope
    monkeypatch.setattr(startup_dependencies, "check_startup_dependencies", lambda cfg: None)
    cfg = minimal()
    del cfg.seqs
    with startup_validation_scope(cfg):
        with pytest.raises(StartupValidationError, match="seqs: required"):
            with startup_validation_scope(cfg, require_inputs=True):
                pytest.fail("input requirement bypassed by permissive scope")


@pytest.mark.parametrize("stages", [None, {}, {"default": None}])
def test_required_stages_preserve_permissive_partial_api(monkeypatch, stages):
    from astrolabe import cli
    cfg = minimal()
    cfg.stages = stages
    validate_startup_config(cfg, check_dependencies=False)
    monkeypatch.setattr(cli, "_maybe_autoselect_cuda_visible_devices", lambda **kw: pytest.fail("GPU selection before stage validation"))
    with pytest.raises(StartupValidationError, match="stages.default: required"):
        cli._run_from_config(cfg, autoselect_gpu=True)


def test_cli_missing_stages_fails_before_gpu_or_application(monkeypatch):
    from astrolabe import cli, app
    cfg = OmegaConf.load(FIXTURES / "invalid_missing_stages.yaml")
    validate_startup_config(cfg, check_dependencies=False)
    monkeypatch.setattr(cli, "_maybe_autoselect_cuda_visible_devices", lambda **kw: pytest.fail("GPU selection before stage validation"))
    monkeypatch.setattr(cli, "_preinitialize_cuda_driver", lambda *a: pytest.fail("CUDA initialization before stage validation"))
    monkeypatch.setattr(app, "run_from_cfg", lambda *a: pytest.fail("application before stage validation"))
    with pytest.raises(StartupValidationError, match="stages.default: required"):
        cli._run_from_config(cfg, autoselect_gpu=True)


def test_registered_custom_search_can_omit_stages_and_sequences(monkeypatch):
    from astrolabe.configuration import SEARCH_FACTORY_CATALOG
    monkeypatch.setitem(SEARCH_FACTORY_CATALOG, ("custom", "independent_inputs"), lambda cfg, repeat: None)
    cfg = minimal()
    del cfg.seqs
    del cfg.stages
    cfg.search = {"algorithm": "custom", "owner": "independent_inputs"}
    validate_startup_config(cfg, check_dependencies=False, require_inputs=True)


def test_bound_execution_passes_actual_run_preflight():
    cfg = OmegaConf.load(FIXTURES / "valid_execution_bound.yaml")
    validate_startup_config(cfg, check_dependencies=False, require_inputs=True)


@pytest.mark.parametrize("resolve_environment", [False, True])
def test_unbound_execution_is_composition_only(resolve_environment):
    cfg = OmegaConf.load(FIXTURES / "valid_execution_unbound_composition.yaml")
    validate_startup_config(cfg, check_dependencies=False, resolve_environment=resolve_environment)
    with pytest.raises(StartupValidationError, match="search.execution.socket_path: required"):
        validate_startup_config(cfg, check_dependencies=False,
                                resolve_environment=resolve_environment, require_inputs=True)


@pytest.mark.parametrize("file", ["invalid_execution_missing.yaml", "invalid_execution_empty.yaml",
    "invalid_execution_partial.yaml", "invalid_execution_unknown.yaml", "invalid_execution_null.yaml",
    "invalid_execution_unbound_actual_run.yaml"])
def test_execution_schema_fails_before_gpu_or_application(monkeypatch, file):
    from astrolabe import cli, app
    cfg = OmegaConf.load(FIXTURES / file)
    monkeypatch.setattr(cli, "_maybe_autoselect_cuda_visible_devices", lambda **kw: pytest.fail("GPU selection before execution validation"))
    monkeypatch.setattr(cli, "_preinitialize_cuda_driver", lambda *a: pytest.fail("CUDA initialization before execution validation"))
    monkeypatch.setattr(app, "run_from_cfg", lambda *a: pytest.fail("application before execution validation"))
    with pytest.raises(StartupValidationError, match="search.execution"):
        cli._run_from_config(cfg, autoselect_gpu=True)
