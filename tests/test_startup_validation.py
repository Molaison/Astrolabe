from __future__ import annotations

import sys

from omegaconf import OmegaConf
import pytest

from astrolabe.startup_validation import StartupValidationError, validate_startup_config


def config(**updates):
    value = {
        "seed": 17, "steps": 2,
        "evaluator": {"evaluator_terms": {"first": {"_type": "sequence"}}},
        "mutator": {"mutator_terms": {"m": {"_type": "mcmc"}}},
        "stages": {"default": {"evaluator": "first", "mutators": ["m"],
                               "weighted_score_expression": "0", "mutation_num": 1}},
    }
    value.update(updates)
    return OmegaConf.create(value)


def test_aggregate_config_errors_without_scientific_imports():
    before = set(sys.modules)
    cfg = config(seed=-1, steps="many", evaluator={"evaluator_terms": {
        "first": {"_type": "unknown-model"},
        "broken": {},
    }}, stages={"default": {"evaluator": "first"},
                "later": {"evaluator": "broken", "weighted_score_expression": "a + )"}})
    with pytest.raises(StartupValidationError) as caught:
        validate_startup_config(cfg, check_dependencies=False)
    text = str(caught.value)
    for fragment in ("unknown-model", "broken._type", "weighted_score_expression", "seed", "steps"):
        assert fragment in text
    assert not {"torch", "pyrosetta", "jax", "run"} & (set(sys.modules) - before)


def test_unused_backend_defaults_are_not_resolved():
    cfg = config()
    cfg.evaluator.alphafold3 = {"model_dir": "${nonexistent.weights}"}
    assert validate_startup_config(cfg, check_dependencies=False) == {}


@pytest.mark.parametrize("mode", ["resident_serial", "adaptive"])
def test_inprocess_af2_rejects_same_process_resource_owner(mode, monkeypatch):
    monkeypatch.delenv("ASTROLABE_RESOURCE_SOCKET", raising=False)
    cfg = config(
        evaluator={"evaluator_terms": {"first": {
            "_type": "alphafold2", "execution_mode": "in_process",
        }}},
        execution={"mode": mode},
    )
    before = set(sys.modules)
    with pytest.raises(StartupValidationError, match="in_process.*execution.mode=legacy"):
        validate_startup_config(cfg, check_dependencies=False)
    assert not {"torch", "pyrosetta", "jax", "run"} & (set(sys.modules) - before)


@pytest.mark.parametrize("execution, inherited", [
    ({"mode": "legacy"}, False),
    ({"mode": "resident_serial"}, True),
])
def test_inprocess_af2_accepts_external_or_legacy_resource_owner(execution, inherited, monkeypatch):
    monkeypatch.delenv("ASTROLABE_RESOURCE_SOCKET", raising=False)
    if inherited:
        monkeypatch.setenv("ASTROLABE_RESOURCE_SOCKET", "/external/owner.sock")
    cfg = config(
        evaluator={"evaluator_terms": {"first": {
            "_type": "alphafold2", "execution_mode": "in_process",
        }}},
        execution=execution,
    )
    assert validate_startup_config(cfg, check_dependencies=False) == {}


def test_selected_backend_defaults_are_resolved():
    cfg = config(evaluator={
        "evaluator_terms": {"first": {"_type": "alphafold3"}},
        "alphafold3": {"model_dir": "${nonexistent.weights}"},
    })
    with pytest.raises(StartupValidationError, match="nonexistent.weights"):
        validate_startup_config(cfg, check_dependencies=False)


def test_later_stage_alias_cannot_disappear():
    cfg = config()
    cfg.stages.later = {"evaluator": "typo"}
    with pytest.raises(StartupValidationError, match="typo.*undefined"):
        validate_startup_config(cfg, check_dependencies=False)


def test_invalid_config_fails_before_gpu_selection_or_application(monkeypatch):
    from astrolabe import cli
    monkeypatch.setattr(cli, "_maybe_autoselect_cuda_visible_devices",
                        lambda **kw: pytest.fail("GPU selection ran before validation"))
    with pytest.raises(StartupValidationError, match="steps"):
        cli._run_from_config(config(steps=-1), autoselect_gpu=True)


def test_python_api_fails_before_runner_import(monkeypatch):
    from astrolabe import app
    monkeypatch.setattr(app, "_load_run_module",
                        lambda: pytest.fail("runner imported before validation"))
    with pytest.raises(StartupValidationError, match="steps"):
        app.Astrolabe(config(steps=-1)).run()


def test_required_loss_parameters_are_checked():
    cfg = config(loss={"loss_terms": {"holes": {"_type": "interface_holes"}}})
    with pytest.raises(StartupValidationError, match="holes.protomer_a"):
        validate_startup_config(cfg, check_dependencies=False)


def test_explicit_inactive_losses_not_validated():
    cfg = config(loss={"loss_terms": {"holes": {"_type": "interface_holes"}}},
                 stages={"default": {"evaluator": "first", "mutators": ["m"],
                                     "weighted_score_expression": "0", "mutation_num": 1,
                                     "active_loss_terms": []}})
    assert validate_startup_config(cfg, check_dependencies=False) == {}


def test_parallel_without_independent_seed_rejected():
    cfg = config()
    cfg.evaluator.initialization = {"workers": 2}
    with pytest.raises(StartupValidationError, match="initialization.seed"):
        validate_startup_config(cfg, check_dependencies=False)


def test_scoped_validation_does_not_recheck_dependencies(monkeypatch):
    from astrolabe import startup_dependencies
    from astrolabe.startup_validation import startup_validation_scope
    calls = []
    monkeypatch.setattr(startup_dependencies, "check_startup_dependencies",
                        lambda cfg: calls.append(cfg))
    cfg = config()
    with startup_validation_scope(cfg):
        with startup_validation_scope(cfg):
            validate_startup_config(cfg)
    assert len(calls) == 1
    validate_startup_config(cfg)
    assert len(calls) == 2


def test_component_merge_overrides_missing_default_and_keeps_root_context():
    from astrolabe.component_config import merge_component_config
    cfg = config()
    cfg.path = "/model"
    cfg.evaluator.sequence = {"model_name": "${missing.path}", "device": "cpu"}
    cfg.evaluator.evaluator_terms.first.model_name = "${path}"
    merged = merge_component_config(cfg.evaluator, ("sequence",), cfg.evaluator.evaluator_terms.first)
    assert merged.model_name == "/model"
    assert merged.device == "cpu"
    assert OmegaConf.to_container(cfg.evaluator.sequence, resolve=False)["model_name"] == "${missing.path}"
    validate_startup_config(cfg, check_dependencies=False)


@pytest.mark.parametrize("readonly,struct", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("resolve", [True, False])
def test_component_merge_preserves_flagged_input(readonly, struct, resolve):
    from astrolabe.component_config import merge_component_config
    cfg = config()
    cfg.path = "/model"
    cfg.evaluator.sequence = {"model_name": "${missing.path}"}
    cfg.evaluator.evaluator_terms.first.model_name = "${path}"
    OmegaConf.set_readonly(cfg, readonly)
    OmegaConf.set_struct(cfg, struct)
    before = OmegaConf.to_container(cfg, resolve=False)
    flags = (OmegaConf.is_readonly(cfg), OmegaConf.is_struct(cfg),
             OmegaConf.is_readonly(cfg.evaluator), OmegaConf.is_struct(cfg.evaluator))

    merged = merge_component_config(
        cfg.evaluator, ("sequence",), cfg.evaluator.evaluator_terms.first,
        resolve=resolve,
    )

    assert OmegaConf.to_container(merged, resolve=True)["model_name"] == "/model"
    assert OmegaConf.to_container(cfg, resolve=False) == before
    assert flags == (OmegaConf.is_readonly(cfg), OmegaConf.is_struct(cfg),
                     OmegaConf.is_readonly(cfg.evaluator), OmegaConf.is_struct(cfg.evaluator))


def test_interpolated_type_selects_real_registry_defaults():
    from astrolabe.component_config import merge_component_config
    from astrolabe.evaluators.utils import _merge_conf_with_defaults
    cfg = config()
    cfg.kind = "sequence"
    cfg.path = "/model"
    cfg.evaluator.sequence = {"model_name": "${missing.path}", "device": "cpu"}
    cfg.evaluator.evaluator_terms.first._type = "${kind}"
    cfg.evaluator.evaluator_terms.first.model_name = "${path}"
    merged = merge_component_config(
        cfg.evaluator, (cfg.evaluator.evaluator_terms.first._type,),
        cfg.evaluator.evaluator_terms.first,
    )
    assert merged.device == "cpu"
    assert merged.model_name == "/model"
    consumer_merged = _merge_conf_with_defaults(
        cfg.evaluator, cfg.evaluator.evaluator_terms.first._type,
        cfg.evaluator.evaluator_terms.first, alias="first",
    )
    assert consumer_merged.device == "cpu"
    assert consumer_merged.model_name == "/model"
    assert "_type" not in consumer_merged
    validate_startup_config(cfg, check_dependencies=False)


def test_resolved_stage_and_expression_cycle():
    cfg = config()
    cfg.chosen = "first"
    cfg.score = "x"
    cfg.stages.default.evaluator = "${chosen}"
    cfg.stages.default.weighted_score_expression = "${score}"
    cfg.loss = {"expression_terms": {"x": "y", "y": "x"}}
    with pytest.raises(StartupValidationError, match="cyclic dependency"):
        validate_startup_config(cfg, check_dependencies=False)
    cfg.score = "0"
    validate_startup_config(cfg, check_dependencies=False)


def test_runtime_loss_metadata_replaces_source_requirements(monkeypatch):
    from astrolabe.losses.base import loss_registry
    previous_fn = loss_registry._loss_functions.get("sequence_residue_fraction")
    previous_meta = loss_registry._metadata.get("sequence_residue_fraction")
    @loss_registry.register(name="sequence_residue_fraction")
    def replacement(eval_result):
        return 0
    try:
        cfg = config(loss={"loss_terms": {"fraction": {"_type": "sequence_residue_fraction"}}})
        assert "fraction" in validate_startup_config(cfg, check_dependencies=False)
        cfg.loss.loss_terms.fraction.typo = 1
        with pytest.raises(StartupValidationError, match="invalid parameter"):
            validate_startup_config(cfg, check_dependencies=False)
    finally:
        if previous_fn is None:
            loss_registry._loss_functions.pop("sequence_residue_fraction", None)
        else:
            loss_registry._loss_functions["sequence_residue_fraction"] = previous_fn
        if previous_meta is None:
            loss_registry._metadata.pop("sequence_residue_fraction", None)
        else:
            loss_registry._metadata["sequence_residue_fraction"] = previous_meta


def test_mutator_preplans_before_first_factory(monkeypatch, tmp_path):
    from astrolabe.mutators.base import build_mutators
    from astrolabe.mutators.registry import MutatorRegistry
    calls = []
    def factory(*args, **kwargs):
        calls.append(1)
        return object()
    monkeypatch.setitem(MutatorRegistry._registry, "probe", factory)
    cfg = config(mutator={"mutator_terms": {
        "first": {"_type": "probe"}, "later": {"_type": "not_a_mutator"}}})
    cfg.stages.default.mutators = ["first", "later"]
    with pytest.raises(StartupValidationError, match="not_a_mutator"):
        validate_startup_config(cfg, check_dependencies=False)
    with pytest.raises(ValueError, match="not_a_mutator"):
        build_mutators(cfg.mutator, tmp_path)
    assert calls == []


def test_overridden_mutator_default_passes_preflight_and_real_factory(tmp_path):
    from astrolabe.mutators.base import build_mutators
    cfg = config(mutator={
        "MCMC": {"mutation_method": "${missing.method}", "select_positions": "plddt"},
        "mutator_terms": {"m": {"_type": "mcmc", "mutation_method": "frequency_adjusted"}},
    })
    validate_startup_config(cfg, check_dependencies=False)
    built = build_mutators(cfg.mutator, tmp_path)
    assert built["m"].mutation_method == "frequency_adjusted"


@pytest.mark.parametrize("overlay,fragment", [
    ({"predocker": {"predocker_terms": {"p": None}}}, "null processor"),
    ({"predocker": {"predocker_terms": {"p": {"_type": "not_a_predocker"}}}}, "unknown predocker"),
    ({"initializer": {"mode": "wrong"}}, "unknown initializer"),
    ({"structure_start": {"mode": "wrong"}}, "structure_start.mode"),
])
def test_other_config_is_rejected_early(overlay, fragment):
    with pytest.raises(StartupValidationError, match=fragment):
        validate_startup_config(config(**overlay), check_dependencies=False)


def test_selected_backend_helper_is_wired_to_merged_mutator():
    cfg = config(mutator={
        "LigandMPNN": {"model_type": "unsupported"},
        "mutator_terms": {"m": {"_type": "ligandmpnn"}},
    })
    with pytest.raises(StartupValidationError, match="mutator.mutator_terms.m.model_type"):
        validate_startup_config(cfg, check_dependencies=False)


def test_dry_run_does_not_check_backend_path_existence(tmp_path):
    cfg = config(evaluator={"evaluator_terms": {
        "first": {"_type": "rf3", "repo_path": str(tmp_path / "not-installed")},
    }})
    validate_startup_config(cfg, check_dependencies=False, resolve_environment=False)
    with pytest.raises(StartupValidationError, match="repo_path"):
        validate_startup_config(cfg, check_dependencies=False)


def test_selected_initializers_and_selector_expressions():
    cfg = config(initializer={"mode": "aa-set", "aa_set": {"segments": {"loop": "AG"}}})
    validate_startup_config(cfg, check_dependencies=False)
    cfg.initializer = {"mode": "mask_mix", "mask_fraction": 2}
    with pytest.raises(StartupValidationError, match="initializer.mask_fraction"):
        validate_startup_config(cfg, check_dependencies=False)
    cfg.initializer = {"mode": "random"}
    cfg.mutator.selector_expression = "m: nonexistent_selector"
    with pytest.raises(StartupValidationError, match="unknown selector"):
        validate_startup_config(cfg, check_dependencies=False)


def test_predocker_chain_reference_and_entrance_expression():
    cfg = config()
    cfg.mutator.mutator_terms.m.predocker = ["missing"]
    with pytest.raises(StartupValidationError, match="unknown predocker alias"):
        validate_startup_config(cfg, check_dependencies=False)
    cfg.mutator.mutator_terms.m.pop("predocker")
    cfg.stages.later = {"_entrance": "x > )", "evaluator": "first"}
    with pytest.raises(StartupValidationError, match="stages.later._entrance"):
        validate_startup_config(cfg, check_dependencies=False)


def test_stage_and_active_loss_fail_before_construction():
    cfg = config()
    cfg.stages.default.pop("mutation_num")
    with pytest.raises(StartupValidationError, match="stages.default.mutation_num"):
        validate_startup_config(cfg, check_dependencies=False)
    cfg.stages.default.mutation_num = 1
    cfg.stages.default.active_loss_terms = ["unknown"]
    with pytest.raises(StartupValidationError, match="unknown loss alias"):
        validate_startup_config(cfg, check_dependencies=False)


@pytest.mark.parametrize("bad", ["nonsense", "NaN", float("nan"), float("inf"), [1]])
@pytest.mark.parametrize("stage_name", ["default", "later"])
def test_invalid_stage_mutation_num_fails_preflight(stage_name, bad):
    cfg = config()
    if stage_name == "later":
        cfg.stages.later = {"_entrance": "True", "mutation_num": bad}
    else:
        cfg.stages.default.mutation_num = bad
    with pytest.raises(StartupValidationError, match=rf"stages\.{stage_name}\.mutation_num"):
        validate_startup_config(cfg, check_dependencies=False)


def test_stage_mutation_num_accepts_percentage_and_negative_native_value():
    from types import SimpleNamespace
    from astrolabe.stage_config import resolve_mutation_num
    from astrolabe.stage_manager import resolve_mutation_num as consumer_resolver
    assert consumer_resolver is resolve_mutation_num
    assert resolve_mutation_num("50%", protomer_pool={"A": SimpleNamespace(mutable_length=20)}) == 10
    cfg = config()
    cfg.stages.default.mutation_num = "25%"
    cfg.stages.later = {"_entrance": "True", "mutation_num": "50%"}
    validate_startup_config(cfg, check_dependencies=False)
    cfg.stages.later.mutation_num = -1
    validate_startup_config(cfg, check_dependencies=False)


def test_nonsense_mutation_num_fails_before_runner_import(monkeypatch):
    from astrolabe import app
    cfg = config()
    cfg.stages.default.mutation_num = "nonsense"
    monkeypatch.setattr(app, "_load_run_module",
                        lambda: pytest.fail("runner imported before mutation_num validation"))
    with pytest.raises(StartupValidationError, match="stages.default.mutation_num"):
        app.Astrolabe(cfg).run()


def test_loss_extra_and_required_parameters():
    cfg = config(loss={"loss_terms": {"fraction": {"_type": "sequence_residue_fraction"}}})
    with pytest.raises(StartupValidationError, match="fraction.included_protomers"):
        validate_startup_config(cfg, check_dependencies=False)
    cfg.loss.loss_terms.fraction.included_protomers = "A"
    cfg.loss.loss_terms.fraction.residues = "A"
    cfg.loss.loss_terms.fraction.typo = 1
    with pytest.raises(StartupValidationError, match="fraction.typo"):
        validate_startup_config(cfg, check_dependencies=False)


def test_loss_names_preserve_native_registry_case():
    from astrolabe.startup_validation import _source_specs
    assert "bindcraft_interface_dG" in _source_specs()["loss"]
    assert "bindcraft_interface_dSASA" in _source_specs()["loss"]
    assert "bindcraft_interface_dg" not in _source_specs()["loss"]
    cfg = config(loss={"loss_terms": {
        "dg": {"_type": "bindcraft_interface_dG", "chain_a": "A", "chain_b": "B"},
        "dsasa": {"_type": "bindcraft_interface_dSASA", "chain_a": "A", "chain_b": "B"},
    }})
    validate_startup_config(cfg, check_dependencies=False, resolve_environment=False)
    cfg.loss.loss_terms.dg._type = "bindcraft_interface_dg"
    with pytest.raises(StartupValidationError, match="unknown loss 'bindcraft_interface_dg'"):
        validate_startup_config(cfg, check_dependencies=False, resolve_environment=False)


def test_intrinsic_oligomer_specific_matches_loss_engine():
    from astrolabe.losses.base import LossEngine
    term = {
        "_type": "sequence_residue_fraction",
        "included_protomers": "A", "residues": "A",
        "oligomer_specific": ["A"],
    }
    cfg = config(loss={"loss_terms": {"x": term}})
    assert validate_startup_config(cfg, check_dependencies=False)["x"] == term
    engine = LossEngine({"loss_terms": {"x": term}})
    engine.setup("x", None, None)
    assert engine.parsed_defined_losses["x"][2]["oligomer_specific"] == ["A"]


def test_live_loss_registry_preserves_exact_case():
    from astrolabe.losses.base import loss_registry
    name = "MixedCaseStartupProbe"
    @loss_registry.register(name=name)
    def probe(eval_result):
        return 0
    try:
        cfg = config(loss={"loss_terms": {"probe": {"_type": name}}})
        validate_startup_config(cfg, check_dependencies=False)
        cfg.loss.loss_terms.probe._type = name.lower()
        with pytest.raises(StartupValidationError, match="unknown loss"):
            validate_startup_config(cfg, check_dependencies=False)
    finally:
        loss_registry._loss_functions.pop(name, None)
        loss_registry._metadata.pop(name, None)


def test_predocker_consumer_merges_selected_alias_before_resolve(monkeypatch):
    from astrolabe.core import global_config
    from astrolabe.structure_processors.base import (
        StructureProcessor, StructureProcessorRegistry, initialize_predockers,
    )
    class Probe(StructureProcessor):
        def __init__(self, *, value, _type):
            self.value = value
        def _run_impl(self, input_file):
            return input_file
    monkeypatch.setitem(StructureProcessorRegistry._registry, "probe", Probe)
    cfg = OmegaConf.create({
        "target": 7,
        "predocker": {
            "probe": {"value": "${missing.value}"},
            "enabled_aliases": ["p"],
            "predocker_terms": {
                "p": {"_type": "probe", "value": "${target}"},
                "unused": {"_type": "unknown"},
            },
        },
    })
    global_config.reset()
    try:
        initialize_predockers(cfg.predocker)
        assert global_config.predockers["p"].value == 7
        assert set(global_config.predockers) == {"p"}
    finally:
        global_config.reset()
