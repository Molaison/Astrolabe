from pathlib import Path
import sys

from omegaconf import OmegaConf
import pytest

from astrolabe.backend_startup import validate_backend_settings
from astrolabe.component_config import merge_component_config
from astrolabe.component_contracts import INFERENCE_FIELDS, normalize_evaluator_options
from astrolabe.configuration import load_composed_config
from astrolabe.startup_validation import StartupValidationError, validate_startup_config


@pytest.mark.parametrize("kind,fields", INFERENCE_FIELDS.items())
def test_common_inference_controls_reach_native_settings(kind, fields):
    public = {name: index + 2 for index, name in enumerate(fields)}
    before = dict(public)
    native = OmegaConf.create(normalize_evaluator_options(kind, public))
    for name, target in fields.items():
        assert OmegaConf.select(native, target) == public[name]
    assert public == before


def test_alias_overrides_defaults_across_public_and_native_spellings():
    config = OmegaConf.create({"budget": 4, "evaluator": {
        "alphafold3": {"recycles": 10, "samples": 5},
        "evaluator_terms": {"fold": {"_type": "alphafold3", "num_recycles": "${budget}", "samples": 2}},
    }})
    before = OmegaConf.to_container(config, resolve=False)
    alias = config.evaluator.evaluator_terms.fold
    merged = merge_component_config(config.evaluator, "alphafold3", alias)
    assert merged.num_recycles == 4
    assert merged.num_samples == 2
    assert OmegaConf.to_container(config, resolve=False) == before


@pytest.mark.parametrize("kind,options", [
    ("alphafold3", {"recycles": 3, "num_recycles": 4}),
    ("protenix", {"recycles": 3, "model": {"N_cycle": 4}}),
    ("boltz", {"samples": 3, "diffusion_samples": 4}),
])
def test_conflicting_spellings_in_same_layer_fail(kind, options):
    with pytest.raises(ValueError, match="conflicts"):
        normalize_evaluator_options(kind, options)


@pytest.mark.parametrize("kind,options", [
    ("alphafold2", {"samples": 2}),
    ("alphafold3", {"diffusion_steps": 10}),
    ("sequence", {"recycles": 3}),
])
def test_unsupported_common_controls_are_not_ignored(kind, options):
    with pytest.raises(ValueError, match="does not expose"):
        normalize_evaluator_options(kind, options)


@pytest.mark.parametrize("options,field", [
    ({"adam": {"lr": 1}}, "adam"),
    ({"c_s": 1}, "c_s"),
    ({"model": {"pairformer": {"n_blocks": 1}}}, "model.pairformer"),
    ({"model": {"constraint_embedder": {"pocket_embedder": {"enable": True}}}}, "model.constraint_embedder"),
    ({"data": {"train_sets": ["bad"]}}, "data.train_sets"),
    ({"sample_diffusion": {"sigma_data": 1}}, "sample_diffusion.sigma_data"),
    ({"sample_diffusion": {"N_sample_mini_rollout": 1}}, "sample_diffusion.N_sample_mini_rollout"),
    ({"skip_amp": {"confidence_head": False}}, "skip_amp"),
    ({"input_json_path": "frozen.json"}, "input_json_path"),
    ({"dump_dir": "/elsewhere"}, "dump_dir"),
])
def test_native_training_architecture_and_request_overrides_fail(options, field):
    before = set(sys.modules)
    errors = validate_backend_settings("evaluator", "protenix", options,
                                      path="evaluator.evaluator_terms.fold", check_paths=False)
    assert any(f".{field}:" in error for error in errors)
    assert not {"torch", "jax", "pyrosetta"} & (set(sys.modules) - before)


def test_proton_native_cap_is_owned_by_current_stage_budget():
    errors = validate_backend_settings("mutator", "proton_pottsmpnn",
                                      {"criteria": {"max_mutations": 3}}, path="mutator.p", check_paths=False)
    assert any("criteria.max_mutations: derived per proposal" in error for error in errors)


@pytest.mark.parametrize("extra", [{"mutation_num": 4}, {"mutator": {"mutation_num": 4}}])
def test_legacy_count_location_fails_before_model_import(extra):
    config = OmegaConf.create({"steps": 0, "seed": 17,
                              "evaluator": {"evaluator_terms": {"fold": {"_type": "sequence"}}},
                              "mutator": {"mutator_terms": {"m": {"_type": "mcmc"}}},
                              "stages": {"default": {"evaluator": "fold", "mutators": ["m"],
                                                      "weighted_score_expression": "0", "mutation_num": 1}}})
    config = OmegaConf.merge(config, extra)
    with pytest.raises(StartupValidationError, match=r"obsolete count location; configure stages\.\*\.mutation_num"):
        validate_startup_config(config, check_dependencies=False)


def test_composed_defaults_leave_unselected_environment_unresolved():
    directory = Path(__file__).resolve().parents[1] / "config"
    config = load_composed_config(directory / "base.yaml")
    merged = merge_component_config(config.evaluator, "alphafold3", {"model_dir": "/weights", "samples": 1})
    assert merged.num_recycles == 10
    assert merged.num_samples == 1
    assert merged.bucket_size == 40
    assert merged.refstructure_cache_max_entries == 20000
    assert "mutation_num" not in config
    assert "mutation_num" not in config.mutator
    assert "adam" not in config.evaluator.protenix
    assert "model" not in config.evaluator.protenix


def test_standalone_af3_owner_receives_native_common_controls(tmp_path):
    from astrolabe.evaluators.af3_service import load_service_config

    path = tmp_path / "owner.yaml"
    path.write_text("recycles: 2\nsamples: 3\nmodel_dir: /weights\n")
    owner = load_service_config(path)
    assert owner.num_recycles == 2
    assert owner.num_samples == 3
    assert "recycles" not in owner


def test_af3_owner_snapshot_uses_native_controls(tmp_path):
    from astrolabe.evaluators.af3_lifecycle import OwnedAF3

    config = OmegaConf.create({"evaluator": {"alphafold3": {
        "recycles": 2, "samples": 3, "model_dir": "/weights"}}})
    owner = OwnedAF3(config, tmp_path, {})
    snapshot = OmegaConf.load(owner.config_path)
    assert snapshot.num_recycles == 2
    assert snapshot.num_samples == 3
    assert config.evaluator.alphafold3.recycles == 2
