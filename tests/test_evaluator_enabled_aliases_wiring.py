from __future__ import annotations

import types
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from astrolabe.core import global_config
from astrolabe.evaluators.utils import construct_evaluators


def test_uses_alphafold3_evaluator_ignores_unused_unresolved_defaults() -> None:
    import run as run_mod

    class _StageUnit:
        evaluator = "Dummy"

    class _StageManager:
        default_setting = _StageUnit()
        stage_args = {"default": _StageUnit()}

    args = types.SimpleNamespace(
        evaluator=OmegaConf.create(
            {
                "alphafold3": {
                    "model_dir": "${weight_path.alphafold3}",
                    "datapipeline": {
                        "jackhmmer_binary_path": "${hmmer.jackhmmer}",
                    },
                },
                "rhofold": {
                    "checkpoint": "${rhofold.checkpoint}",
                },
                "evaluator_terms": {
                    "Dummy": {"_type": "dummy"},
                },
            }
        )
    )

    assert run_mod._uses_alphafold3_evaluator(args, _StageManager()) is False


def test_uses_alphafold3_evaluator_includes_transition_gate_evaluator() -> None:
    import run as run_mod

    class _DefaultStage:
        evaluator = "Protenix"

    class _StageUnit:
        evaluator = "Protenix"
        transition_consistency_gate = types.SimpleNamespace(evaluator="AF3")

    class _StageManager:
        default_setting = _DefaultStage()
        stage_args = {"default": _DefaultStage(), "stage_1": _StageUnit()}

    args = types.SimpleNamespace(
        evaluator=OmegaConf.create(
            {
                "evaluator_terms": {
                    "Protenix": {"_type": "protenix"},
                    "AF3": {"_type": "alphafold3"},
                },
            }
        )
    )

    assert run_mod._uses_alphafold3_evaluator(args, _StageManager()) is True


def test_construct_evaluators_does_not_use_global_stage_manager_for_filtering(
    tmp_path: Path,
) -> None:
    """Without explicit enabled_aliases, invalid evaluator_terms should fail fast."""
    global_config.reset()

    class _StageUnit:
        evaluator = "Dummy"

    class _StageManager:
        default_setting = _StageUnit()
        stage_args = {"default": _StageUnit()}

    global_config.init_const("stage_manager", _StageManager())

    evaluator_conf = types.SimpleNamespace(
        evaluator_terms={
            "Dummy": {"_type": "dummy"},
            "Unused": {"_type": "this_is_not_a_real_evaluator"},
        }
    )

    with pytest.raises(ValueError, match="Unknown evaluator name"):
        construct_evaluators(evaluator_conf, raw_model_results_dir=tmp_path, oligo_names=None)


def test_initialize_pipeline_components_passes_enabled_aliases_to_construct_evaluators(
    tmp_path: Path, monkeypatch
) -> None:
    global_config.reset()

    import run as run_mod

    calls: dict[str, object] = {}

    def _fake_construct_evaluators(
        evaluator_conf,
        raw_model_results_dir,
        oligo_names=None,
        *,
        enabled_aliases=None,
    ):
        calls["enabled_aliases"] = enabled_aliases
        return {alias: object() for alias in enabled_aliases}

    monkeypatch.setattr(run_mod, "construct_evaluators", _fake_construct_evaluators)
    monkeypatch.setattr(run_mod, "initialize_global_config", lambda _args: None)
    monkeypatch.setattr(run_mod, "setup_protomers", lambda _args: None)
    monkeypatch.setattr(run_mod, "apply_initializer", lambda _args: None)
    monkeypatch.setattr(run_mod, "build_mutators", lambda _conf, _dir: {})
    monkeypatch.setattr(run_mod, "construct_standard_input", lambda *_a, **_k: {})
    monkeypatch.setattr(run_mod, "initialize_predockers", lambda _p: None)
    monkeypatch.setattr(run_mod, "_uses_alphafold3_evaluator", lambda *_a, **_k: False)

    args = types.SimpleNamespace(
        raw_model_results_dir=tmp_path,
        mutator=types.SimpleNamespace(mutator_terms={}, enabled_aliases=[]),
        loss={"loss_terms": {}},
        stages=OmegaConf.create(
            {
                "default": {
                    "oligo": ["AB"],
                    "mutation_num": 1,
                    "weighted_score_expression": "plddt",
                    "evaluator": "Dummy",
                    "mutators": ["MCMC"],
                    "mutator_weights": [1.0],
                }
            }
        ),
        evaluator=types.SimpleNamespace(
            evaluator_terms={"Dummy": {"_type": "dummy"}},
            enabled_aliases=["Dummy"],
        ),
        seed=1,
        predocker=None,
        obabel_path=None,
        molfile_to_params=None,
        dssp_path=None,
        usalign_path=None,
    )

    run_mod.initialize_pipeline_components(args)

    assert calls == {}
    assert set(global_config.evaluators) == {"Dummy"}
    assert calls == {}
    assert global_config.stage_manager.current_evaluator is global_config.evaluators["Dummy"]
    assert calls["enabled_aliases"] == ["Dummy"]


@pytest.mark.parametrize("enter_joint", [False, True])
def test_lazy_evaluators_defer_future_models_and_close_only_live_instances(
    tmp_path, monkeypatch, enter_joint
) -> None:
    import run as run_mod
    from astrolabe.evaluators.sequence import SequenceEvaluator

    global_config.reset()
    constructed = []
    closed = []

    def construct(_config, _output, _oligos, *, enabled_aliases):
        alias, = enabled_aliases
        constructed.append(alias)
        evaluator = SequenceEvaluator()
        evaluator.shutdown = lambda: closed.append(alias)
        return {alias: evaluator}

    monkeypatch.setattr(run_mod, "construct_evaluators", construct)
    config = OmegaConf.create({"evaluator_terms": {
        "secondary": {"_type": "sequence"},
        "joint": {"_type": "future_model", "weights": "${missing.weights}"},
        "unused": {"_type": "future_model"},
        "disabled": None,
    }})
    evaluators = run_mod.LazyEvaluatorMap(config, tmp_path, ["AA"])
    global_config.init_const("evaluators", evaluators)

    assert set(evaluators) == {"secondary", "joint", "unused"}
    assert "joint" in evaluators
    assert constructed == []
    scores = evaluators["secondary"].evaluate({"AA": object()})
    assert scores.names == ["AA"]
    assert scores.plddt == {}
    assert scores.iptm == {}
    assert scores.complex_paths == {}
    assert scores.complex_poses is None
    assert constructed == ["secondary"]
    if enter_joint:
        assert evaluators["joint"] is evaluators["joint"]
        assert constructed == ["secondary", "joint"]
    run_mod._shutdown_pipeline_resources()
    assert closed == constructed
    assert "unused" not in constructed


def test_af3_future_stage_does_not_require_initial_af3_token_setup() -> None:
    import run as run_mod

    secondary = types.SimpleNamespace(evaluator="secondary")
    joint = types.SimpleNamespace(evaluator="joint")
    stages = types.SimpleNamespace(
        default_setting=secondary,
        current_setting=secondary,
        stage_args={"default": secondary, "joint": joint},
    )
    args = types.SimpleNamespace(evaluator={"evaluator_terms": {
        "secondary": {"_type": "sequence"},
        "joint": {"_type": "alphafold3"},
    }})
    assert run_mod._uses_alphafold3_evaluator(args, stages) is True
    assert run_mod._uses_alphafold3_evaluator(args, stages, current_only=True) is False
