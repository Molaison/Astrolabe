from __future__ import annotations

import pytest

from astrolabe.core import global_config


@pytest.fixture(autouse=True)
def reset_runtime():
    global_config.reset()
    yield
    global_config.reset()


def test_runtime_context_defaults_for_uninitialized_state() -> None:
    global_config.reset()
    ctx = global_config.build_runtime_context()
    assert ctx.protomers == {}
    assert ctx.mutable_protomers == {}
    assert ctx.mutators == {}
    assert ctx.evaluators == {}
    assert ctx.stage_manager is None
    assert ctx.step == 0


def test_runtime_context_reflects_global_state() -> None:
    global_config.reset()
    ctx = global_config.build_runtime_context()
    global_config.init_const("protomers", {"A": object()})
    global_config.init_const("mutable_protomers", {"A": object()})
    ctx.set_variable("step", 7)
    assert set(ctx.protomers.keys()) == {"A"}
    assert set(ctx.mutable_protomers.keys()) == {"A"}
    assert ctx.step == 7


def test_reset_keeps_old_context_on_old_run_and_new_context_on_new_run():
    global_config.reset()
    old = global_config.build_runtime_context()
    global_config.init_const("protomers", {"old": object()})
    global_config.set_variable("step", 5)
    global_config.reset()
    new = global_config.build_runtime_context()
    with pytest.raises(RuntimeError):
        global_config.get_const("protomers")
    global_config.init_const("protomers", {"new": object()})
    global_config.set_variable("step", 9)
    assert list(old.protomers) == ["old"]
    assert old.step == 5
    old.set_variable("step", 6)
    assert old.step == 6 and new.step == global_config.step == 9
    assert list(new.protomers) == ["new"]


def test_explicit_store_drives_real_stage_and_loss_consumers_and_thread():
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace
    from omegaconf import OmegaConf
    from astrolabe.losses.base import LossEngine
    from astrolabe.stage_manager import StageManager

    global_config.reset()
    global_config.set_variable("step", 100)
    state = global_config.RuntimeState()
    state.init_const("protomers", {"A": SimpleNamespace(sequence="AAAA")})
    state.set_variable("step", 7)
    ctx = global_config.build_runtime_context(state)
    loss = LossEngine({"loss_terms": {}}, runtime_context=ctx, validate_config=False)
    manager = StageManager(OmegaConf.create({"default": {
        "oligo": ["A", "~AB"], "mutation_num": 1, "weighted_score_expression": "step",
        "evaluator": "Eval0", "mutators": ["MCMC"],
    }}), loss_computer=loss, runtime_context=ctx)
    state.init_const("stage_manager", manager)
    manager._sync_design_complexes()
    assert state.get_variable("ignored_complexes") == ["AB"]
    assert global_config.get_variable("ignored_complexes") is None
    assert loss._default_expression_symbols()["step"] == 7
    with ThreadPoolExecutor(max_workers=1) as executor:
        assert executor.submit(lambda: (ctx.step, global_config.step)).result() == (7, 100)
    global_config.reset()
    assert loss._default_expression_symbols()["step"] == 7
    assert manager.runtime_context.state is state


def test_validation_and_legacy_private_monkeypatch_use_one_store(monkeypatch):
    global_config.reset()
    context = global_config.build_runtime_context()
    with pytest.raises(ValueError):
        global_config.init_const("missing", 1)
    with pytest.raises(RuntimeError):
        global_config.update_const("protomers", {})
    global_config.init_const("protomers", {"a": 1})
    global_config.init_const("protomers", {"a": 1})
    with pytest.raises(RuntimeError):
        global_config.init_const("protomers", {"different": 1})
    monkeypatch.setattr(global_config, "_protomers", {"patched": 2})
    assert context.protomers == {"patched": 2}
    monkeypatch.setattr(global_config, "_step", 4)
    assert context.step == 4
    assert "_protomers" not in vars(global_config)
    assert "_step" not in vars(global_config)
