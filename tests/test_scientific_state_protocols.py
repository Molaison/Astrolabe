"""Faults at scientific restore and initialization publication boundaries."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import Protomer, SegmentManager
from astrolabe.design.legacy import LegacyContext, UnsupportedLegacyState
from astrolabe.design.native_support import LazyNativeOwner, ModelReady
from astrolabe.design.store import DataStore, design_codec
from astrolabe.design.types import key_for
from astrolabe.model_execution import capture_random_state


@pytest.fixture(autouse=True)
def run_store():
    global_config.reset()
    for name in ("segments", "protomers", "mutable_protomers", "named_files"):
        global_config.init_const(name, {})
    global_config.init_const("segment_manager", SegmentManager())
    yield
    global_config.reset()


def shared_graph():
    global_config.protomers["A"] = Protomer("A", "AC", "p")
    shared = global_config.segments["A_1"]
    global_config.segments["shared"] = shared
    global_config.protomers["B"] = Protomer("B", "shared", "p")
    return shared, global_config.segment_manager


def test_restore_preserves_shared_preupdate_state_and_rejection_rng():
    shared, manager = shared_graph()
    shared.add_mutations((0, "A", "G"))
    manager.begin_pre_update_step()
    manager.preapply_pending_mutations()
    shared.add_mutations((1, "C", "V"))
    context = LegacyContext()
    snapshot = context.capture_state()
    rng = np.random.get_state()
    baseline_draw = np.random.random()
    manager.finish_pre_update_step(accepted=False)
    baseline = (shared.sequence, shared.mutations)
    context.restore_state(snapshot)
    # State restoration never rewinds RNG; only the existing RNG receipt owner
    # can restore it at a named phase boundary.
    expected_next = np.random.random()
    np.random.set_state(rng)
    assert np.random.random() == baseline_draw
    manager.finish_pre_update_step(accepted=False)
    assert np.random.random() == expected_next
    assert (shared.sequence, shared.mutations) == baseline
    assert global_config.segments["shared"] is shared
    assert len(manager.unique_segments) == 1


def test_invalid_restore_does_not_apply_an_earlier_part_of_snapshot():
    shared, _ = shared_graph()
    context = LegacyContext()
    snapshot = context.capture_state()
    shared.sequence = "GG"

    class BrokenCopy:
        def __deepcopy__(self, memo):
            raise OSError("cut during snapshot decoding")

    broken = replace(snapshot, protomers=(
        replace(snapshot.protomers[0], attributes={"bad": BrokenCopy()}),
        *snapshot.protomers[1:],
    ))
    with pytest.raises(OSError, match="snapshot decoding"):
        context.restore_state(broken)
    assert shared.sequence == "GG"
    assert global_config.protomers["A"].name == "A"


def test_old_context_and_snapshot_cannot_write_a_new_run():
    shared, _ = shared_graph()
    old = global_config.build_runtime_context()
    context = LegacyContext()
    snapshot = context.capture_state()
    global_config.reset()
    assert old.get_const("segments")["A_1"] is shared
    with pytest.raises(RuntimeError, match="not the active run"):
        context.restore_state(snapshot)
    with pytest.raises(RuntimeError, match="not the active run"):
        context.capture_candidate()
    assert global_config.build_runtime_context().run_id != old.run_id


def test_snapshot_rejects_replaced_segment_graph_before_writing():
    shared, manager = shared_graph()
    context = LegacyContext()
    snapshot = context.capture_state()
    manager.unique_segments.clear()
    shared.sequence = "GG"
    with pytest.raises(UnsupportedLegacyState, match="another segment graph"):
        context.restore_state(snapshot)
    assert shared.sequence == "GG"


def test_failed_snapshot_capture_does_not_latch_active_context(monkeypatch):
    shared_graph()
    context = LegacyContext()
    original = context.capture_state
    def fail():
        raise OSError("capture cut")
    monkeypatch.setattr(context, "capture_state", fail)
    with pytest.raises(OSError, match="capture cut"):
        with context.activate():
            pytest.fail("entered after failed snapshot")
    assert not context.active and context.owner_thread is None
    monkeypatch.setattr(context, "capture_state", original)
    with context.activate():
        assert context.active


def test_initialization_receipt_does_not_load_model_and_reload_preserves_rng():
    data = DataStore()
    calls = []
    def construct():
        calls.append(np.random.random())
        return SimpleNamespace(shutdown=lambda: None)
    first = LazyNativeOwner(construct, data, key_for("model"), rng_modules=())
    ready = first._initialize(first.initialization_request)
    first._bind_ready(ready)
    restored = LazyNativeOwner(construct, data, first.identity, rng_modules=())
    restored._bind_ready(ready)
    assert restored.initialization_status.init_proven
    assert restored.initialization_status.residency == "absent"
    assert len(calls) == 1
    before = data.put(capture_random_state(), design_codec)
    restored.get()
    assert calls == [calls[0], calls[0]]
    assert data.put(capture_random_state(), design_codec) == before
    assert restored.initialization_status.residency == "resident"
    first.shutdown()
    restored.shutdown()


def test_failed_construction_cannot_be_replayed_in_same_owner():
    calls = []
    def construct():
        calls.append("native effect")
        raise OSError("constructor cut")
    owner = LazyNativeOwner(construct, DataStore(), key_for("model"), rng_modules=())
    with pytest.raises(OSError, match="constructor cut"):
        owner._initialize(owner.initialization_request)
    with pytest.raises(RuntimeError, match="initialization failed"):
        owner._initialize(owner.initialization_request)
    with pytest.raises(RuntimeError, match="initialization failed"):
        owner.get()
    assert calls == ["native effect"]
    assert owner.initialization_status.residency == "failed"


def test_bad_initialization_receipt_does_not_publish_proof():
    data = DataStore()
    owner = LazyNativeOwner(lambda: pytest.fail("must not load"), data, key_for("model"))
    other_store = DataStore()
    missing = other_store.put({"missing": True}, design_codec)
    with pytest.raises(Exception):
        owner._bind_ready(ModelReady(owner.identity, missing))
    assert not owner.ready


def test_pipeline_partial_initialization_cannot_repeat(monkeypatch):
    from astrolabe.app import _load_run_module
    run = _load_run_module()
    prepared = run.PreparedPipelineState(global_config.build_runtime_context(), "random")
    calls = []
    def fail(*args, **kwargs):
        calls.append("math")
        raise OSError("math cut")
    monkeypatch.setattr(run, "_initialize_pipeline_math", fail)
    args = SimpleNamespace(predocker=None)
    with pytest.raises(OSError, match="math cut"):
        run.prepare_pipeline_math(args, prepared)
    assert prepared.initialization_state == "failed"
    with pytest.raises(RuntimeError, match="initialization is failed"):
        run.prepare_pipeline_math(args, prepared)
    assert calls == ["math"]


def test_lazy_evaluator_failed_batch_is_not_reconstructed(monkeypatch, tmp_path):
    from astrolabe.app import _load_run_module
    run = _load_run_module()
    calls = []
    def fail(*args, **kwargs):
        calls.append("construct")
        raise OSError("weights loaded then failed")
    monkeypatch.setattr(run, "construct_evaluators", fail)
    config = OmegaConf.create({"evaluator_terms": {"seq": {"_type": "sequence"}}})
    evaluators = run.LazyEvaluatorMap(config, tmp_path, ["A"])
    with pytest.raises(OSError, match="weights loaded"):
        evaluators["seq"]
    with pytest.raises(RuntimeError, match="initialization failed"):
        evaluators["seq"]
    assert calls == ["construct"] and evaluators.constructed_items() == ()


def test_score_binding_tracks_parameter_change_and_rejects_cross_run():
    from astrolabe.losses.base import LossEngine
    engine = LossEngine({"loss_terms": {}}, validate_config=False)
    engine.loss_specs = {"metric": ("metric", lambda result, scale: scale, {"scale": 1})}
    scores = SimpleHashableScores()
    assert engine._extract_scores_with_registry(scores)[1]["metric"] == 1
    engine.loss_specs["metric"][2]["scale"] = 2
    assert engine._extract_scores_with_registry(scores)[1]["metric"] == 2
    global_config.reset()
    with pytest.raises(RuntimeError, match="not the active run"):
        engine._extract_scores_with_registry(scores)


class SimpleHashableScores:
    names = ("A",)


def test_validation_scope_cannot_reuse_proof_after_config_mutation(monkeypatch):
    from astrolabe import startup_dependencies
    from astrolabe.startup_validation import (
        StartupValidationError, startup_validation_scope, validate_startup_config,
    )
    monkeypatch.setattr(startup_dependencies, "check_startup_dependencies", lambda cfg: None)
    cfg = OmegaConf.create({
        "seed": 17, "steps": 2,
        "evaluator": {"evaluator_terms": {"first": {"_type": "sequence"}}},
        "mutator": {"mutator_terms": {"m": {"_type": "mcmc"}}},
        "stages": {"default": {"evaluator": "first", "mutators": ["m"],
                               "weighted_score_expression": "0", "mutation_num": 1}},
    })
    with startup_validation_scope(cfg):
        cfg.steps = -1
        with pytest.raises(StartupValidationError, match="steps"):
            validate_startup_config(cfg)
