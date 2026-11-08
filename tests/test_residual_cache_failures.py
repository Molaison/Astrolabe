"""Regressions for stale score/debug reuse and premature native publication."""
import os
import pickle
import random
from types import SimpleNamespace

import numpy as np
import pytest

from astrolabe.core import global_config
from astrolabe.core.folding_input import Input, ProteinChain
from astrolabe.evaluators.utils import StandardScores
from astrolabe.losses.base import LossEngine


@pytest.fixture(autouse=True)
def state(tmp_path):
    global_config.reset()
    global_config.init_const("protomers", {})
    global_config.init_const("out_dir", tmp_path)
    yield
    global_config.reset()


def engine(function, **params):
    result = LossEngine({"loss_terms": {}}, validate_config=False)
    result.loss_specs = {"metric": ("metric", function, params)}
    return result


def scores(**kwargs):
    return StandardScores(names=["A"], plddt={"A": [np.array([0.2, 0.4])]},
                          complex_paths={}, **kwargs)


def test_score_array_mutation_is_seen_without_replacing_result():
    value = scores()
    loss = engine(lambda result: {"A": float(result.plddt["A"][0].mean())})
    assert loss._extract_scores_with_registry(value)[1]["metric"] == pytest.approx(0.3)
    value.plddt["A"][0][0] = 0.8
    assert loss._extract_scores_with_registry(value)[1]["metric"] == pytest.approx(0.6)


def test_score_structure_same_size_and_mtime_cannot_reuse_old_value(tmp_path):
    path = tmp_path / "source.pdb"
    path.write_text("1.0")
    value = scores()
    value.complex_paths["A"] = path
    loss = engine(lambda result: {"A": float(result.complex_paths["A"].read_text())})
    assert loss._extract_scores_with_registry(value)[1]["metric"] == 1
    stamp = path.stat()
    path.write_text("2.0")
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert loss._extract_scores_with_registry(value)[1]["metric"] == 2


def test_opaque_live_pose_cannot_authorize_stale_score_reuse():
    pose = SimpleNamespace(value=1)
    value = scores(complex_poses=SimpleNamespace(poses={"A": pose}))
    loss = engine(lambda result: {"A": result.complex_poses.poses["A"].value})
    assert loss._extract_scores_with_registry(value)[1]["metric"] == 1
    pose.value = 2
    assert loss._extract_scores_with_registry(value)[1]["metric"] == 2


def test_score_parameters_do_not_use_truncated_array_repr():
    params = np.zeros(2000)
    loss = engine(lambda result, weights: {"A": float(weights.sum())}, weights=params)
    value = scores()
    assert loss._extract_scores_with_registry(value)[1]["metric"] == 0
    params[1000] = 5
    assert loss._extract_scores_with_registry(value)[1]["metric"] == 5


def test_mutating_returned_score_mapping_does_not_poison_cache():
    loss = engine(lambda result: {"A": 1})
    value = scores()
    first = loss._extract_scores_with_registry(value)
    first[1]["metric"] = 100
    assert loss._extract_scores_with_registry(value)[1]["metric"] == 1


def test_unchanged_portable_scores_keep_cache_hit():
    calls = []
    loss = engine(lambda result: (calls.append(1) or {"A": 1}))
    value = scores()
    assert loss._extract_scores_with_registry(value) == loss._extract_scores_with_registry(value)
    assert calls == [1]


def test_mask_change_is_seen_and_cannot_be_silently_checkpointed():
    from astrolabe.design.native_scores import snapshot_scores, UnsupportedNativeScoresState
    from astrolabe.design.store import DataStore
    value = scores()
    array = np.ma.array([1.0, 9.0], mask=[False, True])
    value.plddt["A"] = [array]
    loss = engine(lambda result: {"A": float(result.plddt["A"][0].mean())})
    assert loss._extract_scores_with_registry(value)[1]["metric"] == 1
    array.mask[:] = [True, False]
    assert loss._extract_scores_with_registry(value)[1]["metric"] == 9
    with pytest.raises(UnsupportedNativeScoresState, match="explicit native codec"):
        snapshot_scores(value, DataStore())


def test_masked_parameters_do_not_reuse_data_only_identity():
    weights = np.ma.array([1.0, 9.0], mask=[False, True])
    loss = engine(lambda result, weights: {"A": float(weights.mean())}, weights=weights)
    value = scores()
    assert loss._extract_scores_with_registry(value)[1]["metric"] == 1
    weights.mask[:] = [True, False]
    assert loss._extract_scores_with_registry(value)[1]["metric"] == 9


def test_structured_parameter_dtype_cannot_collide():
    weights = np.array([(2.0, 5.0)], dtype=[("a", "f8"), ("b", "f8")])
    loss = engine(lambda result, weights: {"A": float(weights["a"][0])}, weights=weights)
    value = scores()
    assert loss._extract_scores_with_registry(value)[1]["metric"] == 2
    weights.dtype = np.dtype([("b", "f8"), ("a", "f8")])
    assert loss._extract_scores_with_registry(value)[1]["metric"] == 5


class PureBackend:
    alias = "pure"

    def __init__(self, calls, value=0.2):
        self.calls, self.value = calls, value

    def debug_cache_identity(self):
        # Explicit contract: no hidden state to advance and no external effects.
        return {"version": "fixture-pure/v1", "value": self.value}

    def evaluate(self, inputs, runtime):
        self.calls.append(self.value)
        return StandardScores(names=["A"], plddt={"A": [self.value]}, complex_paths={})


@pytest.fixture
def debug_run(tmp_path, monkeypatch):
    from astrolabe.app import _load_run_module
    module = _load_run_module()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEBUG", "1")
    monkeypatch.setattr(module, "_apply_post_eval_hooks", lambda args, result, **kwargs: result)
    monkeypatch.setattr(module, "_finalize_evaluation_result", lambda args, result, loss: result)
    inputs = {"A": Input(name="A", chains=[ProteinChain(id="A", sequence="AA", ptms=[])],
                         rng_seeds=[11])}
    return module, inputs


def test_debug_cache_binds_backend_and_input(debug_run):
    module, inputs = debug_run
    calls = []
    assert module.load_or_evaluate_initial(None, PureBackend(calls), inputs, {}, "sample").plddt["A"] == [0.2]
    assert module.load_or_evaluate_initial(None, PureBackend(calls, 0.7), inputs, {}, "sample").plddt["A"] == [0.7]
    inputs["A"].rng_seeds.append(23)
    module.load_or_evaluate_initial(None, PureBackend(calls, 0.7), inputs, {}, "sample")
    assert calls == [0.2, 0.7, 0.7]


def test_debug_legacy_pickle_is_not_loaded_or_overwritten(debug_run, tmp_path):
    module, inputs = debug_run
    path = tmp_path / "sample_eval_rslt.pkl"
    old = pickle.dumps(scores())
    path.write_bytes(old)
    calls = []
    module.load_or_evaluate_initial(None, PureBackend(calls), inputs, {}, "sample")
    assert calls == [0.2]
    assert path.read_bytes() == old


def test_debug_rng_consuming_backend_cannot_publish_reusable_cache(debug_run):
    module, inputs = debug_run
    calls = []

    class RandomBackend(PureBackend):
        def evaluate(self, inputs, runtime):
            result = super().evaluate(inputs, runtime)
            result.plddt["A"] = [random.random()]
            return result

    random.seed(42)
    first = module.load_or_evaluate_initial(None, RandomBackend(calls), inputs, {}, "random")
    second = module.load_or_evaluate_initial(None, RandomBackend(calls), inputs, {}, "random")
    assert first.plddt != second.plddt
    assert len(calls) == 2


def test_debug_pure_cache_hit_still_runs_hooks(debug_run, monkeypatch):
    module, inputs = debug_run
    calls, hooks = [], []

    def hook(args, result, **kwargs):
        hooks.append(1)
        result.plddt["A"] = [0.8]
        return result

    monkeypatch.setattr(module, "_apply_post_eval_hooks", hook)
    first = module.load_or_evaluate_initial(None, PureBackend(calls), inputs, {}, "pure")
    second = module.load_or_evaluate_initial(None, PureBackend(calls), inputs, {}, "pure")
    assert first.plddt == second.plddt == {"A": [0.8]}
    assert calls == [0.2] and hooks == [1, 1]


def test_unqualified_backend_never_uses_debug_shortcut(debug_run):
    module, inputs = debug_run
    calls = []

    class StatefulBackend(PureBackend):
        def debug_cache_identity(self):
            return None

    for _ in range(2):
        module.load_or_evaluate_initial(None, StatefulBackend(calls), inputs, {}, "stateful")
    assert calls == [0.2, 0.2]


def test_debug_corrupt_publication_is_a_miss(debug_run, tmp_path):
    module, inputs = debug_run
    calls = []
    module.load_or_evaluate_initial(None, PureBackend(calls), inputs, {}, "corrupt")
    path = tmp_path / "corrupt_initial_debug.cache"
    assert path.is_file()
    path.write_bytes(path.read_bytes()[:-7])
    module.load_or_evaluate_initial(None, PureBackend(calls), inputs, {}, "corrupt")
    assert calls == [0.2, 0.2]


def test_debug_publication_and_cleanup_failure_do_not_replace_result(debug_run, monkeypatch):
    from pathlib import Path
    from astrolabe.evaluators import debug_cache
    module, inputs = debug_run
    calls = []

    def fail_replace(*args):
        raise OSError("publication cut")

    real_unlink = Path.unlink

    def fail_staging_unlink(path, *args, **kwargs):
        if path.name.startswith(".fault_initial_debug.cache."):
            raise OSError("cleanup cut")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(debug_cache.os, "replace", fail_replace)
    monkeypatch.setattr(Path, "unlink", fail_staging_unlink)
    result = module.load_or_evaluate_initial(None, PureBackend(calls), inputs, {}, "fault")
    assert result.plddt == {"A": [0.2]} and calls == [0.2]


def test_native_get_cannot_publish_while_registration_is_in_progress(monkeypatch):
    from astrolabe.design import native_support as module
    from astrolabe.design.store import DataStore
    from astrolabe.design.types import key_for
    value = SimpleNamespace(shutdown=lambda: None)
    owner = module.LazyNativeOwner(lambda: value, DataStore(), key_for("owner"), rng_modules=())
    premature = []

    def registration(registered):
        with pytest.raises(RuntimeError, match="construction|initialization|progress"):
            premature.append(registered.get())

    monkeypatch.setattr(module, "register_model_owner", registration)
    receipt = owner._initialize(owner.initialization_request)
    owner._bind_ready(receipt)
    assert premature == []
    assert owner.get() is value
