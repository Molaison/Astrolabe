from types import SimpleNamespace

import pytest

import run
from astrolabe.core import global_config


def test_failed_release_cleans_other_owners_and_never_records_completed(monkeypatch):
    calls, states, raw_cleanup = [], [], []
    class Owner:
        def __init__(self, name, fail=False):
            self.name, self.fail = name, fail
        def close(self):
            calls.append(self.name)
            if self.fail:
                raise RuntimeError("worker outcome unknown")
    first, second = Owner("first", True), Owner("second")
    global_config.reset()
    global_config.init_const("evaluators", {"first": first, "alias": first, "second": second})
    args = SimpleNamespace(out_dir="unused", logging=None, task_name="release")
    monkeypatch.setattr(run, "parse_arguments", lambda _: args)
    monkeypatch.setattr(run, "setup_logging", lambda *a: None)
    monkeypatch.setattr(run, "initialize_pipeline_components", lambda _: (None, SimpleNamespace(current_evaluator=None), {}))
    monkeypatch.setattr(run, "load_or_evaluate_initial", lambda *a, **k: None)
    monkeypatch.setattr(run, "run_optimization_loop", lambda *a: None)
    monkeypatch.setattr(run, "_finish_run_provenance", lambda _args, state, **k: states.append(state))
    monkeypatch.setattr(run, "_cleanup_auto_managed_raw_dir", lambda **k: raw_cleanup.append(k["succeeded"]))
    try:
        with pytest.raises(RuntimeError, match="release was not confirmed"):
            run._run_pipeline({})
        assert calls == ["first", "second"]
        assert states == ["failed"] and raw_cleanup == [False]
    finally:
        global_config.reset()


def test_scientific_failure_is_not_replaced_by_cleanup_failure(monkeypatch):
    states = []
    args = SimpleNamespace(out_dir="unused", logging=None, task_name="release")
    monkeypatch.setattr(run, "parse_arguments", lambda _: args)
    monkeypatch.setattr(run, "setup_logging", lambda *a: None)
    def science(_):
        raise ValueError("scientific failure")
    def cleanup():
        raise RuntimeError("release failed")
    monkeypatch.setattr(run, "initialize_pipeline_components", science)
    monkeypatch.setattr(run, "_shutdown_pipeline_resources", cleanup)
    monkeypatch.setattr(run, "_finish_run_provenance", lambda _args, state, **k: states.append(state))
    monkeypatch.setattr(run, "_cleanup_auto_managed_raw_dir", lambda **k: None)
    with pytest.raises(ValueError, match="scientific failure"):
        run._run_pipeline({})
    assert states == ["failed"]
