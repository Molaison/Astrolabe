from __future__ import annotations

import random
import json
import threading
import os
import time
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest

from astrolabe import evaluator_startup, model_execution
from astrolabe.evaluators.registry import EvaluatorRegistry
from astrolabe.evaluators.utils import construct_evaluators


def test_later_invalid_alias_prevents_all_construction(monkeypatch, tmp_path):
    monkeypatch.setitem(EvaluatorRegistry._registry, "startup_good",
                        lambda *a, **k: pytest.fail("earlier model already constructed"))
    cfg = OmegaConf.create({"evaluator_terms": {
        "first": {"_type": "startup_good"},
        "later": {"_type": "startup_nonexistent"},
    }})
    with pytest.raises(ValueError, match="later"):
        construct_evaluators(cfg, tmp_path)


def test_later_unresolved_defaults_prevent_all_construction(monkeypatch, tmp_path):
    factory = lambda *a, **k: pytest.fail("earlier model already constructed")
    monkeypatch.setitem(EvaluatorRegistry._registry, "startup_good", factory)
    monkeypatch.setitem(EvaluatorRegistry._registry, "startup_bad", factory)
    cfg = OmegaConf.create({"startup_bad": {"path": "${missing.path}"}, "evaluator_terms": {
        "first": {"_type": "startup_good"}, "later": {"_type": "startup_bad"},
    }})
    with pytest.raises(ValueError, match="later"):
        construct_evaluators(cfg, tmp_path)


def test_partial_constructor_owner_and_completed_evaluator_are_closed(monkeypatch, tmp_path):
    closed = []

    class Owner:
        attached = False
        def __init__(self, name):
            self.name = name
        def shutdown(self):
            closed.append(self.name)

    def bad(*args, **kwargs):
        model_execution.register_model_owner(Owner("partial"))
        raise ValueError("checkpoint truncated")

    monkeypatch.setitem(EvaluatorRegistry._registry, "startup_good",
                        lambda *a, **k: Owner("completed"))
    monkeypatch.setitem(EvaluatorRegistry._registry, "startup_bad", bad)
    cfg = OmegaConf.create({"evaluator_terms": {
        "first": {"_type": "startup_good"}, "later": {"_type": "startup_bad"},
    }})
    with pytest.raises(RuntimeError, match="later.*checkpoint truncated"):
        construct_evaluators(cfg, tmp_path)
    assert closed == ["partial", "completed"]


def test_parallel_construction_overlaps_and_preserves_order(monkeypatch):
    monkeypatch.setattr(evaluator_startup, "_INDEPENDENT_IMPORTS", {"a": (), "b": ()})
    barrier = threading.Barrier(2)
    plan = [("one", "a", None, {}), ("two", "b", None, {})]
    def build(entry):
        barrier.wait(timeout=5)
        return entry[0]
    results = evaluator_startup.initialize_plan(plan, build, lambda value: None,
                                                workers=2, seed=17)
    assert list(results) == ["one", "two"]


def test_seeded_serial_parallel_match_without_changing_parent_rng(monkeypatch):
    monkeypatch.setattr(evaluator_startup, "_INDEPENDENT_IMPORTS", {"a": ()})
    plan = [("one", "a", None, {}), ("two", "a", None, {})]
    before = model_execution.capture_random_state()
    def build(entry):
        state = model_execution._INITIALIZATION_RANDOM_STATE.get()
        rng = random.Random()
        rng.setstate(state["python"])
        return rng.random(), state["numpy"], state.get("torch")
    serial = evaluator_startup.initialize_plan(plan, build, lambda value: None,
                                               workers=1, seed=17)
    parallel = evaluator_startup.initialize_plan(plan, build, lambda value: None,
                                                 workers=2, seed=17)
    assert serial == parallel
    assert serial["one"] != serial["two"]
    assert model_execution.capture_random_state() == before


def test_native_client_initialization_does_not_restore_parent_rng(monkeypatch):
    from astrolabe import model_daemon
    seen = []
    class Owner:
        def __init__(self, spec, **kwargs):
            seen.append(spec.settings["_initial_random_state"])
        def request(self, request):
            return {"metadata": {}, "random_state": seen[-1]}
        def shutdown(self):
            pass
    monkeypatch.setattr(model_daemon, "OwnedModelDaemon", Owner)
    monkeypatch.setattr(model_execution, "restore_random_state",
                        lambda state: pytest.fail("restored parent RNG during isolated initialization"))
    with model_execution.independent_initialization(9):
        model_execution.NativeModelClient("dummy", "dummy:factory", {})
    assert seen[0]["torch_seed"] == 9


def test_parallel_failure_drains_and_closes_late_success(monkeypatch):
    monkeypatch.setattr(evaluator_startup, "_INDEPENDENT_IMPORTS", {"a": (), "b": ()})
    barrier = threading.Barrier(2)
    closed = []
    def build(entry):
        barrier.wait(timeout=5)
        if entry[0] == "bad":
            raise ValueError("missing weight")
        return entry[0]
    with pytest.raises(RuntimeError, match="bad.*missing weight"):
        evaluator_startup.initialize_plan(
            [("bad", "a", None, {}), ("good", "b", None, {})],
            build, closed.append, workers=2, seed=17,
        )
    assert closed == ["good"]


def test_context_propagates_model_ownership(monkeypatch):
    monkeypatch.setattr(evaluator_startup, "_INDEPENDENT_IMPORTS", {"a": (), "b": ()})
    closed = []
    def build(entry):
        owner = SimpleNamespace(attached=False, shutdown=lambda: closed.append(entry[0]))
        model_execution.register_model_owner(owner)
        return owner
    with model_execution.model_run_scope():
        evaluator_startup.initialize_plan(
            [("one", "a", None, {}), ("two", "b", None, {})],
            build, lambda value: value.shutdown(), workers=2, seed=17,
        )
        assert closed == []
    assert sorted(closed) == ["one", "two"]


def test_real_cpu_daemons_overlap_preserve_seed_and_release_pids(tmp_path, monkeypatch):
    """Exercise the actual subprocess/describe/RNG/owner path, without models."""
    monkeypatch.setattr(evaluator_startup, "_INDEPENDENT_IMPORTS", {"a": ()})
    (tmp_path / "startup_fixture.py").write_text(
        "import os, random, time\n"
        "from pathlib import Path\n"
        "from astrolabe.model_execution import kernel_handler\n"
        "def factory(settings):\n"
        "    def load(cfg):\n"
        "        if cfg['barrier']:\n"
        "            root = Path(cfg['root'])\n"
        "            (root / (cfg['alias'] + '.ready')).touch()\n"
        "            deadline = time.monotonic() + 10\n"
        "            while len(list(root.glob('*.ready'))) < 2:\n"
        "                if time.monotonic() >= deadline:\n"
        "                    raise RuntimeError('constructors did not overlap')\n"
        "                time.sleep(0.01)\n"
        "        return lambda payload: payload, {'value': random.random(), 'pid': os.getpid()}\n"
        "    return kernel_handler(settings, load)\n",
        encoding="utf-8",
    )
    pythonpath = str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", "")
    plan = [("one", "a", None, {}), ("two", "a", None, {})]
    before = model_execution.capture_random_state()
    outputs = []
    pids = []
    for workers in (1, 2):
        def build(entry):
            client = model_execution.NativeModelClient(
                entry[1], "startup_fixture:factory",
                {"root": str(tmp_path), "alias": entry[0], "barrier": workers > 1},
                env={"PYTHONPATH": pythonpath, "CUDA_VISIBLE_DEVICES": ""},
                startup_timeout=20,
            )
            pids.append(client.pid)
            return client
        with model_execution.model_run_scope():
            clients = evaluator_startup.initialize_plan(
                plan, build, lambda c: c.shutdown(), workers=workers, seed=17,
            )
            outputs.append({alias: c.metadata["value"] for alias, c in clients.items()})
            assert model_execution.capture_random_state() == before
    assert outputs[0] == outputs[1]
    assert outputs[0]["one"] != outputs[0]["two"]
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def test_native_load_sibling_is_cancelled_before_its_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluator_startup, "_INDEPENDENT_IMPORTS", {"a": (), "b": ()})
    (tmp_path / "hung_loader.py").write_text(
        "import os, time\nfrom pathlib import Path\n"
        "def factory(settings):\n"
        "    Path(settings['pidfile']).write_text(str(os.getpid()))\n"
        "    while True: time.sleep(1)\n", encoding="utf-8")
    pidfile = tmp_path / "child.pid"
    plan = [("hung", "a", None, {}), ("bad", "b", None, {})]
    def build(entry):
        if entry[0] == "bad":
            deadline = time.monotonic() + 10
            while not pidfile.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert pidfile.exists()
            raise ValueError("bad checkpoint")
        return model_execution.NativeModelClient(
            "a", "hung_loader:factory", {"pidfile": str(pidfile)},
            env={"PYTHONPATH": str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", ""),
                 "CUDA_VISIBLE_DEVICES": ""}, startup_timeout=15)
    start = time.monotonic()
    with pytest.raises(evaluator_startup.EvaluatorInitializationError, match="bad.*bad checkpoint"):
        evaluator_startup.initialize_plan(plan, build, lambda client: client.shutdown(),
                                          workers=2, seed=11)
    assert time.monotonic() - start < 10
    with pytest.raises(ProcessLookupError):
        os.kill(int(pidfile.read_text()), 0)


def test_delayed_spawn_sibling_failure_returns_before_deadline_and_reaps(tmp_path, monkeypatch):
    from astrolabe import model_daemon
    monkeypatch.setattr(evaluator_startup, "_INDEPENDENT_IMPORTS", {"a": (), "b": ()})
    (tmp_path / "late_factory.py").write_text(
        "def make(settings):\n    return lambda payload: payload\n", encoding="utf-8")
    entered, release = threading.Event(), threading.Event()
    real_popen = model_daemon.subprocess.Popen
    owners = []
    class RecordingOwner(model_daemon.OwnedModelDaemon):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            owners.append(self)
    def delayed_popen(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return real_popen(*args, **kwargs)
    monkeypatch.setattr(model_daemon, "OwnedModelDaemon", RecordingOwner)
    monkeypatch.setattr(model_daemon.subprocess, "Popen", delayed_popen)
    def build(entry):
        if entry[0] == "bad":
            assert entered.wait(5)
            raise ValueError("bad checkpoint")
        return model_execution.NativeModelClient(
            "a", "late_factory:make", {},
            env={"PYTHONPATH": str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", ""),
                 "CUDA_VISIBLE_DEVICES": ""}, startup_timeout=8)
    started = time.monotonic()
    try:
        with pytest.raises(evaluator_startup.EvaluatorInitializationError,
                           match="bad.*bad checkpoint"):
            evaluator_startup.initialize_plan(
                [("late", "a", None, {}), ("bad", "b", None, {})],
                build, lambda client: client.shutdown(), workers=2, seed=11)
        assert time.monotonic() - started < 2
        assert len(owners) == 1
        owners[0].shutdown()
    finally:
        release.set()
        for owner in owners:
            owner._launch_thread.join(timeout=5)
            owner.shutdown()
    assert not owners[0]._launch_thread.is_alive()
    lifecycle = json.loads((owners[0].temp_dir / "lifecycle.json").read_text())
    assert lifecycle["status"] == "stopped"
    with pytest.raises(ProcessLookupError):
        os.kill(lifecycle["pid"], 0)


def test_base_exception_preserves_original_and_reports_cleanup(monkeypatch):
    monkeypatch.setattr(evaluator_startup, "_INDEPENDENT_IMPORTS", {"a": ()})
    class Owner:
        attached = False
        def shutdown(self):
            raise OSError("cleanup failed")
    def build(_entry):
        model_execution.register_model_owner(Owner())
        raise KeyboardInterrupt("interrupted")
    with pytest.raises(KeyboardInterrupt, match="interrupted") as error:
        evaluator_startup.initialize_plan([("one", "a", None, {})], build,
                                          lambda value: None, workers=1, seed=3)
    assert "cleanup failed" in " ".join(error.value.__notes__)


def test_seeded_af2_prepares_import_root_before_module_import(monkeypatch):
    from astrolabe.evaluators import utils
    events = []
    monkeypatch.setattr(utils, "_ensure_alphafold2_import_path",
                        lambda data_dir: events.append(("root", data_dir)))
    monkeypatch.setattr(evaluator_startup.importlib, "import_module",
                        lambda module: events.append(("import", module)))
    plan = [("af2_alias", "alphafold2", None,
             {"execution_mode": "daemon", "data_dir": "/fixture/root"})]
    assert evaluator_startup.initialize_plan(plan, lambda entry: "ready", lambda _: None,
                                             workers=1, seed=7) == {"af2_alias": "ready"}
    assert events[0] == ("root", "/fixture/root")
    assert events[1] == ("import", "astrolabe.evaluators.alphafold2")


def test_seeded_import_error_names_alias_and_type(monkeypatch):
    monkeypatch.setattr(evaluator_startup.importlib, "import_module",
                        lambda module: (_ for _ in ()).throw(ModuleNotFoundError("missing dep")))
    plan = [("af2_alias", "alphafold2", None, {"execution_mode": "daemon"})]
    with pytest.raises(evaluator_startup.EvaluatorInitializationError,
                       match="af2_alias.*alphafold2.*missing dep"):
        evaluator_startup.initialize_plan(plan, lambda entry: pytest.fail("build ran"),
                                          lambda _: None, workers=1, seed=7)


def test_real_failing_factory_reports_alias_type_and_checkpoint(tmp_path, monkeypatch):
    from astrolabe import model_daemon
    monkeypatch.setattr(evaluator_startup, "_INDEPENDENT_IMPORTS", {"a": ()})
    (tmp_path / "bad_factory.py").write_text(
        "def make(settings):\n    raise ValueError('checkpoint header is corrupt')\n",
        encoding="utf-8")
    owners = []
    class RecordingOwner(model_daemon.OwnedModelDaemon):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            owners.append(self)
    monkeypatch.setattr(model_daemon, "OwnedModelDaemon", RecordingOwner)
    def build(entry):
        return model_execution.NativeModelClient(
            "a", "bad_factory:make", {},
            env={"PYTHONPATH": str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", ""),
                 "CUDA_VISIBLE_DEVICES": ""}, startup_timeout=5)
    with pytest.raises(evaluator_startup.EvaluatorInitializationError,
                       match="bad_alias.*ValueError.*checkpoint header is corrupt"):
        evaluator_startup.initialize_plan(
            [("bad_alias", "a", None, {})], build,
            lambda client: client.shutdown(), workers=1, seed=17)
    assert len(owners) == 1
    owner = owners[0]
    owner.shutdown()
    lifecycle = json.loads((owner.temp_dir / "lifecycle.json").read_text())
    assert lifecycle["status"] == "stopped"
    with pytest.raises(ProcessLookupError):
        os.kill(lifecycle["pid"], 0)
