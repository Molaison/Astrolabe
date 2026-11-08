"""CPU fault injection for the remaining r48 lifecycle/auxiliary boundaries.

No model, GPU, network service or real mount is used. Vendor functions are
compiled from their actual AST with dependency stubs, not copied algorithms.
"""
from __future__ import annotations

import ast
import __future__
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor/proton_pottsmpnn/foundry"


def load_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compile_functions(path, names, namespace, *, class_name=None):
    tree = ast.parse(path.read_text())
    nodes = tree.body
    if class_name is not None:
        nodes = next(node for node in nodes if isinstance(node, ast.ClassDef)
                     and node.name == class_name).body
    selected = [node for node in nodes if isinstance(node, ast.FunctionDef)
                and node.name in names]
    assert {node.name for node in selected} == set(names)
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(module, str(path), "exec", flags=__future__.annotations.compiler_flag), namespace)


def test_m06_borrow_release_quarantine_and_failed_cleanup_remain_owned():
    from astrolabe.model_daemon import DaemonSpec, RequestOutcomeUnknown
    from astrolabe.runtime.model_pool import local_model_pool, pool_owns_registration

    class Owner:
        pid = os.getpid()

        def __init__(self, spec):
            self.spec = spec
            self.fail = False
            self.closes = 0
            assert pool_owns_registration(self)

        def request(self, request):
            if self.fail:
                raise RequestOutcomeUnknown("lost result")
            return {"ok": True, "metadata": {}, "random_state": {"python": [1]}}

        def shutdown(self):
            self.closes += 1
            raise RuntimeError("cleanup fault")

    pool = local_model_pool()
    pool.__enter__()
    spec = DaemonSpec("esmfold", "unused:factory", {"_initial_random_state": {"python": [1]}})
    borrowed = pool.acquire(spec, Owner, {}, "identity")
    owner = pool._owners[0]
    borrowed.shutdown()
    borrowed.shutdown()
    assert owner.closes == 0 and borrowed._entry.borrowers == 0
    second = pool.acquire(spec, Owner, {}, "identity")
    owner.fail = True
    with pytest.raises(RequestOutcomeUnknown):
        second.request({"operation": "infer", "seed": 71})
    with pytest.raises(RuntimeError, match="quarantined"):
        pool.acquire(spec, Owner, {}, "identity")
    with pytest.raises(RuntimeError, match="failed to close"):
        pool.close()
    pool.close()  # Ordinary close neither retries nor erases the failed owner.
    assert pool._owners == [owner] and owner.closes == 1
    assert pool.metadata["cleanup_errors"] and not pool.metadata["cleanup_complete"]


def test_m06_partially_constructed_owner_stays_in_pool():
    from astrolabe.model_daemon import DaemonSpec
    from astrolabe.runtime.model_pool import local_model_pool, pool_owns_registration
    partial = []

    class Owner:
        def __init__(self, spec):
            self.spec = spec
            self.closed = False
            partial.append(self)
            assert pool_owns_registration(self)
            raise ValueError("partial construction")

        def shutdown(self):
            self.closed = True

    with local_model_pool() as pool:
        with pytest.raises(ValueError, match="partial construction"):
            pool.acquire(DaemonSpec("esmfold", "unused:factory",
                         {"_initial_random_state": {"python": [1]}}), Owner, {}, "id")
        assert pool._owners == partial and pool._quarantined
    assert partial[0].closed


def test_m07_startup_fault_is_not_replaced_by_cleanup_fault(monkeypatch, tmp_path):
    from astrolabe.evaluators import af3_lifecycle as module
    owner = module.OwnedAF3.__new__(module.OwnedAF3)
    owner.process, owner.stopped, owner.state = None, False, {}
    owner.state_path = tmp_path / "lifecycle.json"
    primary = OSError("startup fault")

    def fail_start(**kwargs):
        raise primary

    def fail_cleanup(*, graceful):
        raise RuntimeError("cleanup fault")

    monkeypatch.setattr(module.tempfile, "mkdtemp", fail_start)
    owner._stop = fail_cleanup
    with pytest.raises(OSError) as caught:
        owner.start()
    assert caught.value is primary
    assert "cleanup fault" in " ".join(primary.__notes__)
    assert owner.state["status"] == "startup_failed"


@pytest.mark.parametrize("propagate", [False, True])
def test_m08_controller_callback_reentry_is_unconfirmed_without_self_wait(propagate):
    from astrolabe.evaluators.af3_pool import DaemonPool
    reentries, failures = [], []

    class Controller:
        settings = SimpleNamespace(max_concurrency=1)

        def close(self):
            try:
                pool.close()
            except RuntimeError as error:
                reentries.append(error)
                assert not pool._close_done
                assert pool._closing_thread is threading.current_thread()
                if propagate:
                    raise

    pool = DaemonPool(["/unused/socket"], controller=Controller())

    def close():
        try:
            pool.close()
        except BaseException as error:
            failures.append(error)

    # A daemon thread keeps a regression from hanging the entire test process;
    # the bounded join still fails deterministically on the old self-wait.
    thread = threading.Thread(target=close, daemon=True)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive(), "controller callback deadlocked in pool.close()"
    assert len(reentries) == 1 and "draining" in str(reentries[0])
    assert pool._close_done and pool._closing_thread is None
    if propagate:
        assert failures == [reentries[0]]
        assert pool.quarantined and pool._close_error is reentries[0]
        with pytest.raises(RuntimeError, match="cleanup failed"):
            pool.close()
    else:
        assert failures == [] and not pool.quarantined and pool._close_error is None
        pool.close()


def test_m08_concurrent_close_waits_for_drain_and_does_not_hide_failure():
    from astrolabe.evaluators.af3_pool import DaemonPool
    entered, release, second_started = threading.Event(), threading.Event(), threading.Event()
    fault = RuntimeError("controller close fault")

    class Controller:
        settings = SimpleNamespace(max_concurrency=1)

        def snapshot(self):
            return {"limit": 1}

        def close(self):
            entered.set()
            assert release.wait(5)
            raise fault

    pool = DaemonPool(["/unused/socket"], controller=Controller())

    def close_again():
        second_started.set()
        pool.close()

    with ThreadPoolExecutor(2) as workers:
        first = workers.submit(pool.close)
        try:
            assert entered.wait(5)
            second = workers.submit(close_again)
            assert second_started.wait(5)
            assert not second.done()
        finally:
            release.set()
        with pytest.raises(RuntimeError, match="controller close fault"):
            first.result(5)
        with pytest.raises(RuntimeError, match="cleanup failed"):
            second.result(5)
    assert pool.quarantined and pool.controller is not None
    snapshot = pool.describe_snapshot()
    assert snapshot["runtime_lifecycle"]["state"] == "quarantined"
    assert not snapshot["cleanup_complete"] and snapshot["cleanup_error"]
    with pytest.raises(RuntimeError, match="cleanup failed"):
        pool.close()


@pytest.mark.parametrize("which", ["esmfold2", "rfd3", "opendde"])
def test_m09_cleanup_failure_retains_last_handle_and_fences_use(which):
    from astrolabe.evaluators.wrappers.esmfold2 import ESMFold2Runner
    from astrolabe.evaluators.wrappers.opendde import OpenDDERunner
    from astrolabe.initializers.wrappers.rfd3 import RFD3Runner
    cls = {"esmfold2": ESMFold2Runner, "rfd3": RFD3Runner, "opendde": OpenDDERunner}[which]
    runner = cls.__new__(cls)
    fault = RuntimeError("cleanup fault")

    class Daemon:
        fail = True

        def shutdown(self):
            if self.fail:
                raise fault

    daemon = Daemon()
    handle = "_owned_daemon" if which == "opendde" else "_daemon"
    settings = "_owned_settings" if which == "opendde" else "_daemon_settings"
    setattr(runner, handle, daemon)
    setattr(runner, settings, {"seed": 71})
    runner._daemon_lock = threading.Lock()
    runner.execution_mode = "daemon"
    with pytest.raises(RuntimeError) as caught:
        runner.shutdown()
    assert caught.value is fault and getattr(runner, handle) is daemon
    assert getattr(runner, settings) == {"seed": 71}
    with pytest.raises(RuntimeError, match="unresolved"):
        if which == "opendde":
            runner.prepare_shared_model()
        else:
            runner._ensure_daemon({"seed": 71})
    daemon.fail = False
    runner.shutdown()
    assert getattr(runner, handle) is None and getattr(runner, settings) is None


def test_m09_rf3_wait_failure_retains_process_and_fences_replacement():
    from astrolabe.evaluators.wrappers.rf3 import RF3RunnerService
    runner = RF3RunnerService.__new__(RF3RunnerService)
    runner._lock = threading.Lock()

    class Process:
        dead = False
        terminated = False

        def poll(self):
            return 0 if self.dead else None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout):
            raise OSError("wait fault")

    proc = runner._proc = Process()
    with pytest.raises(OSError, match="wait fault"):
        runner.shutdown()
    assert proc.terminated and runner._proc is proc
    with pytest.raises(RuntimeError, match="unresolved"):
        runner._ensure_daemon()
    proc.dead = True
    runner.shutdown()
    assert runner._proc is None


def test_m09_rf3_owned_cpu_child_exit_is_confirmed():
    from astrolabe.evaluators.wrappers.rf3 import RF3RunnerService
    runner = RF3RunnerService.__new__(RF3RunnerService)
    runner._lock = threading.Lock()
    proc = runner._proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        runner.shutdown()
        assert proc.poll() is not None and runner._proc is None
        runner.shutdown()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


@pytest.mark.parametrize("failure", ["write", "already_dead"])
def test_m11_shared_exit_failure_retains_registration_and_original_owner(monkeypatch, failure):
    from astrolabe.mutators.wrappers import proton_pottsmpnn_runner as module
    runner = module.ProtonPottsMPNNRunner.__new__(module.ProtonPottsMPNNRunner)
    runner._closed, runner._started = False, True
    runner._shared_registry_key = "fault"
    runner._lock = threading.RLock()
    runner._proc = SimpleNamespace(poll=lambda: 1 if failure == "already_dead" else None,
                                   returncode=1)
    runner._write_request = lambda *args: (_ for _ in ()).throw(OSError("pipe fault"))
    registry = {"fault": ("fp", runner)}
    monkeypatch.setattr(module, "_SHARED_RUNNERS", registry)
    module._close_shared_runners_at_exit()
    assert registry["fault"][1] is runner and runner._shared_registry_key == "fault"
    assert not runner._closed
    with pytest.raises(RuntimeError, match="unresolved"):
        module.get_shared_proton_runner({}, key="fault")
    with pytest.raises(RuntimeError, match="unresolved"):
        runner._ensure_daemon()


def test_m11_shared_runner_cannot_have_two_run_owners(monkeypatch):
    from astrolabe.core import global_config
    from astrolabe.mutators.wrappers import proton_pottsmpnn_runner as module
    first, second = SimpleNamespace(add=lambda *args: None), SimpleNamespace(add=lambda *args: None)
    runner = SimpleNamespace()
    monkeypatch.setattr(global_config, "get_const", lambda name: first)
    module._register_runtime_owner(runner)
    module._register_runtime_owner(runner)
    monkeypatch.setattr(global_config, "get_const", lambda name: second)
    with pytest.raises(RuntimeError, match="another resource owner"):
        module._register_runtime_owner(runner)
    assert runner._runtime_owner is first


def test_r03_quarantine_stopped_and_warming_cannot_be_reset():
    from astrolabe.runtime.evaluation import EvaluatorInferController
    controller = EvaluatorInferController({}, sampler=lambda: None)
    controller.quarantine()
    with pytest.raises(RuntimeError, match="quarantined"):
        controller.control(enabled=True, reset=True)
    assert controller.quarantined
    other = EvaluatorInferController({}, sampler=lambda: None)
    other.warming = 1
    with pytest.raises(RuntimeError, match="drained"):
        other.control(enabled=True, reset=True)
    other.warming = 0
    other.close()
    with pytest.raises(RuntimeError, match="stopped"):
        other.control(enabled=True, reset=True)


def test_o05_signals_do_not_deadlock_on_reader_lock_or_restore_ignore():
    program = '''
import os, signal
from astrolabe import graceful_shutdown as g
signal.signal(signal.SIGTERM, signal.SIG_IGN)
g.install_handlers()
with g._LOCK:
    os.kill(os.getpid(), signal.SIGTERM)
assert g.stop_requested()
assert g.first_signal_info()[0] == signal.SIGTERM
print("first accepted", flush=True)
os.kill(os.getpid(), signal.SIGTERM)
raise AssertionError("second signal ignored")
'''
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, timeout=10)
    assert "first accepted" in result.stdout
    assert result.returncode in {-signal.SIGTERM, 128 + signal.SIGTERM}, result.stderr


@pytest.mark.parametrize("drift", [None, "spec", "artifact", "missing", "directory", "legacy"])
def test_o06_all_requires_selected_fingerprint_and_artifacts(tmp_path, drift):
    module = load_file(ROOT / "scripts/installation/env_setup.py", "remaining_env_setup")
    prefix = tmp_path / "asset"
    prefix.mkdir()
    asset = prefix / "tool"
    asset.write_text("original")
    spec = {"name": "asset", "manager": "path", "path": str(prefix), "required_files": ["tool"]}
    ctx = dict(non_interactive=True, root_prefix=tmp_path / "root", state_root=tmp_path / "state",
               micromamba="unused", substitutions={}, clone_root=tmp_path / "repos", env={},
               force=False, decisions={"asset": "m"})
    assert module._setup_one_env(spec, ctx)["status"] == "marked_done"
    ctx["decisions"] = {"asset": "u"}
    if drift == "spec":
        spec["commands"] = ["must-not-run"]
    elif drift == "artifact":
        asset.write_text("modified")
    elif drift == "missing":
        asset.unlink()
    elif drift == "directory":
        asset.unlink()
        asset.mkdir()
    elif drift == "legacy":
        state = module._load_state(ctx["state_root"], "asset")
        state.pop("completion_receipt")
        module._save_state(ctx["state_root"], "asset", state)
    if drift is None:
        assert module._setup_one_env(spec, ctx)["status"] == "already_done"
    else:
        with pytest.raises((RuntimeError, FileNotFoundError)):
            module._setup_one_env(spec, ctx)


def test_o06_skipped_commands_are_not_completed(tmp_path):
    module = load_file(ROOT / "scripts/installation/env_setup.py", "remaining_env_setup_skip")
    ctx = dict(non_interactive=True, root_prefix=tmp_path / "root", state_root=tmp_path / "state",
               micromamba="unused", substitutions={}, clone_root=tmp_path / "repos", env={},
               force=False, decisions={}, skip_commands=True)
    spec = {"name": "asset", "manager": "path", "path": str(tmp_path / "asset"),
            "commands": ["must-not-run"]}
    assert module._setup_one_env(spec, ctx)["status"] == "commands_skipped"
    state = module._load_state(ctx["state_root"], "asset")
    assert "commands" not in state["completed_steps"]
    assert "commands" not in state["fingerprints"]


def test_o07_corrupt_stamp_rehashes_valid_existing_asset_without_download(tmp_path, monkeypatch):
    from astrolabe import asset_download as module
    asset = tmp_path / "asset"
    asset.write_bytes(b"valid")
    digest = hashlib.sha256(b"valid").hexdigest()
    Path(str(asset) + ".sha256").write_text(f"v2 {digest} invalid 0 0 0 0\n")
    monkeypatch.setattr(module, "_metadata_reuse_is_safe", lambda info: True)
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network used"))
    assert module.download_url("https://unused.invalid/asset", asset, expected_sha256=digest) == asset
    assert asset.read_bytes() == b"valid"


def test_v05_actual_proton_engine_init_failure_publishes_no_ready(monkeypatch, tmp_path):
    from astrolabe.mutators.wrappers import proton_pottsmpnn_daemon as module
    before = random.getstate()
    monkeypatch.setattr(module, "_parse_args", lambda: SimpleNamespace(output_dir=str(tmp_path)))
    monkeypatch.setattr(module, "_load_runtime", lambda args: (_ for _ in ()).throw(ValueError("init fault")))
    published = []
    monkeypatch.setattr(module, "_write", lambda *args: published.append(args))
    with pytest.raises(ValueError, match="init fault"):
        module.main()
    assert published == [] and random.getstate() == before


def test_v05_ev6_live_and_default_construction_failures_do_not_publish(monkeypatch):
    namespace = {"_PREDICTOR": None, "_PREDICTOR_DEFAULT": None}
    path = VENDOR / "models/mpnn/src/mpnn/transforms/extended_vocab_v6.py"
    compile_functions(path, ["_env_prob_thr", "_predictor", "_predictor_default"], namespace)
    dependency = ModuleType("mpnn.transforms.ev6")
    calls = []

    def factory(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise ValueError("partial predictor")
        return SimpleNamespace(options=kwargs)

    dependency.EV6Predictor = factory
    monkeypatch.setitem(sys.modules, "mpnn.transforms.ev6", dependency)
    monkeypatch.setenv("EV6_HIS_PROB_THR", "0.7")
    with pytest.raises(ValueError, match="partial predictor"):
        namespace["_predictor"]()
    assert namespace["_PREDICTOR"] is None
    live = namespace["_predictor"]()
    default = namespace["_predictor_default"]()
    assert live is namespace["_predictor"]() and live is not default
    assert live.options == {"his_thr": 0.7, "acid_thr": None}
    assert default.options == {}


@pytest.mark.parametrize("failure", [None, "work", "join", "both"])
def test_v06_context_retained_until_fork_pool_exit_is_proven(failure):
    calls = []

    class Pool:
        def imap_unordered(self, callback, tasks):
            if failure in {"work", "both"}:
                raise ValueError("work fault")
            return [["result"]]

        def close(self):
            calls.append("close")

        def terminate(self):
            calls.append("terminate")

        def join(self):
            calls.append("join")
            if failure in {"join", "both"}:
                raise OSError("join fault")

    pool = Pool()
    namespace = dict(_POOL_CTX=None, _POOL_ENGINE=None, _POOL_CRITERIA=None, _POOL_SEED=0,
                     _POOL_OWNER=None, _POOL_LOCK=threading.Lock(), _pool_init=lambda: None,
                     _pool_task=lambda task: None, PHDesignSet=list,
                     mp=SimpleNamespace(get_context=lambda mode: SimpleNamespace(Pool=lambda *a, **k: pool)))
    compile_functions(VENDOR / "models/mpnn/src/mpnn/inference_engines/potts_mpnn_ph.py",
                      ["_run_tasks_parallel"], namespace, class_name="PottsMPNNPHEngine")
    run = namespace["_run_tasks_parallel"]
    engine, ctx, criteria = object(), object(), [object()]
    if failure is None:
        assert run(engine, ctx, criteria, 71, [0, 1], 2) == ["result"]
    else:
        with pytest.raises(ValueError if failure in {"work", "both"} else OSError) as caught:
            run(engine, ctx, criteria, 71, [0, 1], 2)
        if failure == "both":
            assert "join fault" in " ".join(caught.value.__notes__)
    assert calls == (["terminate", "join"] if failure in {"work", "both"} else ["close", "join"])
    if failure in {"join", "both"}:
        assert namespace["_POOL_OWNER"] is pool and namespace["_POOL_CTX"] is ctx
        assert namespace["_POOL_ENGINE"] is engine and namespace["_POOL_SEED"] == 71
        with pytest.raises(RuntimeError, match="unresolved"):
            run(engine, object(), [], 99, [0, 1], 2)
    else:
        assert namespace["_POOL_OWNER"] is namespace["_POOL_CTX"] is None
        assert namespace["_POOL_SEED"] == 0 and not namespace["_POOL_LOCK"].locked()


def test_v07_failed_unmount_keeps_table_and_other_mounts_are_attempted(tmp_path, monkeypatch):
    module = load_file(VENDOR / "src/foundry/utils/squashfs.py", "remaining_squashfs")
    manager = module.SquashfsManager
    bad, good = tmp_path / "bad", tmp_path / "good"
    bad.mkdir()
    good.mkdir()
    manager._mounts = {"bad.sqfs": str(bad), "good.sqfs": str(good)}

    def unmount(command, **kwargs):
        if command[-1] == str(bad):
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(module.subprocess, "run", unmount)
    with pytest.raises(RuntimeError, match="registration retained"):
        manager.unmount_all()
    assert manager.list_mounts() == {"bad.sqfs": str(bad)}
    assert bad.is_dir() and not good.exists()
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: None)
    manager.unmount_all()
    assert manager.list_mounts() == {}


def test_vendor_patch_chain_reconstructs_upstream_and_current_bytes(tmp_path):
    import shutil
    source = ROOT / "vendor/proton_pottsmpnn"
    manifest = json.loads((source / "SOURCE_PROVENANCE.json").read_text())
    retained = {}
    for relative in manifest["patch_chain_verified"]["changed_paths"]:
        target = tmp_path / "vendor/proton_pottsmpnn" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)
        retained[relative] = target.read_bytes()
    patches = manifest["local_patches"]
    for reverse, chain in [(True, reversed(patches)), (False, patches)]:
        for relative in chain:
            workdir = tmp_path / manifest["local_patch_workdirs"][relative]
            command = ["git", "apply", "--whitespace=nowarn", str(ROOT / relative)]
            if reverse:
                command.append("--reverse")
            result = subprocess.run(command, cwd=workdir, capture_output=True, text=True, timeout=10)
            assert result.returncode == 0, result.stdout + result.stderr
        for relative, original in retained.items():
            actual = (tmp_path / "vendor/proton_pottsmpnn" / relative).read_bytes()
            if reverse:
                upstream = next(row["upstream_git_blob"] for row in manifest["files"]
                                if row["path"] == relative)
                assert hashlib.sha1(b"blob " + str(len(actual)).encode() + b"\0" + actual).hexdigest() == upstream
            else:
                assert actual == original
