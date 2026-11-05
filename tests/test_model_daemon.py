from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import queue
import threading
import time

import numpy as np
import pytest

from astrolabe.model_daemon import (
    control_request,
    DaemonChildExited,
    DaemonRequestFailed,
    DaemonSpec,
    DaemonStartupError,
    IdentityMismatch,
    ModelDaemonError,
    OwnedModelDaemon,
    RequestOutcomeUnknown,
    model_identity,
)
from astrolabe.model_execution import model_run_scope


ADAPTER_SOURCE = r'''
from pathlib import Path
import numpy as np
import os
import time

LOAD_COUNT = 0


def _increment(path):
    path = Path(path)
    value = int(path.read_text() or "0") if path.exists() else 0
    path.write_text(str(value + 1))


class Handler:
    def __init__(self, settings):
        global LOAD_COUNT
        LOAD_COUNT += 1
        self.settings = settings

    def infer(self, payload):
        action = payload.get("action")
        if action == "provider":
            return {
                "provider_request": self.settings.get("_provider_request"),
                "pid": os.getpid(),
            }
        if action == "snapshot":
            return {
                "load_count": LOAD_COUNT,
                "pid": os.getpid(),
                "value": payload.get("value"),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            }
        if action == "arrays":
            array = np.asarray(payload["array"])
            return {
                "array": array[:, ::-1].copy(),
                "dtype": str(array.dtype),
                "shape": array.shape,
            }
        if action == "fail":
            raise ValueError("adapter failure")
        if action == "wait":
            Path(payload["started"]).write_text("started")
            release = Path(payload["release"])
            deadline = time.monotonic() + 10
            while not release.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            return {"released": release.exists()}
        if action == "slow_count":
            _increment(payload["counter"])
            time.sleep(0.5)
            return {"count": int(Path(payload["counter"]).read_text())}
        if action == "exit":
            os._exit(17)
        raise ValueError(f"unknown action: {action}")


def make(settings):
    return Handler(settings)


def broken(settings):
    raise RuntimeError("factory failed")
'''


def _spec(tmp_path: Path, **updates) -> DaemonSpec:
    adapter_dir = tmp_path / "adapters"
    adapter_dir.mkdir(exist_ok=True)
    (adapter_dir / "adapter.py").write_text(ADAPTER_SOURCE, encoding="utf-8")
    values = {
        "backend": "cpu-test",
        "factory": "adapter:make",
        "settings": {"variant": "a"},
        "python_bin": "python3.11",
        "env": {
            "PYTHONPATH": str(adapter_dir),
            "CUDA_VISIBLE_DEVICES": "",
        },
    }
    values.update(updates)
    return DaemonSpec(**values)


def _wait_for(path: Path, timeout: float = 10) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {path}")


def test_subprocess_loads_once_recovers_failure_and_shuts_down(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    spec.env["UNRELATED_CREDENTIAL"] = "do-not-persist"
    with OwnedModelDaemon(spec, startup_timeout=20) as daemon:
        first = daemon.request({"action": "snapshot", "value": 1})
        second = daemon.request({"action": "snapshot", "value": 2})
        assert first["load_count"] == second["load_count"] == 1
        assert first["pid"] == second["pid"] == daemon.pid
        assert first["cuda_visible_devices"] == ""
        array = np.arange(12, dtype=np.float32).reshape(3, 4)
        array_result = daemon.request({"action": "arrays", "array": array})
        assert array_result["array"].dtype == np.float32
        assert array_result["array"].shape == (3, 4)
        assert np.array_equal(array_result["array"], array[:, ::-1])

        with pytest.raises(DaemonRequestFailed, match="adapter failure"):
            daemon.request({"action": "fail"})
        recovered = daemon.request({"action": "snapshot", "value": 3})
        assert recovered["value"] == 3
        status = daemon.status()
        assert status["state"] == "running"
        assert status["model_load_count"] == 1
        assert status["request_count"] == 5

    assert daemon.log_path is not None and daemon.log_path.exists()
    lifecycle = json.loads((daemon.temp_dir / "lifecycle.json").read_text())
    assert lifecycle["status"] == "stopped"
    persisted = "\n".join(
        path.read_text(errors="replace")
        for path in (daemon.temp_dir / "spec.json", daemon.log_path)
    )
    assert "do-not-persist" not in persisted


def test_attach_is_identity_checked_and_never_stops_owner(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    owner = OwnedModelDaemon(spec, startup_timeout=20)
    attached = None
    try:
        owner.request({"action": "snapshot"})
        socket_path = owner.status()["socket"]
        attached = OwnedModelDaemon(spec, socket_path=socket_path, startup_timeout=20)
        assert attached.request({"action": "snapshot"})["pid"] == owner.pid
        attached.shutdown()
        assert owner.status()["state"] == "running"

        mismatched = OwnedModelDaemon(
            replace(spec, settings={"variant": "b"}),
            socket_path=socket_path,
            startup_timeout=20,
        )
        with pytest.raises(IdentityMismatch):
            mismatched.request({"action": "snapshot"})
        assert owner.status()["state"] == "running"
        lexical_python = replace(spec, python_bin="python3")
        assert model_identity(lexical_python, max_pending=4) != model_identity(spec, max_pending=4)
    finally:
        if attached is not None:
            attached.shutdown()
        owner.shutdown()


def test_attached_client_can_explicitly_stop_owner(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    owner = OwnedModelDaemon(spec, startup_timeout=20)
    try:
        owner.request({"action": "snapshot"})
        attached = OwnedModelDaemon(spec, socket_path=owner.socket_path)
        assert control_request(owner.socket_path, owner.identity)["state"] == "running"
        response = attached.request_stop()
        assert response["result"]["drained"] is True
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                owner.status()
            except DaemonChildExited:
                break
            time.sleep(0.01)
        else:
            raise AssertionError("owner child remained alive after stop acknowledgment")
        attached.shutdown()
    finally:
        owner.shutdown()


def test_control_request_checks_identity_and_can_stop(tmp_path: Path) -> None:
    owner = OwnedModelDaemon(_spec(tmp_path), startup_timeout=20)
    try:
        owner.request({"action": "snapshot"})
        with pytest.raises(IdentityMismatch):
            control_request(owner.socket_path, "not-the-owner")
        response = control_request(owner.socket_path, owner.identity, "stop")
        assert response["result"]["drained"] is True
    finally:
        owner.shutdown()


def test_status_works_while_busy_and_queue_is_bounded(tmp_path: Path) -> None:
    started = tmp_path / "started"
    release = tmp_path / "release"
    daemon = OwnedModelDaemon(_spec(tmp_path), startup_timeout=20, max_pending=1)
    outcomes: queue.Queue = queue.Queue()

    def wait_request() -> None:
        try:
            value = daemon.request(
                {"action": "wait", "started": str(started), "release": str(release)}
            )
            outcomes.put(("result", value))
        except BaseException as error:
            outcomes.put(("error", repr(error)))

    first = threading.Thread(target=wait_request)
    second = threading.Thread(target=wait_request)
    try:
        first.start()
        _wait_for(started)
        second.start()
        time.sleep(0.1)
        with pytest.raises(DaemonRequestFailed, match="pending queue is full"):
            daemon.request(
                {"action": "wait", "started": str(started), "release": str(release)}
            )
        assert daemon.status()["state"] == "running"
        release.write_text("go")
        first.join(timeout=10)
        second.join(timeout=10)
        assert not first.is_alive() and not second.is_alive()
        observed = [outcomes.get_nowait() for _ in range(outcomes.qsize())]
        log = daemon.log_path.read_text(errors="replace") if daemon.log_path else ""
        lifecycle = (
            json.loads((daemon.temp_dir / "lifecycle.json").read_text())
            if daemon.temp_dir is not None
            else {}
        )
        assert observed == [("result", {"released": True}), ("result", {"released": True})], (
            observed,
            log,
            {"test_pid": os.getpid(), "daemon_pid": daemon.pid, "lifecycle": lifecycle},
        )
        assert daemon.request({"action": "snapshot"})["load_count"] == 1
    finally:
        release.write_text("go")
        daemon.shutdown()


def test_timeout_is_outcome_unknown_without_replay(tmp_path: Path) -> None:
    counter = tmp_path / "counter"
    daemon = OwnedModelDaemon(
        _spec(tmp_path),
        startup_timeout=20,
        request_timeout=0.15,
    )
    try:
        with pytest.raises(RequestOutcomeUnknown, match="outcome unknown") as caught:
            daemon.request({"action": "slow_count", "counter": str(counter)})
        assert caught.value.request_id
        assert caught.value.generation == daemon.generation
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and not counter.exists():
            time.sleep(0.01)
        time.sleep(0.6)
        assert counter.read_text() == "1"
        daemon.request_timeout = 2
        snapshot = daemon.request({"action": "snapshot"})
        assert snapshot["load_count"] == 1
        assert counter.read_text() == "1"
    finally:
        daemon.shutdown()


def test_shutdown_stops_admission_and_drains_active_request(tmp_path: Path) -> None:
    started = tmp_path / "drain-started"
    release = tmp_path / "drain-release"
    daemon = OwnedModelDaemon(_spec(tmp_path), startup_timeout=20)
    outcome: queue.Queue = queue.Queue()

    def request() -> None:
        try:
            outcome.put(daemon.request({"action": "wait", "started": str(started), "release": str(release)}))
        except BaseException as error:
            outcome.put(error)

    request_thread = threading.Thread(target=request)
    request_thread.start()
    _wait_for(started)
    shutdown_thread = threading.Thread(target=daemon.shutdown)
    shutdown_thread.start()
    time.sleep(0.15)
    assert shutdown_thread.is_alive()
    release.write_text("go")
    request_thread.join(timeout=10)
    shutdown_thread.join(timeout=10)
    assert not request_thread.is_alive()
    assert not shutdown_thread.is_alive()
    assert outcome.get_nowait() == {"released": True}


def test_startup_failure_is_explicit(tmp_path: Path) -> None:
    spec = _spec(tmp_path, factory="adapter:broken")
    daemon = OwnedModelDaemon(spec, startup_timeout=5)
    with pytest.raises(DaemonStartupError, match="log:"):
        daemon.request({"action": "snapshot"})
    assert daemon.log_path is not None and daemon.log_path.exists()
    daemon.shutdown()


def test_child_death_is_explicit(tmp_path: Path) -> None:
    daemon = OwnedModelDaemon(_spec(tmp_path), startup_timeout=20)
    try:
        with pytest.raises(DaemonChildExited):
            daemon.request({"action": "exit"})
    finally:
        daemon.shutdown()


def test_run_scope_stops_owned_children_and_detaches_clients(tmp_path: Path) -> None:
    outer = OwnedModelDaemon(_spec(tmp_path), startup_timeout=20)
    try:
        outer.request({"action": "snapshot"})
        with model_run_scope():
            inner = OwnedModelDaemon(_spec(tmp_path), startup_timeout=20)
            inner.request({"action": "snapshot"})
            attached = OwnedModelDaemon(
                outer.spec,
                socket_path=outer.socket_path,
                startup_timeout=20,
            )
            attached.request({"action": "snapshot"})
        with pytest.raises(ModelDaemonError, match="closed"):
            inner.status()
        assert outer.status()["state"] == "running"
        attached.shutdown()
        assert outer.status()["state"] == "running"
    finally:
        outer.shutdown()


def test_native_provider_is_prepared_and_activated_before_factory(tmp_path: Path) -> None:
    spec = replace(
        _spec(tmp_path),
        backend="esmfold",
        provider={"provider": "native"},
    )
    with OwnedModelDaemon(spec, startup_timeout=20) as daemon:
        result = daemon.request({"action": "provider"})
        assert result["provider_request"]["provider"] == "native"
        assert result["provider_request"]["mode"] == "off"
        assert daemon.spec.provider == {"provider": "native", "mode": "off",
                                        "model_variant": None, "kit_root": None}
        log = daemon.log_path.read_text(encoding="utf-8")
        assert '"event": "provider_activated"' in log
        assert '"model_identity": "astrolabe-native:esmfold"' in log
