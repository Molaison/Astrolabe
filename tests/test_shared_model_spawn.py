"""Real CPU daemon reuse across simultaneously live spawn consumers."""

from __future__ import annotations

import json
import multiprocessing
import os
from pathlib import Path
import stat
import sys
import traceback

import pytest

from astrolabe.model_daemon import DaemonSpec, OwnedModelDaemon, model_identity
from astrolabe.model_execution import NativeModelClient


ADAPTER = r'''
import json
import os
import torch
from astrolabe.model_execution import kernel_handler

LOAD_COUNT = 0


def factory(settings):
    def loader(raw):
        global LOAD_COUNT
        LOAD_COUNT += 1
        counter = torch.tensor(LOAD_COUNT, device="cpu")
        event = {"owner_pid": os.getpid(), "load_count": LOAD_COUNT,
                 "device": str(counter.device)}
        fd = os.open(raw["load_events"], os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, (json.dumps(event) + "\n").encode())
            os.fsync(fd)
        finally:
            os.close(fd)

        def infer(payload):
            value = torch.tensor(payload["value"], device="cpu") + counter
            return dict(event, value=value.item(), worker_pid=payload["worker_pid"])

        return infer, dict(event, actual_load_count=LOAD_COUNT)

    return kernel_handler(settings, loader, torch_device="cpu",
                          lifecycle={"resource": "cpu", "device": "cpu"})
'''


def _client(spec: DaemonSpec) -> NativeModelClient:
    # The frame bound participates in the exact identity, including on attach.
    return NativeModelClient(
        spec.backend, spec.factory, spec.settings,
        python_bin=spec.python_bin, cwd=spec.cwd, env=spec.env,
        max_frame_bytes=spec.max_frame_bytes,
        startup_timeout=60, request_timeout=60, shutdown_timeout=10,
    )


def _spawn_consumer(spec, barrier, connection):
    client = None
    try:
        client = _client(spec)
        assert isinstance(client.owner, OwnedModelDaemon)
        assert client.owner.attached and client.owner._proc is None
        connection.send({
            "phase": "attached", "worker_pid": os.getpid(),
            "owner_pid": client.pid, "identity": client.owner.identity,
            "metadata": client.metadata,
        })
        barrier.wait(timeout=60)
        responses = []
        for value in (10, 20):
            result = client.request({"value": value, "worker_pid": os.getpid()})
            responses.append({"result": result, "receipt": client.last_runtime_receipt})
        connection.send({"phase": "results", "responses": responses})
        # Remain alive until the parent has observed both workers' responses.
        barrier.wait(timeout=60)
        client.shutdown()
        client = None
        connection.send({"phase": "shutdown", "worker_pid": os.getpid()})
    except BaseException:
        connection.send({"phase": "error", "traceback": traceback.format_exc()})
        barrier.abort()
        raise
    finally:
        if client is not None:
            client.shutdown()
        connection.close()


def _receive(connection, phase):
    assert connection.poll(60), f"spawn consumer timed out waiting for {phase}"
    message = connection.recv()
    assert message["phase"] == phase, message
    return message


def test_spawn_consumers_share_one_real_cpu_load(tmp_path: Path, monkeypatch) -> None:
    for name in (
        "ASTROLABE_RESIDENT_MODEL_REGISTRY", "ASTROLABE_RESOURCE_SOCKET",
        "ASTROLABE_RESOURCE_PROFILES", "ASTROLABE_MODEL_OWNER",
        "ASTROLABE_MODEL_PROVIDERS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "shared_spawn_adapter.py").write_text(ADAPTER, encoding="utf-8")
    load_events = tmp_path / "loader.jsonl"
    pythonpath = os.pathsep.join(filter(None, (
        str(adapter_dir), os.environ.get("PYTHONPATH"),
    )))
    spec = DaemonSpec(
        backend="shared-spawn-cpu", factory="shared_spawn_adapter:factory",
        settings={"load_events": str(load_events)}, python_bin=sys.executable,
        env={"PYTHONPATH": pythonpath, "CUDA_VISIBLE_DEVICES": ""},
        max_frame_bytes=32 * 1024 * 1024,
    )
    identity = model_identity(spec)
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(3)
    workers = []
    connections = []
    parent = None
    try:
        parent = _client(spec)
        assert isinstance(parent.owner, OwnedModelDaemon)
        assert not parent.owner.attached
        daemon = parent.owner._proc
        assert daemon is not None and daemon.poll() is None
        owner_pid = daemon.pid
        assert owner_pid != os.getpid()
        assert parent.pid == owner_pid
        assert parent.owner.identity == identity
        lifecycle = parent.metadata["runtime_lifecycle"]
        model_id = lifecycle["model_id"]
        assert lifecycle["identity"] == identity
        assert lifecycle["pid"] == owner_pid
        assert lifecycle["loader_count"] == parent.metadata["actual_load_count"] == 1
        socket = parent.owner.socket_path
        assert socket is not None and stat.S_ISSOCK(socket.stat().st_mode)
        assert stat.S_IMODE(socket.stat().st_mode) == 0o600
        assert stat.S_IMODE(socket.parent.stat().st_mode) == 0o700
        event = {"owner_pid": owner_pid, "load_count": 1, "device": "cpu"}
        assert [json.loads(line) for line in load_events.read_text().splitlines()] == [event]

        registry = tmp_path / "registry.json"
        registry.write_text(json.dumps({
            "schema": "astrolabe.resident-model-registry/v1",
            "broker_socket": None,
            "models": {identity: {"backend": spec.backend, "socket": str(socket)}},
        }), encoding="utf-8")
        monkeypatch.setenv("ASTROLABE_RESIDENT_MODEL_REGISTRY", str(registry))
        for _ in range(2):
            receiving, sending = context.Pipe(duplex=False)
            connections.append(receiving)
            worker = context.Process(target=_spawn_consumer, args=(spec, barrier, sending))
            workers.append(worker)
            try:
                worker.start()
            finally:
                sending.close()
        attached = [_receive(connection, "attached") for connection in connections]
        worker_pids = [message["worker_pid"] for message in attached]
        assert len(set(worker_pids + [os.getpid(), owner_pid])) == 4
        assert worker_pids == [worker.pid for worker in workers]
        assert all(worker.is_alive() for worker in workers)
        for message in attached:
            assert message["owner_pid"] == owner_pid
            assert message["identity"] == identity
            metadata = message["metadata"]
            assert metadata["actual_load_count"] == 1
            shared = metadata["runtime_lifecycle"]
            assert shared["model_id"] == model_id
            assert shared["pid"] == owner_pid
            assert shared["loader_count"] == 1
            assert shared["resources"] == {"broker_socket": None}
        barrier.wait(timeout=60)
        results = [_receive(connection, "results") for connection in connections]
        assert all(worker.is_alive() for worker in workers)
        for worker_pid, message in zip(worker_pids, results):
            assert len(message["responses"]) == 2
            for value, response in zip((10, 20), message["responses"]):
                assert response["result"] == dict(event, value=value + 1, worker_pid=worker_pid)
                receipt = response["receipt"]
                assert receipt["model_id"] == model_id
                assert receipt["lifecycle"]["pid"] == owner_pid
                assert receipt["lifecycle"]["identity"] == identity
                assert receipt["lifecycle"]["loader_count"] == 1
        barrier.wait(timeout=60)
        for worker, connection in zip(workers, connections):
            assert _receive(connection, "shutdown")["worker_pid"] == worker.pid
            worker.join(timeout=60)
            assert worker.exitcode == 0

        assert daemon.poll() is None
        assert parent.request({"value": 30, "worker_pid": os.getpid()}) == dict(
            event, value=31, worker_pid=os.getpid(),
        )
        assert parent.last_runtime_receipt["model_id"] == model_id
        parent.shutdown()
        assert daemon.poll() == 0
        with pytest.raises(ProcessLookupError):
            os.kill(owner_pid, 0)
        # This independent file catches replacement loads in any daemon process;
        # runtime_lifecycle.loader_count alone is only a per-handler declaration.
        assert [json.loads(line) for line in load_events.read_text().splitlines()] == [event]
        print(json.dumps({"worker_pids": worker_pids, "owner_pid": owner_pid,
                          "model_id": model_id, "actual_loader_events": 1,
                          "requests_per_worker": 2, "parent_after_worker_shutdown": 31,
                          "daemon_exit_code": daemon.returncode}, sort_keys=True))
    finally:
        barrier.abort()
        for worker in workers:
            if worker.pid is not None:
                if worker.is_alive():
                    worker.terminate()
                worker.join(timeout=5)
                if worker.is_alive():
                    worker.kill()
                    worker.join(timeout=5)
        for connection in connections:
            connection.close()
        if parent is not None:
            parent.shutdown()
