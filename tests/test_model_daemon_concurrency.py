"""CPU-only transport contracts; no scientific model or RNG qualification."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import os
import pickle
from pathlib import Path
import socket
import sys
import threading
import time
import weakref

import pytest

from astrolabe.model_daemon import (
    MAX_FRAME_BYTES,
    PROTOCOL,
    DaemonRequestFailed,
    DaemonDisconnected,
    DaemonSpec,
    IdentityMismatch,
    OwnedModelDaemon,
    RequestOutcomeUnknown,
    FrameTooLarge,
    _Server,
    _frame,
    _read_frame,
    _send_frame,
    model_identity,
    control_request,
)


ADAPTER = '''
import json
import os
from pathlib import Path
import threading
import time

class Handler:
    def __init__(self, settings):
        self.directory = Path(settings["directory"])
        self.barrier = threading.Barrier(2)
        if "declaration" in settings:
            self.max_concurrency = settings["declaration"]

    def __call__(self, payload):
        token = payload["token"]
        def marker(name):
            return self.directory / (name + "-" + token)
        with marker("calls").open("a") as handle:
            handle.write("call\\n")
        marker("entered").touch()
        try:
            if payload.get("barrier"):
                self.barrier.wait(timeout=10)
                marker("crossed").touch()
            if payload.get("wait", True):
                deadline = time.monotonic() + 15
                while not marker("release").exists():
                    if (self.directory / "release-all").exists():
                        break
                    if time.monotonic() >= deadline:
                        raise TimeoutError("test gate was not released")
                    time.sleep(0.005)
            if payload.get("fail"):
                raise ValueError("failure-" + token)
            return {"token": token, "pid": os.getpid(), "values": payload.get("values")}
        finally:
            marker("exited").touch()

def make(settings):
    with Path(settings["factory_events"]).open("a") as handle:
        handle.write(json.dumps({"event": "factory_initialized", "pid": os.getpid()}) + "\\n")
    return Handler(settings)
'''

_UNDECLARED = object()


def _wait(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("timed out waiting for transport test condition")


def _marker(directory, kind, token):
    return directory / f"{kind}-{token}"


@contextmanager
def _owner(tmp_path, declaration=_UNDECLARED, *, max_pending=4, pipeline="0"):
    (tmp_path / "transport_adapter.py").write_text(ADAPTER)
    settings = {"directory": str(tmp_path), "factory_events": str(tmp_path / "factory.jsonl")}
    if declaration is not _UNDECLARED:
        settings["declaration"] = declaration
    spec = DaemonSpec(
        backend="cpu-transport-test",
        factory="transport_adapter:make",
        settings=settings,
        python_bin=sys.executable,
        env={"PYTHONPATH": str(tmp_path), "CUDA_VISIBLE_DEVICES": "",
             "ASTROLABE_RESOURCE_SOCKET": "", "ASTROLABE_MODEL_OWNER": "daemon"},
    )
    if pipeline is not None:
        spec.env["ASTROLABE_MODEL_PIPELINE"] = pipeline
    daemon = OwnedModelDaemon(spec, startup_timeout=20, shutdown_timeout=20,
                              max_pending=max_pending)
    pool = ThreadPoolExecutor(max_workers=8)
    try:
        daemon.status()
        yield daemon, pool
    finally:
        (tmp_path / "release-all").touch()
        pool.shutdown(wait=True)
        daemon.shutdown()


def _submit(pool, daemon, token, *, request_id=None, timeout=10, **payload):
    return pool.submit(daemon._exchange,
                       {"type": "infer", "request_id": request_id or token,
                        "payload": {"token": token, **payload}}, timeout=timeout)


def test_declared_two_requests_cross_barrier_in_one_factory_and_pid(tmp_path):
    with _owner(tmp_path, 2) as (daemon, pool):
        first = _submit(pool, daemon, "one", barrier=True, values=[1, 2])
        second = _submit(pool, daemon, "two", barrier=True, values=[3, 4])
        _wait(lambda: all(_marker(tmp_path, "crossed", token).exists()
                          for token in ("one", "two")))
        status = daemon.status()
        assert status["worker_count"] == status["peak_active_requests"] == 2
        assert set(status["active_requests"]) == {"one", "two"}
        assert status["active_request"] in status["active_requests"]
        assert status["request_count"] == 2
        assert status["completed_count"] == status["failed_count"] == 0
        for token in ("one", "two"):
            _marker(tmp_path, "release", token).touch()
        responses = [first.result(10), second.result(10)]
        assert [r["result"]["values"] for r in responses] == [[1, 2], [3, 4]]
        assert {r["pid"] for r in responses} == {daemon.pid}
        assert {r["result"]["pid"] for r in responses} == {daemon.pid}
        assert [r["request_id"] for r in responses] == ["one", "two"]
        assert all(r["generation"] == daemon.generation for r in responses)
        events = [json.loads(line) for line in (tmp_path / "factory.jsonl").read_text().splitlines()]
        assert events == [{"event": "factory_initialized", "pid": daemon.pid}]
        status = daemon.status()
        assert status["completed_count"] == 2 and status["failed_count"] == 0
        assert status["active_requests"] == [] and status["active_request"] is None
        assert status["model_load_count_scope"] == "factory_initializations"
    events = [json.loads(line) for line in daemon.log_path.read_text().splitlines()
              if line.startswith("{")]
    assert sum(event["event"] == "model_load_start" for event in events) == 1
    assert sum(event["event"] == "model_load_end" for event in events) == 1


@pytest.mark.parametrize("pipeline", ["0", "1"])
def test_undeclared_handler_remains_serial(tmp_path, pipeline):
    with _owner(tmp_path, pipeline=pipeline) as (daemon, pool):
        first = _submit(pool, daemon, "one")
        _wait(lambda: _marker(tmp_path, "entered", "one").exists())
        second = _submit(pool, daemon, "two")
        _wait(lambda: daemon.status()["queue_depth"] == 1)
        status = daemon.status()
        assert status["worker_count"] == status["peak_active_requests"] == 1
        assert status["active_requests"] == ["one"]
        assert not _marker(tmp_path, "entered", "two").exists()
        _marker(tmp_path, "release", "one").touch()
        assert first.result(10)["result"]["token"] == "one"
        _wait(lambda: _marker(tmp_path, "entered", "two").exists())
        _marker(tmp_path, "release", "two").touch()
        assert second.result(10)["result"]["token"] == "two"
        assert daemon.status()["peak_active_requests"] == 1


def test_parallel_workers_keep_pending_queue_bounded(tmp_path):
    with _owner(tmp_path, 100, max_pending=2) as (daemon, pool):
        active = [_submit(pool, daemon, token, barrier=True) for token in ("one", "two")]
        _wait(lambda: all(_marker(tmp_path, "crossed", token).exists()
                          for token in ("one", "two")))
        pending = [_submit(pool, daemon, token, wait=False) for token in ("three", "four")]
        _wait(lambda: daemon.status()["queue_depth"] == 2)
        assert daemon.status()["worker_count"] == 2
        with pytest.raises(DaemonRequestFailed, match="pending queue is full"):
            daemon.request({"token": "rejected", "wait": False})
        assert not _marker(tmp_path, "entered", "rejected").exists()
        assert daemon.status()["queue_depth"] == 2
        (tmp_path / "release-all").touch()
        assert {f.result(10)["result"]["token"] for f in active + pending} == {
            "one", "two", "three", "four"}
        assert daemon.status()["peak_active_requests"] == 2


def test_failure_and_colliding_request_ids_are_isolated_by_generation(tmp_path):
    with _owner(tmp_path, 2) as (daemon, pool):
        attached = OwnedModelDaemon(daemon.spec, socket_path=daemon.socket_path,
                                    max_pending=daemon.max_pending)
        try:
            good = _submit(pool, daemon, "good", request_id="same-id", barrier=True)
            bad = _submit(pool, attached, "bad", request_id="same-id", barrier=True, fail=True)
            _wait(lambda: all(_marker(tmp_path, "crossed", token).exists()
                              for token in ("good", "bad")))
            assert daemon.generation != attached.generation
            assert daemon.status()["active_requests"] == ["same-id", "same-id"]
            _marker(tmp_path, "release", "bad").touch()
            with pytest.raises(DaemonRequestFailed, match="failure-bad") as error:
                bad.result(10)
            assert error.value.remote_type == "ValueError"
            assert not good.done()
            status = daemon.status()
            assert status["active_requests"] == ["same-id"]
            assert status["failed_count"] == 1 and status["completed_count"] == 0
            _marker(tmp_path, "release", "good").touch()
            response = good.result(10)
            assert response["request_id"] == "same-id"
            assert response["generation"] == daemon.generation
            assert response["result"]["token"] == "good"
            for token in ("good", "bad"):
                assert _marker(tmp_path, "calls", token).read_text().splitlines() == ["call"]
            assert daemon.request({"token": "after", "wait": False})["token"] == "after"
        finally:
            attached.shutdown()


def test_parallel_timeout_is_unknown_and_never_replayed(tmp_path):
    with _owner(tmp_path, 2) as (daemon, pool):
        unknown = _submit(pool, daemon, "unknown", timeout=0.5)
        other = _submit(pool, daemon, "other")
        _wait(lambda: len(daemon.status()["active_requests"]) == 2)
        with pytest.raises(RequestOutcomeUnknown) as error:
            unknown.result(10)
        assert error.value.request_id == "unknown"
        assert error.value.generation == daemon.generation
        _marker(tmp_path, "release", "unknown").touch()
        _wait(lambda: daemon.status()["completed_count"] == 1)
        assert not other.done()
        assert daemon.status()["active_requests"] == ["other"]
        _marker(tmp_path, "release", "other").touch()
        assert other.result(10)["result"]["token"] == "other"
        assert _marker(tmp_path, "calls", "unknown").read_text().splitlines() == ["call"]
        assert daemon.status()["completed_count"] == 2


@pytest.mark.parametrize("declaration", [True, False, 0, -1, 2.0, "2", None])
def test_concurrency_declaration_requires_strict_positive_integer(tmp_path, declaration):
    def handler(payload):
        return payload
    handler.max_concurrency = declaration
    with pytest.raises(ValueError, match="max_concurrency must be a positive integer"):
        _Server(handler, socket_path=tmp_path / "unused.sock", identity="test",
                backend="cpu-test", max_pending=4, max_frame_bytes=MAX_FRAME_BYTES,
                parent_pid=None)


def test_pipeline_identity_uses_effective_environment_without_mutation(monkeypatch):
    spec = DaemonSpec(backend="cpu-test", factory="json:dumps", settings={"keep": [1, 2]})
    monkeypatch.delenv("ASTROLABE_MODEL_PIPELINE", raising=False)
    default = model_identity(spec)
    monkeypatch.setenv("ASTROLABE_MODEL_PIPELINE", "0")
    assert model_identity(spec) == default
    monkeypatch.setenv("ASTROLABE_MODEL_PIPELINE", "1")
    assert model_identity(spec) != default
    assert spec.settings == {"keep": [1, 2]} and spec.env is None
    assert os.environ["ASTROLABE_MODEL_PIPELINE"] == "1"
    for override in ("0", "1"):
        spec.env = {"ASTROLABE_MODEL_PIPELINE": override}
        overridden = model_identity(spec)
        monkeypatch.setenv("ASTROLABE_MODEL_PIPELINE", "invalid-but-shadowed")
        assert model_identity(spec) == overridden
        assert spec.env == {"ASTROLABE_MODEL_PIPELINE": override}
        assert os.environ["ASTROLABE_MODEL_PIPELINE"] == "invalid-but-shadowed"


@pytest.mark.parametrize("flag", ["", "2", "01", "true", " 1", 1, True, None])
def test_pipeline_identity_rejects_invalid_spec_flag(flag):
    spec = DaemonSpec(backend="cpu-test", factory="json:dumps", settings={},
                      env={"ASTROLABE_MODEL_PIPELINE": flag})
    with pytest.raises(ValueError, match="ASTROLABE_MODEL_PIPELINE must be '0' or '1'"):
        model_identity(spec)


def test_pipeline_identity_rejects_invalid_inherited_flag(monkeypatch):
    monkeypatch.setenv("ASTROLABE_MODEL_PIPELINE", "invalid")
    spec = DaemonSpec(backend="cpu-test", factory="json:dumps", settings={})
    with pytest.raises(ValueError, match="ASTROLABE_MODEL_PIPELINE"):
        model_identity(spec)


def test_inherited_pipeline_flag_cannot_attach_to_serial_owner(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTROLABE_MODEL_PIPELINE", "0")
    with _owner(tmp_path, pipeline=None) as (daemon, _):
        monkeypatch.setenv("ASTROLABE_MODEL_PIPELINE", "1")
        attached = OwnedModelDaemon(daemon.spec, socket_path=daemon.socket_path)
        try:
            assert attached.identity != daemon.identity
            with pytest.raises(IdentityMismatch):
                attached.status()
            assert daemon.status()["worker_count"] == 1
        finally:
            attached.shutdown()


def test_drain_waits_for_all_request_cleanup_and_repeated_stops_are_safe(tmp_path, monkeypatch):
    barrier = threading.Barrier(2)
    cleanup_entered = [threading.Event(), threading.Event()]
    cleanup_release = [threading.Event(), threading.Event()]
    exited = [threading.Event(), threading.Event()]
    stop_events = []
    monkeypatch.setattr("astrolabe.model_daemon._event",
                        lambda event, **fields: stop_events.append(event))

    def handler(payload):
        index = payload["index"]
        try:
            barrier.wait(timeout=10)
            return index
        finally:
            cleanup_entered[index].set()
            if not cleanup_release[index].wait(15):
                raise TimeoutError("test cleanup was not released")
            exited[index].set()

    handler.max_concurrency = 2
    server = _Server(handler, socket_path=tmp_path / "unused.sock", identity="test",
                     backend="cpu-test", max_pending=2, max_frame_bytes=MAX_FRAME_BYTES,
                     parent_pid=None)
    clients = []
    pool = ThreadPoolExecutor(max_workers=6)

    def send(kind, request_id, **payload):
        client, connection = socket.socketpair()
        client.settimeout(10)
        clients.append(client)
        future = pool.submit(server._handle, connection)
        _send_frame(client, {"protocol": PROTOCOL, "identity": "test", "type": kind,
                             "request_id": request_id, "generation": request_id,
                             "payload": payload})
        return client, future

    for worker in server.workers:
        worker.start()
    try:
        first, _ = send("infer", "one", index=0)
        second, _ = send("infer", "two", index=1)
        assert all(event.wait(10) for event in cleanup_entered)
        stop_one, stopping_one = send("stop", "stop-one")
        stop_two, stopping_two = send("stop", "stop-two")
        _wait(lambda: stop_events.count("shutdown_received") == 2)
        assert server._status()["state"] == "draining"
        rejected, _ = send("infer", "late", index=0)
        assert "stopping" in _read_frame(rejected)["message"]
        cleanup_release[0].set()
        assert _read_frame(first)["result"] == 0
        assert exited[0].is_set() and not exited[1].is_set()
        assert not stopping_one.done() and not stopping_two.done()
        status = server._status()
        assert status["active_requests"] == ["two"]
        assert status["completed_count"] == 1
        cleanup_release[1].set()
        assert _read_frame(second)["result"] == 1
        for client, future, request_id in ((stop_one, stopping_one, "stop-one"),
                                            (stop_two, stopping_two, "stop-two")):
            response = _read_frame(client)
            assert response["request_id"] == response["generation"] == request_id
            assert response["result"] == {"drained": True, "completed_count": 2,
                                           "failed_count": 0}
            future.result(10)
        assert all(event.is_set() for event in exited)
        assert all(not worker.is_alive() for worker in server.workers)
        repeated, future = send("stop", "again")
        assert _read_frame(repeated)["result"]["drained"] is True
        future.result(10)
        assert server.tasks.empty() and server.tasks.unfinished_tasks == 0
        assert server.shutdown_complete.is_set()
    finally:
        for event in cleanup_release:
            event.set()
        if not server.workers_stopped:
            client, future = send("stop", "cleanup")
            _read_frame(client)
            future.result(10)
        pool.shutdown(wait=True)
        for client in clients:
            client.close()


@contextmanager
def _socket_transport(tmp_path, handler, *, max_pending=1, io_timeout=2,
                      max_frame_bytes=MAX_FRAME_BYTES):
    """Exercise the transport with small socket buffers and explicit cleanup."""
    server = _Server(handler, socket_path=tmp_path / "toy.sock", identity="toy",
                     backend="cpu-toy", max_pending=max_pending,
                     max_frame_bytes=max_frame_bytes, parent_pid=None,
                     io_timeout=io_timeout)
    clients = []
    pool = ThreadPoolExecutor(max_workers=server.max_connections + 2)

    def connect(kind=None, request_id="toy-request", **payload):
        client, connection = socket.socketpair()
        client.settimeout(5)
        connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024)
        clients.append(client)
        future = pool.submit(server._handle, connection)
        if kind is not None:
            _send_frame(client, dict(protocol=PROTOCOL, identity="toy", type=kind,
                                    request_id=request_id, generation=request_id,
                                    payload=payload))
        return client, future

    for worker in server.workers:
        worker.start()
    try:
        yield server, connect
    finally:
        for client in clients:
            client.close()
        if not server.workers_stopped:
            # Input and output deadlines reclaim any remaining connection slots.
            _wait(lambda: server._status()["connection_count"] == 0)
            client, future = connect("stop", "cleanup")
            try:
                _read_frame(client)
            except (DaemonDisconnected, ConnectionError):
                # Deliberately tiny frame limits can preclude the stop receipt.
                pass
            future.result(10)
        pool.shutdown(wait=True)
        for client in clients:
            client.close()
        assert server.tasks.unfinished_tasks == 0
        assert server._status()["inflight_requests"] == 0
        assert server._status()["connection_count"] == 0
        assert server._status()["output_responses"] == 0
        assert server._status()["output_frame_bytes"] == 0
        assert all(not worker.is_alive() for worker in server.workers)


@pytest.mark.parametrize("finish", ["read", "disconnect", "deadline"])
def test_slow_512k_output_keeps_active_queue_and_connection_credits(tmp_path, finish):
    calls = []

    def handler(payload):
        calls.append(payload["index"])
        return b"x" * (512 * 1024) if payload["index"] == 0 else payload["index"]

    with _socket_transport(tmp_path, handler) as (server, connect):
        first, first_done = connect("infer", "first", index=0)
        _wait(lambda: calls == [0] and server._status()["active_requests"] == ["first"])
        second, second_done = connect("infer", "second", index=1)
        _wait(lambda: server._status()["queue_depth"] == 1)
        for index in (2, 3):
            rejected, future = connect("infer", f"rejected-{index}", index=index)
            assert "pending queue is full" in _read_frame(rejected)["message"]
            future.result(5)
        status_client, future = connect("status", "status")
        status = _read_frame(status_client)
        future.result(5)
        assert status["active_requests"] == ["first"]
        assert status["completed_count"] == 0
        assert status["queue_depth"] == 1
        assert status["inflight_requests"] == status["max_inflight_requests"] == 2
        assert status["max_connections"] == 4
        assert status["peak_connections"] <= 4
        assert status["output_responses"] == 1
        assert 512 * 1024 < status["output_frame_bytes"] <= MAX_FRAME_BYTES + 8
        assert status["peak_output_frame_bytes"] <= status["max_output_frame_bytes"]
        assert status["max_output_responses"] == 4
        assert status["max_output_frame_bytes"] == 4 * (MAX_FRAME_BYTES + 8)
        assert status["max_serialized_output_bytes"] == 2 * 4 * (MAX_FRAME_BYTES + 8)
        assert calls == [0] and not first_done.done() and not second_done.done()
        if finish == "read":
            response = _read_frame(first)
            assert response["result"] == b"x" * (512 * 1024)
            assert response["request_id"] == response["generation"] == "first"
        elif finish == "disconnect":
            first.close()
        # In the deadline case the client never reads or closes the response.
        first_done.result(5)
        assert _read_frame(second)["result"] == 1
        second_done.result(5)
        assert calls == [0, 1]  # No replay after failed/partial delivery.
        assert server._status()["completed_count"] == 2
        assert server._status()["active_requests"] == []
        assert server._status()["inflight_requests"] == 0
        assert server._status()["output_frame_bytes"] == 0
        assert server.tasks.unfinished_tasks == 0


def test_stop_waits_for_unread_output_deadline_and_wakes_partial_input(tmp_path):
    calls = []

    def handler(payload):
        calls.append(payload["index"])
        return b"x" * (512 * 1024)

    with _socket_transport(tmp_path, handler, io_timeout=1) as (server, connect):
        unread, inference = connect("infer", "unread", index=0)
        _wait(lambda: calls == [0])
        partial, reading = connect()
        partial.sendall((1000).to_bytes(8, "big") + b"partial")
        _wait(lambda: server._status()["connection_count"] == 2)
        stopping, stopped = connect("stop", "stop")
        _wait(lambda: server._status()["state"] == "draining")
        assert not stopped.done() and not inference.done()
        reading.result(5)
        receipt = _read_frame(stopping)
        stopped.result(5)
        inference.result(5)
        assert receipt["result"]["drained"] is True
        assert receipt["result"]["completed_count"] == 1
        assert server.shutdown_complete.is_set()
        assert calls == [0] and server.tasks.unfinished_tasks == 0


@pytest.mark.parametrize("prefix", [b"", b"\x00", (1000).to_bytes(8, "big") + b"body"])
def test_partial_frames_release_all_connection_slots_on_absolute_deadline(tmp_path, prefix):
    with _socket_transport(tmp_path, lambda payload: payload, io_timeout=0.3) as (server, connect):
        partials = [connect() for _ in range(server.max_connections)]
        for client, _ in partials:
            if prefix:
                client.sendall(prefix)
        _wait(lambda: server._status()["connection_count"] == server.max_connections)
        excess, closed = connect()
        closed.result(5)
        assert excess.recv(1) == b""
        assert server._status()["peak_connections"] == server.max_connections
        assert server._status()["request_count"] == 0
        # Keep providing bytes faster than the socket idle timeout: the total
        # read deadline still closes every partial frame.
        for _ in range(8):
            for client, _ in partials:
                try:
                    client.sendall(b"\0")
                except OSError:
                    pass
            time.sleep(0.06)
        for _, future in partials:
            future.result(5)
        assert server._status()["connection_count"] == 0
        status_client, future = connect("status")
        assert _read_frame(status_client)["request_count"] == 0
        future.result(5)


def test_listener_uses_fixed_pool_even_when_all_input_slots_are_full(tmp_path):
    server = _Server(lambda payload: payload, socket_path=tmp_path / "listener.sock",
                     identity="toy", backend="cpu-toy", max_pending=1,
                     max_frame_bytes=MAX_FRAME_BYTES, parent_pid=None, io_timeout=0.5)
    owner = threading.Thread(target=server.run)
    clients = []
    owner.start()
    try:
        _wait(lambda: server.socket_path.exists() and len(server.connection_workers) == 4)
        for _ in range(server.max_connections):
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client.settimeout(5)
            client.connect(str(server.socket_path))
            client.sendall(b"\0")
            clients.append(client)
        _wait(lambda: server._status()["connection_count"] == server.max_connections)
        pool_ids = [id(worker) for worker in server.connection_workers]
        # This request waits in the finite OS backlog until an input deadline
        # frees a worker. No new service thread is spawned to read it.
        status = control_request(server.socket_path, "toy", "status", timeout=5)
        assert status["peak_connections"] == status["max_connections"] == 4
        assert [id(worker) for worker in server.connection_workers] == pool_ids
        assert sum(worker.is_alive() for worker in server.connection_workers) == 4
        assert status["request_count"] == 0
        assert control_request(server.socket_path, "toy", "stop", timeout=5)["result"]["drained"]
        owner.join(5)
        assert not owner.is_alive()
        assert all(not worker.is_alive() for worker in server.connection_workers + server.workers)
        assert server._status()["connection_count"] == 0
        assert not server.socket_path.exists()
    finally:
        for client in clients:
            client.close()
        if owner.is_alive():
            control_request(server.socket_path, "toy", "stop", timeout=5)
            owner.join(5)


class _OncePickledOutput:
    def __init__(self, counter):
        self.counter = counter

    def __reduce__(self):
        self.counter.append("pickle")
        if len(self.counter) > 1:
            raise AssertionError("response was serialized more than once")
        return bytes, (b"x" * (512 * 1024),)


def test_output_is_pickled_once_and_last_result_not_retained_by_idle_worker(tmp_path):
    counter, references = [], []

    def handler(payload):
        result = _OncePickledOutput(counter)
        references.append(weakref.ref(result))
        return result

    with _socket_transport(tmp_path, handler) as (server, connect):
        client, done = connect("infer")
        assert _read_frame(client)["result"] == b"x" * (512 * 1024)
        done.result(5)
        assert counter == ["pickle"]
        assert references[0]() is None
        assert server._status()["active_requests"] == []


@pytest.mark.parametrize("failure", ["oversize", "unpickleable", "handler"])
def test_output_failure_releases_credits_and_worker_survives(tmp_path, failure):
    def handler(payload):
        if payload.get("good"):
            return 42
        if failure == "oversize":
            return b"x" * (512 * 1024)
        if failure == "unpickleable":
            return lambda: None
        raise ValueError("cpu toy failure")

    with _socket_transport(tmp_path, handler, max_frame_bytes=4096) as (server, connect):
        failed, done = connect("infer", "failure")
        response = _read_frame(failed)
        done.result(5)
        assert response["type"] == "error"
        assert response["code"] == ("inference_failed" if failure == "handler"
                                      else "response_not_materializable")
        assert server._status()["failed_count"] == 1
        assert server._status()["inflight_requests"] == 0
        assert server.tasks.unfinished_tasks == 0
        good, done = connect("infer", "good", good=True)
        assert _read_frame(good)["result"] == 42
        done.result(5)
        assert server._status()["completed_count"] == 1


def test_serialized_frame_writer_rejects_oversize_before_retaining_it():
    with pytest.raises(FrameTooLarge):
        _frame({"result": b"x" * (512 * 1024)}, max_frame_bytes=1024)
    frame = _frame({"result": 42}, max_frame_bytes=1024)
    assert len(frame) == 8 + int.from_bytes(frame[:8], "big")
    assert pickle.loads(frame[8:]) == {"result": 42}


def test_protocol5_numpy_frame_roundtrip_and_byte_bound():
    np = pytest.importorskip("numpy")
    features = np.arange(32768, dtype=np.float32).reshape(128, 256)
    frame = _frame({"features": features}, max_frame_bytes=features.nbytes + 1024)
    assert len(frame) == 8 + int.from_bytes(frame[:8], "big")
    restored = pickle.loads(frame[8:])["features"]
    assert restored.dtype == features.dtype and restored.shape == features.shape
    assert np.array_equal(restored, features)
    with pytest.raises(FrameTooLarge):
        _frame({"features": features}, max_frame_bytes=features.nbytes - 1)


def test_unmaterializable_error_receipt_does_not_kill_worker_or_strand_drain(tmp_path):
    with _socket_transport(tmp_path, lambda payload: 42, max_frame_bytes=140) as (server, connect):
        # The request fits the toy server's limit, but both full response
        # envelopes exceed it. Closing is the only available transport outcome.
        for _ in range(2):
            client, done = connect("infer", "x")
            assert client.recv(1) == b""
            done.result(5)
            assert server.tasks.unfinished_tasks == 0
            assert server._status()["inflight_requests"] == 0
            assert server.workers[0].is_alive()
        assert server._status()["failed_count"] == 2


def test_disconnect_after_sent_inference_is_unknown_with_identity_and_no_replay(tmp_path):
    endpoint = tmp_path / "dropped.sock"
    tmp_path.chmod(0o700)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(endpoint))
    endpoint.chmod(0o600)
    listener.listen(1)
    spec = DaemonSpec(backend="cpu-toy", factory="json:dumps", settings={})
    attached = OwnedModelDaemon(spec, socket_path=endpoint)
    requests = []

    def drop_response():
        connection, _ = listener.accept()
        with connection:
            requests.append(_read_frame(connection))
            # Simulate a delivery deadline after part of a large response.
            connection.sendall((512 * 1024).to_bytes(8, "big") + b"partial")

    peer = threading.Thread(target=drop_response)
    peer.start()
    try:
        with pytest.raises(RequestOutcomeUnknown, match="disconnected after send") as error:
            attached._exchange({"type": "infer", "request_id": "one", "payload": {}}, timeout=2)
        peer.join(5)
        assert not peer.is_alive()
        assert error.value.request_id == "one"
        assert error.value.generation == attached.generation
        assert len(requests) == 1
        assert requests[0]["identity"] == attached.identity
        assert requests[0]["generation"] == attached.generation
    finally:
        listener.close()
        attached.shutdown()
