"""CPU fake-owner contracts for explicit cross-process model sharing."""

from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import threading

import pytest

from astrolabe import model_daemon, model_execution
from astrolabe.model_daemon import DaemonSpec, OwnedModelDaemon, model_identity
from astrolabe.model_execution import DEFAULT_MODEL_FRAME_BYTES, NativeModelClient
from astrolabe.runtime.admission import ResourceClient
from astrolabe.runtime import model_pool


@pytest.fixture
def shared(tmp_path, monkeypatch):
    for name in ("ASTROLABE_RESIDENT_MODEL_REGISTRY", "ASTROLABE_RESOURCE_SOCKET",
                 "ASTROLABE_RESOURCE_PROFILES", "ASTROLABE_MODEL_OWNER",
                 "ASTROLABE_MODEL_PROVIDERS"):
        monkeypatch.delenv(name, raising=False)
    spec = DaemonSpec("cpu-test", "cpu_shared_fixture:make", {"variant": "a"},
                      max_frame_bytes=DEFAULT_MODEL_FRAME_BYTES)
    fixture = SimpleNamespace(
        spec=spec, registry_path=tmp_path / "registry.json",
        socket=tmp_path / "owner" / "model.sock", broker=tmp_path / "broker.sock",
        owners=[], restored=[], captured={"seed": 17}, description_error=None,
        restore_error=None, snapshot_error=None, snapshot_calls=[],
    )
    fixture.identity = model_identity(spec)
    fixture.description = {
        "ok": True, "random_state": {"seed": 29},
        "metadata": {"device": "cuda:0", "runtime_lifecycle": {
            "backend": spec.backend, "identity": fixture.identity,
            "model_id": f"{fixture.identity}:resident", "pid": os.getpid(),
            "device": "cuda:0", "loader_count": 1,
        }},
    }
    fixture.snapshot = {
        "models": [{"model_id": f"{fixture.identity}:resident", "pid": os.getpid(),
                    "device": "GPU-fixture", "resident_bytes": 128,
                    "resident_ram_bytes": 64}],
        "devices": {"GPU-fixture": {}},
    }
    fixture.registry = {
        "schema": "astrolabe.resident-model-registry/v1",
        "broker_socket": str(fixture.broker),
        "models": {fixture.identity: {"backend": spec.backend,
                                      "socket": str(fixture.socket)}},
    }

    def publish():
        fixture.registry_path.write_text(json.dumps(fixture.registry), encoding="utf-8")
        monkeypatch.setenv("ASTROLABE_RESIDENT_MODEL_REGISTRY", str(fixture.registry_path))
        monkeypatch.setenv("ASTROLABE_RESOURCE_SOCKET", str(fixture.broker))

    fixture.publish = publish
    fixture.make = lambda **kwargs: NativeModelClient(
        spec.backend, spec.factory, spec.settings, **kwargs)

    class FakeOwner(OwnedModelDaemon):
        # Keep real attach construction/shutdown, fake only scientific requests.
        def __init__(self, spec, **kwargs):
            assert kwargs["socket_path"] is not None, "registry must never spawn"
            super().__init__(spec, **kwargs)
            self.shutdown_calls = 0
            self.requests = []
            fixture.owners.append(self)

        def request(self, request):
            self.requests.append(deepcopy(request))
            self._remote_pid = os.getpid()
            if request.get("operation") == "describe":
                if fixture.description_error is not None:
                    raise fixture.description_error
                return deepcopy(fixture.description)
            if request.get("operation") != "infer":
                return deepcopy(request)
            return {"ok": True, "result": request["payload"],
                    "random_state": {"seed": 31},
                    "runtime": {"request_id": request["runtime_request_id"]}}

        def shutdown(self):
            self.shutdown_calls += 1
            super().shutdown()

    def restore(state):
        if fixture.restore_error is not None:
            raise fixture.restore_error
        fixture.restored.append(deepcopy(state))

    def snapshot(client):
        fixture.snapshot_calls.append(client.socket_path)
        if fixture.snapshot_error is not None:
            raise fixture.snapshot_error
        return deepcopy(fixture.snapshot)

    def forbidden(*args, **kwargs):
        pytest.fail("registry cannot load locally, acquire the pool, or unregister the owner")

    monkeypatch.setattr(model_daemon, "OwnedModelDaemon", FakeOwner)
    monkeypatch.setattr(model_daemon, "LocalModelOwner", forbidden)
    monkeypatch.setattr(model_pool, "get_local_model_pool", forbidden)
    monkeypatch.setattr(model_execution, "capture_random_state", lambda: deepcopy(fixture.captured))
    monkeypatch.setattr(model_execution, "restore_random_state", restore)
    monkeypatch.setattr(ResourceClient, "snapshot", snapshot)
    monkeypatch.setattr(ResourceClient, "unregister_model", forbidden)
    publish()
    return fixture


def test_exact_match_attaches_describes_restores_and_keeps_physical_ledger(shared, monkeypatch):
    # Even trajectory mode and a different consumer visibility must attach.
    monkeypatch.setenv("ASTROLABE_MODEL_OWNER", "trajectory")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    shared.identity = model_identity(shared.spec)
    shared.registry["models"] = {shared.identity: {
        "backend": shared.spec.backend, "socket": str(shared.socket)}}
    shared.description["metadata"]["runtime_lifecycle"]["identity"] = shared.identity
    shared.publish()
    client = shared.make(socket_path=shared.socket.parent / "." / shared.socket.name)
    assert isinstance(client.owner, OwnedModelDaemon) and client.owner.attached
    assert client.owner.socket_path == shared.socket
    assert client.owner._proc is None
    assert shared.restored == [{"seed": 29}]
    assert shared.snapshot_calls == [str(shared.broker)]
    assert client.metadata["device"] == "GPU-fixture"
    resources = client.metadata["runtime_lifecycle"]["resources"]
    assert resources == dict(shared.snapshot["models"][0], broker_socket=str(shared.broker))
    assert client.request({"value": 7}) == {"value": 7}
    assert client.owner.requests[-1]["random_state"] == shared.captured
    assert shared.restored[-1] == {"seed": 31}
    client.shutdown()
    assert client.owner._closed is False  # Real attached shutdown leaves owner alive.
    assert len(shared.owners) == 1


def test_null_broker_is_explicit_and_does_not_query_a_broker(shared, monkeypatch):
    shared.registry["broker_socket"] = None
    shared.publish()
    monkeypatch.delenv("ASTROLABE_RESOURCE_SOCKET")
    client = shared.make()
    assert client.owner.attached
    assert client.metadata["runtime_lifecycle"]["resources"] == {"broker_socket": None}
    assert shared.snapshot_calls == []
    client.shutdown()


def test_null_registry_broker_rejects_an_active_broker(shared, monkeypatch):
    monkeypatch.delenv("ASTROLABE_RESOURCE_SOCKET")
    with pytest.raises(ValueError, match="broker_socket"):
        shared.make()
    assert shared.owners == []


def test_cpu_residency_uses_null_physical_device_from_ledger(shared):
    shared.snapshot["models"][0]["device"] = None
    client = shared.make()
    assert client.metadata["device"] == "cpu"
    assert client.metadata["runtime_lifecycle"]["resources"]["device"] is None
    client.shutdown()


def test_current_broker_is_canonicalized(shared, monkeypatch):
    monkeypatch.setenv("ASTROLABE_RESOURCE_SOCKET", str(shared.broker.parent / "unused" / ".." / shared.broker.name))
    shared.make().shutdown()
    assert shared.snapshot_calls == [str(shared.broker)]


@pytest.mark.parametrize("change,match", [
    (lambda s: s.registry.update(schema="other/v1"), "schema"),
    (lambda s: s.registry.pop("broker_socket"), "broker_socket"),
    (lambda s: s.registry.update(broker_socket=None), "broker_socket"),
    (lambda s: s.registry.update(broker_socket=str(s.broker.parent / "other.sock")), "broker_socket"),
    (lambda s: s.registry.update(broker_socket=str(s.broker.parent / "unused" / ".." / s.broker.name)), "broker_socket"),
    (lambda s: s.registry.update(models={}), "exact identity"),
    (lambda s: s.registry.update(models={"other-identity": s.registry["models"][s.identity]}), "exact identity"),
    (lambda s: s.registry["models"][s.identity].update(backend="other"), "backend"),
    (lambda s: s.registry["models"].update({s.identity: None}), "backend"),
    (lambda s: s.registry["models"][s.identity].update(socket=""), "socket"),
    (lambda s: s.registry["models"][s.identity].update(socket="relative.sock"), "socket"),
    (lambda s: s.registry["models"][s.identity].update(socket=None), "socket"),
])
@pytest.mark.parametrize("direct_payload", [False, True])
def test_invalid_registry_never_falls_back(shared, change, match, direct_payload):
    change(shared)
    shared.publish()
    with pytest.raises(ValueError, match=match):
        shared.make(direct_payload=direct_payload)
    assert shared.owners == shared.snapshot_calls == shared.restored == []


def test_explicit_socket_conflict_never_constructs_owner(shared):
    with pytest.raises(ValueError, match="socket_path conflicts"):
        shared.make(socket_path=shared.socket.with_name("other.sock"))
    assert shared.owners == []


@pytest.mark.parametrize("contents", ["{", "[]", "null"])
def test_malformed_registry_never_falls_back(shared, contents):
    shared.registry_path.write_text(contents)
    with pytest.raises(ValueError):
        shared.make()
    assert shared.owners == []


def test_missing_registry_file_never_falls_back(shared):
    shared.registry_path.unlink()
    with pytest.raises(FileNotFoundError):
        shared.make()
    assert shared.owners == []


def test_empty_registry_environment_is_not_opt_out(shared, monkeypatch):
    monkeypatch.setenv("ASTROLABE_RESIDENT_MODEL_REGISTRY", "")
    with pytest.raises(ValueError, match="nonempty"):
        shared.make()
    assert shared.owners == []


@pytest.mark.parametrize("phase", ["describe", "restore", "snapshot"])
def test_attach_error_cleans_up_handle_without_replacement_or_unregister(shared, phase):
    setattr(shared, f"{phase if phase != 'describe' else 'description'}_error",
            RuntimeError(f"{phase} failed"))
    with pytest.raises(RuntimeError, match=f"{phase} failed"):
        shared.make()
    assert len(shared.owners) == 1
    owner = shared.owners[0]
    assert owner.attached and owner.shutdown_calls == 1
    assert owner._proc is None and not owner._closed


@pytest.mark.parametrize("change,match", [
    (lambda s: s.description["metadata"]["runtime_lifecycle"].update(identity="other"), "describe"),
    (lambda s: s.description["metadata"]["runtime_lifecycle"].update(backend="other"), "describe"),
    (lambda s: s.description["metadata"]["runtime_lifecycle"].update(pid=-1), "describe"),
    (lambda s: s.snapshot.update(models=[]), "not registered"),
    (lambda s: s.snapshot["models"][0].update(pid=-1), "not registered"),
    (lambda s: s.snapshot["models"][0].update(device="cuda:0"), "physical device"),
])
def test_describe_and_ledger_must_identify_existing_owner(shared, change, match):
    change(shared)
    with pytest.raises(ValueError, match=match):
        shared.make()
    assert len(shared.owners) == 1 and shared.owners[0].shutdown_calls == 1
    assert shared.restored == []


def test_prepared_provider_spec_identity_is_used(shared, monkeypatch):
    def prepare(spec):
        spec.settings["prepared"] = True
        return spec

    monkeypatch.setattr(model_daemon, "_prepare_provider_spec", prepare)
    prepare(shared.spec)
    identity = model_identity(shared.spec)
    shared.registry["models"] = {identity: {"backend": shared.spec.backend,
                                           "socket": str(shared.socket)}}
    shared.description["metadata"]["runtime_lifecycle"]["identity"] = identity
    shared.publish()
    client = shared.make()
    assert client.owner.identity == identity
    client.shutdown()


@pytest.mark.parametrize("trajectory", [False, True])
def test_no_registry_preserves_default_owner_and_pool_path(shared, monkeypatch, trajectory):
    monkeypatch.delenv("ASTROLABE_RESIDENT_MODEL_REGISTRY")
    if trajectory:
        monkeypatch.setenv("ASTROLABE_MODEL_OWNER", "trajectory")
    calls = []

    class DefaultOwner:
        def __init__(self, spec, **kwargs):
            calls.append((self, kwargs))

        def request(self, request):
            return deepcopy(shared.description)

    monkeypatch.setattr(model_daemon, "LocalModelOwner" if trajectory else "OwnedModelDaemon", DefaultOwner)
    pool_calls = []

    class Pool:
        def acquire(self, spec, owner_type, kwargs, identity):
            pool_calls.append((owner_type, identity))
            return None

    monkeypatch.setattr(model_pool, "get_local_model_pool", lambda: Pool())
    client = shared.make()
    assert isinstance(client.owner, DefaultOwner)
    assert len(calls) == 1 and calls[0][1]["socket_path"] is None
    assert pool_calls[0][0] is DefaultOwner
    assert client.metadata == shared.description["metadata"]
    assert shared.snapshot_calls == []
    assert shared.restored == [{"seed": 29}]


@pytest.mark.parametrize("direct_payload", [False, True])
def test_real_private_socket_transport_shares_one_fake_load_and_survives_shutdown(shared, monkeypatch, direct_payload):
    # Exercise real socket permissions, handshake, and attached shutdown with a
    # CPU handler. No framework, model weights, subprocess or GPU is involved.
    monkeypatch.setattr(model_daemon, "OwnedModelDaemon", OwnedModelDaemon)
    ready = threading.Event()
    failures = []
    loads = []
    shared.socket.parent.mkdir(mode=0o700)

    def handle(request):
        if request.get("operation") == "describe":
            return deepcopy(shared.description)
        if direct_payload:
            if request.get("fail"):
                raise ValueError("native direct failure")
            return request
        return {"ok": True, "result": request["payload"],
                "random_state": request["random_state"]}

    server = model_daemon._Server(
        handle, socket_path=shared.socket, identity=shared.identity,
        backend=shared.spec.backend, max_pending=4,
        max_frame_bytes=DEFAULT_MODEL_FRAME_BYTES, parent_pid=None)
    original_event = model_daemon._event

    def event(name, **kwargs):
        original_event(name, **kwargs)
        if name == "service_ready":
            ready.set()

    monkeypatch.setattr(model_daemon, "_event", event)

    def run():
        loads.append("cpu-handler")
        try:
            server.run()
        except BaseException as error:
            failures.append(error)
            ready.set()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        assert ready.wait(5) and not failures
        first, second = (shared.make(direct_payload=direct_payload),
                         shared.make(direct_payload=direct_payload))
        assert first.pid == second.pid == os.getpid()
        assert first.request({"worker": 1}) == {"worker": 1}
        if direct_payload:
            with pytest.raises(model_daemon.DaemonRequestFailed, match="native direct failure") as failed:
                first.request({"fail": True})
            assert failed.value.remote_type == "ValueError"
        first.shutdown()
        assert second.request({"worker": 2}) == {"worker": 2}
        second.shutdown()
        assert model_daemon.control_request(shared.socket, shared.identity)["state"] == "running"
        assert loads == ["cpu-handler"]

        # Registry lookup alone cannot override the real owner's handshake.
        shared.spec.settings["variant"] = "b"
        different = model_identity(shared.spec)
        shared.registry["models"] = {different: {
            "backend": shared.spec.backend, "socket": str(shared.socket)}}
        shared.publish()
        with pytest.raises(model_daemon.IdentityMismatch):
            shared.make(direct_payload=direct_payload)
        assert model_daemon.control_request(shared.socket, shared.identity)["state"] == "running"

        shared.spec.settings["variant"] = "a"
        shared.registry["models"] = {shared.identity: {
            "backend": shared.spec.backend, "socket": str(shared.socket)}}
        shared.publish()
        shared.socket.chmod(0o666)
        try:
            with pytest.raises(PermissionError, match="0600"):
                shared.make(direct_payload=direct_payload)
        finally:
            shared.socket.chmod(0o600)
        assert model_daemon.control_request(shared.socket, shared.identity)["state"] == "running"
    finally:
        if server.listener is not None:
            model_daemon.control_request(shared.socket, shared.identity, "stop")
        thread.join(timeout=5)
    assert not thread.is_alive() and not failures


def test_direct_payload_keeps_raw_results_rng_and_request_provenance(shared, monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("direct client must not transfer RNG or acquire device admission")

    monkeypatch.setattr(model_execution, "capture_random_state", forbidden)
    monkeypatch.setattr(model_execution, "restore_random_state", forbidden)
    monkeypatch.setattr(model_execution, "device_operation", forbidden)
    trace = tmp_path / "requests.jsonl"
    monkeypatch.setenv("ASTROLABE_MODEL_REQUEST_TRACE_PATH", str(trace))
    client = shared.make(direct_payload=True)
    payload = {"request_dir": "/native/request", "ok": False, "result": "raw"}
    assert client.request(payload) == payload
    assert client.owner.requests == [{"operation": "describe"}, payload]
    assert shared.restored == [] and client.last_runtime_receipt is None
    original = model_daemon.DaemonRequestFailed("native failure", remote_type="ValueError")

    def fail(payload):
        raise original

    monkeypatch.setattr(client.owner, "request", fail)
    with pytest.raises(model_daemon.DaemonRequestFailed) as failed:
        client.request({"request_dir": "/native/failed"})
    assert failed.value is original
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    assert [event["event"] for event in events] == [
        "client_ready", "request_started", "request_finished", "request_started", "request_error"]
    assert events[1]["request_id"] == events[2]["request_id"]
    assert events[3]["request_id"] == events[4]["request_id"]
    assert events[4]["error_type"] == "DaemonRequestFailed"
    assert all(event["model_id"] == client.metadata["runtime_lifecycle"]["model_id"] for event in events)
    client.shutdown()


def test_esmfold2_lazy_shared_client_and_describe_keep_direct_protocol(tmp_path, monkeypatch):
    from astrolabe.evaluators.wrappers import esmfold2

    created = []

    class Client:
        def __init__(self, backend, factory, settings, **kwargs):
            created.append((backend, factory, settings, kwargs))

        def shutdown(self):
            pass

    monkeypatch.setattr(model_execution, "NativeModelClient", Client)
    monkeypatch.setenv("ASTROLABE_EXECUTION_TOKEN", "parent-only")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "ccd.pkl").write_bytes(b"fake CCD")
    for name in ("config.json", "model.safetensors.index.json"):
        (snapshot / name).write_text("{}")
    monkeypatch.setenv("GH_ESMFOLD2_ESMC_SNAPSHOT", str(snapshot))
    runner = esmfold2.ESMFold2Runner(model_name=str(snapshot), device="cpu",
                                   local_files_only=True, python_bin="/isolated/python")
    assert created == []
    client = runner.prepare_shared_model()
    assert runner.prepare_shared_model() is client
    assert len(created) == 1 and created[0][0] == "esmfold2"
    assert created[0][2]["identity"]["model_name"] == str(snapshot)
    assert created[0][3]["direct_payload"] is True
    assert created[0][3]["python_bin"] == "/isolated/python"
    assert set(created[0][3]["env"]) == {"PYTHONPATH"}
    spec = runner._daemon_spec(created[0][2]["identity"])
    monkeypatch.setenv("ASTROLABE_EXECUTION_TOKEN", "worker-only")
    assert model_identity(spec) == model_identity(runner._daemon_spec(spec.settings["identity"]))
    worker = esmfold2._ESMFold2Worker(None, None, {}, "cpu", SimpleNamespace(
        handle=lambda request: {"metadata": {"device": "cpu"}, "random_state": {"seed": 29}}))
    assert worker({"operation": "describe"}) == {
        "metadata": {"device": "cpu"}, "random_state": {"seed": 29}}
    runner.shutdown()
    assert esmfold2.ESMFold2Runner(execution_mode="subprocess").prepare_shared_model() is None
