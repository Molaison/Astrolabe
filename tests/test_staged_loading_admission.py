"""Loading split keeps admission, RNG consumption and request narrowing real."""
from contextlib import contextmanager
import random
from types import SimpleNamespace

import pytest

from astrolabe import model_execution as execution
from astrolabe.runtime.loading import StagedLoader


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_staged_load_charges_prepared_ram_and_releases_only_after_transfer(monkeypatch, device):
    import torch

    events, active, models = [], [], {}
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "init", lambda: events.append("cuda_init"))

    class Client:
        socket_path = "/fixture/shared-broker"

        @contextmanager
        def lease(self, **request):
            assert not active
            active.append(request)
            events.append(("lease", request["resource"], request["phase"]))
            try:
                yield SimpleNamespace(device="GPU-fixture")
            finally:
                active.pop()
                events.append(("release", request["resource"], request["phase"]))

        def register_model(self, model_id, **values):
            assert active
            if values["metadata"]["residency"] == "prepared_cpu_weights":
                assert active[0]["resource"] == "cpu"
                assert values["resident_ram_bytes"] == 30
                assert values["device"] == "cpu"
            else:
                assert len(models) == 1
                assert values["resident_ram_bytes"] == 10
                assert active[0]["resource"] == ("cpu" if device == "cpu" else "gpu")
                assert active[0]["ram_bytes"] == 0
                prepared_id = values["prepared_model_id"]
                assert active[0]["prepared_model_id"] == prepared_id
                assert models.pop(prepared_id)["resident_ram_bytes"] == 30
            models[model_id] = values

        def unregister_model(self, model_id):
            pytest.fail("RAM ownership must transfer atomically at registration")

    monkeypatch.setattr(execution, "_resource_client", lambda: Client())
    random.seed(123)
    state = execution.capture_random_state()
    expected = [random.random() for _ in range(3)]
    stages = []

    def prepare(settings):
        assert active[0]["resource"] == "cpu"
        stages.append(random.random())
        return object()

    def place(prepared):
        assert len(models) == 1
        stages.append(random.random())
        return lambda payload: random.random(), {"device": device}

    with execution.model_kernel_context("fixture", "exact"):
        handler = execution.kernel_handler(
            {"_initial_random_state": state},
            StagedLoader(prepare, place), torch_device=device,
            lifecycle={"resident_ram_bytes": 10, "scratch_ram_bytes": 20},
        )
    description = handler({"operation": "describe"})
    response = handler({"operation": "infer", "payload": {},
                        "random_state": description["random_state"]})
    assert stages == expected[:2]
    assert response["ok"] and response["result"] == expected[2]
    assert len(models) == 1
    transition = description["metadata"]["initialization_transition"]
    assert transition["before"] == state
    assert transition["after"] == description["random_state"]
    assert set(description["metadata"]["load_admission_stages"]) == {
        "cpu_prepare", "device_place"}
    assert events[:2] == [("lease", "cpu", "load"), ("release", "cpu", "load")]


@pytest.mark.parametrize("staged", [True, False])
def test_request_flag_can_narrow_but_never_widen_model_certificate(monkeypatch, staged):
    requests = []

    class Client:
        socket_path = "/fixture/shared-broker"

        @contextmanager
        def lease(self, **request):
            requests.append(request)
            yield SimpleNamespace(device="GPU-fixture")

        def register_model(self, *args, **kwargs):
            pass

    monkeypatch.setattr(execution, "_resource_client", lambda: Client())
    for certificate in (False, True):
        kernel = (execution.KernelStages(lambda x: x, lambda x: x)
                  if staged else lambda x: x)
        with execution.model_kernel_context("fixture", "exact"):
            handler = execution.kernel_handler(
                {"_runtime_profile": {"identity": "exact",
                                      "concurrency_certified": certificate,
                                      "shapes": [execution.resource_shape({})]}},
                lambda settings: (kernel, {}))
        state = handler({"operation": "describe"})["random_state"]
        for allowed in (False, True):
            response = handler({"operation": "infer", "payload": {},
                                "random_state": state,
                                "concurrency_certified": allowed})
            assert response["ok"]
            assert [r for r in requests if r["phase"] == "execute"][-1][
                "concurrency_certified"] is (certificate and allowed)


def test_cpu_prepare_failure_never_takes_gpu_lease_or_registers_model(monkeypatch):
    events = []

    class Client:
        socket_path = "/fixture/shared-broker"

        @contextmanager
        def lease(self, **request):
            events.append(request["resource"])
            yield None

        def register_model(self, *args, **kwargs):
            pytest.fail("Failed preparation cannot register ready weights")

    monkeypatch.setattr(execution, "_resource_client", lambda: Client())
    def fail(settings):
        raise ValueError("bad weights")
    with pytest.raises(ValueError, match="bad weights"):
        execution.kernel_handler({}, StagedLoader(fail, lambda x: pytest.fail("placed")),
                                 lifecycle={"ram_bytes": 30})
    assert events == ["cpu"]


def test_unknown_preparation_memory_keeps_conservative_load_boundary(monkeypatch):
    from contextlib import nullcontext
    active, events = [], []
    monkeypatch.setattr(execution, "_kernel_random_scope",
                        lambda *args, **kwargs: nullcontext())
    class Client:
        socket_path = "/fixture/shared-broker"

        @contextmanager
        def lease(self, **request):
            assert "prepared_model_id" not in request
            active.append(request["resource"])
            events.append(request)
            try:
                yield SimpleNamespace(device="GPU-fixture")
            finally:
                active.pop()

        def register_model(self, model_id, **values):
            assert active == ["gpu"]
            assert "prepared_model_id" not in values

    monkeypatch.setattr(execution, "_resource_client", lambda: Client())
    def prepare(settings):
        assert active == ["gpu"]
        return "weights"
    def place(value):
        assert active == ["gpu"] and value == "weights"
        return lambda x: x, {}
    handler = execution.kernel_handler({}, StagedLoader(prepare, place))
    metadata = handler({"operation": "describe"})["metadata"]
    assert metadata["split_loading"] is False
    assert metadata["split_loading_reason"] == "unknown_prepared_ram_conservative_load"
    assert len(events) == 1 and events[0]["phase"] == "load"


@pytest.mark.parametrize("dependency", [
    "runtime/loading.py", "runtime/model_pool.py",
    "evaluators/esmfold.py", "evaluators/esmfold_stages.py",
])
def test_resident_identity_binds_loading_pool_and_actual_esmfold_code(monkeypatch, tmp_path, dependency):
    from astrolabe import model_daemon

    monkeypatch.setattr(model_daemon, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(model_daemon, "_adapter_hash", lambda spec: "frozen-factory")
    path = tmp_path / "astrolabe" / dependency
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("original dependency")
    spec = model_daemon.DaemonSpec(
        backend="esmfold", factory="fixture:factory", settings={"precision": "same"})
    before = model_daemon.model_identity(spec)
    path.write_text("changed dependency")
    assert model_daemon.model_identity(spec) != before


@pytest.mark.parametrize("failure", ["request_id", "model_id", "missing_runtime", "rng"])
def test_unverifiable_executed_response_quarantines_borrowed_owner_without_replay(monkeypatch, failure):
    import threading

    calls, quarantine = [], []
    class Owner:
        def request(self, request):
            calls.append(request)
            response = {"ok": True, "result": 1, "random_state": {},
                        "runtime": {"request_id": request["runtime_request_id"],
                                    "model_id": "expected-model"}}
            if failure == "request_id":
                response["runtime"]["request_id"] = "another-request"
            if failure == "model_id":
                response["runtime"]["model_id"] = "another-model"
            if failure == "missing_runtime":
                response.pop("runtime")
            return response

        def quarantine(self):
            quarantine.append(True)

    client = execution.NativeModelClient.__new__(execution.NativeModelClient)
    client.owner, client._request_lock = Owner(), threading.RLock()
    client.metadata = {"runtime_lifecycle": {"model_id": "expected-model"}}
    if failure == "rng":
        def reject(state):
            raise ValueError("invalid executed RNG continuation")
        monkeypatch.setattr(execution, "restore_random_state", reject)
    with pytest.raises((RuntimeError, ValueError)):
        client.request({})
    assert len(calls) == 1
    assert quarantine == [True]


def test_actual_kernel_and_broker_transfer_at_preparation_budget_without_headroom(monkeypatch, tmp_path):
    from contextlib import nullcontext
    from astrolabe.runtime.admission import ResourceBroker, ResourceClient

    monkeypatch.setattr(execution, "_kernel_random_scope",
                        lambda *args, **kwargs: nullcontext())
    # Synthetic bytes exercise the actual RPC capacity boundary, not physical RAM.
    broker = ResourceBroker({
        "devices": [], "cpu_slots": 2, "ram_bytes": 40, "output_bytes": 100,
        "socket_dir": str(tmp_path), "shutdown_timeout": 2,
    }, sampler=lambda: []).start()
    client = ResourceClient(broker.socket_path)
    metadata = None
    try:
        def prepare(settings):
            assert client.snapshot()["ram_bytes"] == 30
            return "prepared weights"
        def place(value):
            snapshot = client.snapshot()
            assert value == "prepared weights"
            assert snapshot["resident_ram_bytes"] == 30
            assert snapshot["transient_ram_bytes"] == 0
            return lambda payload: payload, {"device": "cpu"}
        handler = execution.kernel_handler(
            {}, StagedLoader(prepare, place), torch_device="cpu",
            lifecycle={"resident_ram_bytes": 10, "scratch_ram_bytes": 20})
        metadata = handler({"operation": "describe"})["metadata"]
        snapshot = client.snapshot()
        assert metadata["split_loading"] is True
        assert snapshot["resident_models"] == 1
        assert snapshot["resident_ram_bytes"] == 10
        assert snapshot["active"] == 0
        assert snapshot["pending"] == 0
    finally:
        # Test owns no scientific model or real device allocation.
        for model in client.snapshot()["models"]:
            client.unregister_model(model["model_id"])
        broker.close()
