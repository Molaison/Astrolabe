"""Scientific-stream and real completion tests for the common kernel port."""
from concurrent.futures import Future
from contextlib import contextmanager
import random

import pytest

from astrolabe.inference_pipeline import Pending
from astrolabe.model_execution import (
    KernelStages, capture_random_state, kernel_handler, model_kernel_context,
)


def test_stages_keep_random_order_and_load_once_across_requests():
    initial = capture_random_state()
    random.seed(81)
    initial = capture_random_state()
    expected = [random.random() for _ in range(7)]
    loads = []

    def loader(settings):
        loads.append(random.random())
        return KernelStages(
            lambda payload: [random.random()],
            lambda value: value + [random.random()],
            lambda value: value + [random.random()],
        ), {"device": "cpu"}

    with model_kernel_context("fixture", "fixed-checkpoint"):
        handler = kernel_handler(
            {"_initial_random_state": initial}, loader, torch_device="cpu",
            lifecycle={"residency": "resident_model", "weights_loaded": True},
        )
    description = handler({"operation": "describe"})
    first = handler({"operation": "infer", "payload": {},
                     "random_state": description["random_state"],
                     "runtime_request_id": "first"})
    second = handler({"operation": "infer", "payload": {},
                      "random_state": first["random_state"],
                      "runtime_request_id": "second"})
    assert loads == expected[:1]
    assert first["result"] == expected[1:4]
    assert second["result"] == expected[4:7]
    assert description["metadata"]["runtime_lifecycle"]["requests"] == 0
    assert second["runtime"]["lifecycle"]["loader_count"] == 1
    assert second["runtime"]["lifecycle"]["requests"] == 2
    assert second["runtime"]["request_id"] == "second"
    assert set(first["runtime"]["timings"]) == {"prepare", "execute", "finalize"}


def test_async_completion_is_observed_before_cpu_finalize():
    completed = Future()
    order = []

    def loader(settings):
        def execute(value):
            def finish():
                order.append("completed")
                completed.set_result(None)
            return Pending(value, finish, keepalive=object())
        def finalize(value):
            assert completed.done()
            order.append("finalize")
            return value
        return KernelStages(lambda value: value, execute, finalize), {}

    handler = kernel_handler({}, loader)
    response = handler({"operation": "infer", "payload": {"value": 1},
                        "random_state": capture_random_state()})
    assert response["ok"]
    assert response["result"] == {"value": 1}
    assert order == ["completed", "finalize"]


def test_failed_prepare_does_not_run_execute_or_finalize_and_advances_rng():
    random.seed(41)
    state = capture_random_state()
    expected = random.random()
    calls = []

    def prepare(value):
        assert random.random() == expected
        raise ValueError("invalid input")

    handler = kernel_handler({}, lambda settings: (
        KernelStages(prepare, lambda value: calls.append("execute"),
                     lambda value: calls.append("finalize")), {}
    ))
    response = handler({"operation": "infer", "payload": {},
                        "random_state": state})
    assert not response["ok"]
    assert response["error_type"] == "ValueError"
    assert response["random_state"] != state
    assert not calls
    assert response["runtime"]["lifecycle"]["failures"] == 1


def test_unknown_resource_fails_before_backend_execution():
    handler = kernel_handler({}, lambda settings: (
        KernelStages(lambda value: value, lambda value: pytest.fail("ran"),
                     resource="unlimited"), {}
    ))
    response = handler({"operation": "infer", "payload": {},
                        "random_state": capture_random_state()})
    assert not response["ok"] and "Unknown kernel resource" in response["error"]


def test_cpu_phases_release_gpu_and_local_nested_guard_reuses_lease(monkeypatch):
    from astrolabe import model_execution

    active = []
    events = []

    class Client:
        socket_path = "/test/broker"

        @contextmanager
        def lease(self, **request):
            events.append(("start", request["resource"], request["phase"]))
            active.append(request["resource"])
            try:
                yield object()
            finally:
                active.pop()
                events.append(("end", request["resource"], request["phase"]))

        def register_model(self, *args, **kwargs):
            assert active == ["gpu"]

    monkeypatch.setattr(model_execution, "_resource_client", lambda: Client())

    def cpu(value):
        assert active == ["cpu"]
        return value

    def execute(value):
        assert active == ["gpu"]
        with model_execution.device_operation("nested_native_helper"):
            assert active == ["gpu"]
        return value

    with model_kernel_context("fixture", "frozen-model"):
        handler = kernel_handler({}, lambda settings: (
            KernelStages(cpu, execute, cpu), {"device": "cuda:0"}
        ))
    result = handler({"operation": "infer", "payload": {"data": 1},
                      "random_state": capture_random_state()})
    assert result["ok"]
    assert events == [
        ("start", "gpu", "load"), ("end", "gpu", "load"),
        ("start", "cpu", "prepare"), ("end", "cpu", "prepare"),
        ("start", "gpu", "execute"), ("end", "gpu", "execute"),
        ("start", "cpu", "finalize"), ("end", "cpu", "finalize"),
    ]


def test_resource_profile_cannot_change_scientific_loader_settings():
    received = []

    def loader(settings):
        received.append(settings)
        return lambda payload: payload, {}

    with model_kernel_context("fixture", "current-model"):
        with pytest.raises(ValueError, match="different exact model"):
            kernel_handler({"_runtime_profile": {"identity": "other"}}, loader)
        handler = kernel_handler({
            "scientific_setting": 19,
            "_runtime_profile": {
                "identity": "current-model", "resident_bytes": 7,
                "scratch_bytes": 11, "concurrency_certified": True,
            },
        }, loader)
    assert received == [{"scientific_setting": 19}]
    assert handler({"operation": "describe"})["metadata"]["runtime_lifecycle"][
        "resource_profile"]["resident_bytes"] == 7
