"""Request-local RNG receipts and host/device overlap on one loaded model."""
from concurrent.futures import ThreadPoolExecutor
import copy
import random
import threading

import pytest

from astrolabe.model_execution import (
    KernelStages, capture_random_state, kernel_handler,
)


def request(state, index, **payload):
    return dict(operation="infer", random_state=copy.deepcopy(state),
                runtime_request_id=f"request-{index}",
                payload=dict(index=index, **payload))


def test_preparation_overlaps_execution_without_sharing_rng_or_reloading(monkeypatch):
    monkeypatch.setenv("ASTROLABE_MODEL_PIPELINE", "1")
    executing = threading.Event()
    second_prepared = threading.Event()
    loads = []
    model = object()
    active = 0
    peak = 0

    def prepare(payload):
        if payload["index"] == 1:
            assert executing.wait(5), "first device stage never started"
            second_prepared.set()
        return payload

    def execute(payload):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            if payload["index"] == 0:
                executing.set()
                assert second_prepared.wait(5), "host preparation was serialized behind execution"
            return dict(model=id(model), draw=random.random(), index=payload["index"])
        finally:
            active -= 1

    def loader(settings):
        loads.append(id(model))
        return KernelStages(prepare, execute, resource="cpu",
                            pipeline_slots=2, rng_free_host_stages=True), {}

    random.seed(17)
    first_state = capture_random_state()
    expected_first = random.random()
    expected_first_after = capture_random_state()
    random.seed(91)
    second_state = capture_random_state()
    expected_second = random.random()
    expected_second_after = capture_random_state()
    handler = kernel_handler({}, loader)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(handler, request(first_state, 0))
        second = pool.submit(handler, request(second_state, 1))
        results = [first.result(10), second.result(10)]
    assert [r["result"]["draw"] for r in results] == [expected_first, expected_second]
    assert [r["result"]["model"] for r in results] == [id(model), id(model)]
    assert [r["runtime"]["request_id"] for r in results] == ["request-0", "request-1"]
    assert results[0]["random_state"] == expected_first_after
    assert results[1]["random_state"] == expected_second_after
    assert loads == [id(model)]
    assert peak == 1, "default-generator execution must remain serialized"
    lifecycle = handler(dict(operation="describe"))["metadata"]["runtime_lifecycle"]
    assert lifecycle["pipeline_slots"] == lifecycle["peak_active_requests"] == 2
    assert lifecycle["device_execution_slots"] == 1
    assert lifecycle["active_requests"] == 0 and lifecycle["state"] == "ready"


@pytest.mark.parametrize("enabled,qualified,expected", [
    ("0", True, 1), ("1", False, 1), ("1", True, 2),
])
def test_pipeline_requires_backend_contract_and_explicit_enable(monkeypatch, enabled, qualified, expected):
    monkeypatch.setenv("ASTROLABE_MODEL_PIPELINE", enabled)
    handler = kernel_handler({}, lambda _: (
        KernelStages(lambda x: x, lambda x: x, resource="cpu",
                     pipeline_slots=2, rng_free_host_stages=qualified), {}))
    assert handler.max_concurrency == expected


def test_invalid_pipeline_flag_rejects_before_weight_loading(monkeypatch):
    monkeypatch.setenv("ASTROLABE_MODEL_PIPELINE", "true")
    loaded = []
    with pytest.raises(ValueError, match="ASTROLABE_MODEL_PIPELINE"):
        kernel_handler({}, lambda _: loaded.append(True))
    assert loaded == []


def test_optional_stage_clock_preserves_rng_and_execution_order(monkeypatch, tmp_path):
    monkeypatch.setenv("ASTROLABE_MODEL_PIPELINE", "1")
    monkeypatch.setenv("ASTROLABE_MODEL_REQUEST_TRACE_PATH", str(tmp_path / "requests.jsonl"))
    state = capture_random_state()
    handler = kernel_handler({}, lambda _: (
        KernelStages(lambda x: x, lambda x: x, resource="cpu",
                     pipeline_slots=2, rng_free_host_stages=True), {}))
    response = handler(request(state, 0))
    assert response["ok"] and response["random_state"] == state
    timings = response["runtime"]["timings"]
    for name in ("prepare", "execute", "finalize"):
        stage = timings[name]
        assert stage["started_unix"] <= stage["admitted_unix"] <= stage["finished_unix"]
    assert timings["prepare"]["finished_unix"] <= timings["execute"]["started_unix"]
    assert timings["execute"]["finished_unix"] <= timings["finalize"]["started_unix"]


def test_failed_execute_keeps_advanced_rng_and_next_request_is_independent(monkeypatch):
    monkeypatch.setenv("ASTROLABE_MODEL_PIPELINE", "1")
    random.seed(37)
    initial = capture_random_state()
    random.random()
    expected_after = capture_random_state()

    def execute(payload):
        value = random.random()
        if payload.get("fail"):
            raise ValueError("after one draw")
        return value

    handler = kernel_handler({}, lambda _: (
        KernelStages(lambda x: x, execute, resource="cpu",
                     pipeline_slots=2, rng_free_host_stages=True), {}))
    failed = handler(request(initial, 0, fail=True))
    assert not failed["ok"] and failed["random_state"] == expected_after
    succeeded = handler(request(initial, 1))
    assert succeeded["ok"]
    assert succeeded["random_state"] == expected_after
    assert succeeded["runtime"]["request_id"] == "request-1"


def test_failed_prepare_returns_input_rng_without_capturing_another_request(monkeypatch):
    monkeypatch.setenv("ASTROLABE_MODEL_PIPELINE", "1")
    random.seed(37)
    initial = capture_random_state()

    def prepare(payload):
        raise ValueError("invalid input")

    handler = kernel_handler({}, lambda _: (
        KernelStages(prepare, lambda x: x, resource="cpu",
                     pipeline_slots=2, rng_free_host_stages=True), {}))
    random.seed(999)
    response = handler(request(initial, 0))
    assert not response["ok"] and response["random_state"] == initial
    assert response["runtime"]["lifecycle"]["active_requests"] == 0


def test_quarantine_blocks_prepared_following_request(monkeypatch):
    monkeypatch.setenv("ASTROLABE_MODEL_PIPELINE", "1")
    first_execute = threading.Event()
    second_prepare = threading.Event()
    executions = []

    def prepare(payload):
        if payload["index"]:
            assert first_execute.wait(5)
            second_prepare.set()
        return payload

    def execute(payload):
        executions.append(payload["index"])
        if payload["index"] == 0:
            first_execute.set()
            assert second_prepare.wait(5)
            raise RuntimeError("out of memory")
        pytest.fail("prepared request executed after quarantine")

    handler = kernel_handler({}, lambda _: (
        KernelStages(prepare, execute, resource="cpu",
                     pipeline_slots=2, rng_free_host_stages=True), {}))
    state = capture_random_state()
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(handler, request(state, index)) for index in range(2)]
        responses = [f.result(10) for f in futures]
    assert all(not response["ok"] for response in responses)
    assert executions == [0]
    assert handler(dict(operation="describe"))["metadata"]["runtime_lifecycle"]["state"] == "quarantined"


def test_oom_is_published_before_rng_unlock_even_if_failure_response_is_delayed(monkeypatch):
    import astrolabe.model_execution as execution

    monkeypatch.setenv("ASTROLABE_MODEL_PIPELINE", "1")
    entered = threading.Event()
    prepared = threading.Event()
    unlocked = threading.Event()
    return_failure = threading.Event()
    executed = []
    failing_thread = []

    class PauseAfterUnlock:
        def __init__(self):
            self.lock = threading.RLock()

        def __enter__(self):
            self.lock.acquire()

        def __exit__(self, *args):
            self.lock.release()
            if failing_thread and threading.get_ident() == failing_thread[0]:
                unlocked.set()
                assert return_failure.wait(5)

    monkeypatch.setattr(execution, "_KERNEL_RANDOM_LOCK", PauseAfterUnlock())

    def prepare(payload):
        if payload["index"] == 1:
            assert entered.wait(5)
            prepared.set()
        return payload

    def execute(payload):
        executed.append(payload["index"])
        if payload["index"] == 0:
            failing_thread.append(threading.get_ident())
            entered.set()
            assert prepared.wait(5)
            random.random()
            raise RuntimeError("out of memory")
        pytest.fail("execution passed a quarantined RNG owner")

    random.seed(71)
    state = capture_random_state()
    random.random()
    after_failure = capture_random_state()
    handler = kernel_handler({}, lambda _: (
        KernelStages(prepare, execute, resource="cpu",
                     pipeline_slots=2, rng_free_host_stages=True), {}))
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(handler, request(state, 0))
        following = pool.submit(handler, request(state, 1))
        try:
            assert unlocked.wait(5)
            blocked = following.result(5)
            assert not blocked["ok"] and executed == [0]
        finally:
            return_failure.set()
        failed = first.result(5)
    assert not failed["ok"] and failed["random_state"] == after_failure
