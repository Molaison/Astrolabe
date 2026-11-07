"""Stage ownership with a real broker and synthetic device completions."""
from concurrent.futures import Future, ThreadPoolExecutor
import copy
import threading
import uuid

import pytest

import astrolabe.model_execution as execution
from astrolabe.inference_pipeline import Pending
from astrolabe.runtime import AdmissionError, AdmissionTimeout, ResourceBroker, ResourceClient


@pytest.fixture
def broker(tmp_path, monkeypatch):
    for name in ("CUDA_VISIBLE_DEVICES", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS",
                 "ASTROLABE_RESOURCE_SOCKET"):
        monkeypatch.delenv(name, raising=False)
    device = "GPU-stage-test-" + uuid.uuid4().hex
    row = dict(index=0, uuid=device, total_bytes=1000, free_bytes=1000,
               utilization=0, processes=[])
    owner = ResourceBroker(
        dict(devices=[dict(uuid=device, index=0, vram_bytes=800)],
             socket_dir=str(tmp_path), cpu_slots=2, ram_bytes=1000,
             output_bytes=1000, max_pending_bytes=10000,
             sample_interval=.02, sample_max_age=2, shutdown_timeout=2,
             max_concurrency=1),
        lambda: [copy.deepcopy(row)],
    ).start()
    client = ResourceClient(owner.socket_path)
    yield owner, client
    for model_id in list(owner._models):
        client.unregister_model(model_id)
    owner.close()


def records(owner):
    with owner._cv:
        return list(owner._leases.values())


def test_complete_model_slot_and_raw_memory_have_different_lifetimes(broker):
    owner, client = broker
    dispatch = threading.Event()
    finalized = threading.Event()
    release_finalize = threading.Event()
    completion = Future()
    class NativeOutput:
        __module__ = "torch.synthetic"

        def detach(self):
            raise AssertionError("native output was materialized before reduction")

    raw = NativeOutput()
    seen = []

    def prepare(payload):
        assert not any(r["evaluator_inference"] for r in records(owner))
        assert any(r["capacity_only"] for r in records(owner))
        seen.append("prepare")
        return payload

    def infer(payload):
        assert sum(r["evaluator_inference"] for r in records(owner)) == 1
        seen.append("infer")
        dispatch.set()
        return Pending(raw, completion, keepalive=payload)

    def finalize(value):
        assert value is raw, "runtime replaced native output before finalize"
        assert not any(r["evaluator_inference"] for r in records(owner))
        assert any(r["capacity_only"] for r in records(owner))
        seen.append("finalize")
        finalized.set()
        assert release_finalize.wait(5)
        return {"value": "original"}

    handler = execution.kernel_handler({}, lambda _: (
        execution.EvaluatorStages(prepare, infer, finalize), {"device": "cuda:0"}),
        lifecycle=dict(backend="stage-test", device="cuda:0",
                       resident_bytes=100, scratch_bytes=200, output_bytes=20))
    state = execution.capture_random_state()
    with ThreadPoolExecutor(1) as pool:
        result = pool.submit(handler, dict(
            operation="infer", payload={"input": 1}, random_state=state))
        try:
            assert dispatch.wait(5)
            assert not finalized.is_set()
            assert any(r["evaluator_inference"] for r in records(owner))
            # Another GPU module is admitted by capacity despite evaluator
            # inference being active and the legacy compute cap being one.
            with client.lease(backend="mutator", scratch_bytes=100, timeout=.5):
                assert any(r["evaluator_inference"] for r in records(owner))
            completion.set_result(None)
            assert finalized.wait(5)
            assert client.snapshot()["reserved_vram_bytes"] == 300
            assert not result.done()
        finally:
            if not completion.done():
                completion.set_result(None)
            release_finalize.set()
        response = result.result(5)
    assert response["ok"] and response["result"] == {"value": "original"}
    assert response["random_state"] == state
    assert seen == ["prepare", "infer", "finalize"]
    assert set(response["runtime"]["timings"]) == {"prepare", "infer", "finalize"}
    assert client.snapshot()["active"] == 0
    assert client.snapshot()["reserved_vram_bytes"] == 100


def test_capacity_cannot_be_borrowed_by_another_identity_or_released_early(broker):
    owner, client = broker
    memory = client.lease(backend="b", identity="i", scratch_bytes=200,
                          capacity_only=True).__enter__()
    with pytest.raises(AdmissionError, match="another request"):
        with client.lease(backend="b", identity="other",
                          reservation_token=memory.token, timeout=.1):
            pass
    operation = client.lease(backend="b", identity="i",
                              reservation_token=memory.token,
                              evaluator_inference=True).__enter__()
    try:
        with pytest.raises(AdmissionTimeout):
            with client.lease(backend="b", identity="i", phase="finalize",
                              reservation_token=memory.token, timeout=.05):
                pass
        with pytest.raises(AdmissionError, match="lease retained"):
            memory.finish()
        assert any(r["capacity_only"] for r in records(owner))
    finally:
        operation.finish()
        memory.finish()
    assert client.snapshot()["active"] == 0


def test_unknown_memory_stays_conservative_without_consuming_inference(broker):
    owner, client = broker
    with client.lease(backend="b", identity="i", capacity_only=True) as memory:
        assert not any(r["evaluator_inference"] for r in records(owner))
        with client.lease(backend="b", identity="i", reservation_token=memory.token,
                          evaluator_inference=True, timeout=.1):
            with pytest.raises(AdmissionTimeout):
                with client.lease(backend="mutator", scratch_bytes=1, timeout=.05):
                    pass


@pytest.mark.parametrize("failure", ["prepare", "finalize"])
def test_known_cpu_failures_release_capacity_once(broker, failure):
    owner, client = broker
    seen = []

    def phase(name):
        def run(value):
            seen.append(name)
            if name == failure:
                raise ValueError("deliberate " + name)
            return value
        return run

    handler = execution.kernel_handler({}, lambda _: (
        execution.EvaluatorStages(phase("prepare"), phase("infer"), phase("finalize"),
                                  resource="gpu", prepare_resource="cpu",
                                  finalize_resource="cpu"), {}),
        lifecycle=dict(backend="cpu-stage", device="cuda:0",
                       resident_bytes=100, scratch_bytes=200))
    result = handler(dict(operation="infer", payload=1,
                          random_state=execution.capture_random_state()))
    assert not result["ok"] and result["error"] == "deliberate " + failure
    assert seen == ["prepare", "infer", "finalize"][:seen.index(failure) + 1]
    assert client.snapshot()["active"] == 0
    assert result["runtime"]["lifecycle"]["state"] == "ready"


def test_pending_failure_keeps_unknown_device_fenced(broker):
    owner, client = broker
    completion = Future()
    completion.set_exception(RuntimeError("device outcome unknown"))
    finalized = []
    handler = execution.kernel_handler({}, lambda _: (
        execution.EvaluatorStages(lambda x: x, lambda x: Pending(x, completion),
                                  lambda x: finalized.append(x)), {}),
        lifecycle=dict(backend="failure-stage", device="cuda:0",
                       resident_bytes=100, scratch_bytes=200))
    result = handler(dict(operation="infer", payload=1,
                          random_state=execution.capture_random_state()))
    assert not result["ok"] and not finalized
    assert result["runtime"]["lifecycle"]["state"] == "quarantined"
    assert any(r.get("evaluator_inference") for r in records(owner))
    assert any(r.get("capacity_only") for r in records(owner))
    with pytest.raises(RuntimeError, match="quarantined"):
        handler(dict(operation="infer", payload=2,
                     random_state=execution.capture_random_state()))
    # This test's synthetic Future has no actual device work. Explicitly finish
    # only for test cleanup; real unknown device work requires owner exit.
    for record in sorted(records(owner), key=lambda r: r["capacity_only"]):
        client._rpc(dict(op="finish", token=record["token"]))


def test_rejected_inference_admission_releases_memory_and_owner_can_continue(broker, monkeypatch):
    owner, client = broker
    original_lease = client.lease
    reject = [True]
    executed = []

    def lease(**kwargs):
        if kwargs["phase"] == "infer" and reject[0]:
            reject[0] = False
            raise AdmissionError("pre-dispatch rejection")
        return original_lease(**kwargs)

    monkeypatch.setattr(client, "lease", lease)
    monkeypatch.setattr(execution, "_resource_client", lambda: client)
    handler = execution.kernel_handler({}, lambda _: (
        execution.EvaluatorStages(lambda x: x, lambda x: executed.append(x) or x), {}),
        lifecycle=dict(backend="rejected-stage", device="cuda:0",
                       resident_bytes=100, scratch_bytes=200))
    request = dict(operation="infer", payload=1,
                   random_state=execution.capture_random_state())
    result = handler(request)
    assert not result["ok"] and not executed
    assert result["runtime"]["lifecycle"]["state"] == "ready"
    assert client.snapshot()["active"] == 0
    assert handler(request)["ok"] and executed == [1]


def test_cpu_memory_survives_between_stage_admissions(broker, monkeypatch):
    owner, client = broker
    original_lease = client.lease
    observations = []

    def lease(**kwargs):
        if kwargs["phase"] in {"infer", "finalize"}:
            snapshot = client.snapshot()
            observations.append((snapshot["transient_ram_bytes"],
                                 snapshot["output_bytes"]))
        return original_lease(**kwargs)

    monkeypatch.setattr(client, "lease", lease)
    monkeypatch.setattr(execution, "_resource_client", lambda: client)
    handler = execution.kernel_handler({}, lambda _: (
        execution.EvaluatorStages(lambda x: bytearray(200), lambda x: x, lambda x: len(x),
                                  resource="cpu", prepare_resource="cpu",
                                  finalize_resource="cpu"), {}),
        lifecycle=dict(backend="host-stage", resource="cpu",
                       scratch_ram_bytes=200, output_bytes=20))
    response = handler(dict(operation="infer", payload=None,
                            random_state=execution.capture_random_state()))
    assert response["ok"] and response["result"] == 200
    assert observations == [(200, 20), (200, 20)]
    assert client.snapshot()["active"] == 0


@pytest.mark.parametrize("adaptive_enabled", [False, True])
def test_request_local_infer_gate_completion_and_busy_lifetime(monkeypatch, adaptive_enabled):
    """Actual Pending calls overlap only after a measured +1 decision."""
    from astrolabe.runtime import evaluation

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(execution, "_resource_client", lambda: None)
    now = [0.0]
    controller = evaluation.EvaluatorInferController(
        dict(enabled=True, max_concurrency=3, memory_per_request_bytes=100,
             memory_reserve_bytes=10), clock=lambda: now[0])
    monkeypatch.setattr(controller, "start", lambda: None)
    monkeypatch.setattr(evaluation, "EvaluatorInferController", lambda *args, **kw: controller)
    dispatched = {i: threading.Event() for i in (1, 2)}
    completions = {i: Future() for i in (1, 2)}
    finalizing = threading.Event()
    finish_finalize = threading.Event()

    def execute(value):
        if value == 0:
            return value
        dispatched[value].set()
        return Pending(value, completions[value], keepalive=value)

    def finalize(value):
        if value == 1:
            finalizing.set()
            assert finish_finalize.wait(5)
        return value

    handle = execution.kernel_handler({}, lambda _: (
        execution.EvaluatorStages(lambda x: x, execute, finalize,
                                  prepare_resource="cpu", finalize_resource="cpu",
                                  request_local_rng=True, adaptive=dict(enabled=True)), {}),
        lifecycle=dict(backend="alphafold3", device="cuda:0"))
    # Initialization may own default RNG; request execution must not touch it.
    monkeypatch.setattr(execution, "_kernel_random_scope", lambda *a, **k: pytest.fail("global RNG scope"))
    monkeypatch.setattr(execution, "capture_random_state", lambda: pytest.fail("global RNG capture"))
    def request(value):
        return dict(operation="infer", payload=value, random_state={"opaque": value})

    assert handle(request(0))["ok"]
    handle.adaptive_control(enabled=adaptive_enabled, reset=False)
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(handle, request(1))
        second = None
        try:
            assert dispatched[1].wait(5)
            second = pool.submit(handle, request(2))
            with controller._cv:
                assert controller._cv.wait_for(lambda: controller.ready == 1, timeout=5)
            assert not dispatched[2].is_set()
            now[0] = 10
            controller.observe([dict(index=0, uuid="GPU-infer-test", total_bytes=1000,
                                     free_bytes=900, utilization=20, processes=[])], sampled_at=10)
            if adaptive_enabled:
                assert dispatched[2].wait(5), "second real model call did not dispatch"
                assert not completions[1].done()
                assert controller.snapshot()["active"] == 2
            else:
                assert not dispatched[2].is_set()
            completions[1].set_result(None)
            assert finalizing.wait(5)
            assert dispatched[2].wait(5), "finalize must not hold an infer slot"
            with pytest.raises(RuntimeError, match="drain"):
                handle.adaptive_control(enabled=False, reset=True)
            finish_finalize.set()
            response = first.result(5)
            assert response["ok"] and response["random_state"] == {"opaque": 1}
            snapshot = handle.describe_snapshot()["runtime_lifecycle"]
            assert snapshot["state"] == "busy" and snapshot["active_requests"] == 1
        finally:
            finish_finalize.set()
            for completion in completions.values():
                if not completion.done():
                    completion.set_result(None)
        assert second.result(5)["ok"]
    assert handle.describe_snapshot()["runtime_lifecycle"]["state"] == "ready"
    handle.adaptive_control(enabled=True, reset=True)
    assert controller.snapshot()["limit"] == 1 and controller.snapshot()["completed"] == 0
    handle.close()


def test_adaptive_decisions_require_backlog_freshness_and_memory(monkeypatch):
    from astrolabe.runtime.evaluation import EvaluatorInferController

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    now = [0.0]
    events = []
    controller = EvaluatorInferController(
        dict(enabled=True, max_concurrency=3, memory_per_request_bytes=100,
             memory_reserve_bytes=10), clock=lambda: now[0],
        emit=lambda event, **fields: events.append((event, fields)))
    row = dict(index=0, uuid="GPU-control-test", total_bytes=1000,
               free_bytes=900, utilization=20, processes=[])
    def window(**changes):
        now[0] += 10
        controller.observe([dict(row, **changes)], sampled_at=now[0])
        return controller.snapshot()["last_decision"]["reason"]

    assert window() == "starved"
    controller.ready = 1  # Prepared backlog, no model completion yet.
    assert window() == "warming_up"
    controller.ready = 0
    with controller.inference_slot():
        pass
    controller.ready = 1
    now[0] += 10
    controller.observe([row], sampled_at=now[0] - 3)
    assert controller.limit == 1
    assert controller.snapshot()["last_decision"]["reason"] == "telemetry_unavailable"
    assert window(utilization=95) == "hold"
    assert window(free_bytes=109) == "memory_pressure"
    assert window() == "increase" and controller.limit == 2
    assert window() == "increase" and controller.limit == 3
    assert window() == "hold" and controller.limit == 3
    controller.completed += 10
    assert window() == "hold"
    controller.completed += 1
    assert window() == "throughput_drop" and controller.limit == 2
    controller.ready = 0
    assert window() == "starved"
    assert events and all({"utilization", "free_bytes", "ready", "active", "completed", "limit"} <= set(e[1])
                          for e in events)


def test_adaptive_oom_quarantines_owner_without_retry(monkeypatch):
    from astrolabe.runtime.evaluation import EvaluatorInferController

    monkeypatch.setattr(execution, "_resource_client", lambda: None)
    monkeypatch.setattr(EvaluatorInferController, "start", lambda self: None)
    calls = []
    def execute(value):
        calls.append(value)
        raise RuntimeError("RESOURCE_EXHAUSTED: out of memory")
    handle = execution.kernel_handler({}, lambda _: (
        execution.EvaluatorStages(lambda x: x, execute, request_local_rng=True,
                                  adaptive=dict(enabled=True)), {}),
        lifecycle=dict(backend="alphafold3", device="cuda:0"))
    request = dict(operation="infer", payload=1, random_state={})
    assert not handle(request)["ok"]
    assert calls == [1] and handle.describe_snapshot()["adaptive"]["quarantined"]
    with pytest.raises(RuntimeError, match="quarantined"):
        handle(request)
    with pytest.raises(RuntimeError, match="quarantined"):
        handle.adaptive_control(enabled=True, reset=True)
    handle.close()


def test_adaptive_owner_does_not_override_uncertified_broker_gate(broker, monkeypatch):
    from astrolabe.runtime import evaluation

    owner, client = broker
    monkeypatch.setattr(execution, "_resource_client", lambda: client)
    now = [0.0]
    controller = evaluation.EvaluatorInferController(
        dict(enabled=True, max_concurrency=2, memory_per_request_bytes=100,
             memory_reserve_bytes=10), clock=lambda: now[0])
    monkeypatch.setattr(controller, "start", lambda: None)
    monkeypatch.setattr(evaluation, "EvaluatorInferController", lambda *a, **kw: controller)
    dispatched = {i: threading.Event() for i in (1, 2)}
    completions = {i: Future() for i in (1, 2)}
    def execute(value):
        if value == 0:
            return value
        dispatched[value].set()
        return Pending(value, completions[value])
    handle = execution.kernel_handler({}, lambda _: (
        execution.EvaluatorStages(lambda x: x, execute, prepare_resource="cpu",
                                  finalize_resource="cpu", request_local_rng=True,
                                  adaptive=dict(enabled=True)), {}),
        lifecycle=dict(backend="alphafold3", device="cuda:0", resident_bytes=100,
                       scratch_bytes=200, concurrency_certified=False))
    def request(value):
        return dict(operation="infer", payload=value, random_state={})
    assert handle(request(0))["ok"]
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(handle, request(1))
        second = None
        try:
            assert dispatched[1].wait(5)
            second = pool.submit(handle, request(2))
            with controller._cv:
                assert controller._cv.wait_for(lambda: controller.ready == 1, timeout=5)
            now[0] = 10
            controller.observe([dict(index=0, uuid="GPU-test", total_bytes=1000,
                                     free_bytes=900, utilization=10, processes=[])], sampled_at=10)
            assert controller.limit == 2
            with owner._cv:
                assert owner._cv.wait_for(
                    lambda: any(r.get("evaluator_inference") for r in owner._pending), timeout=5)
            assert not dispatched[2].is_set()
            assert sum(r.get("evaluator_inference", False) for r in records(owner)) == 1
            completions[1].set_result(None)
            assert first.result(5)["ok"]
            assert dispatched[2].wait(5)
        finally:
            for completion in completions.values():
                if not completion.done():
                    completion.set_result(None)
        assert second.result(5)["ok"]
    handle.close()
