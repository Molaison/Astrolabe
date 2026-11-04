"""Real broker/ledger admission with a controlled, non-executing worker boundary.

No UNIX transport, process observation or GPU is simulated as working here.
The client raises at dispatch, so these tests certify admission behavior only.
"""
from types import SimpleNamespace
import threading
import time

import pytest

from astrolabe.runtime import broker_state
from astrolabe.runtime.client import OutcomeUnknown
from astrolabe.runtime.contracts import ResourceLimits, request_metadata


class DispatchReached(Exception):
    pass


class ControlledWorker:
    def __init__(self):
        self.hello = {"worker_epoch": "controlled-worker", "capabilities": {}}
        self.lifecycle = {}
        self.last_metrics = {}
        self.dispatches = []
        self.error = DispatchReached("controlled dispatch boundary; no scientific execution")

    def call(self, payload, **kwargs):
        self.dispatches.append(kwargs)
        raise self.error

    def close(self):
        # Controlled post-dispatch cleanup observation; no process/GPU exists.
        self.lifecycle["release_confirmed"] = True


@pytest.fixture
def state(monkeypatch, tmp_path):
    monkeypatch.delenv("ASTROLABE_EXECUTION_ARTIFACT_ROOT", raising=False)
    monkeypatch.setenv("ASTROLABE_RUNTIME_TRACE", str(tmp_path / "trace.jsonl"))
    result = broker_state.State("test", 1, 1, limits=ResourceLimits(
        cpu_slots=1, ram_bytes=4096, prepared_bytes=4096, output_bytes=1024, pending_items=2))
    result.workers["worker"] = {
        "client": ControlledWorker(), "device": "cpu", "clients": {"client": None},
        "busy": 0, "shapes": set(), "cohort": "controlled", "call_lock": threading.Lock(),
        "retained": {}, "delegated": None, "last_used": time.monotonic()}
    request_metadata({})
    yield result
    for token in tuple(result.ledger.pending):
        result.ledger.release(token)
    result.close()


def request(*, timeout=1.0, **overrides):
    return {"epoch": "client", "request_id": "request", "handle": "worker", "timeout": timeout,
            "payload": {"value": 1}, "runtime": {"resources": {"ram_bytes": 0, "output_bytes": 0}},
            **overrides}


def assert_request_released(state, *, retained_pending=()):
    # The ledger token is private to the broker. Only the explicit blockers
    # and an intentionally retained prepared request may remain reserved.
    blockers = {"compute-blocker", "pending-blocker", "blocker"}
    assert not (set(state.ledger.pending) - blockers - set(retained_pending))
    assert not (set(state.ledger.active) - blockers)
    assert state.metrics["waiting"] == 0
    assert state.workers["worker"]["busy"] == 0
    assert state.workers["worker"]["delegated"] is None


def test_pending_and_compute_waits_share_one_total_deadline(state, monkeypatch):
    ledger = state.ledger
    zero = {"ram_bytes": 0, "output_bytes": 0}
    ledger.admit_pending("compute-blocker", zero)
    ledger.acquire_compute("compute-blocker", "cpu")
    ledger.admit_pending("pending-blocker", zero)
    entered, admitted, cancelled = threading.Event(), threading.Event(), threading.Event()
    original = ledger.admit_pending

    def observe_admission(token, resources, **kwargs):
        if token not in {"compute-blocker", "pending-blocker", "blocker"}:
            entered.set()
        original(token, resources, **kwargs)
        if token not in {"compute-blocker", "pending-blocker", "blocker"}:
            admitted.set()

    monkeypatch.setattr(ledger, "admit_pending", observe_admission)

    def release_in_order():
        if entered.wait(2) and not cancelled.wait(.6):
            ledger.release("pending-blocker")
            if admitted.wait(2) and not cancelled.wait(.7):
                ledger.release("compute-blocker")

    thread = threading.Thread(target=release_in_order, daemon=True)
    thread.start()
    try:
        with pytest.raises(TimeoutError):
            state.call(request())
        assert admitted.is_set(), "probe must actually pass through both admission queues"
        assert not state.workers["worker"]["client"].dispatches
        assert_request_released(state)
        assert not state.poisoned
        with pytest.raises(ValueError, match="duplicate request"):
            state.call(request())
    finally:
        cancelled.set()
        ledger.release("pending-blocker")
        ledger.release("compute-blocker")
        thread.join(timeout=3)
    assert not thread.is_alive()


def test_expiration_after_compute_admission_never_gets_a_fresh_millisecond(state, monkeypatch):
    # State and ledger share an absolute clock. Advance it at the observable
    # dispatch boundary; this deterministic test makes no wall-time claim.
    from astrolabe.runtime import resources as resource_module

    clock = [100.0]
    timer = SimpleNamespace(monotonic=lambda: clock[0])
    monkeypatch.setattr(broker_state, "time", timer)
    monkeypatch.setattr(resource_module, "time", timer)
    write = state.events.write

    def expire_before_dispatch(event, **fields):
        write(event, **fields)
        if event == "compute_start":
            clock[0] += 2.0

    monkeypatch.setattr(state.events, "write", expire_before_dispatch)
    with pytest.raises(TimeoutError, match="before submission"):
        state.call(request())
    assert not state.workers["worker"]["client"].dispatches
    assert_request_released(state)
    assert not state.poisoned
    assert state.ledger.snapshot()["cpu_slots"] == 0
    assert state.ledger.snapshot()["prepared_bytes"] == 0


@pytest.mark.parametrize("timeout", [None, 1.0])
def test_live_deadline_preserves_dispatch_identity_and_unbounded_calls(state, timeout):
    with pytest.raises(DispatchReached):
        state.call(request(timeout=timeout))
    [sent] = state.workers["worker"]["client"].dispatches
    assert sent["timeout"] is None if timeout is None else 0 < sent["timeout"] <= timeout
    assert sent["request_id"]
    assert_request_released(state)


def test_expired_staged_execute_preserves_prepared_reservation_before_dispatch(state):
    worker = state.workers["worker"]
    worker["client"].hello["capabilities"]["request_staging"] = True
    resources = {"prepared_bytes": 32, "ram_bytes": 0, "output_bytes": 0}
    state.ledger.admit_pending("prepared", resources)
    worker["retained"][("client", "scientific")] = {
        "token": "prepared", "resources": resources, "state": "prepared"}
    state.ledger.admit_pending("blocker", {"ram_bytes": 0, "output_bytes": 0})
    state.ledger.acquire_compute("blocker", "cpu")
    with pytest.raises(TimeoutError):
        state.call(request(timeout=.03, payload={"method": "execute"},
                           runtime={"scientific_request_id": "scientific"}))
    assert "prepared" in state.ledger.pending
    assert worker["retained"][("client", "scientific")] == {
        "token": "prepared", "resources": resources, "state": "prepared"}
    assert not worker["client"].dispatches
    assert_request_released(state, retained_pending={"prepared"})


def test_unknown_dispatched_outcome_fences_its_cpu_worker_and_is_not_replayed(state):
    client = state.workers["worker"]["client"]
    client.error = OutcomeUnknown("controlled uncertain transport outcome")
    with pytest.raises(OutcomeUnknown):
        state.call(request())
    assert len(client.dispatches) == 1
    assert state.workers["worker"]["fenced"] and state.retain_artifacts
    assert not state.poisoned  # Independent CPU workers do not share a device fence.
    assert_request_released(state)
    with pytest.raises(ValueError, match="duplicate request"):
        state.call(request())
    assert len(client.dispatches) == 1


@pytest.mark.parametrize("waiter_times_out", [False, True], ids=["control", "waiter-timeout"])
def test_waiter_cannot_clear_the_running_request_delegation(state, monkeypatch, waiter_times_out):
    entered, release = threading.Event(), threading.Event()
    worker = state.workers["worker"]
    outcome = {}

    def controlled_call(payload, **kwargs):
        entered.set()
        assert release.wait(2), "test controller must release the controlled worker"
        return {"ok": True}

    monkeypatch.setattr(worker["client"], "call", controlled_call)
    # This test has no resident process. Only the post-response observations
    # are substituted; State/ledger locking, admission and cleanup stay real.
    monkeypatch.setattr(state.ledger, "update_resident", lambda *args, **kwargs: None)
    monkeypatch.setattr(state.ledger, "observe_resident", lambda *args, **kwargs: None)

    def first_call():
        try:
            outcome["result"] = state.call(request(request_id="first"))
        except BaseException as exc:
            outcome["error"] = exc

    thread = threading.Thread(target=first_call)
    thread.start()
    try:
        assert entered.wait(1)
        delegated = worker["delegated"]
        assert delegated is not None
        if waiter_times_out:
            with pytest.raises(TimeoutError):
                state.call(request(timeout=.03))
        assert worker["delegated"] is delegated
        assert worker["busy"] == 1 and state.metrics["waiting"] == 0
        assert len(state.ledger.active) == 1
        assert not state.poisoned
    finally:
        release.set()
        thread.join(timeout=3)
    assert not thread.is_alive()
    assert "error" not in outcome, repr(outcome.get("error"))
    assert outcome["result"]["result"] == {"ok": True}
    assert worker["delegated"] is None and worker["busy"] == 0
    assert state.ledger.snapshot()["pending"] == 0
    assert state.metrics["completed"] == 1


@pytest.mark.parametrize("timeout,expired", [(.03, True), (1.0, False)], ids=["expired", "live"])
def test_entry_lock_wait_consumes_admission_budget(state, monkeypatch, timeout, expired):
    entered = threading.Event()
    errors = []
    track = state._track_request

    def observe_track(value, *args, **kwargs):
        entered.set()
        return track(value, *args, **kwargs)

    monkeypatch.setattr(state, "_track_request", observe_track)

    def call():
        try:
            state.call(request(timeout=timeout))
        except BaseException as exc:
            errors.append(exc)

    with state.lock:
        thread = threading.Thread(target=call)
        thread.start()
        assert entered.wait(1)
        threading.Event().wait(.12)
    thread.join(timeout=2)
    assert not thread.is_alive()
    [error] = errors
    if expired:
        assert isinstance(error, TimeoutError), repr(error)
        assert not state.workers["worker"]["client"].dispatches
    else:
        assert isinstance(error, DispatchReached), repr(error)
        [dispatch] = state.workers["worker"]["client"].dispatches
        assert 0 < dispatch["timeout"] < timeout - .1
    assert_request_released(state)
    assert not state.poisoned
    # An entry lock timeout can occur before the request is recorded. Once
    # recorded, the same ID is still rejected; the runtime never auto-replays.
    if not expired:
        with pytest.raises(ValueError, match="duplicate request"):
            state.call(request(timeout=timeout))


@pytest.mark.parametrize("waiting_retired", [False, True], ids=["before-transition", "after-transition"])
def test_expiry_retires_waiting_count_once_across_admission_transition(state, monkeypatch, waiting_retired):
    """Real admission plus a controlled scheduling delay; no worker executes."""
    observed = {}

    def delay_after_real_admission():
        observed["waiting"] = state.metrics["waiting"]
        observed["active"] = len(state.ledger.active)
        time.sleep(.2)

    if waiting_retired:
        write = state.events.write

        def delayed_event(event, **fields):
            write(event, **fields)
            if event == "compute_start":
                delay_after_real_admission()

        monkeypatch.setattr(state.events, "write", delayed_event)
    else:
        acquire = state.ledger.acquire_compute

        def delayed_acquire(*args, **kwargs):
            acquire(*args, **kwargs)
            delay_after_real_admission()

        monkeypatch.setattr(state.ledger, "acquire_compute", delayed_acquire)

    with pytest.raises(TimeoutError):
        state.call(request(timeout=.1))
    assert observed == {"waiting": 0 if waiting_retired else 1, "active": 1}
    assert not state.workers["worker"]["client"].dispatches
    assert_request_released(state)
    assert state.ledger.snapshot()["cpu_slots"] == 0
    assert not state.poisoned
