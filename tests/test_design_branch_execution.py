from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import threading

import pytest

from astrolabe.design.branch_execution import BranchExecutor, gather_ordered
from astrolabe.design.branch_types import (
    BranchAction,
    BranchEvent,
    BranchJob,
    BranchMember,
    BranchProposal,
    BranchTerminal,
    BranchingState,
)
from astrolabe.design.types import Blob, Key


def _blob(value: str) -> Blob:
    return Blob(Key(value), "branch-execution-test/v1")


def _member(identifier: int) -> BranchMember:
    return BranchMember(
        id=identifier,
        key=f"member-{identifier}",
        value=_blob(f"member-{identifier}"),
        score=float(identifier),
    )


def _job(
    phase: str,
    identifier: int,
    *,
    seed: int = 7,
    slot: int = 0,
) -> BranchJob:
    parent = _member(0)
    proposal = (
        BranchProposal(
            id=identifier,
            parent_id=parent.id,
            key=f"proposal-{identifier}",
            value=_blob(f"proposal-{identifier}"),
        )
        if phase == "evaluate"
        else None
    )
    return BranchJob(
        phase=phase,
        id=identifier,
        generation=1,
        seed=seed,
        parent=parent,
        proposal=proposal,
        slot=slot,
    )


def _terminal(job: BranchJob) -> BranchTerminal:
    value = (
        BranchProposal(
            id=job.id,
            parent_id=0,
            key=f"result-{job.id}",
            value=_blob(f"result-{job.id}"),
        )
        if job.phase == "propose"
        else _member(job.id)
    )
    return BranchTerminal(job=job, status="completed", value=value)


class _ImmediatePool:
    def __init__(self, transform=None) -> None:
        self.calls: list[BranchJob] = []
        self._transform = transform

    def submit(self, worker, job):
        self.calls.append(job)
        future = Future()
        future.set_result(worker(job))
        return future


class _RecordingFuture(Future):
    def __init__(self, events: list[str], label: str) -> None:
        super().__init__()
        self._events = events
        self._label = label

    def result(self, timeout=None):
        self._events.append(f"fetch:{self._label}")
        return super().result(timeout)


def test_gather_ordered_refills_freed_slots_and_preserves_input_order() -> None:
    released = threading.Event()
    fourth_started = threading.Event()
    lock = threading.Lock()
    active = 0
    maximum = 0

    def worker(task):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        try:
            if task["id"] == 0:
                assert fourth_started.wait(5)
                released.set()
            if task["id"] == 3:
                assert not released.is_set()
                fourth_started.set()
            return dict(task, payload=f"payload-{task['id']}")
        finally:
            with lock:
                active -= 1

    tasks = tuple(dict(id=index) for index in range(7))
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = gather_ordered(
            pool,
            tasks,
            3,
            worker,
            lambda task: task["id"],
        )

    assert results == [
        dict(task, payload=f"payload-{task['id']}") for task in tasks
    ]
    assert fourth_started.is_set()
    assert maximum <= 3


def test_gather_ordered_rejects_duplicates_before_submission() -> None:
    class Pool:
        def __init__(self) -> None:
            self.calls = []

        def submit(self, worker, task):
            self.calls.append(task)
            future = Future()
            future.set_result(task)
            return future

    pool = Pool()
    with pytest.raises(ValueError, match="duplicate task identity"):
        gather_ordered(
            pool,
            ({"id": 1}, {"id": 1}),
            1,
            lambda task: task,
            lambda task: task["id"],
        )
    assert pool.calls == []


def test_gather_ordered_checks_whole_ready_group_before_refill() -> None:
    class Pool:
        def __init__(self) -> None:
            self.calls = []
            self.result_calls = []

        def submit(self, worker, task):
            self.calls.append(task["id"])
            pool = self

            class ObservedFuture(Future):
                def result(self, timeout=None):
                    pool.result_calls.append(task["id"])
                    return super().result(timeout)

            future = ObservedFuture()
            if task["id"] != 0:
                future.set_exception(RuntimeError("ready group failed"))
            else:
                future.set_result(task)
            return future

    pool = Pool()
    with pytest.raises(RuntimeError, match="ready group failed"):
        gather_ordered(
            pool,
            ({"id": 0}, {"id": 1}, {"id": 2}),
            3,
            lambda task: task,
            lambda task: task["id"],
        )
    assert pool.calls == [0, 1, 2]
    assert pool.result_calls == [0, 1, 2]


def test_transport_helpers_require_concrete_future() -> None:
    class Pool:
        def submit(self, worker, job):
            return {"result": job}

    with pytest.raises(TypeError, match="concrete Future"):
        BranchExecutor(Pool(), worker=lambda job: _terminal(job)).perform(
            BranchAction(
                BranchingState(),
                jobs=(_job("propose", 1),),
                wait="none",
            )
        )

    with pytest.raises(TypeError, match="concrete Future"):
        gather_ordered(
            Pool(),
            ({"id": 1},),
            1,
            lambda task: task,
            lambda task: task["id"],
        )


def test_branch_executor_submits_each_full_identity_once_and_orders_results() -> None:
    pool = _ImmediatePool()
    executor = BranchExecutor(pool, worker=_terminal)
    first = _job("propose", 2)
    second = _job("evaluate", 1)

    assert executor.request_type is BranchAction
    assert executor.result_type is BranchEvent
    assert executor.perform(
        BranchAction(
            BranchingState(),
            jobs=(first, second),
            wait="none",
        )
    ) == BranchEvent(())
    assert pool.calls == [first, second]
    assert tuple(executor.one_pending.values()) == (first, second)

    event = executor.perform(BranchAction(BranchingState(), wait="any"))

    assert event.terminals == (_terminal(second), _terminal(first))
    assert executor.one_pending == {}
    assert executor.observed == {
        second.identity: _terminal(second),
        first.identity: _terminal(first),
    }

    repeated = executor.perform(
        BranchAction(
            BranchingState(),
            jobs=(first,),
            wait="none",
        )
    )
    assert repeated == BranchEvent(())
    assert pool.calls == [first, second]


def test_restored_job_is_admitted_and_filtered_from_fresh_submission() -> None:
    submitted: list[BranchJob] = []

    def submit(pool, job):
        submitted.append(job)
        future = Future()
        future.set_result(_terminal(job))
        return future

    executor = BranchExecutor(object(), submit=submit)
    restored_job = _job("evaluate", 1)
    restored_future = Future()
    restored_future.set_result(_terminal(restored_job))
    executor.restore(restored_job, restored_future)
    executor.restore(restored_job, restored_future)
    fresh_job = _job("propose", 2)

    event = executor.perform(
        BranchAction(
            BranchingState(),
            jobs=(restored_job, fresh_job),
            wait="all",
        )
    )

    assert submitted == [fresh_job]
    assert event.terminals == (_terminal(restored_job), _terminal(fresh_job))
    assert executor.one_pending == {}


def test_restore_rejects_same_job_identity_conflicts() -> None:
    executor = BranchExecutor(
        object(),
        worker=_terminal,
    )
    job = _job("evaluate", 1)
    future = Future()
    executor.restore(job, future)

    with pytest.raises(ValueError, match="conflicts"):
        executor.restore(job, Future())

    differently_seeded = _job("evaluate", 1, seed=8)
    with pytest.raises(ValueError, match="conflicts"):
        executor.restore(differently_seeded, future)
    assert executor.one_pending == {future: job}


def test_ordered_any_waits_only_on_phase_heads_and_publishes_ready_prefix(monkeypatch) -> None:
    import astrolabe.design.branch_execution as execution

    jobs = (_job("evaluate", 1), _job("evaluate", 2),
            _job("propose", 1), _job("propose", 2))
    futures = [Future() for _ in jobs]
    futures[1].set_result(_terminal(jobs[1]))
    futures[2].set_result(_terminal(jobs[2]))
    published = []
    executor = BranchExecutor(
        object(), worker=_terminal, ordered_results=True,
        resolve=lambda job, raw: published.append(job.identity) or raw,
    )
    for job, future in zip(jobs, futures):
        executor.restore(job, future)
    real_wait = execution.wait
    waits = []

    def wait_on_heads(waiting, *, return_when):
        waits.append(tuple(waiting))
        if len(waits) == 1:
            assert tuple(waiting) == (futures[0], futures[2])
        else:
            # The already-completed evaluation suffix must never wake a wait.
            assert tuple(waiting) == (futures[0], futures[3])
            futures[0].set_result(_terminal(jobs[0]))
        return real_wait(waiting, return_when=return_when)

    monkeypatch.setattr(execution, "wait", wait_on_heads)
    action = BranchAction(BranchingState(), wait="any")
    assert executor.perform(action).terminals == (_terminal(jobs[2]),)
    assert executor.perform(action).terminals == (_terminal(jobs[0]), _terminal(jobs[1]))
    assert published == ["propose/1", "evaluate/1", "evaluate/2"]
    assert tuple(executor.one_pending.values()) == (jobs[3],)


def test_resolver_runs_on_caller_thread_after_all_raw_results_fetch() -> None:
    events: list[str] = []
    resolve_threads: list[int] = []
    jobs = (_job("propose", 2), _job("evaluate", 1))

    def submit(pool, job):
        future = _RecordingFuture(events, job.identity)
        future.set_result({"owner_result": job.identity})
        return future

    def resolve(job, raw):
        events.append(f"resolve:{job.identity}")
        resolve_threads.append(threading.get_ident())
        assert raw == {"owner_result": job.identity}
        return _terminal(job)

    executor = BranchExecutor(object(), submit=submit, resolve=resolve)
    caller_thread = threading.get_ident()

    event = executor.perform(
        BranchAction(
            BranchingState(),
            jobs=jobs,
            wait="all",
        )
    )

    assert events == [
        "fetch:evaluate/1",
        "fetch:propose/2",
        "resolve:evaluate/1",
        "resolve:propose/2",
    ]
    assert resolve_threads == [caller_thread, caller_thread]
    assert event.terminals == (_terminal(jobs[1]), _terminal(jobs[0]))
    assert executor.one_pending == {}


def test_all_ready_errors_call_on_error_before_raise_without_resolve() -> None:
    jobs = (_job("propose", 2), _job("evaluate", 3), _job("evaluate", 1))
    submitted: list[str] = []
    resolver_calls = []
    callback_calls: list[tuple[str, str, int]] = []

    class KnownTerminalFailure(RuntimeError):
        pass

    class UnknownFailure(RuntimeError):
        pass

    def submit(pool, job):
        submitted.append(job.identity)
        future = Future()
        if job.identity == "evaluate/1":
            future.set_exception(KnownTerminalFailure("known"))
        elif job.identity == "evaluate/3":
            future.set_exception(UnknownFailure("unknown"))
        else:
            future.set_result({"owner_result": job.identity})
        return future

    def on_error(job, error):
        status = (
            "unknown" if isinstance(error, UnknownFailure) else "failed"
        )
        callback_calls.append(
            (job.identity, status, threading.get_ident())
        )

    def resolve(job, raw):
        resolver_calls.append((job, raw))
        return _terminal(job)

    executor = BranchExecutor(
        object(),
        submit=submit,
        resolve=resolve,
        on_error=on_error,
    )
    caller_thread = threading.get_ident()

    with pytest.raises(KnownTerminalFailure, match="known"):
        executor.perform(
            BranchAction(
                BranchingState(),
                jobs=jobs,
                wait="all",
            )
        )

    assert callback_calls == [
        ("evaluate/1", "failed", caller_thread),
        ("evaluate/3", "unknown", caller_thread),
    ]
    assert resolver_calls == []
    assert executor.observed == {}
    assert len(executor.one_pending) == 3
    assert submitted == ["propose/2", "evaluate/3", "evaluate/1"]


def test_raw_group_failure_prevents_resolution_and_retains_pending() -> None:
    submitted = []

    def submit(pool, job):
        submitted.append(job.identity)
        future = Future()
        if job.identity == "propose/2":
            future.set_exception(RuntimeError("raw owner failure"))
        else:
            future.set_result({"owner_result": job.identity})
        return future

    resolver_calls = []

    def resolve(job, raw):
        resolver_calls.append((job, raw))
        return _terminal(job)

    executor = BranchExecutor(object(), submit=submit, resolve=resolve)
    jobs = (_job("evaluate", 1), _job("propose", 2))
    with pytest.raises(RuntimeError, match="raw owner failure"):
        executor.perform(
            BranchAction(
                BranchingState(),
                jobs=jobs,
                wait="all",
            )
        )

    assert resolver_calls == []
    assert len(executor.one_pending) == 2
    assert executor.observed == {}

    with pytest.raises(RuntimeError, match="raw owner failure"):
        executor.perform(BranchAction(BranchingState(), wait="all"))
    assert submitted == ["evaluate/1", "propose/2"]


def test_resolved_terminal_must_match_complete_admitted_job() -> None:
    admitted = _job("evaluate", 1, seed=7, slot=0)
    differently_seeded = _job("evaluate", 1, seed=8, slot=0)

    def submit(pool, job):
        future = Future()
        future.set_result({"owner_result": job.identity})
        return future

    executor = BranchExecutor(
        object(),
        submit=submit,
        resolve=lambda job, raw: _terminal(differently_seeded),
    )

    with pytest.raises(ValueError, match="does not match"):
        executor.perform(
            BranchAction(
                BranchingState(),
                jobs=(admitted,),
                wait="all",
            )
        )

    assert list(executor.one_pending.values()) == [admitted]
    assert executor.observed == {}
