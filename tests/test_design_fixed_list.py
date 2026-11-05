from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from astrolabe import model_execution
from astrolabe.design.collection import CollectionExecutionError, JobResult
from astrolabe.design.fixed_list import (
    FixedTask,
    FixedTaskListPolicy,
    run_fixed_task_list,
)
from astrolabe.design.types import key_for


def _completed(job, payload):
    return JobResult(
        job=job,
        status="completed",
        score=0.0,
        payload=payload,
    )


def test_fixed_list_runs_in_order_and_normalizes_completion_order() -> None:
    release = {
        "first": threading.Event(),
        "second": threading.Event(),
    }
    entered: list[str] = []
    lock = threading.Lock()

    def worker(job):
        with lock:
            entered.append(job.job_id)
        if job.job_id == "first":
            assert release["second"].wait(5)
        else:
            release["first"].set()
        if job.job_id == "second":
            release["second"].set()
        return _completed(job, {"job_id": job.job_id})

    tasks = (FixedTask("first"), FixedTask("second"))
    result = run_fixed_task_list(tasks, worker, max_in_flight=2)

    assert set(entered) == {"first", "second"}
    assert [item.job_id for item in result.results] == ["first", "second"]
    assert [item.payload for item in result.results] == [
        {"job_id": "first"},
        {"job_id": "second"},
    ]
    assert result.collections[0].selected is None


def test_fixed_list_serial_never_overlaps_worker_calls() -> None:
    active = 0
    maximum = 0
    lock = threading.Lock()

    def worker(job):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        return _completed(job, job.job_id)

    result = run_fixed_task_list(
        (FixedTask("one"), FixedTask("two"), FixedTask("three")),
        worker,
        max_in_flight=1,
    )

    assert maximum == 1
    assert [item.job_id for item in result.results] == ["one", "two", "three"]


def test_fixed_list_records_ordinary_failure_and_continues() -> None:
    def worker(job):
        if job.job_id == "bad":
            raise RuntimeError("ordinary terminal failure")
        return _completed(job, "ok")

    result = run_fixed_task_list(
        (FixedTask("bad"), FixedTask("good")),
        worker,
        max_in_flight=1,
    )

    assert [(item.job_id, item.status) for item in result.results] == [
        ("bad", "failed"),
        ("good", "completed"),
    ]
    assert result.results[0].error == "ordinary terminal failure"


def test_fixed_list_receipt_hooks_stay_on_control_thread() -> None:
    control_thread = threading.get_ident()
    observed = {}

    def admission(job):
        observed["admission"] = threading.get_ident()
        return None

    def worker(job):
        observed["worker"] = threading.get_ident()
        return _completed(job, "ok")

    def terminal(job, result):
        observed["terminal"] = threading.get_ident()

    result = run_fixed_task_list(
        (FixedTask("one"),),
        worker,
        admission=admission,
        terminal_observed=terminal,
    )

    assert [item.job_id for item in result.results] == ["one"]
    assert observed["admission"] == control_thread
    assert observed["terminal"] == control_thread
    assert observed["worker"] != control_thread


def test_fixed_list_synchronous_worker_uses_control_thread() -> None:
    control_thread = threading.get_ident()
    observed = {}

    def worker(job):
        observed["worker"] = threading.get_ident()
        return _completed(job, "ok")

    def terminal(job, result):
        observed["terminal"] = threading.get_ident()

    result = run_fixed_task_list(
        (FixedTask("one"),),
        worker,
        worker_execution="synchronous",
        terminal_observed=terminal,
    )

    assert [item.job_id for item in result.results] == ["one"]
    assert observed == {
        "worker": control_thread,
        "terminal": control_thread,
    }


def test_fixed_list_synchronous_worker_preserves_python_numpy_rng(
    monkeypatch,
) -> None:
    import random

    random.seed(4101)
    np.random.seed(4102)
    control_thread = threading.get_ident()
    initial = model_execution.capture_random_state()
    observed: list[tuple[str, int]] = []
    restore_calls: list[dict] = []
    worker_waiting = threading.Event()
    transition_window = threading.Event()
    overlap_observed: list[bool] = []
    real_capture = model_execution.capture_random_state
    real_restore = model_execution.restore_random_state

    def spy_capture():
        state = real_capture()
        if worker_waiting.is_set():
            transition_window.set()
            time.sleep(0.05)
        return state

    def spy_restore(state) -> None:
        restore_calls.append(dict(state))
        real_restore(state)

    monkeypatch.setattr(model_execution, "capture_random_state", spy_capture)
    monkeypatch.setattr(model_execution, "restore_random_state", spy_restore)

    def worker(job):
        if threading.get_ident() != control_thread:
            worker_waiting.set()
            if transition_window.wait(5):
                overlap_observed.append(True)
        random.random()
        np.random.random()
        observed.append((job.job_id, threading.get_ident()))
        return _completed(job, job.job_id)

    result = run_fixed_task_list(
        (FixedTask("one"), FixedTask("two")),
        worker,
        worker_execution="synchronous",
    )
    actual = key_for(real_capture())
    real_restore(initial)
    random.random()
    np.random.random()
    random.random()
    np.random.random()
    expected = key_for(real_capture())

    assert [item.payload for item in result.results] == ["one", "two"]
    assert observed == [("one", control_thread), ("two", control_thread)]
    assert overlap_observed == []
    assert restore_calls == []
    assert actual == expected


def test_synchronous_worker_mode_rejects_parallel_execution() -> None:
    with pytest.raises(
        ValueError,
        match="synchronous fixed-list execution requires max_in_flight=1",
    ):
        run_fixed_task_list(
            (FixedTask("one"), FixedTask("two")),
            lambda job: _completed(job, "ok"),
            max_in_flight=2,
            worker_execution="synchronous",
        )


def test_policy_rejects_an_incomplete_collection_without_advancing() -> None:
    from astrolabe.design.collection import ResultCollection
    from astrolabe.design.collection_search import CollectionEvent

    policy = FixedTaskListPolicy(
        (FixedTask("one"), FixedTask("two")),
        batch_size=2,
    )
    state = policy.initial_state()
    action = policy.ask(state)
    state = policy.tell(state, CollectionEvent(action=action))
    drain = policy.ask(state)
    assert drain.kind == "drain"
    partial = ResultCollection(
        collection_index=1,
        trigger="drain",
        results=(_completed(action.jobs[0], "one"),),
        selected=None,
        carried_forward_job_ids=(action.jobs[1].job_id,),
        started_at_s=0.0,
        finished_at_s=1.0,
    )

    with pytest.raises(
        CollectionExecutionError,
        match="incomplete job proof",
    ):
        policy.tell(
            state,
            CollectionEvent(
                action=drain,
                collections=(partial,),
            ),
        )
    assert not policy.finished(state)


def test_policy_uses_explicit_drain_for_each_fixed_batch() -> None:
    from astrolabe.design.collection import ResultCollection
    from astrolabe.design.collection_search import CollectionEvent

    policy = FixedTaskListPolicy(
        (FixedTask("one"), FixedTask("two"), FixedTask("three")),
        batch_size=2,
    )
    state = policy.initial_state()
    submit = policy.ask(state)
    assert submit.kind == "submit"
    state = policy.tell(state, CollectionEvent(action=submit))
    drain = policy.ask(state)
    assert drain.kind == "drain"
    state = policy.tell(
        state,
        CollectionEvent(
            action=drain,
            collections=(
                ResultCollection(
                    collection_index=1,
                    trigger="drain",
                    results=tuple(
                        _completed(job, job.job_id) for job in submit.jobs
                    ),
                    selected=None,
                    carried_forward_job_ids=(),
                    started_at_s=0.0,
                    finished_at_s=1.0,
                ),
            ),
        ),
    )

    assert state.phase == "submit"
    assert state.next_index == 2
    next_submit = policy.ask(state)
    assert next_submit.kind == "submit"
    assert [job.job_id for job in next_submit.jobs] == ["three"]


def test_empty_fixed_list_finishes_without_calling_worker() -> None:
    calls = []
    result = run_fixed_task_list((), lambda job: calls.append(job))

    assert calls == []
    assert result.collections == ()
