"""Rejected design probe: blocking submit can change first-completion selection.

This wraps a scheduler instance only; no production code or scientific workload
is changed. Event-controlled workers preserve the same three-task budget.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from astrolabe.experimental.candidate_batch_policy import (
    CandidatePoolConfig, CandidatePoolScheduler, CandidateTask,
)


def run_case(blocking_capacity):
    entered = [threading.Event() for _ in range(3)]
    release = [threading.Event() for _ in range(3)]
    submitted = threading.Event()
    calls = []
    errors = []

    def worker(task):
        index = int(task.candidate_id)
        entered[index].set()
        if not release[index].wait(5.0):
            raise RuntimeError("controller did not release worker")
        calls.append(index)
        return float(index + 1)

    config = CandidatePoolConfig(
        num_candidates=3, min_completed=1, max_wait_s=5.0, max_in_flight=1
    )
    with CandidatePoolScheduler(config, worker) as scheduler:
        if blocking_capacity:
            slots = threading.BoundedSemaphore(blocking_capacity)
            submit = scheduler._executor.submit

            def limited_submit(*args, **kwargs):
                if not slots.acquire(timeout=5.0):
                    raise RuntimeError("probe admission timed out")
                try:
                    future = submit(*args, **kwargs)
                except BaseException:
                    slots.release()
                    raise
                future.add_done_callback(lambda _future: slots.release())
                return future

            scheduler._executor.submit = limited_submit

        def produce():
            try:
                scheduler.submit_generation(
                    1, [CandidateTask(str(index), 1) for index in range(3)]
                )
            except BaseException as exc:
                errors.append(repr(exc))
            finally:
                submitted.set()

        producer = threading.Thread(target=produce)
        producer.start()
        try:
            assert entered[0].wait(2.0)
            if not blocking_capacity:
                assert submitted.wait(2.0)
            release[0].set()
            assert entered[1].wait(2.0)
            if blocking_capacity:
                # A third submit cannot return until the second task frees its slot.
                release[1].set()
                assert entered[2].wait(2.0)
                assert submitted.wait(2.0)
            assert not errors, errors
            first = scheduler.collect_until_ready(target_generation=1)
            for event in release:
                event.set()
            producer.join(timeout=2.0)
            assert not producer.is_alive()
            rest = scheduler.drain()
        finally:
            for event in release:
                event.set()
            producer.join(timeout=2.0)
    assert first is not None and first.selected is not None
    return {
        "blocking_capacity": blocking_capacity,
        "first_selected_id": first.selected.candidate_id,
        "first_selected_score": first.selected.score,
        "first_collection_count": len(first.completed),
        "total_worker_calls": len(calls),
        "total_completed": len(first.completed) + sum(len(item.completed) for item in rest),
    }


if __name__ == "__main__":
    current = run_case(None)
    blocking = run_case(1)
    assert current["total_worker_calls"] == blocking["total_worker_calls"] == 3
    assert current["total_completed"] == blocking["total_completed"] == 3
    assert current["first_selected_id"] != blocking["first_selected_id"]
    print(json.dumps({
        "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "scope": "Event-controlled ThreadPool protocol probe, not a performance/model/GPU benchmark",
        "current": current,
        "blocking_submission_prototype": blocking,
        "verdict": "reject naive blocking submission: same budget, changed first selected candidate",
    }, indent=2))
