"""Exercise actual pool boundaries with cheap controlled workers, not model/GPU work.

Run from the repository root with ../venv/bin/python and preserve JSON output.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from astrolabe.experimental.candidate_batch_policy import (
    CandidatePoolConfig,
    CandidatePoolScheduler,
    CandidateResult,
    CandidateTask,
)


def run_case(name, batches):
    calls = []
    errors = []
    collections = []

    def worker(task):
        calls.append((task.candidate_id, task.generation))
        return CandidateResult(task, "completed", score=float(task.payload))

    config = CandidatePoolConfig(num_candidates=2, min_completed=1, max_wait_s=2.0)
    with CandidatePoolScheduler(config, worker) as scheduler:
        for declared_generation, tasks in batches:
            try:
                scheduler.submit_generation(declared_generation, tasks)
            except ValueError as exc:
                errors.append(str(exc))
            for collection in scheduler.drain():
                collections.append({
                    "completed": len(collection.completed),
                    "failed": len(collection.failed),
                    "selected": None if collection.selected is None else {
                        "candidate_id": collection.selected.candidate_id,
                        "generation": collection.selected.task.generation,
                        "score": collection.selected.score,
                    },
                })
    return {
        "case": name,
        "worker_calls": len(calls),
        "identity_call_counts": [
            {"candidate_id": identity[0], "generation": identity[1], "count": count}
            for identity, count in sorted(Counter(calls).items())
        ],
        "errors": errors,
        "collections": collections,
    }


def queue_bound_probe():
    release = threading.Event()
    entered = threading.Event()
    calls = []

    def worker(task):
        entered.set()
        if not release.wait(5.0):
            raise RuntimeError("probe controller did not release workers")
        calls.append(task.candidate_id)
        return float(task.payload)

    config = CandidatePoolConfig(
        num_candidates=8, min_completed=1, max_wait_s=2.0, max_in_flight=1
    )
    with CandidatePoolScheduler(config, worker) as scheduler:
        try:
            scheduler.submit_generation(
                1, [CandidateTask(str(index), 1, payload=index) for index in range(8)]
            )
            if not entered.wait(2.0):
                raise RuntimeError("probe worker did not start")
            result = {
                "configured_max_in_flight": 1,
                "submitted_and_unfinished": len(scheduler._pending),
                "running": sum(future.running() for future in scheduler._pending),
                "waiting": sum(not future.running() for future in scheduler._pending),
            }
        finally:
            release.set()
        collections = scheduler.drain()
    result["final_worker_calls"] = len(calls)
    result["final_selected_score"] = collections[0].selected.score
    return result


def main():
    task = CandidateTask("replayed", 1, payload=0.5)
    cases = [
        run_case("valid_control", [(1, [
            CandidateTask("a", 1, payload=0.25),
            CandidateTask("b", 1, payload=0.75),
        ])]),
        run_case("mixed_generation_batch", [(1, [
            CandidateTask("valid_prefix", 1, payload=0.25),
            CandidateTask("wrong_generation", 2, payload=0.75),
        ])]),
        run_case("duplicate_within_batch", [(1, [
            CandidateTask("duplicate", 1, payload=0.25),
            CandidateTask("duplicate", 1, payload=0.75),
        ])]),
        run_case("replay_after_collection", [(1, [task]), (1, [task])]),
        run_case("same_id_different_generation_control", [
            (1, [CandidateTask("allowed", 1, payload=0.25)]),
            (2, [CandidateTask("allowed", 2, payload=0.75)]),
        ]),
    ]
    print(json.dumps({
        "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "scope": "Actual experimental ThreadPool scheduler; controlled inexpensive workers; no model, GPU or broker",
        "cases": cases,
        "queue_bound_probe": queue_bound_probe(),
    }, indent=2))


if __name__ == "__main__":
    main()
