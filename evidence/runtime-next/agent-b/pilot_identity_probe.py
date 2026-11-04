"""Independent actual CPU scheduler identity probe; no GPU/model claims."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("checkout", type=Path)
parser.add_argument("--require-fixed", action="store_true")
args = parser.parse_args()
sys.path.insert(0, str(args.checkout.resolve()))
from astrolabe.experimental.candidate_batch_policy import (
    CandidateExecutionError, CandidatePoolConfig, CandidatePoolScheduler,
    CandidateResult, CandidateTask,
)


def summary(result):
    if result is None:
        return None
    return {"id": result.candidate_id, "generation": result.task.generation,
            "status": result.status, "score": result.score,
            "payload": result.payload, "error": result.error}


def collection_summary(collection):
    if collection is None:
        return None
    return {"selected": summary(collection.selected),
            "results": [summary(r) for r in collection.results]}


cases = []
for returned_id, returned_generation in (("alien", 7), ("submitted", 8),
                                          ("submitted", 6), ("alien", 8)):
    submitted = CandidateTask("submitted", 7, {"sequence": "AAAA"})
    for fail_policy in ("record-and-continue", "fail-fast"):
        for method in ("collect", "drain"):
            foreign = CandidateResult(CandidateTask(returned_id, returned_generation),
                                      "completed", 1000.0, {"sequence": "GGGG"})
            config = CandidatePoolConfig(num_candidates=1, min_completed=1,
                                         max_wait_s=2.0, fail_policy=fail_policy)
            row = {"received": [returned_id, returned_generation],
                   "fail_policy": fail_policy, "method": method}
            with CandidatePoolScheduler(config, lambda task: foreign) as scheduler:
                scheduler.submit_generation(7, [submitted])
                try:
                    if method == "drain":
                        collections = scheduler.drain()
                        row["collections"] = [collection_summary(c) for c in collections]
                    else:
                        collections = [c for g in (7, 6, 8)
                                       if (c := scheduler.collect_until_ready(target_generation=g)) is not None]
                        row["collections"] = [collection_summary(c) for c in collections]
                    row["ready_after"] = [summary(r) for r in scheduler._ready]
                    results = [r for c in collections for r in c.results]
                    row["gate_pass"] = (fail_policy == "record-and-continue"
                        and len(results) == 1 and results[0].status == "failed"
                        and results[0].task is submitted and results[0].payload is None
                        and results[0].score is None
                        and "identity mismatch" in results[0].error
                        and all(c.selected is None for c in collections)
                        and scheduler._ready == [])
                except CandidateExecutionError as exc:
                    row["error"] = str(exc)
                    row["ready_after"] = [summary(r) for r in scheduler._ready]
                    row["gate_pass"] = (fail_policy == "fail-fast"
                        and "Candidate submitted failed: candidate result identity mismatch" in str(exc)
                        and scheduler._ready == [])
            cases.append(row)

normal = []
for scalar in (False, True):
    rng = random.Random(7331)
    expected_rng = random.Random(7331)
    expected = expected_rng.random()
    input_payload = {"sequence": "AAAA", "nested": {"budget": 3}}
    output_payload = {"sequence": "CCCC", "scientific_result": [0.125, 0.25]}
    submitted = CandidateTask("legitimate", 7, input_payload)
    returned = []
    def worker(task):
        score = rng.random()
        result = score if scalar else CandidateResult(
            CandidateTask(task.candidate_id, task.generation), "completed", score,
            output_payload, duration_s=0.75)
        returned.append(result)
        return result
    with CandidatePoolScheduler(CandidatePoolConfig(num_candidates=1, min_completed=1,
                                                    max_wait_s=2.0), worker) as scheduler:
        scheduler.submit_generation(7, [submitted])
        selected = scheduler.collect_until_ready(target_generation=7).selected
        normal.append({"scalar": scalar, "selected": summary(selected),
            "rng_state_exact": rng.getstate() == expected_rng.getstate(),
            "score_exact": selected.score == expected,
            "input_unchanged": input_payload == {"sequence": "AAAA", "nested": {"budget": 3}},
            "return_object_and_payload_preserved": scalar or
                (selected is returned[0] and selected.payload is output_payload and selected.duration_s == 0.75)})

report = {"head": subprocess.check_output(["git", "-C", str(args.checkout), "rev-parse", "HEAD"], text=True).strip(),
          "evidence_kind": "actual CPU ThreadPoolExecutor scheduler; synthetic worker results; no model or GPU",
          "cases": cases, "normal": normal,
          "all_fault_gates_pass": all(r["gate_pass"] for r in cases),
          "normal_gates_pass": all(all(r[k] for k in ("rng_state_exact", "score_exact", "input_unchanged", "return_object_and_payload_preserved")) for r in normal)}
print(json.dumps(report, indent=2))
if args.require_fixed and not (report["all_fault_gates_pass"] and report["normal_gates_pass"]):
    raise SystemExit(1)
