"""Fault-injection evidence only: no runtime changes or GPU/broker access.
Run from repository root: ../venv/bin/python evidence/runtime-next/agent-c/probe_candidate_identity.py
"""
from __future__ import annotations
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from astrolabe.experimental.candidate_batch_policy import (
    CandidateTask, CandidateResult, CandidatePoolConfig, CandidatePoolScheduler,
)


def run_case(name, return_id, return_generation):
    submitted = CandidateTask(candidate_id="expected", generation=1)
    def worker(task):
        return CandidateResult(
            task=CandidateTask(candidate_id=return_id, generation=return_generation),
            status="completed", score=999.0, payload={"sequence": "AAAA"},
        )
    config = CandidatePoolConfig(num_candidates=1, min_completed=1, max_wait_s=0.3)
    with CandidatePoolScheduler(config, worker) as scheduler:
        scheduler.submit_generation(1, [submitted])
        generation_one = scheduler.collect_until_ready(target_generation=1)
        generation_two = scheduler.collect_until_ready(target_generation=2)
    return {
        "case": name,
        "submitted": asdict(submitted),
        "generation_1_selected": asdict(generation_one.selected) if generation_one and generation_one.selected else None,
        "generation_2_selected": asdict(generation_two.selected) if generation_two and generation_two.selected else None,
    }


def main():
    cases = [
        run_case("valid_identity_control", "expected", 1),
        run_case("wrong_candidate_same_generation", "unsubmitted", 1),
        run_case("wrong_candidate_and_future_generation", "unsubmitted", 2),
    ]
    result = {
        "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "python": sys.version,
        "scope": "stdlib ThreadPool experimental candidate scheduler; injected worker replies; no model, no GPU, no broker",
        "cases": cases,
        "observed": {
            "valid_control_selected": bool(cases[0]["generation_1_selected"] and cases[0]["generation_1_selected"]["task"]["candidate_id"] == "expected"),
            "foreign_candidate_selected": bool(cases[1]["generation_1_selected"] and cases[1]["generation_1_selected"]["task"]["candidate_id"] == "unsubmitted"),
            "unsubmitted_generation_selected": cases[2]["generation_2_selected"] is not None,
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
