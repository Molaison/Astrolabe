#!/usr/bin/env python3
"""Fault-inject child metadata into a synthetic CPU checker fixture, no GPU."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile


def probe(repo):
    path = repo / "tests/test_gpu_mutator_control.py"
    spec = importlib.util.spec_from_file_location("b_control_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    protocol = json.loads(module.control.DEFAULT_PROTOCOL.read_text())
    with tempfile.TemporaryDirectory(prefix="c-gpu-gate-review-") as temporary:
        frozen, execution = module.synthetic_run.__wrapped__(Path(temporary), protocol)
        positive = module.control.aggregate(frozen, copy.deepcopy(execution))["status"]
        child = frozen.parent / "trials/g00-seed17-resident/trial.json"
        original = json.loads(child.read_text())
        mutations = {
            "seed": lambda x: x.update(seed=999),
            "mode": lambda x: x.update(mode="legacy"),
            "head": lambda x: x.update(head="foreign-source"),
            "request_indices": lambda x: [r.update(index=0) for r in x["requests"]],
            "request_phases": lambda x: [r.update(phase="first") for r in x["requests"]],
        }
        results = {}
        for name, mutation in mutations.items():
            data = copy.deepcopy(original)
            mutation(data)
            child.write_text(json.dumps(data))
            report = module.control.aggregate(frozen, copy.deepcopy(execution))
            results[name] = {"status": report["status"], "effective_throughput_present":
                "effective_design_requests_per_second" in report["per_mode"]["resident"]}
    return {"scope": "Synthetic CPU checker fault injection; no model, GPU, process or benchmark",
            "commit": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
            "harness_sha256": module.control.sha256(repo / "benchmarks/runtime/gpu_mutator_control.py"),
            "positive_control": positive, "mutations": results,
            "all_invalid_suppressed": all(x["status"] == "INVALID" and not x["effective_throughput_present"] for x in results.values())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = probe(args.repo.resolve())
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
