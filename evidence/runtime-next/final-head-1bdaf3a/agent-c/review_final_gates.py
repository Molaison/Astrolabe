#!/usr/bin/env python3
"""Independently read final CI JUnit and frozen positive-control raw files.

This script does not run tests, scientific work, a model or a benchmark.
"""
import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

FINAL = "1bdaf3a53227c9197b930dc00a01b752acd90067"
FINAL_TREE = "6f763bfb04778ef4f32386ddb8aeb38ce4240bf1"
BASELINE = "d66ee714f2e755948534f95d9d125287ce8dbd91"
STARTUP = "f9ce87b9f5cf37710f0a52ebe503595e5d7fdd7f"
CONTROL_SHA = "6f52867a3d45e83985fc84b844cfdf7227e61761e615355a63b913d83eb4ed26"
WORKLOAD_SHA = "ddbd295188ed22241736750bed4fa613c553dbf29b5423de0f9454b7c90cbde9"


def read(path):
    return json.loads(path.read_text())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()

    def git(*args):
        return subprocess.check_output(["git", "-C", str(a.repo), *args])

    assert git("rev-parse", FINAL + "^{tree}").decode().strip() == FINAL_TREE
    assert git("diff", "--name-only", STARTUP, FINAL, "--", "astrolabe", "run.py", "config") == b""
    wrapper = git("show", FINAL + ":benchmarks/runtime/validate.py")
    constants = {}
    for node in ast.parse(wrapper).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"REQUIRED_TESTS", "TORCH_CONTRACT_TESTS"}:
                    constants[target.id] = list(ast.literal_eval(node.value))
    suites = {"cpu": constants["REQUIRED_TESTS"], "torch": constants["TORCH_CONTRACT_TESTS"]}
    regressions = {}
    for kind, required in suites.items():
        directory = a.root / (kind + "-artifact") / "regressions"
        report = read(directory / "report.json")
        command = read(directory / "command.json")
        assert command[1:5] == ["-m", "pytest", "-q", "-ra"]
        assert command[5:-1] == required and command[-1].startswith("--junitxml=")
        assert report["required_suites"] == required
        tree = ET.parse(directory / "junit.xml")
        cases = list(tree.iter("testcase"))
        identities = [(c.attrib["classname"], c.attrib["name"]) for c in cases]
        assert len(identities) == len(set(identities))
        problems = [dict(c.attrib) for c in cases if any(c.find(tag) is not None for tag in ("error", "failure", "skipped"))]
        assert not problems
        assert all(int(node.attrib[k]) == 0 for node in tree.iter("testsuite") for k in ("errors", "failures", "skipped"))
        assert sum(int(n.attrib["tests"]) for n in tree.iter("testsuite")) == len(cases) == report["test_cases"]
        counts = {}
        for name in required:
            prefix = name[:-3].replace("/", ".")
            counts[name] = sum(c.attrib["classname"] == prefix or c.attrib["classname"].startswith(prefix + ".") for c in cases)
        assert all(counts.values()) and sum(counts.values()) == len(cases)
        assert report["status"] == "DONE" and report["returncode"] == 0 and not report["failures"] and not report["skips"]
        regressions[kind] = {"status": "PASS", "passed": len(cases), "failed": 0, "skipped": 0,
                             "required_source_suites_exact": True, "test_counts_by_suite": counts,
                             "junit_sha256": sha((directory / "junit.xml").read_bytes())}
    assert "tests/test_pyrosetta_lazy_import.py" not in suites["cpu"]
    assert "tests/test_pyrosetta_lazy_import.py" in suites["torch"]
    assert "tests/test_gpu_mutator_control.py" in suites["torch"]
    assert {"tests/test_gpu_slurm_launcher.py", "tests/test_runtime_requirements_source.py"} <= set(suites["cpu"])
    torch = read(a.root / "torch-artifact/environment/torch.json")
    assert torch == {"version": "2.7.1+cpu", "cuda_build": None}

    scientific = a.root / "scientific-positive-01"
    plan = read(scientific / "plan.json")
    assert plan["harness_sha256"] == CONTROL_SHA
    assert sha(git("show", FINAL + ":scripts/validation/runtime_scientific_control.py")) == CONTROL_SHA
    assert sha((scientific / "workload.yaml").read_bytes()) == plan["workload_sha256"] == WORKLOAD_SHA
    assert sha(git("show", FINAL + ":benchmarks/runtime/workloads/cpu_scientific_control.yaml")) == WORKLOAD_SHA
    assert plan["budget"] == {"runs": 2, "repeats_per_run": 1, "seed": 17, "steps_per_run": 4,
        "mutation_attempts_per_run": 3, "expected_evaluations_per_run": 4,
        "expected_score_rows_per_run": 5, "expected_stage_score_recomputations_per_run": 1,
        "walltime_per_run_seconds": 45, "max_output_bytes_per_run": 16777216,
        "gpu_allocation": "", "mode": "direct", "warmups": 0}
    observations = {}
    for variant, head in (("baseline", BASELINE), ("candidate", FINAL)):
        assert plan["sources"][variant]["head"] == head
        directory = scientific / variant
        command = read(directory / "command.json")
        assert command["cpu_only"] and command["performance_data"] is False
        paths = list((directory / "results").rglob("scores.csv"))
        assert len(paths) == 1
        rows = list(csv.DictReader(paths[0].open()))
        attempts = [r for r in rows if int(r["step"]) > 0 and r["note"] != "re-evaluate due to stage change"]
        accepted = [r for r in attempts if r["accepted"].lower() == "true"]
        entries = [r for r in rows if r["note"] == "enter stage"]
        assert len(rows) == 5 and [int(r["step"]) for r in attempts] == [1, 2, 3]
        assert len(accepted) == len(entries) == 1 and int(accepted[0]["step"]) == 1
        assert accepted[0]["pA_sequence"] != "AAAA"
        assert len([r for r in rows if r["note"] == "re-evaluate due to stage change"]) == 1
        state = read(directory / "terminal-scientific-state.json")
        assert state["owner"]["current_stage"] == "accepted_mutation" and not state["owner"]["transition_gate_active"]
        assert state["owner"]["last_accepted"] == 1 and 1 in state["owner"]["accepted_steps"]
        assert len(state["cpu_rng"]["torch_cpu"]) == 5056 and len(state["cpu_rng"]["numpy"]["keys"]) == 624
        events = [json.loads(line) for line in (directory / "runtime.jsonl").read_text().splitlines()]
        phases = Counter(e["phase"] for e in events if e.get("event") == "stage")
        assert dict(phases) == {"cpu_prepare": 4, "cpu_execute": 4, "cpu_finalize": 4}
        outputs = {}
        run_root = paths[0].parent
        for path in sorted(run_root.rglob("*")):
            if path.is_file() and path.suffix in {".csv", ".pdb", ".cif", ".fasta", ".fa", ".npy"}:
                outputs[str(path.relative_to(run_root))] = sha(path.read_bytes().replace(str(run_root).encode(), b"<RUN>"))
        observations[variant] = {"rows": rows, "state": state, "outputs": outputs,
            "counts": {"mutation_attempts": 3, "accepted_mutations": 1, "stage_entries": 1,
                       "score_rows": 5, "actual_evaluator_calls": 4}}
    assert observations["baseline"] == observations["candidate"]
    result = {"status": "PASS", "head": FINAL, "tree": FINAL_TREE, "regressions": regressions,
        "torch": torch, "benchmark_run_by_this_reader": False,
        "production_and_config_bytes_equal_startup_f9": True,
        "validate_wrapper_sha256": sha(wrapper),
        "scientific_control": {"status": "PASS", "baseline": BASELINE, "candidate": FINAL,
            "harness_sha256": CONTROL_SHA, "workload_sha256": WORKLOAD_SHA,
            "unchanged_budget": plan["budget"], "rows_owner_cpu_rng_outputs_exact": True,
            "counts_per_source": observations["candidate"]["counts"],
            "cpu_rng_streams": ["Python", "NumPy", "Torch CPU"],
            "scope": "Actual direct CLI Dummy/MCMC positive control; no structures, pretrained/GPU or crash-resume certification"}}
    with a.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "test_counts": {k: v["passed"] for k, v in regressions.items()},
                      "scientific_counts_per_source": result["scientific_control"]["counts_per_source"]}, indent=2))


if __name__ == "__main__":
    main()
