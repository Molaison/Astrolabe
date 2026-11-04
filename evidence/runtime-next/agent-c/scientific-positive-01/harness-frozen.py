#!/usr/bin/env python3
"""Frozen, correctness-only real CLI control for acceptance and stage entry.

This does not measure throughput and does not alter W0. The child invokes the
public CLI, then reads its surviving scientific owner and CPU RNG states. No
scientific function is patched and no model/GPU/broker execution is substituted.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import traceback


ROOT = Path(__file__).resolve().parents[2]
WORKLOAD = ROOT / "benchmarks/runtime/workloads/cpu_scientific_control.yaml"
IDENTITY = "(0, 17)"
STEPS = 4
EVALUATIONS = 5  # Initial + 3 mutation attempts + 1 target-stage reevaluation.
WALLTIME_PER_RUN = 45
MAX_OUTPUT_BYTES = 16 * 1024**2
READER_FILES = ("runtime_workflow_benchmark.py", "runtime_trace_report.py",
                "runtime_structure_parity.py", "runtime_benchmark_processes.py")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def source_record(repo):
    repo = Path(repo).resolve()
    if git(repo, "diff", "HEAD", "--", "astrolabe", "run.py", "config"):
        raise ValueError(f"uncommitted production/configuration changes in {repo}")
    if git(repo, "ls-files", "--others", "--exclude-standard", "--", "astrolabe", "config"):
        raise ValueError(f"untracked production/configuration inputs in {repo}")
    configs = git(repo, "ls-tree", "-r", "HEAD", "--", "config")
    return {"repository": str(repo), "head": git(repo, "rev-parse", "HEAD"),
            "config_tree_sha256": hashlib.sha256(configs.encode()).hexdigest()}


def freeze(args):
    from omegaconf import OmegaConf

    sources = {name: source_record(getattr(args, name)) for name in ("baseline", "candidate")}
    if sources["baseline"]["config_tree_sha256"] != sources["candidate"]["config_tree_sha256"]:
        raise ValueError("configuration bases differ; refuse an uncontrolled protocol comparison")
    config = OmegaConf.load(WORKLOAD)
    assert (config.seed, config.repeats, config.steps) == (17, 1, STEPS)
    assert config.execution.enabled is False
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    fixture = out / "workload.yaml"
    shutil.copyfile(WORKLOAD, fixture)
    plan = {
        "schema_version": 1, "kind": "correctness_only_cli_positive_control", "performance_data": False,
        "sources": sources, "workload": str(fixture), "workload_sha256": digest(fixture),
        "harness_sha256": digest(__file__), "python": sys.version, "executable": sys.executable,
        "reader_sha256": {name: digest(ROOT / "scripts/validation" / name) for name in READER_FILES},
        "budget": {"runs": 2, "repeats_per_run": 1, "seed": 17, "steps_per_run": STEPS,
                   "mutation_attempts_per_run": 3, "expected_evaluations_per_run": EVALUATIONS,
                   "walltime_per_run_seconds": WALLTIME_PER_RUN, "max_output_bytes_per_run": MAX_OUTPUT_BYTES,
                   "gpu_allocation": "", "mode": "direct", "warmups": 0},
        "required_positive_coverage": ["noninitial accepted sequence change", "stage transition after mutation"],
        "comparison": "exact proposal/score/decision/stage rows, terminal owner, CPU RNG and scientific output hashes",
        "limitations": ["Dummy is not pretrained inference", "no structures are generated", "snapshot is not resume certification"],
    }
    write_json(out / "plan.json", plan)
    print(json.dumps({"plan": str(out / "plan.json"), "workload_sha256": plan["workload_sha256"], "budget": plan["budget"]}))
    return 0


def cpu_rng():
    # Read only already imported modules; never initialize an accelerator or a
    # new library simply to inspect its RNG.
    random = sys.modules["random"]
    numpy = sys.modules["numpy"]
    state = numpy.random.get_state()
    result = {"python": random.getstate(),
              "numpy": {"algorithm": state[0], "keys": state[1].tolist(), "position": state[2],
                        "has_gaussian": state[3], "cached_gaussian_hex": float(state[4]).hex()}}
    torch = sys.modules.get("torch")
    result["torch_cpu"] = None if torch is None else torch.get_rng_state().tolist()
    return result


def terminal_snapshot():
    from astrolabe.core import global_config

    rng = cpu_rng()
    decisions = global_config.get_const("decisions")
    stage = global_config.get_const("stage_manager")
    owner = {
        "step": global_config.get_variable("step"),
        "last_accepted": decisions.last_accepted,
        "accepted_steps": list(decisions.accepted_steps),
        "proposal_sequences": decisions.sequences,
        "weighted_scores_hex": {str(step): float(score).hex() for step, score in decisions.scores.items()},
        "stage_history": decisions.stage_history,
        "current_stage": stage.current_stage,
        "transition_gate_active": stage.transition_consistency_gate_context is not None,
        "accepted_sequences": {name: protomer.sequence for name, protomer in global_config.get_const("protomers").items()},
    }
    assert rng == cpu_rng(), "observer must not consume RNG"
    return {"owner": owner, "cpu_rng": rng, "resume_certified": False}


def child(args):
    repo, directory = args.repo.resolve(), args.directory.resolve()
    sys.path.insert(0, str(repo))
    overrides = (f"out_dir={directory / 'results'} raw_model_results_dir={directory / 'raw'} "
                 "execution.enabled=false execution.max_active=1")
    sys.argv = ["astrolabe", "--config-dir", str(repo / "config"), "run", "--standalone",
                str(args.workload.resolve()), "--no-chdir", "--overrides", overrides]
    code = 0
    try:
        runpy.run_module("astrolabe", run_name="__main__")
    except SystemExit as exc:
        code = int(exc.code or 0)
    except BaseException:
        traceback.print_exc()
        code = 1
    if code == 0:
        try:
            write_json(directory / "terminal-scientific-state.json", terminal_snapshot())
        except Exception:
            traceback.print_exc()
            code = 1
    return code


def inspect_run(directory):
    # Reuse the existing completed-trajectory reader, without invoking timing or
    # changing the frozen W0 measurement implementation.
    sys.path.insert(0, str(ROOT))
    from scripts.validation.runtime_workflow_benchmark import read_completed

    completed, errors = read_completed(directory, expected_identities={IDENTITY})
    rows = completed.get(IDENTITY, [])
    attempts = [row for row in rows if int(row["step"]) > 0 and row["note"] != "re-evaluate due to stage change"]
    accepted = [row for row in attempts if row["accepted"].lower() == "true"]
    if [int(row["step"]) for row in attempts] != list(range(1, STEPS)):
        errors.append("mutation attempt budget differs")
    if not any(row.get("pA_sequence") != "AAAA" for row in accepted):
        errors.append("no accepted noninitial sequence change")
    entries = [row for row in rows if row["note"] == "enter stage" and int(row["step"]) > 0]
    reevaluations = [row for row in rows if row["note"] == "re-evaluate due to stage change"]
    if len(entries) != 1 or len(reevaluations) != 1:
        errors.append("expected exactly one mutation-triggered stage entry/reevaluation")
    snapshot = json.loads((directory / "terminal-scientific-state.json").read_text())
    if snapshot["owner"]["current_stage"] != "accepted_mutation" or snapshot["owner"]["transition_gate_active"]:
        errors.append("unexpected final stage owner")
    if set(snapshot["owner"]["proposal_sequences"]) != {str(step) for step in range(STEPS)}:
        errors.append("proposal history budget differs")
    events = [json.loads(line) for line in (directory / "runtime.jsonl").read_text().splitlines()]
    phases = Counter(event["phase"] for event in events if event.get("event") == "stage")
    if dict(phases) != {name: EVALUATIONS for name in ("cpu_prepare", "cpu_execute", "cpu_finalize")}:
        errors.append("evaluation phase budget differs")
    provenance = list((directory / "results").rglob("provenance.json"))
    run_root = provenance[0].parent if len(provenance) == 1 else None
    outputs = {}
    if run_root is not None:
        for path in sorted(run_root.rglob("*")):
            if path.is_file() and path.suffix in {".csv", ".pdb", ".cif", ".fasta", ".fa", ".npy"}:
                data = path.read_bytes().replace(str(run_root).encode(), b"<RUN>")
                outputs[str(path.relative_to(run_root))] = hashlib.sha256(data).hexdigest()
    if "scores.csv" not in outputs:
        errors.append("missing scientific score output")
    return {"errors": errors, "rows": rows, "terminal_state": snapshot, "scientific_output_sha256": outputs,
            "counts": {"mutation_attempts": len(attempts), "accepted_mutation_steps": [int(row["step"]) for row in accepted],
                       "stage_entries": len(entries), "stage_reevaluations": len(reevaluations),
                       "scientific_rows": len(rows), "phase_counts": dict(phases)}}


def run(args):
    plan_path = args.plan.resolve()
    plan = json.loads(plan_path.read_text())
    if digest(__file__) != plan["harness_sha256"] or digest(plan["workload"]) != plan["workload_sha256"]:
        raise ValueError("frozen harness or workload changed")
    if {name: digest(ROOT / "scripts/validation" / name) for name in READER_FILES} != plan["reader_sha256"]:
        raise ValueError("frozen completed-trajectory reader changed")
    results = {}
    for name in ("baseline", "candidate"):
        source = plan["sources"][name]
        repo = Path(source["repository"])
        if source_record(repo) != source:
            raise ValueError(f"frozen {name} source changed")
        directory = plan_path.parent / name
        directory.mkdir(exist_ok=False)
        env = os.environ.copy()
        for key in ("ASTROLABE_EXECUTION_ENDPOINT", "ASTROLABE_EXECUTION_TOKEN", "ASTROLABE_CPU_OWNER", "DEBUG"):
            env.pop(key, None)
        env.update(CUDA_VISIBLE_DEVICES="", NVIDIA_VISIBLE_DEVICES="void", HIP_VISIBLE_DEVICES="",
                   ROCR_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", GH_CUDA_PREINITIALIZE="0", PYTHONHASHSEED="17",
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
                   PYTHONPATH=str(repo), ASTROLABE_RUNTIME_TRACE=str(directory / "runtime.jsonl"))
        command = [sys.executable, str(Path(__file__).resolve()), "_child", "--repo", str(repo),
                   "--directory", str(directory), "--workload", plan["workload"]]
        write_json(directory / "command.json", {"command": command, "cwd": str(directory),
                   "cpu_only": True, "performance_data": False})
        try:
            with (directory / "cli.log").open("w") as log:
                process = subprocess.run(command, cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT,
                                         timeout=WALLTIME_PER_RUN)
            if process.returncode:
                raise RuntimeError(f"CLI/observer returned {process.returncode}")
            if sum(path.stat().st_size for path in directory.rglob("*") if path.is_file()) > MAX_OUTPUT_BYTES:
                raise RuntimeError("output budget exceeded")
            if source_record(repo) != source:
                raise ValueError(f"{name} source changed during execution")
            results[name] = inspect_run(directory)
        except Exception as exc:
            results[name] = {"errors": [f"{type(exc).__name__}: {exc}"]}
        write_json(directory / "validation.json", results[name])
    checks = {}
    if all(not item["errors"] for item in results.values()):
        for key in ("rows", "terminal_state", "scientific_output_sha256"):
            checks[key + "_exact"] = results["baseline"][key] == results["candidate"][key]
    passed = bool(checks) and all(checks.values())
    report = {"kind": plan["kind"], "performance_data": False, "status": "DONE" if passed else "PARTIAL",
              "passed": passed, "plan_sha256": digest(plan_path), "sources": plan["sources"],
              "comparisons": checks, "runs": results, "limits": plan["limitations"]}
    write_json(plan_path.parent / "report.json", report)
    print(json.dumps({"status": report["status"], "passed": passed, "comparisons": checks,
                      "runs": {name: {key: value for key, value in item.items() if key in {"counts", "errors"}}
                               for name, item in results.items()}}))
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freezing = commands.add_parser("freeze")
    freezing.add_argument("--baseline", type=Path, required=True)
    freezing.add_argument("--candidate", type=Path, required=True)
    freezing.add_argument("--output", type=Path, required=True)
    running = commands.add_parser("run")
    running.add_argument("--plan", type=Path, required=True)
    internal = commands.add_parser("_child")
    internal.add_argument("--repo", type=Path, required=True)
    internal.add_argument("--directory", type=Path, required=True)
    internal.add_argument("--workload", type=Path, required=True)
    args = parser.parse_args()
    return {"freeze": freeze, "run": run, "_child": child}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
