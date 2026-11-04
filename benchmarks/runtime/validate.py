#!/usr/bin/env python3
"""CPU-only CI gates and a fixed, interleaved baseline/candidate W0 comparison.

Measurement, process ownership and scientific comparison are delegated to the
existing validation harness. This module only selects revisions, counts W0 work
and aggregates paired results; it does not implement another process observer.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.validation import runtime_workflow_benchmark as benchmark
from astrolabe.runtime.telemetry import process_namespace_supported

BASELINE = "d66ee714f2e755948534f95d9d125287ce8dbd91"
BASELINE_COMMITS = (BASELINE, "9d62b3eb86cb75a2266af49dc48cb851e69e0341")
PROTOCOL = "f2636de19cdb90c2707a24969da016c398ba13b71c70fc4cde7e2fa00e7855e5"
GROUPS = 5
VARIANTS = ("baseline", "candidate")
MODES = benchmark.MODES
WALL_BUDGET = 900.0
OUTPUT_BUDGET = 260_000_000
IDENTITIES = {str((index, 17 + index)) for index in range(3)}
PHASES = ("cpu_prepare", "cpu_execute", "cpu_finalize")
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
MEASUREMENT_FILES = (
    "scripts/validation/runtime_workflow_benchmark.py",
    "scripts/validation/runtime_benchmark_processes.py",
    "scripts/validation/runtime_trace_report.py",
    "scripts/validation/runtime_structure_parity.py",
)
# The measurement reference stays d66 even when the compared baseline changes.
MEASUREMENT_SHA256 = dict(zip(MEASUREMENT_FILES, (
    "8873b26f63a0d232bdae672c826fba1d7506fa6e4bbc48a9708dcec91e7b1094",
    "e4089c2021586807db8c3e9499ccd99204aee69ae19f3e2cd2e3c4f574c6356c",
    "add56b60843f3b4b3855203c62d1e7a75631d30025ed9c329988e38041ec89f5",
    "1a5c6a0afb2868cba9bc0b9d794614035c917a577e1c5fbec2a344ee2f397242",
)))
REQUIRED_TESTS = (
    "tests/test_candidate_batch_policy.py", "tests/test_candidate_batch_runner.py",
    "tests/test_execution_identity.py", "tests/test_app.py",
    "tests/test_runtime_execution.py", "tests/test_runtime_contracts.py",
    "tests/test_runtime_broker_deadline.py",
    "tests/test_runtime_ingress.py", "tests/test_runtime_request_isolation.py",
    "tests/test_runtime_saturation.py",
    "tests/test_runtime_resources.py", "tests/test_runtime_controller.py",
    "tests/test_runtime_context.py", "tests/test_runtime_process_observations.py",
    "tests/test_runtime_benchmark_processes.py", "tests/test_runtime_trace_report.py",
    "tests/test_runtime_structure_parity.py", "tests/test_workflow_benchmark.py",
    "tests/test_evaluator_lifecycle_contract.py", "tests/test_resource_shutdown.py",
    "tests/test_mutator_subprocess_protocol.py", "tests/test_stage_transaction.py",
    "tests/test_run_stage_transition.py", "tests/test_graceful_shutdown_repeat_boundary.py",
    "tests/test_runtime_validation_harness.py",
    "tests/test_loss_import_lifecycle.py",
    "tests/test_runtime_requirements_source.py",
    "tests/test_gpu_slurm_launcher.py",
)
TORCH_CONTRACT_TESTS = (
    "tests/test_esmfold_output_boundary.py", "tests/test_tree_codec_torch.py",
    "tests/test_evaluator_rng_lifecycle.py", "tests/test_esmfold1_resident.py",
    "tests/test_interaction_ipae.py", "tests/test_repeat_symmetry.py",
    "tests/test_ligand_reference_pose_distance.py", "tests/test_design_screening_geometry.py",
    "tests/test_loss_import_lifecycle.py",
    "tests/test_gpu_mutator_control.py",
    "tests/test_pyrosetta_lazy_import.py",
)


def fresh_directory(path):
    path = path.resolve()
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"refusing to overwrite prior evidence: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def preflight():
    """Probe the required facilities; never replace a denied facility."""
    checks = {"linux": sys.platform == "linux", "python_3_11": sys.version_info[:2] == (3, 11),
              "matching_process_namespace": process_namespace_supported()}
    visibility = {key: os.environ.get(key) for key in (
        "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")}
    checks["no_gpu_allocation"] = (visibility["CUDA_VISIBLE_DEVICES"] == ""
        and visibility["NVIDIA_VISIBLE_DEVICES"] in (None, "", "none", "void")
        and all(visibility[key] in (None, "") for key in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")))
    checks["cpu_framework_policy"] = os.environ.get("JAX_PLATFORMS") == "cpu"
    checks["bounded_cpu_threads"] = all(os.environ.get(key) == "1" for key in THREAD_ENV)
    checks["not_inside_broker"] = not any(os.environ.get(key) for key in (
        "ASTROLABE_EXECUTION_ENDPOINT", "ASTROLABE_EXECUTION_TOKEN", "ASTROLABE_INNER_BROKER_ENDPOINT"))
    unix_error = None
    try:
        with tempfile.TemporaryDirectory(prefix="astrolabe-ci-") as temporary:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
                server.settimeout(2)
                server.bind(str(Path(temporary) / "probe.sock"))
                server.listen(1)
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                    client.settimeout(2)
                    client.connect(server.getsockname())
                    with server.accept()[0] as peer:
                        peer.settimeout(2)
                        client.sendall(b"w0")
                        checks["af_unix"] = peer.recv(2) == b"w0"
    except (OSError, AttributeError) as exc:
        checks["af_unix"] = False
        unix_error = {"type": type(exc).__name__, "errno": getattr(exc, "errno", None), "message": str(exc)}
    packages = {}
    for name in ("numpy", "scipy", "hydra-core", "omegaconf", "biopython", "pytest", "psutil", "jax", "jaxlib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    checks["required_cpu_dependencies"] = all(packages.values())
    return {"status": "DONE" if all(checks.values()) else "BLOCKED", "checks": checks,
            "blocking_reasons": [key for key, passed in checks.items() if not passed],
            "af_unix_error": unix_error, "python": sys.version,
            "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
            "gpu_visibility": visibility, "thread_environment": {key: os.environ.get(key) for key in THREAD_ENV},
            "packages": packages, "gpu_certification": "BLOCKED: no GPU execution in this workflow"}


def source_identity(repository, *, baseline=False, baseline_commit=BASELINE):
    repository = repository.resolve()
    if baseline and baseline_commit not in BASELINE_COMMITS:
        raise ValueError(f"unsupported frozen baseline commit: {baseline_commit}")
    result = benchmark.execution_environment(repository)
    if baseline and result["git_commit"] != baseline_commit:
        raise ValueError(f"baseline must be {baseline_commit}, observed {result['git_commit']!r}")
    if not result["git_commit"] or not (repository / "astrolabe/__main__.py").is_file():
        raise ValueError(f"not a committed Astrolabe checkout: {repository}")
    status = subprocess.run(["git", "-C", str(repository), "status", "--porcelain=v1", "--",
                             "astrolabe", "run.py", "config", "scripts/validation"],
                            check=True, capture_output=True, text=True).stdout
    if status:
        raise ValueError(f"runnable source must match its recorded commit in {repository}: {status}")
    return result


def build_plan(baseline, candidate, out, *, baseline_commit=BASELINE):
    from omegaconf import OmegaConf
    sources = {"baseline": source_identity(baseline, baseline=True, baseline_commit=baseline_commit),
               "candidate": source_identity(candidate)}
    measurement = {}
    for name in MEASUREMENT_FILES:
        expected = MEASUREMENT_SHA256[name]
        for label, repository in (("wrapper", ROOT), ("baseline", baseline), ("candidate", candidate)):
            measured = hashlib.sha256((repository / name).read_bytes()).hexdigest()
            if measured != expected:
                raise ValueError(f"measurement implementation differs from immutable d66 in {label}: {name}")
        measurement[name] = expected
    fixture = candidate / "scripts/validation/fixtures/runtime_cpu.yaml"
    config_dir = candidate / "config"
    base = benchmark.compose(fixture, config_dir)
    if benchmark.fingerprint(benchmark.scientific_projection(base)) != PROTOCOL:
        raise ValueError("W0 scientific protocol changed; refusing a different workload")
    trajectories = benchmark.planned_trajectories(base)
    if {str((row["repeat_index"], row["seed"])) for row in trajectories} != IDENTITIES:
        raise ValueError("W0 requires exactly the original three trajectory identities")
    runs = []
    for group in range(GROUPS):
        order = MODES[group % len(MODES):] + MODES[:group % len(MODES)]
        revisions = VARIANTS if group % 2 == 0 else VARIANTS[::-1]
        for mode in order:
            for variant in revisions:
                directory = out / f"{len(runs) + 1:03d}_{variant}_{mode}"
                directory.mkdir()
                config_path = directory / "config.yaml"
                OmegaConf.save(OmegaConf.create(benchmark.mode_config(base, mode, directory)), config_path)
                repository = baseline if variant == "baseline" else candidate
                # This is the same isolated import/cwd contract as the original harness.
                command = [sys.executable, "-c", "import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); runpy.run_module('astrolabe',run_name='__main__')",
                           str(repository), "--config-dir", str(config_dir), "run", "--standalone", str(config_path), "--no-chdir"]
                runs.append({"run": len(runs) + 1, "repetition": group, "variant": variant, "mode": mode,
                             "repository": str(repository), "directory": str(directory), "command": command,
                             "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()})
    return {"schema_version": 1, "protocol_sha256": PROTOCOL, "configuration": base, "sources": sources,
            "baseline_commit": baseline_commit, "measurement_reference_commit": BASELINE,
            "measurement_source_sha256": measurement,
            "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "working_directory": str(fixture.parent), "expected_trajectories": trajectories,
            "groups": GROUPS, "requested_trials": len(runs), "trajectories_per_trial": 3,
            "evaluations_per_trial": 9, "walltime_budget_seconds": WALL_BUDGET,
            "max_output_bytes": OUTPUT_BUDGET, "gpu_allocation": "", "runs": runs,
            "order": "modes rotate each group; baseline/candidate AB then BA in alternating groups; no warmup"}


def work_counts(rows, trace):
    """Count completed scientific rows and actual Dummy forward stage events."""
    errors = []
    if set(rows) != IDENTITIES:
        errors.append("trajectory identities differ from W0")
    for identity, values in rows.items():
        if len(values) != 3 or [row.get("step") for row in values] != ["0", "1", "2"]:
            errors.append(f"expected exactly steps 0,1,2 for {identity}")
    phases, backend_counts = Counter(), Counter()
    trace_error = None
    try:
        with trace.open() as stream:
            for line in stream:
                event = json.loads(line)
                if event.get("event") == "stage":
                    phases[event.get("phase")] += 1
                    backend_counts[event.get("backend")] += 1
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        trace_error = str(exc)
        errors.append("evaluation trace is unavailable or malformed")
    if trace_error is None:
        from astrolabe.runtime.contracts import identity_key
        dummy = identity_key({"backend": "astrolabe.evaluators.dummy.DummyEvaluator"})
        if dict(phases) != {phase: 9 for phase in PHASES} or dict(backend_counts) != {dummy: 27}:
            errors.append("expected exactly 9 Dummy prepare/execute/finalize calls")
    return {"certified": not errors, "errors": errors, "completed_trajectories": len(rows),
            "scientific_rows": sum(map(len, rows.values())), "trace_available": trace_error is None,
            "evaluation_calls": phases["cpu_execute"] if trace_error is None else None,
            "phase_counts": dict(phases) if trace_error is None else None, "trace_error": trace_error}


def comparison_report(records):
    # Candidate direct results must not expand the baseline's empirical error.
    references = [{**row, "mode": "candidate_direct" if row["variant"] == "candidate"
                   and row["mode"] == "direct" else row["mode"]} for row in records]
    try:
        quality = benchmark.comparison(references)
    except ValueError as exc:
        quality = {"certified": False, "invalid_measurement": True, "reason": str(exc)}
    try:
        structures = benchmark.compare_structures(references)
    except (OSError, ValueError, KeyError, IndexError) as exc:
        structures = {"certified": False, "invalid_measurement": True, "reason": str(exc)}
    quality["delivered_structure_parity"] = structures
    if structures["certified"] is False:
        quality["certified"] = False
    expected = {(group, variant, mode) for group in range(GROUPS) for variant in VARIANTS for mode in MODES}
    observed = [(row["repetition"], row["variant"], row["mode"]) for row in records]
    complete = len(observed) == len(expected) and set(observed) == expected and all(row["valid_workload"] for row in records)
    certified = complete and quality["certified"]
    groups = {}
    for variant in VARIANTS:
        for mode in MODES:
            selected = [row for row in records if (row["variant"], row["mode"]) == (variant, mode)]
            valid = [row for row in selected if row["valid_workload"]]
            seconds = [row["seconds"] for row in valid]
            groups[f"{variant}/{mode}"] = {
                "attempted": len(selected), "valid": len(valid), "failed_or_incomplete": len(selected) - len(valid),
                "completed_trajectories": sum(row["completed"] for row in selected),
                "evaluation_calls": sum(row["work_counts"]["evaluation_calls"] for row in selected)
                    if selected and all(row["work_counts"]["evaluation_calls"] is not None for row in selected) else None,
                "median_seconds": statistics.median(seconds) if seconds else None,
                "range_seconds": [min(seconds), max(seconds)] if seconds else None,
                "effective_trajectories_per_second": sum(row["completed"] for row in selected) / sum(seconds)
                    if certified else None}
    pairs, direct_comparisons = [], []
    by_key = {(row["repetition"], row["variant"], row["mode"]): row for row in records}
    for mode in MODES:
        ratios = [by_key[(group, "baseline", mode)]["seconds"] / by_key[(group, "candidate", mode)]["seconds"]
                  for group in range(GROUPS)] if certified else []
        before, after = groups[f"baseline/{mode}"], groups[f"candidate/{mode}"]
        pairs.append({"mode": mode, "paired_speed_ratios": ratios or None,
                      "median_paired_speed_ratio": statistics.median(ratios) if ratios else None,
                      "candidate_faster_pairs": sum(ratio > 1 for ratio in ratios) if ratios else None,
                      "all_five_pairs_improved": all(ratio > 1 for ratio in ratios) if ratios else None,
                      "strict_range_separation": before["range_seconds"][0] > after["range_seconds"][1] if certified else None})
    best_direct = min(VARIANTS, key=lambda variant: groups[f"{variant}/direct"]["median_seconds"]) if certified else None
    if certified:
        for mode in MODES:
            ratios = [by_key[(group, best_direct, "direct")]["seconds"] / by_key[(group, "candidate", mode)]["seconds"]
                      for group in range(GROUPS)]
            direct_comparisons.append({"candidate_mode": mode, "direct_variant": best_direct,
                                       "paired_speed_ratios": ratios, "median_paired_speed_ratio": statistics.median(ratios)})
    return {"status": "DONE" if certified else "PARTIAL", "protocol_sha256": PROTOCOL,
            "requested_trials": len(expected), "attempted_trials": len(records), "complete": complete,
            "correctness": quality, "groups": groups, "same_mode_pairs": pairs,
            "candidate_vs_best_direct": direct_comparisons, "best_direct_selection": "lower median of the two direct variants",
            "performance_claim": "CPU W0 timings only; inspect paired variation" if certified else "not certified; no speedup may be inferred",
            "gpu_certification": "BLOCKED: CPU Dummy workload, no model weights or GPU execution",
            "runs": records}


def comparison_markdown(result):
    def value(number):
        return "unavailable" if number is None else f"{number:.4f}"
    lines = ["# Runtime CPU W0 comparison", "", f"Status: **{result['status']}**. Completed trial contract: **{result['complete']}**.",
             f"Baseline: `{result.get('baseline_commit') or 'unavailable'}`. Candidate: `{result.get('candidate_commit') or 'unavailable'}`.",
             f"Protocol: `{PROTOCOL}`. {result['attempted_trials']} / {result['requested_trials']} trials attempted.", "",
             "| Revision / mode | Valid / attempted | Evaluations | Median seconds | Range seconds | Effective trajectories/s |",
             "| --- | ---: | ---: | ---: | --- | ---: |"]
    for label, group in result["groups"].items():
        span = "unavailable" if group["range_seconds"] is None else " – ".join(map(value, group["range_seconds"]))
        lines.append(f"| {label} | {group['valid']} / {group['attempted']} | {group['evaluation_calls']} | {value(group['median_seconds'])} | {span} | {value(group['effective_trajectories_per_second'])} |")
    lines += ["", f"Scientific gate across both revisions and every mode: **{result['correctness']['certified']}**.",
              "Baseline direct repeats alone define numerical tolerance. Candidate direct cannot widen it.", "",
              "| Same mode | Median baseline / candidate seconds | Faster pairs / 5 | All five improved | Nonoverlapping time ranges |",
              "| --- | ---: | ---: | --- | --- |"]
    for pair in result["same_mode_pairs"]:
        lines.append(f"| {pair['mode']} | {value(pair['median_paired_speed_ratio'])} | {pair['candidate_faster_pairs']} | {pair['all_five_pairs_improved']} | {pair['strict_range_separation']} |")
    lines += ["", "Ratios above one favor the candidate. Incomplete work or failed scientific/release checks suppress every speed ratio and effective-throughput number.", "",
              "| Candidate mode | Best direct revision | Median direct / candidate seconds |", "| --- | --- | ---: |"]
    for pair in result["candidate_vs_best_direct"]:
        lines.append(f"| {pair['candidate_mode']} | {pair['direct_variant']} | {value(pair['median_paired_speed_ratio'])} |")
    lines += ["", result["performance_claim"], "", result["gpu_certification"], "",
              "All trials start fresh; no warmup, extra trajectories or profiler is added. The existing 30 ms observer times process-tree completion and verifies release. Raw stdout/stderr, configs, traces, telemetry, source manifests, scientific rows and failed trials are retained.", ""]
    return "\n".join(lines)


def compare(args):
    out = fresh_directory(args.out)
    plan = build_plan(args.baseline.resolve(), ROOT, out, baseline_commit=args.baseline_commit)
    provenance = {"baseline_commit": plan["sources"]["baseline"]["git_commit"],
                  "candidate_commit": plan["sources"]["candidate"]["git_commit"],
                  "measurement_reference_commit": BASELINE}
    benchmark.json_write(out / "plan.json", plan)
    if args.plan_only:
        print(f"PLAN ONLY: {out / 'plan.json'}; no work was executed")
        return 0
    environment = preflight()
    benchmark.json_write(out / "preflight.json", environment)
    if environment["status"] != "DONE":
        benchmark.json_write(out / "report.json", {**provenance, "status": "BLOCKED", "preflight": environment,
                                                   "performance_claim": "not measured", "runs": []})
        print("BLOCKED: " + ", ".join(environment["blocking_reasons"]))
        return 3
    started, records, stop_reason = time.monotonic(), [], None
    deadline = started + WALL_BUDGET
    for trial in plan["runs"]:
        remaining = OUTPUT_BUDGET - sum(file.stat().st_size for file in out.rglob("*") if file.is_file())
        if time.monotonic() >= deadline or remaining <= 0:
            stop_reason = "walltime_budget_exhausted" if time.monotonic() >= deadline else "output_budget_exhausted"
            break
        trial_started = time.monotonic()
        try:
            measured = benchmark.run_once(trial["command"], Path(trial["directory"]), Path(plan["working_directory"]),
                                          "", deadline, .5, remaining, Path(trial["repository"]))
        except OSError as exc:
            measured = {"exit_code": None, "seconds": time.monotonic() - trial_started,
                        "stop_reason": "execution_start_or_measurement_failed", "error": str(exc)}
        try:
            rows, failures = benchmark.read_completed(trial["directory"], IDENTITIES)
            structures = benchmark.collect_structures(trial["directory"])
        except (OSError, ValueError) as exc:
            rows, failures, structures = {}, [{"kind": "invalid_result_manifest", "error": str(exc)}], {}
        counts = work_counts(rows, Path(trial["directory"]) / "runtime.jsonl")
        measured.update(trial, scientific_rows=rows, structure_artifacts=structures, completed=len(rows), failures=failures, work_counts=counts)
        measured["valid_workload"] = (measured["exit_code"] == 0 and not failures and counts["certified"]
            and not measured["stop_reason"] and measured.get("process_release", {}).get("release_confirmed") is True)
        records.append(measured)
        checkpoint = {**comparison_report(records), **provenance}
        benchmark.json_write(out / "report.json", checkpoint)
        print(f"{trial['run']:02d}/30 {trial['variant']}/{trial['mode']}: valid={measured['valid_workload']} trajectories={len(rows)} evaluations={counts['evaluation_calls']} seconds={measured['seconds']:.4f}", flush=True)
        if measured["stop_reason"]:
            stop_reason = measured["stop_reason"]
            break
    result = {**comparison_report(records), **provenance}
    result["experiment"] = {"wall_seconds": time.monotonic() - started, "walltime_budget_seconds": WALL_BUDGET,
                            "max_output_bytes": OUTPUT_BUDGET, "stop_reason": stop_reason, "allocated_gpu_hours": 0,
                            "scope": "all trials, failed work and between-trial analysis; bounded observer cleanup may extend a deadline"}
    benchmark.json_write(out / "report.json", result)
    (out / "report.md").write_text(comparison_markdown(result))
    print(out / "report.json")
    return 0 if result["status"] == "DONE" else 2


def junit_result(path, returncode):
    if not path.is_file():
        return {"status": "PARTIAL", "returncode": returncode, "error": "pytest did not produce JUnit evidence"}
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    skips = [{"test": case.get("classname", "") + "::" + case.get("name", ""),
              "reason": skip.get("message") or skip.text} for case in cases for skip in case.findall("skipped")]
    failed = [case.get("classname", "") + "::" + case.get("name", "") for case in cases
              if case.find("failure") is not None or case.find("error") is not None]
    return {"status": "DONE" if returncode == 0 and cases and not skips and not failed else "PARTIAL",
            "returncode": returncode, "test_cases": len(cases), "failures": failed, "skips": skips,
            "skip_policy": "no skips accepted for these required CPU suites"}


def regressions(args):
    out = fresh_directory(args.out)
    required = TORCH_CONTRACT_TESTS if args.torch_contracts else REQUIRED_TESTS
    missing = [name for name in required if not (ROOT / name).is_file()]
    environment = preflight()
    benchmark.json_write(out / "preflight.json", environment)
    if missing or environment["status"] != "DONE":
        result = {"status": "BLOCKED", "missing_required_suites": missing, "preflight": environment}
        benchmark.json_write(out / "report.json", result)
        print(json.dumps(result, indent=2))
        return 3
    command = [sys.executable, "-m", "pytest", "-q", "-ra", *required, f"--junitxml={out / 'junit.xml'}"]
    benchmark.json_write(out / "command.json", command)
    test_env = dict(os.environ)
    test_env["PYTHONPATH"] = str(ROOT) + (os.pathsep + test_env["PYTHONPATH"] if test_env.get("PYTHONPATH") else "")
    benchmark.json_write(out / "environment.json", {"PYTHONPATH": test_env["PYTHONPATH"], "scope": "regression subprocesses only"})
    with (out / "stdout.log").open("w") as log:
        process = subprocess.run(command, cwd=ROOT, env=test_env, stdout=log, stderr=subprocess.STDOUT)
    print((out / "stdout.log").read_text())
    result = junit_result(out / "junit.xml", process.returncode)
    result["required_suites"] = list(required)
    benchmark.json_write(out / "report.json", result)
    return 0 if result["status"] == "DONE" else 2


def summary(args):
    args.out.mkdir(parents=True, exist_ok=True)
    reports = {}
    phases = ("preflight", "regressions") if args.torch_contracts else ("preflight", "regressions", "comparison")
    for phase in phases:
        path = args.out / phase / "report.json"
        try:
            reports[phase] = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            reports[phase] = {"status": "BLOCKED", "reason": "no readable completed report", "error": str(exc)}
    overall = "DONE" if all(report.get("status") == "DONE" for report in reports.values()) else "PARTIAL"
    title = "CPU Torch and structural contract validation" if args.torch_contracts else "runtime validation"
    revision = f"Candidate: `{os.environ.get('GITHUB_SHA', 'local/unpublished')}`."
    baseline_commit = None
    if not args.torch_contracts:
        try:
            plan = json.loads((args.out / "comparison/plan.json").read_text())
            baseline_commit = plan["sources"]["baseline"]["git_commit"]
            revision = f"Candidate: `{plan['sources']['candidate']['git_commit']}`."
        except (OSError, ValueError, KeyError, TypeError):
            pass
        revision += f" Compared baseline: `{baseline_commit or 'unavailable'}`."
    lines = [f"# Astrolabe {title}", "", f"Overall CPU validation: **{overall}**.",
             revision, "",
             "| Gate | Status |", "| --- | --- |"]
    lines.extend(f"| {phase} | {report.get('status', 'BLOCKED')} |" for phase, report in reports.items())
    for phase, report in reports.items():
        if report.get("status") != "DONE":
            lines += ["", f"## {phase} evidence", "", "```json", json.dumps(report, indent=2)[:12000], "```"]
    comparison = args.out / "comparison/report.md"
    if comparison.is_file():
        lines += ["", comparison.read_text()]
    if args.torch_contracts:
        lines += ["", "Scope: CPU Torch worker, output-boundary and lifecycle tests (including local fake-weight fixtures), plus structural loss tests using real SciPy/RDKit. No pretrained inference or performance benchmark is certified by this job."]
    lines += ["", "Real pretrained GPU backend certification: **BLOCKED** in this CPU-only job.",
              "The uploaded artifact preserves raw logs, configuration, traces, telemetry, test cases and skip reasons. Missing runs and failed work are retained and cannot count as acceleration.", ""]
    markdown = "\n".join(lines)
    (args.out / "summary.md").write_text(markdown)
    summary_result = {"status": overall, "gates": {key: value.get("status") for key, value in reports.items()}}
    if not args.torch_contracts:
        summary_result["baseline_commit"] = baseline_commit
    benchmark.json_write(args.out / "summary.json", summary_result)
    if args.step_summary:
        with args.step_summary.open("a") as stream:
            stream.write(markdown)
    print(markdown)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "regressions", "compare", "summary"):
        subparser = commands.add_parser(command)
        subparser.add_argument("--out", type=Path, required=True)
        if command == "compare":
            subparser.add_argument("--baseline", type=Path, required=True)
            subparser.add_argument("--baseline-commit", choices=BASELINE_COMMITS, default=BASELINE,
                                   help="exact expected baseline HEAD; defaults to the original d66 baseline")
            subparser.add_argument("--plan-only", action="store_true")
        if command == "summary":
            subparser.add_argument("--step-summary", type=Path)
        if command in ("regressions", "summary"):
            subparser.add_argument("--torch-contracts", action="store_true",
                                   help="validate CPU Torch worker/lifecycle and structural loss suites; no benchmark")
    args = parser.parse_args(argv)
    try:
        if args.command == "preflight":
            out, result = fresh_directory(args.out), preflight()
            benchmark.json_write(out / "report.json", result)
            print(json.dumps(result, indent=2))
            return 0 if result["status"] == "DONE" else 3
        return {"compare": compare, "regressions": regressions, "summary": summary}[args.command](args)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
