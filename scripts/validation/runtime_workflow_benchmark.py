#!/usr/bin/env python3
"""Fixed-workload, interleaved direct/resident/adaptive workflow comparison.

No extra trajectories are generated for tuning or warm-up. Timing is external to
the measured CLI. Run inside the same allocated job/environment as Astrolabe.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validation.runtime_trace_report import summarize as summarize_trace
from scripts.validation.runtime_structure_parity import collect as collect_structures, compare as compare_structures
from scripts.validation.runtime_benchmark_processes import BenchmarkProcessObserver

MODES = ("direct", "resident_serial", "resident_adaptive")
DECISIONS = {"step", "iteration", "accepted", "is_accepted", "soft_accepted", "stage", "note",
             "early_stop", "is_early_stop", "stage_entry", "acceptance_gate_passed",
             "reference_gate_active", "transition_consistency_gate_active", "prefix_early_stop_triggered"}
NON_SCIENTIFIC_ROOTS = {"execution", "out_dir", "raw_model_results_dir"}


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def scientific_projection(config):
    """Only known execution knobs may differ; sampling knobs remain fingerprinted."""
    result = copy.deepcopy(config)
    for key in NON_SCIENTIFIC_ROOTS:
        result.pop(key, None)
    for section in ("evaluator", "mutator"):
        for entry in (result.get(section) or {}).values():
            if isinstance(entry, dict):
                entry.pop("execution", None)
                entry.pop("execution_resources", None)
                # Terms hold a second level of backend configuration.
                for term in entry.values():
                    if isinstance(term, dict):
                        term.pop("execution", None)
                        term.pop("execution_resources", None)
    return result


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def execution_environment(repository=ROOT):
    def git(*args):
        result = subprocess.run(["git", "-C", str(repository), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    sources = {}
    # Git diff omits new untracked modules. Capture the actual runnable source
    # files too, without recording environment/configuration file contents.
    for location in ("astrolabe", "run.py", "config", "scripts/validation"):
        path = repository / location
        candidates = [path] if path.is_file() else path.rglob("*")
        for file in candidates:
            if file.is_file() and "__pycache__" not in file.parts and file.suffix in {
                    ".py", ".yaml", ".yml", ".json", ".toml", ".ini", ".cfg", ".sh"}:
                sources[str(file.relative_to(repository))] = hashlib.sha256(file.read_bytes()).hexdigest()
    return {"hostname": platform.node(), "platform": platform.platform(),
            "python": sys.version, "executable": sys.executable,
            "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "gpu_allocation": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
            "source_repository": str(repository), "git_commit": git("rev-parse", "HEAD"),
            "source_manifest_sha256": fingerprint(sources), "source_manifest": sources,
            "git_diff_sha256": fingerprint(git("diff", "HEAD", "--", "astrolabe", "run.py", "config")),
            "git_status": git("status", "--porcelain=v1", "--ignore-submodules")}


def compose(path, config_dir):
    from omegaconf import OmegaConf, open_dict
    from astrolabe.cli import _compose_hydra_config, _resolve_template_config, _strip_defaults_key
    base = _compose_hydra_config(config_dir=config_dir,
                                 config_name=_resolve_template_config(config_dir, None), overrides=[])
    with open_dict(base):
        base = OmegaConf.merge(base, _strip_defaults_key(OmegaConf.load(path)))
    # run_repeats resolves interpolations after assigning the per-repeat seed.
    # Resolving here would freeze e.g. a backend's ${seed} at the base seed and
    # silently change the supplied scientific protocol for later repeats.
    return OmegaConf.to_container(base, resolve=False)


def planned_trajectories(base):
    """Read only scheduling fields, matching the existing run_repeats identity."""
    from omegaconf import OmegaConf
    config = OmegaConf.create(base)

    def integer(value, name):
        digits = value[1:] if type(value) is str and value[:1] in "+-" else value
        if type(value) is not int and not (type(digits) is str and digits
                and digits.isascii() and digits.isdecimal()):
            raise ValueError(name + " must be a decimal integer")
        return int(value)

    repeats = integer(config.get("repeats", 1), "repeats")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    raw_seed = config.get("seed")
    seed = integer(raw_seed, "seed") if raw_seed is not None else None
    if seed is not None and (seed < 1 or seed + repeats - 1 > 2**32 - 1):
        raise ValueError("explicit seed sequence must stay within 1..2**32-1")
    task = str(config.get("task_name") or "")
    return [{"repeat_index": index, "seed": seed + index if seed is not None else
             (int(hashlib.sha256(f"{task}:{index}".encode()).hexdigest()[:8], 16) % 1_000_000) + 1}
            for index in range(repeats)]


def mode_config(base, mode, output):
    result = copy.deepcopy(base)
    result["out_dir"] = str(output / "results")
    result["raw_model_results_dir"] = str(output / "raw")
    execution = result.setdefault("execution", {})
    execution["enabled"] = mode != "direct"
    execution["max_active"] = "auto" if mode == "resident_adaptive" else 1
    execution["max_resident_models"] = "auto"
    execution["max_resident_per_device"] = "auto"
    if mode == "direct":
        execution["max_resident_per_device"] = 1
    # Factories own capabilities. Unsupported backend combinations fail explicitly
    # during runtime validation; this harness never substitutes another model.
    for section in ("evaluator", "mutator"):
        terms = (result.get(section) or {}).get(section + "_terms") or {}
        for term in terms.values():
            if mode == "direct" and isinstance(term, dict):
                term.pop("execution_resources", None)
                if section == "mutator":
                    # The old factory does not consume these kwargs, including
                    # when a CPU mutator is explicitly configured as auto.
                    term.pop("execution", None)
            if isinstance(term, dict) and str(term.get("_type", "")).lower() not in {
                "dummy", "rosetta_crystal", "mcmc", "pore_blocker_mcmc", "motifbreaker", "xmix"
            }:
                if mode == "direct" and str(term.get("_type", "")).lower() in {"mol1", "mol-1"}:
                    raise ValueError("Mol-1 has no legacy direct implementation in this baseline; its native reference must be supplied and verified separately")
                if mode == "direct" and section == "mutator":
                    # The preserved legacy factory predates this execution
                    # field. Absence keeps its original direct behavior.
                    term.pop("execution", None)
                    term.pop("execution_resources", None)
                else:
                    term["execution"] = "legacy" if mode == "direct" else "daemon"
    if scientific_projection(base) != scientific_projection(result):
        raise ValueError("benchmark mode changed scientific configuration")
    return result


def read_completed(directory, expected_identities=None):
    """Require terminal provenance and keyed scientific rows, not just exit status."""
    completed, failures, seen = {}, [], set()
    for file in sorted(Path(directory).glob("results/**/*_run/provenance.json")):
        record = json.loads(file.read_text())
        key = (record.get("run", {}).get("repeat_index"), record.get("run", {}).get("seed"))
        if any(type(value) is not int for value in key):
            raise ValueError("invalid scientific trajectory identity: " + repr(key))
        identity = str(key)
        if identity in seen:
            raise ValueError("duplicate scientific trajectory identity: " + identity)
        seen.add(identity)
        if record.get("state") != "completed":
            failures.append({"identity": key, "state": record.get("state"), "path": str(file)})
            continue
        scores = file.parent / "scores.csv"
        if not scores.is_file():
            raise ValueError("missing scores for scientific trajectory identity: " + identity)
        with scores.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError("completed trajectory has no scientific rows: " + str(scores))
        if any(not isinstance(name, str) or not isinstance(value, str)
               for row in rows for name, value in row.items()):
            raise ValueError("malformed scientific score rows: " + str(scores))
        # Paths are presentation fields, normalized to their actual owning run.
        normalized = [{name: value.replace(str(file.parent), "<RUN>") for name, value in row.items()}
                      for row in rows]
        completed[identity] = normalized
    if expected_identities is not None and set(completed) != set(expected_identities):
        failures.append({"kind": "scientific_identity_set_mismatch",
                         "missing": sorted(set(expected_identities) - set(completed)),
                         "unexpected": sorted(set(completed) - set(expected_identities))})
    return completed, failures


def finite_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        raise ValueError("nonfinite scientific measurement: " + str(value))
    return number


def decision_field(name):
    name = name.lower().rsplit(".", 1)[-1]
    return (name in DECISIONS or name.endswith(("_gate_passed", "_gate_active", "_accepted"))
            or ("gate" in name and name.endswith(("_passed", "_attempt", "_attempts")))
            or name in {"selected_sample", "selected_sample_index", "selected_candidate", "selected_candidate_id", "selected_model", "selected_model_index"})


def comparison(records):
    """Infer absolute per-cell tolerances only from repeated direct executions."""
    direct = [r for r in records if r["mode"] == "direct" and r["valid_workload"]]
    if len(direct) < 2:
        return {"certified": False, "reason": "need at least two successful direct repeats"}
    reference = direct[0]["scientific_rows"]
    tolerances = {}
    unstable = []
    for record in direct[1:]:
        other = record["scientific_rows"]
        if other.keys() != reference.keys():
            unstable.append("direct trajectory identities changed")
            continue
        for identity, rows in reference.items():
            if len(rows) != len(other[identity]):
                unstable.append("direct row count changed: " + identity)
                continue
            for index, row in enumerate(rows):
                observed = other[identity][index]
                if row.keys() != observed.keys():
                    unstable.append("direct measurement schema changed")
                    continue
                for key, value in row.items():
                    number = finite_number(value)
                    candidate = finite_number(observed[key])
                    cell = (identity, index, key)
                    if decision_field(key) or "seq" in key.lower() or number is None or candidate is None:
                        if value != observed[key]:
                            unstable.append("direct discrete result changed: " + repr(cell))
                    else:
                        tolerances[cell] = max(tolerances.get(cell, 0.0), abs(number - candidate))
    if unstable:
        return {"certified": False, "reason": "direct scientific baseline is unstable",
                "baseline_differences": unstable}
    findings = []
    for record in records:
        if not record["valid_workload"]:
            findings.append({"run": record["run"], "kind": "incomplete_workload"})
            continue
        if record["mode"] == "direct":
            continue
        other = record["scientific_rows"]
        if other.keys() != reference.keys():
            findings.append({"run": record["run"], "kind": "trajectory_identity"})
            continue
        for identity, rows in reference.items():
            if len(rows) != len(other[identity]):
                findings.append({"run": record["run"], "kind": "row_count", "identity": identity})
                continue
            for index, row in enumerate(rows):
                observed = other[identity][index]
                if row.keys() != observed.keys():
                    findings.append({"run": record["run"], "kind": "measurement_schema"})
                    continue
                for key, value in row.items():
                    number, candidate = finite_number(value), finite_number(observed[key])
                    tolerance = tolerances.get((identity, index, key), 0.0)
                    discrete = decision_field(key) or "seq" in key.lower() or number is None or candidate is None
                    differs = value != observed[key] if discrete else abs(number - candidate) > tolerance
                    if differs:
                        findings.append({"run": record["run"], "identity": identity, "row": index,
                                         "field": key, "kind": "decision" if decision_field(key) else "measurement",
                                         "direct": value, "observed": observed[key], "absolute_tolerance": tolerance})
    return {"certified": not findings, "comparison": "scientific scores and decisions only",
            "direct_repetitions": len(direct), "coordinate_parity": "requires backend conformance evidence",
            "tolerance_source": "maximum same-cell absolute deviation from the first direct repeat",
            "tolerances": [{"identity": i, "row": n, "field": k, "absolute": v}
                           for (i, n, k), v in sorted(tolerances.items())], "differences": findings}


def gpu_sample(allocated):
    if not allocated:
        return {"available": False, "reason": "no_gpu_allocation"}
    try:
        result = subprocess.run(["nvidia-smi", "-i", allocated,
                                 "--query-gpu=uuid,utilization.gpu,memory.used,memory.total",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3)
        if result.returncode:
            return {"available": False, "reason": "telemetry_failed", "error": result.stderr.strip()}
        devices = []
        for row in csv.reader(result.stdout.splitlines()):
            uuid, utilization, used, total = [x.strip() for x in row]
            devices.append({"uuid": uuid, "busy_percent": float(utilization),
                            "used_bytes": int(used) * 1048576, "total_bytes": int(total) * 1048576})
        return {"available": bool(devices), "devices": devices}
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "reason": type(exc).__name__}


def process_sample(process, seen_cpu):
    try:
        import psutil
        parent = psutil.Process(process.pid)
        processes = [parent, *parent.children(recursive=True)]
        rss = 0
        for item in processes:
            try:
                rss += item.memory_info().rss
                cpu = item.cpu_times()
                seen_cpu[(item.pid, item.create_time())] = cpu.user + cpu.system
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return {"rss_bytes": rss, "sampled_cpu_seconds": sum(seen_cpu.values()), "processes": len(processes)}
    except Exception as exc:
        # Measurements are optional; a missing measurement never becomes a zero.
        return {"rss_bytes": None, "sampled_cpu_seconds": None, "error": type(exc).__name__}


def run_once(command, directory, cwd, allocation, deadline, sample_interval, max_output_bytes, repository=ROOT):
    directory.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["ASTROLABE_RUNTIME_TRACE"] = str(directory / "runtime.jsonl")
    env["PYTHONPATH"] = str(repository) + os.pathsep + env.get("PYTHONPATH", "")
    start = time.monotonic()
    seen_cpu, stop_reason = {}, None
    rss_peak, vram_peak, busy_sum, busy_count = None, None, 0.0, 0
    with (directory / "stdout.log").open("wb") as stdout, (directory / "stderr.log").open("wb") as stderr, (directory / "telemetry.jsonl").open("w") as telemetry:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        process_scope = BenchmarkProcessObserver(process)
        exited, terminal = threading.Event(), {}

        def observe_exit():
            try:
                terminal.update(process_scope.observe_until_exit(process, deadline))
            except Exception as exc:
                terminal.update(exit_code=None, completion_monotonic=time.monotonic(),
                                release_confirmed=False, stop_reason="process_observation_unknown",
                                observation_errors=[str(exc)])
            finally:
                exited.set()

        observer = threading.Thread(target=observe_exit, name="benchmark-exit-observer", daemon=True)
        observer.start()
        try:
            while not exited.is_set():
                now = time.monotonic()
                sample = {"elapsed_seconds": now - start, "monotonic_ns": time.monotonic_ns(),
                          "process": process_sample(process, seen_cpu), "gpu": gpu_sample(allocation)}
                rss = sample["process"]["rss_bytes"]
                if rss is not None:
                    rss_peak = max(rss_peak or 0, rss)
                for device in sample["gpu"].get("devices", []):
                    vram_peak = max(vram_peak or 0, device["used_bytes"])
                    busy_sum += device["busy_percent"]
                    busy_count += 1
                telemetry.write(json.dumps(sample, allow_nan=False) + "\n")
                telemetry.flush()
                if not exited.is_set() and time.monotonic() >= deadline:
                    stop_reason = "walltime_budget_exhausted"
                    break
                if sum(p.stat().st_size for p in directory.rglob("*") if p.is_file()) > max_output_bytes:
                    stop_reason = "output_budget_exhausted"
                    break
                exited.wait(min(sample_interval, max(0, deadline - time.monotonic())))
        except Exception as exc:
            stop_reason = "measurement_failed"
            terminal["measurement_error"] = str(exc)
        finally:
            if not exited.is_set():
                process_scope.request_stop(stop_reason or "benchmark_interrupted")
            observer.join(timeout=7)
        if observer.is_alive():
            terminal.update(exit_code=None, completion_monotonic=time.monotonic(),
                            release_confirmed=False, stop_reason="process_observation_unknown",
                            observation_errors=["process observer did not finish bounded cleanup"])
        exit_code = terminal.get("exit_code")
    # A slow telemetry query or sampling interval after process exit must not
    # inflate the independently observed workflow completion time.
    elapsed = terminal["completion_monotonic"] - start
    stop_reason = stop_reason or terminal.get("stop_reason")
    if not terminal.get("release_confirmed"):
        stop_reason = stop_reason or "resource_release_unconfirmed"
    diagnostics = summarize_trace(directory / "runtime.jsonl")
    envelope = time.monotonic() - start
    gpu_count = len(allocation.split(",")) if allocation else 0
    return {"exit_code": exit_code, "seconds": elapsed, "stop_reason": stop_reason,
            "main_process_seconds": terminal["main_exit_monotonic"] - start if terminal.get("main_exit_monotonic") is not None else None,
            "measurement_envelope_seconds": envelope,
            "process_release": terminal,
            "peak_process_tree_rss_bytes": rss_peak,
            "sampled_cpu_seconds_lower_bound": sum(seen_cpu.values()) if seen_cpu else None,
            "peak_device_vram_bytes": vram_peak,
            "mean_sampled_gpu_busy_percent": busy_sum / busy_count if busy_count else None,
            "allocated_gpu_seconds": gpu_count * envelope,
            "workflow_gpu_seconds": gpu_count * elapsed,
            "telemetry_file": str(directory / "telemetry.jsonl"),
            "runtime_trace_file": str(directory / "runtime.jsonl"),
            "runtime_diagnostics": diagnostics,
            "cpu_cost_note": "sampling can miss short-lived children; this is a lower bound",
            "gpu_activity_note": "device-wide activity includes other jobs if allocation is not exclusive"}


def report(records, requested, protocol):
    try:
        quality = comparison(records)
    except ValueError as exc:
        quality = {"certified": False, "reason": str(exc), "invalid_measurement": True}
    if any("structure_artifacts" in record for record in records):
        try:
            structures = compare_structures(records)
        except (OSError, ValueError, KeyError, IndexError) as exc:
            structures = {"certified": False, "reason": str(exc), "invalid_measurement": True}
        quality["delivered_structure_parity"] = structures
        if structures["certified"] is False:
            quality["certified"] = False
    modes = {}
    def peak(runs, field):
        values = [r[field] for r in runs if r.get(field) is not None]
        return max(values) if values else None
    for mode in MODES:
        runs = [r for r in records if r["mode"] == mode]
        valid = [r for r in runs if r["valid_workload"]]
        modes[mode] = {"runs": len(runs), "valid_runs": len(valid),
                       "completed_trajectories": sum(r["completed"] for r in runs),
                       "median_end_to_end_seconds": statistics.median(r["seconds"] for r in valid) if valid else None,
                       "workload_completions_per_second": sum(r["completed"] for r in valid) / sum(r["seconds"] for r in runs) if runs else None,
                       "effective_trajectories_per_second": sum(r["completed"] for r in valid) / sum(r["seconds"] for r in runs) if runs and quality["certified"] else None,
                       "allocated_gpu_hours": sum(r["allocated_gpu_seconds"] for r in runs) / 3600,
                       "failed_or_incomplete_runs": len(runs) - len(valid),
                       "peak_process_tree_rss_bytes": peak(runs, "peak_process_tree_rss_bytes"),
                       "peak_device_vram_bytes": peak(runs, "peak_device_vram_bytes"),
                       "sampled_cpu_seconds_lower_bound": sum(r.get("sampled_cpu_seconds_lower_bound") or 0 for r in runs) if any(r.get("sampled_cpu_seconds_lower_bound") is not None for r in runs) else None}
    complete = len(records) == requested and all(r["valid_workload"] for r in records)
    paired = []
    for repetition in sorted({r["repetition"] for r in records if "repetition" in r}):
        group = {r["mode"]: r for r in records if r.get("repetition") == repetition}
        direct = group.get("direct")
        if not direct or not direct["valid_workload"]:
            continue
        for mode in MODES[1:]:
            run = group.get(mode)
            paired.append({"repetition": repetition, "mode": mode,
                           "end_to_end_speed_ratio": direct["seconds"] / run["seconds"] if run and run["valid_workload"] and quality["certified"] else None})
    return {"schema_version": 1, "protocol_sha256": protocol, "complete": complete,
            "correctness": quality, "modes": modes, "paired_comparisons": paired, "runs": records,
            "performance_claim": "not certified" if not (complete and quality["certified"]) else "measured; inspect paired repetitions and backend coordinate parity",
            "cold_and_steady": "Each CLI run starts cold; per-backend/shape/phase cold and steady observations, loading counters, and control contexts are in runs[].runtime_diagnostics. No extra warm-up samples are generated.",
            "timing": "independent monotonic process-tree observation at 30 ms resolution; no profiler in timed comparison"}


def markdown_report(result):
    def number(value, digits=4):
        return "unavailable" if value is None else f"{value:.{digits}f}"
    lines = ["# Fixed-workload runtime comparison", "",
             f"Protocol SHA-256: `{result['protocol_sha256']}`.", "",
             f"Complete fixed workload: **{result['complete']}**. Scientific score/decision gate: **{result['correctness']['certified']}**.", "",
             "| Mode | Valid / attempted trials | Completed trajectories | Median end-to-end seconds | Effective trajectories / second | Failed or incomplete trials | Allocated GPU hours |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, data in result["modes"].items():
        lines.append(f"| {name} | {data['valid_runs']} / {data['runs']} | {data['completed_trajectories']} | {number(data['median_end_to_end_seconds'])} | {number(data['effective_trajectories_per_second'], 6)} | {data['failed_or_incomplete_runs']} | {number(data['allocated_gpu_hours'], 6)} |")
    lines += ["", "## Interpretation", "", result["cold_and_steady"], "",
              "Scores, selected sequences, stage and acceptance decisions must pass the repeated-direct baseline comparison before effective throughput is reported. Tolerances come from observed direct-repeat error; decision differences are checked separately. Coordinate and cache parity still require the backend's conformance evidence.", "",
              "A ratio above 1 in the paired comparisons means the measured run finished faster. The runs include startup, all configured tasks, and shutdown. GPU activity is diagnostic and is not the throughput objective. CPU sampling may miss short-lived children; device activity and VRAM include other jobs unless allocation is exclusive.", "",
              "Completion timing uses an independent process observer at 30 ms resolution. Allocated GPU cost also charges the trial's observation envelope; report.json records the entire comparison-loop allocation cost. Uncertain release makes resource costs observed lower bounds and stops subsequent trials.", "",
              "## Paired end-to-end comparisons", "",
              "| Repetition | Mode | Direct seconds / mode seconds |", "| ---: | --- | ---: |"]
    for pair in result["paired_comparisons"]:
        lines.append(f"| {pair['repetition']} | {pair['mode']} | {number(pair['end_to_end_speed_ratio'])} |")
    lines += ["", "## Evidence", "",
              "`plan.json` records the resolved scientific configuration, allocation, source identity, command and trial order. `report.json` retains all scientific rows, correctness differences, per-trial costs and runtime diagnostics. Each trial directory preserves stdout, stderr, raw telemetry and any runtime JSONL timeline. No profiler was enabled in timed runs.", "",
              "Model load counters, cold/steady phase observations, resource peaks, release records and sampled low-activity contexts are retained under `runs[].runtime_diagnostics`. Missing counters stay unavailable. Composite native-call wall time is not labeled as device execution. A sampled control context does not prove an exclusive cause of GPU idleness.", ""]
    if not result["correctness"]["certified"]:
        lines += ["## Correctness or completion limitation", "", "```json", json.dumps(result["correctness"], indent=2), "```", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Existing standalone scientific workload YAML.")
    parser.add_argument("--config-dir", type=Path, default=ROOT / "config")
    parser.add_argument("--direct-repo", type=Path, help="Frozen legacy checkout for direct trials; otherwise use this checkout's legacy execution path.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3, help="Direct/serial/adaptive repetitions; all count against budget.")
    parser.add_argument("--walltime-budget", type=float, help="Required total seconds for execution, including failed trials.")
    parser.add_argument("--max-output-bytes", type=int, default=2 * 1024**3)
    parser.add_argument("--sample-interval", type=float, default=0.5)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    if args.repetitions < 2 or args.sample_interval < .1 or args.max_output_bytes < 1:
        parser.error("need >=2 repetitions, sample_interval>=0.1 and positive output budget")
    if not args.plan_only and (args.walltime_budget is None or args.walltime_budget <= 0):
        parser.error("execution requires an explicit positive --walltime-budget")
    from astrolabe.runtime.admission import allocated_devices
    try:
        allocation = ",".join(allocated_devices(os.environ.get("CUDA_VISIBLE_DEVICES", "")))
    except ValueError as exc:
        parser.error(str(exc))
    if args.require_gpu and not allocation:
        parser.error("--require-gpu needs an inherited CUDA_VISIBLE_DEVICES job allocation")
    if not args.plan_only:
        from astrolabe.runtime.telemetry import process_namespace_supported
        if not process_namespace_supported():
            parser.error("execution needs matching process and /proc namespaces for safe measurement and cleanup")
    args.config = args.config.resolve()
    args.out = args.out.resolve()
    direct_repo = args.direct_repo.resolve() if args.direct_repo else ROOT
    if not (direct_repo / "astrolabe/__main__.py").is_file() or not (direct_repo / "run.py").is_file():
        parser.error("--direct-repo must be an existing Astrolabe checkout")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("output directory must be empty; prior evidence will not be overwritten")
    base = compose(args.config, args.config_dir.resolve())
    try:
        trajectories = planned_trajectories(base)
    except ValueError as exc:
        parser.error(str(exc))
    expected = len(trajectories)
    expected_identities = {str((row["repeat_index"], row["seed"])) for row in trajectories}
    protocol = fingerprint(scientific_projection(base))
    from omegaconf import OmegaConf
    runs = []
    for repetition in range(args.repetitions):
        order = MODES[repetition % 3:] + MODES[:repetition % 3]
        for mode in order:
            directory = args.out / f"{len(runs)+1:03d}_{mode}"
            try:
                config = mode_config(base, mode, directory)
            except ValueError as exc:
                parser.error(str(exc))
            config_path = directory / "config.yaml"
            directory.mkdir(parents=True, exist_ok=False)
            OmegaConf.save(OmegaConf.create(config), config_path)
            repository = direct_repo if mode == "direct" else ROOT
            # Pin imports even if the scientific config's directory contains
            # another checkout. CWD stays unchanged for relative input assets.
            command = [sys.executable, "-c", "import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); runpy.run_module('astrolabe',run_name='__main__')", str(repository), "--config-dir", str(args.config_dir.resolve()),
                       "run", "--standalone", str(config_path), "--no-chdir"]
            runs.append({"run": len(runs) + 1, "mode": mode, "repetition": repetition,
                         "directory": str(directory), "command": command, "repository": str(repository)})
    environment = execution_environment()
    json_write(args.out / "plan.json", {"protocol_sha256": protocol, "configuration": base,
                                       "execution_environment": environment,
                                       "direct_environment": execution_environment(direct_repo),
                                       "direct_baseline_kind": "frozen separate checkout" if args.direct_repo else "current checkout legacy path",
                                       "allocation": allocation, "working_directory": str(args.config.parent),
                                       "fixed_trajectories_per_trial": expected, "runs": runs,
                                       "expected_trajectories": trajectories,
                                       "walltime_budget_seconds": args.walltime_budget})
    if args.plan_only:
        print(args.out / "plan.json")
        return 0
    experiment_started, records = time.monotonic(), []
    deadline = experiment_started + args.walltime_budget
    for trial in runs:
        if time.monotonic() >= deadline:
            break
        used_output = sum(path.stat().st_size for path in args.out.rglob("*") if path.is_file())
        remaining_output = args.max_output_bytes - used_output
        if remaining_output <= 0:
            break
        trial_started = time.monotonic()
        try:
            result = run_once(trial["command"], Path(trial["directory"]), args.config.parent,
                              allocation, deadline, args.sample_interval, remaining_output, Path(trial["repository"]))
        except OSError as exc:
            result = {"exit_code": None, "seconds": time.monotonic() - trial_started,
                      "stop_reason": "execution_start_or_measurement_failed", "error": str(exc),
                      "allocated_gpu_seconds": len(allocation.split(",")) * (time.monotonic() - trial_started) if allocation else 0}
        try:
            rows, failures = read_completed(trial["directory"], expected_identities)
            structure_artifacts = collect_structures(trial["directory"])
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            rows, failures = {}, [{"kind": "invalid_result_manifest", "error": str(exc)}]
            structure_artifacts = {}
        result.update(trial, scientific_rows=rows, structure_artifacts=structure_artifacts,
                      completed=len(rows), failures=failures)
        result["valid_workload"] = (result["exit_code"] == 0 and len(rows) == expected and not failures
                                    and not result["stop_reason"]
                                    and result.get("process_release", {}).get("release_confirmed") is True)
        records.append(result)
        json_write(args.out / "report.json", report(records, len(runs), protocol))
        if result["stop_reason"]:
            break
    final = report(records, len(runs), protocol)
    experiment_seconds = time.monotonic() - experiment_started
    final["experiment"] = {"wall_seconds": experiment_seconds,
                           "allocated_gpu_hours": (len(allocation.split(",")) * experiment_seconds / 3600) if allocation else 0,
                           "walltime_budget_seconds": args.walltime_budget,
                           "scope": "entire comparison loop, including between-trial analysis and failed work; per-mode GPU cost includes each trial's observation envelope"}
    json_write(args.out / "report.json", final)
    (args.out / "report.md").write_text(markdown_report(final))
    print(args.out / "report.json")
    return 0 if final["complete"] and final["correctness"]["certified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
