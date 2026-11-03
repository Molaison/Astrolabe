"""Reproduce descriptive CPU evidence from the preserved raw logs (stdlib only)."""
from collections import Counter
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def summarize_attempt(name):
    directory = ROOT / name
    report_path = directory / "report.json"
    report = json.loads(report_path.read_text())
    plan = json.loads((directory / "plan.json").read_text())
    runs = []
    for record in report["runs"]:
        trial = directory / Path(record["directory"]).name
        telemetry = [json.loads(line) for line in (trial / "telemetry.jsonl").read_text().splitlines()]
        previous, peaks = None, []
        for sample in telemetry:
            value = sample["process"].get("sampled_cpu_seconds")
            stamp = sample["elapsed_seconds"]
            if value is None:
                continue
            if previous is not None and stamp > previous[0] and value >= previous[1]:
                peaks.append((value - previous[1]) / (stamp - previous[0]))
            previous = (stamp, value)
        trace = trial / "runtime.jsonl"
        events = [json.loads(line) for line in trace.read_text().splitlines()] if trace.is_file() else []
        controls = [event for event in events if event.get("event") == "controller"]
        phases = Counter()
        phase_seconds = Counter()
        for event in events:
            if event.get("event") == "stage":
                phases[event.get("phase", "unknown")] += 1
                duration = event.get("duration")
                if isinstance(duration, (float, int)):
                    phase_seconds[event.get("phase", "unknown")] += duration
        runs.append({"run": record["run"], "mode": record["mode"],
                     "seconds": record["seconds"], "completed": record["completed"],
                     "valid_workload": record["valid_workload"],
                     "release_confirmed": record["process_release"]["release_confirmed"],
                     "sampled_peak_cpu_core_equivalents": max(peaks) if peaks else None,
                     "maximum_controller_limit": max([event["limit"] for event in controls] or [0]) if controls else None,
                     "controller_reasons": dict(Counter(event.get("reason") for event in controls)),
                     "observed_model_start_events": sum(event.get("event") == "model_loaded" for event in events),
                     "stage_counts": dict(phases), "stage_wall_seconds": dict(phase_seconds),
                     "telemetry_sha256": hashlib.sha256((trial / "telemetry.jsonl").read_bytes()).hexdigest()})
    modes = {}
    for mode, original in report["modes"].items():
        peaks = [run["sampled_peak_cpu_core_equivalents"] for run in runs if run["mode"] == mode
                 and run["sampled_peak_cpu_core_equivalents"] is not None]
        modes[mode] = dict(original, sampled_peak_cpu_core_equivalents=max(peaks) if peaks else None,
                           peak_process_tree_rss_gib=original["peak_process_tree_rss_bytes"] / 1024**3)
    return {"report": name + "/report.json", "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            "protocol_sha256": report["protocol_sha256"],
            "complete": report["complete"], "score_decision_gate": report["correctness"]["certified"],
            "coordinate_gate": report["correctness"].get("delivered_structure_parity"),
            "source_manifest_sha256": plan["execution_environment"]["source_manifest_sha256"],
            "direct_source_manifest_sha256": plan["direct_environment"]["source_manifest_sha256"],
            "fixed_identities": plan["expected_trajectories"], "allocation": plan["allocation"],
            "cpu_affinity": plan["execution_environment"]["cpu_affinity"],
            "modes": modes, "runs": runs, "experiment": report["experiment"]}


def main():
    attempts = [summarize_attempt(name) for name in ("cpu-three-mode-01", "cpu-three-mode-02")]
    assert len({attempt["protocol_sha256"] for attempt in attempts}) == 1
    assert len({attempt["direct_source_manifest_sha256"] for attempt in attempts}) == 1
    assert all(attempt["allocation"] == "" for attempt in attempts)
    result = {"scope": "CPU Dummy/MCMC workflow only; no pretrained model or GPU certification",
              "metric_definitions": {
                  "cpu_peak": "Maximum delta of sampled cumulative process-tree CPU seconds / elapsed sample interval. Approximate 0.5 s windows; not instantaneous peak; discovery and short-lived children limit accuracy.",
                  "rss_peak": "Sum of sampled process RSS, which can double-count shared pages; not physical RAM/PSS.",
                  "throughput": "Completed trajectories in valid fixed trials / total trial completion seconds, gated by repeated-direct score/decision equivalence.",
                  "cold_steady": "All CLI trials start fresh. No neural model in this fixture; GPU model cold/steady throughput and residency are not applicable.",
                  "model_loads": "Dummy has no inference weights; zero model-start events are not evidence about any GPU backend."},
              "attempts": attempts,
              "total_experiment_wall_seconds_including_failure": sum(attempt["experiment"]["wall_seconds"] for attempt in attempts),
              "total_completed_trajectories_including_uncertified_attempt": sum(run["completed"] for attempt in attempts for run in attempt["runs"]),
              "original_total_walltime_budget_seconds": 900,
              "all_allocated_gpu_hours": sum(attempt["experiment"]["allocated_gpu_hours"] for attempt in attempts)}
    (ROOT / "cpu-evidence-summary.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"total_seconds": result["total_experiment_wall_seconds_including_failure"],
                      "final_modes": attempts[-1]["modes"]}, indent=2))


if __name__ == "__main__":
    main()
