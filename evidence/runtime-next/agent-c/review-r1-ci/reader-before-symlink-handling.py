#!/usr/bin/env python3
"""Independently audit persisted W0 CI artifacts; never run a benchmark.

Counts/parity come from every original CSV, provenance, config and runtime trace.
Only individual observer records are read from comparison/report.json; author
aggregates, embedded scientific_rows, correctness and summary files are unused.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import subprocess
import tarfile

from omegaconf import OmegaConf


MODES = ("direct", "resident_serial", "resident_adaptive")
VARIANTS = ("baseline", "candidate")
HEADS = {"baseline": "d66ee714f2e755948534f95d9d125287ce8dbd91",
         "candidate": "71c7592e27d02b2772d984d9b1f5ed9fea47fd6c"}
IDENTITIES = {(0, 17), (1, 18), (2, 19)}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fingerprint(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def plain(value):
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [plain(item) for item in value]
    return str(value) if isinstance(value, Path) else value


def read_yaml(path):
    return plain(OmegaConf.to_container(OmegaConf.load(path), resolve=False))


def source_blobs(repo, head, names):
    archive = subprocess.check_output(["git", "-C", str(repo), "archive", "--format=tar", head, "--", *names])
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as handle:
        return {item.name: handle.extractfile(item).read() for item in handle if item.isfile()}


def audit(args):
    root, out = args.artifacts.resolve(), args.out.resolve()
    if out == root or root in out.parents:
        raise ValueError("write evidence outside the original artifact tree")
    out.mkdir(parents=True, exist_ok=False)
    files = sorted(path for path in root.rglob("*") if path.is_file())
    inventory = {str(path.relative_to(root)): {"bytes": path.stat().st_size, "sha256": sha(path.read_bytes())}
                 for path in files}
    write_json(out / "input-manifest.json", inventory)
    comparison = root / "comparison"
    plan = read_json(comparison / "plan.json")
    # These are the persisted per-trial observations, not aggregate conclusions.
    measured = read_json(comparison / "report.json")["runs"]
    errors = []

    def check(condition, message):
        if not condition:
            errors.append(message)

    check(plan["requested_trials"] == 30 and plan["groups"] == 5, "frozen trial budget")
    check(plan["evaluations_per_trial"] == 9 and plan["trajectories_per_trial"] == 3, "frozen work budget")
    check(plan["gpu_allocation"] == "", "unexpected GPU allocation")
    base = plan["configuration"]
    check((base["seed"], base["steps"], base["repeats"], base["seqs"]) == (17, 3, 3, {"A": "AAAA"}),
          "unexpected scientific fixture")
    source_checks = {}
    for variant, head in HEADS.items():
        source = plan["sources"][variant]
        check(source["git_commit"] == head and source["git_status"] == "", variant + " frozen source")
        expected = source["source_manifest"]
        actual = source_blobs(args.repo, head, list(expected))
        mismatches = [name for name, digest in expected.items() if sha(actual[name]) != digest]
        check(not mismatches, variant + " source bytes do not match Git")
        source_checks[variant] = {"head": head, "verified_files": len(expected), "mismatches": mismatches}
    for path, digest in plan["measurement_source_sha256"].items():
        for variant, head in HEADS.items():
            check(sha(source_blobs(args.repo, head, [path])[path]) == digest, variant + " measurement changed: " + path)
    wrapper = "benchmarks/runtime/validate.py"
    check(sha(source_blobs(args.repo, HEADS["candidate"], [wrapper])[wrapper]) == plan["wrapper_sha256"],
          "comparison wrapper differs from frozen source")
    prior_checks = []
    for prior in args.prior:
        path = prior.resolve() / "comparison/plan.json"
        old = read_json(path)
        same = {key: old[key] == plan[key] for key in
                ("configuration", "protocol_sha256", "measurement_source_sha256", "requested_trials",
                 "evaluations_per_trial", "expected_trajectories", "trajectories_per_trial",
                 "walltime_budget_seconds", "max_output_bytes")}
        check(all(same.values()), "workload/measurement/budget changed since " + str(prior))
        prior_checks.append({"plan": str(path), "sha256": sha(path.read_bytes()), "equal_fields": same})

    expected_order = []
    for group in range(5):
        modes = MODES[group % 3:] + MODES[:group % 3]
        variants = VARIANTS if group % 2 == 0 else VARIANTS[::-1]
        expected_order.extend((group, variant, mode) for mode in modes for variant in variants)
    observed_order = [(row["repetition"], row["variant"], row["mode"]) for row in plan["runs"]]
    check(observed_order == expected_order, "frozen interleaved order differs")
    observed_measurements = {row["run"]: row for row in measured}
    check(len(measured) == len(observed_measurements) == 30, "missing/duplicate measurement trials")
    directories = {path.name for path in comparison.iterdir() if path.is_dir()}
    check(directories == {Path(row["directory"]).name for row in plan["runs"]}, "missing/extra trial directories")

    counts = Counter()
    reference_rows, reference_csv, reference_run_config = {}, {}, None
    trials, outcomes, hardware = [], {}, set()
    previous_completion = None
    for scheduled in plan["runs"]:
        name = Path(scheduled["directory"]).name
        directory = comparison / name
        observation = observed_measurements[scheduled["run"]]
        check(all(observation[key] == value for key, value in scheduled.items()), name + " observation/plan identity")
        mode, variant, group = scheduled["mode"], scheduled["variant"], scheduled["repetition"]
        check(name == f"{scheduled['run']:03d}_{variant}_{mode}", name + " directory identity")
        expected = copy.deepcopy(base)
        expected["out_dir"] = scheduled["directory"] + "/results"
        expected["raw_model_results_dir"] = scheduled["directory"] + "/raw"
        expected["execution"].update(enabled=mode != "direct", max_active="auto" if mode == "resident_adaptive" else 1,
                                     max_resident_models="auto", max_resident_per_device=1 if mode == "direct" else "auto")
        trial_config = read_yaml(directory / "config.yaml")
        check(trial_config == expected, name + " full trial config differs beyond explicit mode/path values")
        check(sha((directory / "config.yaml").read_bytes()) == scheduled["config_sha256"], name + " frozen config hash")

        identities, csv_hashes = set(), {}
        provenance_files = sorted((directory / "results").rglob("provenance.json"))
        check(len(provenance_files) == 3, name + " trajectory directory budget")
        for provenance_file in provenance_files:
            run_dir = provenance_file.parent
            provenance = read_json(provenance_file)
            identity = (provenance["run"]["repeat_index"], provenance["run"]["seed"])
            check(identity in IDENTITIES and identity not in identities, name + " trajectory identity")
            identities.add(identity)
            check(provenance["state"] == "completed" and provenance["terminal"] is True and provenance["error"] is None,
                  name + " incomplete trajectory")
            check(provenance["git"]["commit"] == HEADS[variant] and provenance["git"]["dirty"] is False,
                  name + " executed source identity")
            check(provenance["run"]["repeats"] == 3 and provenance["run"]["task_name"] == "runtime_cpu_protocol",
                  name + " executed trajectory budget")
            run_config = read_yaml(run_dir / "config.yaml")
            check(sha((run_dir / "config.yaml").read_bytes()) == provenance["config"]["sha256"], name + " provenance config hash")
            check((run_config["repeat_index"], run_config["seed"]) == identity, name + " executed config identity")
            check(run_config["execution"] == trial_config["execution"], name + " executed mode config")
            check(run_config["out_dir"] == provenance["run"]["run_dir"], name + " executed output owner")
            check(run_config["raw_model_results_dir"] == scheduled["directory"] + f"/raw/repeat_{identity[0]+1:03d}",
                  name + " executed raw-output owner")
            for key in ("out_dir", "raw_model_results_dir", "repeat_index", "seed"):
                run_config.pop(key)
            run_config["execution"] = base["execution"]
            if reference_run_config is None:
                reference_run_config = run_config
            check(run_config == reference_run_config, name + " executed scientific configuration changed")
            hardware.add(fingerprint({key: provenance[key] for key in ("hardware", "python", "dependencies")}))
            check(provenance["hardware"]["device_visibility_env"]["CUDA_VISIBLE_DEVICES"] == "", name + " GPU visibility")
            with (run_dir / "scores.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            check([row["step"] for row in rows] == ["0", "1", "2"], name + " missing/extra evaluation row")
            check(all(math.isfinite(float(row["weighted_score"])) for row in rows), name + " nonfinite score")
            key = str(identity)
            reference_rows.setdefault(key, rows)
            raw_csv_hash = sha((run_dir / "scores.csv").read_bytes())
            reference_csv.setdefault(key, raw_csv_hash)
            check(rows == reference_rows[key], name + " scientific rows differ for " + key)
            check(raw_csv_hash == reference_csv[key], name + " CSV bytes differ for " + key)
            csv_hashes[key] = raw_csv_hash
            counts["completed_trajectories"] += 1
            for row in rows:
                initial = int(row["step"]) == 0
                accepted = row["accepted"].lower() == "true"
                counts["scientific_rows"] += 1
                counts["initial_rows" if initial else "mutation_attempts"] += 1
                counts["accepted_initial_rows" if initial else "accepted_mutations"] += int(accepted)
                counts["stage_entry_rows"] += int(row["note"] == "enter stage")
                counts["accepted_improving_mutations"] += int(not initial and accepted and
                                                               float(row["weighted_score"]) > float(rows[0]["weighted_score"]))
        check(identities == IDENTITIES, name + " repeat/seed set")
        events = [json.loads(line) for line in (directory / "runtime.jsonl").read_text().splitlines()]
        phases = Counter(event["phase"] for event in events if event.get("event") == "stage")
        check(dict(phases) == {phase: 9 for phase in ("cpu_prepare", "cpu_execute", "cpu_finalize")}, name + " actual evaluation budget")
        counts["evaluation_calls"] += phases["cpu_execute"]
        stop_events = [event for event in events if event.get("event") == "broker_stopped"]
        check(len(stop_events) == int(mode != "direct"), name + " broker terminal count")
        if stop_events:
            stopped = stop_events[0]
            check(stopped["cleanup_confirmed"] is True and not stopped["cleanup_errors"] and not stopped["unreleased"], name + " broker cleanup")
            check(all(not value for key, value in stopped["resources"].items() if key != "limits"), name + " terminal resource ledger")
            counts["empty_terminal_broker_ledgers"] += 1
        release = observation["process_release"]
        check(observation["exit_code"] == release["exit_code"] == 0 and observation["stop_reason"] is None, name + " exit")
        check(release["release_confirmed"] is True and release["observed_tree_empty"] is True and
              not release["observation_errors"] and not release["remaining_identities"] and not release["unverified_process_groups"],
              name + " process release")
        samples = [json.loads(line) for line in (directory / "telemetry.jsonl").read_text().splitlines()]
        check(all(sample["gpu"]["available"] is False for sample in samples), name + " GPU telemetry")
        origins = [sample["monotonic_ns"] / 1e9 - sample["elapsed_seconds"] for sample in samples]
        origin = statistics.median(origins)
        reconstructed = release["completion_monotonic"] - origin
        seconds = observation["seconds"]
        check(seconds > 0 and abs(reconstructed - seconds) < .0001, name + " elapsed/release/telemetry disagreement")
        if previous_completion is not None:
            check(origin > previous_completion, name + " benchmark trials overlap")
        previous_completion = release["completion_monotonic"]
        counts["completed_trials"] += 1
        trials.append({"trial": name, "group": group, "variant": variant, "mode": mode,
                       "seconds": seconds, "reconstructed_seconds": reconstructed,
                       "reconstruction_error_seconds": reconstructed - seconds,
                       "completed_trajectories": len(identities), "evaluation_calls": phases["cpu_execute"],
                       "csv_sha256_by_identity": csv_hashes})
        outcomes[(group, variant, mode)] = seconds
    check(len(hardware) == 1, "provenance hardware/Python/dependencies differ")
    for name, number in {"completed_trials": 30, "completed_trajectories": 90, "evaluation_calls": 270,
                         "scientific_rows": 270, "initial_rows": 90, "accepted_initial_rows": 90,
                         "mutation_attempts": 180, "accepted_mutations": 0, "accepted_improving_mutations": 0,
                         "stage_entry_rows": 0, "empty_terminal_broker_ledgers": 20}.items():
        check(counts[name] == number, "unexpected total " + name)
    groups, pairs = {}, {}
    for mode in MODES:
        times = {}
        for variant in VARIANTS:
            values = [outcomes[(group, variant, mode)] for group in range(5)]
            times[variant] = values
            groups[variant + "/" + mode] = {"seconds": values, "median_seconds": statistics.median(values),
                                           "range_seconds": [min(values), max(values)],
                                           "completed_trajectories_per_second": 15 / sum(values)}
        ratios = [before / after for before, after in zip(times["baseline"], times["candidate"])]
        median = statistics.median(ratios)
        pairs[mode] = {"speed_ratios_by_group": ratios, "median_paired_speed_ratio": median,
                       "median_paired_throughput_improvement_percent": 100 * (median - 1),
                       "median_paired_time_reduction_percent": 100 * (1 - 1 / median),
                       "faster_pairs": sum(value > 1 for value in ratios),
                       "strict_range_separation": max(times["candidate"]) < min(times["baseline"]),
                       "range_gap_seconds": min(times["baseline"]) - max(times["candidate"])}
    best_direct = min(VARIANTS, key=lambda variant: groups[variant + "/direct"]["median_seconds"])
    direct_comparison = {}
    for mode in MODES[1:]:
        ratios = [outcomes[(group, best_direct, "direct")] / outcomes[(group, "candidate", mode)] for group in range(5)]
        direct_comparison[mode] = {"best_direct_variant": best_direct, "speed_ratios_by_group": ratios,
                                  "median_speed_ratio": statistics.median(ratios),
                                  "median_resident_time_multiplier": statistics.median(1 / ratio for ratio in ratios),
                                  "resident_faster_pairs": sum(ratio > 1 for ratio in ratios),
                                  "twenty_percent_throughput_goal_met": statistics.median(ratios) >= 1.2}
    # Raw inputs are immutable during this read-only review.
    check(all(sha((root / name).read_bytes()) == record["sha256"] for name, record in inventory.items()), "raw artifacts changed during review")
    result = {"status": "DONE" if not errors else "PARTIAL", "errors": errors,
              "reviewer": "Agent C", "benchmark_rerun": False, "source_checks": source_checks,
              "raw_root": str(root), "raw_files": len(files), "input_manifest_sha256": sha((out / "input-manifest.json").read_bytes()),
              "protocol_sha256": plan["protocol_sha256"], "prior_plan_checks": prior_checks,
              "counts": dict(counts), "full_trial_configs_verified": 30, "full_executed_configs_verified": 90,
              "scientific_rows_exact_by_repeat_seed": not any("scientific rows differ" in error for error in errors),
              "science_reference": {key: {"all_columns_sha256": fingerprint(rows), "csv_sha256": reference_csv[key],
                                           "trajectory": [{field: row[field] for field in
                                                           ("step", "pA_sequence", "weighted_score", "accepted", "stage", "note")}
                                                          for row in rows]} for key, rows in reference_rows.items()},
              "groups": groups, "same_mode_pairs": pairs, "candidate_resident_vs_best_direct": direct_comparison,
              "all_same_mode_median_speedups_below_twenty_percent": all(pair["median_paired_speed_ratio"] < 1.2 for pair in pairs.values()),
              "max_reconstructed_time_error_seconds": max(abs(row["reconstruction_error_seconds"]) for row in trials),
              "trials": trials,
              "limitations": ["CPU Dummy workload; no pretrained or GPU inference",
                              "Timing is reanalysis of persisted individual observer measurements, not new independent timing",
                              "Five interleaved cold-start samples per source/mode on one runner; no population confidence claim",
                              "Zero accepted mutations or stage changes; completion throughput is not scientific progress",
                              "No structures, full RNG snapshot, checkpoint or retry recovery certification in W0",
                              "Process observer proves release of its observed scope, not arbitrary escaped processes"]}
    write_json(out / "report.json", result)
    print(json.dumps({"status": result["status"], "errors": errors, "counts": result["counts"],
                      "same_mode_pairs": pairs, "candidate_resident_vs_best_direct": direct_comparison}, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--prior", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    raise SystemExit(audit(parser.parse_args()))
