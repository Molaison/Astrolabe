#!/usr/bin/env python3
"""Run two warm canary requests, then validate both against seed-bound references."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "astrolabe.na-mpnn-daemon-timing-warm-final/v1"
PREFLIGHT_SCHEMA = f"{SCHEMA}.preflight"
PROFILE = "na_mpnn_daemon_timing_v1_warm_final_003"
PREFLIGHT_NAME = "preflight.json"
REPORT_NAME = "acceptance.json"
REQUEST_PLANS = (
    {
        "request_index": 0,
        "seed": 267850364,
        "timing_label": "fresh_owner_cold",
    },
    {
        "request_index": 1,
        "seed": 267850363,
        "timing_label": "same_owner_warm",
    },
)
HARD_REQUEST_LIMIT = 2

from scripts.validation import na_mpnn_daemon_timing_acceptance as timing  # noqa: E402
from scripts.validation.na_mpnn_gpu_acceptance import (  # noqa: E402
    AcceptanceError,
    _runner_config,
    _validate_probability_bundle,
    capture_runner_result,
)
from scripts.validation.model_gpu_throughput import (  # noqa: E402
    _flatten_arrays,
    _sha256,
    _value_digest,
)


class FinalWarmError(AcceptanceError):
    """The final warm canary plan cannot be proven."""


def request_plans() -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            **plan,
            "arm": "daemon",
            "temperature": timing.base.TEMPERATURE,
            "batch_size": timing.base.BATCH_SIZE,
            "number_of_batches": timing.base.NUMBER_OF_BATCHES,
            "chains_to_design": "A",
            "parse_these_chains_only": "ABC",
            "design_na_only": 1,
            "parse_na_only": 0,
            "na_shared_tokens": 0,
            "fixed_residues": list(timing.base.FIXED_RESIDUES),
            "design_mask": list(timing.base.EXPECTED_DESIGN_MASK),
        }
        for plan in REQUEST_PLANS
    )


def _file_record(path: Path) -> dict[str, Any]:
    return timing._file_record(path)


def _load_runner():
    from astrolabe.mutators.wrappers.na_mpnn_runner import NAMPNNRunner

    return NAMPNNRunner.__new__(NAMPNNRunner)


def _reference_map(
    reference_root: Path,
    input_data: Mapping[str, Any],
) -> dict[str, Any]:
    root = reference_root.expanduser().resolve()
    manifest_path = root / timing.REFERENCE_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != f"{timing.SCHEMA}.native-reference":
        raise FinalWarmError("Old-native reference manifest schema changed")
    declared_files = manifest.get("files")
    seed_references = manifest.get("seed_references")
    if not isinstance(declared_files, Mapping):
        raise FinalWarmError("Old-native reference file map is missing")
    if not isinstance(seed_references, Mapping):
        raise FinalWarmError("Old-native seed reference map is missing")
    expected_seeds = {str(plan["seed"]) for plan in REQUEST_PLANS}
    if set(seed_references) != expected_seeds:
        raise FinalWarmError(
            "Old-native seed map does not match the complete run plan"
        )
    loader = _load_runner()
    bindings: dict[str, Any] = {}
    for plan in REQUEST_PLANS:
        seed_text = str(plan["seed"])
        seed_map = seed_references[seed_text]
        if not isinstance(seed_map, Mapping):
            raise FinalWarmError(f"Reference map for seed {seed_text} is invalid")
        fasta_relative = str(seed_map.get("fasta") or "")
        npz_relative = str(seed_map.get("specificity_npz") or "")
        if not fasta_relative or not npz_relative:
            raise FinalWarmError(f"Seed {seed_text} lacks explicit FASTA/NPZ paths")
        fasta_path = root / fasta_relative
        npz_path = root / npz_relative
        if not fasta_path.is_file() or not npz_path.is_file():
            raise FinalWarmError(f"Seed {seed_text} reference files are missing")
        fasta_sha = _sha256(fasta_path)
        npz_sha = _sha256(npz_path)
        if declared_files.get(fasta_relative) != fasta_sha:
            raise FinalWarmError(f"Seed {seed_text} FASTA SHA changed")
        if declared_files.get(npz_relative) != npz_sha:
            raise FinalWarmError(f"Seed {seed_text} NPZ SHA changed")
        entries = loader._parse_fasta(fasta_path)
        header_seeds = [
            int(float(entry["metadata"]["seed"]))
            for entry in entries
            if entry["metadata"].get("seed") is not None
        ]
        if len(header_seeds) != len(entries) or set(header_seeds) != {
            int(plan["seed"])
        }:
            raise FinalWarmError(
                f"Seed {seed_text} FASTA header seed binding is invalid"
            )
        bundle = loader._load_specificity_matrix(
            npz_path.parent.parent,
            Path("step_000_ABC_model.cif"),
        )
        if not isinstance(bundle, Mapping):
            raise FinalWarmError(f"Seed {seed_text} reference bundle is missing")
        mask = _validate_probability_bundle(
            bundle,
            seed=int(plan["seed"]),
            input_data=input_data,
        )
        field_records = {
            key: {
                "sha256": _value_digest(value),
                "dtype": str(np.asarray(value).dtype),
                "shape": list(np.asarray(value).shape),
            }
            for key, value in sorted(bundle.items())
        }
        bindings[seed_text] = {
            "seed": int(plan["seed"]),
            "fasta": str(fasta_path.resolve()),
            "fasta_relative": fasta_relative,
            "fasta_sha256": fasta_sha,
            "fasta_header_seeds": header_seeds,
            "specificity_npz": str(npz_path.resolve()),
            "specificity_relative": npz_relative,
            "specificity_sha256": npz_sha,
            "reference_fields": field_records,
            "fixed_mask": mask,
        }
    return {
        "root": str(root),
        "manifest": _file_record(manifest_path),
        "bindings": bindings,
    }


def _prior_completed_request(
    root: Path,
    *,
    seed: int,
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    directory = f"daemon_00_seed_{seed}"
    required = (
        root / "resident-requests.jsonl",
        root / "requests/request_00_caller_before.json",
        root / "requests/daemon/request_00_before.json",
        root / "requests/daemon/request_00_after.json",
        root / "requests/request_00_raw_after.json",
        root / "requests/request_00_live_arrays.npz",
        root / "daemon" / directory / "seqs/step_000_ABC_model.fa",
        root / "daemon" / directory / "specificity/step_000_ABC_model.npz",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FinalWarmError(f"Prior completed-request evidence is missing: {missing}")
    if list((root / "requests").glob("request_01_*")) or list(
        (root / "requests/daemon").glob("request_01_*")
    ):
        raise FinalWarmError(f"Prior request directory contains second-request markers: {root}")
    ledger = [
        json.loads(line)
        for line in (root / "resident-requests.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    if len(ledger) != 1:
        raise FinalWarmError(f"Prior ledger does not contain exactly one request: {root}")
    record = ledger[0]
    if (
        record.get("status") != "completed"
        or int(record.get("request_count", -1)) != 1
        or int(record.get("request_delta", -1)) != 1
        or int(record.get("model_load_count", -1)) != 1
        or str(record.get("gpu_uuid") or "") == ""
    ):
        raise FinalWarmError(f"Prior completed owner ledger is invalid: {root}")
    loader = _load_runner()
    reference_bundle = loader._load_specificity_matrix(
        Path(str(reference["specificity_npz"])).parent.parent,
        Path("step_000_ABC_model.cif"),
    )
    live_bundle = loader._load_specificity_matrix(
        root / "daemon" / directory,
        Path("step_000_ABC_model.cif"),
    )
    from scripts.validation.model_gpu_throughput import compare_outputs

    comparison = compare_outputs(reference_bundle, live_bundle)
    fasta_equal = (
        Path(str(reference["fasta"])).read_bytes()
        == (root / "daemon" / directory / "seqs/step_000_ABC_model.fa").read_bytes()
    )
    if not fasta_equal or not comparison["equal"]:
        raise FinalWarmError(f"Prior completed seed {seed} is not exact")
    caller_before = json.loads(
        (root / "requests/request_00_caller_before.json").read_text()
    )
    daemon_before = json.loads(
        (root / "requests/daemon/request_00_before.json").read_text()
    )
    daemon_after = json.loads(
        (root / "requests/daemon/request_00_after.json").read_text()
    )
    if not (
        caller_before["gpu"]["uuid"]
        == daemon_before["owner"]["gpu"]["uuid"]
        == daemon_after["owner"]["gpu"]["uuid"]
        == record["gpu_uuid"]
    ):
        raise FinalWarmError(f"Prior seed {seed} GPU UUID evidence disagrees")
    return {
        "path": str(root.resolve()),
        "seed": seed,
        "status": "external_validation_PASS",
        "request_count": 1,
        "request_delta": 1,
        "model_load_count": 1,
        "pid": int(record["pid"]),
        "model_token": int(record["model_token"]),
        "gpu_uuid": str(record["gpu_uuid"]),
        "fasta_equal": fasta_equal,
        "all_ten_fields_exact": bool(comparison["equal"]),
        "evidence": {
            str(path.relative_to(root)): _file_record(path)
            for path in required
        },
    }


def _execution_source_manifest() -> dict[str, str]:
    records = dict(timing._execution_source_manifest())
    relative = str(Path(__file__).resolve().relative_to(ROOT))
    records[relative] = _sha256(Path(__file__))
    return records


def full_run_plan(
    *,
    reference_root: Path,
    input_data: Mapping[str, Any],
) -> dict[str, Any]:
    references = _reference_map(reference_root, input_data)
    bindings = []
    for plan in request_plans():
        reference = references["bindings"][str(plan["seed"])]
        bindings.append(
            {
                **plan,
                "reference": reference,
            }
        )
    if len({item["seed"] for item in bindings}) != HARD_REQUEST_LIMIT:
        raise FinalWarmError("Warm run plan does not contain two unique seeds")
    return {
        "schema": f"{SCHEMA}.full-plan",
        "requests": bindings,
        "request_order": [
            {
                "request_index": item["request_index"],
                "seed": item["seed"],
                "timing_label": item["timing_label"],
                "reference_fasta": item["reference"]["fasta_relative"],
                "reference_npz": item["reference"]["specificity_relative"],
            }
            for item in bindings
        ],
    }


def preflight(
    *,
    repo: Path,
    checkpoint: Path,
    input_path: Path,
    python_bin: str,
    reference_root: Path,
    prior_attempt_001: Path,
    prior_attempt_002: Path,
    probe_entrypoint: bool = True,
) -> dict[str, Any]:
    legacy = timing.preflight(
        repo=repo,
        checkpoint=checkpoint,
        input_path=input_path,
        python_bin=python_bin,
        reference_root=reference_root,
        probe_entrypoint=probe_entrypoint,
    )
    input_data = legacy["input"]
    plan_before = full_run_plan(
        reference_root=reference_root,
        input_data=input_data,
    )
    prior_001_before = _prior_completed_request(
        prior_attempt_001,
        seed=267850363,
        reference=plan_before["requests"][1]["reference"],
    )
    prior_002_before = _prior_completed_request(
        prior_attempt_002,
        seed=267850364,
        reference=plan_before["requests"][0]["reference"],
    )
    execution_before = _execution_source_manifest()
    plan_after = full_run_plan(
        reference_root=reference_root,
        input_data=input_data,
    )
    prior_001_after = _prior_completed_request(
        prior_attempt_001,
        seed=267850363,
        reference=plan_after["requests"][1]["reference"],
    )
    prior_002_after = _prior_completed_request(
        prior_attempt_002,
        seed=267850364,
        reference=plan_after["requests"][0]["reference"],
    )
    execution_after = _execution_source_manifest()
    if plan_before != plan_after:
        raise FinalWarmError("Full run-plan reference binding changed")
    if prior_001_before != prior_001_after:
        raise FinalWarmError("Prior completed seed363 evidence changed")
    if prior_002_before != prior_002_after:
        raise FinalWarmError("Prior completed seed364 evidence changed")
    if execution_before != execution_after:
        raise FinalWarmError("Final warm execution source changed")
    report = dict(legacy)
    report.update(
        {
            "schema": PREFLIGHT_SCHEMA,
            "profile": PROFILE,
            "full_run_plan": plan_after,
            "prior_completed_requests": {
                "seed267850363": prior_001_after,
                "seed267850364": prior_002_after,
            },
            "execution_source": {
                "manifest": execution_after,
                "manifest_sha256": timing._manifest_digest(execution_after),
            },
            "request_budget": {
                "benchmark_total_actual": 4,
                "already_completed_prior": 2,
                "new_hard_total": HARD_REQUEST_LIMIT,
                "per_arm": {
                    "subprocess": 0,
                    "daemon": HARD_REQUEST_LIMIT,
                },
                "native_reference_requests": 0,
                "af3_requests": 0,
            },
            "request_plans": list(request_plans()),
        }
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=timing.DEFAULT_REPO)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=timing.DEFAULT_CHECKPOINT,
    )
    parser.add_argument("--input", type=Path, default=timing.DEFAULT_INPUT)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--prior-attempt-001", type=Path, required=True)
    parser.add_argument("--prior-attempt-002", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.out_dir = args.out_dir.expanduser().resolve()
    args.repo = args.repo.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.input = args.input.expanduser().resolve()
    args.python_bin = str(args.python_bin)
    args.reference_root = args.reference_root.expanduser().resolve()
    args.prior_attempt_001 = args.prior_attempt_001.expanduser().resolve()
    args.prior_attempt_002 = args.prior_attempt_002.expanduser().resolve()
    if args.out_dir.exists():
        if not args.out_dir.is_dir():
            parser.error(f"--out-dir is not a directory: {args.out_dir}")
        if any(args.out_dir.iterdir()) and not args.preflight_only:
            parser.error("--out-dir must be absent or empty")
    if not args.input.is_file():
        parser.error(f"Input structure is unavailable: {args.input}")
    if not Path(args.python_bin).is_file():
        parser.error(f"Python executable is unavailable: {args.python_bin}")
    for path, label in (
        (args.reference_root, "reference root"),
        (args.prior_attempt_001, "prior attempt 001"),
        (args.prior_attempt_002, "prior attempt 002"),
    ):
        if not path.is_dir():
            parser.error(f"{label} is unavailable: {path}")
    return args


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = preflight(
        repo=args.repo,
        checkpoint=args.checkpoint,
        input_path=args.input,
        python_bin=args.python_bin,
        reference_root=args.reference_root,
        prior_attempt_001=args.prior_attempt_001,
        prior_attempt_002=args.prior_attempt_002,
        probe_entrypoint=True,
    )
    timing._durable_json(args.out_dir / PREFLIGHT_NAME, report)
    return report


def _raw_result_record(
    *,
    result: Mapping[str, Any],
    root: Path,
    index: int,
) -> dict[str, Any]:
    bundle = result.get("probabilities")
    if not isinstance(bundle, Mapping):
        raise FinalWarmError(f"Request {index} returned no probability bundle")
    arrays_path = root / f"request_{index:02d}_probabilities.npz"
    timing._durable_npz(
        arrays_path,
        {str(key): np.asarray(value) for key, value in bundle.items()},
    )
    fasta_path = Path(str(result["fasta_path"]))
    return {
        "probabilities": _file_record(arrays_path),
        "job_dir": str(result.get("job_dir") or ""),
        "fasta_path": str(fasta_path),
        "fasta": _file_record(fasta_path),
        "sequence_sha256": hashlib.sha256(
            str(result.get("sequence") or "").encode("utf-8")
        ).hexdigest(),
    }


def _resident_summary(
    records: Sequence[Mapping[str, Any]],
    statuses: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(records) != HARD_REQUEST_LIMIT or len(statuses) != HARD_REQUEST_LIMIT:
        raise FinalWarmError("Warm owner evidence must contain two requests")
    pids = {int(item["pid"]) for item in records}
    tokens = {int(item["model_token"]) for item in records}
    devices = {str(item["device"]) for item in records}
    uuids = {str(item["gpu_uuid"]) for item in records}
    if (
        len(pids) != 1
        or len(tokens) != 1
        or len(devices) != 1
        or len(uuids) != 1
        or {int(item["model_load_count"]) for item in records} != {1}
        or [int(item["request_count"]) for item in records] != [1, 2]
        or [int(item["request_delta"]) for item in records] != [1, 1]
        or [int(item["request_count"]) for item in statuses] != [1, 2]
    ):
        raise FinalWarmError("Warm owner residency evidence is inconsistent")
    return {
        "pid": next(iter(pids)),
        "model_token": next(iter(tokens)),
        "device": next(iter(devices)),
        "gpu_uuid": next(iter(uuids)),
        "model_load_count": 1,
        "request_counts": [1, 2],
        "request_deltas": [1, 1],
        "same_owner": True,
        "same_model_token": True,
    }


def observed_duration(
    record: Mapping[str, Any],
) -> tuple[float | None, bool]:
    """Return a duration only when the owner record actually contains it."""
    if "duration_s" not in record:
        return None, False
    value = record["duration_s"]
    if value is None:
        return None, False
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FinalWarmError("Owner duration_s must be numeric or absent")
    duration = float(value)
    if not math.isfinite(duration) or duration < 0:
        raise FinalWarmError("Owner duration_s is not a finite nonnegative value")
    return duration, True


def format_observed_duration(
    duration_s: float | None,
    observed: bool,
) -> str:
    if not observed or duration_s is None:
        return "not observed"
    return f"{duration_s:.6f} s"


def _run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from astrolabe.model_execution import capture_random_state

    if not torch.cuda.is_available():
        raise FinalWarmError("Final warm canary requires a visible CUDA device")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    preflight_report = preflight(
        repo=args.repo,
        checkpoint=args.checkpoint,
        input_path=args.input,
        python_bin=args.python_bin,
        reference_root=args.reference_root,
        prior_attempt_001=args.prior_attempt_001,
        prior_attempt_002=args.prior_attempt_002,
        probe_entrypoint=True,
    )
    timing._durable_json(args.out_dir / PREFLIGHT_NAME, preflight_report)
    source_before = preflight_report["repo"]["source_manifest"]
    execution_before = preflight_report["execution_source"]["manifest"]
    plan_before = preflight_report["full_run_plan"]
    prior_before = preflight_report["prior_completed_requests"]
    input_sha_before = preflight_report["input"]["sha256"]
    checkpoint_sha_before = preflight_report["checkpoint"]["sha256"]
    budget = timing.RequestBudget()
    accepted = {"subprocess": 0, "daemon": 0}
    torch.cuda.current_device()
    torch.cuda.synchronize()

    request_root = args.out_dir / "requests"
    capture_path = args.out_dir / "resident-requests.jsonl"
    Runner = timing._daemon_runner_class()
    runner = Runner(
        _runner_config(
            repo=args.repo,
            checkpoint=args.checkpoint,
            python_bin=args.python_bin,
            execution_mode="daemon",
        ),
        output_root=args.out_dir / "daemon",
        capture_path=capture_path,
        request_root=request_root,
    )
    raw_results: list[dict[str, Any]] = []
    inference_results: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []
    try:
        for index, plan in enumerate(request_plans()):
            accepted["daemon"] = budget.claim("daemon")
            runner.extra_args["seed"] = int(plan["seed"])
            caller_before = capture_random_state()
            caller_before_sha = timing._state_digest(caller_before)
            gpu_before = timing.base._gpu_facts(torch)
            timing._assert_gpu_identity(gpu_before)
            before_path = request_root / f"request_{index:02d}_caller_before.json"
            clock_start = timing._clock()
            perf_start = time.perf_counter()
            timing._durable_json(
                before_path,
                {
                    "schema": f"{SCHEMA}.caller-before",
                    "request_index": index,
                    "request": plan,
                    "gpu": gpu_before,
                    "caller_rng_before_sha256": caller_before_sha,
                    "caller_rng_before": caller_before,
                    "clock": clock_start,
                },
            )
            result = runner.design_sequences(
                structure_path=args.input,
                job_name=f"daemon_{index:02d}_seed_{plan['seed']}",
                chains_to_design="A",
                fixed_residues=list(timing.base.FIXED_RESIDUES),
                parse_these_chains_only="ABC",
            )
            torch.cuda.synchronize()
            elapsed_s = time.perf_counter() - perf_start
            clock_after = timing._clock()
            caller_after = capture_random_state()
            caller_after_sha = timing._state_digest(caller_after)
            owner_status = runner.status()
            statuses.append(owner_status)
            raw_output = _raw_result_record(
                result=result,
                root=request_root,
                index=index,
            )
            raw_after_path = request_root / f"request_{index:02d}_raw_after.json"
            raw_after = {
                "schema": f"{SCHEMA}.caller-after-raw",
                "request_index": index,
                "request": plan,
                "duration_s": elapsed_s,
                "clock_start": clock_start,
                "clock_after": clock_after,
                "caller_rng_before_sha256": caller_before_sha,
                "caller_rng_after_sha256": caller_after_sha,
                "caller_rng_before": caller_before,
                "caller_rng_after": caller_after,
                "gpu_before": gpu_before,
                "owner_status": owner_status,
                "daemon_owner_record": runner.records[index],
                "raw_output": raw_output,
                "comparison_deferred_until_all_paid_requests_complete": True,
            }
            timing._durable_json(raw_after_path, raw_after)
            raw_results.append(raw_after)
            inference_results.append(dict(result))
    finally:
        runner.shutdown()

    records = [
        json.loads(line)
        for line in capture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    resident = _resident_summary(records, statuses)
    if accepted != {"subprocess": 0, "daemon": HARD_REQUEST_LIMIT}:
        raise FinalWarmError(f"Request budget mismatch: {accepted}")
    if len(inference_results) != HARD_REQUEST_LIMIT:
        raise FinalWarmError("Both paid inference requests did not complete")

    comparisons = []
    request_records = []
    for index, (plan, result, raw_after) in enumerate(
        zip(request_plans(), inference_results, raw_results)
    ):
        captured = capture_runner_result(
            runner=runner,
            result=result,
            plan=plan,
            input_data=preflight_report["input"],
        )
        reference = plan_before["requests"][index]["reference"]
        live_arrays_path = request_root / f"request_{index:02d}_live_arrays.npz"
        arrays = {
            f"deterministic.{key}": value
            for key, value in _flatten_arrays(captured["deterministic"]).items()
        }
        timing._durable_npz(live_arrays_path, arrays)
        reference_fasta = Path(reference["fasta"])
        live_fasta = Path(captured["fasta_path"])
        from scripts.validation.model_gpu_throughput import compare_outputs

        probability_comparison = compare_outputs(
            timing._load_reference_bundle(Path(reference["specificity_npz"])),
            captured["deterministic"]["probability_bundle"],
        )
        fasta_equal = reference_fasta.read_bytes() == live_fasta.read_bytes()
        comparison_record = {
            "schema": f"{SCHEMA}.request-comparison",
            "request_index": index,
            "seed": plan["seed"],
            "reference_fasta": reference["fasta_relative"],
            "reference_npz": reference["specificity_relative"],
            "fasta_equal": fasta_equal,
            "probability_bundle_equal": bool(
                probability_comparison["equal"]
            ),
            "probability_comparison": probability_comparison,
            "selected_sequence": captured["deterministic"][
                "selected_fasta_entries"
            ][0]["sequence"],
            "fixed_mask": captured["deterministic"]["fixed_mask"],
        }
        comparison_path = request_root / f"request_{index:02d}_comparison.json"
        daemon_infer_s, daemon_infer_observed = observed_duration(
            raw_after["daemon_owner_record"]
        )
        timing._durable_json(comparison_path, comparison_record)
        comparisons.append(comparison_record)
        request_records.append(
            {
                **plan,
                "caller_before": _file_record(
                    request_root / f"request_{index:02d}_caller_before.json"
                ),
                "raw_after": _file_record(
                    request_root / f"request_{index:02d}_raw_after.json"
                ),
                "live_arrays": _file_record(live_arrays_path),
                "comparison": _file_record(comparison_path),
                "duration_parent_call_s": raw_after["duration_s"],
                "duration_daemon_infer_s": daemon_infer_s,
                "daemon_infer_observed": daemon_infer_observed,
                "daemon_infer_display": format_observed_duration(
                    daemon_infer_s,
                    daemon_infer_observed,
                ),
                "owner_status": raw_after["owner_status"],
                "daemon_owner_record": raw_after["daemon_owner_record"],
            }
        )
    if not all(
        item["fasta_equal"] and item["probability_bundle_equal"]
        for item in comparisons
    ):
        raise FinalWarmError("One or more warm canary references differ")
    caller_checks = [
        {
            "request_index": item["request_index"],
            "equal": (
                item["caller_rng_before_sha256"]
                == item["caller_rng_after_sha256"]
                and item["caller_rng_before"] == item["caller_rng_after"]
            ),
            "before_sha256": item["caller_rng_before_sha256"],
            "after_sha256": item["caller_rng_after_sha256"],
        }
        for item in raw_results
    ]
    if not all(item["equal"] for item in caller_checks):
        raise FinalWarmError("Warm canary changed caller RNG state")
    if timing.base._repo_source_manifest(args.repo) != source_before:
        raise FinalWarmError("Frozen NA-MPNN repository source changed")
    if _execution_source_manifest() != execution_before:
        raise FinalWarmError("Final warm execution source changed")
    plan_after = full_run_plan(
        reference_root=args.reference_root,
        input_data=preflight_report["input"],
    )
    if plan_after != plan_before:
        raise FinalWarmError("Full run-plan reference binding changed")
    prior_after = {
        "seed267850363": _prior_completed_request(
            args.prior_attempt_001,
            seed=267850363,
            reference=plan_after["requests"][1]["reference"],
        ),
        "seed267850364": _prior_completed_request(
            args.prior_attempt_002,
            seed=267850364,
            reference=plan_after["requests"][0]["reference"],
        ),
    }
    if prior_after != prior_before:
        raise FinalWarmError("Prior completed-request evidence changed")
    if _sha256(args.input) != input_sha_before:
        raise FinalWarmError("NA-MPNN input changed during warm canary")
    if _sha256(args.checkpoint) != checkpoint_sha_before:
        raise FinalWarmError("NA-MPNN checkpoint changed during warm canary")

    report = {
        "schema": SCHEMA,
        "profile": PROFILE,
        "verified": True,
        "created_unix_s": time.time(),
        "scope": (
            "One new resident owner performs seed267850364 cold and "
            "seed267850363 warm. Both inference requests and raw evidence "
            "complete before any reference comparison. The two earlier "
            "single-request runs remain separate external evidence."
        ),
        "request_budget": {
            "benchmark_total_actual": 4,
            "already_completed_prior": 2,
            "new_hard_total": HARD_REQUEST_LIMIT,
            "accepted": accepted,
            "native_reference_requests": 0,
            "af3_requests": 0,
        },
        "full_run_plan": plan_before,
        "prior_completed_requests": prior_before,
        "input": preflight_report["input"],
        "repo": preflight_report["repo"],
        "checkpoint": preflight_report["checkpoint"],
        "execution_source": preflight_report["execution_source"],
        "public_entrypoint": preflight_report["public_entrypoint"],
        "gpu": timing.base._gpu_facts(torch),
        "requests": request_records,
        "comparisons": comparisons,
        "resident_daemon": resident,
        "caller_rng": {
            "requests": caller_checks,
            "all_unchanged": True,
        },
        "performance": {
            "labels": [
                "fresh_owner_seed267850364_cold",
                "same_owner_seed267850363_warm",
            ],
            "parent_call_s": [
                float(item["duration_parent_call_s"])
                for item in request_records
            ],
            "daemon_infer_s": [
                item["duration_daemon_infer_s"]
                for item in request_records
            ],
            "daemon_infer_observed": [
                bool(item["daemon_infer_observed"])
                for item in request_records
            ],
            "daemon_infer_display": [
                item["daemon_infer_display"]
                for item in request_records
            ],
            "descriptive_only": True,
            "speedup_claim": None,
            "statistical_claim": None,
            "native_speed_comparison": None,
        },
        "frozen_after_run": {
            "repo_source_manifest_sha256": timing._manifest_digest(
                timing.base._repo_source_manifest(args.repo)
            ),
            "execution_source_manifest_sha256": timing._manifest_digest(
                _execution_source_manifest()
            ),
            "reference_manifest_sha256": _sha256(
                args.reference_root / timing.REFERENCE_MANIFEST_NAME
            ),
            "prior_request_evidence_unchanged": True,
            "input_sha256": _sha256(args.input),
            "checkpoint_sha256": _sha256(args.checkpoint),
        },
    }
    timing._durable_json(args.out_dir / REPORT_NAME, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.preflight_only:
        run_preflight(args)
        print(args.out_dir / PREFLIGHT_NAME)
        return 0
    report = _run(args)
    print(args.out_dir / REPORT_NAME)
    if not report["verified"]:
        raise FinalWarmError("Final warm report is not verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
