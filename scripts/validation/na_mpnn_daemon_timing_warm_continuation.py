#!/usr/bin/env python3
"""Run seed364 cold then seed363 warm in one new resident NA-MPNN owner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "astrolabe.na-mpnn-daemon-timing-warm/v1"
PREFLIGHT_SCHEMA = f"{SCHEMA}.preflight"
PROFILE = "na_mpnn_daemon_timing_v1_warm_002"
PREFLIGHT_NAME = "preflight.json"
REPORT_NAME = "acceptance.json"
SEEDS = (267850364, 267850363)
HARD_REQUEST_LIMIT = 2

from scripts.validation import na_mpnn_daemon_timing_acceptance as timing  # noqa: E402
from scripts.validation.na_mpnn_gpu_acceptance import (  # noqa: E402
    AcceptanceError,
    _runner_config,
    capture_runner_result,
)
from scripts.validation.model_gpu_throughput import (  # noqa: E402
    _flatten_arrays,
    _sha256,
    compare_outputs,
)


class ContinuationError(AcceptanceError):
    """The warm continuation cannot be proven."""


def request_plans() -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "arm": "daemon",
            "request_index": index,
            "seed": seed,
            "timing_label": (
                "fresh_owner_cold"
                if index == 0
                else "same_owner_warm"
            ),
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
        for index, seed in enumerate(SEEDS)
    )


def reference_index(seed: int) -> int:
    """Map request seed to its old-native reference using reference order."""
    return timing.SEEDS.index(int(seed))


def _prior_request_facts(
    prior_root: Path,
    reference_root: Path,
) -> dict[str, Any]:
    from astrolabe.mutators.wrappers.na_mpnn_runner import NAMPNNRunner

    root = prior_root.expanduser().resolve()
    reference = reference_root.expanduser().resolve()
    required = (
        root / "preflight.json",
        root / "resident-requests.jsonl",
        root / "requests/request_00_caller_before.json",
        root / "requests/daemon/request_00_before.json",
        root / "requests/daemon/request_00_after.json",
        root / "requests/request_00_raw_after.json",
        root / "requests/request_00_comparison.json",
        root / "daemon/daemon_00_seed_267850363/seqs/step_000_ABC_model.fa",
        root
        / "daemon/daemon_00_seed_267850363"
        / "specificity/step_000_ABC_model.npz",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ContinuationError(f"Previously completed request evidence is missing: {missing}")
    if list((root / "requests").glob("request_01_*")) or list(
        (root / "requests/daemon").glob("request_01_*")
    ):
        raise ContinuationError("Prior failed attempt contains a second-request start marker")
    if (root / "acceptance.json").exists():
        raise ContinuationError("Prior failed attempt unexpectedly contains acceptance.json")

    ledger = [
        json.loads(line)
        for line in (root / "resident-requests.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    if len(ledger) != 1:
        raise ContinuationError("Prior owner ledger does not contain exactly one request")
    record = ledger[0]
    if (
        record.get("status") != "completed"
        or int(record.get("request_count", -1)) != 1
        or int(record.get("request_delta", -1)) != 1
        or int(record.get("model_load_count", -1)) != 1
        or not str(record.get("gpu_uuid") or "")
    ):
        raise ContinuationError("Prior completed request ledger is inconsistent")

    caller_before = json.loads(
        (root / "requests/request_00_caller_before.json").read_text(
            encoding="utf-8"
        )
    )
    daemon_before = json.loads(
        (root / "requests/daemon/request_00_before.json").read_text(
            encoding="utf-8"
        )
    )
    daemon_after = json.loads(
        (root / "requests/daemon/request_00_after.json").read_text(
            encoding="utf-8"
        )
    )
    if caller_before["gpu"]["uuid"] != record["gpu_uuid"]:
        raise ContinuationError("Prior caller/daemon/ledger GPU UUIDs disagree")
    if daemon_before["owner"]["gpu"]["uuid"] != record["gpu_uuid"]:
        raise ContinuationError("Prior daemon-start GPU UUID disagrees with ledger")
    if daemon_after["status"] != "completed":
        raise ContinuationError("Prior daemon-after record is not completed")

    reference_seed = reference / "subprocess_00_seed_267850363"
    reference_fasta = reference_seed / "seqs/step_000_ABC_model.fa"
    live_fasta = (
        root / "daemon/daemon_00_seed_267850363/seqs/step_000_ABC_model.fa"
    )
    fasta_equal = reference_fasta.read_bytes() == live_fasta.read_bytes()
    reference_bundle = timing._load_reference_bundle(
        reference_seed / "specificity/step_000_ABC_model.npz"
    )
    loader = NAMPNNRunner.__new__(NAMPNNRunner)
    live_bundle = loader._load_specificity_matrix(
        root / "daemon/daemon_00_seed_267850363",
        Path("step_000_ABC_model.cif"),
    )
    strict_comparison = compare_outputs(reference_bundle, live_bundle)
    if not fasta_equal or not strict_comparison["equal"]:
        raise ContinuationError(
            "Previously completed seed363 request is not exact against old native"
        )
    return {
        "path": str(root),
        "status": "external_validation_PASS",
        "request_count": 1,
        "request_delta": 1,
        "model_load_count": 1,
        "pid": int(record["pid"]),
        "model_token": int(record["model_token"]),
        "gpu_uuid": str(record["gpu_uuid"]),
        "fasta_equal": fasta_equal,
        "all_ten_fields_exact": bool(strict_comparison["equal"]),
        "field_sha256": strict_comparison["expected_sha256"],
        "evidence": {
            str(path.relative_to(root)): timing._file_record(path)
            for path in required
        },
    }


def _execution_source_manifest() -> dict[str, str]:
    records = dict(timing._execution_source_manifest())
    relative = str(Path(__file__).resolve().relative_to(ROOT))
    records[relative] = _sha256(Path(__file__))
    return records


def preflight(
    *,
    repo: Path,
    checkpoint: Path,
    input_path: Path,
    python_bin: str,
    reference_root: Path,
    prior_attempt_root: Path,
    probe_entrypoint: bool = True,
) -> dict[str, Any]:
    prior_before = _prior_request_facts(prior_attempt_root, reference_root)
    execution_before = _execution_source_manifest()
    legacy = timing.preflight(
        repo=repo,
        checkpoint=checkpoint,
        input_path=input_path,
        python_bin=python_bin,
        reference_root=reference_root,
        probe_entrypoint=probe_entrypoint,
    )
    execution_after = _execution_source_manifest()
    prior_after = _prior_request_facts(prior_attempt_root, reference_root)
    if execution_before != execution_after:
        raise ContinuationError("Warm execution source changed during preflight")
    if prior_before != prior_after:
        raise ContinuationError("Prior completed request evidence changed during preflight")
    report = dict(legacy)
    report.update(
        {
            "schema": PREFLIGHT_SCHEMA,
            "profile": PROFILE,
            "execution_source": {
                "manifest": execution_after,
                "manifest_sha256": timing._manifest_digest(
                    execution_after
                ),
            },
            "prior_completed_request": prior_after,
            "request_budget": {
                "benchmark_total_actual": 3,
                "already_completed_prior": 1,
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
    parser.add_argument("--prior-attempt-root", type=Path, required=True)
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
    args.prior_attempt_root = args.prior_attempt_root.expanduser().resolve()
    if args.out_dir.exists():
        if not args.out_dir.is_dir():
            parser.error(f"--out-dir is not a directory: {args.out_dir}")
        if any(args.out_dir.iterdir()) and not args.preflight_only:
            parser.error("--out-dir must be absent or empty")
    if not args.input.is_file():
        parser.error(f"Input structure is unavailable: {args.input}")
    if not Path(args.python_bin).is_file():
        parser.error(f"Python executable is unavailable: {args.python_bin}")
    if not args.reference_root.is_dir():
        parser.error(f"Old-native reference is unavailable: {args.reference_root}")
    if not args.prior_attempt_root.is_dir():
        parser.error(f"Prior attempt is unavailable: {args.prior_attempt_root}")
    return args


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = preflight(
        repo=args.repo,
        checkpoint=args.checkpoint,
        input_path=args.input,
        python_bin=args.python_bin,
        reference_root=args.reference_root,
        prior_attempt_root=args.prior_attempt_root,
        probe_entrypoint=True,
    )
    timing._durable_json(args.out_dir / PREFLIGHT_NAME, report)
    return report


def _single_resident_summary(
    records: Sequence[Mapping[str, Any]],
    statuses: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(records) != HARD_REQUEST_LIMIT or len(statuses) != HARD_REQUEST_LIMIT:
        raise ContinuationError("Warm owner must contain exactly two requests")
    pids = {int(item["pid"]) for item in records}
    tokens = {int(item["model_token"]) for item in records}
    devices = {str(item["device"]) for item in records}
    uuids = {str(item["gpu_uuid"]) for item in records}
    request_counts = [int(item["request_count"]) for item in records]
    request_deltas = [int(item["request_delta"]) for item in records]
    loads = {int(item["model_load_count"]) for item in records}
    status_records = [
        int(item["request_count"])
        for item in statuses
    ]
    if (
        len(pids) != 1
        or len(tokens) != 1
        or len(devices) != 1
        or len(uuids) != 1
        or loads != {1}
        or request_counts != [1, 2]
        or request_deltas != [1, 1]
        or status_records != [1, 2]
    ):
        raise ContinuationError("Warm owner residency evidence is inconsistent")
    return {
        "pid": next(iter(pids)),
        "model_token": next(iter(tokens)),
        "device": next(iter(devices)),
        "gpu_uuid": next(iter(uuids)),
        "model_load_count": 1,
        "request_counts": request_counts,
        "request_deltas": request_deltas,
        "same_owner": True,
        "same_model_token": True,
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from astrolabe.model_execution import capture_random_state

    if not torch.cuda.is_available():
        raise ContinuationError("Warm continuation requires a visible CUDA device")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    preflight_report = preflight(
        repo=args.repo,
        checkpoint=args.checkpoint,
        input_path=args.input,
        python_bin=args.python_bin,
        reference_root=args.reference_root,
        prior_attempt_root=args.prior_attempt_root,
        probe_entrypoint=True,
    )
    timing._durable_json(
        args.out_dir / PREFLIGHT_NAME,
        preflight_report,
    )
    source_before = preflight_report["repo"]["source_manifest"]
    execution_before = preflight_report["execution_source"]["manifest"]
    reference_before = preflight_report["reference"]
    prior_before = preflight_report["prior_completed_request"]
    input_sha_before = preflight_report["input"]["sha256"]
    checkpoint_sha_before = preflight_report["checkpoint"]["sha256"]
    budget = timing.RequestBudget()
    accepted = {"subprocess": 0, "daemon": 0}
    torch.cuda.current_device()
    torch.cuda.synchronize()

    capture_path = args.out_dir / "resident-requests.jsonl"
    request_root = args.out_dir / "requests"
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
    results: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []
    caller_checks: list[dict[str, Any]] = []
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
            captured = capture_runner_result(
                runner=runner,
                result=result,
                plan=plan,
                input_data=preflight_report["input"],
            )
            live_arrays_path = request_root / f"request_{index:02d}_live_arrays.npz"
            live_arrays = timing._raw_output_snapshot(
                live_arrays_path,
                captured,
            )
            raw_path = request_root / f"request_{index:02d}_raw_after.json"
            timing._durable_json(
                raw_path,
                {
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
                    "job_dir": captured["job_dir"],
                    "fasta_path": captured["fasta_path"],
                    "fasta": timing._file_record(Path(captured["fasta_path"])),
                    "raw_arrays": live_arrays,
                    "selected_raw_sha256": hashlib.sha256(
                        str(captured["selected_raw"]).encode("utf-8")
                    ).hexdigest(),
                },
            )
            comparison = timing._compare_reference(
                seed=int(plan["seed"]),
                index=reference_index(int(plan["seed"])),
                reference_root=args.reference_root,
                captured=captured,
            )
            comparison_path = (
                request_root / f"request_{index:02d}_comparison.json"
            )
            timing._durable_json(
                comparison_path,
                {
                    "schema": f"{SCHEMA}.request-comparison",
                    "request_index": index,
                    "seed": plan["seed"],
                    "reference_content_bound": True,
                    **comparison,
                },
            )
            caller_equal = (
                caller_before == caller_after
                and caller_before_sha == caller_after_sha
            )
            caller_checks.append(
                {
                    "request_index": index,
                    "equal": caller_equal,
                    "before_sha256": caller_before_sha,
                    "after_sha256": caller_after_sha,
                }
            )
            result_record = {
                **plan,
                "duration_parent_call_s": elapsed_s,
                "duration_daemon_infer_s": float(
                    runner.records[index].get("duration_s", 0.0)
                ),
                "caller_before": timing._file_record(before_path),
                "raw_after": timing._file_record(raw_path),
                "comparison": timing._file_record(comparison_path),
                "owner_status": owner_status,
                "daemon_owner_record": runner.records[index],
            }
            results.append(result_record)
            if not comparison["fasta_equal"]:
                raise ContinuationError(
                    f"Seed {plan['seed']} FASTA differs from old native reference"
                )
            if not comparison["probability_bundle_equal"]:
                raise ContinuationError(
                    f"Seed {plan['seed']} probabilities differ from old native reference"
                )
            if not caller_equal:
                raise ContinuationError(
                    f"Seed {plan['seed']} changed caller RNG state"
                )
    finally:
        runner.shutdown()

    records = [
        json.loads(line)
        for line in capture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    resident = _single_resident_summary(records, statuses)
    if accepted != {"subprocess": 0, "daemon": HARD_REQUEST_LIMIT}:
        raise ContinuationError(f"Request budget mismatch: {accepted}")
    if len(results) != HARD_REQUEST_LIMIT:
        raise ContinuationError("Warm continuation did not complete two requests")
    if timing.base._repo_source_manifest(args.repo) != source_before:
        raise ContinuationError("Frozen NA-MPNN repository source changed")
    if _execution_source_manifest() != execution_before:
        raise ContinuationError("Astrolabe warm execution source changed")
    if timing.reference_facts(args.reference_root) != reference_before:
        raise ContinuationError("Old-native reference changed during warm benchmark")
    if _prior_request_facts(
        args.prior_attempt_root,
        args.reference_root,
    ) != prior_before:
        raise ContinuationError("Previously completed request evidence changed")
    if _sha256(args.input) != input_sha_before:
        raise ContinuationError("NA-MPNN input changed during warm benchmark")
    if _sha256(args.checkpoint) != checkpoint_sha_before:
        raise ContinuationError("NA-MPNN checkpoint changed during warm benchmark")

    report = {
        "schema": SCHEMA,
        "profile": PROFILE,
        "verified": True,
        "created_unix_s": time.time(),
        "scope": (
            "A new resident owner runs seed267850364 cold and seed267850363 "
            "warm. Both requests are exactly compared with preserved old-native "
            "outputs. The earlier completed seed267850363 request remains a "
            "separate external validation and is not overwritten."
        ),
        "request_budget": {
            "benchmark_total_actual": 3,
            "already_completed_prior": 1,
            "new_hard_total": HARD_REQUEST_LIMIT,
            "accepted": accepted,
            "native_reference_requests": 0,
            "af3_requests": 0,
        },
        "prior_completed_request": prior_before,
        "reference": reference_before,
        "input": preflight_report["input"],
        "repo": preflight_report["repo"],
        "checkpoint": preflight_report["checkpoint"],
        "execution_source": preflight_report["execution_source"],
        "public_entrypoint": preflight_report["public_entrypoint"],
        "gpu": timing.base._gpu_facts(torch),
        "requests": results,
        "resident_daemon": resident,
        "caller_rng": {
            "requests": caller_checks,
            "all_unchanged": all(item["equal"] for item in caller_checks),
        },
        "performance": {
            "labels": [
                "fresh_owner_seed267850364_cold",
                "same_owner_seed267850363_warm",
            ],
            "parent_call_s": [
                float(item["duration_parent_call_s"])
                for item in results
            ],
            "daemon_infer_s": [
                float(item["duration_daemon_infer_s"])
                for item in results
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
        raise ContinuationError("Warm continuation report is not verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
