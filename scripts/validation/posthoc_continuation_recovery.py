#!/usr/bin/env python3
"""CPU-only recovery: reuse committed caller/baseline and prove zero-call cache."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import posthoc_continuation as common


SCHEMA = "astrolabe.posthoc-refold.continuation-recovery/v1"


def _verify_recovery_bundle(
    *,
    base_root: Path,
    recovery_root: Path,
    patch_manifest: Path,
) -> dict[str, Any]:
    value = common._read_json(patch_manifest)
    if value.get("schema") != "astrolabe.posthoc-refold.recovery-bundle/v1":
        raise common.ContinuationError("recovery bundle manifest has another schema")
    if Path(value["base_capsule"]).resolve() != base_root.resolve():
        raise common.ContinuationError("recovery base capsule path differs")
    if Path(value["recovery_bundle"]).resolve() != recovery_root.resolve():
        raise common.ContinuationError("recovery bundle path differs")
    base_manifest = base_root / "source_manifest.json"
    if value["base_source_manifest_sha256"] != common._sha256(base_manifest):
        raise common.ContinuationError("recovery base source manifest differs")
    payload = common._read_json(base_manifest)
    files = payload.get("files")
    if not isinstance(files, Mapping):
        raise common.ContinuationError("recovery base manifest has no files")
    for relative, expected in files.items():
        observed = common._sha256(base_root / relative)
        if observed != expected:
            raise common.ContinuationError(
                f"recovery base file differs from frozen v4: {relative}"
            )
    added = value.get("added_files")
    if not isinstance(added, Mapping) or not added:
        raise common.ContinuationError("recovery added-file contract is absent")
    for relative, expected in added.items():
        if common._sha256(recovery_root / relative) != expected:
            raise common.ContinuationError(
                f"recovery validator file differs: {relative}"
            )
    if value.get("runtime_source_overlays"):
        raise common.ContinuationError(
            "CPU recovery must execute the original v4 source without overlays"
        )
    return dict(value)


def _existing_baseline(
    harness,
    *,
    out_root: Path,
) -> dict[str, Any]:
    capture_dir = out_root / "evidence" / "requests" / "native_baseline"
    requests = harness._request_evidence(capture_dir)
    expected = {
        "rpc_delivered": 1,
        "request_reservations": 1,
        "inner_run_inference_invocations": 1,
        "inner_run_inference_completed": 1,
        "entrypoint_completed": 1,
        "evaluator_invocations": 1,
        "completed_evaluator_invocations": 1,
        "failed_or_denied_model_requests": 0,
    }
    observed = {key: requests[key] for key in expected}
    if observed != expected:
        raise common.ContinuationError(
            f"existing baseline request receipt differs: {observed}"
        )
    result_path = out_root / "baseline" / "result.json"
    result = common._read_json(result_path)
    if result.get("status") != "completed":
        raise common.ContinuationError("existing baseline did not complete")
    structure_paths = result.get("structure_paths")
    if not isinstance(structure_paths, Mapping) or not structure_paths:
        raise common.ContinuationError("existing baseline has no structure path")
    missing = [
        path for path in structure_paths.values() if not Path(path).is_file()
    ]
    if missing:
        raise common.ContinuationError(
            f"existing baseline structures are absent: {missing}"
        )
    return {
        "result": result,
        "requests": requests,
        "capture_dir": str(capture_dir),
        "result_path": str(result_path),
    }


def _run_guarded_cache(
    harness,
    *,
    out_root: Path,
    comparison_root: Path,
    prepared: Mapping[str, Any],
    base_root: Path,
    timeout_s: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    capture_dir = comparison_root / "evidence" / "requests" / "cache_resume"
    if capture_dir.exists() and any(capture_dir.iterdir()):
        raise common.ContinuationError(
            f"recovery cache evidence exists; refusing replay: {capture_dir}"
        )
    harness._install_capture(capture_dir)
    environment = harness._capture_environment(
        repo_root=base_root,
        capture_dir=capture_dir,
        stage="cache_resume_cpu_recovery",
        max_requests=0,
    )
    process = harness._run_process(
        prepared["commands"]["public_caller"],
        environment=environment,
        stdout_path=comparison_root / "logs" / "cache.stdout.json",
        stderr_path=comparison_root / "logs" / "cache.stderr.log",
        timeout_s=timeout_s,
    )
    summary = harness._parse_json_stdout(
        comparison_root / "logs" / "cache.stdout.json"
    )
    requests = harness._request_evidence(capture_dir)
    expected_zero = {
        "rpc_delivered": 0,
        "request_reservations": 0,
        "inner_run_inference_invocations": 0,
        "inner_run_inference_completed": 0,
        "entrypoint_completed": 0,
        "evaluator_invocations": 0,
        "failed_or_denied_model_requests": 0,
    }
    observed_zero = {key: requests[key] for key in expected_zero}
    if observed_zero != expected_zero:
        raise common.ContinuationError(
            f"recovery cache request layers differ: {observed_zero}"
        )
    if (
        int(summary.get("ran_af3", -1)) != 0
        or int(summary.get("skipped", -1)) != 1
        or int(summary.get("receipt_cache_hits_af3", -1)) != 1
        or int(summary.get("failed", -1)) != 0
    ):
        raise common.ContinuationError(
            f"recovery cache summary has unexpected counts: {summary}"
        )
    return {
        "process": process,
        "summary": summary,
        "requests": requests,
        "capture_dir": str(capture_dir),
    }, requests


def run_recovery(
    *,
    out_root: Path,
    comparison_root: Path,
    base_root: Path,
    recovery_root: Path,
    patch_manifest: Path,
    timeout_s: int = 3600,
    verify_only: bool = False,
) -> dict[str, Any]:
    out_root = Path(out_root).expanduser().resolve()
    comparison_root = Path(comparison_root).expanduser().resolve()
    base_root = Path(base_root).expanduser().resolve()
    recovery_root = Path(recovery_root).expanduser().resolve()
    patch_record = _verify_recovery_bundle(
        base_root=base_root,
        recovery_root=recovery_root,
        patch_manifest=patch_manifest,
    )
    harness = common._load_harness(base_root)
    frozen = harness.verify_frozen_inputs(out_root)
    prepared = frozen["prepared"]
    item = frozen["manifest_item"]
    if Path(prepared["repo_root"]).resolve() != base_root:
        raise common.ContinuationError(
            "prepared repository root differs from original v4 capsule"
        )
    original_failure_path = out_root / "failure.json"
    original_failure = common._read_json(original_failure_path)
    if "child stdout is not one JSON document" not in str(
        original_failure.get("error", "")
    ):
        raise common.ContinuationError(
            "original failure is not the known caller stdout failure"
        )
    original_failure_sha256 = common._sha256(original_failure_path)
    caller = common._committed_caller(harness, out_root=out_root, item=item)
    baseline = _existing_baseline(harness, out_root=out_root)

    state_path = comparison_root / "recovery_state.json"
    if state_path.exists():
        raise common.ContinuationError(
            f"recovery state exists; refusing duplicate execution: {state_path}"
        )
    if verify_only:
        result = {
            "schema": "astrolabe.posthoc-refold.continuation-recovery-preflight/v1",
            "status": "PASS",
            "caller_reused_not_reexecuted": True,
            "baseline_reused_not_reexecuted": True,
            "original_failure_sha256": original_failure_sha256,
            "caller_stdout": caller["stdout_parse"],
            "caller_request_counts": {
                key: caller["requests"][key]
                for key in (
                    "rpc_delivered",
                    "request_reservations",
                    "inner_run_inference_invocations",
                    "inner_run_inference_completed",
                    "entrypoint_completed",
                )
            },
            "baseline_request_counts": {
                key: baseline["requests"][key]
                for key in (
                    "rpc_delivered",
                    "request_reservations",
                    "inner_run_inference_invocations",
                    "inner_run_inference_completed",
                    "entrypoint_completed",
                    "evaluator_invocations",
                )
            },
            "base_capsule": {
                "path": str(base_root),
                "source_manifest_sha256": common._sha256(
                    base_root / "source_manifest.json"
                ),
                "capsule_metadata_sha256": common._sha256(
                    base_root / "capsule_metadata.json"
                ),
            },
            "recovery_bundle": {
                "path": str(recovery_root),
                "patch_manifest": str(Path(patch_manifest).resolve()),
                "patch_manifest_sha256": common._sha256(Path(patch_manifest)),
                "patch": patch_record,
            },
            "cache_plan": {
                "source":"original frozen v4",
                "model_runner_max_requests": 0,
                "container_nv": False,
                "expected_model_layers": 0,
            },
        }
        common._atomic_json(
            comparison_root / "recovery_preflight.json", result
        )
        return result

    common._atomic_json(
        state_path,
        {
            "schema": SCHEMA,
            "status": "running",
            "caller_reused": True,
            "baseline_reused": True,
            "started_unix_s": time.time(),
        },
    )
    try:
        caller_before_cache = harness._tree_snapshot(
            out_root / "caller" / "runs"
        )
        common._atomic_json(
            comparison_root / "evidence" / "caller_before_cache.json",
            caller_before_cache,
        )
        cache, cache_requests = _run_guarded_cache(
            harness,
            out_root=out_root,
            comparison_root=comparison_root,
            prepared=prepared,
            base_root=base_root,
            timeout_s=timeout_s,
        )
        caller_after_cache = harness._tree_snapshot(
            out_root / "caller" / "runs"
        )
        common._atomic_json(
            comparison_root / "evidence" / "caller_after_cache.json",
            caller_after_cache,
        )
        scientific = harness._scientific_check(
            out_root=comparison_root,
            prepared=prepared,
            item=item,
            caller=caller,
            baseline=baseline,
            caller_requests=caller["requests"],
            baseline_requests=baseline["requests"],
            cache_requests=cache_requests,
            caller_before_cache=caller_before_cache,
            caller_after_cache=caller_after_cache,
        )
        if scientific["status"] != "PASS":
            raise common.ContinuationError(
                "read-only scientific comparison failed"
            )
        summary = {
            "schema": SCHEMA,
            "status": "PASS",
            "caller_reused_not_reexecuted": True,
            "baseline_reused_not_reexecuted": True,
            "cache_new_physical_model_requests": 0,
            "original_failure_sha256": original_failure_sha256,
            "caller_stdout": caller["stdout_parse"],
            "request_layers": {
                "caller_reused": caller["requests"],
                "baseline_reused": baseline["requests"],
                "cache_resume": cache_requests,
            },
            "base_capsule": {
                "path": str(base_root),
                "source_manifest_sha256": common._sha256(
                    base_root / "source_manifest.json"
                ),
                "capsule_metadata_sha256": common._sha256(
                    base_root / "capsule_metadata.json"
                ),
            },
            "recovery_bundle": {
                "path": str(recovery_root),
                "patch_manifest": str(Path(patch_manifest).resolve()),
                "patch_manifest_sha256": common._sha256(Path(patch_manifest)),
                "patch": patch_record,
            },
            "scientific": scientific,
            "cache_summary": cache["summary"],
            "cache_logs": {
                "stdout": str(comparison_root / "logs" / "cache.stdout.json"),
                "stderr": str(comparison_root / "logs" / "cache.stderr.log"),
            },
        }
        common._atomic_json(
            comparison_root / "recovery_summary.json", summary
        )
        common._atomic_json(
            state_path,
            {
                "schema": SCHEMA,
                "status": "completed",
                "completed_unix_s": time.time(),
                "summary": str(comparison_root / "recovery_summary.json"),
            },
        )
        return summary
    except BaseException as error:
        failure = {
            "schema": SCHEMA,
            "status": "FAIL",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "automatic_retry": False,
        }
        common._atomic_json(
            comparison_root / "recovery_failure.json", failure
        )
        common._atomic_json(
            state_path,
            {
                "schema": SCHEMA,
                "status": "failed",
                "failed_unix_s": time.time(),
                "failure": str(comparison_root / "recovery_failure.json"),
            },
        )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--comparison-root", required=True)
    parser.add_argument("--base-root", required=True)
    parser.add_argument("--recovery-root", required=True)
    parser.add_argument("--patch-manifest", required=True)
    parser.add_argument("--timeout-s", type=int, default=3600)
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_recovery(
            out_root=Path(args.out_root),
            comparison_root=Path(args.comparison_root),
            base_root=Path(args.base_root),
            recovery_root=Path(args.recovery_root),
            patch_manifest=Path(args.patch_manifest),
            timeout_s=int(args.timeout_s),
            verify_only=bool(args.verify_only),
        )
    except BaseException as error:
        print(f"posthoc continuation recovery failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": result["status"],
                "kind": (
                    "preflight"
                    if args.verify_only
                    else "cache_and_external_comparison"
                ),
                "result": (
                    str(
                        Path(args.comparison_root).expanduser().resolve()
                        / "recovery_preflight.json"
                    )
                    if args.verify_only
                    else str(
                        Path(args.comparison_root).expanduser().resolve()
                        / "recovery_summary.json"
                    )
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
