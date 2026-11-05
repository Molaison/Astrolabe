#!/usr/bin/env python3
"""Final read-only comparison for reused posthoc caller/baseline evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import posthoc_continuation as common
import posthoc_continuation_recovery as recovery


SCHEMA = "astrolabe.posthoc-refold.continuation-final-comparison/v1"


def _entrypoint_request_evidence(
    harness,
    capture_dir: Path,
) -> dict[str, Any]:
    evidence = harness._request_evidence(capture_dir)
    events = harness._read_jsonl(capture_dir / "events.jsonl")
    completions = [
        event
        for event in events
        if event.get("event") == "model_entrypoint_completed"
    ]
    if len(completions) != 1:
        raise common.ContinuationError(
            f"expected one completed entrypoint event in {capture_dir}"
        )
    evidence = dict(evidence)
    evidence["completed_entrypoint_events"] = completions
    evidence["requests"] = completions
    return evidence


def _existing_cache(
    harness,
    *,
    recovery_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    stdout_path = recovery_root / "logs" / "cache.stdout.json"
    summary = harness._parse_json_stdout(stdout_path)
    if (
        int(summary.get("ran_af3", -1)) != 0
        or int(summary.get("skipped", -1)) != 1
        or int(summary.get("receipt_cache_hits_af3", -1)) != 1
        or int(summary.get("failed", -1)) != 0
    ):
        raise common.ContinuationError(
            f"completed recovery cache summary differs: {summary}"
        )
    capture_dir = recovery_root / "evidence" / "requests" / "cache_resume"
    events_path = capture_dir / "events.jsonl"
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
            f"completed recovery cache request layers differ: {observed_zero}"
        )
    return {
        "summary": summary,
        "requests": requests,
        "capture_dir": str(capture_dir),
        "events_path": str(events_path),
        "events_file_absent": not events_path.exists(),
    }, requests


def run_final_compare(
    *,
    out_root: Path,
    recovered_cache_root: Path,
    final_root: Path,
    base_root: Path,
) -> dict[str, Any]:
    out_root = Path(out_root).expanduser().resolve()
    recovered_cache_root = Path(recovered_cache_root).expanduser().resolve()
    final_root = Path(final_root).expanduser().resolve()
    base_root = Path(base_root).expanduser().resolve()
    harness = common._load_harness(base_root)
    frozen = harness.verify_frozen_inputs(out_root)
    prepared = frozen["prepared"]
    item = frozen["manifest_item"]
    caller = common._committed_caller(harness, out_root=out_root, item=item)
    caller_requests = _entrypoint_request_evidence(
        harness,
        Path(caller["capture_dir"]),
    )
    caller["requests"] = caller_requests
    baseline = recovery._existing_baseline(harness, out_root=out_root)
    baseline_requests = _entrypoint_request_evidence(
        harness,
        Path(baseline["capture_dir"]),
    )
    baseline["requests"] = baseline_requests
    cache, cache_requests = _existing_cache(
        harness,
        recovery_root=recovered_cache_root,
    )
    caller_before_cache = common._read_json(
        recovered_cache_root / "evidence" / "caller_before_cache.json"
    )
    caller_after_cache = common._read_json(
        recovered_cache_root / "evidence" / "caller_after_cache.json"
    )
    if (
        harness._tree_changes(caller_before_cache, caller_after_cache)
        != []
    ):
        raise common.ContinuationError(
            "committed caller tree changed during the completed cache run"
        )

    cache_logs = final_root / "logs"
    common._atomic_json(cache_logs / "cache.stdout.json", cache["summary"])
    inputs_link = final_root / "inputs"
    if not inputs_link.exists():
        inputs_link.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(
            out_root / "inputs",
            inputs_link,
            target_is_directory=True,
        )
    scientific = harness._scientific_check(
        out_root=final_root,
        prepared=prepared,
        item=item,
        caller=caller,
        baseline=baseline,
        caller_requests=caller_requests,
        baseline_requests=baseline_requests,
        cache_requests=cache_requests,
        caller_before_cache=caller_before_cache,
        caller_after_cache=caller_after_cache,
    )
    if scientific["status"] != "PASS":
        failed_checks = [
            key for key, value in scientific["checks"].items() if not value
        ]
        common._atomic_json(
            final_root / "final_comparison_failure.json",
            {
                "schema": SCHEMA,
                "status": "FAIL",
                "new_model_requests": 0,
                "failed_checks": failed_checks,
                "scientific": scientific,
            },
        )
        raise common.ContinuationError(
            "final read-only scientific comparison failed: "
            + ",".join(failed_checks)
        )
    summary = {
        "schema": SCHEMA,
        "status": "PASS",
        "new_model_requests": 0,
        "caller_reused_not_reexecuted": True,
        "baseline_reused_not_reexecuted": True,
        "cache_reused_not_reexecuted": True,
        "recovered_cache_root": str(recovered_cache_root),
        "recovered_cache": cache,
        "request_layers": {
            "caller": caller_requests,
            "native_baseline": baseline_requests,
            "cache_resume": cache_requests,
        },
        "caller_stdout": caller["stdout_parse"],
        "scientific": scientific,
    }
    common._atomic_json(final_root / "final_comparison.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--recovered-cache-root", required=True)
    parser.add_argument("--final-root", required=True)
    parser.add_argument("--base-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = run_final_compare(
            out_root=Path(args.out_root),
            recovered_cache_root=Path(args.recovered_cache_root),
            final_root=Path(args.final_root),
            base_root=Path(args.base_root),
        )
    except BaseException as error:
        print(f"posthoc final comparison failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": summary["status"],
                "new_model_requests": 0,
                "result": str(
                    Path(args.final_root).expanduser().resolve()
                    / "final_comparison.json"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
