#!/usr/bin/env python3
"""Continue a committed posthoc caller without replaying its paid model call."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback
from typing import Any, Mapping, Sequence


SCHEMA = "astrolabe.posthoc-refold.continuation/v1"
FAILURE_SCHEMA = "astrolabe.posthoc-refold.continuation-failure/v1"
FEATURISING_FIRST = "Featurising data with seed 17."
FEATURISING_SECOND = re.compile(
    r"^Featurising data with seed 17 took [0-9]+\.[0-9]{2} seconds\.$"
)


class ContinuationError(RuntimeError):
    """Committed caller evidence cannot be continued safely."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _hash_value(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_harness(repo_root: Path):
    path = Path(repo_root) / "scripts" / "validation" / "posthoc_gpu_acceptance.py"
    name = "_posthoc_continuation_base_harness"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ContinuationError(f"cannot load base acceptance harness: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_committed_caller_stdout(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Strictly allow only the observed two AF3 progress lines before JSON."""

    path = Path(path)
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if len(lines) < 3:
        raise ContinuationError("caller stdout has no complete prefixed JSON")
    if lines[0] != FEATURISING_FIRST:
        raise ContinuationError(
            f"caller stdout line 1 is not the known AF3 progress line: {lines[0]!r}"
        )
    if FEATURISING_SECOND.fullmatch(lines[1]) is None:
        raise ContinuationError(
            f"caller stdout line 2 is not the known AF3 timing line: {lines[1]!r}"
        )
    if not lines[2].startswith("{"):
        raise ContinuationError(
            f"caller stdout JSON must begin on line 3: {lines[2]!r}"
        )
    payload_text = "\n".join(lines[2:])
    decoder = json.JSONDecoder()
    try:
        payload, end = decoder.raw_decode(payload_text)
    except json.JSONDecodeError as error:
        raise ContinuationError(
            f"caller stdout JSON is invalid: {error}"
        ) from error
    if payload_text[end:].strip():
        raise ContinuationError(
            "caller stdout has non-whitespace residue after JSON"
        )
    if not isinstance(payload, dict):
        raise ContinuationError("caller stdout JSON is not an object")
    return payload, {
        "path": str(path),
        "raw_sha256": _sha256(path),
        "raw_size": path.stat().st_size,
        "allowed_prefix_line_count": 2,
        "allowed_prefix_lines": lines[:2],
        "json_start_line": 3,
        "json_sha256": _hash_value(payload),
        "json_trailing_non_whitespace": False,
    }


def _committed_caller(
    harness,
    *,
    out_root: Path,
    item: Mapping[str, Any],
) -> dict[str, Any]:
    stdout_path = out_root / "logs" / "caller.stdout.json"
    summary, parse_record = parse_committed_caller_stdout(stdout_path)
    if (
        int(summary.get("ran_af3", -1)) != 1
        or int(summary.get("processed", -1)) != 1
        or int(summary.get("failed", -1)) != 0
    ):
        raise ContinuationError("committed caller summary has unexpected counts")

    capture_dir = out_root / "evidence" / "requests" / "caller"
    requests = harness._request_evidence(capture_dir)
    expected = {
        "rpc_delivered": 1,
        "request_reservations": 1,
        "inner_run_inference_invocations": 1,
        "inner_run_inference_completed": 1,
        "entrypoint_completed": 1,
        "failed_or_denied_model_requests": 0,
    }
    observed = {key: requests[key] for key in expected}
    if observed != expected:
        raise ContinuationError(
            f"committed caller request receipt differs: {observed}"
        )

    run_dir = harness._expected_run_dir(out_root / "caller" / "runs", item)
    if not harness._model_done_marker(run_dir, item).is_file():
        raise ContinuationError("committed caller completion marker is absent")
    if not harness._model_structure(run_dir, item).is_file():
        raise ContinuationError("committed caller CIF is absent")
    refold_receipt = harness._refold_receipt_evidence(run_dir, item)
    return {
        "summary": summary,
        "stdout_parse": parse_record,
        "requests": requests,
        "capture_dir": str(capture_dir),
        "run_dir": str(run_dir),
        "refold_receipt": refold_receipt,
    }


def _patched_cache_command(
    base_command: Sequence[str],
) -> list[str]:
    command = list(base_command)
    try:
        index = command.index("-m")
    except ValueError as error:
        raise ContinuationError("public command has no python -m module form") from error
    if command[index + 1 : index + 2] != ["astrolabe.cli"]:
        raise ContinuationError("public command does not invoke astrolabe.cli")
    return [command[0], "-m", "astrolabe.cli", *command[index + 2 :]]


def _invoke_patched_cache(
    harness,
    *,
    out_root: Path,
    prepared: Mapping[str, Any],
    continuation_root: Path,
    timeout_s: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    capture_dir = out_root / "evidence" / "requests" / "cache_resume"
    if capture_dir.exists() and any(capture_dir.iterdir()):
        raise ContinuationError(
            f"cache request evidence already exists; refusing replay: {capture_dir}"
        )
    harness._install_capture(capture_dir)
    environment = harness._capture_environment(
        repo_root=continuation_root,
        capture_dir=capture_dir,
        stage="cache_resume",
        max_requests=0,
    )
    command = _patched_cache_command(prepared["commands"]["public_caller"])
    process = harness._run_process(
        command,
        environment=environment,
        stdout_path=out_root / "logs" / "cache.stdout.json",
        stderr_path=out_root / "logs" / "cache.stderr.log",
        timeout_s=timeout_s,
    )
    summary = harness._parse_json_stdout(out_root / "logs" / "cache.stdout.json")
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
        raise ContinuationError(
            f"cache/resume request layers differ: {observed_zero}"
        )
    if (
        int(summary.get("ran_af3", -1)) != 0
        or int(summary.get("skipped", -1)) != 1
        or int(summary.get("receipt_cache_hits_af3", -1)) != 1
        or int(summary.get("failed", -1)) != 0
    ):
        raise ContinuationError(
            f"cache/resume summary has unexpected counts: {summary}"
        )
    return {
        "process": process,
        "summary": summary,
        "requests": requests,
        "capture_dir": str(capture_dir),
    }, requests


def _verify_patch_manifest(
    *,
    base_root: Path,
    continuation_root: Path,
    path: Path,
) -> dict[str, Any]:
    value = _read_json(path)
    if value.get("schema") != "astrolabe.posthoc-refold.continuation-bundle/v1":
        raise ContinuationError("continuation patch manifest has another schema")
    if Path(value["base_capsule"]).resolve() != base_root.resolve():
        raise ContinuationError("continuation base capsule path differs")
    if Path(value["continuation_root"]).resolve() != continuation_root.resolve():
        raise ContinuationError("continuation root path differs")
    base_manifest = base_root / "source_manifest.json"
    if value["base_source_manifest_sha256"] != _sha256(base_manifest):
        raise ContinuationError("base source manifest hash differs")
    manifest_payload = _read_json(base_manifest)
    base_files = manifest_payload.get("files")
    if not isinstance(base_files, Mapping):
        raise ContinuationError("base source manifest has no file hashes")
    changed = value.get("changed_files")
    if not isinstance(changed, Mapping) or not changed:
        raise ContinuationError("continuation has no explicit changed-file contract")
    added = value.get("added_files", {})
    if not isinstance(added, Mapping):
        raise ContinuationError("continuation added-file contract is invalid")
    if set(changed) & set(added):
        raise ContinuationError("continuation file is both changed and added")
    if not set(changed).issubset(base_files):
        raise ContinuationError("changed-file contract includes a non-base file")
    if set(added) & set(base_files):
        raise ContinuationError("added-file contract includes an existing base file")
    for relative, hashes in changed.items():
        base_path = base_root / relative
        patched_path = continuation_root / relative
        if _sha256(base_path) != hashes["base_sha256"]:
            raise ContinuationError(f"base patch source differs: {relative}")
        if base_files[relative] != hashes["base_sha256"]:
            raise ContinuationError(f"base patch source differs from manifest: {relative}")
        if _sha256(patched_path) != hashes["patched_sha256"]:
            raise ContinuationError(f"patched continuation source differs: {relative}")
        if hashes["base_sha256"] == hashes["patched_sha256"]:
            raise ContinuationError(f"changed-file contract has no change: {relative}")
    for relative, expected in added.items():
        if _sha256(continuation_root / relative) != expected:
            raise ContinuationError(f"added continuation source differs: {relative}")
    for relative, expected in base_files.items():
        if relative in changed:
            continue
        if _sha256(base_root / relative) != expected:
            raise ContinuationError(f"base source differs from manifest: {relative}")
        if _sha256(continuation_root / relative) != expected:
            raise ContinuationError(
                f"continuation changed an unapproved base file: {relative}"
            )
    return dict(value)


def run_continuation(
    *,
    out_root: Path,
    base_root: Path,
    continuation_root: Path,
    patch_manifest: Path,
    timeout_s: int = 3600,
    verify_only: bool = False,
) -> dict[str, Any]:
    out_root = Path(out_root).expanduser().resolve()
    base_root = Path(base_root).expanduser().resolve()
    continuation_root = Path(continuation_root).expanduser().resolve()
    state_path = out_root / "continuation_state.json"
    if state_path.exists():
        raise ContinuationError(
            f"continuation state already exists; refusing duplicate execution: {state_path}"
        )
    patch_record = _verify_patch_manifest(
        base_root=base_root,
        continuation_root=continuation_root,
        path=Path(patch_manifest),
    )
    harness = _load_harness(base_root)
    frozen = harness.verify_frozen_inputs(out_root)
    prepared = frozen["prepared"]
    item = frozen["manifest_item"]
    if Path(prepared["repo_root"]).resolve() != base_root:
        raise ContinuationError("prepared repository root differs from base capsule")
    original_failure_path = out_root / "failure.json"
    original_failure = _read_json(original_failure_path)
    if original_failure.get("error_type") != "AcceptanceError" or (
        "child stdout is not one JSON document"
        not in str(original_failure.get("error", ""))
    ):
        raise ContinuationError("original failure is not the expected stdout-only failure")
    original_failure_sha256 = _sha256(original_failure_path)

    caller = _committed_caller(harness, out_root=out_root, item=item)
    if verify_only:
        result = {
            "schema": "astrolabe.posthoc-refold.continuation-preflight/v1",
            "status": "PASS",
            "caller_reused_not_reexecuted": True,
            "original_failure_path": str(original_failure_path),
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
            "base_capsule": {
                "path": str(base_root),
                "source_manifest_sha256": _sha256(base_root / "source_manifest.json"),
                "capsule_metadata_sha256": _sha256(
                    base_root / "capsule_metadata.json"
                ),
            },
            "continuation_bundle": {
                "path": str(continuation_root),
                "patch_manifest": str(Path(patch_manifest).resolve()),
                "patch_manifest_sha256": _sha256(Path(patch_manifest)),
                "cli_sha256": _sha256(continuation_root / "astrolabe/cli.py"),
                "base_cli_sha256": _sha256(base_root / "astrolabe/cli.py"),
                "patch": patch_record,
            },
            "planned_new_requests": {
                "native_baseline": 1,
                "cache_resume": 0,
                "maximum_new_physical_inner_requests": 1,
            },
        }
        _atomic_json(out_root / "continuation_preflight.json", result)
        return result

    _atomic_json(
        state_path,
        {
            "schema": SCHEMA,
            "status": "running",
            "caller_reused": True,
            "started_unix_s": time.time(),
        },
    )
    try:
        if (out_root / "baseline").exists() and any(
            (out_root / "baseline").iterdir()
        ):
            raise ContinuationError("baseline output exists; refusing replay")
        baseline, _ = harness._invoke_baseline(
            out_root=out_root,
            prepared=prepared,
            timeout_s=timeout_s,
        )
        caller_before_cache = harness._tree_snapshot(
            out_root / "caller" / "runs"
        )
        harness._atomic_json(
            out_root / "evidence" / "continuation_cache_before.json",
            caller_before_cache,
        )
        cache, cache_requests = _invoke_patched_cache(
            harness,
            out_root=out_root,
            prepared=prepared,
            continuation_root=continuation_root,
            timeout_s=timeout_s,
        )
        caller_after_cache = harness._tree_snapshot(
            out_root / "caller" / "runs"
        )
        harness._atomic_json(
            out_root / "evidence" / "continuation_cache_after.json",
            caller_after_cache,
        )
        scientific = harness._scientific_check(
            out_root=out_root,
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
            raise ContinuationError(
                "scientific comparison failed; inspect continuation summary"
            )
        summary = {
            "schema": SCHEMA,
            "status": "PASS",
            "caller_reused_not_reexecuted": True,
            "original_failure_path": str(original_failure_path),
            "original_failure_sha256": original_failure_sha256,
            "caller_stdout": caller["stdout_parse"],
            "request_layers": {
                "caller_reused": caller["requests"],
                "native_baseline": baseline["requests"],
                "cache_resume": cache_requests,
                "new_physical_inner_requests": (
                    baseline["requests"]["inner_run_inference_invocations"]
                    + cache_requests["inner_run_inference_invocations"]
                ),
                "maximum_new_physical_inner_requests": 1,
            },
            "base_capsule": {
                "path": str(base_root),
                "source_manifest_sha256": _sha256(base_root / "source_manifest.json"),
                "capsule_metadata_sha256": _sha256(base_root / "capsule_metadata.json"),
            },
            "continuation_bundle": {
                "path": str(continuation_root),
                "patch_manifest": str(Path(patch_manifest).resolve()),
                "patch_manifest_sha256": _sha256(Path(patch_manifest)),
                "cli_sha256": _sha256(continuation_root / "astrolabe/cli.py"),
                "base_cli_sha256": _sha256(base_root / "astrolabe/cli.py"),
                "patch": patch_record,
            },
            "scientific": scientific,
            "baseline": baseline,
            "cache_summary": cache["summary"],
        }
        harness._atomic_json(out_root / "continuation_summary.json", summary)
        _atomic_json(
            state_path,
            {
                "schema": SCHEMA,
                "status": "completed",
                "completed_unix_s": time.time(),
                "summary": str(out_root / "continuation_summary.json"),
            },
        )
        return summary
    except BaseException as error:
        failure = {
            "schema": FAILURE_SCHEMA,
            "status": "FAIL",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "original_failure_sha256": original_failure_sha256,
            "automatic_retry": False,
        }
        _atomic_json(out_root / "continuation_failure.json", failure)
        _atomic_json(
            state_path,
            {
                "schema": SCHEMA,
                "status": "failed",
                "failed_unix_s": time.time(),
                "failure": str(out_root / "continuation_failure.json"),
            },
        )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--base-root", required=True)
    parser.add_argument("--continuation-root", required=True)
    parser.add_argument("--patch-manifest", required=True)
    parser.add_argument("--timeout-s", type=int, default=3600)
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = run_continuation(
            out_root=Path(args.out_root),
            base_root=Path(args.base_root),
            continuation_root=Path(args.continuation_root),
            patch_manifest=Path(args.patch_manifest),
            timeout_s=int(args.timeout_s),
            verify_only=bool(args.verify_only),
        )
    except BaseException as error:
        print(f"posthoc continuation failed: {error}", file=sys.stderr)
        return 1
    output = {"status": summary["status"]}
    if args.verify_only:
        output["preflight"] = str(
            Path(args.out_root).expanduser().resolve()
            / "continuation_preflight.json"
        )
    else:
        output["summary"] = str(
            Path(args.out_root).expanduser().resolve()
            / "continuation_summary.json"
        )
        output["new_physical_inner_requests"] = summary["request_layers"][
            "new_physical_inner_requests"
        ]
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
