#!/usr/bin/env python3
"""Run layout acceptance, then the public interface root and completed resume."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
LAYOUT_PATH = Path(__file__).with_name("isolated_gpu_acceptance.py")
SCHEMA = "astrolabe.layout-public-acceptance/v1"
ROOT_RESULT = Path("candidates/000000/result.json")


def _load_layout_helpers():
    name = "astrolabe_layout_public_shared"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, LAYOUT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load layout acceptance helpers")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SHARED = _load_layout_helpers()


def _source_manifest() -> dict[str, str]:
    manifest = dict(SHARED._source_manifest())
    current = Path(__file__).resolve()
    manifest[current.relative_to(REPO_ROOT).as_posix()] = SHARED._sha256(current)
    return manifest


def _source_fingerprint(manifest: Mapping[str, str]) -> str:
    return hashlib.sha256(
        json.dumps(dict(manifest), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _layout_command(
    *,
    out_root: Path,
    case_config: Path,
    prepare_only: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        str(LAYOUT_PATH),
        "--out-root",
        str(out_root),
        "--case-config",
        str(case_config),
    ]
    if prepare_only:
        command.append("--prepare-only")
    return command


def _public_command(
    *,
    input_path: Path,
    target_homology: str,
    out_root: Path,
    resume: bool = False,
    prepare_only: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "astrolabe.cli",
        "refine",
        "interface",
        "--input",
        str(input_path),
        "--target-homology",
        str(target_homology),
        "--out-dir",
        str(out_root),
        "--budget",
        "1",
        "--target-chain",
        "B",
        "--binder-chain",
        "A",
    ]
    if resume:
        command.append("--resume")
    if prepare_only:
        command.append("--prepare-only")
    return command


def _run(
    command: Sequence[str],
    *,
    label: str,
    log_root: Path,
    timeout_s: float,
) -> int:
    stdout_path = log_root / f"{label}.stdout.log"
    stderr_path = log_root / f"{label}.stderr.log"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            env=os.environ.copy(),
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            close_fds=True,
        )
        identity = SHARED._process_identity(process.pid)
        try:
            return process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired as error:
            SHARED._cleanup_owned_process_group(process, identity)
            raise SHARED.AcceptanceError(f"{label} timed out") from error
        except BaseException:
            SHARED._cleanup_owned_process_group(process, identity)
            raise


def _snapshot(output: Path) -> dict[str, Any]:
    service_root = output / "services"
    services = sorted(path.name for path in service_root.glob("*") if path.is_dir())
    owner_logs = {
        path: path.read_bytes()
        for path in sorted(service_root.glob("*/owner.jsonl"))
    }
    ledger = output / "cases" / "D5056_D15" / "af3_calls.json"
    return {
        "services": services,
        "owner_logs": owner_logs,
        "ledger": ledger.read_bytes(),
        "events": SHARED._inference_event_projection(output),
    }


def _validate_public_first(output: Path) -> None:
    if not (output / "interface_plan.json").is_file():
        raise SHARED.AcceptanceError("public interface wrote no plan")
    if not (output / "interface_results.json").is_file():
        raise SHARED.AcceptanceError("public interface wrote no result summary")
    results = json.loads((output / "interface_results.json").read_text())
    if (
        len(results) != 1
        or results[0].get("status") != "complete"
        or results[0].get("budget_complete") is not True
    ):
        raise SHARED.AcceptanceError("public root did not complete the one-call budget")
    snapshot = _snapshot(output)
    if snapshot["events"]["counts"] != {
        "request_received": 1,
        "gpu_start": 1,
        "gpu_end": 1,
        "gpu_ok": 1,
    }:
        raise SHARED.AcceptanceError("public root did not complete exactly one AF3 request")
    ledger = json.loads(snapshot["ledger"])
    entries = list(ledger.get("entries", {}).values())
    if len(entries) != 1 or entries[0].get("status") != "issued":
        raise SHARED.AcceptanceError("public root did not retain exactly one issued call")
    case_output = output / "cases" / "D5056_D15"
    if not (case_output / ROOT_RESULT).is_file():
        raise SHARED.AcceptanceError("public root wrote no terminal candidate")
    if not (case_output / "candidates" / "000000" / "scores.pkl").is_file():
        raise SHARED.AcceptanceError("public root wrote no score snapshot")


def _execute(
    *,
    case_config: Path,
    input_path: Path,
    target_homology: str,
    out_root: Path,
    timeout_s: float,
) -> dict[str, Any]:
    out_root = out_root.expanduser().resolve()
    if out_root.exists() and any(out_root.iterdir()):
        raise SHARED.AcceptanceError(f"output is not empty: {out_root}")
    layout_root = out_root / "layout"
    public_root = out_root / "public"
    log_root = out_root / "logs"
    out_root.mkdir(parents=True)
    source_manifest = _source_manifest()
    source_fingerprint = _source_fingerprint(source_manifest)
    SHARED._atomic_text(
        out_root / "source_manifest.json",
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n",
    )

    layout_code = _run(
        _layout_command(out_root=layout_root, case_config=case_config),
        label="layout",
        log_root=log_root,
        timeout_s=timeout_s,
    )
    layout_summary_path = layout_root / "summary.json"
    if layout_code != 0 or not layout_summary_path.is_file():
        raise SHARED.AcceptanceError("layout acceptance failed before public interface")
    layout_summary = json.loads(layout_summary_path.read_text())
    if layout_summary.get("status") != "PASS":
        raise SHARED.AcceptanceError("layout acceptance was not PASS; public phase blocked")

    public_code = _run(
        _public_command(
            input_path=input_path,
            target_homology=target_homology,
            out_root=public_root,
        ),
        label="public-root",
        log_root=log_root,
        timeout_s=timeout_s,
    )
    if public_code != 0:
        raise SHARED.AcceptanceError("public interface root run failed")
    _validate_public_first(public_root)
    before = _snapshot(public_root)

    resume_code = _run(
        _public_command(
            input_path=input_path,
            target_homology=target_homology,
            out_root=public_root,
            resume=True,
        ),
        label="public-resume",
        log_root=log_root,
        timeout_s=timeout_s,
    )
    if resume_code != 0:
        raise SHARED.AcceptanceError("completed public interface resume failed")
    after = _snapshot(public_root)
    issues = []
    for key in ("services", "owner_logs", "ledger", "events"):
        if after[key] != before[key]:
            issues.append(f"completed resume changed {key}")
    if _source_fingerprint(_source_manifest()) != source_fingerprint:
        issues.append("source changed during sequential acceptance")
    summary = {
        "schema": f"{SCHEMA}.summary",
        "status": "PASS" if not issues else "ISSUES",
        "source_fingerprint": source_fingerprint,
        "layout_status": layout_summary["status"],
        "layout_native_calls": layout_summary.get("comparison", {}).get(
            "native_call_count"
        ),
        "public_root_native_calls": before["events"]["counts"]["request_received"],
        "public_completed_resume_native_calls": (
            after["events"]["counts"]["request_received"]
            - before["events"]["counts"]["request_received"]
        ),
        "layout_output": str(layout_root),
        "public_output": str(public_root),
        "issues": issues,
        "automatic_retry": False,
    }
    SHARED._atomic_json(out_root / "summary.json", summary)
    return summary


def _prepare(
    *,
    case_config: Path,
    input_path: Path,
    target_homology: str,
    out_root: Path,
) -> dict[str, Any]:
    out_root = out_root.expanduser().resolve()
    if out_root.exists() and any(out_root.iterdir()):
        raise SHARED.AcceptanceError(f"output is not empty: {out_root}")
    layout_root = out_root / "preflight-layout"
    public_root = out_root / "preflight-public"
    log_root = out_root / "logs"
    out_root.mkdir(parents=True)
    source_manifest = _source_manifest()
    source_fingerprint = _source_fingerprint(source_manifest)
    SHARED._atomic_text(
        out_root / "source_manifest.json",
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n",
    )
    layout_code = _run(
        _layout_command(
            out_root=layout_root,
            case_config=case_config,
            prepare_only=True,
        ),
        label="prepare-layout",
        log_root=log_root,
        timeout_s=300,
    )
    public_code = _run(
        _public_command(
            input_path=input_path,
            target_homology=target_homology,
            out_root=public_root,
            prepare_only=True,
        ),
        label="prepare-public",
        log_root=log_root,
        timeout_s=300,
    )
    if layout_code != 0 or public_code != 0:
        raise SHARED.AcceptanceError("no-GPU preflight failed")
    layout_plan = json.loads((layout_root / "plan.json").read_text())
    public_plan = json.loads((public_root / "interface_plan.json").read_text())
    summary = {
        "schema": f"{SCHEMA}.preflight",
        "status": "PASS",
        "source_fingerprint": source_fingerprint,
        "layout_ready": layout_plan.get("full_run_ready"),
        "layout_blockers": layout_plan.get("full_run_blockers"),
        "public_cases": len(public_plan.get("cases", [])),
        "new_af3_calls": 0,
    }
    SHARED._atomic_json(out_root / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--case-config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--target-homology", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=7200.0)
    arguments = parser.parse_args(argv)
    if arguments.timeout_s <= 0:
        parser.error("--timeout-s must be positive")
    if threading.current_thread() is not threading.main_thread():
        parser.error("acceptance must run in the main thread")
    previous = {
        signum: signal.getsignal(signum)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }

    def terminate(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    for signum in previous:
        signal.signal(signum, terminate)
    try:
        if arguments.prepare_only:
            summary = _prepare(
                case_config=arguments.case_config,
                input_path=arguments.input,
                target_homology=arguments.target_homology,
                out_root=arguments.out_root,
            )
        else:
            summary = _execute(
                case_config=arguments.case_config,
                input_path=arguments.input,
                target_homology=arguments.target_homology,
                out_root=arguments.out_root,
                timeout_s=arguments.timeout_s,
            )
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    print(json.dumps({"status": summary["status"], "summary": str(arguments.out_root / "summary.json")}))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
