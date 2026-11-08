#!/usr/bin/env python3
"""Exercise issued-no-terminal recovery through the real public interface CLI."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence
import uuid


REPO_ROOT = Path(__file__).resolve().parents[2]
ISSUED_PATH = Path(__file__).with_name("issued_cut_gpu_acceptance.py")
PUBLIC_PATH = Path(__file__).with_name("public_interface_final.py")
SCHEMA = "astrolabe.issued-public-acceptance/v1"
ROOT_RESULT = Path("candidates/000000/result.json")


def _load(name: str, path: Path):
    loaded = sys.modules.get(name)
    if loaded is not None:
        return loaded
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load acceptance helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ISSUED = _load("astrolabe_issued_public_issued", ISSUED_PATH)
PUBLIC = _load("astrolabe_issued_public_public", PUBLIC_PATH)
LAYOUT = PUBLIC.LAYOUT
SHARED = PUBLIC.SHARED


def _source_manifest() -> dict[str, str]:
    manifest = dict(SHARED._source_manifest())
    for path in (
        ISSUED_PATH.resolve(),
        LAYOUT.LAYOUT_PATH.resolve(),
        PUBLIC_PATH.resolve(),
        Path(__file__).resolve(),
    ):
        manifest[path.relative_to(REPO_ROOT).as_posix()] = SHARED._sha256(path)
    return manifest


def _source_fingerprint(manifest: Mapping[str, str]) -> str:
    return hashlib.sha256(
        json.dumps(dict(manifest), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _public_command(
    *,
    input_path: Path,
    target_homology: str,
    out_root: Path,
    resume: bool = False,
    prepare_only: bool = False,
) -> list[str]:
    return LAYOUT._public_command(
        input_path=input_path,
        target_homology=target_homology,
        out_root=out_root,
        resume=resume,
        prepare_only=prepare_only,
    )


def _prepare(
    *,
    input_path: Path,
    target_homology: str,
    preflight_root: Path,
    execution_root: Path,
    target_python: Path,
) -> dict[str, Any]:
    preflight_root = preflight_root.expanduser().resolve()
    execution_root = execution_root.expanduser().resolve()
    if execution_root.exists():
        raise SHARED.AcceptanceError(
            f"execution root must not exist during preflight: {execution_root}"
        )
    if preflight_root.exists() and any(preflight_root.iterdir()):
        raise SHARED.AcceptanceError(f"preflight root is not empty: {preflight_root}")
    preflight_root.mkdir(parents=True, exist_ok=True)
    public_root = preflight_root / "public"
    code = LAYOUT._run(
        _public_command(
            input_path=input_path,
            target_homology=target_homology,
            out_root=public_root,
            prepare_only=True,
        ),
        label="prepare-public",
        log_root=preflight_root / "logs",
        timeout_s=600,
    )
    if code != 0:
        raise SHARED.AcceptanceError("public prepare-only command failed")
    if (public_root / "services").exists():
        raise SHARED.AcceptanceError("public prepare-only created a model owner")
    plan = json.loads((public_root / "interface_plan.json").read_text())
    if len(plan.get("cases", [])) != 1:
        raise SHARED.AcceptanceError("public prepare-only did not produce one case")
    from astrolabe.protein_beam import load_config

    cfg = load_config(public_root / plan["cases"][0]["config"])
    science = {
        "max_evaluations": int(cfg.protein_beam.max_evaluations),
        "max_generations": cfg.protein_beam.max_generations,
        "max_total": int(cfg.protein_beam.af3_call_budget.max_total),
        "max_per_case": int(cfg.protein_beam.af3_call_budget.max_per_case),
        "num_samples": int(cfg.evaluator.alphafold3.num_samples),
        "num_recycles": int(cfg.evaluator.alphafold3.num_recycles),
        "seed": int(cfg.seed),
    }
    expected = {
        "max_evaluations": 1,
        "max_generations": None,
        "max_total": 1,
        "max_per_case": 1,
        "num_samples": 1,
        "num_recycles": 3,
        "seed": 10079,
    }
    if science != expected:
        raise SHARED.AcceptanceError(
            f"public preset scientific config changed: {science} != {expected}"
        )
    closure = PUBLIC._runtime_closure(target_python)
    manifest = _source_manifest()
    record = {
        "schema": f"{SCHEMA}.plan",
        "status": "prepared",
        "source_fingerprint": _source_fingerprint(manifest),
        "input": str(input_path),
        "target_homology": str(target_homology),
        "preflight_root": str(preflight_root),
        "execution_root": str(execution_root),
        "public_preflight_config": str(public_root / plan["cases"][0]["config"]),
        "science": science,
        "runtime_closure": closure,
        "cut_boundary": ISSUED.CUT_BOUNDARY,
        "gpu_execution_interrupted": False,
        "expected_before_cut": {
            "delivered_requests": 1,
            "gpu_start": 0,
            "gpu_end": 0,
            "model_load": 1,
        },
        "expected_after_resume": {
            "new_requests": 0,
            "new_model_loads": 0,
            "new_services": 0,
        },
        "resource_envelope": {
            "cpus": 4,
            "gpu": 1,
            "memory_gb": 28,
            "partition": "boron",
            "time_minutes": 60,
        },
        "new_af3_calls": 0,
        "model_started": False,
    }
    SHARED._atomic_json(preflight_root / "plan.json", record)
    SHARED._atomic_text(
        preflight_root / "source_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    return record


def _environment(instrumentation: Path, *, receipt: Path | None, token: str | None) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        item
        for item in (
            str(instrumentation),
            str(REPO_ROOT),
            environment.get("PYTHONPATH", ""),
        )
        if item
    )
    if receipt is not None:
        environment.update(
            ASTROLABE_ACCEPTANCE_ISSUED_RECEIPT=str(receipt),
            ASTROLABE_ACCEPTANCE_ISSUED_TOKEN=str(token),
        )
    else:
        environment.pop("ASTROLABE_ACCEPTANCE_ISSUED_RECEIPT", None)
        environment.pop("ASTROLABE_ACCEPTANCE_ISSUED_TOKEN", None)
    return environment


def _start_public(
    *,
    input_path: Path,
    target_homology: str,
    out_root: Path,
    log_root: Path,
    label: str,
    resume: bool,
    environment: Mapping[str, str],
) -> tuple[subprocess.Popen[bytes], Mapping[str, Any] | None]:
    stdout_path = log_root / f"{label}.stdout.log"
    stderr_path = log_root / f"{label}.stderr.log"
    log_root.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            _public_command(
                input_path=input_path,
                target_homology=target_homology,
                out_root=out_root,
                resume=resume,
            ),
            cwd=str(REPO_ROOT),
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            close_fds=True,
        )
    identity = SHARED._process_identity(process.pid)
    return process, identity


def _snapshot(service_root: Path, case_output: Path) -> dict[str, Any]:
    services = sorted(
        path.name for path in (service_root / "services").glob("*") if path.is_dir()
    )
    owner_logs = {
        str(path): path.read_bytes()
        for path in sorted((service_root / "services").glob("*/owner.jsonl"))
    }
    diagnostics = {
        str(path): path.read_bytes()
        for path in sorted((service_root / "services").glob("*/owner.stderr.log"))
    }
    ledger_path = case_output / "af3_calls.json"
    all_counts: dict[str, int] = {}
    for content in owner_logs.values():
        for line in content.decode("utf-8").splitlines():
            if not line.startswith("{"):
                raise SHARED.AcceptanceError("owner stdout contains non-event data")
            event = json.loads(line)
            if not isinstance(event.get("event"), str):
                raise SHARED.AcceptanceError("owner stdout event lacks event name")
            name = event["event"]
            all_counts[name] = all_counts.get(name, 0) + 1
    return {
        "services": services,
        "owner_logs": owner_logs,
        "diagnostics": diagnostics,
        "ledger": ledger_path.read_bytes() if ledger_path.is_file() else None,
        "events": SHARED._inference_event_projection(service_root),
        "all_event_counts": dict(sorted(all_counts.items())),
    }


def _execute(
    *,
    input_path: Path,
    target_homology: str,
    preflight_root: Path,
    out_root: Path,
    timeout_s: float,
) -> dict[str, Any]:
    preflight = json.loads((preflight_root / "plan.json").read_text())
    fingerprint = _source_fingerprint(_source_manifest())
    if fingerprint != preflight.get("source_fingerprint"):
        raise SHARED.AcceptanceError("source changed after CPU preflight")
    out_root = out_root.expanduser().resolve()
    if out_root.exists():
        raise SHARED.AcceptanceError(f"public output already exists: {out_root}")
    control_root = out_root.parent / "control"
    log_root = control_root / "logs"
    instrumentation = control_root / "instrumentation"
    instrumentation.mkdir(parents=True, exist_ok=False)
    SHARED._atomic_text(
        instrumentation / "sitecustomize.py",
        ISSUED._instrumentation_source(),
    )
    receipt = control_root / "cut" / "issued.receipt.json"
    token = uuid.uuid4().hex
    process = None
    identity = None
    try:
        process, identity = _start_public(
            input_path=input_path,
            target_homology=target_homology,
            out_root=out_root,
            log_root=log_root,
            label="public-issued",
            resume=False,
            environment=_environment(instrumentation, receipt=receipt, token=token),
        )
        if (
            identity is None
            or identity.get("process_group") != process.pid
            or identity.get("session") != process.pid
        ):
            raise SHARED.AcceptanceError("cannot establish public CLI process group")
        cut = ISSUED._wait_for_receipt(process, receipt, timeout_s=timeout_s)
        output_plan = json.loads((out_root / "interface_plan.json").read_text())
        if len(output_plan.get("cases", [])) != 1:
            raise SHARED.AcceptanceError("public issued run did not create one case")
        case = output_plan["cases"][0]
        case_output = out_root / case["output"]
        prepared = ISSUED.PreparedIssuedCut(
            out_root=control_root,
            case_config=Path(preflight["public_preflight_config"]),
            source_fingerprint=fingerprint,
            config=out_root / case["config"],
            output=case_output,
            service_root=out_root,
            instrumentation=instrumentation,
        )
        cut_evidence = ISSUED._validate_issued_cut(
            prepared,
            cut,
            coordinator_pid=process.pid,
            token=token,
        )
        SHARED._atomic_json(control_root / "cut" / "issued.cut.json", cut_evidence)
    finally:
        if process is not None:
            ISSUED._cleanup(process, identity, out_root)

    case = json.loads((out_root / "interface_plan.json").read_text())["cases"][0]
    case_output = out_root / case["output"]
    before = _snapshot(out_root, case_output)
    resume_process, resume_identity = _start_public(
        input_path=input_path,
        target_homology=target_homology,
        out_root=out_root,
        log_root=log_root,
        label="public-resume-preflight",
        resume=True,
        environment=_environment(instrumentation, receipt=None, token=None),
    )
    try:
        try:
            resume_code = resume_process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired as error:
            SHARED._cleanup_owned_process_group(resume_process, resume_identity)
            raise SHARED.AcceptanceError("public resume preflight timed out") from error
    finally:
        if resume_process.poll() is None:
            SHARED._cleanup_owned_process_group(resume_process, resume_identity)
    after = _snapshot(out_root, case_output)
    resume_stderr = (log_root / "public-resume-preflight.stderr.log").read_text(
        errors="replace"
    )
    checks = {
        "cut_has_one_delivered_request": (
            before["events"]["counts"].get("request_received") == 1
        ),
        "cut_has_zero_gpu_start": before["events"]["counts"].get("gpu_start", 0) == 0,
        "cut_has_zero_gpu_end": before["events"]["counts"].get("gpu_end", 0) == 0,
        "cut_loaded_model_once": (
            before["all_event_counts"].get("model_load_start") == 1
            and before["all_event_counts"].get("model_load_end") == 1
        ),
        "cut_retained_issued_cost": (
            before["ledger"] is not None
            and len(json.loads(before["ledger"]).get("entries", {})) == 1
        ),
        "cut_has_no_terminal_root": not (case_output / ROOT_RESULT).exists(),
        "resume_stopped_on_missing_terminal": (
            resume_code != 0
            and "Interrupted inference lacks a terminal result" in resume_stderr
        ),
        "resume_added_no_requests": (
            after["events"]["counts"].get("request_received", 0)
            == before["events"]["counts"].get("request_received", 0)
        ),
        "resume_added_no_model_load": (
            after["events"]["counts"].get("model_load_start", 0)
            == before["events"]["counts"].get("model_load_start", 0)
            and after["all_event_counts"].get("model_load_start", 0)
            == before["all_event_counts"].get("model_load_start", 0)
        ),
        "resume_added_no_services": after["services"] == before["services"],
        "resume_preserved_owner_logs": after["owner_logs"] == before["owner_logs"],
        "resume_preserved_diagnostics": after["diagnostics"] == before["diagnostics"],
        "resume_preserved_issued_ledger": after["ledger"] == before["ledger"],
    }
    summary = {
        "schema": f"{SCHEMA}.summary",
        "attempt": "issued_public_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "ISSUES",
        "source_fingerprint": fingerprint,
        "checks": checks,
        "cut": json.loads((control_root / "cut" / "issued.cut.json").read_text()),
        "before_resume": {
            "services": before["services"],
            "events": before["events"],
            "all_event_counts": before["all_event_counts"],
            "ledger": before["ledger"].decode() if before["ledger"] is not None else None,
        },
        "after_resume": {
            "services": after["services"],
            "events": after["events"],
            "all_event_counts": after["all_event_counts"],
            "ledger": after["ledger"].decode() if after["ledger"] is not None else None,
        },
        "resume_returncode": resume_code,
        "resume_stderr": str(log_root / "public-resume-preflight.stderr.log"),
        "gpu_execution_interrupted": False,
        "automatic_retry": False,
        "runtime_closure": preflight["runtime_closure"],
    }
    SHARED._atomic_json(out_root.parent / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--preflight-root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--target-homology", required=True)
    parser.add_argument("--target-python", type=Path, default=Path("/alphafold3_venv/bin/python"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=3600.0)
    arguments = parser.parse_args(argv)
    if not (arguments.timeout_s > 0):
        parser.error("--timeout-s must be positive")
    try:
        if arguments.prepare_only:
            result = _prepare(
                input_path=arguments.input,
                target_homology=arguments.target_homology,
                preflight_root=arguments.preflight_root,
                execution_root=arguments.out_root,
                target_python=arguments.target_python,
            )
        else:
            result = _execute(
                input_path=arguments.input,
                target_homology=arguments.target_homology,
                preflight_root=arguments.preflight_root,
                out_root=arguments.out_root,
                timeout_s=arguments.timeout_s,
            )
    except Exception as error:
        summary_path = (
            arguments.preflight_root / "summary.json"
            if arguments.prepare_only
            else arguments.out_root.parent / "summary.json"
        )
        SHARED._atomic_json(
            summary_path,
            {
                "schema": f"{SCHEMA}.summary",
                "status": "ISSUES",
                "error_type": type(error).__name__,
                "error": str(error),
                "automatic_retry": False,
            },
        )
        print(json.dumps({"status": "ISSUES", "summary": str(summary_path), "error": str(error)}))
        return 1
    print(json.dumps({"status": result["status"], "summary": str(
        arguments.preflight_root / "plan.json"
        if arguments.prepare_only else arguments.out_root.parent / "summary.json"
    )}))
    return 0 if result["status"] in {"prepared", "PASS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
