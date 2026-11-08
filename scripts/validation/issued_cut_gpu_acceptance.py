#!/usr/bin/env python3
"""Bounded GPU check for an issued native request with no terminal result."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence
import uuid


REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_PATH = Path(__file__).with_name("isolated_gpu_acceptance.py")
ISSUED_BUDGET = 1
ROOT_RESULT = Path("candidates/000000/result.json")
SCHEMA = "astrolabe.issued-gpu-acceptance/v1"
CUT_BOUNDARY = (
    "service recorded request_received, then received a controlled SIGSTOP "
    "before executor.submit and gpu_start; this proves a submitted request "
    "with unknown terminal, not interruption of executing GPU inference"
)


def _load_shared():
    name = "astrolabe_issued_cut_shared"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SHARED_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the isolated acceptance helpers")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SHARED = _load_shared()


@dataclass(frozen=True)
class PreparedIssuedCut:
    out_root: Path
    case_config: Path
    source_fingerprint: str
    config: Path
    output: Path
    service_root: Path
    instrumentation: Path


def _source_manifest() -> dict[str, str]:
    manifest = dict(SHARED._source_manifest())
    current = Path(__file__).resolve()
    manifest[current.relative_to(REPO_ROOT).as_posix()] = SHARED._sha256(current)
    return manifest


def _source_fingerprint(manifest: Mapping[str, str]) -> str:
    return hashlib.sha256(
        json.dumps(dict(manifest), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _instrumentation_source() -> str:
    return r'''
from __future__ import annotations
import json
import os
from pathlib import Path
import signal
import time

_receipt = os.environ.get("ASTROLABE_ACCEPTANCE_ISSUED_RECEIPT")
_token = os.environ.get("ASTROLABE_ACCEPTANCE_ISSUED_TOKEN")

if _receipt and _token:
    _original_write = os.write
    _armed = True

    def _identity(pid):
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {
            "pid": pid,
            "start_ticks": fields[19],
            "state": fields[0],
            "process_group": int(fields[2]),
            "session": int(fields[3]),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        }

    def write(fd, data):
        global _armed
        written = _original_write(fd, data)
        if not _armed or fd != 1:
            return written
        try:
            event = json.loads(data)
        except (TypeError, ValueError):
            return written
        if event.get("event") != "request_received":
            return written
        _armed = False
        payload = {
            "schema": "astrolabe.issued-gpu-cut/v1",
            "token": _token,
            "coordinator_pid": os.getppid(),
            "service": _identity(os.getpid()),
            "request_event": event,
            "created_unix_s": time.time(),
        }
        receipt = Path(_receipt)
        receipt.parent.mkdir(parents=True, exist_ok=True)
        temporary = receipt.with_name(receipt.name + "." + str(os.getpid()) + ".tmp")
        with temporary.open("w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, receipt)
        descriptor = os.open(receipt.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.kill(os.getpid(), signal.SIGSTOP)
        return written

    os.write = write
'''.lstrip()


def _prepare(case_config: Path, out_root: Path) -> PreparedIssuedCut:
    from omegaconf import OmegaConf, open_dict

    from astrolabe.protein_beam import validate_config

    out_root = out_root.expanduser().resolve()
    if out_root.exists() and any(out_root.iterdir()):
        raise SHARED.AcceptanceError(f"acceptance output is not empty: {out_root}")
    case_config = SHARED._resolve_case_config(case_config)
    output = out_root / "runs" / "issued"
    service_root = out_root / "runs" / "issued-service"
    SHARED._validate_service_beam_layout(output, service_root)
    cfg = SHARED._bounded_config(case_config, output)
    with open_dict(cfg):
        cfg.protein_beam.max_evaluations = ISSUED_BUDGET
        cfg.protein_beam.af3_call_budget.max_total = ISSUED_BUDGET
        cfg.protein_beam.af3_call_budget.max_per_case = ISSUED_BUDGET
    validate_config(cfg)

    manifest = _source_manifest()
    fingerprint = _source_fingerprint(manifest)
    out_root.mkdir(parents=True, exist_ok=True)
    config = out_root / "configs" / "issued.yaml"
    SHARED._atomic_text(config, OmegaConf.to_yaml(cfg, resolve=False))
    instrumentation = out_root / "instrumentation"
    instrumentation.mkdir()
    SHARED._atomic_text(
        instrumentation / "sitecustomize.py",
        _instrumentation_source(),
    )
    SHARED._atomic_text(
        out_root / "source_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    reasons = SHARED._full_run_reasons(cfg, case_config)
    plan = {
        "schema": f"{SCHEMA}.plan",
        "status": "prepared",
        "case_config": str(case_config),
        "case_config_sha256": SHARED._sha256(case_config),
        "source_fingerprint": fingerprint,
        "seed": int(cfg.seed),
        "evaluation_budget_including_root": ISSUED_BUDGET,
        "af3_quota": {"max_total": ISSUED_BUDGET, "max_per_case": ISSUED_BUDGET},
        "config": str(config),
        "output": str(output),
        "service_root": str(service_root),
        "cut": CUT_BOUNDARY,
        "gpu_execution_interrupted": False,
        "resume": {"retry_interrupted": False},
        "full_run_ready": not reasons,
        "full_run_blockers": reasons,
        "one_owner_at_a_time": True,
    }
    SHARED._atomic_json(out_root / "plan.json", plan)
    return PreparedIssuedCut(
        out_root=out_root,
        case_config=case_config,
        source_fingerprint=fingerprint,
        config=config,
        output=output,
        service_root=service_root,
        instrumentation=instrumentation,
    )


def _coordinator_class() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--service-root", type=Path, required=True)
    parser.add_argument("--source-fingerprint", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser


def _coordinator_main(arguments: argparse.Namespace) -> int:
    from astrolabe.evaluators.af3_lifecycle import (
        OwnedAF3,
        recover_owned_services,
        service_settings,
    )
    from astrolabe.interface import _check_interrupted_attempts, cpu_coordinator_environment
    from astrolabe.protein_beam import load_config, run_beam

    fingerprint = _source_fingerprint(_source_manifest())
    if fingerprint != arguments.source_fingerprint:
        raise SHARED.AcceptanceError("Astrolabe source changed before coordinator start")
    output = arguments.output.expanduser().resolve()
    service_root = arguments.service_root.expanduser().resolve()
    SHARED._validate_service_beam_layout(output, service_root)
    cfg = load_config(arguments.config)
    if arguments.resume:
        recover_owned_services(service_root)
        _check_interrupted_attempts(output, retry=False)
    with cpu_coordinator_environment() as gpu_environment:
        owner = OwnedAF3(
            cfg,
            service_root,
            gpu_environment,
            settings=service_settings(cfg),
        )
        with owner as socket_path:
            state = run_beam(
                cfg,
                output,
                socket_path,
                workers=int(cfg.protein_beam.workers),
                in_flight=int(cfg.protein_beam.in_flight),
                resume=bool(arguments.resume),
                retry_interrupted=False,
                on_abort=owner.abort,
            )
    SHARED._atomic_json(
        output / "coordinator_result.json",
        {
            "status": state["status"],
            "source_fingerprint": fingerprint,
            "resume": bool(arguments.resume),
            "retry_interrupted": False,
        },
    )
    return 0


def _command(
    prepared: PreparedIssuedCut,
    *,
    resume: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--coordinator",
        "--config",
        str(prepared.config),
        "--output",
        str(prepared.output),
        "--service-root",
        str(prepared.service_root),
        "--source-fingerprint",
        prepared.source_fingerprint,
    ]
    if resume:
        command.append("--resume")
    return command


def _environment(
    prepared: PreparedIssuedCut,
    *,
    receipt: Path | None,
    token: str | None,
) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (
            str(prepared.instrumentation),
            str(REPO_ROOT),
            environment.get("PYTHONPATH", ""),
        )
    )
    if receipt is not None:
        environment.update(
            ASTROLABE_ACCEPTANCE_ISSUED_RECEIPT=str(receipt.resolve()),
            ASTROLABE_ACCEPTANCE_ISSUED_TOKEN=str(token),
        )
    else:
        environment.pop("ASTROLABE_ACCEPTANCE_ISSUED_RECEIPT", None)
        environment.pop("ASTROLABE_ACCEPTANCE_ISSUED_TOKEN", None)
    return environment


def _validate_issued_cut(
    prepared: PreparedIssuedCut,
    receipt: Mapping[str, Any],
    *,
    coordinator_pid: int,
    token: str,
) -> dict[str, Any]:
    if receipt.get("schema") != "astrolabe.issued-gpu-cut/v1":
        raise SHARED.AcceptanceError("unknown issued-cut receipt schema")
    if receipt.get("token") != token:
        raise SHARED.AcceptanceError("issued-cut receipt belongs to another harness")
    if int(receipt.get("coordinator_pid", -1)) != coordinator_pid:
        raise SHARED.AcceptanceError("issued-cut receipt belongs to another coordinator")
    service = receipt.get("service")
    if not isinstance(service, Mapping) or not SHARED._same_process(service):
        raise SHARED.AcceptanceError("issued-cut service identity is invalid")
    lifecycle = next(
        (
            path
            for path in (prepared.service_root / "services").glob("*/lifecycle.json")
            if int(json.loads(path.read_text()).get("pid", -1)) == int(service["pid"])
        ),
        None,
    )
    if lifecycle is None:
        raise SHARED.AcceptanceError("issued-cut service has no lifecycle receipt")
    state = json.loads(lifecycle.read_text())
    if (
        str(state.get("pid_start_ticks")) != str(service.get("start_ticks"))
        or state.get("boot_id") != service.get("boot_id")
    ):
        raise SHARED.AcceptanceError("issued-cut service lifecycle identity changed")
    ledger = json.loads((prepared.output / "af3_calls.json").read_text())
    entries = list(ledger.get("entries", {}).values())
    if len(entries) != 1 or entries[0].get("status") != "issued":
        raise SHARED.AcceptanceError("issued-cut did not retain exactly one issued call")
    beam = json.loads((prepared.output / "beam_state.json").read_text())
    running = [record for record in beam.get("records", []) if record.get("status") == "running"]
    if not running or any(record.get("id") != 0 for record in running):
        raise SHARED.AcceptanceError("issued-cut has no running root attempt")
    root_result = prepared.output / ROOT_RESULT
    if root_result.exists():
        raise SHARED.AcceptanceError("issued-cut unexpectedly has a terminal root result")
    events = SHARED._inference_event_projection(prepared.service_root)
    if events["counts"]["request_received"] != 1:
        raise SHARED.AcceptanceError("issued-cut must contain exactly one native request")
    if events["counts"]["gpu_start"] != 0 or events["counts"]["gpu_end"] != 0:
        raise SHARED.AcceptanceError("issued-cut crossed into GPU execution")
    return {
        "schema": f"{SCHEMA}.cut",
        "boundary": CUT_BOUNDARY,
        "gpu_execution_interrupted": False,
        "proves": "request_received was recorded and no terminal result exists",
        "does_not_prove": "an in-progress GPU inference was interrupted",
        "service": dict(service),
        "lifecycle": str(lifecycle),
        "ledger_entries": entries,
        "beam_status": beam.get("status"),
        "running_record_ids": [record.get("id") for record in running],
        "root_result_exists": False,
        "event_counts": events["counts"],
        "retry_interrupted": False,
    }


def _wait_for_receipt(
    process: subprocess.Popen[bytes],
    receipt: Path,
    *,
    timeout_s: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise SHARED.AcceptanceError(
                f"issued-cut coordinator exited {process.returncode} before the cut"
            )
        if receipt.is_file():
            value = json.loads(receipt.read_text())
            if not isinstance(value, dict):
                raise SHARED.AcceptanceError("issued-cut receipt is not a JSON object")
            return value
        time.sleep(0.01)
    raise SHARED.AcceptanceError("timed out waiting for issued request cut")


def _cleanup(
    process: subprocess.Popen[bytes] | None,
    coordinator_identity: Mapping[str, Any] | None,
    service_root: Path,
) -> None:
    if process is not None:
        SHARED._cleanup_owned_process_group(process, coordinator_identity)
    SHARED._terminate_owned_services(service_root)
    SHARED._wait_owned_services_ended(service_root, timeout_s=30)


def _run_issued_cut(
    prepared: PreparedIssuedCut,
    *,
    timeout_s: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = prepared.out_root / "cut" / "issued.receipt.json"
    token = uuid.uuid4().hex
    stdout_log = prepared.out_root / "logs" / "issued.stdout.log"
    stderr_log = prepared.out_root / "logs" / "issued.stderr.log"
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    with stdout_log.open("wb") as stdout, stderr_log.open("wb") as stderr:
        process = None
        identity = None
        try:
            process = subprocess.Popen(
                _command(prepared, resume=False),
                cwd=str(REPO_ROOT),
                env=_environment(prepared, receipt=receipt, token=token),
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
                close_fds=True,
            )
            identity = SHARED._process_identity(process.pid)
            if identity is None or identity["process_group"] != process.pid:
                raise SHARED.AcceptanceError("cannot establish coordinator process group")
            cut = _wait_for_receipt(process, receipt, timeout_s=timeout_s)
            evidence = _validate_issued_cut(
                prepared,
                cut,
                coordinator_pid=process.pid,
                token=token,
            )
            SHARED._atomic_json(prepared.out_root / "cut" / "issued.cut.json", evidence)
        finally:
            _cleanup(process, identity, prepared.service_root)
    if process.returncode not in {-signal.SIGTERM, 128 + signal.SIGTERM}:
        raise SHARED.AcceptanceError(
            f"issued-cut coordinator exited {process.returncode} instead of SIGTERM"
        )
    return cut, evidence


def _run_resume(
    prepared: PreparedIssuedCut,
    *,
    timeout_s: float,
) -> dict[str, Any]:
    stdout_log = prepared.out_root / "logs" / "resume.stdout.log"
    stderr_log = prepared.out_root / "logs" / "resume.stderr.log"
    with stdout_log.open("wb") as stdout, stderr_log.open("wb") as stderr:
        process = subprocess.Popen(
            _command(prepared, resume=True),
            cwd=str(REPO_ROOT),
            env=_environment(prepared, receipt=None, token=None),
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            close_fds=True,
        )
        identity = SHARED._process_identity(process.pid)
        try:
            returncode = process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired as error:
            SHARED._cleanup_owned_process_group(process, identity)
            raise SHARED.AcceptanceError("resume preflight timed out") from error
    error_text = stderr_log.read_text(errors="replace")
    if returncode == 0 or "Interrupted inference lacks a terminal result" not in error_text:
        raise SHARED.AcceptanceError(
            "resume did not stop at the missing native terminal preflight"
        )
    return {
        "returncode": returncode,
        "expected_error": "Interrupted inference lacks a terminal result",
        "stderr": str(stderr_log),
    }


def _execute(
    prepared: PreparedIssuedCut,
    *,
    timeout_s: float,
) -> dict[str, Any]:
    from astrolabe.protein_beam import load_config

    reasons = SHARED._full_run_reasons(
        load_config(prepared.config),
        prepared.case_config,
    )
    if reasons:
        raise SHARED.AcceptanceError("; ".join(reasons))
    cut, cut_evidence = _run_issued_cut(prepared, timeout_s=timeout_s)
    before_ledger = (prepared.output / "af3_calls.json").read_bytes()
    owner_logs_before = {
        path: path.read_bytes()
        for path in (prepared.service_root / "services").glob("*/owner.jsonl")
    }
    services_before = sorted(
        path.name for path in (prepared.service_root / "services").glob("*")
    )
    resume = _run_resume(prepared, timeout_s=timeout_s)
    owner_logs_after = {
        path: path.read_bytes()
        for path in (prepared.service_root / "services").glob("*/owner.jsonl")
    }
    services_after = sorted(
        path.name for path in (prepared.service_root / "services").glob("*")
    )
    ledger_after = (prepared.output / "af3_calls.json").read_bytes()
    issues = []
    if owner_logs_after != owner_logs_before:
        issues.append("resume changed prior owner logs")
    if services_after != services_before:
        issues.append("resume created or removed a model owner record")
    if ledger_after != before_ledger:
        issues.append("resume changed retained issued-call cost")
    summary = {
        "schema": f"{SCHEMA}.summary",
        "status": "PASS" if not issues else "ISSUES",
        "source_fingerprint": prepared.source_fingerprint,
        "evaluation_budget_including_root": ISSUED_BUDGET,
        "cut": cut,
        "cut_evidence": cut_evidence,
        "gpu_execution_interrupted": False,
        "resume": resume,
        "issues": issues,
        "automatic_retry": False,
    }
    SHARED._atomic_json(prepared.out_root / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    if "--coordinator" in (argv if argv is not None else sys.argv[1:]):
        arguments, _unknown = _coordinator_class().parse_known_args(
            (argv if argv is not None else sys.argv[1:])
        )
        return _coordinator_main(arguments)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--case-config", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=10800.0)
    arguments = parser.parse_args(argv)
    if not math.isfinite(arguments.timeout_s) or arguments.timeout_s <= 0:
        parser.error("--timeout-s must be finite and positive")
    prepared = _prepare(arguments.case_config, arguments.out_root)
    if arguments.prepare_only:
        print(json.dumps({"status": "prepared", "plan": str(prepared.out_root / "plan.json")}))
        return 0
    summary = _execute(prepared, timeout_s=arguments.timeout_s)
    print(json.dumps({"status": summary["status"], "summary": str(prepared.out_root / "summary.json")}))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
