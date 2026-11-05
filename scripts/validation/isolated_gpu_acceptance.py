#!/usr/bin/env python3
"""Bounded real-GPU acceptance for the isolated protein-beam owner."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence
import uuid


REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_BUDGET = 2
ROOT_CANDIDATE = Path("candidates/000000/result.json")
SCHEMA = "astrolabe.isolated-gpu-acceptance/v1"


class AcceptanceError(RuntimeError):
    pass


@dataclass(frozen=True)
class PreparedAcceptance:
    out_root: Path
    case_config: Path
    source_fingerprint: str
    baseline_config: Path
    recovery_config: Path
    baseline_output: Path
    recovery_output: Path
    instrumentation: Path
    baseline_service_root: Path | None = None
    recovery_service_root: Path | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_service_beam_layout(beam_output: Path, service_root: Path) -> None:
    """The model owner is a sibling of its beam case, as in run_interface()."""
    beam_output = beam_output.expanduser().resolve()
    service_root = service_root.expanduser().resolve()
    if beam_output == service_root:
        raise AcceptanceError("AF3 service root must differ from beam output")
    if beam_output.parent != service_root.parent:
        raise AcceptanceError("AF3 service root must be a sibling of beam output")
    if beam_output in service_root.parents or service_root in beam_output.parents:
        raise AcceptanceError("AF3 service and beam output must not nest")


def _boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def _process_identity(pid: int) -> dict[str, Any] | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {
            "pid": pid,
            "start_ticks": fields[19],
            "state": fields[0],
            "process_group": int(fields[2]),
            "session": int(fields[3]),
            "boot_id": _boot_id(),
        }
    except FileNotFoundError:
        return None


def _same_process(identity: Mapping[str, Any], *, require_running: bool = True) -> bool:
    if identity.get("boot_id") != _boot_id():
        return False
    current = _process_identity(int(identity["pid"]))
    return bool(
        current
        and current["start_ticks"] == str(identity["start_ticks"])
        and (not require_running or current["state"] != "Z")
    )


def _owned_group_members(identity: Mapping[str, Any]) -> list[dict[str, Any]]:
    if identity.get("boot_id") != _boot_id():
        return []
    coordinator_pid = int(identity["pid"])
    if (
        int(identity.get("process_group", -1)) != coordinator_pid
        or int(identity.get("session", -1)) != coordinator_pid
    ):
        return []
    members = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == os.getpid():
            continue
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            uid_fields = Path(f"/proc/{pid}/status").read_text().splitlines()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        uid = next(
            (
                int(line.split()[1])
                for line in uid_fields
                if line.startswith("Uid:")
            ),
            -1,
        )
        if (
            int(fields[2]) == coordinator_pid
            and int(fields[3]) == coordinator_pid
            and fields[0] != "Z"
            and uid == os.getuid()
        ):
            members.append(
                {
                    "pid": pid,
                    "start_ticks": fields[19],
                    "state": fields[0],
                }
            )
    return members


def _resolve_case_config(value: Path) -> Path:
    path = value.expanduser().resolve()
    if path.is_file():
        return path
    if not path.is_dir():
        raise AcceptanceError(f"case config does not exist: {path}")
    for name in ("03_D5056_D15.yaml", "02_fixture.yaml"):
        candidates = sorted(path.rglob(name))
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise AcceptanceError(
                f"case-config directory has multiple {name} files: "
                + ", ".join(str(item) for item in candidates)
            )
    raise AcceptanceError(
        "case-config directory has no unique 03_D5056_D15.yaml or 02_fixture.yaml"
    )


def _clone_config(cfg: Any) -> Any:
    from omegaconf import OmegaConf

    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))


def _bounded_config(case_config: Path, output: Path) -> Any:
    from omegaconf import OmegaConf, open_dict

    from astrolabe.protein_beam import load_config, validate_config

    cfg = load_config(case_config)
    cfg = _clone_config(cfg)
    cfg._set_flag("allow_objects", True)
    output = output.expanduser().resolve()
    with open_dict(cfg):
        cfg.out_dir = str(output)
        cfg.raw_model_results_dir = str(output / "raw_model_results")
        cfg.protein_beam.max_evaluations = EVALUATION_BUDGET
        cfg.protein_beam.max_generations = None
        quota = cfg.protein_beam.get("af3_call_budget")
        if quota is None:
            cfg.protein_beam.af3_call_budget = {}
            quota = cfg.protein_beam.af3_call_budget
        quota.max_total = EVALUATION_BUDGET
        quota.max_per_case = EVALUATION_BUDGET
        quota.ledger = str(output / "af3_calls.json")
    validate_config(cfg)
    return cfg


def _homology_path(specification: Any, config_path: Path) -> Path | None:
    if not isinstance(specification, str):
        return None
    value = specification
    if ".json:" in value:
        value = value.split(".json:", 1)[0] + ".json"
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def _full_run_reasons(cfg: Any, case_config: Path) -> list[str]:
    from astrolabe.evaluators.af3_lifecycle import service_settings

    reasons: list[str] = []
    try:
        service_settings(cfg)
    except (TypeError, ValueError) as error:
        reasons.append(f"AF3 service settings are invalid: {error}")
    case_input = cfg.get("case_input") or {}
    structure_value = case_input.get("structure")
    if not structure_value:
        reasons.append("case_input.structure is required for real geometry")
    else:
        structure = Path(str(structure_value)).expanduser()
        if not structure.is_absolute():
            structure = case_config.parent / structure
        if not structure.resolve().is_file():
            reasons.append(f"case_input.structure is unavailable: {structure.resolve()}")

    homology = _homology_path((cfg.get("homologies") or {}).get("B"), case_config)
    if homology is None:
        reasons.append("homologies.B must name the fixed target MSA JSON")
    elif not homology.is_file():
        reasons.append(f"fixed target MSA is unavailable: {homology}")

    alpha = cfg.evaluator.alphafold3
    if int(alpha.get("num_samples", 0)) != 1:
        reasons.append("AF3 num_samples must remain 1")
    if int(alpha.get("num_recycles", 0)) != 3:
        reasons.append("AF3 num_recycles must remain 3")
    model_dir = Path(str(alpha.get("model_dir", "???"))).expanduser()
    if str(model_dir) == "???" or "${" in str(model_dir):
        reasons.append("AF3 model_dir is unresolved; set GH_WEIGHT_AF3")
    elif not (model_dir / "af3.bin").is_file():
        reasons.append(f"AF3 weights are unavailable: {model_dir / 'af3.bin'}")

    settings = cfg.get("protein_beam") or {}
    calibration = settings.get("root_calibration") or {}
    fixed_root_context = bool(
        calibration.get("enabled", False)
        or settings.get("root_parity_reference")
        or settings.get("initial_state")
        or case_input.get("root_structure")
    )
    if not fixed_root_context:
        reasons.append("native root calibration or parity context is required")
    parity = settings.get("root_parity_reference")
    if parity:
        parity_path = Path(str(parity)).expanduser()
        if not parity_path.is_file():
            reasons.append(f"root parity reference is unavailable: {parity_path}")
    gates = settings.get("gates") or {}
    if not gates.get("geometry_max"):
        reasons.append("real geometry gates must be preserved")

    alias = cfg.stages.default.mutators[0]
    term = cfg.mutator.mutator_terms[alias]
    base = cfg.mutator.LigandMPNN
    model_type = str(term.get("model_type", base.model_type))
    checkpoint_key = {
        "ligand_mpnn": "checkpoint_ligand_mpnn",
        "protein_mpnn": "checkpoint_protein_mpnn",
        "soluble_mpnn": "checkpoint_soluble_mpnn",
    }.get(model_type)
    if checkpoint_key is None:
        reasons.append(f"unsupported LigandMPNN model_type: {model_type}")
    else:
        checkpoint = Path(
            str(term.get(checkpoint_key, base.get(checkpoint_key, "???")))
        ).expanduser()
        if str(checkpoint) == "???" or "${" in str(checkpoint):
            reasons.append(
                "LigandMPNN checkpoint is unresolved; set GH_WEIGHT_LIGANDMPNN"
            )
        elif not checkpoint.is_file():
            reasons.append(f"LigandMPNN checkpoint is unavailable: {checkpoint}")
    return reasons


def _source_manifest() -> dict[str, str]:
    paths = sorted((REPO_ROOT / "astrolabe").rglob("*.py"))
    paths.extend([REPO_ROOT / "run.py", Path(__file__).resolve()])
    return {
        str(path.relative_to(REPO_ROOT)): _sha256(path)
        for path in paths
    }


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

_root = os.environ.get("ASTROLABE_ACCEPTANCE_ROOT_RESULT")
_receipt = os.environ.get("ASTROLABE_ACCEPTANCE_CUT_RECEIPT")
_token = os.environ.get("ASTROLABE_ACCEPTANCE_CUT_TOKEN")

if _root and _receipt and _token:
    _original_replace = os.replace

    def _identity(pid):
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {
            "pid": pid,
            "start_ticks": fields[19],
            "state": fields[0],
            "process_group": int(fields[2]),
            "session": int(fields[3]),
        }

    def _publish_cut(source, destination):
        root = Path(_root).resolve()
        target = Path(destination).resolve()
        if target != root:
            return None
        payload = {
            "schema": "astrolabe.isolated-gpu-cut/v1",
            "token": _token,
            "coordinator_pid": os.getppid(),
            "worker": dict(
                _identity(os.getpid()),
                boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            ),
            "root_result": str(root),
            "root_sha256": __import__("hashlib").sha256(root.read_bytes()).hexdigest(),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
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
        _original_replace(temporary, receipt)
        descriptor = os.open(receipt.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.kill(os.getpid(), signal.SIGSTOP)
        return payload

    def replace(source, destination):
        result = _original_replace(source, destination)
        _publish_cut(source, destination)
        return result

    os.replace = replace
'''.lstrip()


def _prepare_acceptance(case_config: Path, out_root: Path) -> PreparedAcceptance:
    out_root = out_root.expanduser().resolve()
    if out_root.exists() and any(out_root.iterdir()):
        raise AcceptanceError(f"acceptance output is not empty: {out_root}")
    case_config = _resolve_case_config(case_config)
    baseline_output = out_root / "runs" / "baseline"
    baseline_service_root = out_root / "runs" / "baseline-service"
    recovery_output = out_root / "runs" / "recovery"
    recovery_service_root = out_root / "runs" / "recovery-service"
    _validate_service_beam_layout(baseline_output, baseline_service_root)
    _validate_service_beam_layout(recovery_output, recovery_service_root)
    baseline_cfg = _bounded_config(case_config, baseline_output)
    recovery_cfg = _bounded_config(case_config, recovery_output)
    reasons = _full_run_reasons(recovery_cfg, case_config)

    manifest = _source_manifest()
    fingerprint = _source_fingerprint(manifest)
    out_root.mkdir(parents=True, exist_ok=True)
    instrumentation = out_root / "instrumentation"
    instrumentation.mkdir()
    _atomic_text(instrumentation / "sitecustomize.py", _instrumentation_source())
    baseline_config = out_root / "configs" / "baseline.yaml"
    recovery_config = out_root / "configs" / "recovery.yaml"
    baseline_config.parent.mkdir()
    from omegaconf import OmegaConf

    _atomic_text(
        baseline_config, OmegaConf.to_yaml(baseline_cfg, resolve=False)
    )
    _atomic_text(
        recovery_config, OmegaConf.to_yaml(recovery_cfg, resolve=False)
    )
    _atomic_text(
        out_root / "source_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    plan = {
        "schema": f"{SCHEMA}.plan",
        "status": "prepared",
        "case_config": str(case_config),
        "case_config_sha256": _sha256(case_config),
        "source_fingerprint": fingerprint,
        "seed": int(recovery_cfg.seed),
        "evaluation_budget_including_root": EVALUATION_BUDGET,
        "af3_quota": {
            "max_total": EVALUATION_BUDGET,
            "max_per_case": EVALUATION_BUDGET,
        },
        "runs": {
            "baseline": {
                "config": str(baseline_config),
                "output": str(baseline_output),
                "service_root": str(baseline_service_root),
            },
            "recovery": {
                "config": str(recovery_config),
                "output": str(recovery_output),
                "service_root": str(recovery_service_root),
                "cut_after": str(recovery_output / ROOT_CANDIDATE),
                "resume": True,
                "retry_interrupted": False,
            },
        },
        "owner": "astrolabe.evaluators.af3_lifecycle.OwnedAF3",
        "coordinator": "astrolabe.protein_beam.run_beam",
        "one_owner_at_a_time": True,
        "full_run_ready": not reasons,
        "full_run_blockers": reasons,
        "required_runtime": {
            "gpu": True,
            "python": sys.executable,
            "model_assets": "must resolve from the preserved case config",
            "accepted_case_root": "must remain visible at the paths embedded in case_config",
            "writable_paths": [str(out_root), "/tmp"],
        },
    }
    _atomic_json(out_root / "plan.json", plan)
    return PreparedAcceptance(
        out_root=out_root,
        case_config=case_config,
        source_fingerprint=fingerprint,
        baseline_config=baseline_config,
        recovery_config=recovery_config,
        baseline_output=baseline_output,
        recovery_output=recovery_output,
        instrumentation=instrumentation,
        baseline_service_root=baseline_service_root,
        recovery_service_root=recovery_service_root,
    )


def _assert_full_run_ready(prepared: PreparedAcceptance) -> None:
    from astrolabe.protein_beam import load_config

    cfg = load_config(prepared.recovery_config)
    reasons = _full_run_reasons(cfg, prepared.case_config)
    if reasons:
        raise AcceptanceError("; ".join(reasons))


def _scientific_preflight(config_path: Path) -> dict[str, Any]:
    from astrolabe.protein_beam import _scientific_contract_config, load_config
    from astrolabe.protein_beam_contract import scientific_contract

    cfg = load_config(config_path)
    contract, digest = scientific_contract(_scientific_contract_config(cfg))
    return {
        "digest": digest,
        "source_sha256": contract["source_sha256"],
        "data_sha256": contract["data_sha256"],
        "readonly_weight_asset_stat": contract["readonly_weight_asset_stat"],
    }


def _coordinator_command(
    *,
    python_bin: Path,
    config: Path,
    output: Path,
    service_root: Path,
    resume: bool,
    source_fingerprint: str,
    receipt: Path | None,
) -> list[str]:
    command = [
        str(python_bin),
        str(Path(__file__).resolve()),
        "--coordinator",
        "--config",
        str(config),
        "--output",
        str(output),
        "--service-root",
        str(service_root),
        "--source-fingerprint",
        source_fingerprint,
    ]
    if resume:
        command.append("--resume")
    if receipt is not None:
        command.extend(["--cut-receipt", str(receipt)])
    return command


def _coordinator_main(arguments: argparse.Namespace) -> int:
    from astrolabe.evaluators.af3_lifecycle import (
        OwnedAF3,
        recover_owned_services,
        service_settings,
    )
    from astrolabe.interface import _check_interrupted_attempts, cpu_coordinator_environment
    from astrolabe.protein_beam import load_config, run_beam

    current_fingerprint = _source_fingerprint(_source_manifest())
    if current_fingerprint != arguments.source_fingerprint:
        raise AcceptanceError("Astrolabe source changed before coordinator start")
    cfg = load_config(arguments.config)
    transport_settings = service_settings(cfg)
    output = arguments.output.expanduser().resolve()
    service_root = arguments.service_root.expanduser().resolve()
    _validate_service_beam_layout(output, service_root)
    if arguments.resume:
        recover_owned_services(service_root)
        # Missing native terminal evidence must stop before creating another
        # model owner. This preserves retained cost and unknown status.
        _check_interrupted_attempts(output, retry=False)
    with cpu_coordinator_environment() as gpu_environment:
        service = OwnedAF3(
            cfg,
            service_root,
            gpu_environment,
            settings=transport_settings,
        )
        with service as socket_path:
            state = run_beam(
                cfg,
                output,
                socket_path,
                workers=int(cfg.protein_beam.workers),
                in_flight=int(cfg.protein_beam.in_flight),
                resume=bool(arguments.resume),
                retry_interrupted=False,
                on_abort=service.abort,
            )
    _atomic_json(
        output / "coordinator_result.json",
        {
            "status": state["status"],
            "source_fingerprint": current_fingerprint,
            "resume": bool(arguments.resume),
            "retry_interrupted": False,
        },
    )
    return 0


def _coordinator_environment(
    prepared: PreparedAcceptance,
    *,
    receipt: Path | None,
    token: str | None,
) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        item
        for item in (
            str(prepared.instrumentation),
            str(REPO_ROOT),
            env.get("PYTHONPATH", ""),
        )
        if item
    )
    env["ASTROLABE_ACCEPTANCE_SOURCE_FINGERPRINT"] = prepared.source_fingerprint
    if receipt is not None:
        env.update(
            ASTROLABE_ACCEPTANCE_ROOT_RESULT=str(
                (prepared.recovery_output / ROOT_CANDIDATE).resolve()
            ),
            ASTROLABE_ACCEPTANCE_CUT_RECEIPT=str(receipt.resolve()),
            ASTROLABE_ACCEPTANCE_CUT_TOKEN=str(token),
        )
    else:
        for key in (
            "ASTROLABE_ACCEPTANCE_ROOT_RESULT",
            "ASTROLABE_ACCEPTANCE_CUT_RECEIPT",
            "ASTROLABE_ACCEPTANCE_CUT_TOKEN",
        ):
            env.pop(key, None)
    return env


def _run_coordinator(
    prepared: PreparedAcceptance,
    *,
    config: Path,
    output: Path,
    service_root: Path,
    resume: bool,
    label: str,
    timeout_s: float,
) -> dict[str, Any]:
    stdout_log = prepared.out_root / "logs" / f"{label}.stdout.log"
    stderr_log = prepared.out_root / "logs" / f"{label}.stderr.log"
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    with stdout_log.open("wb") as stdout, stderr_log.open("wb") as stderr:
        process = subprocess.Popen(
            _coordinator_command(
                python_bin=Path(sys.executable),
                config=config,
                output=output,
                service_root=service_root,
                resume=resume,
                source_fingerprint=prepared.source_fingerprint,
                receipt=None,
            ),
            cwd=str(REPO_ROOT),
            env=_coordinator_environment(
                prepared,
                receipt=None,
                token=None,
            ),
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            close_fds=True,
        )
        identity = _process_identity(process.pid)
        try:
            returncode = process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired as error:
            if identity is None or not _terminate_owned_group(
                process.pid, identity["start_ticks"]
            ):
                process.kill()
            process.wait(timeout=30)
            raise AcceptanceError(f"{label} coordinator timed out") from error
    if returncode != 0:
        raise AcceptanceError(
            f"{label} coordinator exited {returncode}; see {stderr_log}"
        )
    result_path = output / "coordinator_result.json"
    if not result_path.is_file():
        raise AcceptanceError(f"{label} coordinator wrote no terminal receipt")
    return _read_json(result_path)


def _validate_cut_receipt(
    receipt: Mapping[str, Any],
    *,
    coordinator_pid: int,
    root_result: Path,
    token: str,
) -> dict[str, Any]:
    if receipt.get("schema") != "astrolabe.isolated-gpu-cut/v1":
        raise AcceptanceError("unknown cut receipt schema")
    if receipt.get("token") != token:
        raise AcceptanceError("cut receipt belongs to another harness invocation")
    if int(receipt.get("coordinator_pid", -1)) != coordinator_pid:
        raise AcceptanceError("cut receipt belongs to another coordinator")
    if Path(str(receipt.get("root_result", ""))).resolve() != root_result.resolve():
        raise AcceptanceError("cut receipt names another root result")
    if receipt.get("root_sha256") != _sha256(root_result):
        raise AcceptanceError("root result changed after the cut receipt")
    worker = receipt.get("worker")
    if not isinstance(worker, Mapping) or not _same_process(worker):
        raise AcceptanceError("cut worker process identity is no longer valid")
    if int(worker.get("process_group", -1)) != coordinator_pid:
        raise AcceptanceError("cut worker is outside the harness coordinator group")
    record = _read_json(root_result)
    if record.get("id") != 0 or record.get("status") != "done":
        raise AcceptanceError("root result is not a completed native root")
    return record


def _terminate_owned_group(coordinator_pid: int, expected_start_ticks: str) -> bool:
    current = _process_identity(coordinator_pid)
    if not current or current["start_ticks"] != str(expected_start_ticks):
        return False
    if current["process_group"] != coordinator_pid:
        return False
    try:
        os.killpg(coordinator_pid, signal.SIGTERM)
        os.killpg(coordinator_pid, signal.SIGCONT)
    except ProcessLookupError:
        return True
    return True


def _terminate_recorded_group(identity: Mapping[str, Any]) -> bool:
    members = _owned_group_members(identity)
    if not members:
        return False
    try:
        os.killpg(int(identity["process_group"]), signal.SIGTERM)
        os.killpg(int(identity["process_group"]), signal.SIGCONT)
    except ProcessLookupError:
        return False
    return True


def _cleanup_owned_process_group(
    process: subprocess.Popen[bytes],
    identity: Mapping[str, Any] | None,
) -> None:
    if process.poll() is not None and not (
        identity and _owned_group_members(identity)
    ):
        return
    verified = bool(
        identity
        and identity.get("process_group") == process.pid
        and identity.get("session") == process.pid
        and identity.get("boot_id") == _boot_id()
    )
    if verified:
        if process.poll() is None and _same_process(identity):
            _terminate_owned_group(process.pid, str(identity["start_ticks"]))
        else:
            _terminate_recorded_group(identity)
    else:
        process.kill()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        if verified:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        process.wait(timeout=30)
    if verified:
        deadline = time.monotonic() + 5
        while _owned_group_members(identity) and time.monotonic() < deadline:
            time.sleep(0.05)
        members = _owned_group_members(identity)
        if members:
            try:
                os.killpg(int(identity["process_group"]), signal.SIGKILL)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + 5
            while _owned_group_members(identity) and time.monotonic() < deadline:
                time.sleep(0.05)


def _wait_for_cut(
    process: subprocess.Popen[bytes],
    *,
    receipt: Path,
    timeout_s: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        returncode = process.poll()
        if returncode is not None:
            raise AcceptanceError(
                f"interrupted coordinator exited {returncode} before the durable cut"
            )
        if receipt.is_file():
            data = _read_json(receipt)
            if not isinstance(data, dict):
                raise AcceptanceError("cut receipt is not a JSON object")
            return data
        time.sleep(0.05)
    raise AcceptanceError("timed out waiting for the durable root cut receipt")


def _owned_service_identities(service_root: Path) -> list[dict[str, Any]]:
    identities = []
    for lifecycle in sorted((service_root / "services").glob("*/lifecycle.json")):
        state = _read_json(lifecycle)
        pid = state.get("pid")
        start_ticks = state.get("pid_start_ticks", state.get("start_ticks"))
        if not pid or not start_ticks or not state.get("boot_id"):
            continue
        identities.append(
            {
                "pid": int(pid),
                "start_ticks": str(start_ticks),
                "boot_id": str(state["boot_id"]),
                "process_group": int(pid),
                "session": int(pid),
                "lifecycle": str(lifecycle),
                "status": state.get("status"),
                "ended_at": state.get("ended_at"),
                "exit_code": state.get("exit_code"),
            }
        )
    return identities


def _service_evidence(service_root: Path) -> list[dict[str, Any]]:
    identities = _owned_service_identities(service_root)
    if not identities:
        raise AcceptanceError("AF3 service sibling has no verifiable lifecycle")
    evidence = []
    for identity in identities:
        members = _owned_group_members(identity)
        live = _same_process(identity)
        if not live and not members:
            raise AcceptanceError("AF3 service ended before the durable cut")
        evidence.append({**identity, "process_group_members": members})
    return evidence


def _terminate_owned_services(service_root: Path) -> None:
    for identity in _owned_service_identities(service_root):
        if _owned_group_members(identity):
            _terminate_recorded_group(identity)


def _wait_owned_services_ended(service_root: Path, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        active = []
        for identity in _owned_service_identities(service_root):
            if _same_process(identity) or _owned_group_members(identity):
                active.append(identity["lifecycle"])
        if not active:
            return
        if time.monotonic() >= deadline:
            raise AcceptanceError(
                "owned AF3 process did not end after coordinator termination: "
                + ", ".join(active)
            )
        time.sleep(0.1)


def _run_interrupted(
    prepared: PreparedAcceptance,
    *,
    timeout_s: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = prepared.out_root / "cut" / "root.receipt.json"
    token = uuid.uuid4().hex
    root_result = (prepared.recovery_output / ROOT_CANDIDATE).resolve()
    if prepared.recovery_service_root is None:
        raise AcceptanceError("Recovery service root is not prepared")
    stdout_log = prepared.out_root / "logs" / "recovery-first.stdout.log"
    stderr_log = prepared.out_root / "logs" / "recovery-first.stderr.log"
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    with stdout_log.open("wb") as stdout, stderr_log.open("wb") as stderr:
        process = None
        identity = None
        try:
            process = subprocess.Popen(
                _coordinator_command(
                    python_bin=Path(sys.executable),
                    config=prepared.recovery_config,
                    output=prepared.recovery_output,
                    service_root=prepared.recovery_service_root,
                    resume=False,
                    source_fingerprint=prepared.source_fingerprint,
                    receipt=receipt,
                ),
                cwd=str(REPO_ROOT),
                env=_coordinator_environment(
                    prepared,
                    receipt=receipt,
                    token=token,
                ),
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
                close_fds=True,
            )
            identity = _process_identity(process.pid)
            if identity is None or identity["process_group"] != process.pid:
                raise AcceptanceError(
                    "could not establish the coordinator process group"
                )
            identity_path = prepared.out_root / "cut" / "coordinator.receipt.json"
            _atomic_json(
                identity_path,
                {
                    "schema": "astrolabe.isolated-gpu-coordinator/v1",
                    "token": token,
                    "pid": process.pid,
                    "start_ticks": identity["start_ticks"],
                    "process_group": identity["process_group"],
                    "session": identity["session"],
                    "boot_id": _boot_id(),
                },
            )
            cut = _wait_for_cut(process, receipt=receipt, timeout_s=timeout_s)
            _validate_cut_receipt(
                cut,
                coordinator_pid=process.pid,
                root_result=root_result,
                token=token,
            )
            service_evidence = _service_evidence(prepared.recovery_service_root)
            _atomic_json(
                prepared.out_root / "cut" / "root.cut.json",
                {
                    "status": "root-durable-before-termination",
                    "root_result": str(root_result),
                    "root_sha256": _sha256(root_result),
                    "coordinator": identity,
                    "worker": cut["worker"],
                    "services": service_evidence,
                    "receipt": str(receipt),
                },
            )
            if not _terminate_owned_group(process.pid, identity["start_ticks"]):
                raise AcceptanceError(
                    "coordinator identity changed before controlled termination"
                )
            try:
                returncode = process.wait(timeout=30)
            except subprocess.TimeoutExpired as error:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=30)
                raise AcceptanceError(
                    "interrupted coordinator did not terminate"
                ) from error
        finally:
            if process is not None:
                _cleanup_owned_process_group(process, identity)
            if prepared.recovery_service_root is not None:
                _terminate_owned_services(prepared.recovery_service_root)
                _wait_owned_services_ended(
                    prepared.recovery_service_root,
                    timeout_s=30,
                )
    if returncode not in {-signal.SIGTERM, 128 + signal.SIGTERM}:
        raise AcceptanceError(
            f"interrupted coordinator exited {returncode} instead of controlled SIGTERM"
        )
    return cut, _read_json(root_result)


def _flatten_numbers(value: Any, prefix: str = "") -> dict[str, str]:
    result: dict[str, str] = {}
    if isinstance(value, Mapping):
        for key in sorted(value, key=str):
            result.update(_flatten_numbers(value[key], f"{prefix}.{key}".lstrip(".")))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            result.update(_flatten_numbers(item, f"{prefix}[{index}]"))
    elif type(value) in {int, float}:
        result[prefix] = float(value).hex()
    return result


_SCORE_ARRAY_FIELDS = (
    "plddt",
    "per_proto_scores",
    "per_proto_score_terms",
    "ptm",
    "pae",
    "pde",
    "iptm",
    "chain_pair_scores",
)


def _raw_score_arrays(value: Any, prefix: str = "") -> dict[str, Any]:
    import numpy as np

    arrays: dict[str, Any] = {}
    if value is None:
        return arrays
    if isinstance(value, Mapping):
        for key in sorted(value, key=lambda item: str(item)):
            child = f"{prefix}.{key}".lstrip(".")
            arrays.update(_raw_score_arrays(value[key], child))
        return arrays
    if isinstance(value, (list, tuple)):
        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            array = np.asarray(value, dtype=object)
        if array.dtype.kind in "biufc":
            arrays[prefix] = np.array(array, copy=True, order="C")
            return arrays
        for index, item in enumerate(value):
            arrays.update(_raw_score_arrays(item, f"{prefix}[{index}]"))
        return arrays
    array = np.asarray(value)
    if array.dtype.kind not in "biufc":
        raise ValueError(f"score snapshot field {prefix!r} is not numeric")
    arrays[prefix] = np.array(array, copy=True, order="C")
    return arrays


def _score_array_evidence(snapshot_path: Path) -> dict[str, Any]:
    import numpy as np

    snapshot_path = snapshot_path.expanduser().resolve()
    with snapshot_path.open("rb") as handle:
        scores = pickle.load(handle)
    arrays: dict[str, Any] = {}
    for field in _SCORE_ARRAY_FIELDS:
        arrays.update(_raw_score_arrays(getattr(scores, field, None), field))
    manifest = {}
    for name, value in sorted(arrays.items()):
        array = np.ascontiguousarray(value)
        digest = hashlib.sha256()
        digest.update(array.dtype.str.encode())
        digest.update(json.dumps(list(array.shape)).encode())
        digest.update(array.tobytes(order="C"))
        manifest[name] = {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "sha256": digest.hexdigest(),
        }
    aggregate = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "snapshot": snapshot_path.name,
        "array_count": len(manifest),
        "aggregate_sha256": aggregate,
        "arrays": manifest,
    }


def _inference_event_projection(service_root: Path) -> dict[str, Any]:
    services = []
    for directory in sorted((service_root / "services").glob("*")):
        lifecycle = directory / "lifecycle.json"
        if not lifecycle.is_file():
            continue
        state = _read_json(lifecycle)
        services.append(
            (
                float(state.get("started_at", 0)),
                str(directory.name),
                directory / "owner.jsonl",
            )
        )
    requests: list[dict[str, Any]] = []
    counts = {
        "request_received": 0,
        "gpu_start": 0,
        "gpu_end": 0,
        "gpu_ok": 0,
    }
    for _started, _service_name, log_path in sorted(services):
        if not log_path.is_file():
            continue
        owner_requests: dict[int, dict[str, Any]] = {}
        for line_number, line in enumerate(
            log_path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except ValueError as error:
                raise AcceptanceError(
                    f"invalid AF3 owner event at {log_path}:{line_number}"
                ) from error
            kind = event.get("event")
            if kind not in counts:
                continue
            counts[kind] += 1
            request_id = int(event["request_id"])
            record = owner_requests.setdefault(
                request_id,
                {
                    "request_received": 0,
                    "gpu_start": 0,
                    "gpu_end": 0,
                    "response_ok": None,
                },
            )
            record[kind] += 1
            if kind == "gpu_end":
                record["response_ok"] = bool(event.get("ok"))
                if record["response_ok"]:
                    counts["gpu_ok"] += 1
        requests.extend(
            owner_requests[request_id]
            for request_id in sorted(owner_requests)
        )
    response_complete = sum(item["gpu_end"] for item in requests)
    request_without_response = sum(
        item["request_received"] > 0 and item["gpu_end"] == 0 for item in requests
    )
    return {
        "source": "AF3 owned-service owner.jsonl",
        "response_marker": "gpu_end",
        "response_sent_marker_available": False,
        "counts": counts,
        "request_received_count": counts["request_received"],
        "response_complete_count": response_complete,
        "requests_without_response": request_without_response,
        "requests": requests,
    }


def _parse_cif_line(line: str) -> list[str]:
    values: list[str] = []
    current: list[str] = []
    quote: str | None = None
    for character in line.strip():
        if quote is not None:
            if character == quote:
                quote = None
            else:
                current.append(character)
        elif character in {"'", '"'}:
            quote = character
        elif character.isspace():
            if current:
                values.append("".join(current))
                current = []
        else:
            current.append(character)
    if current:
        values.append("".join(current))
    return values


def _structure_coordinates(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".pdb":
        coordinates, labels = [], {"chain": [], "residue": [], "atom": []}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.startswith(("ATOM", "HETATM")):
                continue
            coordinates.append(
                [
                    float(line[30:38]).hex(),
                    float(line[38:46]).hex(),
                    float(line[46:54]).hex(),
                ]
            )
            labels["atom"].append(line[12:16].strip())
            labels["residue"].append(line[17:20].strip())
            labels["chain"].append(line[21:22].strip())
        return {"coordinates": coordinates, "labels": labels}
    if suffix not in {".cif", ".mmcif"}:
        raise AcceptanceError(f"unsupported structure evidence: {path}")
    headers: list[str] = []
    rows: list[list[str]] = []
    in_atoms = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("_atom_site."):
            headers.append(line[len("_atom_site.") :])
            in_atoms = True
        elif in_atoms and not line:
            continue
        elif in_atoms and line.startswith("#"):
            break
        elif in_atoms and not line.startswith(("loop_", "_")):
            values = _parse_cif_line(line)
            if len(values) == len(headers):
                rows.append(values)
    index = {name: position for position, name in enumerate(headers)}
    required = ("Cartn_x", "Cartn_y", "Cartn_z")
    if not rows or any(name not in index for name in required):
        raise AcceptanceError(f"no exact atom coordinates in {path}")
    label_columns = {
        "chain": "auth_asym_id",
        "residue": "label_comp_id",
        "sequence_id": "label_seq_id",
        "atom": "label_atom_id",
    }
    return {
        "coordinates": [
            [
                float(row[index["Cartn_x"]]).hex(),
                float(row[index["Cartn_y"]]).hex(),
                float(row[index["Cartn_z"]]).hex(),
            ]
            for row in rows
        ],
        "labels": {
            key: [row[index[column]] for row in rows]
            for key, column in label_columns.items()
            if column in index
        },
    }


def _native_snapshot(
    output: Path,
    config_path: Path,
    service_root: Path | None = None,
) -> dict[str, Any]:
    from omegaconf import OmegaConf

    output = output.expanduser().resolve()
    candidates = []
    for result_path in sorted((output / "candidates").glob("[0-9][0-9][0-9][0-9][0-9][0-9]/result.json")):
        record = _read_json(result_path)
        if record.get("native_artifacts"):
            from astrolabe.protein_beam_contract import verify_native_artifacts

            verify_native_artifacts(record["native_artifacts"])
        structure = Path(record["structure_path"]).expanduser().resolve()
        score_arrays = _score_array_evidence(Path(record["snapshot"]))
        candidates.append(
            {
                "id": record["id"],
                "sequence": record["sequence"],
                "status": record["status"],
                "evaluation_origin": record.get("evaluation_origin"),
                "eligible": record.get("eligible"),
                "numeric": _flatten_numbers(
                    {
                        "score": record["score"],
                        "metrics": record["metrics"],
                        "geometry": record.get("geometry"),
                    }
                ),
                "structure": _structure_coordinates(structure),
                "score_arrays": score_arrays,
            }
        )
    events = _inference_event_projection(service_root or output)
    record_fresh_calls = sum(
        item["evaluation_origin"] == "fresh_af3" for item in candidates
    )
    ledger_path = output / "af3_calls.json"
    ledger = _read_json(ledger_path) if ledger_path.is_file() else {"entries": {}}
    entries = [
        {
            "token": token,
            "candidate_id": entry["candidate_id"],
            "attempt": entry["attempt"],
            "seed": entry["seed"],
            "status": entry["status"],
        }
        for token, entry in sorted(ledger.get("entries", {}).items())
    ]
    cfg = OmegaConf.load(config_path)
    return {
        "seed": int(cfg.seed),
        "source_fingerprint": _source_fingerprint(_source_manifest()),
        "budget": int(cfg.protein_beam.max_evaluations),
        "quota_contract": ledger.get("contract"),
        "candidates": candidates,
        "native_call_count": events["request_received_count"],
        "record_fresh_call_count": record_fresh_calls,
        "inference_events": events,
        "quota_entries": entries,
    }


def _scientific_config_projection(config_path: Path) -> dict[str, Any]:
    from omegaconf import OmegaConf

    data = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    data.pop("out_dir", None)
    data.pop("raw_model_results_dir", None)
    quota = (data.get("protein_beam") or {}).get("af3_call_budget")
    if isinstance(quota, dict):
        quota.pop("ledger", None)
    return data


def compare_runs(
    baseline: Mapping[str, Any],
    recovery: Mapping[str, Any],
    *,
    baseline_config: Path,
    recovery_config: Path,
) -> dict[str, Any]:
    issues: list[str] = []
    for key in ("seed", "source_fingerprint", "budget", "quota_contract"):
        if baseline.get(key) != recovery.get(key):
            issues.append(f"{key} differs")
    if baseline.get("native_call_count") != recovery.get("native_call_count"):
        issues.append("native call count differs")
    calls = recovery.get("native_call_count")
    if calls != EVALUATION_BUDGET:
        issues.append("native call count is not exactly root plus one child")
    for key in ("quota_entries", "inference_events"):
        if baseline.get(key) != recovery.get(key):
            issues.append(f"{key} differ")
    def candidate_projection(snapshot: Mapping[str, Any], key: str) -> Any:
        return [
            {
                "id": item["id"],
                key: item[key],
            }
            for item in snapshot.get("candidates") or []
        ]

    for key, issue in (
        ("sequence", "sequences differ"),
        ("status", "candidate status differs"),
        ("evaluation_origin", "candidate provenance differs"),
        ("eligible", "candidate acceptance eligibility differs"),
        ("numeric", "candidate numeric metrics differ"),
        ("structure", "candidate coordinates differ"),
        ("score_arrays", "raw score snapshot arrays differ"),
    ):
        if candidate_projection(baseline, key) != candidate_projection(recovery, key):
            issues.append(issue)
    if recovery.get("record_fresh_call_count") != calls:
        issues.append("event requests differ from terminal fresh-call records")
    if (
        recovery.get("inference_events", {}).get("response_complete_count")
        != calls
    ):
        issues.append("event requests lack one AF3 response completion each")
    if _scientific_config_projection(baseline_config) != _scientific_config_projection(recovery_config):
        issues.append("scientific config differs")
    quota = recovery.get("quota_contract") or {}
    if quota != {"max_total": EVALUATION_BUDGET, "max_per_case": EVALUATION_BUDGET}:
        issues.append("AF3 quota is not synchronized to 2")
    return {
        "status": "PASS" if not issues else "ISSUES",
        "exact_metrics": "candidate numeric metrics differ" not in issues,
        "exact_coordinates": "candidate coordinates differ" not in issues,
        "exact_raw_score_arrays": "raw score snapshot arrays differ" not in issues,
        "exact_inference_events": not any(
            item in issues for item in ("inference_events differ",)
        ),
        "native_call_count": calls,
        "candidate_count": len(recovery.get("candidates") or []),
        "issues": issues,
    }


def _record_failure(out_root: Path, error: BaseException) -> None:
    _atomic_json(
        out_root / "summary.json",
        {
            "schema": f"{SCHEMA}.summary",
            "status": "ISSUES",
            "error": repr(error),
            "automatic_retry": False,
        },
    )


def _gpu_manifest() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,driver_version,memory.total",
        "--format=csv,noheader",
    ]
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise AcceptanceError(
            f"nvidia-smi failed before model execution: {completed.stderr.strip()}"
        )
    return {
        "command": command,
        "stdout": completed.stdout.strip(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
    }


def _execute_acceptance(
    prepared: PreparedAcceptance,
    *,
    timeout_s: float,
) -> dict[str, Any]:
    _assert_full_run_ready(prepared)
    if prepared.baseline_service_root is None or prepared.recovery_service_root is None:
        raise AcceptanceError("Acceptance service roots are not prepared")
    _atomic_json(
        prepared.out_root / "environment.json",
        {
            "schema": f"{SCHEMA}.environment",
            "python": sys.executable,
            "python_version": sys.version,
            "gpu": _gpu_manifest(),
        },
    )
    preflight = {
        "baseline": _scientific_preflight(prepared.baseline_config),
        "recovery": _scientific_preflight(prepared.recovery_config),
    }
    _atomic_json(prepared.out_root / "scientific_preflight.json", preflight)
    baseline_coordinator = _run_coordinator(
        prepared,
        config=prepared.baseline_config,
        output=prepared.baseline_output,
        service_root=prepared.baseline_service_root,
        resume=False,
        label="baseline",
        timeout_s=timeout_s,
    )
    cut, root_record = _run_interrupted(prepared, timeout_s=timeout_s)
    recovered_coordinator = _run_coordinator(
        prepared,
        config=prepared.recovery_config,
        output=prepared.recovery_output,
        service_root=prepared.recovery_service_root,
        resume=True,
        label="recovery-resume",
        timeout_s=timeout_s,
    )
    baseline_snapshot = _native_snapshot(
        prepared.baseline_output,
        prepared.baseline_config,
        prepared.baseline_service_root,
    )
    recovery_snapshot = _native_snapshot(
        prepared.recovery_output,
        prepared.recovery_config,
        prepared.recovery_service_root,
    )
    comparison = compare_runs(
        baseline_snapshot,
        recovery_snapshot,
        baseline_config=prepared.baseline_config,
        recovery_config=prepared.recovery_config,
    )
    _atomic_json(
        prepared.out_root / "baseline_snapshot.json",
        baseline_snapshot,
    )
    _atomic_json(
        prepared.out_root / "recovery_snapshot.json",
        recovery_snapshot,
    )
    _atomic_json(
        prepared.out_root / "comparison.json",
        comparison,
    )
    summary = {
        "schema": f"{SCHEMA}.summary",
        "status": comparison["status"],
        "source_fingerprint": prepared.source_fingerprint,
        "seed": int(recovery_snapshot["seed"]),
        "evaluation_budget_including_root": EVALUATION_BUDGET,
        "automatic_retry": False,
        "baseline": baseline_coordinator,
        "service_roots": {
            "baseline": str(prepared.baseline_service_root),
            "recovery": str(prepared.recovery_service_root),
        },
        "durable_root_cut": {
            "record": root_record,
            "receipt": cut,
        },
        "recovery": recovered_coordinator,
        "comparison": comparison,
    }
    _atomic_json(prepared.out_root / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the D5056_D15-derived isolated interface acceptance with a "
            "two-native-call budget, one AF3 owner, and a durable root cut."
        )
    )
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--case-config", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=10800.0)
    parser.add_argument("--coordinator", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--config", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--service-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--resume", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--source-fingerprint", help=argparse.SUPPRESS)
    parser.add_argument("--cut-receipt", type=Path, help=argparse.SUPPRESS)
    arguments = parser.parse_args(argv)

    if arguments.coordinator:
        if not all(
            (
                arguments.config,
                arguments.output,
                arguments.service_root,
                arguments.source_fingerprint,
            )
        ):
            parser.error(
                "internal coordinator requires --config, --output, "
                "--service-root and --source-fingerprint"
            )
        return _coordinator_main(arguments)

    if arguments.out_root is None or arguments.case_config is None:
        parser.error("--out-root and --case-config are required")
    if not math.isfinite(arguments.timeout_s) or arguments.timeout_s <= 0:
        parser.error("--timeout-s must be finite and positive")
    prepared = _prepare_acceptance(arguments.case_config, arguments.out_root)
    if arguments.prepare_only:
        print(json.dumps({"status": "prepared", "plan": str(prepared.out_root / "plan.json")}))
        return 0
    try:
        summary = _execute_acceptance(
            prepared,
            timeout_s=arguments.timeout_s,
        )
    except BaseException as error:
        try:
            _record_failure(prepared.out_root, error)
        except BaseException as evidence_error:
            error.add_note(f"Acceptance failure evidence could not be written: {evidence_error!r}")
        raise
    print(
        json.dumps(
            {
                "status": summary["status"],
                "summary": str(prepared.out_root / "summary.json"),
            }
        )
    )
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
