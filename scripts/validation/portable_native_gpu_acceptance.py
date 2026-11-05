#!/usr/bin/env python3
"""Bounded real-GPU acceptance for the public portable-native frontend.

The harness reuses the evaluator capture, evidence projection, and comparison
helpers from ``unified_gpu_acceptance.py``.  Public execution enters through
``astrolabe.app.Astrolabe.run`` and therefore exercises the real
``run._run_pipeline`` recovery route.  The cut injector only wraps the actual
``Runtime.call`` method to terminate after a named receipt is durable.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
UNIFIED_PATH = Path(__file__).with_name("unified_gpu_acceptance.py")
DEFAULT_PROFILE = REPO_ROOT / "config" / "container_af3_smoke.yaml"
SUPPORTED_CUTS = ("after-raw-prediction",)
RAW_PREDICTION_CUT = "after-raw-prediction"
CUT_BOUNDARY = {"after-raw-prediction": "root/raw_predict"}
CUT_EXIT = 97
NATIVE_REFERENCE_REQUESTS = 2
NORMAL_PUBLIC_REQUESTS = 2
BASE_REQUESTS = NATIVE_REFERENCE_REQUESTS + NORMAL_PUBLIC_REQUESTS
REQUESTS_PER_CUT = 2
MAX_REQUESTS = 8
NO_RETRY = True


class AcceptanceError(RuntimeError):
    """The portable-native acceptance run cannot produce trustworthy evidence."""


@dataclass(frozen=True)
class AcceptanceCase:
    name: str
    mode: str
    execution: str
    recovery_resume: bool | None
    cut: str | None
    reference: str
    frontdoor: str
    fixture: str = "single-20aa"


def _load_unified():
    module_name = "portable_native_unified_gpu_acceptance_helpers"
    loaded = sys.modules.get(module_name)
    if loaded is not None:
        return loaded
    spec = importlib.util.spec_from_file_location(module_name, UNIFIED_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load unified GPU acceptance helpers: {UNIFIED_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def acceptance_matrix(
    cuts: Sequence[str],
    *,
    include_normal_public: bool = True,
) -> tuple[AcceptanceCase, ...]:
    selected = tuple(cuts)
    unknown = set(selected) - set(SUPPORTED_CUTS)
    if unknown:
        raise ValueError(f"unsupported cuts: {sorted(unknown)!r}")
    if len(set(selected)) != len(selected):
        raise ValueError("cut values must not repeat")
    matrix = [
        AcceptanceCase(
            name="native-in-process",
            mode="native-reference",
            execution="in_process",
            recovery_resume=None,
            cut=None,
            reference="none",
            frontdoor="run.py _initialize_optimization_state + _advance_optimization_state",
        )
    ]
    if include_normal_public:
        matrix.append(
            AcceptanceCase(
                name="public-api-durable",
                mode="public-api",
                execution="daemon",
                recovery_resume=False,
                cut=None,
                reference="native-in-process",
                frontdoor="astrolabe.app.Astrolabe.run -> run._run_pipeline",
            )
        )
    for cut in selected:
        matrix.append(
            AcceptanceCase(
                name=f"public-api-{cut}",
                mode="public-api-cut",
                execution="daemon",
                recovery_resume=False,
                cut=cut,
                reference="native-in-process",
                frontdoor="astrolabe.app.Astrolabe.run -> run._run_pipeline",
            )
        )
    return tuple(matrix)


def request_budget(
    cuts: Sequence[str],
    *,
    include_normal_public: bool = True,
) -> int:
    budget = NATIVE_REFERENCE_REQUESTS + REQUESTS_PER_CUT * len(tuple(cuts))
    if include_normal_public:
        budget += NORMAL_PUBLIC_REQUESTS
    if budget > MAX_REQUESTS:
        raise ValueError(
            f"selected cuts require {budget} model requests; hard cap is {MAX_REQUESTS}"
        )
    return budget


def _reject_existing_output(out_root: Path) -> None:
    if out_root.exists() and any(out_root.iterdir()):
        raise AcceptanceError(
            f"--out-root must be absent or empty; refusing to overwrite {out_root}"
        )
    out_root.mkdir(parents=True, exist_ok=True)
    try:
        out_root.chmod(0o700)
    except OSError:
        pass


def _case_config(
    *,
    profile: Path,
    case: AcceptanceCase,
    out_dir: Path,
):
    from omegaconf import open_dict

    unified = _load_unified()
    cfg = unified._build_config(
        profile=profile,
        fixture=case.fixture,
        out_dir=out_dir,
        task_name=f"portable_acceptance_{case.name}",
        execution=case.execution,
        generic_search=False,
        seed=unified.SEED,
    )
    with open_dict(cfg):
        if case.recovery_resume is None:
            cfg.pop("recovery", None)
        else:
            cfg.recovery = {
                "mode": "durable",
                "resume": case.recovery_resume,
            }
    return cfg


def _write_config(config: Any, path: Path) -> None:
    unified = _load_unified()
    unified._save_standalone(config, path)


def _worker_command(
    *,
    python_bin: Path,
    mode: str,
    config_path: Path,
    cut: str | None,
) -> list[str]:
    command = [
        str(python_bin),
        str(Path(__file__).resolve()),
        "--internal-worker",
        "--mode",
        mode,
        "--config",
        str(config_path),
    ]
    if cut is not None:
        command.extend(("--cut", cut))
    return command


def _write_cut_marker(path: Path, *, boundary: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "astrolabe.portable-native.cut-marker/v1",
        "boundary": boundary,
        "pid": os.getpid(),
    }
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        encoded = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _install_runtime_cut(boundary: str, marker: Path):
    from astrolabe.design.runtime import Runtime

    if boundary != CUT_BOUNDARY[RAW_PREDICTION_CUT]:
        raise ValueError(f"unsupported runtime cut boundary {boundary!r}")
    original = Runtime.call
    fired = False

    def cut_after_return(
        self,
        named_round,
        operation,
        request,
        *,
        work_id=None,
    ):
        nonlocal fired
        result = original(
            self,
            named_round,
            operation,
            request,
            work_id=work_id,
        )
        if not fired and named_round == boundary:
            fired = True
            _write_cut_marker(marker, boundary=boundary)
            os._exit(CUT_EXIT)
        return result

    Runtime.call = cut_after_return
    return original


def _run_persisted_run(run_root: Path) -> Path:
    candidates = sorted(
        path.parent.resolve()
        for path in run_root.rglob("config.yaml")
        if path.is_file()
    )
    if len(candidates) != 1:
        raise AcceptanceError(
            f"cut run must contain exactly one persisted config.yaml, found {candidates}"
        )
    persisted = candidates[0]
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(persisted / "config.yaml")
    recorded = Path(str(cfg.get("out_dir", ""))).expanduser().resolve()
    if recorded != persisted:
        raise AcceptanceError(
            "persisted config.out_dir does not identify its actual run root: "
            f"recorded={recorded} actual={persisted}"
        )
    return persisted


def _prepare_resume_config(persisted_root: Path, destination: Path) -> Any:
    from omegaconf import OmegaConf, open_dict

    persisted_path = persisted_root / "config.yaml"
    before = persisted_path.read_bytes()
    cfg = OmegaConf.load(persisted_path)
    recorded = Path(str(cfg.get("out_dir", ""))).expanduser().resolve()
    if recorded != persisted_root.resolve():
        raise AcceptanceError("resume must use the exact parsed run root")
    recovery = cfg.get("recovery")
    if recovery is None or str(recovery.get("mode")) != "durable":
        raise AcceptanceError("persisted config has no durable recovery contract")
    if recovery.get("resume") is not False:
        raise AcceptanceError("persisted initial config must have recovery.resume=false")
    with open_dict(cfg):
        cfg.out_dir = str(persisted_root.resolve())
        cfg.recovery.resume = True
    OmegaConf.save(cfg, destination)
    if persisted_path.read_bytes() != before:
        raise AcceptanceError("preparing resume config modified the failed run config")
    return OmegaConf.load(destination)


def _snapshot_failed_metadata(persisted_root: Path, destination: Path) -> dict[str, Any]:
    from omegaconf import OmegaConf

    destination.mkdir(parents=True, exist_ok=False)
    names = ("config.yaml", "provenance.json")
    unified = _load_unified()
    fingerprints = {}
    for name in names:
        source = persisted_root / name
        if not source.is_file():
            raise AcceptanceError(f"failed run is missing protected file {source}")
        shutil.copy2(source, destination / name)
        fingerprints[name] = unified._sha256(source)
    persisted = OmegaConf.load(persisted_root / "config.yaml")
    payload = {
        "schema": "astrolabe.portable-native.failed-metadata/v1",
        "run_root": str(persisted_root),
        "persisted_out_dir": str(persisted.get("out_dir")),
        "files": fingerprints,
    }
    unified._atomic_json(destination / "manifest.json", payload)
    return payload


def validate_raw_cut_accounting(
    cut_records: Sequence[Mapping[str, Any]],
    resume_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(cut_records) != 1:
        raise AcceptanceError(
            f"cut process must issue exactly one model request, observed {len(cut_records)}"
        )
    if len(resume_records) != 1:
        raise AcceptanceError(
            f"resume process must issue exactly one remaining model request, observed {len(resume_records)}"
        )
    cut_steps = [int(record["step"]) for record in cut_records]
    resume_steps = [int(record["step"]) for record in resume_records]
    if cut_steps != [0]:
        raise AcceptanceError(f"cut process issued unexpected steps {cut_steps}")
    if resume_steps != [1]:
        raise AcceptanceError(f"resume issued duplicate or unexpected steps {resume_steps}")
    duplicates = sorted(set(cut_steps) & set(resume_steps))
    if duplicates:
        raise AcceptanceError(f"resume replayed durable model steps {duplicates}")
    return {
        "cut_model_requests": len(cut_records),
        "resume_model_requests": len(resume_records),
        "remaining_model_requests": 1,
        "duplicate_model_requests": 0,
        "cut_steps": cut_steps,
        "resume_steps": resume_steps,
        "total_case_model_requests": len(cut_records) + len(resume_records),
    }


def _worker_main(
    *,
    mode: str,
    config_path: Path,
    cut: str | None = None,
) -> int:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(config_path)
    unified = _load_unified()
    if mode == "native-reference":
        unified._run_native_reference(cfg)
        return 0
    if mode not in {"public-api", "public-api-cut"}:
        raise ValueError(f"unsupported worker mode {mode!r}")
    if mode == "public-api-cut":
        if cut not in CUT_BOUNDARY:
            raise ValueError("public-api-cut requires a supported --cut")
        marker = os.environ.get("ASTROLABE_PORTABLE_CUT_MARKER")
        if not marker:
            raise RuntimeError("public-api-cut requires ASTROLABE_PORTABLE_CUT_MARKER")
        _install_runtime_cut(CUT_BOUNDARY[cut], Path(marker))
    from astrolabe.app import Astrolabe

    Astrolabe(cfg).run()
    return 0


def _plan_for(case: AcceptanceCase):
    unified = _load_unified()
    return unified.CasePlan(
        name=case.name,
        reference=case.reference,
        frontdoor=case.frontdoor,
        fixture=case.fixture,
    )


def _run_simple_case(
    *,
    case: AcceptanceCase,
    profile: Path,
    out_root: Path,
    environment: Mapping[str, Any],
    python_bin: Path,
    timeout_s: float,
) -> tuple[dict[str, Any], Any, Path]:
    case_root = out_root / "cases" / case.name
    case_root.mkdir(parents=True, exist_ok=False)
    config = _case_config(
        profile=profile,
        case=case,
        out_dir=case_root / "output",
    )
    config_path = case_root / "input.yaml"
    _write_config(config, config_path)
    unified = _load_unified()
    capture, _claims, env = unified._install_capture_environment(case_root)
    command = _worker_command(
        python_bin=python_bin,
        mode=case.mode,
        config_path=config_path,
        cut=None,
    )
    process = unified._run_process(
        command,
        cwd=REPO_ROOT,
        env=env,
        stdout_log=case_root / "stdout.log",
        stderr_log=case_root / "stderr.log",
        timeout_s=timeout_s,
    )
    plan = _plan_for(case)
    try:
        result = unified._write_case_evidence(
            case_root=case_root,
            run_root=case_root / "output",
            process=process,
            capture_path=capture,
            env_manifest=environment,
            plan=plan,
            config=config,
        )
    except Exception as error:
        result = unified._write_evidence_failure(
            case_root=case_root,
            process=process,
            capture_path=capture,
            env_manifest=environment,
            plan=plan,
            config=config,
            error=error,
        )
    result["command"] = command
    unified._atomic_json(case_root / "result.json", result)
    return result, config, case_root


def _run_cut_case(
    *,
    case: AcceptanceCase,
    profile: Path,
    out_root: Path,
    environment: Mapping[str, Any],
    python_bin: Path,
    timeout_s: float,
) -> tuple[dict[str, Any], Any, Path]:
    if case.cut != RAW_PREDICTION_CUT:
        raise ValueError(f"unsupported acceptance cut {case.cut!r}")
    case_root = out_root / "cases" / case.name
    case_root.mkdir(parents=True, exist_ok=False)
    config = _case_config(
        profile=profile,
        case=case,
        out_dir=case_root / "output",
    )
    config_path = case_root / "input.yaml"
    _write_config(config, config_path)
    unified = _load_unified()
    capture, _claims, env = unified._install_capture_environment(case_root)
    marker = case_root / "cut-marker.json"
    cut_env = dict(env)
    cut_env["ASTROLABE_PORTABLE_CUT_MARKER"] = str(marker)
    cut_command = _worker_command(
        python_bin=python_bin,
        mode=case.mode,
        config_path=config_path,
        cut=case.cut,
    )
    cut_process = unified._run_process(
        cut_command,
        cwd=REPO_ROOT,
        env=cut_env,
        stdout_log=case_root / "cut.stdout.log",
        stderr_log=case_root / "cut.stderr.log",
        timeout_s=timeout_s,
    )
    if cut_process.timed_out or cut_process.returncode != CUT_EXIT:
        raise AcceptanceError(
            "cut process did not terminate at the declared boundary: "
            f"rc={cut_process.returncode} timeout={cut_process.timed_out}"
        )
    if not marker.is_file():
        raise AcceptanceError("cut process exited without writing its marker")
    cut_records = unified._read_capture_records(capture)
    if len(cut_records) != 1:
        raise AcceptanceError(
            f"cut process issued {len(cut_records)} model requests before the cut"
        )
    persisted_root = _run_persisted_run(case_root / "output")
    config_before = unified._sha256(persisted_root / "config.yaml")
    failed_metadata = _snapshot_failed_metadata(
        persisted_root,
        case_root / "cut_before",
    )
    resume_config_path = case_root / "resume-input.yaml"
    resume_config = _prepare_resume_config(persisted_root, resume_config_path)
    resume_command = _worker_command(
        python_bin=python_bin,
        mode="public-api",
        config_path=resume_config_path,
        cut=None,
    )
    resume_process = unified._run_process(
        resume_command,
        cwd=REPO_ROOT,
        env=env,
        stdout_log=case_root / "resume.stdout.log",
        stderr_log=case_root / "resume.stderr.log",
        timeout_s=timeout_s,
    )
    config_after = unified._sha256(persisted_root / "config.yaml")
    if config_after != config_before:
        raise AcceptanceError("resume modified the original failed run config.yaml")
    all_records = unified._read_capture_records(capture)
    resume_records = all_records[len(cut_records) :]
    accounting = validate_raw_cut_accounting(cut_records, resume_records)
    plan = _plan_for(case)
    try:
        result = unified._write_case_evidence(
            case_root=case_root,
            run_root=persisted_root,
            process=resume_process,
            capture_path=capture,
            env_manifest=environment,
            plan=plan,
            config=config,
        )
    except Exception as error:
        result = unified._write_evidence_failure(
            case_root=case_root,
            process=resume_process,
            capture_path=capture,
            env_manifest=environment,
            plan=plan,
            config=config,
            error=error,
        )
    result.update(
        {
            "cut": case.cut,
            "cut_boundary": CUT_BOUNDARY[case.cut],
            "cut_process": {
                "returncode": cut_process.returncode,
                "timed_out": cut_process.timed_out,
                "stdout_log": str(cut_process.stdout_log),
                "stderr_log": str(cut_process.stderr_log),
            },
            "resume_process": {
                "returncode": resume_process.returncode,
                "timed_out": resume_process.timed_out,
                "stdout_log": str(resume_process.stdout_log),
                "stderr_log": str(resume_process.stderr_log),
            },
            "resume_accounting": accounting,
            "resume_from": str(persisted_root),
            "resume_config": str(resume_config_path),
            "failed_metadata_snapshot": str(case_root / "cut_before"),
            "failed_metadata": failed_metadata,
            "failed_config_sha256_before": config_before,
            "failed_config_sha256_after": config_after,
            "no_retry": NO_RETRY,
            "commands": {
                "cut": cut_command,
                "resume": resume_command,
            },
        }
    )
    unified._atomic_json(case_root / "result.json", result)
    return result, config, case_root


def _dry_run(
    *,
    out_root: Path,
    profile: Path,
    cuts: Sequence[str],
    include_normal_public: bool,
) -> int:
    from omegaconf import OmegaConf

    matrix = acceptance_matrix(
        cuts,
        include_normal_public=include_normal_public,
    )
    rows = []
    for case in matrix:
        cfg = _case_config(
            profile=profile,
            case=case,
            out_dir=out_root / "dry-run" / case.name / "output",
        )
        rows.append(
            {
                "name": case.name,
                "mode": case.mode,
                "execution": case.execution,
                "cut": case.cut,
                "recovery": (
                    None
                    if cfg.get("recovery") is None
                    else OmegaConf.to_container(cfg.recovery, resolve=True)
                ),
                "frontdoor": case.frontdoor,
                "fixture": case.fixture,
            }
        )
    payload = {
        "schema": "astrolabe.portable-native.gpu-acceptance.plan/v1",
        "status": "dry-run",
        "request_budget": request_budget(
            cuts,
            include_normal_public=include_normal_public,
        ),
        "hard_request_cap": MAX_REQUESTS,
        "no_retry": NO_RETRY,
        "include_normal_public": include_normal_public,
        "cuts": list(cuts),
        "cases": rows,
    }
    _load_unified()._atomic_json(out_root / "plan.json", payload)
    print(json.dumps({"status": "dry-run", "plan": str(out_root / "plan.json")}))
    return 0


def run_acceptance(
    *,
    out_root: Path,
    profile: Path,
    python_bin: Path,
    timeout_s: float,
    cuts: Sequence[str],
    include_normal_public: bool = True,
) -> int:
    matrix = acceptance_matrix(
        cuts,
        include_normal_public=include_normal_public,
    )
    budget = request_budget(
        cuts,
        include_normal_public=include_normal_public,
    )
    _reject_existing_output(out_root)
    unified = _load_unified()
    source_manifest = unified._read_source_manifest()
    unified._atomic_json(out_root / "source_manifest.json", source_manifest.as_dict())
    baseline_case = matrix[0]
    baseline_cfg = _case_config(
        profile=profile,
        case=baseline_case,
        out_dir=out_root / "cases" / baseline_case.name / "output",
    )
    environment = unified._environment_manifest(baseline_cfg, source_manifest)
    unified._atomic_json(out_root / "environment_manifest.json", environment)
    unified._atomic_json(
        out_root / "plan.json",
        {
            "schema": "astrolabe.portable-native.gpu-acceptance.plan/v1",
            "status": "ready",
            "request_budget": budget,
            "hard_request_cap": MAX_REQUESTS,
            "no_retry": NO_RETRY,
            "include_normal_public": include_normal_public,
            "cuts": list(cuts),
            "cases": [case.name for case in matrix],
        },
    )

    native_result, _native_cfg, native_root = _run_simple_case(
        case=matrix[0],
        profile=profile,
        out_root=out_root,
        environment=environment,
        python_bin=python_bin,
        timeout_s=timeout_s,
    )
    if native_result.get("status") != "evidence-complete":
        raise AcceptanceError("native in_process reference did not complete")

    results: dict[str, dict[str, Any]] = {"native-in-process": native_result}
    for case in matrix[1:]:
        if case.cut is None:
            result, _cfg, _root = _run_simple_case(
                case=case,
                profile=profile,
                out_root=out_root,
                environment=environment,
                python_bin=python_bin,
                timeout_s=timeout_s,
            )
        else:
            result, _cfg, _root = _run_cut_case(
                case=case,
                profile=profile,
                out_root=out_root,
                environment=environment,
                python_bin=python_bin,
                timeout_s=timeout_s,
            )
        results[case.name] = result
        if result.get("status") != "evidence-complete":
            raise AcceptanceError(f"case {case.name} did not produce complete evidence")

    comparisons = {
        name: unified.compare_case(
            reference_root=native_root,
            case_root=out_root / "cases" / name,
        )
        for name in results
        if name != "native-in-process"
    }
    actual_requests = sum(int(result["model_calls"]) for result in results.values())
    exact_counts = all(int(result["model_calls"]) == 2 for result in results.values())
    cut_accounting = {
        name: result["resume_accounting"]
        for name, result in results.items()
        if "resume_accounting" in result
    }
    passed = (
        actual_requests == budget
        and actual_requests <= MAX_REQUESTS
        and exact_counts
        and all(item.get("status") == "PASS" for item in comparisons.values())
        and all(
            item["duplicate_model_requests"] == 0
            and item["remaining_model_requests"] == 1
            for item in cut_accounting.values()
        )
    )
    summary = {
        "schema": "astrolabe.portable-native.gpu-acceptance.summary/v1",
        "status": "PASS" if passed else "ISSUES",
        "request_budget": budget,
        "actual_true_gpu_requests": actual_requests,
        "hard_request_cap": MAX_REQUESTS,
        "exact_case_request_counts": exact_counts,
        "no_retry": NO_RETRY,
        "include_normal_public": include_normal_public,
        "cuts": list(cuts),
        "cases": {
            name: {
                "status": result.get("status"),
                "model_calls": result.get("model_calls"),
                "result": str(out_root / "cases" / name / "result.json"),
            }
            for name, result in results.items()
        },
        "comparisons": {
            name: {
                "status": item.get("status"),
                "path": str(out_root / "cases" / name / "comparison.json"),
            }
            for name, item in comparisons.items()
        },
        "resume_accounting": cut_accounting,
        "environment_manifest": str(out_root / "environment_manifest.json"),
        "source_manifest": str(out_root / "source_manifest.json"),
    }
    for name, comparison in comparisons.items():
        unified._atomic_json(
            out_root / "cases" / name / "comparison.json",
            comparison,
        )
    unified._atomic_json(out_root / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "summary": str(out_root / "summary.json")}))
    return 0 if passed else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run bounded public portable-native GPU acceptance with exact request and "
            "cross-process resume accounting."
        )
    )
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument(
        "--cut",
        action="append",
        choices=SUPPORTED_CUTS,
        default=None,
        help="Cut family to run; the initial implementation supports one raw-prediction cut.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-normal-public",
        action="store_true",
        help=(
            "Skip the separate non-cut public API case and run only the native "
            "reference plus raw-prediction cut/resume (exact four requests)."
        ),
    )
    parser.add_argument("--internal-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mode", help=argparse.SUPPRESS)
    parser.add_argument("--config", type=Path, help=argparse.SUPPRESS)
    arguments = parser.parse_args(argv)

    if arguments.internal_worker:
        if arguments.mode is None or arguments.config is None:
            parser.error("--internal-worker requires --mode and --config")
        cut = arguments.cut[0] if arguments.cut else None
        return _worker_main(
            mode=arguments.mode,
            config_path=arguments.config,
            cut=cut,
        )
    if arguments.out_root is None:
        parser.error("--out-root is required")
    cuts = tuple(arguments.cut or (RAW_PREDICTION_CUT,))
    out_root = arguments.out_root.expanduser().resolve()
    profile = arguments.profile.expanduser().resolve()
    python_bin = arguments.python_bin.expanduser().resolve()
    owned_empty_root = not out_root.exists() or (
        out_root.is_dir() and not any(out_root.iterdir())
    )
    try:
        if arguments.dry_run:
            _reject_existing_output(out_root)
            return _dry_run(
                out_root=out_root,
                profile=profile,
                cuts=cuts,
                include_normal_public=not arguments.skip_normal_public,
            )
        return run_acceptance(
            out_root=out_root,
            profile=profile,
            python_bin=python_bin,
            timeout_s=arguments.timeout_s,
            cuts=cuts,
            include_normal_public=not arguments.skip_normal_public,
        )
    except Exception as error:
        if owned_empty_root and out_root.is_dir():
            unified = _load_unified()
            unified._atomic_json(
                out_root / "summary.json",
                {
                    "schema": "astrolabe.portable-native.gpu-acceptance.summary/v1",
                    "status": "ISSUES",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "no_retry": NO_RETRY,
                },
            )
        print(
            json.dumps(
                {
                    "status": "ISSUES",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
        )
        return 1


__all__ = [
    "AcceptanceCase",
    "AcceptanceError",
    "CUT_EXIT",
    "MAX_REQUESTS",
    "RAW_PREDICTION_CUT",
    "acceptance_matrix",
    "request_budget",
    "run_acceptance",
    "validate_raw_cut_accounting",
]


if __name__ == "__main__":
    raise SystemExit(main())
