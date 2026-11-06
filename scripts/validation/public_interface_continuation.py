#!/usr/bin/env python3
"""Complete public-interface acceptance without replaying its successful root."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
FINAL_PATH = Path(__file__).with_name("public_interface_final.py")
SCHEMA = "astrolabe.public-interface-continuation/v1"
MAX_NEW_ACTUAL_REQUESTS = 1


def _load_final():
    name = "astrolabe_public_interface_continuation_final"
    loaded = sys.modules.get(name)
    if loaded is not None:
        return loaded
    spec = importlib.util.spec_from_file_location(name, FINAL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load public interface acceptance helpers")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


FINAL = _load_final()
SHARED = FINAL.SHARED


def _source_manifest() -> dict[str, str]:
    manifest = dict(FINAL._source_manifest())
    current = Path(__file__).resolve()
    manifest[current.relative_to(REPO_ROOT).as_posix()] = SHARED._sha256(current)
    return manifest


def _source_fingerprint(manifest: Mapping[str, str]) -> str:
    return hashlib.sha256(
        json.dumps(dict(manifest), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _resume_command(public_root: Path) -> list[str]:
    plan = json.loads((public_root / "interface_plan.json").read_text())
    cases = plan.get("cases", [])
    if len(cases) != 1:
        raise SHARED.AcceptanceError("existing public output must contain one interface case")
    case = cases[0]
    target_homology = f"{plan['homology']}:{case['target_chain']}"
    return FINAL._public_command(
        input_path=Path(plan["input"]),
        target_homology=target_homology,
        out_root=public_root,
        resume=True,
    )


def _case_output(public_root: Path) -> Path:
    plan = json.loads((public_root / "interface_plan.json").read_text())
    cases = plan.get("cases", [])
    if len(cases) != 1:
        raise SHARED.AcceptanceError("existing public output must contain one interface case")
    return public_root / cases[0]["output"]


def _continuation_paths(root: Path) -> dict[str, Path]:
    return {
        "root": root,
        "baseline": root / "baseline",
        "service": root / "baseline-service",
        "control": root / "baseline-control",
        "logs": root / "logs",
        "summary": root / "summary.json",
    }


def _resume_checks(
    *,
    returncode: int,
    tree_before: Mapping[str, str],
    tree_after: Mapping[str, str],
    owner_before: Mapping[str, Any],
    owner_after: Mapping[str, Any],
) -> dict[str, Any]:
    before_counts = owner_before["events"]["counts"]
    after_counts = owner_after["events"]["counts"]
    before_all = owner_before["all_event_counts"]
    after_all = owner_after["all_event_counts"]
    new_requests = (
        after_counts.get("request_received", 0)
        - before_counts.get("request_received", 0)
    )
    new_model_loads = (
        after_all.get("model_load_start", 0)
        - before_all.get("model_load_start", 0)
    )
    checks = {
        "returncode_zero": returncode == 0,
        "tree_unchanged": dict(tree_before) == dict(tree_after),
        "owner_events_unchanged": owner_before == owner_after,
        "new_requests_zero": new_requests == 0,
        "new_model_loads_zero": new_model_loads == 0,
    }
    return {
        "passed": all(checks.values()),
        "returncode": returncode,
        "checks": checks,
        "new_requests": new_requests,
        "new_model_loads": new_model_loads,
    }


def _prepare(
    *,
    existing_public: Path,
    continuation_root: Path,
    preflight_root: Path,
    target_python: Path,
) -> dict[str, Any]:
    existing_public = existing_public.expanduser().resolve()
    continuation_root = continuation_root.expanduser().resolve()
    preflight_root = preflight_root.expanduser().resolve()
    if continuation_root.exists():
        raise SHARED.AcceptanceError(
            f"continuation root must not exist during preflight: {continuation_root}"
        )
    if preflight_root.exists() and any(preflight_root.iterdir()):
        raise SHARED.AcceptanceError(f"preflight root is not empty: {preflight_root}")
    FINAL.LAYOUT._validate_public_first(existing_public)
    public_prediction = FINAL._prediction_evidence(_case_output(existing_public) / FINAL.ROOT_RESULT)
    resume_command = _resume_command(existing_public)
    manifest = _source_manifest()
    closure = FINAL._runtime_closure(target_python)
    paths = _continuation_paths(continuation_root)
    record = {
        "schema": f"{SCHEMA}.plan",
        "status": "prepared",
        "source_fingerprint": _source_fingerprint(manifest),
        "existing_public": str(existing_public),
        "existing_public_prediction": public_prediction,
        "resume_command": resume_command,
        "continuation_root": str(continuation_root),
        "baseline_root": str(paths["baseline"]),
        "baseline_service_root": str(paths["service"]),
        "baseline_control_root": str(paths["control"]),
        "runtime_closure": closure,
        "max_new_actual_model_requests": MAX_NEW_ACTUAL_REQUESTS,
        "public_root_replayed": False,
        "new_af3_calls": 0,
        "model_started": False,
        "resource_envelope": {
            "cpus": 4,
            "gpu": 1,
            "memory_gb": 28,
            "partition": "boron",
            "time_minutes": 60,
        },
    }
    preflight_root.mkdir(parents=True, exist_ok=True)
    SHARED._atomic_json(preflight_root / "plan.json", record)
    SHARED._atomic_text(
        preflight_root / "source_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    return record


def _execute(
    *,
    existing_public: Path,
    continuation_root: Path,
    preflight_root: Path,
    timeout_s: float,
) -> dict[str, Any]:
    preflight = json.loads((preflight_root / "plan.json").read_text())
    fingerprint = _source_fingerprint(_source_manifest())
    if fingerprint != preflight.get("source_fingerprint"):
        raise SHARED.AcceptanceError("source changed after CPU preflight")
    existing_public = existing_public.expanduser().resolve()
    continuation_root = continuation_root.expanduser().resolve()
    paths = _continuation_paths(continuation_root)
    if continuation_root.exists():
        raise SHARED.AcceptanceError("continuation paths already exist")
    FINAL.LAYOUT._validate_public_first(existing_public)
    continuation_root.mkdir(parents=True)
    paths["logs"].mkdir()

    tree_before = FINAL._tree_manifest(existing_public)
    owner_before = FINAL._owner_evidence(existing_public)
    resume_code = FINAL._run(
        _resume_command(existing_public),
        label="public-completed-resume-continuation",
        log_root=paths["logs"],
        timeout_s=timeout_s,
    )
    tree_after = FINAL._tree_manifest(existing_public)
    owner_after = FINAL._owner_evidence(existing_public)
    completed_resume = _resume_checks(
        returncode=resume_code,
        tree_before=tree_before,
        tree_after=tree_after,
        owner_before=owner_before,
        owner_after=owner_after,
    )
    if not completed_resume["passed"]:
        raise SHARED.AcceptanceError(
            "completed public resume changed artifacts or added model work"
        )

    cfg, config_evidence = FINAL._prepare_public_config(
        public_root=existing_public,
        baseline_root=paths["baseline"],
        baseline_service_root=paths["service"],
    )
    from astrolabe.evaluators.af3_lifecycle import OwnedAF3
    from astrolabe.interface import cpu_coordinator_environment
    from astrolabe.protein_beam import run_beam

    with cpu_coordinator_environment() as gpu_environment:
        owner = OwnedAF3(cfg, paths["service"], gpu_environment)
        with owner as socket_path:
            baseline_state = run_beam(
                cfg,
                paths["baseline"],
                socket_path,
                workers=int(cfg.protein_beam.workers),
                in_flight=int(cfg.protein_beam.in_flight),
                resume=False,
                retry_interrupted=False,
                on_abort=owner.abort,
            )
    comparison = FINAL._strict_comparison(
        FINAL._prediction_evidence(_case_output(existing_public) / FINAL.ROOT_RESULT),
        FINAL._prediction_evidence(paths["baseline"] / FINAL.ROOT_RESULT),
    )
    public_owner = FINAL._owner_evidence(existing_public)
    baseline_owner = FINAL._owner_evidence(paths["service"])
    public_counts = public_owner["events"]["counts"]
    baseline_counts = baseline_owner["events"]["counts"]
    public_all_counts = public_owner["all_event_counts"]
    baseline_all_counts = baseline_owner["all_event_counts"]
    checks = {
        "existing_public_root_reused": (
            public_counts.get("request_received") == 1
            and public_counts.get("gpu_start") == 1
            and public_counts.get("gpu_ok") == 1
            and public_all_counts.get("model_load_start") == 1
        ),
        "completed_public_resume_zero_model": completed_resume["passed"],
        "independent_native_root_one_request": (
            baseline_counts.get("request_received") == 1
            and baseline_counts.get("gpu_start") == 1
            and baseline_counts.get("gpu_ok") == 1
            and baseline_all_counts.get("model_load_start") == 1
            and baseline_state.get("status") == "complete"
        ),
        "strict_scientific_parity": comparison["passed"],
        "new_actual_requests_at_most_one": (
            baseline_counts.get("request_received", 0) <= MAX_NEW_ACTUAL_REQUESTS
            and completed_resume["new_requests"] == 0
        ),
        "config_delta_output_only": set(config_evidence["differences"]) <= set(
            config_evidence["allowed_differences"]
        ),
    }
    summary = {
        "schema": f"{SCHEMA}.summary",
        "attempt": "public_interface_continuation_001",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "ISSUES",
        "source_fingerprint": fingerprint,
        "checks": checks,
        "completed_resume": completed_resume,
        "comparison": comparison,
        "config": config_evidence,
        "public_owner": public_owner,
        "baseline_owner": baseline_owner,
        "existing_public": str(existing_public),
        "public_replayed": False,
        "baseline_output": str(paths["baseline"]),
        "baseline_service_root": str(paths["service"]),
        "new_actual_requests": baseline_counts.get("request_received", 0),
        "max_new_actual_requests": MAX_NEW_ACTUAL_REQUESTS,
        "new_af3_calls_during_preflight": 0,
        "runtime_closure": preflight["runtime_closure"],
    }
    SHARED._atomic_json(paths["summary"], summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-public", type=Path, required=True)
    parser.add_argument("--continuation-root", type=Path, required=True)
    parser.add_argument("--preflight-root", type=Path, required=True)
    parser.add_argument(
        "--target-python",
        type=Path,
        default=Path("/alphafold3_venv/bin/python"),
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=3600.0)
    arguments = parser.parse_args(argv)
    if arguments.timeout_s <= 0:
        parser.error("--timeout-s must be positive")
    try:
        if arguments.prepare_only:
            result = _prepare(
                existing_public=arguments.existing_public,
                continuation_root=arguments.continuation_root,
                preflight_root=arguments.preflight_root,
                target_python=arguments.target_python,
            )
        else:
            result = _execute(
                existing_public=arguments.existing_public,
                continuation_root=arguments.continuation_root,
                preflight_root=arguments.preflight_root,
                timeout_s=arguments.timeout_s,
            )
    except Exception as error:
        summary_path = (
            arguments.preflight_root / "summary.json"
            if arguments.prepare_only
            else _continuation_paths(arguments.continuation_root)["summary"]
        )
        summary_path.parent.mkdir(parents=True, exist_ok=True)
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
    output = (
        arguments.preflight_root / "plan.json"
        if arguments.prepare_only
        else _continuation_paths(arguments.continuation_root)["summary"]
    )
    print(json.dumps({"status": result["status"], "summary": str(output)}))
    return 0 if result["status"] in {"prepared", "PASS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
