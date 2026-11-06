#!/usr/bin/env python3
"""Pair the real public interface root with an independent native root."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
LAYOUT_PATH = Path(__file__).with_name("layout_then_public_gpu_acceptance.py")
SCHEMA = "astrolabe.public-interface-final/v2"
ROOT_RESULT = Path("candidates/000000/result.json")
MAX_ACTUAL_REQUESTS = 2


def _load_layout():
    name = "astrolabe_public_interface_final_layout"
    loaded = sys.modules.get(name)
    if loaded is not None:
        return loaded
    spec = importlib.util.spec_from_file_location(name, LAYOUT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load public interface acceptance helpers")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


LAYOUT = _load_layout()
SHARED = LAYOUT.SHARED


def _source_manifest() -> dict[str, str]:
    manifest = dict(SHARED._source_manifest())
    for path in (LAYOUT_PATH.resolve(), Path(__file__).resolve()):
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


def _run(
    command: Sequence[str],
    *,
    label: str,
    log_root: Path,
    timeout_s: float,
) -> int:
    return LAYOUT._run(
        command,
        label=label,
        log_root=log_root,
        timeout_s=timeout_s,
    )


def _runtime_closure(target_python: Path) -> dict[str, Any]:
    from astrolabe.evaluators.af3_lifecycle import af3_child_environment

    target_python = target_python.expanduser().resolve()
    runtime = af3_child_environment(target_python, os.environ)
    script = '''
import ctypes,json,sys
names=[
    "libnvrtc.so.12",
    "libnvrtc-builtins.so.12.6",
    "libcudnn.so.9",
    "libcudnn_engines_runtime_compiled.so.9",
]
for name in names:
    ctypes.CDLL(name)
print(json.dumps({
    "loaded": names,
    "torch_imported": "torch" in sys.modules,
    "jax_imported": "jax" in sys.modules,
}))
'''
    completed = subprocess.run(
        [runtime.python_bin, "-I", "-B", "-c", script],
        env=runtime.environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if completed.returncode:
        raise SHARED.AcceptanceError(
            f"target Python GPU runtime closure failed: {completed.stderr}"
        )
    result = json.loads(completed.stdout)
    if result.get("torch_imported") or result.get("jax_imported"):
        raise SHARED.AcceptanceError(
            "runtime closure imported Torch or JAX before the model guard"
        )
    return {
        "python_bin": runtime.python_bin,
        "library_directories": list(runtime.library_directories),
        "ld_library_path": runtime.environment.get("LD_LIBRARY_PATH"),
        "cdll": result,
    }


def _tree_manifest(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): SHARED._sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _normalize(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise SHARED.AcceptanceError("object dtype in score snapshot")
        array = np.ascontiguousarray(value)
        return {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
    if isinstance(value, np.generic):
        array = np.asarray(value)
        return {
            "dtype": array.dtype.str,
            "sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
        }
    if isinstance(value, Path):
        return "<output-path>"
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value.hex()
    if isinstance(value, dict):
        return {
            str(key): _normalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "attributes": _normalize(attributes),
        }
    raise SHARED.AcceptanceError(f"unsupported score value {type(value)!r}")


def _prediction_evidence(result_path: Path) -> dict[str, Any]:
    result = json.loads(result_path.read_text())
    scores_path = result_path.parent / "scores.pkl"
    with scores_path.open("rb") as stream:
        scores = pickle.load(stream)
    from astrolabe.protein_beam_root_parity import _atoms, _confidence, _digest

    structure = Path(result["structure_path"])
    normalized = _normalize(scores)
    return {
        "result_path": str(result_path),
        "scores_path": str(scores_path),
        "scores_file_sha256": SHARED._sha256(scores_path),
        "scores_normalized_sha256": hashlib.sha256(
            json.dumps(
                normalized,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "coordinates_sha256": _digest(_atoms(structure)),
        "confidence_sha256": _digest(_confidence(result)),
        "scientific": {
            key: result.get(key)
            for key in (
                "id",
                "parent_id",
                "generation",
                "kind",
                "evaluation_origin",
                "af3_call_token",
                "sequence",
                "seed",
                "metrics",
                "holes",
                "geometry",
                "score",
                "status",
            )
        },
    }


def _strict_comparison(public: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    def digest(value: Any) -> str:
        return hashlib.sha256(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()

    checks = {
        "raw_score_tree_exact": (
            public["scores_normalized_sha256"]
            == baseline["scores_normalized_sha256"]
        ),
        "all_atom_coordinates_exact": (
            public["coordinates_sha256"] == baseline["coordinates_sha256"]
        ),
        "full_native_confidence_exact": (
            public["confidence_sha256"] == baseline["confidence_sha256"]
        ),
        "physics_and_result_scientific_exact": (
            digest(public["scientific"]) == digest(baseline["scientific"])
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "public": public,
        "baseline": baseline,
    }


def _owner_evidence(service_root: Path) -> dict[str, Any]:
    states = []
    all_counts: dict[str, int] = {}
    for lifecycle in sorted((service_root / "services").glob("*/lifecycle.json")):
        state = json.loads(lifecycle.read_text())
        stdout_path = Path(str(state.get("stdout_path", state.get("log", ""))))
        stderr_path = Path(str(state.get("stderr_path", "")))
        if not stdout_path.is_file() or not stderr_path.is_file():
            raise SHARED.AcceptanceError(f"owner stream paths missing: {lifecycle}")
        event_lines = stdout_path.read_text().splitlines()
        parsed_events = []
        for line in event_lines:
            if not line.startswith("{"):
                raise SHARED.AcceptanceError(
                    f"owner stdout contains non-event data: {stdout_path}"
                )
            event = json.loads(line)
            if not isinstance(event.get("event"), str):
                raise SHARED.AcceptanceError(
                    f"owner stdout event lacks event name: {stdout_path}"
                )
            parsed_events.append(event)
            name = event["event"]
            all_counts[name] = all_counts.get(name, 0) + 1
        if not parsed_events:
            raise SHARED.AcceptanceError(f"owner stdout is empty: {stdout_path}")
        states.append({
            "lifecycle": str(lifecycle),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "stdout_sha256": SHARED._sha256(stdout_path),
            "stderr_sha256": SHARED._sha256(stderr_path),
        })
    return {
        "states": states,
        "events": SHARED._inference_event_projection(service_root),
        "all_event_counts": dict(sorted(all_counts.items())),
    }


def _prepare_public_config(
    *,
    public_root: Path,
    baseline_root: Path,
    baseline_service_root: Path,
) -> tuple[Any, dict[str, Any]]:
    from omegaconf import OmegaConf, open_dict

    from astrolabe.protein_beam import load_config

    plan = json.loads((public_root / "interface_plan.json").read_text())
    if len(plan.get("cases", [])) != 1:
        raise SHARED.AcceptanceError("public interface must prepare exactly one case")
    case = plan["cases"][0]
    public_config_path = public_root / case["config"]
    public_cfg = load_config(public_config_path)
    original = OmegaConf.to_container(public_cfg, resolve=False)
    cfg = OmegaConf.create(original)
    with open_dict(cfg):
        cfg.out_dir = str(baseline_root)
        cfg.protein_beam.af3_call_budget.ledger = str(
            baseline_root / "af3_calls.json"
        )
    baseline_root.mkdir(parents=True, exist_ok=False)
    control_root = baseline_root.parent / "baseline-control"
    control_root.mkdir(parents=True, exist_ok=False)
    config_path = control_root / "config.yaml"
    SHARED._atomic_text(config_path, OmegaConf.to_yaml(cfg, resolve=False))
    changed = OmegaConf.to_container(cfg, resolve=False)
    differences = []

    def walk(left: Any, right: Any, prefix: str = "") -> None:
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) | set(right)):
                walk(left.get(key), right.get(key), f"{prefix}.{key}".strip("."))
            return
        if left != right:
            differences.append(prefix)

    walk(original, changed)
    allowed = {
        "out_dir",
        "protein_beam.af3_call_budget.ledger",
    }
    unknown = sorted(set(differences) - allowed)
    if unknown:
        raise SHARED.AcceptanceError(
            f"native baseline changed scientific config fields: {unknown}"
        )
    SHARED._validate_service_beam_layout(baseline_root, baseline_service_root)
    return cfg, {
        "public_config": str(public_config_path),
        "public_config_sha256": SHARED._sha256(public_config_path),
        "baseline_config": str(config_path),
        "baseline_config_sha256": SHARED._sha256(config_path),
        "differences": differences,
        "allowed_differences": sorted(allowed),
    }


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
    code = _run(
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
    closure = _runtime_closure(target_python)
    manifest = _source_manifest()
    plan_record = {
        "schema": f"{SCHEMA}.plan",
        "status": "prepared",
        "source_fingerprint": _source_fingerprint(manifest),
        "input": str(input_path),
        "target_homology": str(target_homology),
        "preflight_root": str(preflight_root),
        "execution_root": str(execution_root),
        "baseline_root": str(execution_root.parent / "baseline"),
        "baseline_service_root": str(execution_root.parent / "baseline-service"),
        "public_preflight_config": str(public_root / plan["cases"][0]["config"]),
        "science": science,
        "runtime_closure": closure,
        "execution_command": _public_command(
            input_path=input_path,
            target_homology=target_homology,
            out_root=execution_root,
        ),
        "max_actual_model_requests": MAX_ACTUAL_REQUESTS,
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
    SHARED._atomic_json(preflight_root / "plan.json", plan_record)
    SHARED._atomic_text(
        preflight_root / "source_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    return plan_record


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
    baseline_root = out_root.parent / "baseline"
    baseline_service_root = out_root.parent / "baseline-service"
    if out_root.exists() or baseline_root.exists() or baseline_service_root.exists():
        raise SHARED.AcceptanceError("execution paths already exist")
    log_root = out_root.parent / "logs"
    out_root.parent.mkdir(parents=True, exist_ok=True)

    code = _run(
        _public_command(
            input_path=input_path,
            target_homology=target_homology,
            out_root=out_root,
        ),
        label="public-root",
        log_root=log_root,
        timeout_s=timeout_s,
    )
    if code != 0:
        raise SHARED.AcceptanceError("public interface root run failed")
    LAYOUT._validate_public_first(out_root)
    public_tree_before = _tree_manifest(out_root)
    public_owner_before = _owner_evidence(out_root)

    resume_code = _run(
        _public_command(
            input_path=input_path,
            target_homology=target_homology,
            out_root=out_root,
            resume=True,
        ),
        label="public-completed-resume",
        log_root=log_root,
        timeout_s=timeout_s,
    )
    public_tree_after = _tree_manifest(out_root)
    public_owner_after = _owner_evidence(out_root)
    completed_resume = {
        "returncode": resume_code,
        "tree_unchanged": public_tree_before == public_tree_after,
        "owner_events_unchanged": public_owner_before == public_owner_after,
        "new_requests": (
            public_owner_after["events"]["counts"].get("request_received", 0)
            - public_owner_before["events"]["counts"].get("request_received", 0)
        ),
        "new_model_loads": (
            public_owner_after["all_event_counts"].get("model_load_start", 0)
            - public_owner_before["all_event_counts"].get("model_load_start", 0)
        ),
    }
    completed_resume["passed"] = (
        resume_code == 0
        and completed_resume["tree_unchanged"]
        and completed_resume["owner_events_unchanged"]
        and completed_resume["new_requests"] == 0
        and completed_resume["new_model_loads"] == 0
    )

    cfg, config_evidence = _prepare_public_config(
        public_root=out_root,
        baseline_root=baseline_root,
        baseline_service_root=baseline_service_root,
    )
    from astrolabe.evaluators.af3_lifecycle import OwnedAF3
    from astrolabe.interface import cpu_coordinator_environment
    from astrolabe.protein_beam import run_beam

    with cpu_coordinator_environment() as gpu_environment:
        owner = OwnedAF3(cfg, baseline_service_root, gpu_environment)
        with owner as socket_path:
            baseline_state = run_beam(
                cfg,
                baseline_root,
                socket_path,
                workers=int(cfg.protein_beam.workers),
                in_flight=int(cfg.protein_beam.in_flight),
                resume=False,
                retry_interrupted=False,
                on_abort=owner.abort,
            )
    case_name = json.loads((out_root / "interface_plan.json").read_text())["cases"][0]["name"]
    public_result = out_root / "cases" / case_name / ROOT_RESULT
    baseline_result = baseline_root / ROOT_RESULT
    comparison = _strict_comparison(
        _prediction_evidence(public_result),
        _prediction_evidence(baseline_result),
    )
    public_owner = _owner_evidence(out_root)
    baseline_owner = _owner_evidence(baseline_service_root)
    public_counts = public_owner["events"]["counts"]
    baseline_counts = baseline_owner["events"]["counts"]
    public_all_counts = public_owner["all_event_counts"]
    baseline_all_counts = baseline_owner["all_event_counts"]
    call_accounting = {
        "public_requests": public_counts.get("request_received", 0),
        "baseline_requests": baseline_counts.get("request_received", 0),
        "completed_resume_requests": completed_resume["new_requests"],
        "total_actual_requests": (
            public_counts.get("request_received", 0)
            + baseline_counts.get("request_received", 0)
        ),
        "max_actual_requests": MAX_ACTUAL_REQUESTS,
        "no_retry": True,
    }
    checks = {
        "public_one_completed_root": (
            public_counts.get("request_received") == 1
            and public_counts.get("gpu_start") == 1
            and public_counts.get("gpu_ok") == 1
            and public_all_counts.get("model_load_start") == 1
        ),
        "native_baseline_one_completed_root": (
            baseline_counts.get("request_received") == 1
            and baseline_counts.get("gpu_start") == 1
            and baseline_counts.get("gpu_ok") == 1
            and baseline_state.get("status") == "complete"
            and baseline_all_counts.get("model_load_start") == 1
        ),
        "completed_resume_zero_model": completed_resume["passed"],
        "strict_scientific_parity": comparison["passed"],
        "two_actual_requests": call_accounting["total_actual_requests"] == 2,
        "config_delta_output_only": set(config_evidence["differences"]) <= set(
            config_evidence["allowed_differences"]
        ),
        "stderr_separated": all(
            item["states"]
            for item in (public_owner, baseline_owner)
        ),
    }
    summary = {
        "schema": f"{SCHEMA}.summary",
        "attempt": "public_interface_final_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "ISSUES",
        "source_fingerprint": fingerprint,
        "checks": checks,
        "call_accounting": call_accounting,
        "completed_resume": completed_resume,
        "config": config_evidence,
        "comparison": comparison,
        "public_owner": public_owner,
        "baseline_owner": baseline_owner,
        "public_output": str(out_root),
        "baseline_output": str(baseline_root),
        "baseline_service_root": str(baseline_service_root),
        "no_retry": True,
        "new_af3_calls_during_preflight": 0,
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
