#!/usr/bin/env python3
"""Preflight and bounded execution for the frozen portable_public_v4 CLI test.

The scientific runtime and CLI recovery harness are loaded from the frozen
portable_public_v4 capsule.  This controller only chooses whether its existing
API baseline is reusable in the current allocation and enforces the combined
request cap across an optional fresh baseline and the CLI harness.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "69d4df40246cb0c5f04b070f258c4e2b8f553d2e409156f5122eebcae7e12bfe"
)
EXPECTED_SOURCE_FINGERPRINT = (
    "05bb426a66c222d6888a8a95fb020ecc3e1f130a4101bb1c21b88fad7840f09a"
)
ORIGINAL_PAPER_INPUT_SHA256 = (
    "00bbed74f1bb47d85acc6f793fa25d1c116b94592d30ad5d1b574de055e5903d"
)
CLI_BUDGET = 6
DEFAULT_BASELINE_BUDGET = 2
HARD_REQUEST_CAP = 8
NO_RETRY = True


class ControllerError(RuntimeError):
    """The public-CLI recovery task cannot proceed safely."""


@dataclass(frozen=True)
class BaselineStrategy:
    mode: str
    request_budget: int
    reason: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


_CPU_PRELUDE_SCRIPT = r'''
import json
import os
from pathlib import Path
import sys
import importlib.util

from omegaconf import OmegaConf


CAPSULE = Path(sys.argv[1]).resolve()
CONFIG = Path(sys.argv[2]).resolve()
OUTPUT = Path(sys.argv[3]).resolve()
cfg = OmegaConf.load(CONFIG)

run_path = CAPSULE / "run.py"
spec = importlib.util.spec_from_file_location("astrolabe_run_cpu_prelude", run_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load run.py from {CAPSULE}")
run_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = run_module
spec.loader.exec_module(run_module)

prepared = run_module.prepare_pipeline_state(
    cfg,
    apply_initialization=False,
)
loss_computer, stage_manager = run_module._initialize_pipeline_math(
    cfg,
    prepared,
)
payload = {
    "schema": "astrolabe.public-cli-recovery.cpu-prelude-result/v1",
    "status": "PASS",
    "config": str(CONFIG),
    "capsule_root": str(CAPSULE),
    "apply_initialization": False,
    "initializer_mode": prepared.initializer_mode,
    "initializer_result_is_none": prepared.initializer_result is None,
    "runtime_context_type": type(prepared.runtime_context).__name__,
    "loss_computer_type": type(loss_computer).__name__,
    "stage_manager_type": type(stage_manager).__name__,
    "current_oligo": getattr(stage_manager, "current_oligo", None),
    "model_calls": 0,
}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
temporary = OUTPUT.with_name(f".{OUTPUT.name}.tmp-{os.getpid()}")
temporary.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.replace(temporary, OUTPUT)
'''.lstrip()


def _write_cpu_prelude_script(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_CPU_PRELUDE_SCRIPT, encoding="utf-8")
    return path


def _tree_manifest(root: Path) -> dict[str, str]:
    if not root.is_dir():
        return {}
    return {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _run_once(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    stdout_path: Path,
    stderr_path: Path,
    timeout_s: float,
) -> dict[str, Any]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            [str(item) for item in command],
            cwd=str(cwd),
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            close_fds=True,
        )
        timed_out = False
        try:
            returncode = process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
            returncode = process.wait()
    return {
        "returncode": returncode,
        "timed_out": timed_out,
        "duration_s": time.monotonic() - started,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }


def run_cpu_prelude(
    *,
    capsule_root: Path,
    config_path: Path,
    out_root: Path,
    python_bin: Path,
    timeout_s: float,
) -> dict[str, Any]:
    if not config_path.is_file():
        raise ControllerError(f"failed resume config is missing: {config_path}")
    claims_root = config_path.parent / "call-claims"
    claims_before = _tree_manifest(claims_root)
    prelude_root = out_root / "cpu-prelude"
    prelude_script = _write_cpu_prelude_script(
        prelude_root / "run_cpu_prelude.py"
    )
    result_path = prelude_root / "result.json"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (
            str(capsule_root),
            environment.get("PYTHONPATH", ""),
        )
        if part
    )
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["GH_CUDA_PREINITIALIZE"] = "0"
    environment["ASTROLABE_RUNTIME_ACTIVE"] = "1"
    environment["ASTROLABE_ACCEPTANCE_MAX_EVALUATIONS"] = "2"
    environment["PYTHONHASHSEED"] = "17"
    command = [
        str(python_bin),
        str(prelude_script),
        str(capsule_root),
        str(config_path),
        str(result_path),
    ]
    process = _run_once(
        command,
        cwd=capsule_root,
        environment=environment,
        stdout_path=prelude_root / "stdout.log",
        stderr_path=prelude_root / "stderr.log",
        timeout_s=timeout_s,
    )
    result = (
        json.loads(result_path.read_text(encoding="utf-8"))
        if result_path.is_file()
        else None
    )
    claims_after = _tree_manifest(claims_root)
    no_model_call = claims_after == claims_before
    passed = (
        process["returncode"] == 0
        and not process["timed_out"]
        and isinstance(result, Mapping)
        and result.get("status") == "PASS"
        and result.get("apply_initialization") is False
        and result.get("initializer_result_is_none") is True
        and no_model_call
    )
    evidence = {
        "schema": "astrolabe.public-cli-recovery.cpu-prelude/v1",
        "status": "PASS" if passed else "ISSUES",
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "script": str(prelude_script),
        "command": command,
        "process": process,
        "result": result,
        "claims_before": {"files": claims_before, "count": len(claims_before)},
        "claims_after": {
            "files": claims_after,
            "count": len(claims_after),
        },
        "model_calls": 0 if no_model_call else max(1, len(claims_after) - len(claims_before)),
        "scope": (
            "CPU prelude only: parsed config -> "
            "prepare_pipeline_state(apply_initialization=False) -> "
            "prepare_pipeline_math. This is not full cold-resume evidence."
        ),
    }
    _atomic_json(prelude_root / "evidence.json", evidence)
    if not passed:
        raise ControllerError(
            "CPU prelude did not complete without model calls: "
            f"rc={process['returncode']} timeout={process['timed_out']} "
            f"result={json.dumps(result, sort_keys=True)} "
            f"claims_before={len(claims_before)} "
            f"claims_after={len(claims_after)}"
        )
    return evidence


def _load_module(path: Path, module_name: str):
    loaded = sys.modules.get(module_name)
    if loaded is not None:
        return loaded
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ControllerError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def validate_capsule(
    capsule_root: Path,
    *,
    expected_manifest_sha256: str | None = None,
    expected_fingerprint: str | None = None,
) -> dict[str, Any]:
    expected_manifest_sha256 = (
        EXPECTED_SOURCE_MANIFEST_SHA256
        if expected_manifest_sha256 is None
        else expected_manifest_sha256
    )
    expected_fingerprint = (
        EXPECTED_SOURCE_FINGERPRINT
        if expected_fingerprint is None
        else expected_fingerprint
    )
    manifest_path = capsule_root / "source_manifest.json"
    if not manifest_path.is_file():
        raise ControllerError(f"capsule source manifest is missing: {manifest_path}")
    observed_sha256 = _sha256(manifest_path)
    if observed_sha256 != expected_manifest_sha256:
        raise ControllerError(
            "capsule source manifest changed: "
            f"expected={expected_manifest_sha256} observed={observed_sha256}"
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    observed_fingerprint = payload.get("fingerprint")
    if observed_fingerprint != expected_fingerprint:
        raise ControllerError(
            "capsule source fingerprint changed: "
            f"expected={expected_fingerprint} observed={observed_fingerprint}"
        )
    metadata_path = capsule_root / "capsule_metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file()
        else {}
    )
    return {
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": observed_sha256,
        "source_fingerprint": observed_fingerprint,
        "capsule_metadata": metadata,
    }


def validate_paper_overlay(original: Path, overlay: Path) -> dict[str, Any]:
    from omegaconf import OmegaConf

    if not original.is_file() or not overlay.is_file():
        raise ControllerError("both original and portable paper inputs are required")
    original_sha256 = _sha256(original)
    if original_sha256 != ORIGINAL_PAPER_INPUT_SHA256:
        raise ControllerError(
            "original paper input changed: "
            f"expected={ORIGINAL_PAPER_INPUT_SHA256} observed={original_sha256}"
        )
    original_payload = OmegaConf.to_container(
        OmegaConf.load(original),
        resolve=False,
    )
    overlay_payload = OmegaConf.to_container(
        OmegaConf.load(overlay),
        resolve=False,
    )
    if not isinstance(original_payload, dict) or not isinstance(overlay_payload, dict):
        raise ControllerError("paper inputs must be YAML mappings")
    if overlay_payload.get("skip_complex_poses") is not True:
        raise ControllerError("portable paper overlay must set skip_complex_poses=true")
    overlay_without_transport = dict(overlay_payload)
    overlay_without_transport.pop("skip_complex_poses", None)
    if overlay_without_transport != original_payload:
        raise ControllerError(
            "portable paper overlay changes scientific inputs beyond "
            "skip_complex_poses"
        )
    return {
        "original_path": str(original),
        "original_sha256": original_sha256,
        "overlay_path": str(overlay),
        "overlay_sha256": _sha256(overlay),
        "structural_difference": {"path": "$.skip_complex_poses", "value": True},
        "transport_only": True,
    }


def select_baseline_strategy(choice: Any) -> BaselineStrategy:
    """Authorize fresh default only for an otherwise reusable GPU mismatch."""

    if bool(getattr(choice, "reuse", False)):
        return BaselineStrategy(
            mode="reuse",
            request_budget=CLI_BUDGET,
            reason="API and CLI allocations match",
        )
    reasons = tuple(getattr(choice, "reasons", ()) or ())
    source_changes = tuple(getattr(choice, "source_changes", ()) or ())
    config_differences = tuple(
        getattr(choice, "config_differences", ()) or ()
    )
    weights_match = getattr(choice, "weights_match", None)
    runtime_image_match = getattr(choice, "runtime_image_match", None)
    rng_evidence_present = bool(
        getattr(choice, "rng_evidence_present", False)
    )
    gpu_match = getattr(choice, "gpu_match", None)
    if (
        reasons == ("historical baseline GPU allocation differs",)
        and gpu_match is False
        and not source_changes
        and not config_differences
        and weights_match is True
        and runtime_image_match is True
        and rng_evidence_present
    ):
        return BaselineStrategy(
            mode="fresh-default",
            request_budget=DEFAULT_BASELINE_BUDGET + CLI_BUDGET,
            reason=(
                "API baseline differs only in GPU allocation; recreate the "
                "default comparison baseline in the CLI allocation"
            ),
        )
    raise ControllerError(
        "API baseline is not reusable for a reason other than GPU allocation: "
        + json.dumps(
            {
                "reasons": list(reasons),
                "source_changes": list(source_changes),
                "config_differences": list(config_differences),
                "weights_match": weights_match,
                "runtime_image_match": runtime_image_match,
                "rng_evidence_present": rng_evidence_present,
                "gpu_match": gpu_match,
            },
            sort_keys=True,
        )
    )


def _gpu_comparison(
    api_baseline: Path,
    current_environment: Mapping[str, Any],
) -> dict[str, Any]:
    result_path = api_baseline / "result.json"
    if not result_path.is_file():
        raise ControllerError(
            f"API native-in-process baseline result is missing: {result_path}"
        )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    baseline_environment = result.get("environment_manifest")
    if not isinstance(baseline_environment, Mapping):
        raise ControllerError("API baseline environment manifest is missing")
    return {
        "api_baseline_gpu": baseline_environment.get("gpu"),
        "cli_allocation_gpu": current_environment.get("gpu"),
        "equal": baseline_environment.get("gpu")
        == current_environment.get("gpu"),
    }


def _require_empty_root(out_root: Path) -> None:
    if out_root.exists() and any(out_root.iterdir()):
        raise ControllerError(
            f"--out-root must be absent or empty; refusing to overwrite {out_root}"
        )
    out_root.mkdir(parents=True, exist_ok=True)
    try:
        out_root.chmod(0o700)
    except OSError:
        pass


def _preflight_payload(
    *,
    capsule: Mapping[str, Any],
    paper_overlay: Mapping[str, Any],
    api_baseline: Path,
    failed_config: Path | None,
    out_root: Path,
    status: str,
    stage: str,
) -> dict[str, Any]:
    return {
        "schema": "astrolabe.public-cli-recovery.controller/v1",
        "status": status,
        "stage": stage,
        "capsule": dict(capsule),
        "paper_input": dict(paper_overlay),
        "api_baseline": str(api_baseline),
        "failed_config": (
            str(failed_config)
            if failed_config is not None
            else None
        ),
        "request_policy": {
            "cli_budget": CLI_BUDGET,
            "fresh_default_budget": DEFAULT_BASELINE_BUDGET,
            "hard_cap": HARD_REQUEST_CAP,
            "no_retry": NO_RETRY,
        },
        "out_root": str(out_root),
    }


def run_controller(
    *,
    capsule_root: Path,
    expected_source_manifest_sha256: str,
    expected_source_fingerprint: str,
    out_root: Path,
    api_baseline: Path,
    failed_config: Path | None,
    paper_inputs: Path,
    paper_overlay: Path,
    paper_baseline: Path,
    profile: Path,
    python_bin: Path,
    af3_module_path: Path,
    timeout_s: float,
    dry_run: bool,
) -> int:
    _require_empty_root(out_root)
    capsule = validate_capsule(
        capsule_root,
        expected_manifest_sha256=expected_source_manifest_sha256,
        expected_fingerprint=expected_source_fingerprint,
    )
    paper = validate_paper_overlay(paper_inputs, paper_overlay)
    preflight = _preflight_payload(
        capsule=capsule,
        paper_overlay=paper,
        api_baseline=api_baseline,
        failed_config=failed_config,
        out_root=out_root,
        status="PASS" if dry_run else "IN_PROGRESS",
        stage="static-preflight" if dry_run else "environment-preflight",
    )
    _atomic_json(out_root / "preflight.json", preflight)
    if dry_run:
        preflight["conditional_plan"] = {
            "api_gpu_match": {
                "mode": "reuse",
                "request_budget": CLI_BUDGET,
            },
            "api_gpu_mismatch_only": {
                "mode": "fresh-default",
                "fresh_default_requests": DEFAULT_BASELINE_BUDGET,
                "request_budget": DEFAULT_BASELINE_BUDGET + CLI_BUDGET,
            },
            "other_baseline_mismatch": "STOP_BEFORE_MODEL_REQUEST",
            "cpu_prelude": {
                "required": True,
                "apply_initialization": False,
                "full_cold_resume": (
                    "validated by formal GPU CLI cut/resume comparison; "
                    "the CPU prelude is not recovery evidence"
                ),
                "request_budget": 0,
            },
        }
        _atomic_json(out_root / "preflight.json", preflight)
        print(json.dumps({"status": "PASS", "preflight": str(out_root / "preflight.json")}))
        return 0
    if failed_config is None:
        raise ControllerError(
            "a real failed resume config is required for the CPU prelude"
        )

    harness_path = (
        capsule_root
        / "scripts"
        / "validation"
        / "public_cli_recovery_gpu_acceptance.py"
    )
    harness = _load_module(
        harness_path,
        "astrolabe_public_cli_recovery_v4_harness",
    )
    unified = harness._load_unified()
    current_source = unified._read_source_manifest()
    if current_source.fingerprint != capsule["source_fingerprint"]:
        raise ControllerError(
            "loaded source fingerprint does not match the frozen capsule: "
            f"expected={capsule['source_fingerprint']} "
            f"observed={current_source.fingerprint}"
        )
    cpu_prelude = run_cpu_prelude(
        capsule_root=capsule_root,
        config_path=failed_config,
        out_root=out_root,
        python_bin=python_bin,
        timeout_s=timeout_s,
    )
    environment = harness._run_environment(
        profile=profile,
        out_root=out_root / "environment-preflight",
        current_source=current_source,
    )
    bound_cases = harness._bind_case_specs(
        default_baseline=api_baseline,
        paper_baseline=paper_baseline,
    )
    choice = harness._baseline_choice(
        case=bound_cases["default-cli"],
        profile=profile,
        environment=environment,
        af3_module_path=af3_module_path,
    )
    gpu_comparison = _gpu_comparison(api_baseline, environment)
    if gpu_comparison["equal"] != choice.gpu_match:
        raise ControllerError(
            "frozen harness and controller disagree about GPU equality"
        )
    strategy = select_baseline_strategy(choice)
    preflight.update(
        {
            "status": "PASS",
            "stage": "allocation-preflight",
            "api_baseline_choice": {
                "reuse": choice.reuse,
                "reasons": list(choice.reasons),
                "source_changes": list(choice.source_changes),
                "config_differences": list(choice.config_differences),
                "weights_match": choice.weights_match,
                "gpu_match": choice.gpu_match,
                "runtime_image_match": choice.runtime_image_match,
                "rng_evidence_present": choice.rng_evidence_present,
            },
            "gpu_comparison": gpu_comparison,
            "cpu_prelude": cpu_prelude,
            "strategy": {
                "mode": strategy.mode,
                "reason": strategy.reason,
                "request_budget": strategy.request_budget,
            },
        }
    )
    _atomic_json(out_root / "preflight.json", preflight)

    default_baseline = api_baseline
    fresh_default_requests = 0
    if strategy.mode == "fresh-default":
        fresh_root = out_root / "default-baseline"
        fresh_root.mkdir(parents=True, exist_ok=False)
        baseline_budget = harness.RequestBudget(
            fresh_root / "request_budget.json",
            limit=DEFAULT_BASELINE_BUDGET,
        )
        default_baseline = harness._fresh_baseline(
            case=bound_cases["default-cli"],
            profile=profile,
            out_root=fresh_root,
            environment=environment,
            python_bin=python_bin,
            timeout_s=timeout_s,
            budget=baseline_budget,
        )
        baseline_result = json.loads(
            (default_baseline / "result.json").read_text(encoding="utf-8")
        )
        fresh_default_requests = int(baseline_result.get("model_calls") or 0)
        if (
            baseline_result.get("status") != "evidence-complete"
            or fresh_default_requests != DEFAULT_BASELINE_BUDGET
        ):
            raise ControllerError(
                "fresh default baseline did not produce exactly two requests"
            )

    cli_root = out_root / "public-cli"
    cli_status = harness.run_acceptance(
        out_root=cli_root,
        profile=profile,
        paper_input=paper_overlay,
        python_bin=python_bin,
        timeout_s=timeout_s,
        af3_module_path=af3_module_path,
        default_baseline=default_baseline,
        paper_baseline=paper_baseline,
    )
    summary_path = cli_root / "summary.json"
    if not summary_path.is_file():
        raise ControllerError("public CLI acceptance summary was not written")
    cli_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cli_requests = int(cli_summary.get("actual_observed_evaluator_requests") or 0)
    total_requests = fresh_default_requests + cli_requests
    expected_total = strategy.request_budget
    passed = (
        cli_status == 0
        and cli_summary.get("status") == "PASS"
        and total_requests == expected_total
        and total_requests <= HARD_REQUEST_CAP
    )
    summary = {
        "schema": "astrolabe.public-cli-recovery.controller-summary/v1",
        "status": "PASS" if passed else "ISSUES",
        "capsule": capsule,
        "paper_input": paper,
        "api_baseline": str(api_baseline),
        "default_baseline": str(default_baseline),
        "strategy": {
            "mode": strategy.mode,
            "reason": strategy.reason,
        },
        "gpu_comparison": gpu_comparison,
        "cpu_prelude": cpu_prelude,
        "fresh_default_requests": fresh_default_requests,
        "public_cli_requests": cli_requests,
        "total_model_requests": total_requests,
        "request_budget": expected_total,
        "hard_request_cap": HARD_REQUEST_CAP,
        "no_retry": NO_RETRY,
        "preflight": str(out_root / "preflight.json"),
        "public_cli_summary": str(summary_path),
    }
    _atomic_json(out_root / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "summary": str(out_root / "summary.json")}))
    return 0 if passed else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight the frozen portable_public_v4 API baseline and run the "
            "same-capsule public CLI recovery harness with a hard 8-request cap."
        )
    )
    parser.add_argument("--capsule-root", type=Path, required=True)
    parser.add_argument(
        "--expected-source-manifest-sha256",
        default=EXPECTED_SOURCE_MANIFEST_SHA256,
    )
    parser.add_argument(
        "--expected-source-fingerprint",
        default=EXPECTED_SOURCE_FINGERPRINT,
    )
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--api-baseline", type=Path, required=True)
    parser.add_argument("--failed-config", type=Path)
    parser.add_argument("--paper-inputs", type=Path, required=True)
    parser.add_argument("--paper-overlay", type=Path, required=True)
    parser.add_argument("--paper-baseline", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--af3-module-path",
        type=Path,
        default=Path("/alphafold3_venv/lib/python3.11/site-packages/alphafold3"),
    )
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args(argv)
    capsule_root = arguments.capsule_root.expanduser().resolve()
    profile = (
        arguments.profile.expanduser().resolve()
        if arguments.profile is not None
        else capsule_root / "config" / "container_af3_smoke.yaml"
    )
    try:
        return run_controller(
            capsule_root=capsule_root,
            expected_source_manifest_sha256=(
                arguments.expected_source_manifest_sha256
            ),
            expected_source_fingerprint=arguments.expected_source_fingerprint,
            out_root=arguments.out_root.expanduser().resolve(),
            api_baseline=arguments.api_baseline.expanduser().resolve(),
            failed_config=(
                arguments.failed_config.expanduser().resolve()
                if arguments.failed_config is not None
                else None
            ),
            paper_inputs=arguments.paper_inputs.expanduser().resolve(),
            paper_overlay=arguments.paper_overlay.expanduser().resolve(),
            paper_baseline=arguments.paper_baseline.expanduser().resolve(),
            profile=profile,
            python_bin=arguments.python_bin.expanduser().resolve(),
            af3_module_path=arguments.af3_module_path.expanduser().resolve(),
            timeout_s=arguments.timeout_s,
            dry_run=arguments.dry_run,
        )
    except Exception as error:
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


if __name__ == "__main__":
    raise SystemExit(main())
