#!/usr/bin/env python3
"""Bounded public-CLI durable-recovery acceptance for default and paper frontdoors.

The recovery processes enter through the real installed CLI commands.  A
sitecustomize shim wraps the production ``Runtime.call`` only to terminate the
CLI process after the root raw-prediction receipt is durable.  Resume then
invokes the same public CLI again with transport-only overrides.  The hidden
worker from ``unified_gpu_acceptance.py`` is used only to produce a fresh
native comparison baseline when a historical baseline cannot be reused.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
UNIFIED_PATH = Path(__file__).with_name("unified_gpu_acceptance.py")
PORTABLE_PATH = Path(__file__).with_name("portable_native_gpu_acceptance.py")
DEFAULT_PROFILE = REPO_ROOT / "config" / "container_af3_smoke.yaml"
DEFAULT_PAPER_INPUT = (
    REPO_ROOT
    / "outputs"
    / "unified-gpu-acceptance"
    / "final_callers_v2"
    / "attempt_001"
    / "cases"
    / "paper-cli"
    / "paper_inputs.yaml"
)
DEFAULT_BASELINE = (
    REPO_ROOT
    / "outputs"
    / "unified-gpu-acceptance"
    / "portable_public_v5"
    / "attempt_001"
    / "cases"
    / "native-in-process"
)
PAPER_BASELINE = (
    REPO_ROOT
    / "outputs"
    / "unified-gpu-acceptance"
    / "final_callers_v2"
    / "attempt_001"
    / "references"
    / "native-pair"
)

TASK_NAME = "astrolabe-public-cli-recovery-v1"
SCHEMA_ROOT = "astrolabe.public-cli-recovery"
SEED = 17
CUT_BOUNDARY = "root/raw_predict"
CUT_EXIT = 97
COLD_GUARD_BOUNDARY = "runtime.factory"
COLD_GUARD_EXIT = 86
MAX_REQUESTS = 8
REQUESTS_PER_COMPLETE_RUN = 2
REQUESTS_PER_RECOVERY_CASE = 2
SUPPORTED_CASES = ("default-cli", "paper-cli")
NO_RETRY = True
DEFAULT_AF3_MODULE_PATH = (
    "/alphafold3_venv/lib/python3.11/site-packages/alphafold3"
)


class AcceptanceError(RuntimeError):
    """The public-CLI recovery evidence cannot be trusted."""


@dataclass(frozen=True)
class CaseSpec:
    name: str
    fixture: str
    reference: str
    historical_baseline: Path
    frontdoor: str


@dataclass(frozen=True)
class BaselineChoice:
    name: str
    reuse: bool
    reasons: tuple[str, ...]
    source_changes: tuple[Mapping[str, Any], ...]
    config_differences: tuple[Mapping[str, Any], ...]
    weights_match: bool | None
    gpu_match: bool | None
    runtime_image_match: bool | None
    rng_evidence_present: bool


def _load_module(name: str, path: Path):
    loaded = sys.modules.get(name)
    if loaded is not None:
        return loaded
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load validation helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_unified():
    return _load_module(
        "public_cli_recovery_unified_helpers",
        UNIFIED_PATH,
    )


def _load_portable():
    return _load_module(
        "public_cli_recovery_portable_helpers",
        PORTABLE_PATH,
    )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _load_unified()._atomic_json(path, dict(payload))


def _sha256(path: Path) -> str:
    return _load_unified()._sha256(path)


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


def _case_spec(name: str) -> CaseSpec:
    if name == "default-cli":
        return CaseSpec(
            name=name,
            fixture="single-20aa",
            reference="native-single",
            historical_baseline=DEFAULT_BASELINE,
            frontdoor="python -m astrolabe.cli run --standalone",
        )
    if name == "paper-cli":
        return CaseSpec(
            name=name,
            fixture="pair-shared-fixed",
            reference="native-pair",
            historical_baseline=PAPER_BASELINE,
            frontdoor=(
                "python -m astrolabe.cli paper run evaluate-af3 "
                "--inputs <original-paper-inputs> --overrides <transport-only>"
            ),
        )
    raise ValueError(f"unsupported public CLI recovery case {name!r}")


def _selected_cases(names: Sequence[str] | None = None) -> tuple[str, ...]:
    if names is None:
        return SUPPORTED_CASES
    requested = tuple(names)
    if not requested:
        raise AcceptanceError("at least one public CLI case must be selected")
    unknown = sorted(set(requested) - set(SUPPORTED_CASES))
    if unknown:
        raise AcceptanceError(f"unsupported public CLI cases: {unknown}")
    if len(set(requested)) != len(requested):
        raise AcceptanceError("public CLI cases must not repeat")
    selected = set(requested)
    return tuple(name for name in SUPPORTED_CASES if name in selected)


def _maximum_selected_request_cap(names: Sequence[str]) -> int:
    selected = _selected_cases(names)
    if selected == SUPPORTED_CASES:
        return MAX_REQUESTS
    return (
        REQUESTS_PER_RECOVERY_CASE * len(selected)
        + (REQUESTS_PER_COMPLETE_RUN if "paper-cli" in selected else 0)
    )


def _bound_case_specs(
    *,
    default_baseline: Path,
    paper_baseline: Path,
    selected_cases: Sequence[str] | None = None,
) -> dict[str, CaseSpec]:
    available = {
        "default-cli": replace(
            _case_spec("default-cli"),
            historical_baseline=default_baseline.expanduser().resolve(),
        ),
        "paper-cli": replace(
            _case_spec("paper-cli"),
            historical_baseline=paper_baseline.expanduser().resolve(),
        ),
    }
    return {
        name: available[name]
        for name in _selected_cases(selected_cases)
    }


def request_budget(
    *,
    fresh_default: bool,
    fresh_paper: bool,
    cases: Sequence[str] | None = None,
) -> int:
    """Return the merged-freeze budget; default baseline recreation is forbidden."""

    if fresh_default:
        raise AcceptanceError(
            "merged mode requires the default native-in-process baseline from "
            "the same portable capsule; refusing to recreate it"
        )
    selected = _selected_cases(cases)
    budget = REQUESTS_PER_RECOVERY_CASE * len(selected)
    if fresh_paper and "paper-cli" in selected:
        budget += REQUESTS_PER_COMPLETE_RUN
    if budget > MAX_REQUESTS:
        raise AcceptanceError(
            f"request budget {budget} exceeds hard cap {MAX_REQUESTS}"
        )
    return budget


class RequestBudget:
    """Durably reserve model requests before each child process is launched."""

    def __init__(self, path: Path, *, limit: int = MAX_REQUESTS) -> None:
        self.path = path
        self.limit = limit
        self.reservations: list[dict[str, Any]] = []
        self.used = 0
        self._write(status="initialized")

    def _write(self, *, status: str) -> None:
        _atomic_json(
            self.path,
            {
                "schema": f"{SCHEMA_ROOT}.request-budget/v1",
                "task_name": TASK_NAME,
                "status": status,
                "limit": self.limit,
                "reserved_before_launch": self.used,
                "remaining": self.limit - self.used,
                "no_retry": NO_RETRY,
                "reservations": self.reservations,
            },
        )

    def reserve(self, label: str, count: int) -> None:
        if not isinstance(count, int) or count < 1:
            raise ValueError("request reservation must be a positive integer")
        if self.used + count > self.limit:
            raise AcceptanceError(
                f"request budget exceeded before launching {label!r}: "
                f"reserved={self.used} requested={count} limit={self.limit}"
            )
        self.reservations.append(
            {
                "label": label,
                "requests": count,
                "status": "reserved-before-launch",
            }
        )
        self.used += count
        self._write(status="reserved")

    def record_observed(self, label: str, observed: int) -> None:
        if observed < 0 or observed > self._reserved_for(label):
            raise AcceptanceError(
                f"observed request count for {label!r} exceeds its hard reservation: "
                f"observed={observed}"
            )
        for row in self.reservations:
            if row["label"] == label:
                row["observed"] = observed
                row["status"] = "observed"
                break
        self._write(status="observed")

    def _reserved_for(self, label: str) -> int:
        amount = sum(
            int(row["requests"])
            for row in self.reservations
            if row["label"] == label
        )
        if amount < 1:
            raise KeyError(label)
        return amount


def _transport_overrides(*, out_dir: Path, resume: bool) -> str:
    tokens = (
        "recovery.mode=durable",
        f"recovery.resume={'true' if resume else 'false'}",
        f"out_dir={out_dir}",
        "skip_complex_poses=true",
    )
    return shlex.join(tokens)


def public_cli_command(
    *,
    case: CaseSpec,
    python_bin: Path,
    config_path: Path | None,
    paper_input: Path | None,
    task_name: str,
    out_dir: Path,
    resume: bool,
    cli_out_dir: Path | None = None,
    dry_run: bool = False,
) -> list[str]:
    """Build the real public CLI command; no validation worker entrypoint is used."""

    overrides = _transport_overrides(out_dir=out_dir, resume=resume)
    if case.name == "default-cli":
        if config_path is None:
            raise ValueError("default-cli requires config_path")
        command = [
            str(python_bin),
            "-m",
            "astrolabe.cli",
            "run",
            "--standalone",
            str(config_path),
            "--no-chdir",
            "--overrides",
            overrides,
        ]
    elif case.name == "paper-cli":
        if paper_input is None:
            raise ValueError("paper-cli requires paper_input")
        command = [
            str(python_bin),
            "-m",
            "astrolabe.cli",
            "paper",
            "run",
            "evaluate-af3",
            "--inputs",
            str(paper_input),
            "--config-env",
            "env_container",
            "--task-name",
            task_name,
            "--out-dir",
            str(cli_out_dir or out_dir),
            "--seed",
            str(SEED),
            "--num-runs",
            "1",
            "--overrides",
            overrides,
        ]
    else:
        raise ValueError(case.name)
    if dry_run:
        command.extend(("--dry-run", "--print-config"))
    if "--internal-worker" in command or "--api-worker" in command:
        raise AssertionError("recovery command must not use a validation worker")
    return command


def _write_instrumented_sitecustomize(case_root: Path, mode: str) -> Path:
    if mode not in {"cut", "cold-guard"}:
        raise ValueError(f"unsupported CLI instrumentation mode {mode!r}")
    bootstrap = case_root / f"{mode}-instrumentation"
    bootstrap.mkdir(parents=True, exist_ok=False)
    capture_sitecustomize = case_root / "instrumentation" / "sitecustomize.py"
    text = f'''\
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys


def _load_capture():
    path = Path({str(capture_sitecustomize)!r})
    spec = importlib.util.spec_from_file_location(
        "_astrolabe_public_cli_capture_sitecustomize",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load capture sitecustomize: {{path}}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)


def _write_marker(path, boundary, **extra):
    payload = {{
        "schema": "astrolabe.public-cli-recovery.boundary-marker/v1",
        "task_name": {TASK_NAME!r},
        "boundary": boundary,
        "pid": os.getpid(),
        **extra,
    }}
    encoded = (json.dumps(payload, sort_keys=True) + "\\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        offset = 0
        while offset < len(encoded):
            offset += os.write(descriptor, encoded[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _install_cut():
    marker_text = os.environ.get("ASTROLABE_PUBLIC_CLI_CUT_MARKER")
    boundary = os.environ.get("ASTROLABE_PUBLIC_CLI_CUT_BOUNDARY")
    if not marker_text or not boundary:
        return
    if boundary != {CUT_BOUNDARY!r}:
        raise RuntimeError(f"Unsupported public CLI cut boundary: {{boundary}}")

    from astrolabe.design.runtime import Runtime

    marker = Path(marker_text)
    marker.parent.mkdir(parents=True, exist_ok=True)
    original = Runtime.call
    fired = False

    def cut_after_durable_return(
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
            _write_marker(marker, boundary)
            os._exit({CUT_EXIT})
        return result

    Runtime.call = cut_after_durable_return


def _install_cold_guard():
    marker_text = os.environ.get("ASTROLABE_PUBLIC_CLI_COLD_GUARD_MARKER")
    boundary = os.environ.get("ASTROLABE_PUBLIC_CLI_COLD_GUARD_BOUNDARY")
    if not marker_text or not boundary:
        return
    if boundary != {COLD_GUARD_BOUNDARY!r}:
        raise RuntimeError(f"Unsupported public CLI cold boundary: {{boundary}}")

    from astrolabe.design.runtime import Runtime

    marker = Path(marker_text)
    marker.parent.mkdir(parents=True, exist_ok=True)
    original = Runtime.__init__

    def cold_after_runtime_factory(self, *args, **kwargs):
        original(self, *args, **kwargs)
        _write_marker(
            marker,
            boundary,
            runtime_root=str(self.root) if self.root is not None else None,
            operation_aliases=sorted(self.operations),
            scientific_contract=self.scientific_contract.digest,
            operation_identities={{
                alias: component.descriptor.identity.digest
                for alias, component in sorted(self.operations.items())
            }},
            budgets=dict(self.budgets),
        )
        os._exit({COLD_GUARD_EXIT})

    Runtime.__init__ = cold_after_runtime_factory


_load_capture()
{"_install_cut()" if mode == "cut" else "_install_cold_guard()"}
'''
    (bootstrap / "sitecustomize.py").write_text(text, encoding="utf-8")
    return bootstrap


def _write_cut_sitecustomize(case_root: Path) -> Path:
    return _write_instrumented_sitecustomize(case_root, "cut")


def _capture_environment(
    case_root: Path,
    *,
    mode: str,
) -> tuple[Path, Path, dict[str, str]]:
    if mode not in {"cut", "cold-guard", "resume"}:
        raise ValueError(f"unsupported capture mode {mode!r}")
    unified = _load_unified()
    capture, claims, env = unified._install_capture_environment(case_root)
    result_env = dict(env)
    if mode != "resume":
        bootstrap = _write_instrumented_sitecustomize(case_root, mode)
        existing = result_env.get("PYTHONPATH")
        result_env["PYTHONPATH"] = os.pathsep.join(
            part
            for part in (str(bootstrap), existing or "")
            if part
        )
        if mode == "cut":
            marker = case_root / "cut-marker.json"
            result_env["ASTROLABE_PUBLIC_CLI_CUT_MARKER"] = str(marker)
            result_env["ASTROLABE_PUBLIC_CLI_CUT_BOUNDARY"] = CUT_BOUNDARY
        else:
            marker = case_root / "cold-guard-marker.json"
            result_env["ASTROLABE_PUBLIC_CLI_COLD_GUARD_MARKER"] = str(marker)
            result_env["ASTROLABE_PUBLIC_CLI_COLD_GUARD_BOUNDARY"] = (
                COLD_GUARD_BOUNDARY
            )
    return capture, claims, result_env


def _config_payload(cfg: Any) -> dict[str, Any]:
    from omegaconf import OmegaConf

    value = OmegaConf.to_container(cfg, resolve=False)
    if not isinstance(value, dict):
        raise AcceptanceError("composed configuration must be a mapping")
    value.pop("defaults", None)
    return value


def _baseline_scientific_projection(cfg: Any) -> dict[str, Any]:
    """Compare all non-runtime config fields without resolving unused backends."""

    payload = _config_payload(cfg)
    for key in (
        "task_name",
        "out_dir",
        "repeat_index",
        "raw_model_results_dir",
    ):
        payload.pop(key, None)
    return payload


def validate_resume_override_diff(
    *,
    persisted_cfg: Any,
    resume_cfg: Any,
    persisted_root: Path,
) -> dict[str, Any]:
    """Accept only out_dir/recovery-resume transport changes."""

    persisted_task = str(persisted_cfg.get("task_name"))
    resume_task = str(resume_cfg.get("task_name"))
    if persisted_task != resume_task:
        raise AcceptanceError(
            "resume changed task_name: "
            f"{persisted_task!r} -> {resume_task!r}"
        )
    recovery = resume_cfg.get("recovery")
    if recovery is None:
        raise AcceptanceError("resume CLI did not add recovery configuration")
    if str(recovery.get("mode")) != "durable":
        raise AcceptanceError("resume recovery.mode must be durable")
    if recovery.get("resume") is not True:
        raise AcceptanceError("resume recovery.resume must be true")
    resumed_out_dir = Path(str(resume_cfg.get("out_dir"))).expanduser().resolve()
    if resumed_out_dir != persisted_root.resolve():
        raise AcceptanceError(
            "resume out_dir is not the exact persisted run root: "
            f"configured={resumed_out_dir} persisted={persisted_root.resolve()}"
        )

    unified = _load_unified()
    raw_left = _config_payload(persisted_cfg)
    raw_right = _config_payload(resume_cfg)
    raw_differences = unified._config_differences(raw_left, raw_right)
    return {
        "schema": f"{SCHEMA_ROOT}.resume-config-diff/v1",
        "status": "PENDING_CANONICAL_RUNTIME_CHECK",
        "task_name": persisted_task,
        "out_dir": str(resumed_out_dir),
        "recovery": {"mode": "durable", "resume": True},
        "raw_config_differences": raw_differences,
        "allowed_transport_changes": [
            "out_dir",
            "recovery.resume",
            "repeat_index",
            "raw_model_results_dir",
        ],
    }


def validate_command_transport_diff(
    *,
    initial_command: Sequence[str],
    resume_command: Sequence[str],
) -> dict[str, Any]:
    """Require identical public CLI arguments except transport overrides."""

    initial = list(initial_command)
    resume = list(resume_command)
    if initial.count("--overrides") != 1 or resume.count("--overrides") != 1:
        raise AcceptanceError(
            "cold guard commands must each contain exactly one --overrides"
        )
    initial_index = initial.index("--overrides")
    resume_index = resume.index("--overrides")
    initial_overrides = shlex.split(initial[initial_index + 1])
    resume_overrides = shlex.split(resume[resume_index + 1])

    def parse(tokens: Sequence[str]) -> dict[str, str]:
        values = {}
        for token in tokens:
            key, separator, value = token.partition("=")
            if not separator or not key:
                raise AcceptanceError(f"invalid override token: {token!r}")
            values[key] = value
        return values

    before = list(initial)
    after = list(resume)
    before[initial_index + 1] = "<overrides>"
    after[resume_index + 1] = "<overrides>"
    if before != after:
        differences = [
            {"index": index, "initial": left, "resume": right}
            for index, (left, right) in enumerate(zip(before, after))
            if left != right
        ]
        raise AcceptanceError(
            "initial and resume public CLI commands differ outside overrides: "
            f"{differences}"
        )
    initial_values = parse(initial_overrides)
    resume_values = parse(resume_overrides)
    changed = {
        key
        for key in set(initial_values) | set(resume_values)
        if initial_values.get(key) != resume_values.get(key)
    }
    allowed = {"out_dir", "recovery.resume"}
    if changed != allowed:
        raise AcceptanceError(
            "resume override diff must be exactly out_dir and "
            f"recovery.resume; observed={sorted(changed)}"
        )
    return {
        "schema": f"{SCHEMA_ROOT}.command-transport-diff/v1",
        "status": "PASS",
        "changed_overrides": sorted(changed),
        "initial_overrides": initial_values,
        "resume_overrides": resume_values,
        "all_non_override_argv_equal": True,
    }


def _case_config(
    *,
    case: CaseSpec,
    profile: Path,
    out_dir: Path,
    task_name: str,
):
    unified = _load_unified()
    return unified._build_config(
        profile=profile,
        fixture=case.fixture,
        out_dir=out_dir,
        task_name=task_name,
        execution="daemon",
        generic_search=False,
        seed=SEED,
    )


def _baseline_config(
    *,
    case: CaseSpec,
    profile: Path,
    out_dir: Path,
):
    unified = _load_unified()
    return unified._build_config(
        profile=profile,
        fixture=case.fixture,
        out_dir=out_dir,
        task_name=f"{TASK_NAME}-baseline-{case.name}",
        execution="in_process",
        generic_search=False,
        seed=SEED,
    )


def _historical_config_path(root: Path) -> Path:
    for name in ("config.yaml", "input.yaml"):
        path = root / name
        if path.is_file():
            return path
    return root / "config.yaml"


def _baseline_choice(
    *,
    case: CaseSpec,
    profile: Path,
    environment: Mapping[str, Any] | None,
    af3_module_path: Path,
) -> BaselineChoice:
    unified = _load_unified()
    root = case.historical_baseline.expanduser().resolve()
    reasons: list[str] = []
    source_changes: tuple[Mapping[str, Any], ...] = ()
    config_differences: tuple[Mapping[str, Any], ...] = ()
    weights_match: bool | None = None
    gpu_match: bool | None = None
    runtime_image_match: bool | None = None
    rng_evidence_present = (
        (root / "evidence" / "rng_trace.json").is_file()
        and (root / "evidence" / "rng_trace.npz").is_file()
    )

    result_path = root / "result.json"
    config_path = _historical_config_path(root)
    if not result_path.is_file() or not config_path.is_file():
        reasons.append("historical baseline result/config is missing")
        return BaselineChoice(
            case.reference,
            False,
            tuple(reasons),
            source_changes,
            config_differences,
            weights_match,
            gpu_match,
            runtime_image_match,
            rng_evidence_present,
        )

    result = unified._load_json(result_path)
    if result.get("status") != "evidence-complete":
        reasons.append("historical baseline is not evidence-complete")
    if result.get("model_calls") != REQUESTS_PER_COMPLETE_RUN:
        reasons.append(
            "historical baseline does not contain exactly "
            f"{REQUESTS_PER_COMPLETE_RUN} evaluator requests"
        )
    if not rng_evidence_present:
        reasons.append("historical baseline RNG evidence is missing")

    from omegaconf import OmegaConf

    baseline_cfg = OmegaConf.load(config_path)
    current_cfg = _baseline_config(
        case=case,
        profile=profile,
        out_dir=root / "scientific-config-check",
    )
    baseline_projection = unified._comparable_scientific_config(
        result.get("config")
    )
    current_projection = unified._comparable_scientific_config(
        unified._digest_config(current_cfg)
    )
    config_differences = tuple(
        unified._config_differences(
            baseline_projection,
            current_projection,
        )
    )
    if config_differences:
        reasons.append("historical baseline scientific config differs")
    baseline_file_projection = _baseline_scientific_projection(baseline_cfg)
    current_file_projection = _baseline_scientific_projection(current_cfg)
    file_config_differences = unified._config_differences(
        baseline_file_projection,
        current_file_projection,
    )
    if file_config_differences:
        config_differences = tuple(
            [*config_differences, *file_config_differences]
        )
        reasons.append("historical baseline full config differs")

    environment_payload = result.get("environment_manifest")
    if not isinstance(environment_payload, Mapping):
        reasons.append("historical baseline environment manifest is missing")
    else:
        embedded_source = environment_payload.get("source_manifest")
        if not isinstance(embedded_source, Mapping):
            reasons.append("historical baseline numerical source manifest is missing")
        else:
            include_af3_vendor = not str(af3_module_path).startswith(
                "/alphafold3_venv/"
            )
            baseline_source = unified._source_manifest_from_payload(
                embedded_source,
                include_af3_vendor=include_af3_vendor,
            )
            current_numerical = unified._read_source_manifest(
                numerical_only=True,
                include_af3_vendor=include_af3_vendor,
            )
            source_changes = tuple(
                unified._source_changes(baseline_source, current_numerical)
            )
            if source_changes:
                reasons.append("historical baseline numerical source differs")
        if environment is not None:
            weights_match = unified._scientific_equal(
                environment_payload.get("weights"),
                environment.get("weights"),
            )
            gpu_match = unified._scientific_equal(
                environment_payload.get("gpu"),
                environment.get("gpu"),
            )
            if not weights_match:
                reasons.append("historical baseline AF3 weights differ")
            if not gpu_match:
                reasons.append("historical baseline GPU allocation differs")
            baseline_runtime = (
                environment_payload.get("environment")
                if isinstance(environment_payload.get("environment"), Mapping)
                else {}
            )
            current_runtime = (
                environment.get("environment")
                if isinstance(environment.get("environment"), Mapping)
                else {}
            )
            runtime_keys = ("APPTAINER_CONTAINER", "SINGULARITY_CONTAINER")
            runtime_image_match = unified._scientific_equal(
                {key: baseline_runtime.get(key) for key in runtime_keys},
                {key: current_runtime.get(key) for key in runtime_keys},
            )
            if not runtime_image_match:
                reasons.append("historical baseline model runtime image differs")
        else:
            pass

    return BaselineChoice(
        name=case.reference,
        reuse=not reasons,
        reasons=tuple(reasons),
        source_changes=source_changes,
        config_differences=config_differences,
        weights_match=weights_match,
        gpu_match=gpu_match,
        runtime_image_match=runtime_image_match,
        rng_evidence_present=rng_evidence_present,
    )


def _copy_reused_baseline(source: Path, destination: Path) -> None:
    if destination.exists():
        raise AcceptanceError(
            f"refusing to overwrite adopted baseline: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("result.json",):
        source_path = source / name
        if not source_path.is_file():
            raise AcceptanceError(f"baseline file is missing: {source_path}")
        shutil.copy2(source_path, destination / name)
    config_source = _historical_config_path(source)
    if not config_source.is_file():
        raise AcceptanceError(
            f"baseline scientific config is missing: {config_source}"
        )
    shutil.copy2(config_source, destination / "config.yaml")
    evidence = source / "evidence"
    if not evidence.is_dir():
        raise AcceptanceError(f"baseline evidence directory is missing: {evidence}")
    shutil.copytree(evidence, destination / "evidence")


def _fresh_baseline(
    *,
    case: CaseSpec,
    profile: Path,
    out_root: Path,
    environment: Mapping[str, Any],
    python_bin: Path,
    timeout_s: float,
    budget: RequestBudget,
):
    unified = _load_unified()
    reference_root = out_root / "references" / f"{case.reference}-fresh"
    reference_root.mkdir(parents=True, exist_ok=False)
    config = _baseline_config(
        case=case,
        profile=profile,
        out_dir=reference_root / "run",
    )
    config_path = reference_root / "config.yaml"
    unified._save_standalone(config, config_path)
    capture, _claims, env = unified._install_capture_environment(reference_root)
    worker_mode = (
        "native-generic"
        if case.reference == "native-generic"
        else "native-reference"
    )
    command = [
        str(python_bin),
        str(UNIFIED_PATH),
        "--api-worker",
        "--mode",
        worker_mode,
        "--config",
        str(config_path),
        "--fixture",
        case.fixture,
    ]
    budget.reserve(f"fresh-baseline:{case.reference}", 2)
    process = unified._run_process(
        command,
        cwd=REPO_ROOT,
        env=env,
        stdout_log=reference_root / "stdout.log",
        stderr_log=reference_root / "stderr.log",
        timeout_s=timeout_s,
    )
    plan = unified.CasePlan(
        name=case.reference,
        reference="none",
        frontdoor=(
            "native in-process root and one advance; builder not executed"
        ),
        fixture=case.fixture,
    )
    try:
        result = unified._write_case_evidence(
            case_root=reference_root,
            run_root=reference_root / "run",
            process=process,
            capture_path=capture,
            env_manifest=environment,
            plan=plan,
            config=config,
        )
    except Exception as error:
        result = unified._write_evidence_failure(
            case_root=reference_root,
            process=process,
            capture_path=capture,
            env_manifest=environment,
            plan=plan,
            config=config,
            error=error,
        )
    budget.record_observed(
        f"fresh-baseline:{case.reference}",
        int(result.get("model_calls") or 0),
    )
    if result.get("status") != "evidence-complete":
        raise AcceptanceError(
            f"fresh baseline {case.reference} failed: "
            f"{result.get('error')}"
        )
    _atomic_json(
        reference_root / "commands.json",
        {
            "schema": f"{SCHEMA_ROOT}.fresh-baseline-command/v1",
            "case": case.reference,
            "command": command,
            "role": (
                "Current-environment native comparison baseline only; "
                "this is not public-CLI recovery evidence."
            ),
        },
    )
    return reference_root


def _dry_run_config(
    *,
    command: Sequence[str],
    env: Mapping[str, str],
    stdout_log: Path,
    stderr_log: Path,
    timeout_s: float,
) -> Any:
    unified = _load_unified()
    process = unified._run_process(
        command,
        cwd=REPO_ROOT,
        env=env,
        stdout_log=stdout_log,
        stderr_log=stderr_log,
        timeout_s=timeout_s,
    )
    if process.returncode != 0 or process.timed_out:
        raise AcceptanceError(
            "public CLI dry-run preflight failed: "
            f"rc={process.returncode} timeout={process.timed_out} "
            f"stderr={stderr_log}"
        )
    from omegaconf import OmegaConf

    return OmegaConf.load(stdout_log)


def _run_recovery_case(
    *,
    case: CaseSpec,
    profile: Path,
    paper_input: Path,
    out_root: Path,
    environment: Mapping[str, Any],
    python_bin: Path,
    timeout_s: float,
    budget: RequestBudget,
    source_guard,
) -> tuple[dict[str, Any], Path]:
    unified = _load_unified()
    portable = _load_portable()
    case_root = out_root / "cases" / case.name
    case_root.mkdir(parents=True, exist_ok=False)
    task_name = f"{TASK_NAME}-{case.name}"
    initial_out = case_root / "output"
    config_path: Path | None = None
    if case.name == "default-cli":
        initial_cfg = _case_config(
            case=case,
            profile=profile,
            out_dir=initial_out,
            task_name=task_name,
        )
        config_path = case_root / "input.yaml"
        unified._save_standalone(initial_cfg, config_path)
    else:
        initial_cfg = _case_config(
            case=case,
            profile=profile,
            out_dir=initial_out,
            task_name=task_name,
        )
    paper_before = _sha256(paper_input) if case.name == "paper-cli" else None

    capture, claims, base_env = _capture_environment(case_root, mode="cut")
    marker = case_root / "cut-marker.json"
    cut_command = public_cli_command(
        case=case,
        python_bin=python_bin,
        config_path=config_path,
        paper_input=paper_input if case.name == "paper-cli" else None,
        task_name=task_name,
        out_dir=initial_out,
        resume=False,
    )
    budget.reserve(f"{case.name}:cut", 1)
    cut_process = unified._run_process(
        cut_command,
        cwd=REPO_ROOT,
        env=base_env,
        stdout_log=case_root / "cut.stdout.log",
        stderr_log=case_root / "cut.stderr.log",
        timeout_s=timeout_s,
    )
    if cut_process.timed_out or cut_process.returncode != CUT_EXIT:
        observed = len(unified._read_capture_records(capture))
        budget.record_observed(f"{case.name}:cut", observed)
        raise AcceptanceError(
            "public CLI cut did not terminate at the declared durable boundary: "
            f"rc={cut_process.returncode} timeout={cut_process.timed_out}"
        )
    if not marker.is_file():
        budget.record_observed(f"{case.name}:cut", 0)
        raise AcceptanceError("public CLI cut exited without writing its marker")
    cut_records = unified._read_capture_records(capture)
    budget.record_observed(f"{case.name}:cut", len(cut_records))
    if len(cut_records) != 1 or int(cut_records[0].get("step")) != 0:
        raise AcceptanceError(
            "public CLI cut must issue exactly root step 0 before exit; "
            f"observed={cut_records}"
        )
    if case.name == "paper-cli" and _sha256(paper_input) != paper_before:
        raise AcceptanceError("public paper CLI modified its original --inputs file")

    persisted_root = portable._run_persisted_run(initial_out)
    config_before = _sha256(persisted_root / "config.yaml")
    failed_metadata = portable._snapshot_failed_metadata(
        persisted_root,
        case_root / "cut_before",
    )
    source_guard(f"case:{case.name}:before-resume")

    resume_command = public_cli_command(
        case=case,
        python_bin=python_bin,
        config_path=config_path,
        paper_input=paper_input if case.name == "paper-cli" else None,
        task_name=task_name,
        out_dir=persisted_root,
        resume=True,
        cli_out_dir=initial_out,
    )
    resume_preflight_command = public_cli_command(
        case=case,
        python_bin=python_bin,
        config_path=config_path,
        paper_input=paper_input if case.name == "paper-cli" else None,
        task_name=task_name,
        out_dir=persisted_root,
        resume=True,
        cli_out_dir=initial_out,
        dry_run=True,
    )
    _, _, resume_env = _capture_environment(case_root, mode="resume")
    from omegaconf import OmegaConf

    persisted_cfg = OmegaConf.load(persisted_root / "config.yaml")
    resume_cfg = _dry_run_config(
        command=resume_preflight_command,
        env=resume_env,
        stdout_log=case_root / "resume-preflight.stdout.yaml",
        stderr_log=case_root / "resume-preflight.stderr.log",
        timeout_s=timeout_s,
    )
    resume_diff = validate_resume_override_diff(
        persisted_cfg=persisted_cfg,
        resume_cfg=resume_cfg,
        persisted_root=persisted_root,
    )
    _atomic_json(case_root / "resume_config_diff.json", resume_diff)

    budget.reserve(f"{case.name}:resume", 1)
    resume_process = unified._run_process(
        resume_command,
        cwd=REPO_ROOT,
        env=resume_env,
        stdout_log=case_root / "resume.stdout.log",
        stderr_log=case_root / "resume.stderr.log",
        timeout_s=timeout_s,
    )
    source_guard(f"case:{case.name}:after-resume")
    config_after = _sha256(persisted_root / "config.yaml")
    if config_after != config_before:
        budget.record_observed(f"{case.name}:resume", 0)
        raise AcceptanceError("resume modified the failed run's config.yaml")
    all_records = unified._read_capture_records(capture)
    resume_records = all_records[len(cut_records) :]
    budget.record_observed(f"{case.name}:resume", len(resume_records))
    accounting = portable.validate_raw_cut_accounting(
        cut_records,
        resume_records,
    )
    if case.name == "paper-cli" and _sha256(paper_input) != paper_before:
        raise AcceptanceError("public paper resume modified its original --inputs file")

    plan = unified.CasePlan(
        name=case.name,
        reference=case.reference,
        frontdoor=case.frontdoor,
        fixture=case.fixture,
    )
    try:
        result = unified._write_case_evidence(
            case_root=case_root,
            run_root=persisted_root,
            process=resume_process,
            capture_path=capture,
            env_manifest=environment,
            plan=plan,
            config=initial_cfg,
        )
    except Exception as error:
        result = unified._write_evidence_failure(
            case_root=case_root,
            process=resume_process,
            capture_path=capture,
            env_manifest=environment,
            plan=plan,
            config=initial_cfg,
            error=error,
        )
    result.update(
        {
            "cut_boundary": CUT_BOUNDARY,
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
            "failed_metadata_snapshot": str(case_root / "cut_before"),
            "failed_metadata": failed_metadata,
            "failed_config_sha256_before": config_before,
            "failed_config_sha256_after": config_after,
            "resume_config_diff": resume_diff,
            "paper_input_sha256": paper_before,
            "no_retry": NO_RETRY,
            "commands": {
                "cut": cut_command,
                "resume_preflight": resume_preflight_command,
                "resume": resume_command,
            },
        }
    )
    _atomic_json(case_root / "result.json", result)
    return result, case_root


def _run_cold_guard_case(
    *,
    case: CaseSpec,
    profile: Path,
    paper_input: Path,
    out_root: Path,
    python_bin: Path,
    timeout_s: float,
    source_guard,
) -> dict[str, Any]:
    """Prove initial and resume CLI paths reach Runtime with zero inference."""

    unified = _load_unified()
    portable = _load_portable()
    case_root = out_root / "cold_guard" / case.name
    case_root.mkdir(parents=True, exist_ok=False)
    task_name = f"{TASK_NAME}-cold-{case.name}"
    output = case_root / "output"
    config_path: Path | None = None
    if case.name == "default-cli":
        cfg = _case_config(
            case=case,
            profile=profile,
            out_dir=output,
            task_name=task_name,
        )
        config_path = case_root / "input.yaml"
        unified._save_standalone(cfg, config_path)
    paper_before = _sha256(paper_input) if case.name == "paper-cli" else None
    command = public_cli_command(
        case=case,
        python_bin=python_bin,
        config_path=config_path,
        paper_input=paper_input if case.name == "paper-cli" else None,
        task_name=task_name,
        out_dir=output,
        resume=False,
    )
    capture, initial_claims, env = _capture_environment(
        case_root,
        mode="cold-guard",
    )
    initial_marker = case_root / "cold-guard-marker.json"
    source_guard(f"cold-guard:{case.name}:before")
    initial_process = unified._run_process(
        command,
        cwd=REPO_ROOT,
        env=env,
        stdout_log=case_root / "stdout.log",
        stderr_log=case_root / "stderr.log",
        timeout_s=timeout_s,
    )
    source_guard(f"cold-guard:{case.name}:after")
    initial_records = unified._read_capture_records(capture)
    if (
        initial_process.timed_out
        or initial_process.returncode != COLD_GUARD_EXIT
    ):
        raise AcceptanceError(
            "public CLI cold guard did not stop at Runtime factory: "
            f"case={case.name} rc={initial_process.returncode} "
            f"timeout={initial_process.timed_out}"
        )
    if initial_records:
        raise AcceptanceError(
            "public CLI initial cold guard issued model requests: "
            f"{len(initial_records)}"
        )
    if not initial_marker.is_file():
        raise AcceptanceError(
            f"public CLI cold guard did not write Runtime marker: {case.name}"
        )
    initial_marker_payload = unified._load_json(initial_marker)
    if initial_marker_payload.get("boundary") != COLD_GUARD_BOUNDARY:
        raise AcceptanceError(
            "public CLI cold guard reached wrong boundary: "
            f"{initial_marker_payload}"
        )
    aliases = initial_marker_payload.get("operation_aliases")
    if not isinstance(aliases, list) or not aliases:
        raise AcceptanceError(
            "Runtime factory cold guard recorded no operations: "
            f"{initial_marker_payload}"
        )

    persisted_root = portable._run_persisted_run(output)
    from omegaconf import OmegaConf

    persisted_cfg = OmegaConf.load(persisted_root / "config.yaml")
    resume_case_root = case_root / "resume"
    resume_case_root.mkdir(parents=True, exist_ok=False)
    resume_command = public_cli_command(
        case=case,
        python_bin=python_bin,
        config_path=config_path,
        paper_input=paper_input if case.name == "paper-cli" else None,
        task_name=task_name,
        out_dir=persisted_root,
        resume=True,
        cli_out_dir=output,
    )
    command_transport_diff = validate_command_transport_diff(
        initial_command=command,
        resume_command=resume_command,
    )
    _atomic_json(
        case_root / "command_transport_diff.json",
        command_transport_diff,
    )
    resume_preflight_command = public_cli_command(
        case=case,
        python_bin=python_bin,
        config_path=config_path,
        paper_input=paper_input if case.name == "paper-cli" else None,
        task_name=task_name,
        out_dir=persisted_root,
        resume=True,
        cli_out_dir=output,
        dry_run=True,
    )
    _, _, resume_preflight_env = _capture_environment(
        resume_case_root,
        mode="resume",
    )
    resume_cfg = _dry_run_config(
        command=resume_preflight_command,
        env=resume_preflight_env,
        stdout_log=resume_case_root / "preflight.stdout.yaml",
        stderr_log=resume_case_root / "preflight.stderr.log",
        timeout_s=timeout_s,
    )
    resume_diff = validate_resume_override_diff(
        persisted_cfg=persisted_cfg,
        resume_cfg=resume_cfg,
        persisted_root=persisted_root,
    )
    _atomic_json(resume_case_root / "resume_config_diff.json", resume_diff)

    resume_capture, resume_claims, resume_env = _capture_environment(
        resume_case_root,
        mode="cold-guard",
    )
    resume_marker = resume_case_root / "cold-guard-marker.json"
    source_guard(f"cold-guard:{case.name}:resume-before")
    resume_process = unified._run_process(
        resume_command,
        cwd=REPO_ROOT,
        env=resume_env,
        stdout_log=resume_case_root / "stdout.log",
        stderr_log=resume_case_root / "stderr.log",
        timeout_s=timeout_s,
    )
    source_guard(f"cold-guard:{case.name}:resume-after")
    resume_records = unified._read_capture_records(resume_capture)
    if resume_process.timed_out or resume_process.returncode != COLD_GUARD_EXIT:
        raise AcceptanceError(
            "public CLI resume cold guard did not stop at Runtime factory: "
            f"case={case.name} rc={resume_process.returncode} "
            f"timeout={resume_process.timed_out}"
        )
    if resume_records:
        raise AcceptanceError(
            "public CLI resume cold guard issued model requests: "
            f"{len(resume_records)}"
        )
    if not resume_marker.is_file():
        raise AcceptanceError(
            f"public CLI resume cold guard did not write marker: {case.name}"
        )
    resume_marker_payload = unified._load_json(resume_marker)
    if resume_marker_payload.get("boundary") != COLD_GUARD_BOUNDARY:
        raise AcceptanceError(
            "public CLI resume cold guard reached wrong boundary: "
            f"{resume_marker_payload}"
        )
    contract_keys = (
        "runtime_root",
        "operation_aliases",
        "scientific_contract",
        "operation_identities",
        "budgets",
    )
    initial_contract = {
        key: initial_marker_payload.get(key) for key in contract_keys
    }
    resume_contract = {
        key: resume_marker_payload.get(key) for key in contract_keys
    }
    if not unified._scientific_equal(initial_contract, resume_contract):
        raise AcceptanceError(
            "initial and resume public CLI Runtime contracts differ: "
            f"initial={initial_contract} resume={resume_contract}"
        )
    resume_diff = {
        **resume_diff,
        "status": "PASS",
        "canonical_runtime_contract_equal": True,
        "canonical_scientific_contract": initial_contract[
            "scientific_contract"
        ],
    }
    _atomic_json(resume_case_root / "resume_config_diff.json", resume_diff)
    if case.name == "paper-cli" and _sha256(paper_input) != paper_before:
        raise AcceptanceError(
            "paper cold guard or resume modified its original --inputs file"
        )
    result = {
        "schema": f"{SCHEMA_ROOT}.cold-guard/v1",
        "status": "PASS",
        "case": case.name,
        "frontdoor": case.frontdoor,
        "boundary": COLD_GUARD_BOUNDARY,
        "persisted_root": str(persisted_root),
        "commands": {
            "initial": command,
            "resume_preflight": resume_preflight_command,
            "resume": resume_command,
        },
        "initial_process": {
            "returncode": initial_process.returncode,
            "timed_out": initial_process.timed_out,
            "duration_s": initial_process.duration_s,
            "stdout_log": str(initial_process.stdout_log),
            "stderr_log": str(initial_process.stderr_log),
        },
        "resume_process": {
            "returncode": resume_process.returncode,
            "timed_out": resume_process.timed_out,
            "duration_s": resume_process.duration_s,
            "stdout_log": str(resume_process.stdout_log),
            "stderr_log": str(resume_process.stderr_log),
        },
        "initial_model_requests": 0,
        "resume_model_requests": 0,
        "model_requests": 0,
        "initial_claim_count": len(list(initial_claims.glob("*.claim"))),
        "resume_claim_count": len(list(resume_claims.glob("*.claim"))),
        "initial_runtime_marker": initial_marker_payload,
        "resume_runtime_marker": resume_marker_payload,
        "runtime_contracts_equal": True,
        "runtime_contract": initial_contract,
        "resume_config_diff": resume_diff,
        "command_transport_diff": command_transport_diff,
        "paper_input_sha256": paper_before,
    }
    _atomic_json(case_root / "result.json", result)
    return result


def _run_environment(
    *,
    profile: Path,
    out_root: Path,
    current_source,
) -> Mapping[str, Any]:
    unified = _load_unified()
    cfg = unified._build_config(
        profile=profile,
        fixture="single-20aa",
        out_dir=out_root / "environment-preflight",
        task_name=f"{TASK_NAME}-environment",
        execution="daemon",
        generic_search=False,
        seed=SEED,
    )
    environment = unified._environment_manifest(cfg, current_source)
    _atomic_json(out_root / "environment_manifest.json", environment)
    return environment


def _run_cold_guard_environment(
    *,
    profile: Path,
    out_root: Path,
    current_source,
) -> dict[str, Any]:
    """Hash the real runtime assets without requiring a GPU for cold guards."""

    import platform
    import sys

    unified = _load_unified()
    cfg = unified._build_config(
        profile=profile,
        fixture="single-20aa",
        out_dir=out_root / "environment-preflight",
        task_name=f"{TASK_NAME}-cold-environment",
        execution="daemon",
        generic_search=False,
        seed=SEED,
    )
    selected_environment = {
        key: os.environ.get(key)
        for key in (
            "APPTAINER_CONTAINER",
            "APPTAINER_NAME",
            "SINGULARITY_CONTAINER",
            "SINGULARITY_NAME",
            "CUDA_VISIBLE_DEVICES",
            "GH_WEIGHT_AF3",
            "XLA_PYTHON_CLIENT_PREALLOCATE",
            "GH_CUDA_PREINITIALIZE",
        )
    }
    environment = {
        "schema": f"{SCHEMA_ROOT}.cold-environment/v1",
        "repo_root": str(REPO_ROOT),
        "source_manifest": current_source.as_dict(),
        "weights": unified._model_manifest(cfg),
        "environment": selected_environment,
        "python": {
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
            "version": sys.version,
        },
        "gpu": {
            "inspected": False,
            "reason": "cold guard stops before any inference or GPU allocation",
        },
    }
    _atomic_json(out_root / "environment_manifest.json", environment)
    return environment


def _validate_runtime_origin(
    *,
    out_root: Path,
    expected_af3_module_path: Path,
) -> dict[str, Any]:
    unified = _load_unified()
    runtime_imports = unified._preflight_runtime_imports()
    af3_modules = [
        item
        for item in runtime_imports.get("modules", [])
        if item.get("module") == "alphafold3" and item.get("file")
    ]
    if len(af3_modules) != 1:
        raise AcceptanceError(
            "runtime preflight did not identify exactly one AlphaFold3 module"
        )
    actual = Path(str(af3_modules[0]["file"])).resolve()
    expected = expected_af3_module_path.expanduser().resolve()
    matches = (
        actual == expected
        if expected.suffix == ".py"
        else actual.is_relative_to(expected)
    )
    record = {
        "schema": f"{SCHEMA_ROOT}.runtime-origin/v1",
        "status": "PASS" if matches else "ISSUES",
        "expected_af3_module_path": str(expected),
        "observed_af3_module_path": str(actual),
        "matches": matches,
        "runtime_imports": runtime_imports,
    }
    _atomic_json(out_root / "runtime_origin.json", record)
    if not matches:
        raise AcceptanceError(
            "observed AlphaFold3 module origin differs from the requested "
            f"path: expected={expected} observed={actual}"
        )
    return record


def _plan_payload(
    *,
    cases: Mapping[str, CaseSpec],
    choices: Mapping[str, BaselineChoice],
    source_fingerprint: str,
    budget: int | None,
    environment_checked: bool,
    selected_cases: Sequence[str] | None = None,
    request_cap: int = MAX_REQUESTS,
) -> dict[str, Any]:
    selected = _selected_cases(selected_cases)
    scientific_profile: dict[str, Any] = {}
    if "default-cli" in selected:
        scientific_profile["default"] = {
            "sequence": "ACDEFGHIKLMNPQRSTVWY",
            "oligo": ["A"],
        }
    if "paper-cli" in selected:
        scientific_profile["paper"] = {
            "seqs": {
                "A": "+ACDEFGHIKL@shared",
                "B": "shared +MNPQRSTVWY",
            },
            "oligo": ["AB"],
            "steps": 2,
        }
    scientific_profile["alphafold3"] = {
        "num_seeds": 1,
        "num_recycles": 1,
        "num_samples": 1,
    }
    return {
        "schema": f"{SCHEMA_ROOT}.plan/v1",
        "task_name": TASK_NAME,
        "status": "planned",
        "selected_cases": list(selected),
        "request_budget": budget,
        "hard_request_cap": request_cap,
        "no_retry": NO_RETRY,
        "seed": SEED,
        "num_recycles": 1,
        "num_samples": 1,
        "source_fingerprint": source_fingerprint,
        "environment_checked": environment_checked,
        "cold_guard": {
            "boundary": COLD_GUARD_BOUNDARY,
            "model_requests": 0,
            "required_before_model_requests": True,
        },
        "cases": [
            {
                "name": name,
                "fixture": cases[name].fixture,
                "frontdoor": cases[name].frontdoor,
                "historical_baseline": str(cases[name].historical_baseline),
                "reuse_historical_baseline": choices[name].reuse,
                "fresh_baseline_requests": (
                    REQUESTS_PER_COMPLETE_RUN
                    if name == "paper-cli" and not choices[name].reuse
                    else 0
                ),
                "fresh_baseline_allowed": name == "paper-cli",
                "recovery_requests": REQUESTS_PER_RECOVERY_CASE,
                "reasons": list(choices[name].reasons),
                "source_changes": list(choices[name].source_changes),
                "config_differences": list(choices[name].config_differences),
                "weights_match": choices[name].weights_match,
                "gpu_match": choices[name].gpu_match,
                "runtime_image_match": choices[name].runtime_image_match,
                "rng_evidence_present": choices[name].rng_evidence_present,
            }
            for name in selected
        ],
        "scientific_profile": scientific_profile,
        "resources": {
            "gpu": 1,
            "cpu": 4,
            "memory_gib": 24,
            "wall_time_minutes": 60,
            "partition": "boron",
        },
    }


def _dry_run_prepare(
    *,
    out_root: Path,
    profile: Path,
    af3_module_path: Path,
    default_baseline: Path,
    paper_baseline: Path,
    selected_cases: Sequence[str] | None = None,
) -> int:
    unified = _load_unified()
    selected = _selected_cases(selected_cases)
    os.environ["ASTROLABE_ACCEPTANCE_MAX_EVALUATIONS"] = "2"
    unified._configure_max_evaluations(2)
    current_source = unified._read_source_manifest(
        numerical_only=True,
        include_af3_vendor=False,
    )
    cases = _bound_case_specs(
        default_baseline=default_baseline,
        paper_baseline=paper_baseline,
        selected_cases=selected,
    )
    choices = {
        name: _baseline_choice(
            case=cases[name],
            profile=profile,
            environment=None,
            af3_module_path=af3_module_path,
        )
        for name in selected
    }
    default_valid = (
        "default-cli" not in selected
        or choices["default-cli"].reuse
    )
    budget = (
        request_budget(
            fresh_default=False,
            fresh_paper=(
                "paper-cli" in selected
                and not choices["paper-cli"].reuse
            ),
            cases=selected,
        )
        if default_valid
        else None
    )
    request_cap = (
        budget
        if budget is not None and len(selected) != len(SUPPORTED_CASES)
        else MAX_REQUESTS
    )
    payload = _plan_payload(
        cases=cases,
        choices=choices,
        source_fingerprint=current_source.fingerprint,
        budget=budget,
        environment_checked=False,
        selected_cases=selected,
        request_cap=request_cap,
    )
    payload["status"] = "dry-run" if default_valid else "blocked"
    payload["note"] = (
        "Weights/GPU equality is checked during the real preflight. "
        "Default baseline recreation is forbidden in merged mode. "
        "Only a paper baseline may be recreated after runtime preflight."
    )
    _atomic_json(out_root / "plan.json", payload)
    print(
        json.dumps(
            {"status": payload["status"], "plan": str(out_root / "plan.json")}
        )
    )
    return 0 if default_valid else 1


def run_cold_guard_preflight(
    *,
    out_root: Path,
    profile: Path,
    paper_input: Path,
    python_bin: Path,
    timeout_s: float,
    af3_module_path: Path,
    default_baseline: Path,
    paper_baseline: Path,
    selected_cases: Sequence[str] | None = None,
) -> int:
    """Run selected public CLI paths to Runtime factory without inference."""

    _reject_existing_output(out_root)
    unified = _load_unified()
    selected = _selected_cases(selected_cases)
    os.environ["ASTROLABE_ACCEPTANCE_MAX_EVALUATIONS"] = "2"
    unified._configure_max_evaluations(2)
    current_source = unified._read_source_manifest(
        numerical_only=True,
        include_af3_vendor=False,
    )
    _atomic_json(out_root / "source_manifest.json", current_source.as_dict())
    environment = _run_cold_guard_environment(
        profile=profile,
        out_root=out_root,
        current_source=current_source,
    )
    runtime_origin = _validate_runtime_origin(
        out_root=out_root,
        expected_af3_module_path=af3_module_path,
    )
    cases = _bound_case_specs(
        default_baseline=default_baseline,
        paper_baseline=paper_baseline,
        selected_cases=selected,
    )
    source_checks: list[dict[str, Any]] = []

    def guard(phase: str) -> None:
        current = unified._read_source_manifest(
            numerical_only=True,
            include_af3_vendor=False,
        )
        changes = unified._source_changes(current_source, current)
        if changes:
            raise AcceptanceError(
                f"Astrolabe numerical source changed at {phase!r}: "
                f"{json.dumps(changes, sort_keys=True)}"
            )
        source_checks.append(
            {"phase": phase, "fingerprint": current.fingerprint}
        )

    cold_guards = {
        name: _run_cold_guard_case(
            case=cases[name],
            profile=profile,
            paper_input=paper_input,
            out_root=out_root,
            python_bin=python_bin,
            timeout_s=timeout_s,
            source_guard=guard,
        )
        for name in selected
    }
    summary = {
        "schema": f"{SCHEMA_ROOT}.cold-guard-summary/v1",
        "task_name": TASK_NAME,
        "status": "PASS",
        "selected_cases": list(selected),
        "model_requests": 0,
        "hard_request_cap": _maximum_selected_request_cap(selected),
        "environment_manifest": str(out_root / "environment_manifest.json"),
        "runtime_origin": str(out_root / "runtime_origin.json"),
        "observed_af3_module_path": runtime_origin[
            "observed_af3_module_path"
        ],
        "source_checks": source_checks,
        "cases": cold_guards,
    }
    _atomic_json(out_root / "cold_guard_summary.json", summary)
    print(
        json.dumps(
            {
                "status": "PASS",
                "summary": str(out_root / "cold_guard_summary.json"),
            }
        )
    )
    return 0


def run_acceptance(
    *,
    out_root: Path,
    profile: Path,
    paper_input: Path,
    python_bin: Path,
    timeout_s: float,
    af3_module_path: Path,
    default_baseline: Path,
    paper_baseline: Path,
    selected_cases: Sequence[str] | None = None,
) -> int:
    _reject_existing_output(out_root)
    unified = _load_unified()
    selected = _selected_cases(selected_cases)
    os.environ["ASTROLABE_ACCEPTANCE_MAX_EVALUATIONS"] = "2"
    unified._configure_max_evaluations(2)
    current_source = unified._read_source_manifest(
        numerical_only=True,
        include_af3_vendor=False,
    )
    _atomic_json(out_root / "source_manifest.json", current_source.as_dict())
    environment = _run_environment(
        profile=profile,
        out_root=out_root,
        current_source=current_source,
    )
    runtime_origin = _validate_runtime_origin(
        out_root=out_root,
        expected_af3_module_path=af3_module_path,
    )
    cases = _bound_case_specs(
        default_baseline=default_baseline,
        paper_baseline=paper_baseline,
        selected_cases=selected,
    )
    choices = {
        name: _baseline_choice(
            case=cases[name],
            profile=profile,
            environment=environment,
            af3_module_path=af3_module_path,
        )
        for name in selected
    }
    if (
        "default-cli" in selected
        and not choices["default-cli"].reuse
    ):
        _atomic_json(
            out_root / "default_baseline_revalidation.json",
            {
                "schema": f"{SCHEMA_ROOT}.baseline-revalidation/v1",
                "status": "ISSUES",
                "model_requests_issued": 0,
                "baseline": str(cases["default-cli"].historical_baseline),
                "reasons": list(choices["default-cli"].reasons),
                "source_changes": list(
                    choices["default-cli"].source_changes
                ),
                "config_differences": list(
                    choices["default-cli"].config_differences
                ),
                "weights_match": choices["default-cli"].weights_match,
                "gpu_match": choices["default-cli"].gpu_match,
                "runtime_image_match": (
                    choices["default-cli"].runtime_image_match
                ),
            },
        )
        raise AcceptanceError(
            "merged default baseline is not reusable; refusing a duplicate "
            "default baseline before any model request"
        )
    budget_total = request_budget(
        fresh_default=False,
        fresh_paper=(
            "paper-cli" in selected
            and not choices["paper-cli"].reuse
        ),
        cases=selected,
    )
    request_cap = (
        budget_total
        if len(selected) != len(SUPPORTED_CASES)
        else MAX_REQUESTS
    )
    plan = _plan_payload(
        cases=cases,
        choices=choices,
        source_fingerprint=current_source.fingerprint,
        budget=budget_total,
        environment_checked=True,
        selected_cases=selected,
        request_cap=request_cap,
    )
    plan["status"] = "ready"
    plan["af3_module_path"] = str(af3_module_path.resolve())
    _atomic_json(out_root / "plan.json", plan)
    request_ledger = RequestBudget(
        out_root / "request_budget.json",
        limit=request_cap,
    )
    source_checks: list[dict[str, Any]] = []

    def guard(phase: str) -> None:
        current = unified._read_source_manifest(
            numerical_only=True,
            include_af3_vendor=False,
        )
        changes = unified._source_changes(current_source, current)
        _atomic_json(
            out_root
            / "source_checks"
            / f"{len(source_checks) + 1:03d}-{phase.replace(':', '-')}.json",
            {
                "schema": f"{SCHEMA_ROOT}.source-check/v1",
                "phase": phase,
                "expected_fingerprint": current_source.fingerprint,
                "observed_fingerprint": current.fingerprint,
                "matches_expected": not changes,
                "changes": changes,
            },
        )
        if changes:
            raise AcceptanceError(
                f"Astrolabe numerical source changed at {phase!r}: "
                f"{json.dumps(changes, sort_keys=True)}"
            )
        source_checks.append(
            {
                "phase": phase,
                "fingerprint": current.fingerprint,
            }
        )

    cold_guards = {
        name: _run_cold_guard_case(
            case=cases[name],
            profile=profile,
            paper_input=paper_input,
            out_root=out_root,
            python_bin=python_bin,
            timeout_s=timeout_s,
            source_guard=guard,
        )
        for name in selected
    }
    _atomic_json(
        out_root / "cold_guard_summary.json",
        {
            "schema": f"{SCHEMA_ROOT}.cold-guard-summary/v1",
            "status": "PASS",
            "selected_cases": list(selected),
            "model_requests": 0,
            "cases": cold_guards,
        },
    )

    baselines: dict[str, tuple[dict[str, Any], Path]] = {}
    for name in selected:
        case = cases[name]
        choice = choices[name]
        if choice.reuse:
            local_root = out_root / "references" / f"{case.reference}-reused"
            _copy_reused_baseline(case.historical_baseline, local_root)
            baselines[name] = (
                unified._load_json(local_root / "result.json"),
                local_root,
            )
        else:
            guard(f"fresh-baseline:{name}:before")
            fresh_root = _fresh_baseline(
                case=case,
                profile=profile,
                out_root=out_root,
                environment=environment,
                python_bin=python_bin,
                timeout_s=timeout_s,
                budget=request_ledger,
            )
            guard(f"fresh-baseline:{name}:after")
            baselines[name] = (
                unified._load_json(fresh_root / "result.json"),
                fresh_root,
            )

    case_results: dict[str, dict[str, Any]] = {}
    case_roots: dict[str, Path] = {}
    for name in selected:
        case = cases[name]
        guard(f"case:{name}:before")
        result, case_root = _run_recovery_case(
            case=case,
            profile=profile,
            paper_input=paper_input,
            out_root=out_root,
            environment=environment,
            python_bin=python_bin,
            timeout_s=timeout_s,
            budget=request_ledger,
            source_guard=guard,
        )
        guard(f"case:{name}:after")
        case_results[name] = result
        case_roots[name] = case_root
        if result.get("status") != "evidence-complete":
            raise AcceptanceError(
                f"public CLI recovery case {name} failed: {result.get('error')}"
            )

    comparisons: dict[str, dict[str, Any]] = {}
    for name in selected:
        baseline_result, baseline_root = baselines[name]
        comparison = unified.compare_case(
            reference_root=baseline_root,
            case_root=case_roots[name],
        )
        _atomic_json(case_roots[name] / "comparison.json", comparison)
        comparisons[name] = comparison

    case_requests = sum(
        int(row.get("model_calls") or 0)
        for row in case_results.values()
    )
    fresh_baseline_requests = sum(
        int(result.get("model_calls") or 0)
        for name, (result, _root) in baselines.items()
        if not choices[name].reuse
    )
    actual_requests = case_requests + fresh_baseline_requests
    accounting_passed = all(
        result["resume_accounting"]["cut_model_requests"] == 1
        and result["resume_accounting"]["resume_model_requests"] == 1
        and result["resume_accounting"]["duplicate_model_requests"] == 0
        for result in case_results.values()
    )
    comparison_passed = all(
        row.get("status") == "PASS"
        for row in comparisons.values()
    )
    passed = (
        actual_requests == budget_total
        and actual_requests <= request_cap
        and accounting_passed
        and comparison_passed
    )
    summary = {
        "schema": f"{SCHEMA_ROOT}.summary/v1",
        "task_name": TASK_NAME,
        "status": "PASS" if passed else "ISSUES",
        "selected_cases": list(selected),
        "request_budget": budget_total,
        "actual_observed_evaluator_requests": actual_requests,
        "hard_request_cap": request_cap,
        "no_retry": NO_RETRY,
        "baseline_decisions": {
            name: {
                "reuse_historical_baseline": choices[name].reuse,
                "reasons": list(choices[name].reasons),
                "source_changes": list(choices[name].source_changes),
                "config_differences": list(choices[name].config_differences),
                "weights_match": choices[name].weights_match,
                "gpu_match": choices[name].gpu_match,
                "runtime_image_match": choices[name].runtime_image_match,
                "rng_evidence_present": choices[name].rng_evidence_present,
                "result": str(baselines[name][1] / "result.json"),
            }
            for name in selected
        },
        "cases": {
            name: {
                "status": case_results[name].get("status"),
                "model_calls": case_results[name].get("model_calls"),
                "cut_boundary": CUT_BOUNDARY,
                "resume_accounting": case_results[name].get(
                    "resume_accounting"
                ),
                "result": str(case_roots[name] / "result.json"),
                "comparison": str(case_roots[name] / "comparison.json"),
            }
            for name in selected
        },
        "source_checks": source_checks,
        "cold_guard_summary": str(out_root / "cold_guard_summary.json"),
        "cold_guard_model_requests": 0,
        "request_ledger": str(out_root / "request_budget.json"),
        "environment_manifest": str(out_root / "environment_manifest.json"),
        "runtime_origin": str(out_root / "runtime_origin.json"),
        "observed_af3_module_path": runtime_origin[
            "observed_af3_module_path"
        ],
        "source_manifest": str(out_root / "source_manifest.json"),
        "paper_input_sha256": (
            _sha256(paper_input)
            if "paper-cli" in selected
            else None
        ),
    }
    _atomic_json(out_root / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "summary": str(out_root / "summary.json")}))
    return 0 if passed else 1


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run bounded durable recovery through the real default and paper "
            "public CLI frontdoors. No worker entrypoint is used for recovery."
        )
    )
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=SUPPORTED_CASES,
        help=(
            "Public CLI cases to run. Omit to preserve the default of both "
            "default-cli and paper-cli."
        ),
    )
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--paper-inputs", type=Path, default=DEFAULT_PAPER_INPUT)
    parser.add_argument(
        "--default-baseline",
        type=Path,
        help=(
            "Native-in-process baseline generated by portable_public_v5 in "
            "the same frozen capsule."
        ),
    )
    parser.add_argument(
        "--paper-baseline",
        type=Path,
        default=PAPER_BASELINE,
        help="Historical native-pair baseline; a fresh one is allowed if needed.",
    )
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument(
        "--af3-module-path",
        type=Path,
        default=Path(DEFAULT_AF3_MODULE_PATH),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--cold-guard-only",
        action="store_true",
        help="Run selected real CLI paths to Runtime construction with zero inference.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_argument_parser()
    arguments = parser.parse_args(argv)
    selected_cases = _selected_cases(arguments.cases)

    out_root = arguments.out_root.expanduser().resolve()
    profile = arguments.profile.expanduser().resolve()
    paper_input = arguments.paper_inputs.expanduser().resolve()
    python_bin = arguments.python_bin.expanduser().resolve()
    af3_module_path = arguments.af3_module_path.expanduser().resolve()
    paper_baseline = arguments.paper_baseline.expanduser().resolve()
    needs_default_baseline = "default-cli" in selected_cases
    if arguments.cold_guard_only or not needs_default_baseline:
        default_baseline = (
            arguments.default_baseline.expanduser().resolve()
            if arguments.default_baseline is not None
            else DEFAULT_BASELINE.resolve()
        )
    elif arguments.default_baseline is None:
        parser.error(
            "--default-baseline is required for dry-run or recovery execution"
        )
        raise AssertionError("unreachable")
    else:
        default_baseline = arguments.default_baseline.expanduser().resolve()
    owned_empty_root = not out_root.exists() or (
        out_root.is_dir() and not any(out_root.iterdir())
    )
    try:
        if arguments.dry_run:
            _reject_existing_output(out_root)
            return _dry_run_prepare(
                out_root=out_root,
                profile=profile,
                af3_module_path=af3_module_path,
                default_baseline=default_baseline,
                paper_baseline=paper_baseline,
                selected_cases=selected_cases,
            )
        if arguments.cold_guard_only:
            return run_cold_guard_preflight(
                out_root=out_root,
                profile=profile,
                paper_input=paper_input,
                python_bin=python_bin,
                timeout_s=arguments.timeout_s,
                af3_module_path=af3_module_path,
                default_baseline=default_baseline,
                paper_baseline=paper_baseline,
                selected_cases=selected_cases,
            )
        return run_acceptance(
            out_root=out_root,
            profile=profile,
            paper_input=paper_input,
            python_bin=python_bin,
            timeout_s=arguments.timeout_s,
            af3_module_path=af3_module_path,
            default_baseline=default_baseline,
            paper_baseline=paper_baseline,
            selected_cases=selected_cases,
        )
    except Exception as error:
        if (
            owned_empty_root
            and out_root.is_dir()
            and not (out_root / "summary.json").exists()
        ):
            unified = _load_unified()
            unified._atomic_json(
                out_root / "summary.json",
                {
                    "schema": f"{SCHEMA_ROOT}.summary/v1",
                    "task_name": TASK_NAME,
                    "status": "ISSUES",
                    "selected_cases": list(selected_cases),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "hard_request_cap": _maximum_selected_request_cap(
                        selected_cases
                    ),
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
    "AcceptanceError",
    "BaselineChoice",
    "COLD_GUARD_BOUNDARY",
    "COLD_GUARD_EXIT",
    "CUT_BOUNDARY",
    "CUT_EXIT",
    "MAX_REQUESTS",
    "RequestBudget",
    "SUPPORTED_CASES",
    "TASK_NAME",
    "public_cli_command",
    "request_budget",
    "run_cold_guard_preflight",
    "run_acceptance",
    "validate_command_transport_diff",
    "validate_resume_override_diff",
]


if __name__ == "__main__":
    raise SystemExit(main())
