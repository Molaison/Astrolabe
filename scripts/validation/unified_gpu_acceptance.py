#!/usr/bin/env python3
"""Bounded GPU acceptance evidence for unified Astrolabe AF3 call paths.

The harness deliberately keeps one launch per case and never retries a model
call.  It compares real evaluator results against a direct native callback
reference run with the same frozen input and scientific settings.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import secrets
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE = REPO_ROOT / "config" / "container_af3_smoke.yaml"
SINGLE_SEQUENCE = "ACDEFGHIKLMNPQRSTVWY"
PAIR_SHARED_SEQUENCE = "ACDEFGHIKL"
PAIR_UNIQUE_SEQUENCE = "MNPQRSTVWY"
SEED = 17
# Candidate batch derives the first candidate seed from base_seed + 1.
CANDIDATE_BATCH_BASE_SEED = SEED - 1
DEFAULT_MAX_EVALUATIONS = 2
THROUGHPUT_MAX_EVALUATIONS = 4
ALLOWED_MAX_EVALUATIONS = (
    DEFAULT_MAX_EVALUATIONS,
    THROUGHPUT_MAX_EVALUATIONS,
)
MAX_EVALUATIONS_ENV = "ASTROLABE_ACCEPTANCE_MAX_EVALUATIONS"
THROUGHPUT_CASES = frozenset({"default-api", "default-cli"})
MAX_EVALUATIONS = DEFAULT_MAX_EVALUATIONS
AF3_MODEL_DIR = "/models/alphafold3"
DEFAULT_RESUME_CASE = "default-api"
CASE_NAMES = (
    "default-api",
    "default-cli",
    "generic-api",
    "paper-cli",
    "candidate-batch",
)
PRIORITY_CASES = ("default-api", "generic-api")
REFERENCE_FOR_CASE = {
    "default-api": "native-single",
    "default-cli": "native-single",
    "generic-api": "native-generic",
    "paper-cli": "native-pair",
    "candidate-batch": "native-single",
}


class AcceptanceError(RuntimeError):
    """The acceptance run cannot produce trustworthy evidence."""


class UnsupportedEvidence(AcceptanceError):
    """A returned object has no lossless harness representation."""


class SourceChanged(AcceptanceError):
    """Execution inputs changed after the acceptance source was frozen."""


def parse_max_evaluations(value: str | None) -> int:
    """Parse the explicit process-wide acceptance request cap."""

    if value is None or value == "":
        return DEFAULT_MAX_EVALUATIONS
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise AcceptanceError(
            f"{MAX_EVALUATIONS_ENV} must be exactly 2 or 4, got {value!r}"
        ) from error
    if parsed not in ALLOWED_MAX_EVALUATIONS:
        raise AcceptanceError(
            f"{MAX_EVALUATIONS_ENV} must be exactly 2 or 4, got {value!r}"
        )
    return parsed


def acceptance_max_evaluations_from_environment(
    environ: Mapping[str, str] | None = None,
) -> int:
    environment = os.environ if environ is None else environ
    return parse_max_evaluations(environment.get(MAX_EVALUATIONS_ENV))


def _configure_max_evaluations(value: int) -> None:
    global MAX_EVALUATIONS
    if value not in ALLOWED_MAX_EVALUATIONS:
        raise AcceptanceError(
            f"acceptance max evaluations must be 2 or 4, got {value}"
        )
    MAX_EVALUATIONS = value


def acceptance_budget_record() -> dict[str, Any]:
    """Return the frozen acceptance request budget recorded in evidence."""

    return {
        "schema": "astrolabe.gpu-acceptance.request-budget/v1",
        "max_evaluations_per_process": MAX_EVALUATIONS,
        "environment_variable": MAX_EVALUATIONS_ENV,
        "environment_value": os.environ.get(MAX_EVALUATIONS_ENV) or None,
        "profile": (
            "default"
            if MAX_EVALUATIONS == DEFAULT_MAX_EVALUATIONS
            else "throughput"
        ),
        "throughput_cases": sorted(THROUGHPUT_CASES),
    }


def recorded_max_evaluations(environment: Mapping[str, Any]) -> int:
    """Read a stored acceptance cap, accepting pre-field two-call baselines."""

    acceptance = environment.get("acceptance")
    if isinstance(acceptance, Mapping):
        value = acceptance.get("max_evaluations_per_process")
        if value in ALLOWED_MAX_EVALUATIONS:
            return int(value)
    config = environment.get("config")
    if isinstance(config, Mapping):
        value = config.get("steps")
        if value in ALLOWED_MAX_EVALUATIONS:
            return int(value)
    return DEFAULT_MAX_EVALUATIONS


@dataclass(frozen=True)
class CasePlan:
    name: str
    reference: str
    frontdoor: str
    fixture: str


@dataclass(frozen=True)
class ProcessResult:
    returncode: int | None
    timed_out: bool
    duration_s: float
    stdout_log: Path
    stderr_log: Path


@dataclass(frozen=True)
class SourceManifest:
    files: tuple[tuple[str, str], ...]
    fingerprint: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "astrolabe.gpu-acceptance.source-manifest/v1",
            "fingerprint": self.fingerprint,
            "files": dict(self.files),
        }


def case_plan(name: str) -> CasePlan:
    if name not in CASE_NAMES:
        raise ValueError(f"unknown acceptance case {name!r}")
    if name == "paper-cli":
        fixture = "pair-shared-fixed"
    else:
        fixture = "single-20aa"
    frontdoors = {
        "default-api": "astrolabe.app.Astrolabe.run",
        "default-cli": "python -m astrolabe.cli run",
        "generic-api": "astrolabe.app.Astrolabe.run search.native_refinement",
        "paper-cli": "python -m astrolabe.cli paper run evaluate-af3",
        "candidate-batch": (
            "python -m astrolabe.cli experimental candidate-batch "
            "-> scripts/astrolabe run"
        ),
    }
    return CasePlan(
        name=name,
        reference=REFERENCE_FOR_CASE[name],
        frontdoor=frontdoors[name],
        fixture=fixture,
    )


def _sha256(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _source_files(
    root: Path = REPO_ROOT,
    *,
    numerical_only: bool = False,
    include_af3_vendor: bool = True,
) -> tuple[Path, ...]:
    root = root.expanduser().resolve()
    files = set((root / "astrolabe").rglob("*.py"))
    files.add(root / "run.py")
    if not numerical_only:
        files.update(
            {
                root / "scripts" / "astrolabe",
                Path(__file__).resolve(),
                root / "config" / "base.yaml",
                root / "config" / "container_af3_smoke.yaml",
                root / "config" / "env" / "env.yaml",
                root / "config" / "env" / "env_container.yaml",
                root / "config" / "paper" / "evaluate_af3.yaml",
            }
        )
    vendor_root = root / "vendor" / "alphafold3" / "src"
    if include_af3_vendor and vendor_root.is_dir():
        files.update(vendor_root.rglob("*.py"))
    return tuple(sorted(path.resolve() for path in files if path.is_file()))


def _read_source_manifest(
    root: Path = REPO_ROOT,
    *,
    numerical_only: bool = False,
    include_af3_vendor: bool = True,
) -> SourceManifest:
    root = root.expanduser().resolve()
    files = tuple(
        (str(path.relative_to(root)), _sha256(path))
        for path in _source_files(
            root,
            numerical_only=numerical_only,
            include_af3_vendor=include_af3_vendor,
        )
    )
    fingerprint = hashlib.sha256(
        json.dumps(
            dict(files),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return SourceManifest(files=files, fingerprint=fingerprint)


def _source_manifest_from_payload(
    value: Any,
    *,
    include_af3_vendor: bool = True,
) -> SourceManifest:
    if not isinstance(value, Mapping):
        raise AcceptanceError("source manifest must be a JSON object")
    embedded = value.get("files") if "files" in value else value
    if not isinstance(embedded, Mapping):
        raise AcceptanceError("source manifest has no file-hash mapping")
    files = []
    for raw_path, raw_digest in embedded.items():
        path = str(raw_path).replace("\\", "/")
        is_numerical = (
            path == "run.py"
            or (
                path.startswith("astrolabe/")
                and path.endswith(".py")
            )
            or (
                include_af3_vendor
                and
                path.startswith("vendor/alphafold3/src/")
                and path.endswith(".py")
            )
        )
        if not is_numerical:
            continue
        digest = str(raw_digest)
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise AcceptanceError(
                f"source manifest has an invalid SHA256 for {path!r}"
            )
        files.append((path, digest))
    if not files:
        raise AcceptanceError(
            "source manifest contains no numerical application sources"
        )
    files = tuple(sorted(files))
    return SourceManifest(
        files=files,
        fingerprint=hashlib.sha256(
            json.dumps(
                dict(files),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    )


def _load_source_manifest(
    path: Path,
    *,
    include_af3_vendor: bool = True,
) -> SourceManifest:
    return _source_manifest_from_payload(
        _load_json(path),
        include_af3_vendor=include_af3_vendor,
    )


def _source_changes(
    expected: SourceManifest,
    current: SourceManifest,
) -> list[dict[str, Any]]:
    before = dict(expected.files)
    after = dict(current.files)
    changes = []
    for path in sorted(set(before) | set(after)):
        if before.get(path) != after.get(path):
            changes.append(
                {
                    "path": path,
                    "before": before.get(path),
                    "after": after.get(path),
                }
            )
    return changes


def _check_source_manifest(
    expected: SourceManifest,
    *,
    out_root: Path,
    phase: str,
    check_index: int,
) -> SourceManifest:
    current = _read_source_manifest()
    record = {
        "schema": "astrolabe.gpu-acceptance.source-check/v1",
        "phase": phase,
        "fingerprint": current.fingerprint,
        "matches_expected": current == expected,
        "changes": _source_changes(expected, current),
        "files": dict(current.files),
    }
    path = (
        out_root
        / "source_checks"
        / f"{check_index:03d}_{_safe_key(phase)}.json"
    )
    _atomic_json(path, record)
    if current != expected:
        _atomic_json(out_root / "source_change.json", record)
        raise SourceChanged(
            f"Astrolabe source changed at {phase!r}: "
            f"expected {expected.fingerprint}, observed {current.fingerprint}"
        )
    return current


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise UnsupportedEvidence("object-dtype JSON evidence is unsupported")
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _reject_existing_output(out_root: Path) -> None:
    if out_root.exists() and any(out_root.iterdir()):
        raise AcceptanceError(
            f"--out-root must be absent or empty; refusing to mix evidence in {out_root}"
        )
    out_root.mkdir(parents=True, exist_ok=True)
    try:
        out_root.chmod(0o700)
    except OSError:
        pass


def _nvidia_manifest() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise AcceptanceError("nvidia-smi is unavailable; GPU acceptance requires a GPU runtime")
    command = [
        executable,
        "--query-gpu=uuid,name,driver_version,memory.total",
        "--format=csv,noheader",
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise AcceptanceError(
            "nvidia-smi failed before model execution: "
            f"rc={completed.returncode} stderr={completed.stderr.strip()}"
        )
    rows = [
        [field.strip() for field in line.split(",")]
        for line in completed.stdout.splitlines()
        if line.strip()
    ]
    if not rows:
        raise AcceptanceError("nvidia-smi reported no visible GPU")
    return {
        "command": command,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "devices": [
            {
                "uuid": row[0],
                "name": row[1],
                "driver_version": row[2],
                "memory_total": row[3],
            }
            for row in rows
            if len(row) >= 4
        ],
    }


def _compose_smoke(profile: Path) -> Any:
    from astrolabe.configuration import load_composed_config

    return load_composed_config(profile)


def _compose_paper_af3() -> Any:
    from astrolabe import config_dir
    from astrolabe.cli import _compose_hydra_config

    return _compose_hydra_config(
        config_dir=Path(config_dir()),
        config_name="paper/evaluate_af3",
        overrides=["env=env_container"],
    )


def _apply_common_config(
    cfg: Any,
    *,
    out_dir: Path,
    task_name: str,
    seed: int,
    execution: str,
) -> Any:
    from omegaconf import open_dict

    with open_dict(cfg):
        cfg.task_name = task_name
        cfg.out_dir = str(out_dir)
        cfg.seed = int(seed)
        cfg.repeats = 1
        # AcceptedSuccessorPolicy rounds are max(0, steps - 1), so the
        # explicit acceptance cap is also the root-plus-successors call count.
        cfg.steps = MAX_EVALUATIONS
        cfg.mutation_num = 1
        cfg.enable_pyrosetta = False
        cfg.skip_complex_poses = True
        cfg.stages.default.mutation_num = 1
        cfg.stages.default.weighted_score_expression = "plddt"
        cfg.evaluator.alphafold3.execution = execution
        cfg.evaluator.alphafold3.model_dir = AF3_MODEL_DIR
        cfg.evaluator.alphafold3.num_seeds = 1
        cfg.evaluator.alphafold3.num_recycles = 1
        cfg.evaluator.alphafold3.num_samples = 1
        cfg.evaluator.alphafold3.run_data_pipeline = False
        cfg.evaluator.alphafold3.run_inference = True
        cfg.evaluator.alphafold3.flash_attention_implementation = "cudnn"
        # The selected term is what evaluator construction merges into the
        # shared alphafold3 defaults, so keep the same overrides there.
        term = cfg.evaluator.evaluator_terms.Alphafold3
        term.num_seeds = 1
        term.num_recycles = 1
        term.num_samples = 1
        term.run_data_pipeline = False
        term.run_inference = True
        term.flash_attention_implementation = "cudnn"
        term.execution = execution
        cfg.weight_path.alphafold3 = AF3_MODEL_DIR
    return cfg


def _apply_pair_fixture(cfg: Any) -> Any:
    from omegaconf import open_dict

    with open_dict(cfg):
        cfg.types = {"A": "p", "B": "p"}
        cfg.seqs = {
            "A": f"+{PAIR_SHARED_SEQUENCE}@shared",
            "B": f"shared +{PAIR_UNIQUE_SEQUENCE}",
        }
        cfg.homologies = {}
        cfg.stages.default.oligo = ["AB"]
    return cfg


def _apply_single_fixture(cfg: Any) -> Any:
    from omegaconf import open_dict

    with open_dict(cfg):
        cfg.types = {"A": "p"}
        cfg.seqs = {"A": SINGLE_SEQUENCE}
        cfg.homologies = {}
        cfg.stages.default.oligo = ["A"]
    return cfg


def _apply_generic_search(cfg: Any) -> Any:
    from omegaconf import open_dict

    with open_dict(cfg):
        cfg.search = {
            "algorithm": "branching",
            "owner": "native_refinement",
            "width": 1,
            "proposals_per_parent": 1,
            "max_evaluations": MAX_EVALUATIONS,
            "proposal_cap": 10,
            "measurement": "weighted_score",
            "summary_format": "json",
        }
    return cfg


def _save_standalone(cfg: Any, path: Path) -> None:
    from omegaconf import OmegaConf

    value = OmegaConf.to_container(cfg, resolve=False)
    if not isinstance(value, dict):
        raise TypeError("composed config must resolve to a mapping")
    value.pop("defaults", None)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        OmegaConf.to_yaml(OmegaConf.create(value), resolve=False),
        encoding="utf-8",
    )


def _build_config(
    *,
    profile: Path,
    fixture: str,
    out_dir: Path,
    task_name: str,
    execution: str,
    generic_search: bool,
    seed: int = SEED,
) -> Any:
    if fixture == "pair-shared-fixed":
        cfg = _compose_paper_af3()
        _apply_pair_fixture(cfg)
    elif fixture == "single-20aa":
        cfg = _compose_smoke(profile)
        _apply_single_fixture(cfg)
    else:
        raise ValueError(f"unknown fixture {fixture!r}")
    _apply_common_config(
        cfg,
        out_dir=out_dir,
        task_name=task_name,
        seed=seed,
        execution=execution,
    )
    if fixture == "pair-shared-fixed":
        _apply_pair_fixture(cfg)
    if generic_search:
        _apply_generic_search(cfg)
    return cfg


def _invocation_seed(case_name: str) -> int:
    """Return the config-level seed needed to produce the requested child seed."""

    return CANDIDATE_BATCH_BASE_SEED if case_name == "candidate-batch" else SEED


def _candidate_executed_config(
    case_root: Path,
    invocation_config: Any,
) -> tuple[Any, dict[str, Any]]:
    """Load the one real candidate child config used as comparison evidence."""

    from omegaconf import OmegaConf

    candidate_configs = sorted(
        (case_root / "run" / "candidate_pool").glob("*/config.yaml")
    )
    if len(candidate_configs) != 1:
        raise AcceptanceError(
            "candidate-batch acceptance requires exactly one candidate child "
            f"config, found {len(candidate_configs)}"
        )
    config_path = candidate_configs[0]
    config = OmegaConf.load(config_path)
    candidate_id = config_path.parent.name
    suffix = candidate_id.rsplit("_", 1)[-1]
    if not suffix.isdigit():
        raise AcceptanceError(
            f"candidate id has no numeric seed offset: {candidate_id!r}"
        )
    offset = int(suffix)
    base_seed = int(invocation_config.seed)
    execution_seed = int(config.seed)
    if execution_seed != base_seed + offset:
        raise AcceptanceError(
            "candidate child seed does not equal base seed plus candidate offset: "
            f"base={base_seed} offset={offset} child={execution_seed}"
        )
    if execution_seed != SEED:
        raise AcceptanceError(
            "candidate-batch harness did not produce the requested scientific "
            f"seed {SEED}; child seed is {execution_seed}"
        )
    if int(config.steps) != MAX_EVALUATIONS:
        raise AcceptanceError(
            "candidate child steps differ from the acceptance budget: "
            f"{config.steps}"
        )
    return config, {
        "candidate_id": candidate_id,
        "config_path": str(config_path),
        "base_seed": base_seed,
        "seed_offset": offset,
        "execution_seed": execution_seed,
    }


def _paper_input_overlay(cfg: Any) -> dict[str, Any]:
    """Return only the paper CLI input fields; the protocol stays authoritative."""
    from omegaconf import OmegaConf

    selected = OmegaConf.create(
        {
            "seed": int(cfg.seed),
            "steps": int(cfg.steps),
            "mutation_num": 1,
            "types": OmegaConf.to_container(cfg.types, resolve=True),
            "seqs": OmegaConf.to_container(cfg.seqs, resolve=True),
            "homologies": {},
            "evaluator": {
                "evaluator_terms": {
                    "Alphafold3": OmegaConf.to_container(
                        cfg.evaluator.evaluator_terms.Alphafold3,
                        resolve=True,
                    )
                },
                "alphafold3": {
                    "model_dir": AF3_MODEL_DIR,
                    "num_seeds": 1,
                    "num_recycles": 1,
                    "num_samples": 1,
                    "run_data_pipeline": False,
                    "run_inference": True,
                    "flash_attention_implementation": "cudnn",
                    "execution": "daemon",
                },
            },
            "stages": {
                "default": {
                    "oligo": ["AB"],
                    "mutation_num": 1,
                    "weighted_score_expression": "plddt",
                }
            },
        }
    )
    return OmegaConf.to_container(selected, resolve=False)


def _digest_config(cfg: Any) -> dict[str, Any]:
    from omegaconf import OmegaConf

    payload = OmegaConf.to_container(cfg, resolve=True)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_default,
    ).encode("utf-8")
    term = cfg.evaluator.evaluator_terms.Alphafold3
    return {
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "acceptance": acceptance_budget_record(),
        "seed": int(cfg.seed),
        "steps": int(cfg.steps),
        "repeats": int(cfg.repeats),
        "mutation_num": int(cfg.stages.default.mutation_num),
        "weighted_score_expression": str(
            cfg.stages.default.weighted_score_expression
        ),
        "oligo": list(cfg.stages.default.oligo),
        "sequences": {
            str(key): str(value) for key, value in dict(cfg.seqs).items()
        },
        "model": {
            "backend": "AlphaFold3",
            "model_dir": str(cfg.evaluator.alphafold3.model_dir),
            "num_seeds": int(term.num_seeds),
            "num_recycles": int(term.num_recycles),
            "num_samples": int(term.num_samples),
            "flash_attention_implementation": str(
                term.flash_attention_implementation
            ),
            "bfloat16": "all (vendor default; not overridden)",
            "host_materialization": "bfloat16 -> float32",
            "run_data_pipeline": bool(term.run_data_pipeline),
            "run_inference": bool(term.run_inference),
            "execution": str(
                cfg.evaluator.alphafold3.get("execution", "daemon")
            ),
        },
        "search": (
            None
            if cfg.get("search") is None
            else OmegaConf.to_container(cfg.search, resolve=True)
        ),
    }


_VOLATILE_CONFIG_KEYS = {
    "task_name",
    "out_dir",
    "repeat_index",
    "raw_model_results_dir",
}


def _scientific_config_projection(cfg: Any) -> dict[str, Any]:
    from omegaconf import OmegaConf

    value = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(value, dict):
        raise TypeError("configuration must resolve to a mapping")
    return {
        key: item
        for key, item in value.items()
        if key not in _VOLATILE_CONFIG_KEYS
    }


def _config_digest_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if key != "sha256"
    }


def _comparable_scientific_config(value: Mapping[str, Any] | None) -> dict[str, Any]:
    projection = _config_digest_projection(value or {})
    model = projection.get("model")
    if isinstance(model, Mapping):
        projection["model"] = {
            key: item
            for key, item in model.items()
            if key != "execution"
        }
    return projection


def _config_differences(
    left: Any,
    right: Any,
    *,
    path: str = "$",
) -> list[dict[str, Any]]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        differences: list[dict[str, Any]] = []
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}"
            if key not in left:
                differences.append(
                    {"path": child, "reference": "<missing>", "case": right[key]}
                )
            elif key not in right:
                differences.append(
                    {"path": child, "reference": left[key], "case": "<missing>"}
                )
            else:
                differences.extend(
                    _config_differences(left[key], right[key], path=child)
                )
        return differences
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [{"path": path, "reference": left, "case": right}]
        differences: list[dict[str, Any]] = []
        for index, (a, b) in enumerate(zip(left, right)):
            differences.extend(
                _config_differences(a, b, path=f"{path}[{index}]")
            )
        return differences
    return [] if _scientific_equal(left, right) else [
        {"path": path, "reference": left, "case": right}
    ]


def _model_manifest(cfg: Any) -> dict[str, Any]:
    model_dir = Path(str(cfg.evaluator.alphafold3.model_dir)).expanduser()
    if not model_dir.is_absolute():
        model_dir = (REPO_ROOT / model_dir).resolve()
    if not model_dir.is_dir():
        raise AcceptanceError(f"AF3 model directory does not exist: {model_dir}")
    files = []
    for path in sorted(model_dir.rglob("*")):
        if not path.is_file():
            continue
        stat = path.stat()
        files.append(
            {
                "path": str(path),
                "relative_path": str(path.relative_to(model_dir)),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": _sha256(path),
            }
        )
    if not files:
        raise AcceptanceError(f"AF3 model directory is empty: {model_dir}")
    return {"root": str(model_dir), "files": files}


def _environment_manifest(
    cfg: Any,
    source_manifest: SourceManifest,
) -> dict[str, Any]:
    return {
        "schema": "astrolabe.gpu-acceptance.environment/v1",
        "created_unix_s": time.time(),
        "repo_root": str(REPO_ROOT),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "implementation": sys.implementation.name,
        },
        "gpu": _nvidia_manifest(),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "NVIDIA_VISIBLE_DEVICES",
                "XLA_PYTHON_CLIENT_PREALLOCATE",
                "JAX_COMPILATION_CACHE_DIR",
                "NUMBA_CACHE_DIR",
                "ASTROLABE_RUNTIME_ACTIVE",
                "APPTAINER_NAME",
                "APPTAINER_CONTAINER",
                "SINGULARITY_NAME",
                "SINGULARITY_CONTAINER",
                MAX_EVALUATIONS_ENV,
            )
        },
        "acceptance": acceptance_budget_record(),
        "compilation_cache": {
            "jax": {
                "environment_variable": os.environ.get(
                    "JAX_COMPILATION_CACHE_DIR"
                ),
                "configured_path": (
                    None
                    if cfg.evaluator.alphafold3.get(
                        "jax_compilation_cache_dir"
                    )
                    is None
                    else str(
                        cfg.evaluator.alphafold3.jax_compilation_cache_dir
                    )
                ),
                "observed_only": True,
                "cleared": False,
                "note": (
                    "Shared or persistent JAX compilation cache may be reused; "
                    "the harness does not clear it or infer a cold compilation."
                ),
            }
        },
        "source_manifest": source_manifest.as_dict(),
        "config": _digest_config(cfg),
        "weights": _model_manifest(cfg),
    }


_REQUIRED_RUNTIME_IMPORTS = (
    "attrs",
    "jsonschema",
    "jsonschema_specifications",
    "referencing",
    "rpds",
    "hydra",
    "omegaconf",
    "numpy",
    "jax",
    "haiku",
    "alphafold3",
)


def _preflight_runtime_imports() -> dict[str, Any]:
    import importlib
    from importlib import metadata

    modules = []
    failures = []
    for name in _REQUIRED_RUNTIME_IMPORTS:
        try:
            module = importlib.import_module(name)
        except Exception as error:
            failures.append(
                {
                    "module": name,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            continue
        modules.append(
            {
                "module": name,
                "file": getattr(module, "__file__", None),
                "version": getattr(module, "__version__", None),
            }
        )
    if failures:
        raise AcceptanceError(
            "runtime import preflight failed before model execution: "
            + json.dumps(failures, sort_keys=True)
        )
    return {
        "schema": "astrolabe.gpu-acceptance.runtime-import-preflight/v1",
        "status": "passed",
        "python": sys.executable,
        "metadata_version": metadata.version("jsonschema"),
        "modules": modules,
    }


def _evidence_file_hashes(root: Path) -> dict[str, str]:
    evidence = root / "evidence"
    required = (
        "standard_scores.npz",
        "standard_scores.json",
        "rng_trace.npz",
        "rng_trace.json",
        "structure_coordinates.npz",
        "structure_coordinates.json",
        "decisions.json",
        "sequence_and_scores.json",
    )
    result = {}
    for name in required:
        path = evidence / name
        if not path.is_file():
            raise AcceptanceError(f"baseline evidence is missing {path}")
        result[name] = _sha256(path)
    result["result.json"] = _sha256(root / "result.json")
    result["config.yaml"] = _sha256(root / "config.yaml")
    return result


def _validate_resume_baseline(
    *,
    out_root: Path,
    profile: Path,
    environment: Mapping[str, Any],
    current_source: SourceManifest,
    runtime_imports: Mapping[str, Any],
    expected_af3_module_path: Path,
    frozen_source_root: Path | None,
    frozen_source_manifest: Path | None,
) -> tuple[dict[str, Any], Any]:
    failed_root = out_root / "cases" / DEFAULT_RESUME_CASE
    failed_result_path = failed_root / "result.json"
    if not failed_result_path.is_file():
        raise AcceptanceError(
            f"resume requires the original failed case result: {failed_result_path}"
        )
    failed_result = _load_json(failed_result_path)
    if failed_result.get("status") != "failed":
        raise AcceptanceError("resume requires an existing failed default-api result")
    if failed_result.get("model_calls") != 0:
        raise AcceptanceError(
            "refusing resume unless the failed default-api case proves model_calls=0"
        )

    reference_root = out_root / "references" / "native-single"
    reference_result_path = reference_root / "result.json"
    if not reference_result_path.is_file():
        raise AcceptanceError(
            f"native-single baseline is missing: {reference_result_path}"
        )
    baseline = _load_json(reference_result_path)
    if baseline.get("status") != "evidence-complete":
        raise AcceptanceError("native-single baseline is not evidence-complete")
    baseline_environment = baseline.get("environment_manifest") or {}
    baseline_max_evaluations = recorded_max_evaluations(baseline_environment)
    current_max_evaluations = recorded_max_evaluations(environment)
    if baseline_max_evaluations != current_max_evaluations:
        raise AcceptanceError(
            "resume acceptance budget mismatch: "
            f"baseline={baseline_max_evaluations}, "
            f"current={current_max_evaluations}"
        )
    if baseline.get("model_calls") != MAX_EVALUATIONS:
        raise AcceptanceError(
            "native-single baseline does not prove exactly "
            f"{MAX_EVALUATIONS} model calls"
        )

    reference_cfg = _build_config(
        profile=profile,
        fixture="single-20aa",
        out_dir=reference_root / "run",
        task_name="resume_reference_validation",
        execution="in_process",
        generic_search=False,
    )
    baseline_config_path = reference_root / "config.yaml"
    if not baseline_config_path.is_file():
        raise AcceptanceError("native-single baseline config.yaml is missing")
    from omegaconf import OmegaConf

    baseline_config = OmegaConf.load(baseline_config_path)
    if not _scientific_equal(
        _scientific_config_projection(baseline_config),
        _scientific_config_projection(reference_cfg),
    ):
        raise AcceptanceError(
            "native-single baseline input/config differs from the current "
            "acceptance profile"
        )

    current_digest = _config_digest_projection(_digest_config(reference_cfg))
    baseline_digest = _config_digest_projection(
        dict(baseline.get("config") or {})
    )
    if not _scientific_equal(current_digest, baseline_digest):
        raise AcceptanceError(
            "native-single baseline result metadata differs from the current config"
        )

    if not _scientific_equal(
        baseline_environment.get("weights"),
        environment.get("weights"),
    ):
        raise AcceptanceError(
            "AF3 weights changed since the native-single baseline"
        )
    if not _scientific_equal(
        baseline_environment.get("gpu"),
        environment.get("gpu"),
    ):
        raise AcceptanceError(
            "resume must run on the same GPU allocation as baseline"
        )

    af3_modules = [
        item
        for item in runtime_imports.get("modules", [])
        if item.get("module") == "alphafold3" and item.get("file")
    ]
    if len(af3_modules) != 1:
        raise AcceptanceError(
            "runtime preflight did not record exactly one AF3 module origin"
        )
    actual_af3_module_path = Path(af3_modules[0]["file"]).resolve()
    expected_path = expected_af3_module_path.expanduser().resolve()
    if actual_af3_module_path != expected_path:
        raise AcceptanceError(
            "current AF3 module origin differs from the measured old baseline: "
            f"expected {expected_path}, got {actual_af3_module_path}"
        )
    actual_path_text = str(actual_af3_module_path)
    current_vendor_root = (REPO_ROOT / "vendor" / "alphafold3").resolve()
    if actual_path_text.startswith("/alphafold3_venv/"):
        af3_code_source = "sif-site-packages"
        include_af3_vendor = False
    elif actual_af3_module_path.is_relative_to(current_vendor_root):
        af3_code_source = "repository-vendor"
        include_af3_vendor = True
    else:
        raise AcceptanceError(
            "AF3 module origin is neither the immutable SIF site-packages "
            f"nor repository vendor: {actual_af3_module_path}"
        )

    embedded_source = baseline_environment.get("source_manifest")
    current_numerical = _read_source_manifest(
        numerical_only=True,
        include_af3_vendor=include_af3_vendor,
    )
    manifest_from_file = (
        None
        if frozen_source_manifest is None
        else _load_source_manifest(
            frozen_source_manifest,
            include_af3_vendor=include_af3_vendor,
        )
    )
    manifest_from_root = (
        None
        if frozen_source_root is None
        else _read_source_manifest(
            frozen_source_root,
            numerical_only=True,
            include_af3_vendor=include_af3_vendor,
        )
    )
    if (
        manifest_from_file is not None
        and manifest_from_root is not None
        and manifest_from_file != manifest_from_root
    ):
        raise AcceptanceError(
            "capsule source manifest and --resume-source-root disagree"
        )
    if manifest_from_file is not None:
        frozen_manifest = manifest_from_file
    elif manifest_from_root is not None:
        frozen_manifest = manifest_from_root
    elif isinstance(embedded_source, Mapping):
        frozen_manifest = _source_manifest_from_payload(
            embedded_source,
            include_af3_vendor=include_af3_vendor,
        )
    else:
        raise AcceptanceError(
            "baseline has no source fingerprint; provide "
            "--resume-source-manifest or --resume-source-root"
        )
    if frozen_manifest != current_numerical:
        raise AcceptanceError(
            "numerical source differs from the frozen native-single baseline"
        )

    report = {
        "schema": "astrolabe.gpu-acceptance.resume-baseline-validation/v1",
        "status": "validated",
        "failed_case_model_calls": 0,
        "baseline_model_calls": MAX_EVALUATIONS,
        "acceptance_budget": acceptance_budget_record(),
        "baseline_config_matches": True,
        "baseline_result_metadata_matches": True,
        "baseline_weights_match": True,
        "baseline_gpu_matches": True,
        "numerical_source_matches": True,
        "af3_code_source": af3_code_source,
        "af3_module_path": actual_path_text,
        "numerical_source_fingerprint": frozen_manifest.fingerprint,
        "current_source_fingerprint": current_source.fingerprint,
        "frozen_source_root": (
            None if frozen_source_root is None else str(frozen_source_root)
        ),
        "frozen_source_manifest": (
            None
            if frozen_source_manifest is None
            else str(frozen_source_manifest)
        ),
        "failed_case_evidence_sha256": {
            path.name: _sha256(path)
            for path in sorted(failed_root.glob("*"))
            if path.is_file()
        },
        "baseline_evidence_sha256": _evidence_file_hashes(reference_root),
    }
    return report, reference_cfg


_SITECUSTOMIZE = r'''
from __future__ import annotations
import json
import os
from pathlib import Path
import pickle
import time
import traceback

def _acceptance_install():
    capture = os.environ.get("ASTROLABE_ACCEPTANCE_CAPTURE")
    claims = os.environ.get("ASTROLABE_ACCEPTANCE_CLAIMS")
    maximum = os.environ.get("ASTROLABE_ACCEPTANCE_MAX_CALLS")
    if not capture or not claims or not maximum:
        return
    capture_path = Path(capture)
    claims_path = Path(claims)
    capture_path.parent.mkdir(parents=True, exist_ok=True)
    claims_path.mkdir(parents=True, exist_ok=True)
    max_calls = int(maximum)

    from astrolabe.evaluators.base import Evaluator
    from astrolabe.evaluators.utils import StandardScores
    from astrolabe.model_execution import capture_random_state

    original = Evaluator.evaluate

    def _resident_observation(evaluator):
        runner = getattr(evaluator, "model_runner", None)
        client = getattr(runner, "_client", None)
        owner = getattr(client, "owner", None)
        if owner is None:
            return {
                "available": False,
                "reason": "no_resident_daemon_owner",
            }
        status_method = getattr(owner, "status", None)
        if not callable(status_method):
            return {
                "available": False,
                "reason": "resident_owner_has_no_status",
            }
        try:
            # status() lazily starts an owned daemon.  Reading pid is
            # observational and lets the harness avoid creating a model owner
            # or loading a model just to time an unavailable status.
            if getattr(owner, "pid", None) is None:
                return {
                    "available": False,
                    "reason": "resident_owner_not_started",
                }
            status = status_method()
        except Exception as error:
            return {
                "available": False,
                "reason": "resident_status_failed",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        return {
            "available": True,
            "pid": status.get("pid"),
            "model_token": status.get("identity"),
            "generation": status.get("generation"),
            "state": status.get("state"),
            "request_count": status.get("request_count"),
            "completed_count": status.get("completed_count"),
            "failed_count": status.get("failed_count"),
            "model_load_count": status.get("model_load_count"),
        }

    def _claim():
        for index in range(max_calls + 1):
            marker = claims_path / (str(index) + ".claim")
            try:
                descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                continue
            os.close(descriptor)
            return index
        raise RuntimeError("acceptance evaluator-call claim space exhausted")

    def _record(record):
        payload = pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL)
        descriptor = os.open(capture_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(descriptor, len(payload).to_bytes(8, "big"))
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def wrapped(self, inputs=None, runtime_config=None):
        index = _claim()
        if index >= max_calls:
            raise RuntimeError(
                "acceptance budget exceeded before model call "
                f"({index + 1} > {max_calls}); no replay"
            )
        before = capture_random_state()
        resident_before = _resident_observation(self)
        base = {
            "index": index,
            "evaluator": type(self).__name__,
            "evaluator_alias": getattr(self, "alias", None),
            "step": None if runtime_config is None else runtime_config.get("step"),
            "rng_before": before,
            "resident_status_before": resident_before,
        }
        started = time.perf_counter()
        try:
            result = original(self, inputs, runtime_config)
        except BaseException as error:
            elapsed_s = time.perf_counter() - started
            base.update({
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "wall_elapsed_s": elapsed_s,
                "resident_status_after": _resident_observation(self),
            })
            _record(base)
            raise
        elapsed_s = time.perf_counter() - started
        resident_after = _resident_observation(self)
        if type(result) is StandardScores:
            try:
                _record({
                    **base,
                    "status": "ok",
                    "scores": result,
                    "wall_elapsed_s": elapsed_s,
                    "rng_after": capture_random_state(),
                    "resident_status_after": resident_after,
                })
            except BaseException as error:
                _record({
                    **base,
                    "status": "capture_failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "wall_elapsed_s": elapsed_s,
                    "rng_after": capture_random_state(),
                    "resident_status_after": resident_after,
                })
        else:
            _record({
                **base,
                "status": "unsupported_result",
                "result_type": type(result).__name__,
                "wall_elapsed_s": elapsed_s,
                "rng_after": capture_random_state(),
                "resident_status_after": resident_after,
            })
        return result

    if not getattr(Evaluator.evaluate, "_astrolabe_acceptance_wrapped", False):
        wrapped._astrolabe_acceptance_wrapped = True
        Evaluator.evaluate = wrapped

_acceptance_install()
'''.lstrip()


def _install_capture_environment(case_root: Path) -> tuple[Path, Path, dict[str, str]]:
    instrumentation = case_root / "instrumentation"
    instrumentation.mkdir(parents=True, exist_ok=True)
    (instrumentation / "sitecustomize.py").write_text(_SITECUSTOMIZE, encoding="utf-8")
    capture = case_root / "capture.pickle"
    claims = case_root / "call-claims"
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (str(instrumentation), str(REPO_ROOT), existing or "")
        if part
    )
    env["ASTROLABE_ACCEPTANCE_CAPTURE"] = str(capture)
    env["ASTROLABE_ACCEPTANCE_CLAIMS"] = str(claims)
    env["ASTROLABE_ACCEPTANCE_MAX_CALLS"] = str(MAX_EVALUATIONS)
    env[MAX_EVALUATIONS_ENV] = str(MAX_EVALUATIONS)
    env["ASTROLABE_RUNTIME_ACTIVE"] = "1"
    # The reference and all candidate frontdoors must use the same requested
    # stream. Explicit empty CUDA_VISIBLE_DEVICES remains untouched.
    env["PYTHONHASHSEED"] = str(SEED)
    return capture, claims, env


def _run_process(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    stdout_log: Path,
    stderr_log: Path,
    timeout_s: float,
) -> ProcessResult:
    """Launch exactly once; a timeout terminates the process group and returns."""

    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    stderr_log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with stdout_log.open("wb") as stdout, stderr_log.open("wb") as stderr:
        process = subprocess.Popen(
            [str(item) for item in command],
            cwd=str(cwd),
            env=dict(env),
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
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                returncode = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                returncode = process.wait()
    return ProcessResult(
        returncode=returncode,
        timed_out=timed_out,
        duration_s=time.monotonic() - started,
        stdout_log=stdout_log,
        stderr_log=stderr_log,
    )


def _read_capture_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        while True:
            header = handle.read(8)
            if not header:
                break
            if len(header) != 8:
                raise AcceptanceError(f"truncated capture header in {path}")
            size = int.from_bytes(header, "big")
            if size <= 0 or size > 256 * 1024 * 1024:
                raise AcceptanceError(f"invalid capture frame size {size} in {path}")
            payload = handle.read(size)
            if len(payload) != size:
                raise AcceptanceError(f"truncated capture payload in {path}")
            records.append(pickle.loads(payload))
    return records


def _encode_tree(
    value: Any,
    *,
    prefix: str,
    arrays: dict[str, np.ndarray],
) -> Any:
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        arrays[prefix] = np.asarray(value, dtype=np.float64)
        return {
            "$array": {
                "key": prefix,
                "dtype": np.dtype(np.float64).str,
                "shape": [],
            }
        }
    if isinstance(value, Path):
        return {"$path": str(value)}
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise UnsupportedEvidence("object-dtype score array has no exact codec")
        key = prefix
        arrays[key] = np.asarray(value)
        return {
            "$array": {
                "key": key,
                "dtype": value.dtype.str,
                "shape": list(value.shape),
            }
        }
    if isinstance(value, np.generic):
        if value.dtype.hasobject:
            raise UnsupportedEvidence("object-dtype score scalar has no exact codec")
        arrays[prefix] = np.asarray(value)
        return {
            "$array": {
                "key": prefix,
                "dtype": value.dtype.str,
                "shape": [],
            }
        }
    if isinstance(value, Mapping):
        return {
            str(key): _encode_tree(
                item,
                prefix=f"{prefix}__{key}",
                arrays=arrays,
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _encode_tree(item, prefix=f"{prefix}__{index:06d}", arrays=arrays)
            for index, item in enumerate(value)
        ]
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "$dataclass": type(value).__qualname__,
            "fields": {
                field.name: _encode_tree(
                    getattr(value, field.name),
                    prefix=f"{prefix}__{field.name}",
                    arrays=arrays,
                )
                for field in fields(value)
            },
        }
    raise UnsupportedEvidence(
        f"{type(value).__module__}.{type(value).__qualname__} has no exact evidence codec"
    )


def _score_snapshot(scores: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    from astrolabe.evaluators.utils import StandardScores

    if type(scores) is not StandardScores:
        raise UnsupportedEvidence(
            f"expected StandardScores, got {type(scores).__module__}.{type(scores).__qualname__}"
        )
    poses = getattr(scores, "complex_poses", None)
    if poses not in (None, {}) and getattr(poses, "poses", None) not in (None, {}):
        raise UnsupportedEvidence(
            "StandardScores.complex_poses is non-empty; an exact pose codec is required"
        )
    arrays: dict[str, np.ndarray] = {}
    metadata = _encode_tree(scores, prefix="scores", arrays=arrays)
    extensions = {
        key: value
        for key, value in getattr(scores, "__dict__", {}).items()
        if key not in getattr(scores, "__dataclass_fields__", {})
    }
    metadata["$extensions"] = _encode_tree(
        extensions,
        prefix="scores__extensions",
        arrays=arrays,
    )
    return arrays, metadata


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


def _structure_atoms(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".pdb":
        coords = []
        metadata: dict[str, list[str]] = {
            "record": [],
            "chain": [],
            "residue": [],
            "sequence_id": [],
            "atom": [],
        }
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.startswith(("ATOM", "HETATM")):
                continue
            try:
                coords.append(
                    [
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ]
                )
            except ValueError as error:
                raise AcceptanceError(f"invalid PDB coordinate in {path}: {line}") from error
            metadata["record"].append(line[:6].strip())
            metadata["atom"].append(line[12:16].strip())
            metadata["residue"].append(line[17:20].strip())
            metadata["chain"].append(line[21:22].strip())
            metadata["sequence_id"].append(line[22:26].strip())
        return np.asarray(coords, dtype=np.float64), metadata

    if suffix not in {".cif", ".mmcif"}:
        raise UnsupportedEvidence(f"unsupported structure format for atom evidence: {path}")
    headers: list[str] = []
    rows: list[list[str]] = []
    in_atom_site = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("_atom_site."):
            headers.append(line[len("_atom_site.") :])
            in_atom_site = True
            continue
        if not in_atom_site or not headers:
            continue
        if not line or line.startswith("#"):
            break
        if line == "loop_" or line.startswith("_"):
            continue
        values = _parse_cif_line(line)
        if len(values) == len(headers):
            rows.append(values)
    if not rows:
        raise AcceptanceError(f"no atom_site rows in {path}")
    index = {name: position for position, name in enumerate(headers)}
    required = ["Cartn_x", "Cartn_y", "Cartn_z"]
    missing = [name for name in required if name not in index]
    if missing:
        raise AcceptanceError(f"missing atom coordinate columns {missing} in {path}")
    coords = np.asarray(
        [
            [float(row[index["Cartn_x"]]), float(row[index["Cartn_y"]]), float(row[index["Cartn_z"]])]
            for row in rows
        ],
        dtype=np.float64,
    )
    selected = {
        "record": "group_PDB",
        "chain": "auth_asym_id",
        "label_chain": "label_asym_id",
        "residue": "label_comp_id",
        "sequence_id": "label_seq_id",
        "atom": "label_atom_id",
    }
    metadata = {
        key: [row[index[column]] for row in rows]
        for key, column in selected.items()
        if column in index
    }
    return coords, metadata


def _safe_key(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in value
    )


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **dict(arrays))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _score_and_coordinate_evidence(
    case_root: Path,
    records: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[str, np.ndarray],
    Any,
    dict[str, np.ndarray],
    Any,
    dict[str, np.ndarray],
    Any,
]:
    score_arrays: dict[str, np.ndarray] = {}
    score_metadata: list[Any] = []
    rng_arrays: dict[str, np.ndarray] = {}
    rng_metadata: list[Any] = []
    coordinate_arrays: dict[str, np.ndarray] = {}
    coordinate_metadata: list[Any] = []
    structures = case_root / "evidence" / "structures"
    structures.mkdir(parents=True, exist_ok=True)
    for record_index, record in enumerate(records):
        score_row: dict[str, Any] = {
            "index": int(record["index"]),
            "status": record.get("status"),
            "evaluator": record.get("evaluator"),
            "evaluator_alias": record.get("evaluator_alias"),
            "step": record.get("step"),
            "rng_before": _encode_tree(
                record.get("rng_before"),
                prefix=f"rng__{record_index:03d}__before",
                arrays=rng_arrays,
            ),
            "rng_after": _encode_tree(
                record.get("rng_after"),
                prefix=f"rng__{record_index:03d}__after",
                arrays=rng_arrays,
            ),
        }
        rng_metadata.append(
            {
                "index": int(record["index"]),
                "step": record.get("step"),
                "before": score_row["rng_before"],
                "after": score_row["rng_after"],
            }
        )
        if record.get("status") != "ok":
            score_row.update(
                {
                    "error_type": record.get("error_type"),
                    "error": record.get("error"),
                    "traceback": record.get("traceback"),
                }
            )
            score_metadata.append(score_row)
            continue
        arrays, encoded_scores = _score_snapshot(record["scores"])
        for key, value in arrays.items():
            full_key = f"evaluation_{record_index:03d}__{key}"
            if full_key in score_arrays:
                raise AcceptanceError(f"duplicate score evidence key {full_key}")
            score_arrays[full_key] = value
        score_row["scores"] = encoded_scores
        score_metadata.append(score_row)

        scores = record["scores"]
        for assembly, source in dict(scores.complex_paths).items():
            source_path = Path(source)
            if not source_path.is_file():
                raise AcceptanceError(
                    f"evaluator returned missing structure for {assembly!r}: {source_path}"
                )
            destination = structures / (
                f"evaluation_{record_index:03d}__{_safe_key(str(assembly))}"
                f"{source_path.suffix or '.cif'}"
            )
            shutil.copy2(source_path, destination)
            coords, atom_metadata = _structure_atoms(source_path)
            key = (
                f"evaluation_{record_index:03d}__"
                f"assembly_{_safe_key(str(assembly))}"
            )
            coordinate_arrays[f"{key}__coordinates"] = coords
            coordinate_metadata.append(
                {
                    "evaluation_index": record_index,
                    "assembly": str(assembly),
                    "source_path": str(source_path),
                    "snapshot_path": str(destination),
                    "source_sha256": _sha256(source_path),
                    "coordinate_dtype": coords.dtype.str,
                    "coordinate_shape": list(coords.shape),
                    "atoms": atom_metadata,
                }
            )
    return (
        score_arrays,
        score_metadata,
        rng_arrays,
        rng_metadata,
        coordinate_arrays,
        coordinate_metadata,
    )


def _counter_delta(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    key: str,
) -> int | None:
    left = before.get(key)
    right = after.get(key)
    if (
        isinstance(left, int)
        and not isinstance(left, bool)
        and isinstance(right, int)
        and not isinstance(right, bool)
    ):
        return right - left
    return None


def _evaluator_timing_evidence(
    records: Sequence[Mapping[str, Any]],
    *,
    process_duration_s: float,
    compilation_cache: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    calls = []
    for record in records:
        raw_duration = record.get("wall_elapsed_s")
        duration_s = (
            float(raw_duration)
            if isinstance(raw_duration, (int, float))
            and not isinstance(raw_duration, bool)
            else None
        )
        before = record.get("resident_status_before") or {}
        after = record.get("resident_status_after") or {}
        before_available = bool(before.get("available"))
        after_available = bool(after.get("available"))
        resident_available = before_available and after_available
        calls.append(
            {
                "index": record.get("index"),
                "wall_elapsed_s": duration_s,
                "resident": {
                    "available": resident_available,
                    "before": before,
                    "after": after,
                    "same_pid": (
                        before.get("pid") == after.get("pid")
                        if resident_available
                        else None
                    ),
                    "same_model_token": (
                        before.get("model_token") == after.get("model_token")
                        if resident_available
                        else None
                    ),
                    "same_generation": (
                        before.get("generation") == after.get("generation")
                        if resident_available
                        else None
                    ),
                    "completed_delta": (
                        _counter_delta(before, after, "completed_count")
                        if resident_available
                        else None
                    ),
                    "request_delta": (
                        _counter_delta(before, after, "request_count")
                        if resident_available
                        else None
                    ),
                    "failed_delta": (
                        _counter_delta(before, after, "failed_count")
                        if resident_available
                        else None
                    ),
                },
            }
        )
    durations = [item["wall_elapsed_s"] for item in calls]
    resident_calls = [
        item["resident"]
        for item in calls
        if item["resident"]["available"]
    ]
    observed_pids = {
        status.get("pid")
        for item in resident_calls
        for status in (item["before"], item["after"])
        if status.get("pid") is not None
    }
    observed_tokens = {
        status.get("model_token")
        for item in resident_calls
        for status in (item["before"], item["after"])
        if status.get("model_token") is not None
    }
    observed_load_counts = {
        status.get("model_load_count")
        for item in resident_calls
        for status in (item["before"], item["after"])
        if status.get("model_load_count") is not None
    }
    complete_durations = [
        value for value in durations if value is not None
    ]
    return {
        "schema": "astrolabe.gpu-acceptance.evaluator-timing/v1",
        "evaluator_call_count": len(calls),
        "request_order_semantics": (
            "first/subsequent refer only to Evaluator.evaluate order within "
            "this process; no cold-compilation or cold-kernel claim"
        ),
        "first_request_s": (
            calls[0]["wall_elapsed_s"] if calls else None
        ),
        "subsequent_request_s": [
            item["wall_elapsed_s"] for item in calls[1:]
        ],
        "requests": calls,
        "compilation_cache": dict(compilation_cache or {}),
        "total_evaluator_wall_elapsed_s": (
            math.fsum(complete_durations)
            if len(complete_durations) == len(durations)
            else None
        ),
        "whole_process_duration_s": process_duration_s,
        "resident": {
            "observed_call_count": len(resident_calls),
            "pids": sorted(observed_pids),
            "model_tokens": sorted(observed_tokens),
            "model_load_counts": sorted(observed_load_counts),
            "completed_deltas": [
                item["completed_delta"] for item in resident_calls
            ],
            "no_reload_observed": bool(resident_calls)
            and len(observed_pids) == 1
            and len(observed_tokens) == 1
            and len(observed_load_counts) == 1
            and all(
                item["same_pid"]
                and item["same_model_token"]
                and item["same_generation"]
                for item in resident_calls
            ),
        },
        "performance_scope": (
            "single-sample evaluator-request observation only; shared or "
            "persistent compilation caches may be reused; "
            f"{len(calls)} request(s) are described descriptively and are "
            "insufficient for a statistical performance claim"
        ),
    }


def _read_decisions(run_root: Path) -> tuple[list[dict[str, Any]], str | None]:
    score_files = sorted(run_root.rglob("scores.csv"))
    rows: list[dict[str, Any]] = []
    for path in score_files:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append({"source": str(path), **row})
    if rows:
        return rows, "trajectory-scores-csv"

    beams = sorted(run_root.rglob("03_final_beam.json"))
    if not beams:
        return [], None
    projection = []
    for path in beams:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload.get("evaluations", []):
            projection.append(
                {
                    "source": str(path),
                    "evaluation_id": item.get("evaluation_id"),
                    "score": item.get("score"),
                    "candidate_identity": item.get("candidate_identity"),
                    "selected": item.get("selected"),
                }
            )
    return projection, "branching-final-beam"


def _sequence_projection(
    score_metadata: Sequence[Mapping[str, Any]],
    coordinate_metadata: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    from astrolabe.core.structure_parser import create_structure_parser

    by_eval: dict[int, dict[str, str]] = {}
    for row in coordinate_metadata:
        index = int(row["evaluation_index"])
        parser = create_structure_parser(str(row["source_path"]))
        chains = parser.parse()
        sequence = "".join(value[1] for _, value in sorted(chains.items()))
        by_eval.setdefault(index, {})[str(row["assembly"])] = sequence
    return [
        {"evaluation": str(index), **by_eval[index]}
        for index in sorted(by_eval)
    ]


def _weighted_score_projection(decisions: Sequence[Mapping[str, Any]]) -> list[Any]:
    values = []
    for row in decisions:
        raw = row.get("weighted_score", row.get("score"))
        if raw in (None, ""):
            continue
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            values.append(raw)
    return values


def _write_case_evidence(
    *,
    case_root: Path,
    run_root: Path,
    process: ProcessResult,
    capture_path: Path,
    env_manifest: Mapping[str, Any],
    plan: CasePlan,
    config: Any,
    effective_config: Any | None = None,
    execution_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    records = _read_capture_records(capture_path)
    (
        score_arrays,
        score_metadata,
        rng_arrays,
        rng_metadata,
        coordinate_arrays,
        coordinate_metadata,
    ) = _score_and_coordinate_evidence(case_root, records)
    evidence = case_root / "evidence"
    _write_npz(evidence / "standard_scores.npz", score_arrays)
    _write_npz(evidence / "rng_trace.npz", rng_arrays)
    _write_npz(evidence / "structure_coordinates.npz", coordinate_arrays)
    _atomic_json(evidence / "standard_scores.json", score_metadata)
    _atomic_json(evidence / "rng_trace.json", rng_metadata)
    _atomic_json(evidence / "structure_coordinates.json", coordinate_metadata)
    evaluator_timing = _evaluator_timing_evidence(
        records,
        process_duration_s=process.duration_s,
        compilation_cache=env_manifest.get("compilation_cache"),
    )
    _atomic_json(evidence / "evaluator_timing.json", evaluator_timing)

    decisions, decision_kind = _read_decisions(run_root)
    sequences = _sequence_projection(score_metadata, coordinate_metadata)
    weighted_scores = _weighted_score_projection(decisions)
    _atomic_json(evidence / "decisions.json", decisions)
    _atomic_json(
        evidence / "sequence_and_scores.json",
        {
            "seed": SEED,
            "sequences": sequences,
            "weighted_scores": weighted_scores,
        },
    )

    failed_capture = [
        row
        for row in score_metadata
        if row.get("status") != "ok"
    ]
    unexpected_evaluator = [
        row
        for row in score_metadata
        if row.get("evaluator") != "AlphaFold3Evaluator"
    ]
    successful_calls = len(records) - len(failed_capture)
    evidence_config = config if effective_config is None else effective_config
    result = {
        "schema": "astrolabe.gpu-acceptance.case/v1",
        "case": plan.name,
        "frontdoor": plan.frontdoor,
        "reference": plan.reference,
        "fixture": plan.fixture,
        "seed": SEED,
        "max_evaluations": MAX_EVALUATIONS,
        "acceptance_budget": acceptance_budget_record(),
        "model_calls": len(records),
        "successful_model_calls": successful_calls,
        "decision_kind": decision_kind,
        "process": {
            "returncode": process.returncode,
            "timed_out": process.timed_out,
            "duration_s": process.duration_s,
            "stdout_log": str(process.stdout_log),
            "stderr_log": str(process.stderr_log),
        },
        "environment_manifest": dict(env_manifest),
        "config": _digest_config(evidence_config),
        "artifacts": {
            "standard_scores_npz": str(evidence / "standard_scores.npz"),
            "standard_scores_json": str(evidence / "standard_scores.json"),
            "rng_trace_npz": str(evidence / "rng_trace.npz"),
            "rng_trace_json": str(evidence / "rng_trace.json"),
            "evaluator_timing_json": str(evidence / "evaluator_timing.json"),
            "structure_coordinates_npz": str(
                evidence / "structure_coordinates.npz"
            ),
            "structure_coordinates_json": str(
                evidence / "structure_coordinates.json"
            ),
            "decisions_json": str(evidence / "decisions.json"),
            "sequence_and_scores_json": str(
                evidence / "sequence_and_scores.json"
            ),
        },
        "sequences": sequences,
        "weighted_scores": weighted_scores,
        "evaluator_timing": {
            "first_request_s": evaluator_timing["first_request_s"],
            "subsequent_request_s": evaluator_timing[
                "subsequent_request_s"
            ],
            "compilation_cache": evaluator_timing["compilation_cache"],
            "whole_process_duration_s": evaluator_timing[
                "whole_process_duration_s"
            ],
            "resident": evaluator_timing["resident"],
            "performance_scope": evaluator_timing["performance_scope"],
        },
    }
    if effective_config is not None:
        result["invocation_config"] = _digest_config(config)
        result["execution_context"] = dict(execution_context or {})
    if failed_capture:
        result["status"] = "failed"
        result["error"] = "one or more evaluator calls did not return a supported StandardScores"
        result["failed_capture"] = failed_capture
    elif unexpected_evaluator:
        result["status"] = "failed"
        result["error"] = "acceptance captured a non-AF3 evaluator call"
        result["unexpected_evaluator"] = unexpected_evaluator
    elif process.timed_out:
        result["status"] = "failed"
        result["error"] = "frontdoor timed out; process group terminated; no retry"
        result["outcome"] = "unknown"
    elif process.returncode != 0:
        result["status"] = "failed"
        result["error"] = f"frontdoor returned nonzero status {process.returncode}; no retry"
    elif len(records) != MAX_EVALUATIONS:
        result["status"] = "failed"
        result["error"] = (
            f"expected exactly {MAX_EVALUATIONS} evaluator calls, observed {len(records)}"
        )
    elif not score_arrays or not coordinate_arrays:
        result["status"] = "failed"
        result["error"] = "missing StandardScores arrays or structure coordinates"
    elif decision_kind is None:
        result["status"] = "failed"
        result["error"] = "no trajectory decisions or branching report found"
    else:
        result["status"] = "evidence-complete"
    _atomic_json(case_root / "result.json", result)
    return result


def _write_evidence_failure(
    *,
    case_root: Path,
    process: ProcessResult,
    capture_path: Path,
    env_manifest: Mapping[str, Any],
    plan: CasePlan,
    config: Any,
    error: BaseException,
) -> dict[str, Any]:
    try:
        records = _read_capture_records(capture_path)
        model_calls = len(records)
    except BaseException:
        model_calls = None
    result = {
        "schema": "astrolabe.gpu-acceptance.case/v1",
        "case": plan.name,
        "frontdoor": plan.frontdoor,
        "reference": plan.reference,
        "fixture": plan.fixture,
        "seed": SEED,
        "max_evaluations": MAX_EVALUATIONS,
        "acceptance_budget": acceptance_budget_record(),
        "model_calls": model_calls,
        "successful_model_calls": None,
        "status": (
            "unsupported"
            if isinstance(error, UnsupportedEvidence)
            else "failed"
        ),
        "error_type": type(error).__name__,
        "error": str(error),
        "no_retry": True,
        "capture": str(capture_path),
        "process": {
            "returncode": process.returncode,
            "timed_out": process.timed_out,
            "duration_s": process.duration_s,
            "stdout_log": str(process.stdout_log),
            "stderr_log": str(process.stderr_log),
        },
        "environment_manifest": dict(env_manifest),
        "config": _digest_config(config),
    }
    _atomic_json(case_root / "result.json", result)
    return result


def _load_tree_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _compare_arrays(
    left_path: Path,
    right_path: Path,
) -> dict[str, Any]:
    left = _load_tree_arrays(left_path)
    right = _load_tree_arrays(right_path)
    failures = []
    max_abs: float | None = None
    for key in sorted(set(left) | set(right)):
        if key not in left or key not in right:
            failures.append({"key": key, "reason": "missing-key"})
            continue
        a = left[key]
        b = right[key]
        if a.dtype != b.dtype:
            failures.append(
                {
                    "key": key,
                    "reason": "dtype",
                    "left": a.dtype.str,
                    "right": b.dtype.str,
                }
            )
            continue
        if a.shape != b.shape:
            failures.append(
                {
                    "key": key,
                    "reason": "shape",
                    "left": list(a.shape),
                    "right": list(b.shape),
                }
            )
            continue
        if not np.array_equal(a, b, equal_nan=True):
            failures.append({"key": key, "reason": "value"})
            if np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
                difference = np.abs(a.astype(np.float64) - b.astype(np.float64))
                finite = difference[np.isfinite(difference)]
                if finite.size:
                    current = float(np.max(finite))
                    max_abs = current if max_abs is None else max(max_abs, current)
    return {
        "equal": not failures,
        "array_count": len(left),
        "failures": failures,
        "max_abs_difference": max_abs,
    }


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _scientific_equal(left: Any, right: Any) -> bool:
    if isinstance(left, float) and isinstance(right, float):
        return (math.isnan(left) and math.isnan(right)) or left == right
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _scientific_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _scientific_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _without_path_values(value: Any) -> Any:
    if isinstance(value, Mapping):
        if set(value) == {"$path"}:
            return {"$path": "<artifact-path>"}
        return {
            key: _without_path_values(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_without_path_values(item) for item in value]
    return value


def _compatible_decision_projection(
    result: Mapping[str, Any],
) -> list[dict[str, Any]]:
    projections = []
    for index, score in enumerate(result.get("weighted_scores", [])):
        projections.append(
            {
                "index": index,
                "weighted_score": score,
                "sequences": (
                    result.get("sequences", [])[index]
                    if index < len(result.get("sequences", []))
                    else None
                ),
            }
        )
    return projections


def _beam_selection_projection(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    evaluations = []
    selected = []
    for row in rows:
        if row.get("evaluation_id") is None or row.get("score") is None:
            continue
        item = {
            "evaluation_id": row["evaluation_id"],
            "score": float(row["score"]),
        }
        evaluations.append(item)
        if bool(row.get("selected", False)):
            selected.append(item)
    ranked = sorted(
        evaluations,
        key=lambda item: (-item["score"], int(item["evaluation_id"])),
    )
    expected = ranked[: len(selected)]
    return selected, _scientific_equal(selected, expected)


def _coordinate_identity_projection(rows: Sequence[Mapping[str, Any]]) -> list[Any]:
    return [
        {
            "evaluation_index": row.get("evaluation_index"),
            "assembly": row.get("assembly"),
            "coordinate_dtype": row.get("coordinate_dtype"),
            "coordinate_shape": row.get("coordinate_shape"),
            "atoms": row.get("atoms"),
        }
        for row in rows
    ]


def compare_case(
    *,
    reference_root: Path,
    case_root: Path,
) -> dict[str, Any]:
    reference = _load_json(reference_root / "result.json")
    case = _load_json(case_root / "result.json")
    if (
        reference.get("status") != "evidence-complete"
        or case.get("status") != "evidence-complete"
    ):
        return {
            "schema": "astrolabe.gpu-acceptance.comparison/v1",
            "case": case.get("case"),
            "reference": reference.get("case"),
            "status": "ISSUES",
            "reference_status": reference.get("status"),
            "case_status": case.get("status"),
            "reference_error": reference.get("error"),
            "case_error": case.get("error"),
            "exact_numeric_match": False,
            "exact_rng_trace_match": False,
            "exact_rng_trace_detail_match": False,
            "exact_score_field_topology_match": False,
            "exact_structure_coordinate_match": False,
            "exact_structure_atom_identity_match": False,
            "exact_sequence_and_weighted_score_match": False,
            "exact_decision_projection_match": False,
            "same_gpu": False,
            "same_weights": False,
            "same_scientific_config_except_execution": False,
            "config_differences": [],
        }
    score_compare = _compare_arrays(
        reference_root / "evidence" / "standard_scores.npz",
        case_root / "evidence" / "standard_scores.npz",
    )
    rng_compare = _compare_arrays(
        reference_root / "evidence" / "rng_trace.npz",
        case_root / "evidence" / "rng_trace.npz",
    )
    coordinate_compare = _compare_arrays(
        reference_root / "evidence" / "structure_coordinates.npz",
        case_root / "evidence" / "structure_coordinates.npz",
    )
    coordinate_identity_equal = _scientific_equal(
        _coordinate_identity_projection(
            _load_json(reference_root / "evidence" / "structure_coordinates.json")
        ),
        _coordinate_identity_projection(
            _load_json(case_root / "evidence" / "structure_coordinates.json")
        ),
    )
    score_topology_equal = _scientific_equal(
        _without_path_values(
            _load_json(reference_root / "evidence" / "standard_scores.json")
        ),
        _without_path_values(
            _load_json(case_root / "evidence" / "standard_scores.json")
        ),
    )
    rng_trace_detail_equal = _scientific_equal(
        _load_json(reference_root / "evidence" / "rng_trace.json"),
        _load_json(case_root / "evidence" / "rng_trace.json"),
    )
    reference_beam, reference_beam_valid = _beam_selection_projection(
        _load_json(reference_root / "evidence" / "decisions.json")
    )
    case_beam, case_beam_valid = _beam_selection_projection(
        _load_json(case_root / "evidence" / "decisions.json")
    )
    beam_selection_equal = _scientific_equal(reference_beam, case_beam)
    reference_sequences = reference.get("sequences")
    case_sequences = case.get("sequences")
    reference_weighted = reference.get("weighted_scores")
    case_weighted = case.get("weighted_scores")
    sequence_score_equal = (
        _scientific_equal(reference_sequences, case_sequences)
        and _scientific_equal(reference_weighted, case_weighted)
    )
    decision_protocol_equal = reference.get("decision_kind") == case.get("decision_kind")
    decisions_equal = (
        decision_protocol_equal
        and _scientific_equal(
            _compatible_decision_projection(reference),
            _compatible_decision_projection(case),
        )
    )
    environment_left = reference.get("environment_manifest", {}).get("gpu")
    environment_right = case.get("environment_manifest", {}).get("gpu")
    config_left = _comparable_scientific_config(reference.get("config"))
    config_right = _comparable_scientific_config(case.get("config"))
    same_gpu = _scientific_equal(environment_left, environment_right)
    same_scientific_config = _scientific_equal(config_left, config_right)
    config_differences = _config_differences(config_left, config_right)
    same_weights = _scientific_equal(
        reference.get("environment_manifest", {}).get("weights"),
        case.get("environment_manifest", {}).get("weights"),
    )
    passed = (
        reference.get("status") == "evidence-complete"
        and case.get("status") == "evidence-complete"
        and same_gpu
        and same_weights
        and same_scientific_config
        and score_compare["equal"]
        and rng_compare["equal"]
        and rng_trace_detail_equal
        and score_topology_equal
        and coordinate_compare["equal"]
        and coordinate_identity_equal
        and sequence_score_equal
        and decisions_equal
        and reference_beam_valid
        and case_beam_valid
        and beam_selection_equal
    )
    return {
        "schema": "astrolabe.gpu-acceptance.comparison/v1",
        "case": case["case"],
        "reference": reference["case"],
        "status": "PASS" if passed else "ISSUES",
        "exact_numeric_match": score_compare["equal"],
        "exact_rng_trace_match": rng_compare["equal"],
        "exact_rng_trace_detail_match": rng_trace_detail_equal,
        "exact_score_field_topology_match": score_topology_equal,
        "exact_structure_coordinate_match": coordinate_compare["equal"],
        "exact_structure_atom_identity_match": coordinate_identity_equal,
        "exact_sequence_and_weighted_score_match": sequence_score_equal,
        "exact_decision_projection_match": decisions_equal,
        "exact_beam_selection_match": beam_selection_equal,
        "reference_beam_selection_matches_score_order": reference_beam_valid,
        "case_beam_selection_matches_score_order": case_beam_valid,
        "same_gpu": same_gpu,
        "same_weights": same_weights,
        "same_scientific_config_except_execution": same_scientific_config,
        "config_differences": config_differences,
        "score_arrays": score_compare,
        "rng_trace": rng_compare,
        "structure_coordinates": coordinate_compare,
        "reference_sequences": reference_sequences,
        "case_sequences": case_sequences,
        "reference_weighted_scores": reference_weighted,
        "case_weighted_scores": case_weighted,
        "reference_decision_kind": reference.get("decision_kind"),
        "case_decision_kind": case.get("decision_kind"),
        "reference_beam_selection": reference_beam,
        "case_beam_selection": case_beam,
    }


def _api_worker_main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--mode",
        choices=(
            "default-api",
            "generic-api",
            "native-reference",
            "native-generic",
        ),
        required=True,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fixture", required=True)
    arguments = parser.parse_args(argv)

    from omegaconf import OmegaConf

    cfg = OmegaConf.load(arguments.config)
    if arguments.mode in {"default-api", "generic-api"}:
        from astrolabe.app import Astrolabe

        Astrolabe(cfg).run()
        return 0
    if arguments.mode == "native-reference":
        _run_native_reference(cfg)
        return 0
    if arguments.mode == "native-generic":
        _run_native_generic_reference(cfg)
        return 0
    raise AssertionError(arguments.mode)


def _run_native_reference(cfg: Any) -> None:
    """Perform the configured bounded native root/advance callbacks."""
    from astrolabe.app import _load_run_module
    from astrolabe.core import global_config
    from astrolabe.model_execution import model_run_scope

    run = _load_run_module()
    global_config.reset()
    args = run.parse_arguments(cfg)
    run.setup_logging(args.out_dir, getattr(args, "logging", None))
    try:
        loss_computer, stage_manager, inputs = run.initialize_pipeline_components(args)
        runtime_config = {
            "mutator_stack_pre_update": bool(
                getattr(getattr(args, "mutator", None), "stack_pre_update", False)
            ),
            "state_conflict_diagnostics": getattr(
                getattr(args, "mutator", None),
                "state_conflict_diagnostics",
                None,
            ),
        }
        with model_run_scope():
            initial = run.load_or_evaluate_initial(
                args,
                stage_manager.current_evaluator,
                inputs,
                runtime_config,
                args.task_name,
                loss_computer=loss_computer,
            )
            decisions, current = run._initialize_optimization_state(
                args,
                loss_computer,
                initial,
                stage_manager,
                runtime_config,
            )
            if int(args.steps) != MAX_EVALUATIONS:
                raise AcceptanceError(
                    "native reference requires "
                    f"steps={MAX_EVALUATIONS}, got {args.steps}"
                )
            for step in range(1, MAX_EVALUATIONS):
                current, _accepted, early_stop = (
                    run._advance_optimization_state(
                        args,
                        loss_computer,
                        current,
                        stage_manager,
                        runtime_config,
                        decisions,
                        step,
                    )
                )
                if early_stop:
                    raise AcceptanceError(
                        "native reference stopped before the requested "
                        f"{MAX_EVALUATIONS} evaluator calls"
                    )
            decisions.dump_results(global_config.out_dir, incremental=False)
    finally:
        run._shutdown_pipeline_resources()


def _run_native_generic_reference(cfg: Any) -> None:
    """Run the exact generic root/proposal/root-successor prediction protocol."""
    from astrolabe.app import _load_run_module
    from astrolabe.argparser import set_random_seeds
    from astrolabe.core import global_config
    from astrolabe.design.configured_refinement import _generic_proposal_seed
    from astrolabe.model_execution import model_run_scope

    run = _load_run_module()
    global_config.reset()
    args = run.parse_arguments(cfg)
    run.setup_logging(args.out_dir, getattr(args, "logging", None))
    protocol = {
        "mode": "native-generic-reference",
        "seed": SEED,
        "root_step": 0,
        "proposal_step": 1,
        "proposal_seed": _generic_proposal_seed(SEED, 1),
        "prediction_seed": SEED,
    }
    try:
        loss_computer, stage_manager, _inputs = (
            run.initialize_pipeline_components(args)
        )
        runtime_config = {
            "mutator_stack_pre_update": False,
            "state_conflict_diagnostics": getattr(
                getattr(args, "mutator", None),
                "state_conflict_diagnostics",
                None,
            ),
        }
        evaluator = stage_manager.current_evaluator
        mutators = tuple(stage_manager.current_mutators)
        assemblies = list(stage_manager.current_oligo)
        with model_run_scope():
            global_config.set_variable("step", 0)
            root_scores = run.perform_evaluation(
                args,
                evaluator,
                runtime_config,
                0,
                assemblies,
                loss_computer=loss_computer,
            )
            root_score, _ = loss_computer.compute_loss(
                root_scores,
                runtime_config,
            )

            set_random_seeds(protocol["proposal_seed"])
            global_config.set_variable("step", 1)
            run.execute_mutation_step(
                mutators,
                root_scores,
                runtime_config,
                1,
            )
            global_config.segment_manager.apply_mutations()
            child_scores = run.perform_evaluation(
                args,
                evaluator,
                runtime_config,
                1,
                assemblies,
                loss_computer=loss_computer,
            )
            child_score, _ = loss_computer.compute_loss(
                child_scores,
                runtime_config,
            )

        evaluations = [
            {
                "evaluation_id": 0,
                "generation": 0,
                "parent_id": None,
                "proposal_step": 0,
                "score": float(root_score),
                "seed": SEED,
            },
            {
                "evaluation_id": 1,
                "generation": 1,
                "parent_id": 0,
                "proposal_step": 1,
                "score": float(child_score),
                "seed": SEED,
            },
        ]
        if not all(math.isfinite(row["score"]) for row in evaluations):
            raise ValueError("native generic reference produced a non-finite score")
        ranked = sorted(
            evaluations,
            key=lambda row: (-row["score"], row["evaluation_id"]),
        )
        selected_id = ranked[0]["evaluation_id"]
        for evaluation in evaluations:
            evaluation["selected"] = (
                evaluation["evaluation_id"] == selected_id
            )
        payload = {
            "routing": {
                "algorithm": "branching",
                "owner": "native_refinement",
                "reference_mode": "native-generic-reference",
            },
            "reference_protocol": protocol,
            "evaluations": evaluations,
            "selected_evaluation_ids": [selected_id],
        }
        _atomic_json(Path(args.out_dir) / "03_final_beam.json", payload)
        _atomic_json(Path(args.out_dir) / "reference_protocol.json", protocol)
    finally:
        run._shutdown_pipeline_resources()


def _task_name(kind: str, case_name: str) -> str:
    token = _safe_key(f"{kind}_{case_name}")[:80]
    return f"gpu_acceptance_{token}_{secrets.token_hex(3)}"


def _prepare_reference(
    *,
    profile: Path,
    out_root: Path,
    plan: CasePlan,
    environment: Mapping[str, Any],
    python_bin: Path,
    timeout_s: float,
) -> tuple[dict[str, Any], Any]:
    reference = case_plan(plan.name).reference
    fixture = "pair-shared-fixed" if reference == "native-pair" else "single-20aa"
    reference_root = out_root / "references" / reference
    if (reference_root / "result.json").is_file():
        result = _load_json(reference_root / "result.json")
        cfg = _build_config(
            profile=profile,
            fixture=fixture,
            out_dir=reference_root / "run",
            task_name=result["config"].get(
                "task_name",
                _task_name("reference", reference),
            ),
            execution="in_process",
            generic_search=False,
        )
        return result, cfg

    reference_root.mkdir(parents=True, exist_ok=True)
    task_name = _task_name("reference", reference)
    config = _build_config(
        profile=profile,
        fixture=fixture,
        out_dir=reference_root / "run",
        task_name=task_name,
        execution="in_process",
        generic_search=False,
    )
    config_path = reference_root / "config.yaml"
    _save_standalone(config, config_path)
    capture_path, _claims, proc_env = _install_capture_environment(reference_root)
    worker_mode = (
        "native-generic"
        if reference == "native-generic"
        else "native-reference"
    )
    command = [
        str(python_bin),
        str(Path(__file__).resolve()),
        "--api-worker",
        "--mode",
        worker_mode,
        "--config",
        str(config_path),
        "--fixture",
        fixture,
    ]
    process = _run_process(
        command,
        cwd=REPO_ROOT,
        env=proc_env,
        stdout_log=reference_root / "stdout.log",
        stderr_log=reference_root / "stderr.log",
        timeout_s=timeout_s,
    )
    evidence_plan = CasePlan(
        name=reference,
        reference="none",
        frontdoor=(
            "run.py perform_evaluation root + "
            "_generic_proposal_seed(17,1) + execute_mutation_step + "
            "perform_evaluation"
            if reference == "native-generic"
            else "run.py _initialize_optimization_state + _advance_optimization_state"
        ),
        fixture=fixture,
    )
    try:
        result = _write_case_evidence(
            case_root=reference_root,
            run_root=reference_root / "run",
            process=process,
            capture_path=capture_path,
            env_manifest=environment,
            plan=evidence_plan,
            config=config,
        )
    except Exception as error:
        result = _write_evidence_failure(
            case_root=reference_root,
            process=process,
            capture_path=capture_path,
            env_manifest=environment,
            plan=evidence_plan,
            config=config,
            error=error,
        )
    if reference == "native-generic":
        protocol_files = sorted(reference_root.rglob("reference_protocol.json"))
        if len(protocol_files) != 1:
            result["status"] = "failed"
            result["error"] = (
                "native-generic reference did not emit exactly one "
                "reference_protocol.json"
            )
        else:
            snapshot = reference_root / "reference_protocol.json"
            shutil.copy2(protocol_files[0], snapshot)
            result["reference_protocol"] = _load_json(snapshot)
        _atomic_json(reference_root / "result.json", result)
    return result, config


def _cli_command(
    *,
    plan: CasePlan,
    python_bin: Path,
    config_path: Path,
    paper_input: Path,
    case_root: Path,
) -> tuple[list[str], Path]:
    if plan.name == "default-cli":
        command = [
            str(python_bin),
            "-m",
            "astrolabe.cli",
            "run",
            "--standalone",
            str(config_path),
            "--no-chdir",
        ]
        return command, case_root / "run"
    if plan.name == "paper-cli":
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
            str(_load_paper_task_name(paper_input)),
            "--out-dir",
            str(case_root / "run"),
            "--seed",
            str(SEED),
            "--num-runs",
            "1",
        ]
        return command, case_root / "run"
    if plan.name == "candidate-batch":
        command = [
            str(python_bin),
            "-m",
            "astrolabe.cli",
            "experimental",
            "candidate-batch",
            "--config-yaml",
            str(config_path),
            "--out-root",
            str(case_root / "run"),
            "--runner",
            str(REPO_ROOT / "scripts" / "astrolabe"),
            "--num-candidates",
            "1",
            "--min-completed",
            "1",
            "--max-in-flight",
            "1",
            "--max-wait-s",
            "900",
            "--decision-rounds",
            "1",
            "--candidate-steps",
            str(MAX_EVALUATIONS),
            "--format",
            "json",
        ]
        return command, case_root / "run"
    raise ValueError(plan.name)


def _load_paper_task_name(path: Path) -> str:
    from omegaconf import OmegaConf

    value = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(value, dict):
        raise TypeError("paper input must be a mapping")
    return str(value["task_name"])


def _prepare_case(
    *,
    profile: Path,
    out_root: Path,
    plan: CasePlan,
    environment: Mapping[str, Any],
    python_bin: Path,
    timeout_s: float,
    case_root: Path | None = None,
) -> tuple[dict[str, Any], Any]:
    case_root = (
        out_root / "cases" / plan.name
        if case_root is None
        else Path(case_root)
    )
    case_root.mkdir(parents=True, exist_ok=True)
    task_name = _task_name("case", plan.name)
    generic = plan.name == "generic-api"
    config = _build_config(
        profile=profile,
        fixture=plan.fixture,
        out_dir=case_root / "run",
        task_name=task_name,
        execution="daemon",
        generic_search=generic,
        seed=_invocation_seed(plan.name),
    )
    config_path = case_root / "config.yaml"
    _save_standalone(config, config_path)
    paper_input = case_root / "paper_inputs.yaml"
    if plan.name == "paper-cli":
        from omegaconf import OmegaConf

        overlay = _paper_input_overlay(config)
        overlay["task_name"] = task_name
        paper_input.write_text(
            OmegaConf.to_yaml(OmegaConf.create(overlay), resolve=False),
            encoding="utf-8",
        )

    capture_path, _claims, proc_env = _install_capture_environment(case_root)
    if plan.name in {"default-api", "generic-api"}:
        command = [
            str(python_bin),
            str(Path(__file__).resolve()),
            "--api-worker",
            "--mode",
            plan.name,
            "--config",
            str(config_path),
            "--fixture",
            plan.fixture,
        ]
    else:
        command, _run_root = _cli_command(
            plan=plan,
            python_bin=python_bin,
            config_path=config_path,
            paper_input=paper_input,
            case_root=case_root,
        )
    process = _run_process(
        command,
        cwd=REPO_ROOT,
        env=proc_env,
        stdout_log=case_root / "stdout.log",
        stderr_log=case_root / "stderr.log",
        timeout_s=timeout_s,
    )
    try:
        effective_config = config
        execution_context = None
        if plan.name == "candidate-batch":
            effective_config, execution_context = _candidate_executed_config(
                case_root,
                config,
            )
        result = _write_case_evidence(
            case_root=case_root,
            run_root=case_root / "run",
            process=process,
            capture_path=capture_path,
            env_manifest=environment,
            plan=plan,
            config=config,
            effective_config=effective_config,
            execution_context=execution_context,
        )
    except Exception as error:
        result = _write_evidence_failure(
            case_root=case_root,
            process=process,
            capture_path=capture_path,
            env_manifest=environment,
            plan=plan,
            config=config,
            error=error,
        )
    result["command"] = command
    _atomic_json(case_root / "result.json", result)
    return result, config


def _parse_cases(arguments: argparse.Namespace) -> tuple[str, ...]:
    selected = list(arguments.case)
    if arguments.all:
        selected = list(CASE_NAMES)
    if not selected:
        selected = list(PRIORITY_CASES)
    if len(set(selected)) != len(selected):
        raise ValueError("--case values must not repeat")
    return tuple(selected)


def _dry_run(
    *,
    out_root: Path,
    profile: Path,
    cases: Sequence[str],
) -> None:
    plan_rows = []
    for name in cases:
        plan = case_plan(name)
        case_root = out_root / "cases" / name
        case_root.mkdir(parents=True, exist_ok=True)
        config = _build_config(
            profile=profile,
            fixture=plan.fixture,
            out_dir=case_root / "run",
            task_name=f"gpu_acceptance_{_safe_key(name)}",
            execution="daemon",
            generic_search=name == "generic-api",
            seed=_invocation_seed(name),
        )
        config_path = case_root / "config.yaml"
        _save_standalone(config, config_path)
        paper_input = case_root / "paper_inputs.yaml"
        if name == "paper-cli":
            from omegaconf import OmegaConf

            overlay = _paper_input_overlay(config)
            overlay["task_name"] = f"gpu_acceptance_{_safe_key(name)}"
            paper_input.write_text(
                OmegaConf.to_yaml(OmegaConf.create(overlay), resolve=False),
                encoding="utf-8",
            )
        if name in {"default-api", "generic-api"}:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--api-worker",
                "--mode",
                name,
                "--config",
                str(config_path),
                "--fixture",
                plan.fixture,
            ]
        else:
            command, _ = _cli_command(
                plan=plan,
                python_bin=Path(sys.executable),
                config_path=config_path,
                paper_input=paper_input,
                case_root=case_root,
            )
        plan_rows.append(
            {
                "case": name,
                "reference": plan.reference,
                "fixture": plan.fixture,
                "frontdoor": plan.frontdoor,
                "command": command,
                "config": _digest_config(config),
                "invocation_seed": int(config.seed),
                "effective_seed": (
                    SEED if name == "candidate-batch" else int(config.seed)
                ),
            }
        )
    unique_references = {
        case_plan(name).reference for name in cases
    }
    _atomic_json(
        out_root / "plan.json",
        {
            "schema": "astrolabe.gpu-acceptance.plan/v1",
            "status": "planned",
            "dry_run": True,
            "coverage_scope": "implemented-five-family-only",
            "out_root": str(out_root),
            "acceptance_budget": acceptance_budget_record(),
            "native_call_budget": {
                "reference_count": len(unique_references),
                "reference_calls": len(unique_references) * MAX_EVALUATIONS,
                "case_count": len(cases),
                "case_calls": len(cases) * MAX_EVALUATIONS,
                "total_calls": (
                    len(unique_references) + len(cases)
                )
                * MAX_EVALUATIONS,
            },
            "cases": plan_rows,
        },
    )


def _next_resume_root(out_root: Path) -> Path:
    root = out_root / "resumes"
    root.mkdir(parents=True, exist_ok=True)
    for index in range(1, 10_000):
        candidate = root / f"default_api_{index:03d}"
        if not candidate.exists():
            candidate.mkdir()
            return candidate
    raise AcceptanceError("resume attempt namespace is exhausted")


def _resume_default_api(
    *,
    out_root: Path,
    profile: Path,
    python_bin: Path,
    timeout_s: float,
    frozen_source_root: Path | None,
    frozen_source_manifest: Path | None,
    expected_af3_module_path: Path,
) -> int:
    attempt_root = _next_resume_root(out_root)
    try:
        runtime_imports = _preflight_runtime_imports()
    except Exception as error:
        failure = {
            "schema": "astrolabe.gpu-acceptance.runtime-import-preflight/v1",
            "status": "failed",
            "model_calls": 0,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        _atomic_json(attempt_root / "runtime_import_preflight.json", failure)
        print(
            json.dumps(
                {
                    "status": "ISSUES",
                    "attempt": str(attempt_root),
                    "error": str(error),
                }
            )
        )
        return 1
    _atomic_json(
        attempt_root / "runtime_import_preflight.json",
        runtime_imports,
    )
    source_manifest = _read_source_manifest()
    _atomic_json(
        attempt_root / "source_manifest.json",
        source_manifest.as_dict(),
    )
    manifest_cfg = _build_config(
        profile=profile,
        fixture="single-20aa",
        out_dir=attempt_root / "environment-preflight",
        task_name="resume_environment_preflight",
        execution="daemon",
        generic_search=False,
    )
    try:
        environment = _environment_manifest(manifest_cfg, source_manifest)
        _atomic_json(
            attempt_root / "resume_environment_manifest.json",
            environment,
        )
        validation, _reference_cfg = _validate_resume_baseline(
            out_root=out_root,
            profile=profile,
            environment=environment,
            current_source=source_manifest,
            runtime_imports=runtime_imports,
            expected_af3_module_path=expected_af3_module_path,
            frozen_source_root=frozen_source_root,
            frozen_source_manifest=frozen_source_manifest,
        )
    except Exception as error:
        failure = {
            "schema": "astrolabe.gpu-acceptance.resume-baseline-validation/v1",
            "status": "failed",
            "model_calls": 0,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        _atomic_json(attempt_root / "baseline_validation.json", failure)
        print(
            json.dumps(
                {
                    "status": "ISSUES",
                    "attempt": str(attempt_root),
                    "error": str(error),
                }
            )
        )
        return 1

    _atomic_json(attempt_root / "baseline_validation.json", validation)
    plan = case_plan(DEFAULT_RESUME_CASE)
    source_checks = []

    def guard(phase: str) -> None:
        current = _check_source_manifest(
            source_manifest,
            out_root=attempt_root,
            phase=phase,
            check_index=len(source_checks) + 1,
        )
        source_checks.append(
            {
                "phase": phase,
                "fingerprint": current.fingerprint,
                "matches_expected": True,
            }
        )
        _atomic_json(
            attempt_root / "source_checks.json",
            {
                "schema": "astrolabe.gpu-acceptance.source-check-summary/v1",
                "fingerprint": source_manifest.fingerprint,
                "checks": source_checks,
            },
        )

    try:
        guard("resume:case:default-api:before")
        case_result, _cfg = _prepare_case(
            profile=profile,
            out_root=out_root,
            plan=plan,
            environment=environment,
            python_bin=python_bin,
            timeout_s=timeout_s,
            case_root=attempt_root,
        )
        case_result["resume_from"] = str(
            out_root / "references" / "native-single"
        )
        case_result["resume_baseline_validation"] = str(
            attempt_root / "baseline_validation.json"
        )
        guard("resume:case:default-api:after")
        case_result["source_fingerprint"] = source_manifest.fingerprint
        baseline_hashes_after = _evidence_file_hashes(
            out_root / "references" / "native-single"
        )
        if baseline_hashes_after != validation["baseline_evidence_sha256"]:
            raise SourceChanged("native-single baseline evidence changed during resume")
        comparison = compare_case(
            reference_root=out_root / "references" / "native-single",
            case_root=attempt_root,
        )
        _atomic_json(attempt_root / "comparison.json", comparison)
        _atomic_json(attempt_root / "result.json", case_result)
        status = (
            "PASS"
            if case_result.get("status") == "evidence-complete"
            and comparison.get("status") == "PASS"
            else "ISSUES"
        )
        summary = {
            "schema": "astrolabe.gpu-acceptance.resume-summary/v1",
            "status": status,
            "case": DEFAULT_RESUME_CASE,
            "attempt": str(attempt_root),
            "baseline_reused": True,
            "baseline_model_calls": MAX_EVALUATIONS,
            "case_model_calls": case_result.get("model_calls"),
            "no_retry": True,
            "failed_logs_preserved": str(
                out_root / "cases" / DEFAULT_RESUME_CASE
            ),
            "comparison": str(attempt_root / "comparison.json"),
        }
        _atomic_json(attempt_root / "summary.json", summary)
        print(json.dumps({"status": status, "summary": str(attempt_root / "summary.json")}))
        return 0 if status == "PASS" else 1
    except Exception as error:
        existing_result = attempt_root / "result.json"
        if existing_result.is_file():
            try:
                failed_case_result = _load_json(existing_result)
            except Exception:
                failed_case_result = None
            if isinstance(failed_case_result, dict):
                failed_case_result["status"] = "failed"
                failed_case_result["error_type"] = type(error).__name__
                failed_case_result["error"] = str(error)
                _atomic_json(existing_result, failed_case_result)
        failure = {
            "schema": "astrolabe.gpu-acceptance.resume-summary/v1",
            "status": "ISSUES",
            "case": DEFAULT_RESUME_CASE,
            "attempt": str(attempt_root),
            "error_type": type(error).__name__,
            "error": str(error),
            "no_retry": True,
            "failed_logs_preserved": str(
                out_root / "cases" / DEFAULT_RESUME_CASE
            ),
        }
        _atomic_json(attempt_root / "summary.json", failure)
        print(json.dumps({"status": "ISSUES", "summary": str(attempt_root / "summary.json")}))
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run bounded real-GPU AF3 acceptance for actual Astrolabe frontdoors. "
            "The process-wide evaluator cap defaults to two; the explicit "
            "throughput profile allows four only for default-api/default-cli. "
            "No case is retried."
        )
    )
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--case", action="append", choices=CASE_NAMES, default=[])
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Run all five implemented acceptance families only; this does not "
            "assert coverage of every repository caller."
        ),
    )
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse a validated native-single baseline for default-api only. "
            "Creates a new numbered resume attempt without modifying failed logs."
        ),
    )
    parser.add_argument(
        "--resume-source-root",
        type=Path,
        default=None,
        help=(
            "Frozen default_v1 source tree used to verify numerical source "
            "identity before a resumed model call."
        ),
    )
    parser.add_argument(
        "--resume-source-manifest",
        type=Path,
        default=None,
        help=(
            "Capsule source_manifest.json used to verify numerical source "
            "identity when the baseline predates embedded source fingerprints."
        ),
    )
    parser.add_argument(
        "--resume-af3-module-path",
        type=Path,
        default=None,
        help=(
            "Exact AF3 package origin measured in the original baseline. "
            "Required for --resume."
        ),
    )
    parser.add_argument("--api-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--mode",
        choices=(
            "default-api",
            "generic-api",
            "native-reference",
            "native-generic",
        ),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--config", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--fixture", help=argparse.SUPPRESS)
    arguments = parser.parse_args(argv)

    try:
        configured_max_evaluations = (
            acceptance_max_evaluations_from_environment()
        )
    except AcceptanceError as error:
        parser.error(str(error))
    _configure_max_evaluations(configured_max_evaluations)

    if arguments.api_worker:
        if arguments.mode is None or arguments.config is None:
            parser.error("internal API worker requires --mode and --config")
        if (
            MAX_EVALUATIONS == THROUGHPUT_MAX_EVALUATIONS
            and arguments.mode in {"generic-api", "native-generic"}
        ):
            parser.error(
                f"{MAX_EVALUATIONS_ENV}=4 is unsupported for "
                f"internal mode {arguments.mode}"
            )
        return _api_worker_main(
            [
                "--mode",
                arguments.mode,
                "--config",
                str(arguments.config),
                "--fixture",
                str(arguments.fixture or ""),
            ]
        )

    if MAX_EVALUATIONS == THROUGHPUT_MAX_EVALUATIONS:
        if arguments.all:
            parser.error(
                f"{MAX_EVALUATIONS_ENV}=4 cannot be combined with --all"
            )
        if not arguments.case:
            parser.error(
                f"{MAX_EVALUATIONS_ENV}=4 requires explicit --case "
                "default-api or --case default-cli"
            )
        if len(arguments.case) != 1:
            parser.error(
                f"{MAX_EVALUATIONS_ENV}=4 requires exactly one case per run"
            )
        unsupported_cases = sorted(set(arguments.case) - THROUGHPUT_CASES)
        if unsupported_cases:
            parser.error(
                f"{MAX_EVALUATIONS_ENV}=4 supports only default-api and "
                f"default-cli; got {', '.join(unsupported_cases)}"
            )

    if arguments.out_root is None:
        parser.error("--out-root is required")
    if arguments.timeout_s <= 0:
        parser.error("--timeout-s must be positive")
    python_bin = arguments.python_bin.expanduser().resolve()
    if not python_bin.is_file() or not os.access(python_bin, os.X_OK):
        parser.error(f"--python-bin is not executable: {python_bin}")
    out_root = arguments.out_root.expanduser().resolve()
    profile = arguments.profile.expanduser().resolve()
    if not profile.is_file():
        parser.error(f"profile does not exist: {profile}")
    if arguments.resume:
        if arguments.all:
            parser.error("--resume cannot be combined with --all")
        if arguments.dry_run:
            parser.error("--resume cannot be combined with --dry-run")
        selected = tuple(arguments.case) or (DEFAULT_RESUME_CASE,)
        if selected != (DEFAULT_RESUME_CASE,):
            parser.error("--resume supports exactly --case default-api")
        if not out_root.is_dir():
            parser.error(f"--resume output root does not exist: {out_root}")
        frozen_source_root = (
            None
            if arguments.resume_source_root is None
            else arguments.resume_source_root.expanduser().resolve()
        )
        if frozen_source_root is not None and not frozen_source_root.is_dir():
            parser.error(
                f"--resume-source-root is not a directory: {frozen_source_root}"
            )
        frozen_source_manifest = (
            None
            if arguments.resume_source_manifest is None
            else arguments.resume_source_manifest.expanduser().resolve()
        )
        if (
            frozen_source_manifest is not None
            and not frozen_source_manifest.is_file()
        ):
            parser.error(
                "--resume-source-manifest is not a file: "
                f"{frozen_source_manifest}"
            )
        if arguments.resume_af3_module_path is None:
            parser.error(
                "--resume requires --resume-af3-module-path from the "
                "original baseline runtime evidence"
            )
        expected_af3_module_path = (
            arguments.resume_af3_module_path.expanduser().resolve()
        )
        return _resume_default_api(
            out_root=out_root,
            profile=profile,
            python_bin=python_bin,
            timeout_s=arguments.timeout_s,
            frozen_source_root=frozen_source_root,
            frozen_source_manifest=frozen_source_manifest,
            expected_af3_module_path=expected_af3_module_path,
        )

    _reject_existing_output(out_root)
    cases = _parse_cases(arguments)
    if arguments.dry_run:
        _dry_run(out_root=out_root, profile=profile, cases=cases)
        print(json.dumps({"status": "planned", "plan": str(out_root / "plan.json")}))
        return 0

    try:
        runtime_imports = _preflight_runtime_imports()
    except Exception as error:
        failure = {
            "schema": "astrolabe.gpu-acceptance.runtime-import-preflight/v1",
            "status": "failed",
            "model_calls": 0,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        _atomic_json(out_root / "runtime_import_preflight.json", failure)
        print(
            json.dumps(
                {
                    "status": "ISSUES",
                    "error": str(error),
                    "artifact": str(out_root / "runtime_import_preflight.json"),
                }
            )
        )
        return 1
    _atomic_json(out_root / "runtime_import_preflight.json", runtime_imports)

    source_manifest = _read_source_manifest()
    _atomic_json(out_root / "source_manifest.json", source_manifest.as_dict())
    source_checks: list[dict[str, Any]] = []

    def guard_sources(phase: str) -> SourceManifest:
        current = _check_source_manifest(
            source_manifest,
            out_root=out_root,
            phase=phase,
            check_index=len(source_checks) + 1,
        )
        source_checks.append(
            {
                "phase": phase,
                "fingerprint": current.fingerprint,
                "matches_expected": True,
            }
        )
        _atomic_json(
            out_root / "source_checks.json",
            {
                "schema": "astrolabe.gpu-acceptance.source-check-summary/v1",
                "fingerprint": source_manifest.fingerprint,
                "checks": source_checks,
            },
        )
        return current

    # Compose before probing GPU/weights so the manifest is the scientific
    # profile actually used. The primary single-chain fixture supplies the
    # shared environment manifest.
    manifest_cfg = _build_config(
        profile=profile,
        fixture="single-20aa",
        out_dir=out_root / "manifest-run",
        task_name="gpu_acceptance_manifest",
        execution="daemon",
        generic_search=False,
    )
    environment = _environment_manifest(manifest_cfg, source_manifest)
    _atomic_json(out_root / "environment_manifest.json", environment)

    references: dict[str, tuple[dict[str, Any], Any]] = {}
    reference_order = []
    for name in cases:
        reference = case_plan(name).reference
        if reference not in reference_order:
            reference_order.append(reference)
    for reference in reference_order:
        plan = case_plan(next(name for name in cases if case_plan(name).reference == reference))
        guard_sources(f"reference:{reference}:before")
        references[reference] = _prepare_reference(
            profile=profile,
            out_root=out_root,
            plan=plan,
            environment=environment,
            python_bin=python_bin,
            timeout_s=arguments.timeout_s,
        )
        guard_sources(f"reference:{reference}:after")
        references[reference][0]["source_fingerprint"] = source_manifest.fingerprint
        _atomic_json(
            out_root / "references" / reference / "result.json",
            references[reference][0],
        )

    summaries = []
    for name in cases:
        plan = case_plan(name)
        guard_sources(f"case:{name}:before")
        case_result, _cfg = _prepare_case(
            profile=profile,
            out_root=out_root,
            plan=plan,
            environment=environment,
            python_bin=python_bin,
            timeout_s=arguments.timeout_s,
        )
        try:
            guard_sources(f"case:{name}:after")
            case_result["source_fingerprint"] = source_manifest.fingerprint
        except SourceChanged as error:
            case_result["status"] = "failed"
            case_result["error"] = str(error)
            case_result["source_fingerprint"] = source_manifest.fingerprint
        _atomic_json(out_root / "cases" / name / "result.json", case_result)
        reference_root = out_root / "references" / plan.reference
        comparison = compare_case(
            reference_root=reference_root,
            case_root=out_root / "cases" / name,
        )
        _atomic_json(out_root / "cases" / name / "comparison.json", comparison)
        summaries.append(
            {
                "case": name,
                "case_status": case_result["status"],
                "comparison_status": comparison["status"],
                "model_calls": case_result["model_calls"],
                "comparison": str(out_root / "cases" / name / "comparison.json"),
            }
        )
    overall = (
        "PASS"
        if all(
            item["case_status"] == "evidence-complete"
            and item["comparison_status"] == "PASS"
            for item in summaries
        )
        else "ISSUES"
    )
    _atomic_json(
        out_root / "summary.json",
        {
            "schema": "astrolabe.gpu-acceptance.summary/v1",
            "status": overall,
            "seed": SEED,
            "max_evaluations_per_case": MAX_EVALUATIONS,
            "acceptance_budget": acceptance_budget_record(),
            "no_retry": True,
            "coverage_scope": "implemented-five-family-only",
            "cases": summaries,
        },
    )
    print(json.dumps({"status": overall, "summary": str(out_root / "summary.json")}))
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
