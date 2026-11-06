#!/usr/bin/env python3
"""CPU/GPU acceptance harness for two parallel posthoc daemon slots.

The harness owns validation orchestration only.  It does not implement or
modify refold production control, fixed-list collection, receipts, or numeric
model configuration.  It reuses the two native AlphaFold3 baselines from the
frozen in-process acceptance and runs one ``workers=2`` public refold
invocation with two groups whose only scientific-config transport difference
is ``execution: daemon``.

The public parent is observed through a process-local pre-consume barrier.
Only the outer ``_run_shard`` worker triggers it: after
``refold.run_fixed_task_list`` has returned with both child waits terminal and
both slot receipts committed, and before its results are consumed by the
parent.  The controller verifies PID/start-time/boot-id before SIGSTOP and
SIGKILL.  If that boundary is not observed, this harness fails closed; it never
guesses a cut with an arbitrary delay.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "astrolabe.posthoc-refold.parallel-daemon/v1"
PREPARED_SCHEMA = "astrolabe.posthoc-refold.parallel-daemon.prepared/v1"
TASK_NAME = "posthoc_parallel_final"
OLIGO = "A"
SEQUENCE = "ACDEFGHIKLMNPQRSTVWY"
SEED = 17
NUM_SEEDS = 1
NUM_SAMPLES = 1
MAX_NEW_MODEL_REQUESTS = 4

REPO_ROOT_DEFAULT = Path(
    os.environ.get("ASTROLABE_REPO_ROOT", "/opt/Astrolabe")
).expanduser()
MODEL_DIR_DEFAULT = Path(
    os.environ.get("ASTROLABE_AF3_MODEL_DIR", "/models/alphafold3")
).expanduser()
SIF_PATH = Path(
    "/xcf26/zpzeng/.gh_container/images/"
    "GuidedHallucination_af3_licensed_a13_20260722.sif"
)
SIF_PYTHON = Path("/alphafold3_venv/bin/python")
BASELINE_ATTEMPT_DEFAULT = Path(
    os.environ.get(
        "ASTROLABE_POSTHOC_BASELINE_ATTEMPT",
        "/baseline_attempt",
    )
).expanduser()


class ParallelAcceptanceError(RuntimeError):
    """The parallel acceptance evidence cannot be trusted."""


@dataclass(frozen=True)
class Profile:
    name: str
    num_recycles: int

    @property
    def run(self) -> str:
        return f"single20_seed17_rec{self.num_recycles}"


PROFILES = (
    Profile(name="rec1", num_recycles=1),
    Profile(name="rec2", num_recycles=2),
)


def _load_harness():
    path = Path(__file__).with_name("posthoc_gpu_acceptance.py")
    name = "_posthoc_parallel_daemon_common"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ParallelAcceptanceError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_DAEMON_SITECUSTOMIZE_APPENDIX = r'''

if _ENABLED:
    from astrolabe.evaluators.native_daemons import (
        NativeAF3Runner as _NativeAF3Runner,
    )

    _original_native_af3_run_inference = _NativeAF3Runner.run_inference

    def _daemon_owner_snapshot(runner):
        client = getattr(runner, "_client", None)
        owner = getattr(client, "owner", None)
        if owner is None:
            raise RuntimeError("NativeAF3Runner has no owned daemon")
        owner._start()
        daemon_pid = owner.pid
        if daemon_pid is None:
            raise RuntimeError("NativeAF3Runner daemon has no PID")
        status = owner.status()
        settings = dict(getattr(runner, "_settings", {}) or {})
        lifecycle_path = (
            None
            if owner.temp_dir is None
            else owner.temp_dir / "lifecycle.json"
        )
        return {
            "transport": "daemon",
            "client_pid": os.getpid(),
            "client_ppid": os.getppid(),
            "daemon_pid": int(daemon_pid),
            "daemon_parent_pid": int(status["parent_pid"]),
            "daemon_state": status.get("state"),
            "daemon_model_load_count": int(status.get("model_load_count", -1)),
            "daemon_request_count": int(status.get("request_count", -1)),
            "daemon_completed_count": int(status.get("completed_count", -1)),
            "daemon_failed_count": int(status.get("failed_count", -1)),
            "daemon_identity": owner.identity,
            "daemon_socket": (
                None if owner.socket_path is None else str(owner.socket_path)
            ),
            "daemon_lifecycle_path": (
                None if lifecycle_path is None else str(lifecycle_path)
            ),
            "daemon_log_path": (
                None if owner.log_path is None else str(owner.log_path)
            ),
            "runner_type": _type_name(runner),
            "in_process_runner": False,
            "model_dir": settings.get("model_dir"),
            "num_recycles": settings.get("num_recycles"),
            "num_samples": settings.get("num_samples"),
        }

    def _observed_native_af3_run_inference(
        self,
        featurised_example,
        rng_key,
    ):
        before = _daemon_owner_snapshot(self)
        started = time.time_ns()
        _append(
            {
                "event": "model_daemon_rpc_started",
                "stage": _stage,
                **before,
                "time_ns": started,
            }
        )
        try:
            result = _original_native_af3_run_inference(
                self,
                featurised_example,
                rng_key,
            )
        except BaseException as error:
            _append(
                {
                    "event": "model_daemon_rpc_failed",
                    "stage": _stage,
                    **before,
                    "time_ns": time.time_ns(),
                    "duration_ns": time.time_ns() - started,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            raise
        after = _daemon_owner_snapshot(self)
        completed = time.time_ns()
        _append(
            {
                "event": "model_daemon_rpc_completed",
                "stage": _stage,
                **after,
                "time_ns": completed,
                "duration_ns": completed - started,
            }
        )
        return result

    _NativeAF3Runner.run_inference = _observed_native_af3_run_inference
'''


def _install_capture(directory: Path) -> None:
    harness = _load_harness()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "sitecustomize.py"
    path.write_text(
        harness._SITECUSTOMIZE + _DAEMON_SITECUSTOMIZE_APPENDIX,
        encoding="utf-8",
    )
    try:
        path.chmod(0o600)
    except OSError:
        pass


def _atomic_json(path: Path, value: Any) -> None:
    _load_harness()._atomic_json(path, value)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ParallelAcceptanceError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ParallelAcceptanceError(
                f"{path}:{line_number} is not a JSON object"
            )
        records.append(value)
    return records


def _numeric_manifest_projection(
    source_manifest: Mapping[str, Any],
) -> dict[str, str]:
    files = source_manifest.get("files")
    if not isinstance(files, Mapping):
        raise ParallelAcceptanceError(
            "source manifest has no file-hash mapping"
        )
    projection = {
        str(path): str(digest)
        for path, digest in files.items()
        if path == "run.py"
        or (
            str(path).startswith("astrolabe/")
            and str(path).endswith(".py")
        )
    }
    if not projection:
        raise ParallelAcceptanceError(
            "source manifest has no numeric application closure"
        )
    return projection


def _sha256(path: Path) -> str:
    return _load_harness()._sha256(path)


def _runtime_source_manifest(repo_root: Path) -> dict[str, Any]:
    harness = _load_harness()
    root = Path(repo_root).expanduser().resolve()
    source_manifest = harness.runtime_source_manifest(root)
    common_harness = (
        root / "scripts/validation/posthoc_gpu_acceptance.py"
    )
    source_manifest["files"][
        str(common_harness.relative_to(root))
    ] = _sha256(common_harness)
    parallel_harness = (
        root / "scripts/validation/posthoc_parallel_daemon.py"
    )
    source_manifest["files"][
        str(parallel_harness.relative_to(root))
    ] = _sha256(parallel_harness)
    source_manifest["fingerprint"] = harness._hash_value(
        source_manifest["files"]
    )
    return source_manifest


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _config_text(profile: Profile, *, cache_root: Path, source_dir: Path) -> str:
    return f"""task_name: {TASK_NAME}
out_dir: {source_dir}
seed: {SEED}
repeats: 1
steps: 1
mutation_num: 0
enable_pyrosetta: false
skip_complex_poses: true
types:
  A: p
seqs:
  A: {SEQUENCE}
homologies: {{}}
cyclization: null
nonstandard_residues: null
evaluator:
  evaluator_terms:
    Alphafold3:
      _type: Alphafold3
      execution: daemon
      flash_attention_implementation: cudnn
      num_seeds: {NUM_SEEDS}
      num_recycles: {profile.num_recycles}
      num_samples: {NUM_SAMPLES}
      run_data_pipeline: false
      run_inference: true
      bucket_pad_tolerance: 0
      save_all_samples: false
      jax_compilation_cache_dir: {cache_root / profile.name}
  alphafold3:
    execution: daemon
    model_dir: MODEL_DIR_PLACEHOLDER
    flash_attention_implementation: cudnn
    num_seeds: {NUM_SEEDS}
    num_recycles: {profile.num_recycles}
    num_samples: {NUM_SAMPLES}
    save_embeddings: false
    run_data_pipeline: false
    run_inference: true
    bucket_pad_tolerance: 0
    save_all_samples: false
    jax_compilation_cache_dir: {cache_root / profile.name}
weight_path:
  alphafold3: MODEL_DIR_PLACEHOLDER
stages:
  default:
    evaluator: Alphafold3
    oligo:
    - A
    mutators: []
    mutation_num: 0
    weighted_score_expression: plddt
"""


def _manifest_item(profile: Profile, *, src_run_dir: Path, src_config: Path) -> dict[str, Any]:
    item = {
        "created_at": 0,
        "cyclization": None,
        "homologies": {"A": None},
        "id": f"{TASK_NAME}__{profile.run}__step_000__A",
        "needed_evaluators": ["af3"],
        "nonstandard_residues": None,
        "oligo": OLIGO,
        "run": profile.run,
        "seqs": {"A": SEQUENCE},
        "src_config": str(src_config),
        "src_run_dir": str(src_run_dir),
        "stage1": {
            "failed": [],
            "missing": [],
            "passed": True,
            "selector": "posthoc-parallel-final",
            "selector_kind": "acceptance",
        },
        "step": 0,
        "task": TASK_NAME,
        "types": {"A": "p"},
    }
    return item


def _flatten_config(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {prefix: value}
    flattened: dict[str, Any] = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        flattened.update(_flatten_config(item, path))
    return flattened


def _config_record(path: Path) -> dict[str, Any]:
    from omegaconf import OmegaConf

    flattened = _flatten_config(
        OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    )
    execution = {
        key: flattened[key]
        for key in (
            "evaluator.evaluator_terms.Alphafold3.execution",
            "evaluator.alphafold3.execution",
        )
        if key in flattened
    }
    projection = {
        key: value
        for key, value in flattened.items()
        if key != "out_dir"
        and not key.endswith(".jax_compilation_cache_dir")
        and not key.endswith(".execution")
    }
    return {"execution": execution, "scientific_projection": projection}


def _baseline_reuse_record(
    *,
    baseline_attempt: Path,
    profile: Profile,
    daemon_config: Path,
) -> dict[str, Any]:
    harness = _load_harness()
    baseline_attempt = Path(baseline_attempt).expanduser().resolve()
    capture_dir = (
        baseline_attempt
        / f"evidence/requests/native_baseline_{profile.name}"
    )
    source_config = (
        baseline_attempt / f"inputs/configs/{profile.name}.yaml"
    )
    result_path = (
        baseline_attempt / f"baselines/{profile.name}/result.json"
    )
    requests = _assert_request_counts(
        capture_dir=capture_dir,
        stage=f"reused_native_baseline_{profile.name}",
        expected=1,
    )
    result = _read_json(result_path)
    if result.get("status") != "completed":
        raise ParallelAcceptanceError(
            f"reused baseline is not completed: {profile.name}"
        )
    source = _config_record(source_config)
    daemon = _config_record(daemon_config)
    expected_transport = {
        "evaluator.evaluator_terms.Alphafold3.execution": "in_process",
        "evaluator.alphafold3.execution": "in_process",
    }
    observed_transport = {
        "evaluator.evaluator_terms.Alphafold3.execution": "daemon",
        "evaluator.alphafold3.execution": "daemon",
    }
    if (
        source["execution"] != expected_transport
        or daemon["execution"] != observed_transport
    ):
        raise ParallelAcceptanceError(
            f"transport selectors differ for {profile.name}: "
            f"{source['execution']} -> {daemon['execution']}"
        )
    if source["scientific_projection"] != daemon["scientific_projection"]:
        source_values = source["scientific_projection"]
        daemon_values = daemon["scientific_projection"]
        changes = {
            key: {
                "baseline": source_values.get(key),
                "daemon": daemon_values.get(key),
            }
            for key in sorted(set(source_values) | set(daemon_values))
            if source_values.get(key) != daemon_values.get(key)
        }
        raise ParallelAcceptanceError(
            f"scientific config differs beyond execution for "
            f"{profile.name}: {changes}"
        )

    events = _events(capture_dir)
    completion = [
        event
        for event in events
        if event.get("event") == "model_entrypoint_completed"
    ]
    evaluator = [
        event
        for event in events
        if event.get("event") == "evaluator_completed"
    ]
    if len(completion) != 1 or len(evaluator) != 1:
        raise ParallelAcceptanceError(
            f"reused baseline evidence differs for {profile.name}"
        )
    evidence_paths = {
        "result": result_path,
        "raw_result": capture_dir / Path(
            completion[0]["result_path"]
        ).name,
        "rng": capture_dir / Path(
            completion[0]["rng_key_path"]
        ).name,
        "input_snapshot": capture_dir / Path(
            completion[0]["input_snapshot_path"]
        ).name,
        "scores": capture_dir / Path(evaluator[0]["scores_path"]).name,
        "structure": (
            baseline_attempt
            / f"baselines/{profile.name}/structures"
            / Path(result["structure_paths"][OLIGO]).name
        ),
        "events": capture_dir / "events.jsonl",
    }
    hashes = {
        name: {"path": str(path), "sha256": _sha256(path)}
        for name, path in evidence_paths.items()
    }
    return {
        "source_attempt": str(baseline_attempt),
        "profile": profile.name,
        "source_config": str(source_config),
        "daemon_config": str(daemon_config),
        "transport_differences": [
            {
                "path": key,
                "baseline": source["execution"][key],
                "daemon": daemon["execution"][key],
            }
            for key in sorted(expected_transport)
        ],
        "scientific_config_equal_after_transport_normalization": True,
        "result": result,
        "result_path": str(result_path),
        "capture_dir": str(capture_dir),
        "requests": requests,
        "evidence_hashes": hashes,
        "runtime": result["runtime"],
    }


def _verify_baseline_reuse(
    prepared: Mapping[str, Any],
    profile: Profile,
) -> dict[str, Any]:
    record = prepared["baseline_reuse"][profile.name]
    for name, value in record["evidence_hashes"].items():
        path = Path(value["path"])
        if not path.is_file() or _sha256(path) != value["sha256"]:
            raise ParallelAcceptanceError(
                f"reused baseline evidence changed: {name}: {path}"
            )
    return {
        "process": {
            "reused": True,
            "source_attempt": record["source_attempt"],
            "model_execution": False,
        },
        "result": record["result"],
        "result_path": record["result_path"],
        "capture_dir": record["capture_dir"],
        "requests": record["requests"],
        "reuse_record": record,
    }


def _command(
    *,
    python: Path,
    harness: Path,
    subcommand: str,
    manifest: Path,
    out_root: Path,
    config: Path | None = None,
    workers: int | None = None,
) -> list[str]:
    command = [
        str(python),
        str(harness),
        subcommand,
        "--manifest",
        str(manifest),
        "--out-root",
        str(out_root),
    ]
    if config is not None:
        command.extend(["--config", str(config)])
    if workers is not None:
        command.extend(["--workers", str(workers)])
    return command


def _public_command(
    *,
    python: Path,
    manifest: Path,
    out_root: Path,
    workers: int,
) -> list[str]:
    return [
        str(python),
        "-m",
        "astrolabe.cli",
        "posthoc",
        "refold-run",
        "--manifest",
        str(manifest),
        "--out-root",
        str(out_root),
        "--evaluators",
        "af3",
        "--workers",
        str(workers),
        "--format",
        "json",
    ]


def prepare_acceptance(
    out_root: Path,
    *,
    repo_root: Path = REPO_ROOT_DEFAULT,
    model_dir: Path = MODEL_DIR_DEFAULT,
    sif_provenance: Path | None = None,
    baseline_attempt: Path = BASELINE_ATTEMPT_DEFAULT,
) -> dict[str, Any]:
    out_root = Path(out_root).expanduser().resolve()
    repo_root = Path(repo_root).expanduser().resolve()
    model_dir = Path(model_dir).expanduser().resolve()
    baseline_attempt = Path(baseline_attempt).expanduser().resolve()
    if not baseline_attempt.is_dir():
        raise ParallelAcceptanceError(
            f"baseline attempt is absent: {baseline_attempt}"
        )
    if out_root.exists() and any(out_root.iterdir()):
        raise ParallelAcceptanceError(
            f"acceptance root is not empty: {out_root}"
        )
    out_root.mkdir(parents=True, exist_ok=True)
    inputs = out_root / "inputs"
    configs = inputs / "configs"
    source_runs = inputs / "source_runs"
    configs.mkdir(parents=True)
    source_runs.mkdir(parents=True)

    items = []
    input_hashes: dict[str, str] = {}
    for profile in PROFILES:
        source_dir = source_runs / profile.name
        source_dir.mkdir(parents=True)
        config_path = configs / f"{profile.name}.yaml"
        config_text = _config_text(
            profile,
            cache_root=out_root / "cache",
            source_dir=source_dir,
        ).replace("MODEL_DIR_PLACEHOLDER", str(model_dir))
        _write_text_atomic(config_path, config_text)
        item = _manifest_item(
            profile,
            src_run_dir=source_dir,
            src_config=config_path,
        )
        item["inputs_hash"] = _load_harness()._manifest_inputs_hash(item)
        items.append(item)
        input_hashes[f"config_{profile.name}"] = _sha256(config_path)

        baseline_manifest = inputs / f"manifest_{profile.name}.jsonl"
        _write_text_atomic(
            baseline_manifest,
            json.dumps(item, sort_keys=True) + "\n",
        )
        input_hashes[f"manifest_{profile.name}"] = _sha256(
            baseline_manifest
        )

    manifest = inputs / "manifest.jsonl"
    _write_text_atomic(
        manifest,
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in items),
    )
    input_hashes["manifest"] = _sha256(manifest)
    baseline_reuse = {
        profile.name: _baseline_reuse_record(
            baseline_attempt=baseline_attempt,
            profile=profile,
            daemon_config=configs / f"{profile.name}.yaml",
        )
        for profile in PROFILES
    }

    provenance = None
    if sif_provenance is not None:
        provenance = _load_harness()._load_sif_provenance(
            sif_provenance,
            expected_path=SIF_PATH,
        )
    source_manifest = _runtime_source_manifest(repo_root)
    _atomic_json(out_root / "source_manifest.json", source_manifest)
    baseline_prepared = _read_json(baseline_attempt / "prepared.json")
    baseline_source_manifest = _read_json(
        baseline_attempt / "source_manifest.json"
    )
    numeric_projection = _numeric_manifest_projection(source_manifest)
    baseline_numeric_projection = _numeric_manifest_projection(
        baseline_source_manifest
    )
    if numeric_projection != baseline_numeric_projection:
        changed = sorted(
            path
            for path in set(numeric_projection)
            | set(baseline_numeric_projection)
            if numeric_projection.get(path)
            != baseline_numeric_projection.get(path)
        )
        raise ParallelAcceptanceError(
            f"numeric source differs from reused baseline: {changed}"
        )
    if not provenance is None:
        baseline_sif_sha256 = baseline_prepared.get("sif_sha256")
        if (
            baseline_sif_sha256 is not None
            and baseline_sif_sha256 != provenance["sha256"]
        ):
            raise ParallelAcceptanceError(
                "daemon harness SIF differs from reused baseline SIF"
            )
    if str(Path(baseline_prepared["model_dir"])) != str(model_dir):
        raise ParallelAcceptanceError(
            "daemon harness model directory differs from reused baseline"
        )

    harness_path = repo_root / "scripts/validation/posthoc_parallel_daemon.py"
    caller_root = out_root / "caller/runs"
    prepared = {
        "schema": PREPARED_SCHEMA,
        "status": "prepared",
        "acceptance_schema": SCHEMA,
        "out_root": str(out_root),
        "repo_root": str(repo_root),
        "model_dir": str(model_dir),
        "model_file": str(model_dir / "af3.bin"),
        "sif": str(SIF_PATH),
        "sif_sha256": None if provenance is None else provenance["sha256"],
        "sif_provenance": provenance,
        "sif_python": str(SIF_PYTHON),
        "harness_path": str(harness_path),
        "fixture": {
            "sequence": SEQUENCE,
            "types": {"A": "p"},
            "oligo": OLIGO,
        },
        "profiles": [
            {
                "name": profile.name,
                "run": profile.run,
                "seed": SEED,
                "num_samples": NUM_SAMPLES,
                "num_recycles": profile.num_recycles,
                "manifest": str(
                    inputs / f"manifest_{profile.name}.jsonl"
                ),
                "config": str(configs / f"{profile.name}.yaml"),
            }
            for profile in PROFILES
        ],
        "inputs": {
            "manifest": str(manifest),
            "baseline_manifests": {
                profile.name: str(
                    inputs / f"manifest_{profile.name}.jsonl"
                )
                for profile in PROFILES
            },
            "configs": {
                profile.name: str(configs / f"{profile.name}.yaml")
                for profile in PROFILES
            },
        },
        "input_hashes": input_hashes,
        "baseline_attempt": str(baseline_attempt),
        "baseline_reuse": baseline_reuse,
        "baseline_provenance": {
            "source_attempt": str(baseline_attempt),
            "source_fingerprint": baseline_prepared["source_fingerprint"],
            "numeric_source_file_count": len(numeric_projection),
            "numeric_source_unchanged": True,
            "sif_sha256": baseline_prepared.get("sif_sha256"),
            "model_dir": str(baseline_prepared["model_dir"]),
        },
        "source_fingerprint": source_manifest["fingerprint"],
        "commands": {
            "public_actual_workers2": _public_command(
                python=SIF_PYTHON,
                manifest=manifest,
                out_root=caller_root,
                workers=2,
            ),
            "public_parallel_cache_workers2": _public_command(
                python=SIF_PYTHON,
                manifest=manifest,
                out_root=caller_root,
                workers=2,
            ),
            "public_serial_cache_workers1": _public_command(
                python=SIF_PYTHON,
                manifest=manifest,
                out_root=caller_root,
                workers=1,
            ),
        },
        "request_contract": {
            "native_baseline_reused": 2,
            "native_baseline_new": 0,
            "public_daemon_workers2": 2,
            "public_parent_kill_recovery": 0,
            "public_parallel_cache": 0,
            "public_serial_cache": 0,
            "maximum_new_model_requests": MAX_NEW_MODEL_REQUESTS,
        },
        "control_contract": {
            "public_workers": 2,
            "groups": 2,
            "resource_groups": 2,
            "shard_child_workers": 1,
            "slot_worker_execution": "synchronous",
            "fixed_list_engine": "astrolabe.design.fixed_list.run_fixed_task_list",
            "cut_boundary": (
                "refold.run_fixed_task_list returned after both outer "
                "_run_shard child waits and committed slot receipts, before "
                "the parent consumes FixedTaskListRun results"
            ),
            "cut_hook_scope": "outer _run_shard caller only",
            "resume_contract": (
                "per-slot committed receipts; outer fixed-list control has "
                "no durable journal"
            ),
            "cut_signal_scope": "verified owner CLI pid/start-time/boot-id only",
            "cut_ready_schema": (
                "astrolabe.posthoc-refold.parallel-cut-ready/v2"
            ),
        },
        "cpu_preflight": {
            "daemon_transport": True,
            "reused_native_baseline": True,
            "generated_sitecustomize_fresh_process": True,
            "jax_random_key": True,
            "empty_array_npz_fail_closed": True,
            "model_execution": False,
        },
    }
    if not harness_path.is_file():
        raise ParallelAcceptanceError(
            f"harness is absent from the source root: {harness_path}"
        )
    _atomic_json(out_root / "prepared.json", prepared)
    _atomic_json(
        out_root / "plan.json",
        {
            "schema": SCHEMA,
            "status": "prepared",
            "profiles": prepared["profiles"],
            "commands": prepared["commands"],
            "request_contract": prepared["request_contract"],
            "control_contract": prepared["control_contract"],
        },
    )
    return prepared


def _verify_prepared(out_root: Path) -> dict[str, Any]:
    out_root = Path(out_root).expanduser().resolve()
    prepared = _read_json(out_root / "prepared.json")
    source_path = out_root / "source_manifest.json"
    expected_source = _read_json(source_path)
    repo_root = Path(prepared["repo_root"]).expanduser().resolve()
    observed_source = _runtime_source_manifest(repo_root)
    if expected_source.get("fingerprint") != observed_source["fingerprint"]:
        raise ParallelAcceptanceError("runtime source changed after preparation")
    for key, expected in prepared["input_hashes"].items():
        if key == "manifest":
            path = Path(prepared["inputs"]["manifest"])
        elif key.startswith("manifest_"):
            path = Path(
                prepared["inputs"]["baseline_manifests"][key.removeprefix("manifest_")]
            )
        elif key.startswith("config_"):
            path = Path(
                prepared["inputs"]["configs"][key.removeprefix("config_")]
            )
        else:
            raise ParallelAcceptanceError(f"unknown frozen input key: {key}")
        if _sha256(path) != expected:
            raise ParallelAcceptanceError(f"frozen input changed: {path}")

    manifest_path = Path(prepared["inputs"]["manifest"])
    items = _read_jsonl(manifest_path)
    if len(items) != 2:
        raise ParallelAcceptanceError(
            f"parallel fixture requires two manifest rows, got {len(items)}"
        )
    by_run = {str(item["run"]): item for item in items}
    from omegaconf import OmegaConf

    for profile in PROFILES:
        item = by_run.get(profile.run)
        if item is None:
            raise ParallelAcceptanceError(
                f"manifest profile is absent: {profile.run}"
            )
        cfg_path = Path(prepared["inputs"]["configs"][profile.name])
        cfg = OmegaConf.load(cfg_path)
        observed = {
            "seed": int(cfg.seed),
            "num_seeds": int(cfg.evaluator.alphafold3.num_seeds),
            "num_samples": int(cfg.evaluator.alphafold3.num_samples),
            "num_recycles": int(cfg.evaluator.alphafold3.num_recycles),
            "execution": str(cfg.evaluator.alphafold3.execution),
            "run_inference": bool(cfg.evaluator.alphafold3.run_inference),
            "run_data_pipeline": bool(
                cfg.evaluator.alphafold3.run_data_pipeline
            ),
        }
        expected = {
            "seed": SEED,
            "num_seeds": NUM_SEEDS,
            "num_samples": NUM_SAMPLES,
            "num_recycles": profile.num_recycles,
            "execution": "daemon",
            "run_inference": True,
            "run_data_pipeline": False,
        }
        if observed != expected:
            raise ParallelAcceptanceError(
                f"scientific profile differs for {profile.name}: "
                f"{observed} != {expected}"
            )
        if item.get("seqs") != {"A": SEQUENCE}:
            raise ParallelAcceptanceError(
                f"sequence differs for {profile.name}"
            )
        if Path(item["src_config"]).resolve() != cfg_path.resolve():
            raise ParallelAcceptanceError(
                f"manifest config is not the frozen config for {profile.name}"
            )
        _verify_baseline_reuse(prepared, profile)
    return prepared


def _capture_environment(
    *,
    prepared: Mapping[str, Any],
    capture_dir: Path,
    stage: str,
    max_requests: int,
) -> dict[str, str]:
    harness = _load_harness()
    environment = harness._capture_environment(
        repo_root=Path(prepared["repo_root"]),
        capture_dir=capture_dir,
        stage=stage,
        max_requests=max_requests,
    )
    environment["ASTROLABE_POSTHOC_ACCEPTANCE_GLOBAL_LEDGER"] = "1"
    environment["JAX_PLATFORMS"] = (
        "cpu" if int(max_requests) == 0 else "cuda"
    )
    return environment


def _arm_total_request_budget(
    environment: dict[str, str],
    *,
    out_root: Path,
    maximum: int = MAX_NEW_MODEL_REQUESTS,
) -> None:
    environment.update(
        {
            "ASTROLABE_POSTHOC_ACCEPTANCE_TOTAL_LEDGER_DIR": str(
                out_root / "evidence/request_budget_ledger"
            ),
            "ASTROLABE_POSTHOC_ACCEPTANCE_TOTAL_MAX_REQUESTS": str(
                maximum
            ),
        }
    )


def _request_evidence(capture_dir: Path) -> dict[str, Any]:
    return _load_harness()._request_evidence(capture_dir)


def _events(capture_dir: Path) -> list[dict[str, Any]]:
    return _load_harness()._read_jsonl(capture_dir / "events.jsonl")


def _assert_request_counts(
    *,
    capture_dir: Path,
    stage: str,
    expected: int,
) -> dict[str, Any]:
    evidence = _request_evidence(capture_dir)
    keys = (
        "rpc_delivered",
        "request_reservations",
        "inner_run_inference_invocations",
        "inner_run_inference_completed",
        "entrypoint_completed",
        "observed_model_requests",
        "failed_or_denied_model_requests",
    )
    expected_record = {
        "rpc_delivered": expected,
        "request_reservations": expected,
        "inner_run_inference_invocations": expected,
        "inner_run_inference_completed": expected,
        "entrypoint_completed": expected,
        "observed_model_requests": expected,
        "failed_or_denied_model_requests": 0,
    }
    observed = {key: evidence[key] for key in keys}
    if observed != expected_record:
        raise ParallelAcceptanceError(
            f"{stage} request layers differ: {observed} != {expected_record}"
        )
    return evidence


def _nonzero_npz(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ParallelAcceptanceError(f"array evidence is absent: {path}")
    with np.load(path, allow_pickle=False) as archive:
        keys = list(archive.files)
        arrays = {key: archive[key] for key in keys}
    if not arrays:
        raise ParallelAcceptanceError(
            f"array evidence is an empty NPZ archive: {path}"
        )
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "keys": keys,
        "dtype_shape": {
            key: {
                "dtype": value.dtype.str,
                "shape": list(value.shape),
            }
            for key, value in arrays.items()
        },
    }


def _owner_key(event: Mapping[str, Any]) -> int | None:
    owner = event.get("resource_owner")
    if not isinstance(owner, Mapping):
        return None
    value = owner.get("num_recycles")
    return None if value is None else int(value)


def _request_intervals(capture_dir: Path) -> dict[str, Any]:
    events = _events(capture_dir)
    started = {
        int(event["request_id"]): event
        for event in events
        if event.get("event") == "model_inner_invocation_started"
    }
    completed = {
        int(event["request_id"]): event
        for event in events
        if event.get("event") == "model_inner_invocation_completed"
    }
    if set(started) != set(completed):
        raise ParallelAcceptanceError(
            "inner invocation start/completion request IDs differ"
        )
    records = []
    for request_id in sorted(started):
        begin = started[request_id]
        end = completed[request_id]
        owner = begin.get("resource_owner")
        owner_after = end.get("resource_owner")
        if not isinstance(owner, Mapping) or not isinstance(owner_after, Mapping):
            raise ParallelAcceptanceError(
                f"request {request_id} has no resource-owner evidence"
            )
        if owner.get("pid") != begin.get("pid"):
            raise ParallelAcceptanceError(
                f"request {request_id} owner PID differs"
            )
        records.append(
            {
                "request_id": request_id,
                "pid": int(begin["pid"]),
                "ppid": int(owner["ppid"]),
                "start_ns": int(begin["time_ns"]),
                "completed_ns": int(end["time_ns"]),
                "num_recycles": owner.get("num_recycles"),
                "instance_token": owner.get("instance_token"),
                "in_process_runner": owner.get("in_process_runner"),
                "model_dir": owner.get("model_dir"),
                "device": owner.get("device"),
                "model_state": owner_after.get("model_state"),
            }
        )
    if len(records) < 2:
        raise ParallelAcceptanceError(
            f"parallel overlap needs at least two actual requests, got {len(records)}"
        )
    overlap_start = max(record["start_ns"] for record in records)
    overlap_end = min(record["completed_ns"] for record in records)
    return {
        "request_count": len(records),
        "distinct_pids": len({record["pid"] for record in records}),
        "distinct_owners": len(
            {record["instance_token"] for record in records}
        ),
        "overlap_ns": max(0, overlap_end - overlap_start),
        "overlap": overlap_end > overlap_start,
        "requests": records,
    }


def _daemon_transport_evidence(capture_dir: Path) -> dict[str, Any]:
    events = _events(capture_dir)
    failed = [
        event
        for event in events
        if event.get("event") == "model_daemon_rpc_failed"
    ]
    if failed:
        raise ParallelAcceptanceError(
            f"daemon transport recorded failures: {failed}"
        )
    started = [
        event
        for event in events
        if event.get("event") == "model_daemon_rpc_started"
    ]
    completed = [
        event
        for event in events
        if event.get("event") == "model_daemon_rpc_completed"
    ]
    if len(started) != 2 or len(completed) != 2:
        raise ParallelAcceptanceError(
            "daemon transport did not record exactly two start/completion pairs"
        )
    inner = [
        event
        for event in events
        if event.get("event") == "model_inner_invocation_completed"
    ]
    inner_started = [
        event
        for event in events
        if event.get("event") == "model_inner_invocation_started"
    ]
    records = []
    for event in sorted(completed, key=lambda value: int(value["time_ns"])):
        client_pid = int(event["client_pid"])
        daemon_pid = int(event["daemon_pid"])
        matching_started = [
            item
            for item in started
            if int(item.get("client_pid", -1)) == client_pid
            and int(item.get("daemon_pid", -1)) == daemon_pid
        ]
        if len(matching_started) != 1:
            raise ParallelAcceptanceError(
                f"daemon status start is not unique for client {client_pid}"
            )
        started_event = matching_started[0]
        if client_pid == daemon_pid:
            raise ParallelAcceptanceError(
                "daemon transport reused the client PID"
            )
        if int(event["daemon_parent_pid"]) != client_pid:
            raise ParallelAcceptanceError(
                "daemon owner parent PID differs from its client"
            )
        if (
            int(started_event.get("daemon_model_load_count", -1)) != 1
            or int(started_event.get("daemon_request_count", -1)) != 1
            or int(started_event.get("daemon_completed_count", -1)) != 1
            or int(started_event.get("daemon_failed_count", -1)) != 0
        ):
            raise ParallelAcceptanceError(
                f"daemon pre-inference status differs for client {client_pid}"
            )
        # NativeAF3Runner starts its daemon with one ``describe`` RPC,
        # followed by exactly one inference RPC. Physical model calls remain
        # proven by the unique inner ModelRunner start/completion pair below,
        # never by the daemon request counter alone.
        if (
            event.get("transport") != "daemon"
            or event.get("in_process_runner") is not False
            or int(event.get("daemon_model_load_count", -1)) != 1
            or int(event.get("daemon_request_count", -1)) != 2
            or int(event.get("daemon_completed_count", -1)) != 2
            or int(event.get("daemon_failed_count", -1)) != 0
        ):
            raise ParallelAcceptanceError(
                f"daemon status differs for client {client_pid}"
            )
        matching_inner = [
            item
            for item in inner
            if int(item.get("pid", -1)) == daemon_pid
            and int(
                (item.get("resource_owner") or {}).get("ppid", -1)
            )
            == client_pid
        ]
        matching_start = [
            item
            for item in inner_started
            if int(item.get("pid", -1)) == daemon_pid
            and int(
                (item.get("resource_owner") or {}).get("ppid", -1)
            )
            == client_pid
        ]
        if len(matching_inner) != 1 or len(matching_start) != 1:
            raise ParallelAcceptanceError(
                "daemon transport has no unique inner ModelRunner pair"
            )
        lifecycle_path = Path(str(event["daemon_lifecycle_path"]))
        lifecycle = _read_json(lifecycle_path)
        if (
            lifecycle.get("status") != "stopped"
            or int(lifecycle.get("pid", -1)) != daemon_pid
            or int(lifecycle.get("parent_pid", -1)) != client_pid
            or int(lifecycle.get("exit_code", -1)) != 0
        ):
            raise ParallelAcceptanceError(
                f"daemon lifecycle is not cleanly stopped: {lifecycle_path}"
            )
        log_path = Path(str(event["daemon_log_path"]))
        if not log_path.is_file():
            raise ParallelAcceptanceError(
                f"daemon owner log is absent: {log_path}"
            )
        inner_event = matching_inner[0]
        inner_start = matching_start[0]
        records.append(
            {
                "client_pid": client_pid,
                "client_ppid": int(event["client_ppid"]),
                "daemon_pid": daemon_pid,
                "daemon_parent_pid": int(event["daemon_parent_pid"]),
                "daemon_identity": event["daemon_identity"],
                "daemon_socket": event["daemon_socket"],
                "daemon_lifecycle_path": str(lifecycle_path),
                "daemon_lifecycle_sha256": _sha256(lifecycle_path),
                "daemon_log_path": str(log_path),
                "daemon_log_sha256": _sha256(log_path),
                "daemon_exit_code": int(lifecycle["exit_code"]),
                "num_recycles": int(event["num_recycles"]),
                "num_samples": int(event["num_samples"]),
                "model_dir": event["model_dir"],
                "start_status_counts": {
                    "model_load": int(
                        started_event["daemon_model_load_count"]
                    ),
                    "requests": int(
                        started_event["daemon_request_count"]
                    ),
                    "completed": int(
                        started_event["daemon_completed_count"]
                    ),
                    "failed": int(
                        started_event["daemon_failed_count"]
                    ),
                },
                "end_status_counts": {
                    "model_load": int(event["daemon_model_load_count"]),
                    "requests": int(event["daemon_request_count"]),
                    "completed": int(event["daemon_completed_count"]),
                    "failed": int(event["daemon_failed_count"]),
                },
                "device": (
                    inner_event.get("resource_owner") or {}
                ).get("device"),
                "inner_request_id": int(inner_event["request_id"]),
                "inner_start_ns": int(inner_start["time_ns"]),
                "inner_completed_ns": int(inner_event["time_ns"]),
            }
        )
    if len({record["client_pid"] for record in records}) != 2:
        raise ParallelAcceptanceError(
            "daemon requests did not use two client processes"
        )
    if len({record["daemon_pid"] for record in records}) != 2:
        raise ParallelAcceptanceError(
            "daemon requests did not use two model-owner processes"
        )
    overlap_start = max(record["inner_start_ns"] for record in records)
    overlap_end = min(record["inner_completed_ns"] for record in records)
    return {
        "transport": "daemon",
        "request_count": len(records),
        "distinct_clients": len(
            {record["client_pid"] for record in records}
        ),
        "distinct_daemons": len(
            {record["daemon_pid"] for record in records}
        ),
        "overlap_ns": max(0, overlap_end - overlap_start),
        "records": sorted(
            records,
            key=lambda value: value["inner_start_ns"],
        ),
    }


def _entrypoint_preflight_worker(
    *,
    manifest_path: Path,
    config_path: Path,
    model_dir: Path,
    out_dir: Path,
    guard: bool,
) -> int:
    import jax
    from alphafold3.constants import chemical_components
    from alphafold3.data import featurisation
    import sitecustomize as generated
    from astrolabe.core import global_config
    from astrolabe.evaluators.alphafold3 import AlphaFold3Evaluator
    from astrolabe.evaluators.wrappers.alphafold3 import (
        ModelRunner,
        make_model_config,
    )
    from astrolabe.posthoc import refold
    from astrolabe.utils.af3_compat import (
        allow_all_x_protein_chains_during_featurisation,
    )
    from astrolabe.utils.base import construct_standard_input

    item = _read_jsonl(manifest_path)
    if len(item) != 1:
        raise ParallelAcceptanceError(
            "entrypoint preflight requires one manifest row"
        )
    item = item[0]
    cfg = refold._load_cfg_for_refold(config_path)
    runtime = refold._load_refold_runtime_overrides(cfg)
    context = refold._load_refold_input_context(cfg, item=item)
    homologies = refold._resolve_homology_value(
        v=item.get("homologies") or {},
        relpath_base=None,
        path_maps=(),
        require_abs=True,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    refold._init_refold_global_config(out_dir=out_dir, **runtime, **context)
    refold._build_protomers(
        seqs=item["seqs"],
        types=item["types"],
        homologies=homologies,
    )
    global_config.set_variable("step", int(item["step"]))
    inputs = construct_standard_input([str(item["oligo"])], seed=int(cfg.seed))
    model_config = make_model_config(
        num_recycles=int(cfg.evaluator.alphafold3.num_recycles),
        num_diffusion_samples=int(
            cfg.evaluator.alphafold3.num_samples
        ),
    )
    runner = ModelRunner(
        model_config,
        jax.devices("cpu")[0],
        Path(model_dir),
    )
    evaluator = AlphaFold3Evaluator(
        runner,
        out_dir / "unused_evaluator_output",
        cfg.evaluator.alphafold3,
    )
    prepared_inputs = evaluator.prepare_input(inputs)
    fold_input = prepared_inputs[str(item["oligo"])]
    ccd = chemical_components.Ccd(user_ccd=fold_input.user_ccd)
    with allow_all_x_protein_chains_during_featurisation(fold_input):
        examples = featurisation.featurise_input(
            fold_input=fold_input,
            buckets=evaluator.buckets[str(item["oligo"])],
            ccd=ccd,
            verbose=False,
            **evaluator.kwargs[str(item["oligo"])],
        )
    if len(examples) != 1:
        raise ParallelAcceptanceError(
            f"expected one real featurised fixture, got {len(examples)}"
        )
    real_bundle = examples[0]

    def guarded_inner(_runner, featurised_example, _rng_key):
        if featurised_example is not real_bundle:
            raise ParallelAcceptanceError(
                "entrypoint preflight feature identity changed"
            )
        if guard:
            raise RuntimeError(
                "INNER_GPU_BOUNDARY_GUARD: reached after feature capture, "
                "request reservation, and RNG capture; model was not launched"
            )
        return {
            "identifier": np.asarray([17], dtype=np.int64),
            "positions": np.asarray(
                [[0.0, 1.0, 2.0]],
                dtype=np.float32,
            ),
        }

    generated._original_inference = guarded_inner
    result = ModelRunner.run_inference(
        runner,
        real_bundle,
        jax.random.PRNGKey(SEED),
    )
    _atomic_json(
        out_dir / "entrypoint_probe.json",
        {
            "status": "PASS",
            "model_execution": False,
            "guard": bool(guard),
            "num_recycles": int(
                cfg.evaluator.alphafold3.num_recycles
            ),
            "rng_platform": jax.default_backend(),
            "result_identifier": np.asarray(
                result["identifier"]
            ).tolist(),
        },
    )
    return 0


def _validate_entrypoint_preflight(
    *,
    capture_dir: Path,
    profile: Profile,
) -> dict[str, Any]:
    counts = _assert_request_counts(
        capture_dir=capture_dir,
        stage=f"entrypoint_preflight_{profile.name}",
        expected=1,
    )
    events = _events(capture_dir)
    completions = [
        event
        for event in events
        if event.get("event") == "model_entrypoint_completed"
    ]
    if len(completions) != 1:
        raise ParallelAcceptanceError(
            "entrypoint preflight did not produce one completed event"
        )
    rng = _nonzero_npz(Path(completions[0]["rng_key_path"]))
    with np.load(
        completions[0]["rng_key_path"],
        allow_pickle=False,
    ) as archive:
        values = np.asarray(archive["rng_key"])
    if (
        values.dtype != np.dtype(np.uint32)
        or values.shape != (2,)
        or values.tolist() != [0, SEED]
    ):
        raise ParallelAcceptanceError(
            f"unexpected JAX PRNG key: {values.dtype} {values.shape} {values}"
        )
    owner = completions[0].get("resource_owner")
    if not isinstance(owner, Mapping) or owner.get("in_process_runner") is not True:
        raise ParallelAcceptanceError(
            "entrypoint preflight has no in-process resource owner"
        )
    return {
        "status": "PASS",
        "profile": profile.name,
        "request_counts": {
            key: counts[key]
            for key in (
                "rpc_delivered",
                "request_reservations",
                "inner_run_inference_invocations",
                "inner_run_inference_completed",
                "entrypoint_completed",
            )
        },
        "rng": rng,
        "resource_owner": owner,
        "input_digest": completions[0]["input_digest"],
        "model_execution": False,
    }


def _validate_entrypoint_guard(
    *,
    capture_dir: Path,
    profile: Profile,
    returncode: int,
    stderr: str,
) -> dict[str, Any]:
    if returncode != 1 or "INNER_GPU_BOUNDARY_GUARD" not in stderr:
        raise ParallelAcceptanceError(
            "entrypoint preflight did not stop at the inner GPU guard"
        )
    evidence = _request_evidence(capture_dir)
    expected = {
        "rpc_delivered": 1,
        "request_reservations": 1,
        "inner_run_inference_invocations": 1,
        "inner_run_inference_completed": 0,
        "entrypoint_completed": 0,
        "failed_or_denied_model_requests": 1,
    }
    observed = {key: evidence[key] for key in expected}
    if observed != expected:
        raise ParallelAcceptanceError(
            f"entrypoint guard request layers differ: {observed}"
        )
    events = _events(capture_dir)
    started = [
        event
        for event in events
        if event.get("event") == "model_inner_invocation_started"
    ]
    if len(started) != 1:
        raise ParallelAcceptanceError(
            "entrypoint guard did not start exactly one inner boundary"
        )
    rng_paths = sorted(capture_dir.glob("*_rng_key.npz"))
    snapshots = sorted(capture_dir.glob("*_input_snapshot.json"))
    if len(rng_paths) != 1 or len(snapshots) != 1:
        raise ParallelAcceptanceError(
            "entrypoint guard did not retain RNG and input snapshots"
        )
    rng = _nonzero_npz(rng_paths[0])
    owner = started[0].get("resource_owner")
    if not isinstance(owner, Mapping):
        raise ParallelAcceptanceError(
            "entrypoint guard has no resource-owner evidence"
        )
    return {
        "status": "PASS",
        "profile": profile.name,
        "model_execution": False,
        "gpu_model_calls": 0,
        "inner_guard_reached": True,
        "request_counts": observed,
        "rng": rng,
        "input_snapshot_path": str(snapshots[0]),
        "input_snapshot_sha256": _sha256(snapshots[0]),
        "resource_owner": owner,
    }


def _process_identity(pid: int) -> dict[str, Any]:
    stat_path = Path(f"/proc/{pid}/stat")
    boot_path = Path("/proc/sys/kernel/random/boot_id")
    try:
        line = stat_path.read_text(encoding="utf-8").strip()
        boot_id = boot_path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise ParallelAcceptanceError(
            f"cannot observe owned process {pid}: {error}"
        ) from error
    closing = line.rfind(")")
    if closing < 0:
        raise ParallelAcceptanceError(f"invalid /proc stat for {pid}")
    remainder = line[closing + 2 :].split()
    if len(remainder) < 20:
        raise ParallelAcceptanceError(f"short /proc stat for {pid}")
    return {
        "pid": pid,
        "state": remainder[0],
        "starttime_ticks": int(remainder[19]),
        "boot_id": boot_id,
    }


def _wait_for_state(
    pid: int,
    *,
    states: set[str],
    timeout_s: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            identity = _process_identity(pid)
        except ParallelAcceptanceError:
            identity = {"pid": pid, "state": "absent"}
        if identity["state"] in states:
            return identity
        time.sleep(0.05)
    raise ParallelAcceptanceError(
        f"process {pid} did not enter states {sorted(states)}"
    )


def _expected_receipt_paths(
    caller_root: Path,
) -> dict[str, Path]:
    return {
        profile.name: (
            caller_root
            / TASK_NAME
            / profile.run
            / ".framework"
            / "refold"
            / "af3"
            / "step_000_A"
            / "receipt.json"
        )
        for profile in PROFILES
    }


def _validate_cut_ready(
    *,
    ready_path: Path,
    parent_pid: int,
    caller_root: Path,
) -> dict[str, Any]:
    if not ready_path.is_file():
        raise ParallelAcceptanceError(
            f"pre-collect cut barrier is absent: {ready_path}"
        )
    ready = _read_json(ready_path)
    if ready.get("schema") != "astrolabe.posthoc-refold.parallel-cut-ready/v2":
        raise ParallelAcceptanceError("pre-collect cut barrier has another schema")
    if (
        int(ready.get("pid", -1)) != parent_pid
        or int(ready.get("parent_pid", -1)) != parent_pid
    ):
        raise ParallelAcceptanceError(
            "pre-collect cut barrier belongs to another parent PID"
        )
    if (
        ready.get("boundary")
        != (
            "refold_run_fixed_task_list_after_child_wait_"
            "and_receipt_commit_before_result_consume"
        )
        or not re.fullmatch(
            r"fixedlist-[0-9a-f]{64}",
            str(ready.get("action_id", "")),
        )
        or ready.get("ordered_jobs")
        != ["shard_0001", "shard_0002"]
        or int(ready.get("successful_child_waits", -1)) != 2
        or ready.get("outer_worker_name") != "_run_shard"
        or int(ready.get("max_in_flight", -1)) != 2
        or ready.get("resume_contract")
        != "per_slot_committed_receipts_with_no_outer_journal"
    ):
        raise ParallelAcceptanceError(
            "outer refold fixed-list cut marker differs"
        )
    expected = _expected_receipt_paths(caller_root)
    observed_paths = {
        Path(record["path"]).resolve()
        for record in ready.get("receipts", [])
        if isinstance(record, Mapping) and record.get("path")
    }
    expected_paths = {path.resolve() for path in expected.values()}
    if observed_paths != expected_paths:
        raise ParallelAcceptanceError(
            "cut receipts do not match both expected slots"
        )
    for record in ready["receipts"]:
        path = Path(record["path"])
        value = _read_json(path)
        if (
            value.get("schema") != "astrolabe.posthoc-refold.receipt/v1"
            or value.get("status") != "completed"
            or value.get("backend") != "af3"
            or value.get("origin") != "inference"
            or not value.get("identity_sha256")
            or not value.get("outputs", {}).get("fingerprint")
        ):
            raise ParallelAcceptanceError(
                f"cut receipt is not completed inference proof: {path}"
            )
        if record.get("sha256") != _sha256(path):
            raise ParallelAcceptanceError(
                f"cut receipt hash changed: {path}"
            )
    summaries = ready.get("worker_summaries", [])
    if len(summaries) != 2:
        raise ParallelAcceptanceError(
            "cut marker has no two completed child summaries"
        )
    for record in summaries:
        path = Path(record["path"])
        if not path.is_file() or record.get("sha256") != _sha256(path):
            raise ParallelAcceptanceError(
                f"cut child summary is absent or changed: {path}"
            )
    return ready


def _actual_shard_evidence(
    *,
    caller_root: Path,
    parent_pid: int,
) -> dict[str, Any]:
    parallel_root = caller_root / "_parallel_workers"
    generations = sorted(
        path
        for path in parallel_root.glob("run_*")
        if (path / "generation.json").is_file()
    )
    if len(generations) != 1:
        raise ParallelAcceptanceError(
            "actual parallel run must have exactly one generation before recovery"
        )
    generation_root = generations[0]
    generation = _read_json(generation_root / "generation.json")
    if (
        generation.get("schema")
        != "astrolabe.posthoc-refold.parallel-generation/v1"
        or int(generation.get("parent_pid", -1)) != parent_pid
        or int(generation.get("workers", -1)) != 2
    ):
        raise ParallelAcceptanceError(
            "actual generation ownership/worker count differs"
        )
    summaries = sorted(generation_root.glob("worker_*.summary.json"))
    if len(summaries) != 2:
        raise ParallelAcceptanceError(
            "actual parallel parent did not produce two child summaries"
        )
    records = []
    for path in summaries:
        summary = _read_json(path)
        expected = {
            "workers_requested": 1,
            "workers": 1,
            "processed": 1,
            "skipped": 0,
            "failed": 0,
            "ran_af3": 1,
            "receipt_cache_hits_af3": 0,
            "groups": 1,
            "resource_groups": 1,
        }
        observed = {key: summary.get(key) for key in expected}
        if observed != expected:
            raise ParallelAcceptanceError(
                f"actual shard summary differs: {path}: {observed}"
            )
        records.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "summary": summary,
            }
        )
    return {
        "generation_root": str(generation_root),
        "generation": generation,
        "worker_summaries": records,
    }


def _run_public_with_precollect_cut(
    *,
    out_root: Path,
    prepared: Mapping[str, Any],
    timeout_s: int,
) -> dict[str, Any]:
    harness = _load_harness()
    capture_dir = out_root / "evidence/requests/public_actual"
    if capture_dir.exists() and any(capture_dir.iterdir()):
        raise ParallelAcceptanceError(
            f"public capture is not empty: {capture_dir}"
        )
    _install_capture(capture_dir)
    ready_path = out_root / "evidence/public_cut_ready.json"
    if ready_path.exists():
        raise ParallelAcceptanceError(
            f"cut barrier is not empty: {ready_path}"
        )
    caller_root = out_root / "caller/runs"
    environment = _capture_environment(
        prepared=prepared,
        capture_dir=capture_dir,
        stage="public_actual_workers2",
        max_requests=2,
    )
    environment.update(
        {
            "ASTROLABE_POSTHOC_PARALLEL_CUT": "1",
            "ASTROLABE_POSTHOC_PARALLEL_CUT_EXPECTED": "2",
            "ASTROLABE_POSTHOC_PARALLEL_CUT_ROOT": str(caller_root),
            "ASTROLABE_POSTHOC_PARALLEL_CUT_READY": str(ready_path),
        }
    )
    _arm_total_request_budget(
        environment,
        out_root=out_root,
        maximum=MAX_NEW_MODEL_REQUESTS,
    )
    stdout_path = out_root / "logs/public_actual.stdout.log"
    stderr_path = out_root / "logs/public_actual.stderr.log"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    started_unix_s = time.time()
    with stdout_path.open("w", encoding="utf-8") as stdout, (
        stderr_path.open("w", encoding="utf-8")
    ) as stderr:
        process = subprocess.Popen(
            prepared["commands"]["public_actual_workers2"],
            env=environment,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
    identity = _process_identity(process.pid)
    deadline = time.monotonic() + timeout_s
    while not ready_path.exists():
        if process.poll() is not None:
            raise ParallelAcceptanceError(
                "public parent exited before both committed receipts were "
                f"ready for the cut; rc={process.returncode}"
            )
        if time.monotonic() >= deadline:
            raise ParallelAcceptanceError(
                "public parent did not reach the committed-receipt cut"
            )
        time.sleep(0.05)
    ready = _validate_cut_ready(
        ready_path=ready_path,
        parent_pid=process.pid,
        caller_root=caller_root,
    )
    stopped = _wait_for_state(
        process.pid,
        states={"T", "t"},
        timeout_s=10,
    )
    before_kill = _process_identity(process.pid)
    if (
        before_kill["starttime_ticks"] != identity["starttime_ticks"]
        or before_kill["boot_id"] != identity["boot_id"]
    ):
        raise ParallelAcceptanceError(
            "public parent identity changed before SIGKILL"
        )
    marker_start = ready.get("process_start") or {}
    if (
        int(marker_start.get("starttime_ticks", -1))
        != identity["starttime_ticks"]
        or marker_start.get("boot_id") != identity["boot_id"]
    ):
        raise ParallelAcceptanceError(
            "outer refold fixed-list marker process identity differs"
        )
    artifact_root = caller_root / TASK_NAME
    artifact_before_recovery = harness._tree_snapshot(artifact_root)
    os.kill(process.pid, signal.SIGKILL)
    try:
        returncode = process.wait(timeout=30)
    except subprocess.TimeoutExpired as error:
        raise ParallelAcceptanceError(
            "owned public parent did not terminate after SIGKILL"
        ) from error
    if returncode != -signal.SIGKILL:
        raise ParallelAcceptanceError(
            f"public parent termination status differs: {returncode}"
        )
    evidence = _assert_request_counts(
        capture_dir=capture_dir,
        stage="public_actual_workers2",
        expected=2,
    )
    shard_evidence = _actual_shard_evidence(
        caller_root=caller_root,
        parent_pid=process.pid,
    )
    return {
        "pid": process.pid,
        "identity_before": identity,
        "identity_stopped": stopped,
        "returncode": returncode,
        "started_unix_s": started_unix_s,
        "finished_unix_s": time.time(),
        "capture_dir": str(capture_dir),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "cut_ready": ready,
        "shard_evidence": shard_evidence,
        "request_evidence": evidence,
        "artifact_before_recovery": artifact_before_recovery,
    }


def _invoke_baseline(
    *,
    out_root: Path,
    prepared: Mapping[str, Any],
    profile: Profile,
    timeout_s: int,
) -> dict[str, Any]:
    del out_root, timeout_s
    return _verify_baseline_reuse(prepared, profile)


def _invoke_cache(
    *,
    out_root: Path,
    prepared: Mapping[str, Any],
    label: str,
    command: Sequence[str],
    expected_workers: int,
    expected_parallel: bool,
    timeout_s: int,
) -> dict[str, Any]:
    harness = _load_harness()
    capture_dir = out_root / f"evidence/requests/{label}"
    if capture_dir.exists() and any(capture_dir.iterdir()):
        raise ParallelAcceptanceError(
            f"cache capture is not empty: {capture_dir}"
        )
    _install_capture(capture_dir)
    environment = _capture_environment(
        prepared=prepared,
        capture_dir=capture_dir,
        stage=label,
        max_requests=0,
    )
    _arm_total_request_budget(
        environment,
        out_root=out_root,
        maximum=0,
    )
    stdout_path = out_root / f"logs/{label}.stdout.json"
    stderr_path = out_root / f"logs/{label}.stderr.log"
    process = harness._run_process(
        command,
        environment=environment,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        timeout_s=timeout_s,
    )
    summary = harness._parse_json_stdout(stdout_path)
    expected = {
        "processed": 0,
        "skipped": 2,
        "failed": 0,
        "ran_af3": 0,
        "receipt_cache_hits_af3": 2,
        "workers": expected_workers,
        "groups": 2,
        "resource_groups": 2,
    }
    observed = {key: summary.get(key) for key in expected}
    if observed != expected:
        raise ParallelAcceptanceError(
            f"{label} summary differs: {observed} != {expected}"
        )
    if expected_parallel != ("parallel_generation" in summary):
        raise ParallelAcceptanceError(
            f"{label} parallel-generation shape differs"
        )
    requests = _assert_request_counts(
        capture_dir=capture_dir,
        stage=label,
        expected=0,
    )
    worker_summaries = []
    if expected_parallel:
        generation_root = Path(summary["parallel_generation"])
        worker_paths = sorted(
            generation_root.glob("worker_*.summary.json")
        )
        if len(worker_paths) != 2:
            raise ParallelAcceptanceError(
                f"{label} did not run two fixed-list shards"
            )
        for path in worker_paths:
            value = _read_json(path)
            if (
                value.get("workers_requested") != 1
                or value.get("workers") != 1
                or value.get("receipt_cache_hits_af3") != 1
                or value.get("ran_af3") != 0
                or value.get("skipped") != 1
            ):
                raise ParallelAcceptanceError(
                    f"{label} shard did not use worker=1 cache semantics: "
                    f"{path}"
                )
            worker_summaries.append(
                {
                    "path": str(path),
                    "sha256": _sha256(path),
                    "summary": value,
                }
            )
    return {
        "process": process,
        "summary": summary,
        "requests": requests,
        "capture_dir": str(capture_dir),
        "worker_summaries": worker_summaries,
    }


def _run_public_recovery_and_caches(
    *,
    out_root: Path,
    prepared: Mapping[str, Any],
    artifact_paid: Mapping[str, Any],
    timeout_s: int,
) -> dict[str, Any]:
    harness = _load_harness()
    artifact_root = out_root / "caller/runs" / TASK_NAME
    recovery = _invoke_cache(
        out_root=out_root,
        prepared=prepared,
        label="public_parent_kill_recovery_workers2",
        command=prepared["commands"]["public_actual_workers2"],
        expected_workers=2,
        expected_parallel=True,
        timeout_s=timeout_s,
    )
    artifact_after_recovery = harness._tree_snapshot(artifact_root)
    if artifact_after_recovery != artifact_paid:
        raise ParallelAcceptanceError(
            "parent-kill cold recovery changed scientific artifacts"
        )
    parallel_cache = _invoke_cache(
        out_root=out_root,
        prepared=prepared,
        label="public_parallel_cache_workers2",
        command=prepared["commands"]["public_parallel_cache_workers2"],
        expected_workers=2,
        expected_parallel=True,
        timeout_s=timeout_s,
    )
    artifact_parallel_cache = harness._tree_snapshot(artifact_root)
    if artifact_parallel_cache != artifact_paid:
        raise ParallelAcceptanceError(
            "parallel cache changed scientific artifacts"
        )
    serial_cache = _invoke_cache(
        out_root=out_root,
        prepared=prepared,
        label="public_serial_cache_workers1",
        command=prepared["commands"]["public_serial_cache_workers1"],
        expected_workers=1,
        expected_parallel=False,
        timeout_s=timeout_s,
    )
    artifact_serial_cache = harness._tree_snapshot(artifact_root)
    if artifact_serial_cache != artifact_paid:
        raise ParallelAcceptanceError(
            "serial cache changed scientific artifacts"
        )
    return {
        "public_parent_kill_recovery": recovery,
        "public_parallel_cache": parallel_cache,
        "public_serial_cache": serial_cache,
        "artifact_fingerprints": {
            "paid_before_recovery": artifact_paid["fingerprint"],
            "after_parent_kill_recovery": (
                artifact_after_recovery["fingerprint"]
            ),
            "after_parallel_cache": artifact_parallel_cache["fingerprint"],
            "after_serial_cache": artifact_serial_cache["fingerprint"],
            "unchanged": (
                artifact_paid
                == artifact_after_recovery
                == artifact_parallel_cache
                == artifact_serial_cache
            ),
        },
    }


def _event_for_profile(
    events: Sequence[Mapping[str, Any]],
    *,
    event_name: str,
    num_recycles: int,
) -> dict[str, Any]:
    matches = [
        dict(event)
        for event in events
        if event.get("event") == event_name
        and _owner_key(event) == num_recycles
    ]
    if len(matches) != 1:
        raise ParallelAcceptanceError(
            f"expected one {event_name} for recycles={num_recycles}, "
            f"got {len(matches)}"
        )
    return matches[0]


def _normalize_empty_array_strides(value, path="$", changed=None):
    if changed is None:
        changed = []
    if isinstance(value, Mapping):
        normalized = {
            key: _normalize_empty_array_strides(
                item,
                f"{path}.{key}",
                changed,
            )
            for key, item in value.items()
        }
        kind = value.get("kind")
        shape = value.get("shape")
        if (
            kind in {"numeric_array", "object_array"}
            and isinstance(shape, list)
            and any(int(dimension) == 0 for dimension in shape)
        ):
            canonical = [0] * len(shape)
            if normalized.get("strides") != canonical:
                normalized["strides"] = canonical
                changed.append(path)
        return normalized
    if isinstance(value, list):
        return [
            _normalize_empty_array_strides(
                item,
                f"{path}[{index}]",
                changed,
            )
            for index, item in enumerate(value)
        ]
    return value


def _snapshot_tree_digest(tree: Any) -> str:
    encoded = json.dumps(
        tree,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _input_snapshot_comparison(
    public_completion: Mapping[str, Any],
    baseline_completion: Mapping[str, Any],
) -> dict[str, Any]:
    public_path = Path(public_completion["input_snapshot_path"])
    baseline_path = Path(baseline_completion["input_snapshot_path"])
    public_snapshot = _read_json(public_path)
    baseline_snapshot = _read_json(baseline_path)
    public_normalized_paths: list[str] = []
    baseline_normalized_paths: list[str] = []
    public_normalized = _normalize_empty_array_strides(
        public_snapshot,
        changed=public_normalized_paths,
    )
    baseline_normalized = _normalize_empty_array_strides(
        baseline_snapshot,
        changed=baseline_normalized_paths,
    )
    public_digest = _snapshot_tree_digest(public_snapshot["tree"])
    baseline_digest = _snapshot_tree_digest(baseline_snapshot["tree"])
    normalized_public_digest = _snapshot_tree_digest(
        public_normalized["tree"]
    )
    normalized_baseline_digest = _snapshot_tree_digest(
        baseline_normalized["tree"]
    )
    return {
        "equal": public_normalized == baseline_normalized,
        "public": {
            "sha256": normalized_public_digest,
            "raw_tree_sha256": public_digest,
            "counts": public_normalized["counts"],
            "snapshot_path": str(public_path),
        },
        "baseline": {
            "sha256": normalized_baseline_digest,
            "raw_tree_sha256": baseline_digest,
            "counts": baseline_normalized["counts"],
            "snapshot_path": str(baseline_path),
        },
        "raw_tree_digests_equal": public_digest == baseline_digest,
        "normalization": {
            "zero_length_strides_only": True,
            "public_changed_paths": public_normalized_paths,
            "baseline_changed_paths": baseline_normalized_paths,
        },
    }


def _scientific_profile_check(
    *,
    out_root: Path,
    prepared: Mapping[str, Any],
    profile: Profile,
    public_capture: Path,
    public_run_dir: Path,
    baseline: Mapping[str, Any],
    daemon_transport: Mapping[str, Any],
) -> dict[str, Any]:
    harness = _load_harness()
    public_events = _events(public_capture)
    baseline_capture = Path(baseline["capture_dir"])
    baseline_events = _events(baseline_capture)
    public_completion = _event_for_profile(
        public_events,
        event_name="model_entrypoint_completed",
        num_recycles=profile.num_recycles,
    )
    baseline_completion = _event_for_profile(
        baseline_events,
        event_name="model_entrypoint_completed",
        num_recycles=profile.num_recycles,
    )
    public_score_event = _event_for_profile(
        public_events,
        event_name="evaluator_completed",
        num_recycles=profile.num_recycles,
    )
    baseline_score_event = _event_for_profile(
        baseline_events,
        event_name="evaluator_completed",
        num_recycles=profile.num_recycles,
    )

    input_compare = _input_snapshot_comparison(
        public_completion,
        baseline_completion,
    )
    rng_compare = harness.compare_npz(
        Path(public_completion["rng_key_path"]),
        Path(baseline_completion["rng_key_path"]),
    )
    raw_compare = harness.compare_npz(
        Path(public_completion["result_path"]),
        Path(baseline_completion["result_path"]),
    )

    evidence_root = out_root / "evidence/comparisons" / profile.name
    public_scores, _public_scores_json = harness._scores_evidence(
        Path(public_score_event["scores_path"]),
        evidence_root / "public_scores",
    )
    baseline_scores, _baseline_scores_json = harness._scores_evidence(
        Path(baseline_score_event["scores_path"]),
        evidence_root / "baseline_scores",
    )
    scores_compare = harness.compare_npz(
        public_scores,
        baseline_scores,
    )

    public_structure = harness._model_structure(
        public_run_dir,
        {
            "step": 0,
            "oligo": OLIGO,
        },
    )
    result = baseline["result"]
    baseline_structure = Path(
        result["structure_paths"][OLIGO]
    )
    public_coord_npz, _public_coord_json, public_coord = (
        harness._coordinate_evidence(
            name=f"{profile.name}_public",
            structure=public_structure,
            root=evidence_root / "coordinates",
        )
    )
    baseline_coord_npz, _baseline_coord_json, baseline_coord = (
        harness._coordinate_evidence(
            name=f"{profile.name}_baseline",
            structure=baseline_structure,
            root=evidence_root / "coordinates",
        )
    )
    coordinate_compare = harness.compare_npz(
        public_coord_npz,
        baseline_coord_npz,
    )
    coordinate_identity_fields = (
        "coordinate_dtype",
        "coordinate_shape",
        "atom_identity",
    )
    coordinate_identity = all(
        public_coord.get(key) == baseline_coord.get(key)
        for key in coordinate_identity_fields
    )
    public_raw = harness._load_npz(
        Path(public_completion["result_path"])
    )
    baseline_raw = harness._load_npz(
        Path(baseline_completion["result_path"])
    )
    identifier_key = "result____identifier__"
    same_model_identifier = (
        identifier_key in public_raw
        and identifier_key in baseline_raw
        and np.array_equal(
            public_raw[identifier_key],
            baseline_raw[identifier_key],
            equal_nan=True,
        )
    )
    item = next(
        row
        for row in _read_jsonl(Path(prepared["inputs"]["manifest"]))
        if str(row["run"]) == profile.run
    )
    receipt = harness._refold_receipt_evidence(public_run_dir, item)
    config_path = Path(
        prepared["inputs"]["configs"][profile.name]
    )
    baseline_config = _config_record(
        Path(baseline["reuse_record"]["source_config"])
    )
    daemon_config = _config_record(config_path)
    same_scientific_config = (
        baseline_config["scientific_projection"]
        == daemon_config["scientific_projection"]
        and int(result["seed"]) == SEED
        and int(result["num_seeds"]) == NUM_SEEDS
        and int(result["num_samples"]) == NUM_SAMPLES
        and int(result["num_recycles"]) == profile.num_recycles
        and int(
            public_completion["resource_owner"]["num_recycles"]
        )
        == profile.num_recycles
        and baseline_config["execution"]
        == {
            "evaluator.evaluator_terms.Alphafold3.execution": "in_process",
            "evaluator.alphafold3.execution": "in_process",
        }
        and daemon_config["execution"]
        == {
            "evaluator.evaluator_terms.Alphafold3.execution": "daemon",
            "evaluator.alphafold3.execution": "daemon",
        }
    )
    transport_records = [
        record
        for record in daemon_transport["records"]
        if int(record["num_recycles"]) == profile.num_recycles
    ]
    daemon_owner_complete = (
        len(transport_records) == 1
        and int(transport_records[0]["num_samples"]) == NUM_SAMPLES
        and int(transport_records[0]["daemon_exit_code"]) == 0
        and public_completion["resource_owner"]["pid"]
        == transport_records[0]["daemon_pid"]
        and public_completion["resource_owner"]["ppid"]
        == transport_records[0]["client_pid"]
        and transport_records[0]["device"]
        == public_completion["resource_owner"]["device"]
    )
    expected_device = result["runtime"]["devices"][0]
    same_runtime_device = (
        public_completion["resource_owner"]["device"] == expected_device
        and daemon_owner_complete
    )
    checks = {
        "exact_featurised_input_digest": input_compare["equal"],
        "exact_rng_key": rng_compare["equal"],
        "exact_raw_result_arrays_dtype_shape_values_equal_nan": (
            raw_compare["equal"]
        ),
        "exact_standard_score_arrays": scores_compare["equal"],
        "exact_atomic_coordinates": coordinate_compare["equal"],
        "exact_atom_identity": coordinate_identity,
        "same_model_identifier": same_model_identifier,
        "same_scientific_config": same_scientific_config,
        "daemon_transport_and_cleanup": daemon_owner_complete,
        "same_runtime_device": same_runtime_device,
        "committed_refold_receipt": (
            receipt.get("status") == "completed"
            and isinstance(receipt.get("identity_sha256"), str)
            and isinstance(
                receipt.get("output_contract_fingerprint"),
                str,
            )
            and int(receipt.get("output_file_count", 0)) > 0
        ),
    }
    return {
        "profile": profile.name,
        "num_recycles": profile.num_recycles,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "raw_model_result_arrays": raw_compare,
        "rng_key": rng_compare,
        "standard_score_arrays": scores_compare,
        "atomic_coordinates": coordinate_compare,
        "atom_identity_equal": coordinate_identity,
        "featurised_input_digest": input_compare,
        "same_model_identifier": same_model_identifier,
        "same_scientific_config": same_scientific_config,
        "daemon_transport": transport_records[0]
        if len(transport_records) == 1
        else transport_records,
        "same_runtime_device": same_runtime_device,
        "refold_receipt": receipt,
        "public_structure": str(public_structure),
        "baseline_structure": str(baseline_structure),
    }


def _daemon_preflight_worker(
    *,
    baseline_capture: Path,
    out_root: Path,
) -> int:
    import sitecustomize as generated
    from astrolabe.evaluators.native_daemons import NativeAF3Runner

    out_root.mkdir(parents=True, exist_ok=True)
    events = _events(baseline_capture)
    completion = [
        event
        for event in events
        if event.get("event") == "model_entrypoint_completed"
    ]
    if len(completion) != 1:
        raise ParallelAcceptanceError(
            "baseline capture has no unique entrypoint completion"
        )
    source_event = completion[0]
    rng_path = (
        Path(baseline_capture)
        / Path(source_event["rng_key_path"]).name
    )
    with np.load(rng_path, allow_pickle=False) as archive:
        if not archive.files or "rng_key" not in archive.files:
            raise ParallelAcceptanceError(
                "baseline RNG evidence is empty"
            )
        rng = np.asarray(archive["rng_key"])
    daemon_root = out_root / "synthetic_daemon"
    daemon_root.mkdir(parents=True, exist_ok=True)
    lifecycle_path = daemon_root / "lifecycle.json"
    log_path = daemon_root / "owner.jsonl"
    _atomic_json(
        lifecycle_path,
        {
            "status": "ready",
            "pid": os.getpid() + 1,
            "parent_pid": os.getpid(),
        },
    )
    log_path.write_text("", encoding="utf-8")

    class _Owner:
        identity = "daemon-preflight"
        pid = os.getpid() + 1
        socket_path = daemon_root / "model.sock"
        temp_dir = daemon_root

        def __init__(self):
            self.log_path = log_path
            self.request_count = 1
            self.completed_count = 1

        def _start(self):
            return None

        def status(self):
            return {
                "state": "running",
                "parent_pid": os.getpid(),
                "model_load_count": 1,
                "request_count": self.request_count,
                "completed_count": self.completed_count,
                "failed_count": 0,
            }

    class _Client:
        def __init__(self):
            self.owner = _Owner()

        def request(self, payload):
            self.owner.request_count += 1
            self.owner.completed_count += 1
            return {"preflight_only": True}

    owner = _Owner()
    runner = NativeAF3Runner.__new__(NativeAF3Runner)
    runner._client = _Client()
    runner._cyclic_asym_ids = ()
    runner._cyclic_relative_patches = ()
    runner._settings = {
        "model_dir": "$MODEL_DIR",
        "num_recycles": int(
            source_event["resource_owner"]["num_recycles"]
        ),
        "num_samples": int(
            source_event["resource_owner"]["num_samples"]
        ),
    }
    model_dir = str(
        source_event["resource_owner"].get("model_dir") or ""
    )
    runner._settings["model_dir"] = model_dir
    runner.run_inference(
        {
            "baseline_input_snapshot": source_event[
                "input_snapshot_sha256"
            ]
        },
        rng,
    )
    _atomic_json(
        lifecycle_path,
        {
            "status": "stopped",
            "pid": owner.pid,
            "parent_pid": os.getpid(),
            "exit_code": 0,
        },
    )
    transport = _events(Path(os.environ["ASTROLABE_POSTHOC_ACCEPTANCE_CAPTURE_DIR"]))
    kinds = [event.get("event") for event in transport]
    if (
        kinds.count("model_daemon_rpc_started") != 1
        or kinds.count("model_daemon_rpc_completed") != 1
        or any(kind.startswith("model_inner_") for kind in kinds)
    ):
        raise ParallelAcceptanceError(
            f"generated daemon transport preflight differs: {kinds}"
        )
    _atomic_json(
        out_root / "daemon_preflight.json",
        {
            "schema": "astrolabe.posthoc-refold.daemon-preflight/v1",
            "status": "PASS",
            "transport_shape_only": True,
            "model_execution": False,
            "new_model_requests": 0,
            "generated_sitecustomize": Path(
                generated.__file__
            ).resolve().as_posix(),
            "baseline_input_snapshot": str(
                Path(baseline_capture)
                / Path(source_event["input_snapshot_path"]).name
            ),
            "baseline_input_snapshot_sha256": source_event[
                "input_snapshot_sha256"
            ],
            "baseline_rng": str(rng_path),
            "baseline_rng_sha256": _sha256(rng_path),
            "baseline_rng_dtype": str(rng.dtype),
            "baseline_rng_shape": list(rng.shape),
        },
    )
    return 0


def _run_daemon_preflights(
    *,
    out_root: Path,
    prepared: Mapping[str, Any],
    timeout_s: int,
) -> dict[str, Any]:
    harness = _load_harness()
    results = {}
    for profile in PROFILES:
        capture_dir = out_root / f"evidence/preflight/daemon_{profile.name}"
        if capture_dir.exists() and any(capture_dir.iterdir()):
            raise ParallelAcceptanceError(
                f"daemon preflight capture is not empty: {capture_dir}"
            )
        _install_capture(capture_dir)
        environment = _capture_environment(
            prepared=prepared,
            capture_dir=capture_dir,
            stage=f"daemon_preflight_{profile.name}",
            max_requests=0,
        )
        environment["JAX_PLATFORMS"] = "cpu"
        baseline_capture = Path(
            prepared["baseline_reuse"][profile.name]["capture_dir"]
        )
        command = [
            str(Path(prepared["sif_python"])),
            str(Path(prepared["harness_path"])),
            "_daemon-preflight-worker",
            "--baseline-capture",
            str(baseline_capture),
            "--out-root",
            str(out_root / f"preflight/daemon_{profile.name}"),
        ]
        stdout_path = (
            out_root
            / f"logs/preflight_daemon_{profile.name}.stdout.log"
        )
        stderr_path = (
            out_root
            / f"logs/preflight_daemon_{profile.name}.stderr.log"
        )
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            completed = subprocess.run(
                command,
                env=environment,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            stdout_path.write_text(
                error.stdout or "",
                encoding="utf-8",
            )
            stderr_path.write_text(
                error.stderr or "",
                encoding="utf-8",
            )
            raise ParallelAcceptanceError(
                f"daemon preflight timed out: {profile.name}"
            ) from error
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise ParallelAcceptanceError(
                f"daemon preflight failed for {profile.name}: "
                f"{completed.stderr}"
            )
        request_counts = _assert_request_counts(
            capture_dir=capture_dir,
            stage=f"daemon_preflight_{profile.name}",
            expected=0,
        )
        report = _read_json(
            out_root
            / f"preflight/daemon_{profile.name}/daemon_preflight.json"
        )
        if (
            report.get("status") != "PASS"
            or report.get("model_execution") is not False
            or report.get("new_model_requests") != 0
            or report.get("transport_shape_only") is not True
        ):
            raise ParallelAcceptanceError(
                f"daemon preflight report differs for {profile.name}"
            )
        results[profile.name] = {
            "process": {
                "command": command,
                "returncode": completed.returncode,
                "stdout_log": str(stdout_path),
                "stderr_log": str(stderr_path),
            },
            "evidence": {
                **report,
                "profile": profile.name,
                "request_counts": {
                    key: request_counts[key]
                    for key in (
                        "rpc_delivered",
                        "request_reservations",
                        "inner_run_inference_invocations",
                        "inner_run_inference_completed",
                        "entrypoint_completed",
                    )
                },
                "capture_dir": str(capture_dir),
            },
        }
    return results


def _verify_fixedlist_control_contract(summary: Mapping[str, Any]) -> None:
    import inspect
    from astrolabe.design import fixed_list

    if not hasattr(fixed_list, "run_fixed_task_list"):
        raise ParallelAcceptanceError(
            "Galileo fixed-list collect-all API is absent"
        )
    parameters = inspect.signature(
        fixed_list.run_fixed_task_list
    ).parameters
    if (
        "worker_execution" not in parameters
        or "max_in_flight" not in parameters
    ):
        raise ParallelAcceptanceError(
            "fixed-list interface lacks bounded inline/parallel execution"
        )
    if (
        int(summary.get("workers_requested", -1)) != 2
        or int(summary.get("workers", -1)) != 2
        or int(summary.get("groups", -1)) != 2
        or int(summary.get("resource_groups", -1)) != 2
    ):
        raise ParallelAcceptanceError(
            "parallel CLI summary does not prove two workers and resource groups"
        )


def _fixedlist_interface_evidence(
    prepared: Mapping[str, Any],
) -> dict[str, Any]:
    import inspect
    from astrolabe.design import fixed_list

    _verify_fixedlist_control_contract(
        {
            "workers_requested": 2,
            "workers": 2,
            "groups": 2,
            "resource_groups": 2,
        }
    )
    repo_root = Path(prepared["repo_root"])
    refold_path = repo_root / "astrolabe/posthoc/refold.py"
    source = refold_path.read_text(encoding="utf-8")
    required_fragments = (
        "run_fixed_task_list(",
        'worker_execution="synchronous"',
        "max_in_flight=worker_count",
        "workers=1",
    )
    missing = [
        fragment for fragment in required_fragments
        if fragment not in source
    ]
    if missing:
        raise ParallelAcceptanceError(
            "posthoc does not match Galileo's final fixed-list wiring: "
            f"missing={missing}"
        )
    return {
        "status": "PASS",
        "engine": (
            f"{fixed_list.run_fixed_task_list.__module__}."
            f"{fixed_list.run_fixed_task_list.__qualname__}"
        ),
        "signature": str(
            inspect.signature(fixed_list.run_fixed_task_list)
        ),
        "synchronous_slot_worker": True,
        "parent_max_in_flight": 2,
        "child_workers": 1,
        "source": {
            "fixed_list": str(Path(inspect.getsourcefile(fixed_list))),
            "posthoc_refold": str(refold_path),
        },
    }


def run_cpu_preflight(
    out_root: Path,
    *,
    timeout_s: int = 600,
) -> dict[str, Any]:
    out_root = Path(out_root).expanduser().resolve()
    prepared = _verify_prepared(out_root)
    control = _fixedlist_interface_evidence(prepared)
    from astrolabe.posthoc import refold_receipts

    numeric_source = refold_receipts.numeric_source_record(
        "af3",
        repo_root=Path(prepared["repo_root"]),
    )
    expected_control_files = {
        str(
            (
                Path(prepared["repo_root"])
                / relative
            ).resolve()
        )
        for relative in (
            "astrolabe/design/fixed_list.py",
            "astrolabe/design/collection.py",
            "astrolabe/design/collection_search.py",
            "astrolabe/design/coordinator.py",
            "astrolabe/design/composition.py",
            "astrolabe/posthoc/refold.py",
            "astrolabe/posthoc/refold_receipts.py",
            "run.py",
        )
    }
    numeric_paths = {
        str(Path(record["path"]).resolve())
        for record in numeric_source["files"]
    }
    missing_numeric_paths = sorted(
        expected_control_files - numeric_paths
    )
    if missing_numeric_paths:
        raise ParallelAcceptanceError(
            "numeric source closure omits control files: "
            f"{missing_numeric_paths}"
        )
    _atomic_json(
        out_root / "preflight/numeric_source.json",
        numeric_source,
    )
    result = {
        "schema": SCHEMA,
        "status": "PASS",
        "phase": "cpu-preflight",
        "model_execution": False,
        "new_model_requests": 0,
        "daemon_preflight": _run_daemon_preflights(
            out_root=out_root,
            prepared=prepared,
            timeout_s=timeout_s,
        ),
        "fixedlist_control": control,
        "numeric_source": {
            "schema": numeric_source["schema"],
            "backend": numeric_source["backend"],
            "origin": numeric_source["origin"],
            "own_package_file_count": numeric_source[
                "own_package_file_count"
            ],
            "selected_third_party_file_count": numeric_source[
                "selected_third_party_file_count"
            ],
            "record_sha256": _load_harness()._hash_value(
                numeric_source
            ),
            "manifest_path": str(
                out_root / "preflight/numeric_source.json"
            ),
        },
        "source_fingerprint": prepared["source_fingerprint"],
    }
    source_after = _runtime_source_manifest(
        Path(prepared["repo_root"])
    )
    if source_after["fingerprint"] != prepared["source_fingerprint"]:
        raise ParallelAcceptanceError(
            "runtime source changed during CPU preflight"
        )
    result["source_fingerprint_after"] = source_after["fingerprint"]
    _atomic_json(out_root / "cpu_preflight.json", result)
    return result


def run_acceptance(
    out_root: Path,
    *,
    timeout_s: int = 3600,
) -> dict[str, Any]:
    harness = _load_harness()
    out_root = Path(out_root).expanduser().resolve()
    state_path = out_root / "run_state.json"
    if state_path.exists():
        raise ParallelAcceptanceError(
            f"run state exists; refusing duplicate execution: {state_path}"
        )
    prepared = _verify_prepared(out_root)
    control = _fixedlist_interface_evidence(prepared)
    _atomic_json(
        out_root / "preflight/fixedlist_control.json",
        control,
    )
    runtime = harness._runtime_preflight(prepared)
    _atomic_json(out_root / "runtime_preflight.json", runtime)
    daemon_preflight = _run_daemon_preflights(
        out_root=out_root,
        prepared=prepared,
        timeout_s=timeout_s,
    )
    _atomic_json(
        out_root / "preflight/entrypoint_evidence.json",
        {
            "schema": SCHEMA,
            "status": "PASS",
            "model_execution": False,
            "new_model_requests": 0,
            "profiles": daemon_preflight,
        },
    )
    _atomic_json(
        state_path,
        {
            "schema": SCHEMA,
            "status": "running",
            "maximum_new_model_requests": MAX_NEW_MODEL_REQUESTS,
            "started_unix_s": time.time(),
        },
    )
    try:
        baselines = {
            profile.name: _invoke_baseline(
                out_root=out_root,
                prepared=prepared,
                profile=profile,
                timeout_s=timeout_s,
            )
            for profile in PROFILES
        }
        public_actual = _run_public_with_precollect_cut(
            out_root=out_root,
            prepared=prepared,
            timeout_s=timeout_s,
        )
        public_capture = Path(public_actual["capture_dir"])
        intervals = _request_intervals(public_capture)
        daemon_transport = _daemon_transport_evidence(public_capture)
        parent_pid = int(public_actual["pid"])
        if intervals["distinct_pids"] != 2:
            raise ParallelAcceptanceError(
                "public workers=2 did not execute in two distinct child processes"
            )
        if intervals["overlap_ns"] <= 0:
            raise ParallelAcceptanceError(
                "the two actual public requests did not overlap in time"
            )
        if daemon_transport["request_count"] != 2:
            raise ParallelAcceptanceError(
                "daemon transport did not carry exactly two model requests"
            )
        if daemon_transport["overlap_ns"] <= 0:
            raise ParallelAcceptanceError(
                "the two daemon-owned GPU requests did not overlap in time"
            )
        for request in intervals["requests"]:
            if request["in_process_runner"] is not True:
                raise ParallelAcceptanceError(
                    "daemon process has no in-process GPU resource owner"
                )
            state = request.get("model_state") or {}
            if (
                state.get("params_cached_after") is not True
                or state.get("jit_model_cached_after") is not True
            ):
                raise ParallelAcceptanceError(
                    "actual request did not establish resident model ownership"
                )
        if sorted(
            int(request["num_recycles"])
            for request in intervals["requests"]
        ) != [1, 2]:
            raise ParallelAcceptanceError(
                "parallel request owners do not cover recycles 1 and 2"
            )
        matching_transport = [
            record
            for record in daemon_transport["records"]
            if int(record["client_ppid"]) == parent_pid
        ]
        if len(matching_transport) != 2:
            raise ParallelAcceptanceError(
                "daemon clients are not both children of the cut parent"
            )
        daemon_by_recycles = {
            int(record["num_recycles"]): record
            for record in daemon_transport["records"]
        }
        inner_by_recycles = {
            int(record["num_recycles"]): record
            for record in intervals["requests"]
        }
        for rec in (1, 2):
            transport = daemon_by_recycles.get(rec)
            inner = inner_by_recycles.get(rec)
            if (
                transport is None
                or inner is None
                or int(transport["daemon_pid"]) != int(inner["pid"])
                or int(transport["client_pid"]) != int(inner["ppid"])
            ):
                raise ParallelAcceptanceError(
                    f"daemon/inner owner identity differs for recycles={rec}"
                )
        total_request_ids = sorted(
            int(event["total_reservation_index"])
            for event in _events(public_capture)
            if event.get("event") == "model_request_reserved"
        )
        if total_request_ids != [1, 2]:
            raise ParallelAcceptanceError(
                "global pre-call request hard gate did not observe exactly "
                f"two daemon reservations: {total_request_ids}"
            )

        caller_root = out_root / "caller/runs"
        public_run_dirs = {
            profile.name: caller_root / TASK_NAME / profile.run
            for profile in PROFILES
        }
        for run_dir in public_run_dirs.values():
            if not run_dir.is_dir():
                raise ParallelAcceptanceError(
                    f"public run output is absent: {run_dir}"
                )
        artifact_root = caller_root / TASK_NAME
        artifact_paid = harness._tree_snapshot(artifact_root)
        if artifact_paid != public_actual["artifact_before_recovery"]:
            raise ParallelAcceptanceError(
                "scientific artifacts changed around the parent SIGKILL"
            )

        recovery_and_caches = _run_public_recovery_and_caches(
            out_root=out_root,
            prepared=prepared,
            artifact_paid=artifact_paid,
            timeout_s=timeout_s,
        )
        recovery = recovery_and_caches[
            "public_parent_kill_recovery"
        ]
        parallel_cache = recovery_and_caches["public_parallel_cache"]
        serial_cache = recovery_and_caches["public_serial_cache"]

        scientific = {
            profile.name: _scientific_profile_check(
                out_root=out_root,
                prepared=prepared,
                profile=profile,
                public_capture=public_capture,
                public_run_dir=public_run_dirs[profile.name],
                baseline=baselines[profile.name],
                daemon_transport=daemon_transport,
            )
            for profile in PROFILES
        }
        if any(value["status"] != "PASS" for value in scientific.values()):
            raise ParallelAcceptanceError(
                "one or more scientific profile comparisons failed"
            )
        total_actual_requests = int(
            public_actual["request_evidence"]["observed_model_requests"]
        )
        if total_actual_requests != 2:
            raise ParallelAcceptanceError(
                "actual request total differs from the frozen budget: "
                f"{total_actual_requests}"
            )
        if (
            recovery["requests"]["observed_model_requests"] != 0
            or parallel_cache["requests"]["observed_model_requests"] != 0
            or serial_cache["requests"]["observed_model_requests"] != 0
        ):
            raise ParallelAcceptanceError(
                "recovery or cache runs observed model requests"
            )

        source_after = _runtime_source_manifest(
            Path(prepared["repo_root"])
        )
        if source_after["fingerprint"] != prepared["source_fingerprint"]:
            raise ParallelAcceptanceError(
                "runtime source changed during acceptance"
            )
        summary = {
            "schema": SCHEMA,
            "status": "PASS",
            "profiles": prepared["profiles"],
            "fixedlist_control": control,
            "runtime_preflight": runtime,
            "daemon_preflight": {
                "status": "PASS",
                "model_execution": False,
                "new_model_requests": 0,
                "profiles": daemon_preflight,
            },
            "native_baseline": baselines,
            "daemon_transport": daemon_transport,
            "public_actual": public_actual,
            "public_parent_kill_recovery": recovery,
            "parallel_overlap": intervals,
            "public_parallel_cache": parallel_cache,
            "public_serial_cache": serial_cache,
            "scientific": scientific,
            "artifact_fingerprints": (
                recovery_and_caches["artifact_fingerprints"]
            ),
            "request_budget": {
                "native_baseline_reused": 2,
                "native_baseline_new": 0,
                "public_daemon_workers2": 2,
                "public_parent_kill_recovery": 0,
                "public_parallel_cache": 0,
                "public_serial_cache": 0,
                "actual_total": total_actual_requests,
                "maximum": MAX_NEW_MODEL_REQUESTS,
                "global_reservation_indices": total_request_ids,
            },
            "source_after_fingerprint": source_after["fingerprint"],
        }
        _atomic_json(out_root / "summary.json", summary)
        _atomic_json(
            state_path,
            {
                "schema": SCHEMA,
                "status": "completed",
                "completed_unix_s": time.time(),
                "summary": str(out_root / "summary.json"),
            },
        )
        return summary
    except BaseException as error:
        failure = {
            "schema": SCHEMA,
            "status": "FAIL",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "automatic_retry": False,
        }
        _atomic_json(out_root / "failure.json", failure)
        _atomic_json(
            state_path,
            {
                "schema": SCHEMA,
                "status": "failed",
                "failed_unix_s": time.time(),
                "failure": str(out_root / "failure.json"),
            },
        )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--out-root", required=True)
    prepare.add_argument(
        "--repo-root",
        default=str(REPO_ROOT_DEFAULT),
    )
    prepare.add_argument(
        "--model-dir",
        default=str(MODEL_DIR_DEFAULT),
    )
    prepare.add_argument("--sif-provenance")
    prepare.add_argument(
        "--baseline-attempt",
        default=str(BASELINE_ATTEMPT_DEFAULT),
    )

    cpu = subparsers.add_parser("cpu-preflight")
    cpu.add_argument("--out-root", required=True)
    cpu.add_argument("--timeout-s", type=int, default=600)

    run = subparsers.add_parser("run")
    run.add_argument("--out-root", required=True)
    run.add_argument("--timeout-s", type=int, default=3600)

    baseline = subparsers.add_parser("_baseline-worker")
    baseline.add_argument("--manifest", required=True)
    baseline.add_argument("--config", required=True)
    baseline.add_argument("--out-root", required=True)

    entrypoint = subparsers.add_parser("_entrypoint-worker")
    entrypoint.add_argument("--manifest", required=True)
    entrypoint.add_argument("--config", required=True)
    entrypoint.add_argument("--model-dir", required=True)
    entrypoint.add_argument("--out-root", required=True)
    entrypoint.add_argument("--guard", action="store_true")
    daemon_preflight = subparsers.add_parser(
        "_daemon-preflight-worker"
    )
    daemon_preflight.add_argument("--baseline-capture", required=True)
    daemon_preflight.add_argument("--out-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_acceptance(
                Path(args.out_root),
                repo_root=Path(args.repo_root),
                model_dir=Path(args.model_dir),
                sif_provenance=(
                    None
                    if not args.sif_provenance
                    else Path(args.sif_provenance)
                ),
                baseline_attempt=Path(args.baseline_attempt),
            )
            print(
                json.dumps(
                    {
                        "status": result["status"],
                        "prepared": str(
                            Path(args.out_root).resolve()
                            / "prepared.json"
                        ),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "cpu-preflight":
            result = run_cpu_preflight(
                Path(args.out_root),
                timeout_s=int(args.timeout_s),
            )
            print(
                json.dumps(
                    {
                        "status": result["status"],
                        "new_model_requests": 0,
                        "result": str(
                            Path(args.out_root).resolve()
                            / "cpu_preflight.json"
                        ),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "run":
            result = run_acceptance(
                Path(args.out_root),
                timeout_s=int(args.timeout_s),
            )
            print(
                json.dumps(
                    {
                        "status": result["status"],
                        "actual_total": result["request_budget"][
                            "actual_total"
                        ],
                        "result": str(
                            Path(args.out_root).resolve()
                            / "summary.json"
                        ),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "_baseline-worker":
            return _load_harness()._baseline_worker(
                manifest_path=Path(args.manifest),
                config_path=Path(args.config),
                out_root=Path(args.out_root),
            )
        if args.command == "_entrypoint-worker":
            return _entrypoint_preflight_worker(
                manifest_path=Path(args.manifest),
                config_path=Path(args.config),
                model_dir=Path(args.model_dir),
                out_dir=Path(args.out_root),
                guard=bool(args.guard),
            )
        if args.command == "_daemon-preflight-worker":
            return _daemon_preflight_worker(
                baseline_capture=Path(args.baseline_capture),
                out_root=Path(args.out_root),
            )
    except BaseException as error:
        print(f"posthoc parallel daemon failed: {error}", file=sys.stderr)
        return 1
    raise ParallelAcceptanceError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
