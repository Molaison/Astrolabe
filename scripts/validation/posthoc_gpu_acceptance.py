#!/usr/bin/env python3
"""Bounded real-GPU acceptance for the public posthoc refold caller.

This harness freezes one 20-residue protein manifest, runs that manifest through
``python -m astrolabe.cli posthoc refold-run``, and compares the result with one
direct AlphaFold3 evaluator baseline.  A third invocation, the cache/resume
check, is run with a zero-request middleware allowance.

The two model requests are counted by wrapping
``astrolabe.evaluators.wrappers.alphafold3.ModelRunner.run_inference`` in each
child process.  File counts are used only for artifact identity, never as a
substitute for observed model calls.

Typical bounded execution inside the prepared AF3 SIF::

    /alphafold3_venv/bin/python \
      /opt/Astrolabe/scripts/validation/posthoc_gpu_acceptance.py prepare \
      --out-root /evidence/posthoc-refold-acceptance \
      --repo-root /opt/Astrolabe

    /alphafold3_venv/bin/python \
      /opt/Astrolabe/scripts/validation/posthoc_gpu_acceptance.py run \
      --out-root /evidence/posthoc-refold-acceptance \
      --repo-root /opt/Astrolabe

``prepare`` is CPU-only.  This worker does not launch the GPU command; the
parent owns the GPU run and its scheduling.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "astrolabe.posthoc-refold-gpu-acceptance/v1"
PREPARED_SCHEMA = "astrolabe.posthoc-refold-gpu-acceptance.prepared/v1"
SOURCE_SCHEMA = "astrolabe.posthoc-refold-gpu-acceptance.source/v1"
SUMMARY_SCHEMA = "astrolabe.posthoc-refold-gpu-acceptance.summary/v1"

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
SIF_PROVENANCE_SCHEMA = "astrolabe.af3-sif-provenance/v1"
RUNTIME_PREFLIGHT_SCHEMA = "astrolabe.posthoc-refold.runtime-preflight/v1"

TASK_NAME = "posthoc_refold_gpu_acceptance"
RUN_NAME = "single20_seed17"
OLIGO = "A"
SEQUENCE = "ACDEFGHIKLMNPQRSTVWY"
SEED = 17
NUM_SEEDS = 1
NUM_SAMPLES = 1
NUM_RECYCLES = 1
MAX_REAL_GPU_REQUESTS = 2


class AcceptanceError(RuntimeError):
    """The acceptance evidence cannot be trusted."""


class FrozenInputChanged(AcceptanceError):
    """A frozen source or input changed."""


_SHARED_VALIDATION_MODULE: Any = None


def _shared_validation():
    """Load existing validation helpers without modifying their owner script."""

    global _SHARED_VALIDATION_MODULE
    if _SHARED_VALIDATION_MODULE is None:
        path = Path(__file__).with_name("unified_gpu_acceptance.py")
        spec = importlib.util.spec_from_file_location(
            "_posthoc_shared_gpu_acceptance",
            path,
        )
        if spec is None or spec.loader is None:
            raise AcceptanceError(f"cannot load shared validation helper: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _SHARED_VALIDATION_MODULE = module
    return _SHARED_VALIDATION_MODULE


def _atomic_json(path: Path, value: Any) -> None:
    _shared_validation()._atomic_json(path, value)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise AcceptanceError(
                f"{path}:{line_number} is not a JSON object"
            )
        records.append(value)
    return records


def _sha256(path: Path) -> str:
    return _shared_validation()._sha256(path)


def _hash_value(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _manifest_inputs_hash(item: Mapping[str, Any]) -> str:
    value = {
        "task": item.get("task"),
        "run": item.get("run"),
        "step": item.get("step"),
        "oligo": item.get("oligo"),
        "seqs": item.get("seqs"),
        "types": item.get("types"),
        "homologies": item.get("homologies"),
        "cyclization": item.get("cyclization"),
        "nonstandard_residues": item.get("nonstandard_residues"),
    }
    return _hash_value(value)[:16]


def runtime_source_manifest(repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.expanduser().resolve()
    shared = _shared_validation()
    files = dict(shared._read_source_manifest(repo_root).files)
    harness = Path(__file__).resolve()
    try:
        harness_relative = str(harness.relative_to(repo_root))
    except ValueError:
        pass
    else:
        files[harness_relative] = _sha256(harness)
    if not files:
        raise AcceptanceError(f"no runtime sources found under {repo_root}")
    fingerprint = _hash_value(files)
    return {
        "schema": SOURCE_SCHEMA,
        "root": str(repo_root),
        "fingerprint": fingerprint,
        "files": files,
    }


def _source_changes(
    expected: Mapping[str, str],
    observed: Mapping[str, str],
) -> list[dict[str, Any]]:
    changes = []
    for path in sorted(set(expected) | set(observed)):
        before = expected.get(path)
        after = observed.get(path)
        if before != after:
            changes.append({"path": path, "before": before, "after": after})
    return changes


def _manifest_record(path: Path) -> dict[str, Any]:
    records = _read_jsonl(path)
    if len(records) != 1:
        raise AcceptanceError(
            f"acceptance requires exactly one manifest record, got {len(records)}"
        )
    return records[0]


def _load_sif_provenance(
    path: Path,
    *,
    expected_path: Path,
) -> dict[str, Any]:
    provenance_path = Path(path).expanduser().resolve()
    if not provenance_path.is_file():
        raise AcceptanceError(
            f"host SIF provenance is absent from the container view: {provenance_path}"
        )
    value = _read_json(provenance_path)
    sif_path = Path(str(value.get("path") or "")).expanduser()
    sha256 = value.get("sha256")
    if value.get("schema") != SIF_PROVENANCE_SCHEMA:
        raise AcceptanceError("host SIF provenance has another schema")
    if sif_path != Path(expected_path).expanduser():
        raise AcceptanceError(
            f"host SIF provenance path differs: {sif_path} != {expected_path}"
        )
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
    ):
        raise AcceptanceError("host SIF provenance has an invalid SHA256")
    return {
        "file": str(provenance_path),
        "file_sha256": _sha256(provenance_path),
        "path": str(sif_path),
        "sha256": sha256,
    }


def verify_frozen_inputs(out_root: Path) -> dict[str, Any]:
    """Verify source, manifest, and config hashes recorded by ``prepare``."""

    out_root = Path(out_root).expanduser().resolve()
    prepared_path = out_root / "prepared.json"
    if not prepared_path.is_file():
        raise AcceptanceError(f"prepared inputs are absent: {prepared_path}")
    prepared = _read_json(prepared_path)
    repo_root = Path(prepared["repo_root"]).expanduser().resolve()

    expected_source_path = out_root / "source_manifest.json"
    expected_source = _read_json(expected_source_path)
    observed_source = runtime_source_manifest(repo_root)
    if expected_source.get("fingerprint") != observed_source["fingerprint"]:
        changes = _source_changes(
            expected_source.get("files", {}),
            observed_source["files"],
        )
        raise FrozenInputChanged(
            "runtime source changed after preparation: "
            + json.dumps(changes[:20], sort_keys=True)
        )

    checked_inputs = {}
    for key in ("manifest", "config"):
        path = out_root / prepared["inputs"][key]
        expected = prepared["input_hashes"][key]
        if not path.is_file():
            raise FrozenInputChanged(f"frozen {key} is missing: {path}")
        observed = _sha256(path)
        checked_inputs[key] = {
            "path": str(path),
            "expected_sha256": expected,
            "observed_sha256": observed,
            "equal": observed == expected,
        }
        if observed != expected:
            raise FrozenInputChanged(
                f"frozen {key} changed: expected {expected}, observed {observed}"
            )

    item = _manifest_record(out_root / prepared["inputs"]["manifest"])
    if item.get("inputs_hash") != _manifest_inputs_hash(item):
        raise FrozenInputChanged("manifest inputs_hash is inconsistent")
    if prepared.get("sif_provenance"):
        recorded = prepared["sif_provenance"]
        observed = _load_sif_provenance(
            Path(recorded["file"]),
            expected_path=Path(prepared["sif"]),
        )
        if observed != recorded:
            raise FrozenInputChanged("host SIF provenance changed")

    from omegaconf import OmegaConf

    cfg = OmegaConf.load(str(out_root / prepared["inputs"]["config"]))
    expected_scientific = {
        "seed": SEED,
        "num_seeds": NUM_SEEDS,
        "num_samples": NUM_SAMPLES,
        "num_recycles": NUM_RECYCLES,
    }
    observed_scientific = {
        "seed": int(cfg.seed),
        "num_seeds": int(cfg.evaluator.alphafold3.num_seeds),
        "num_samples": int(cfg.evaluator.alphafold3.num_samples),
        "num_recycles": int(cfg.evaluator.alphafold3.num_recycles),
    }
    if observed_scientific != expected_scientific:
        raise FrozenInputChanged(
            "scientific settings changed: "
            f"expected {expected_scientific}, observed {observed_scientific}"
        )
    term = cfg.evaluator.evaluator_terms.Alphafold3
    for key, expected in (
        ("num_seeds", NUM_SEEDS),
        ("num_samples", NUM_SAMPLES),
        ("num_recycles", NUM_RECYCLES),
        ("run_inference", True),
        ("run_data_pipeline", False),
    ):
        if getattr(term, key) != expected:
            raise FrozenInputChanged(
                f"evaluator term {key} differs: expected {expected!r}"
            )
    for key, expected in (
        ("run_inference", True),
        ("run_data_pipeline", False),
        ("execution", "in_process"),
    ):
        if getattr(cfg.evaluator.alphafold3, key) != expected:
            raise FrozenInputChanged(
                f"evaluator.alphafold3.{key} differs: expected {expected!r}"
            )
    if dict(cfg.types) != {OLIGO: "p"}:
        raise AcceptanceError("fixture types differ from one protein chain A")
    if item.get("seqs") != {OLIGO: SEQUENCE}:
        raise AcceptanceError("manifest sequence differs from the frozen fixture")

    source_run_dir = Path(item["src_run_dir"])
    source_marker = (
        source_run_dir
        / "raw_model_results"
        / "alphafold3"
        / f"step_{int(item['step']):03d}_{OLIGO}"
        / f"step_{int(item['step']):03d}_{OLIGO}_summary_confidences.json"
    )
    if source_marker.exists():
        raise AcceptanceError(
            "source run unexpectedly contains an AF3 completion marker; the "
            "first public caller could reuse it without exercising inference"
        )

    return {
        "prepared": prepared,
        "manifest_item": item,
        "source": observed_source,
        "checked_inputs": checked_inputs,
    }


def posthoc_cache_assessment() -> dict[str, Any]:
    """State the exact cache boundary this acceptance run relies on."""

    return {
        "identical_manifest_same_out_root": {
            "expected_model_requests": 0,
            "reason": (
                "A committed destination receipt matches the scientific "
                "identity, selected assets, numerical sources, and output "
                "contract, so run_refold skips before evaluator construction."
            ),
            "requires_committed_receipt": True,
            "requires_output_contract_match": True,
        },
        "changed_manifest_or_config_same_out_root": {
            "detected_by_committed_receipt": True,
            "classification": "REJECTED_WITHOUT_REPLAY",
        },
        "source_run_output_reuse": {
            "committed_receipt_required": True,
            "identity_checked": True,
            "legacy_receiptless_outputs_reused": False,
            "classification": "SAFE",
        },
        "acceptance_guards": [
            "Freeze all numerical runtime source hashes before the first request.",
            "Freeze manifest and source config hashes before the first request.",
            "Reject a source run with a pre-existing AF3 completion marker.",
            "Require a completed refold receipt with matching input and output contracts.",
            "Treat issued, legacy, or mismatched target receipts as terminal cache conflicts.",
            "Snapshot every caller artifact before and after the cache run.",
            "Deny any cache-run model callback before the original GPU call.",
        ],
        "conclusion": (
            "Zero-request resume is evidence-safe when the completed receipt "
            "matches the current identity and output contract. A completion "
            "marker alone is never a cache proof."
        ),
    }


def _fixture_config(out_root: Path, model_dir: Path) -> dict[str, Any]:
    af3 = {
        "execution": "in_process",
        "model_dir": str(model_dir),
        "flash_attention_implementation": "cudnn",
        "num_seeds": NUM_SEEDS,
        "num_recycles": NUM_RECYCLES,
        "num_samples": NUM_SAMPLES,
        "save_embeddings": False,
        "run_data_pipeline": False,
        "run_inference": True,
        "bucket_pad_tolerance": 0,
        "save_all_samples": False,
        "jax_compilation_cache_dir": str(out_root / "cache" / "jax"),
    }
    term = {
        "_type": "Alphafold3",
        **{
            key: value
            for key, value in af3.items()
            if key not in {"model_dir", "save_embeddings"}
        },
    }
    return {
        "task_name": TASK_NAME,
        "out_dir": str(out_root / "inputs" / "source_run"),
        "seed": SEED,
        "repeats": 1,
        "steps": 1,
        "mutation_num": 0,
        "enable_pyrosetta": False,
        "skip_complex_poses": True,
        "types": {OLIGO: "p"},
        "seqs": {OLIGO: SEQUENCE},
        "homologies": {},
        "cyclization": None,
        "nonstandard_residues": None,
        "evaluator": {
            "evaluator_terms": {"Alphafold3": term},
            "alphafold3": af3,
        },
        "weight_path": {"alphafold3": str(model_dir)},
        "stages": {
            "default": {
                "evaluator": "Alphafold3",
                "oligo": [OLIGO],
                "mutators": [],
                "mutation_num": 0,
                "weighted_score_expression": "plddt",
            }
        },
    }


def _prepared_commands(
    *,
    repo_root: Path,
    out_root: Path,
    manifest: Path,
    config: Path,
) -> dict[str, list[str]]:
    script = repo_root / "scripts" / "validation" / "posthoc_gpu_acceptance.py"
    public = [
        sys.executable,
        "-m",
        "astrolabe.cli",
        "posthoc",
        "refold-run",
        "--manifest",
        str(manifest),
        "--out-root",
        str(out_root / "caller" / "runs"),
        "--evaluators",
        "af3",
        "--config-yaml",
        str(config),
        "--workers",
        "1",
        "--format",
        "json",
    ]
    baseline = [
        sys.executable,
        str(script),
        "_native-baseline",
        "--manifest",
        str(manifest),
        "--config",
        str(config),
        "--out-root",
        str(out_root / "baseline"),
    ]
    return {"public_caller": public, "native_baseline": baseline}


def prepare_acceptance(
    out_root: Path,
    *,
    repo_root: Path = REPO_ROOT_DEFAULT,
    model_dir: Path = MODEL_DIR_DEFAULT,
    sif_provenance: Path | None = None,
) -> dict[str, Any]:
    """Create the frozen manifest/config and source hashes without model work."""

    out_root = Path(out_root).expanduser().resolve()
    repo_root = Path(repo_root).expanduser().resolve()
    model_dir = Path(model_dir).expanduser().resolve()
    prepared_path = out_root / "prepared.json"
    if prepared_path.exists():
        verify_frozen_inputs(out_root)
        return _read_json(prepared_path)
    if out_root.exists() and any(out_root.iterdir()):
        raise AcceptanceError(
            f"out-root is not empty and has no prepared.json: {out_root}"
        )
    if not (repo_root / "astrolabe" / "posthoc" / "refold.py").is_file():
        raise AcceptanceError(f"invalid Astrolabe repo root: {repo_root}")

    out_root.mkdir(parents=True, exist_ok=True)
    try:
        out_root.chmod(0o700)
    except OSError:
        pass
    source_dir = out_root / "inputs" / "source_run"
    source_dir.mkdir(parents=True)
    config_path = source_dir / "config.yaml"
    manifest_path = out_root / "inputs" / "manifest.jsonl"

    from omegaconf import OmegaConf

    OmegaConf.save(
        config=OmegaConf.create(_fixture_config(out_root, model_dir)),
        f=str(config_path),
    )
    item: dict[str, Any] = {
        "id": "posthoc_gpu_acceptance__single20_seed17__step_000__A",
        "task": TASK_NAME,
        "run": RUN_NAME,
        "step": 0,
        "oligo": OLIGO,
        "src_run_dir": str(source_dir),
        "src_config": str(config_path),
        "seqs": {OLIGO: SEQUENCE},
        "types": {OLIGO: "p"},
        "homologies": {OLIGO: None},
        "needed_evaluators": ["af3"],
        "cyclization": None,
        "nonstandard_residues": None,
        "stage1": {
            "selector_kind": "acceptance",
            "selector": "frozen-single20aa",
            "passed": True,
            "failed": [],
            "missing": [],
        },
        "created_at": 0,
    }
    item["inputs_hash"] = _manifest_inputs_hash(item)
    manifest_path.write_text(
        json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    source_manifest = runtime_source_manifest(repo_root)
    _atomic_json(out_root / "source_manifest.json", source_manifest)
    provenance = (
        _load_sif_provenance(
            sif_provenance,
            expected_path=SIF_PATH,
        )
        if sif_provenance is not None
        else None
    )
    prepared = {
        "schema": PREPARED_SCHEMA,
        "acceptance_schema": SCHEMA,
        "out_root": str(out_root),
        "repo_root": str(repo_root),
        "model_dir": str(model_dir),
        "model_file": str(model_dir / "af3.bin"),
        "sif": str(SIF_PATH),
        "sif_sha256": provenance["sha256"] if provenance else None,
        "sif_provenance": provenance,
        "sif_python": sys.executable,
        "fixture": {
            "name": "single-20aa",
            "sequence": SEQUENCE,
            "types": {OLIGO: "p"},
            "oligo": OLIGO,
        },
        "scientific_settings": {
            "seed": SEED,
            "num_seeds": NUM_SEEDS,
            "num_samples": NUM_SAMPLES,
            "num_recycles": NUM_RECYCLES,
            "run_data_pipeline": False,
            "run_inference": True,
        },
        "inputs": {
            "manifest": str(manifest_path.relative_to(out_root)),
            "config": str(config_path.relative_to(out_root)),
        },
        "input_hashes": {
            "manifest": _sha256(manifest_path),
            "config": _sha256(config_path),
        },
        "source_fingerprint": source_manifest["fingerprint"],
        "commands": _prepared_commands(
            repo_root=repo_root,
            out_root=out_root,
            manifest=manifest_path,
            config=config_path,
        ),
        "request_contract": {
            "caller": 1,
            "native_baseline": 1,
            "cache_resume": 0,
            "maximum_real_gpu_requests": MAX_REAL_GPU_REQUESTS,
            "count_source": (
                "model_inner_invocation_started/completed events after "
                "model_rpc_delivered and model_request_reserved; the pre-call "
                "budget remains before the original inner ModelRunner.run_inference"
            ),
        },
        "cache_assessment": posthoc_cache_assessment(),
    }
    _atomic_json(prepared_path, prepared)
    return prepared


_SITECUSTOMIZE = r'''
"""Process-local observability installed only for posthoc acceptance children."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import pickle
import dataclasses
import datetime
import enum
from collections.abc import Mapping as _Mapping
import signal
import threading
import time

import numpy as np


_ENABLED = os.environ.get("ASTROLABE_POSTHOC_ACCEPTANCE_CAPTURE") == "1"


def _safe(value):
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def _append(record):
    path = Path(os.environ["ASTROLABE_POSTHOC_ACCEPTANCE_CAPTURE_DIR"])
    path.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(
        path / "events.jsonl",
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o600,
    )
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _collect(value, prefix, arrays):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise RuntimeError(f"object-dtype array is unsupported at {prefix}")
        arrays[prefix] = np.asarray(value)
        return
    if isinstance(value, np.generic):
        if value.dtype.hasobject:
            raise RuntimeError(f"object-dtype scalar is unsupported at {prefix}")
        arrays[prefix] = np.asarray(value)
        return
    # AF3 receives its PRNG key as a jaxlib ArrayImpl, which is not a numpy
    # ndarray even though it has numeric dtype/shape and supports np.asarray.
    if type(value).__module__.split(".", 1)[0] in {"jax", "jaxlib"}:
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise RuntimeError(f"object-dtype array is unsupported at {prefix}")
        arrays[prefix] = array
        return
    if isinstance(value, (bytes, bytearray, memoryview)):
        arrays[prefix] = np.frombuffer(bytes(value), dtype=np.uint8)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _collect(item, f"{prefix}__{_safe(key)}", arrays)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _collect(item, f"{prefix}__{index:06d}", arrays)


def _write_arrays(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_json_atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, separators=(",", ":"))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _type_name(value):
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _resource_owner(value):
    pid = os.getpid()
    device = getattr(value, "_device", None)
    model_config = getattr(value, "_model_config", None)
    model_dir = getattr(value, "_model_dir", None)
    model_params = getattr(model_config, "num_recycles", None)
    diffusion = getattr(model_config, "heads", None)
    try:
        num_samples = int(diffusion.diffusion.eval.num_samples)
    except Exception:
        num_samples = None
    record = {
        "pid": pid,
        "ppid": os.getppid(),
        "runner_type": _type_name(value),
        "instance_token": hashlib.sha256(
            f"{pid}:{id(value)}".encode("utf-8")
        ).hexdigest(),
        "in_process_runner": device is not None,
        "model_dir": None if model_dir is None else str(model_dir),
        "num_recycles": (
            None if model_params is None else int(model_params)
        ),
        "num_samples": num_samples,
        "device": None,
        "model_state": {
            "params_cached_before": "model_params" in vars(value),
            "jit_model_cached_before": "_model" in vars(value),
        },
    }
    if device is not None:
        record["device"] = {
            "platform": str(getattr(device, "platform", "")),
            "kind": str(getattr(device, "device_kind", "")),
            "id": int(getattr(device, "id", -1)),
        }
    return record


def _global_claim(kind, *, maximum, root=None):
    claims = (
        _capture_dir if root is None else Path(root)
    ) / "_claims"
    claims.mkdir(parents=True, exist_ok=True)
    for index in range(1, maximum + 1):
        try:
            descriptor = os.open(
                claims / f"{kind}_{index:06d}.claim",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError:
            continue
        try:
            os.write(
                descriptor,
                f"{os.getpid()}:{time.time_ns()}\n".encode("ascii"),
            )
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return index
    raise RuntimeError("acceptance global request claim space exhausted")


def _scalar(value):
    if value is None or isinstance(value, (bool, int, str)):
        return {
            "kind": "scalar",
            "type": _type_name(value),
            "value": value,
        }
    if isinstance(value, float):
        return {
            "kind": "scalar",
            "type": _type_name(value),
            "value_hex": value.hex(),
        }
    if isinstance(value, (bytes, bytearray, memoryview)):
        payload = bytes(value)
        return {
            "kind": "bytes",
            "type": _type_name(value),
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    if isinstance(value, (datetime.date, datetime.datetime, datetime.time)):
        return {
            "kind": "datetime",
            "type": _type_name(value),
            "value": value.isoformat(),
        }
    if isinstance(value, enum.Enum):
        return {
            "kind": "enum",
            "type": _type_name(value),
            "name": value.name,
            "value": _scalar(value.value),
        }
    return None


def _canonical_tree(value, prefix, counts):
    scalar = _scalar(value)
    if scalar is not None:
        return scalar

    if isinstance(value, np.generic):
        scalar = _scalar(value.item())
        if scalar is not None:
            scalar["numpy_type"] = _type_name(value)
            return scalar

    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        # Empty ndarrays have no addressable elements, and NumPy transports
        # may reconstruct them with either stride zero or the item-size
        # stride. Canonicalize only this semantically empty case; nonzero-size
        # strides remain part of the digest and fail closed.
        strides = (
            [0] * array.ndim
            if array.size == 0
            else list(array.strides)
        )
        common = {
            "path": prefix,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "strides": strides,
        }
        counts["array_count"] += 1
        if not array.dtype.hasobject:
            contiguous = np.ascontiguousarray(array)
            counts["numeric_array_count"] += 1
            return {
                "kind": "numeric_array",
                **common,
                "size": int(contiguous.size),
                "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
            }

        counts["object_array_count"] += 1
        items = [
            _canonical_tree(item, f"{prefix}__{index:06d}", counts)
            for index, item in enumerate(array.reshape(-1))
        ]
        compact = bool(items) and all(
            item.get("kind") in {"scalar", "bytes", "datetime", "enum"}
            for item in items
        )
        if compact and array.ndim:
            encoded = json.dumps(
                items,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            return {
                "kind": "object_array",
                **common,
                "size": int(array.size),
                "element_sha256": hashlib.sha256(encoded).hexdigest(),
                "element_types": sorted(
                    {item["type"] for item in items}
                ),
                "nonempty_element_count": sum(
                    item.get("value") not in {None, ""}
                    or item.get("size", 0) > 0
                    for item in items
                ),
            }
        return {
            "kind": "object_array",
            **common,
            "size": int(array.size),
            "items": items,
        }

    if isinstance(value, _Mapping):
        counts["mapping_count"] += 1
        if all(isinstance(key, str) for key in value):
            items = {
                key: _canonical_tree(
                    value[key],
                    f"{prefix}__{_safe(key)}",
                    counts,
                )
                for key in sorted(value)
            }
        else:
            pairs = [
                {
                    "key": _canonical_tree(
                        key,
                        f"{prefix}__key_{index:06d}",
                        counts,
                    ),
                    "value": _canonical_tree(
                        value[key],
                        f"{prefix}__value_{index:06d}",
                        counts,
                    ),
                }
                for index, key in enumerate(
                    sorted(value, key=lambda item: repr(item))
                )
            ]
            items = pairs
        return {"kind": "mapping", "items": items}

    type_name = _type_name(value)
    if type_name == "alphafold3.structure.structure.Structure":
        counts["native_object_count"] += 1
        table_names = tuple(sorted(getattr(value, "tables", ())))
        return {
            "kind": "af3_structure",
            "type": type_name,
            "fields": {
                "scalars": {
                    name: _canonical_tree(
                        getattr(value, name),
                        f"{prefix}__{name}",
                        counts,
                    )
                    for name in (
                        "name",
                        "release_date",
                        "resolution",
                        "structure_method",
                        "bioassembly_data",
                        "chemical_components_data",
                    )
                },
                "tables": {
                    name: _canonical_tree(
                        value.get_table(name),
                        f"{prefix}__tables__{name}",
                        counts,
                    )
                    for name in table_names
                },
            },
        }

    if dataclasses.is_dataclass(value):
        counts["native_object_count"] += 1
        return {
            "kind": "dataclass",
            "type": type_name,
            "fields": {
                field.name: _canonical_tree(
                    getattr(value, field.name),
                    f"{prefix}__{field.name}",
                    counts,
                )
                for field in dataclasses.fields(value)
            },
        }

    if isinstance(value, tuple) and hasattr(value, "_fields"):
        counts["native_object_count"] += 1
        return {
            "kind": "namedtuple",
            "type": type_name,
            "fields": {
                name: _canonical_tree(
                    getattr(value, name),
                    f"{prefix}__{name}",
                    counts,
                )
                for name in value._fields
            },
        }

    if isinstance(value, (list, tuple)):
        counts["sequence_count"] += 1
        return {
            "kind": "sequence",
            "type": type_name,
            "items": [
                _canonical_tree(
                    item,
                    f"{prefix}__{index:06d}",
                    counts,
                )
                for index, item in enumerate(value)
            ],
        }

    if isinstance(value, (set, frozenset)):
        counts["sequence_count"] += 1
        items = [
            _canonical_tree(
                item,
                f"{prefix}__{index:06d}",
                counts,
            )
            for index, item in enumerate(
                sorted(value, key=lambda item: repr(item))
            )
        ]
        encoded = json.dumps(
            items,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return {
            "kind": "set",
            "type": type_name,
            "size": len(items),
            "items_sha256": hashlib.sha256(encoded).hexdigest(),
        }

    raise TypeError(
        "unsupported native object in featurised input at "
        f"{prefix}: {type_name}"
    )


def _digest_tree(value):
    counts = {
        "array_count": 0,
        "numeric_array_count": 0,
        "object_array_count": 0,
        "native_object_count": 0,
        "mapping_count": 0,
        "sequence_count": 0,
    }
    tree = _canonical_tree(value, "input", counts)
    encoded = json.dumps(
        tree,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return (
        {"sha256": digest, **counts},
        {
            "schema": "astrolabe.posthoc-refold.featurised-input-snapshot/v1",
            "scope": {
                "numeric_arrays": (
                    "all recursively reachable numeric ndarray buffers are "
                    "hashed with dtype, shape, path, and raw bytes"
                ),
                "object_arrays": (
                    "0-D native objects are expanded through declared AF3 "
                    "Structure, dataclass, mapping, sequence, enum, date, "
                    "primitive, and bytes fields; multi-dimensional primitive "
                    "object arrays are hashed elementwise"
                ),
                "unsupported_objects": "fail closed; never dropped",
                "raw_buffers_in_json": False,
            },
            "counts": counts,
            "tree": tree,
        },
    )


if _ENABLED:
    from astrolabe.evaluators.alphafold3 import AlphaFold3Evaluator
    from astrolabe.evaluators.wrappers.alphafold3 import ModelRunner

    _capture_dir = Path(
        os.environ["ASTROLABE_POSTHOC_ACCEPTANCE_CAPTURE_DIR"]
    )
    _capture_dir.mkdir(parents=True, exist_ok=True)
    _stage = os.environ.get("ASTROLABE_POSTHOC_ACCEPTANCE_STAGE", "unknown")
    _max_requests = max(
        0,
        int(os.environ.get("ASTROLABE_POSTHOC_ACCEPTANCE_MAX_REQUESTS", "0")),
    )
    _total_ledger_root = Path(
        os.environ.get(
            "ASTROLABE_POSTHOC_ACCEPTANCE_TOTAL_LEDGER_DIR",
            str(_capture_dir),
        )
    ).resolve()
    _total_max_requests = max(
        0,
        int(
            os.environ.get(
                "ASTROLABE_POSTHOC_ACCEPTANCE_TOTAL_MAX_REQUESTS",
                str(_max_requests),
            )
        ),
    )
    _delivery_count = 0
    _reservation_count = 0
    _inner_invocation_count = 0
    _evaluator_count = 0
    _lock = threading.Lock()
    _original_inference = ModelRunner.run_inference
    _original_evaluate = AlphaFold3Evaluator.evaluate
    _global_ledger = (
        os.environ.get("ASTROLABE_POSTHOC_ACCEPTANCE_GLOBAL_LEDGER") == "1"
    )

    def _observed_inference(self, featurised_example, rng_key):
        global _delivery_count, _reservation_count, _inner_invocation_count
        owner_before = _resource_owner(self)
        with _lock:
            _delivery_count += 1
            delivery_index = _delivery_count
        runner = type(self).__module__ + "." + type(self).__qualname__
        delivered = time.time_ns()
        _append(
            {
                "event": "model_rpc_delivered",
                "stage": _stage,
                "pid": os.getpid(),
                "delivery_index": delivery_index,
                "runner": runner,
                "resource_owner": owner_before,
                "time_ns": delivered,
            }
        )
        with _lock:
            _reservation_count += 1
            local_reservation_index = _reservation_count
        reservation_index = (
            _global_claim("request", maximum=_max_requests + 1)
            if _global_ledger
            else local_reservation_index
        )
        total_reservation_index = _global_claim(
            "request_total",
            maximum=_total_max_requests + 1,
            root=_total_ledger_root,
        )
        within_budget = (
            reservation_index <= _max_requests
            and total_reservation_index <= _total_max_requests
        )
        _append(
            {
                "event": "model_request_reserved",
                "stage": _stage,
                "pid": os.getpid(),
                "delivery_index": delivery_index,
                "local_reservation_index": local_reservation_index,
                "reservation_index": reservation_index,
                "total_reservation_index": total_reservation_index,
                "global_ledger": _global_ledger,
                "within_budget": within_budget,
                "max_requests": _max_requests,
                "total_max_requests": _total_max_requests,
                "resource_owner": owner_before,
                "time_ns": time.time_ns(),
            }
        )
        if not within_budget:
            _append(
                {
                    "event": "request_budget_exceeded",
                    "stage": _stage,
                    "pid": os.getpid(),
                    "delivery_index": delivery_index,
                    "reservation_index": reservation_index,
                    "max_requests": _max_requests,
                    "time_ns": time.time_ns(),
                }
            )
            raise RuntimeError(
                f"acceptance request budget {_max_requests} exceeded before "
                f"inner ModelRunner.run_inference reservation {reservation_index}; "
                f"total reservation {total_reservation_index} has maximum "
                f"{_total_max_requests}"
            )

        stem = f"request_{reservation_index:03d}_pid_{os.getpid()}"
        started = time.time_ns()
        try:
            input_digest, input_snapshot = _digest_tree(featurised_example)
            rng_arrays = {}
            _collect(rng_key, "rng_key", rng_arrays)
            rng_path = _capture_dir / f"{stem}_rng_key.npz"
            _write_arrays(rng_path, rng_arrays)
            input_snapshot_path = (
                _capture_dir / f"{stem}_input_snapshot.json"
            )
            _write_json_atomic(input_snapshot_path, input_snapshot)
            input_snapshot_sha256 = hashlib.sha256(
                input_snapshot_path.read_bytes()
            ).hexdigest()
        except BaseException as error:
            _append(
                {
                    "event": "model_entrypoint_failed",
                    "stage": _stage,
                    "pid": os.getpid(),
                    "delivery_index": delivery_index,
                    "reservation_index": reservation_index,
                    "phase": "input_capture_before_inner",
                    "time_ns": time.time_ns(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            raise

        with _lock:
            _inner_invocation_count += 1
            inner_index = _inner_invocation_count
        inner_started = time.time_ns()
        _append(
            {
                "event": "model_inner_invocation_started",
                "stage": _stage,
                "pid": os.getpid(),
                "delivery_index": delivery_index,
                "reservation_index": reservation_index,
                "inner_index": inner_index,
                "runner": runner,
                "request_id": reservation_index,
                "total_request_id": total_reservation_index,
                "resource_owner": owner_before,
                "time_ns": inner_started,
            }
        )
        try:
            result = _original_inference(self, featurised_example, rng_key)
        except BaseException as error:
            _append(
                {
                    "event": "model_inner_invocation_failed",
                    "stage": _stage,
                    "pid": os.getpid(),
                    "delivery_index": delivery_index,
                    "reservation_index": reservation_index,
                    "inner_index": inner_index,
                    "time_ns": time.time_ns(),
                    "duration_ns": time.time_ns() - inner_started,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            raise
        inner_completed = time.time_ns()
        owner_after = _resource_owner(self)
        owner_after["model_state"].update(
            {
                "params_cached_after": "model_params" in vars(self),
                "jit_model_cached_after": "_model" in vars(self),
            }
        )
        _append(
            {
                "event": "model_inner_invocation_completed",
                "stage": _stage,
                "pid": os.getpid(),
                "delivery_index": delivery_index,
                "reservation_index": reservation_index,
                "inner_index": inner_index,
                "runner": runner,
                "request_id": reservation_index,
                "total_request_id": total_reservation_index,
                "resource_owner": owner_after,
                "time_ns": inner_completed,
                "duration_ns": inner_completed - inner_started,
            }
        )
        try:
            result_arrays = {}
            _collect(result, "result", result_arrays)
            result_path = _capture_dir / f"{stem}_result.npz"
            _write_arrays(result_path, result_arrays)
        except BaseException as error:
            _append(
                {
                    "event": "model_raw_capture_failed",
                    "stage": _stage,
                    "pid": os.getpid(),
                    "delivery_index": delivery_index,
                    "reservation_index": reservation_index,
                    "inner_index": inner_index,
                    "phase": "result_capture_after_inner",
                    "time_ns": time.time_ns(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            raise
        _append(
            {
                "event": "model_entrypoint_completed",
                "stage": _stage,
                "pid": os.getpid(),
                "delivery_index": delivery_index,
                "reservation_index": reservation_index,
                "inner_index": inner_index,
                "request_id": reservation_index,
                "total_request_id": total_reservation_index,
                "resource_owner": owner_after,
                "time_ns": time.time_ns(),
                "duration_ns": time.time_ns() - started,
                "input_digest": input_digest,
                "input_snapshot_path": str(input_snapshot_path),
                "input_snapshot_sha256": input_snapshot_sha256,
                "rng_key_path": str(rng_path),
                "result_path": str(result_path),
                "result_sha256": hashlib.sha256(
                    result_path.read_bytes()
                ).hexdigest(),
            }
        )
        return result

    def _observed_evaluate(self, inputs=None, runtime_config=None):
        global _evaluator_count
        model_runner = getattr(self, "model_runner", None)
        evaluator_owner = (
            None
            if model_runner is None
            else _resource_owner(model_runner)
        )
        if _global_ledger:
            evaluator_index = _global_claim(
                "evaluator",
                maximum=100000,
            )
        else:
            with _lock:
                _evaluator_count += 1
                evaluator_index = _evaluator_count
        path = (
            _capture_dir
            / f"scores_{os.getpid()}_{evaluator_index:06d}.pkl"
        )
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        _append(
            {
                "event": "evaluator_started",
                "stage": _stage,
                "pid": os.getpid(),
                "evaluator_index": evaluator_index,
                "resource_owner": evaluator_owner,
                "time_ns": time.time_ns(),
            }
        )
        try:
            scores = _original_evaluate(self, inputs, runtime_config)
            with temporary.open("wb") as handle:
                pickle.dump(scores, handle, protocol=5)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except BaseException as error:
            _append(
                {
                    "event": "evaluator_failed",
                    "stage": _stage,
                    "pid": os.getpid(),
                    "evaluator_index": evaluator_index,
                    "resource_owner": evaluator_owner,
                    "time_ns": time.time_ns(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            raise
        _append(
            {
                "event": "evaluator_completed",
                "stage": _stage,
                "pid": os.getpid(),
                "evaluator_index": evaluator_index,
                "resource_owner": (
                    _resource_owner(model_runner)
                    if model_runner is not None
                    else None
                ),
                "time_ns": time.time_ns(),
                "scores_path": str(path),
                "scores_sha256": hashlib.sha256(
                    path.read_bytes()
                ).hexdigest(),
            }
        )
        return scores

    ModelRunner.run_inference = _observed_inference
    AlphaFold3Evaluator.evaluate = _observed_evaluate

    _cut_enabled = (
        os.environ.get("ASTROLABE_POSTHOC_PARALLEL_CUT") == "1"
    )
    if _cut_enabled:
        _cut_root = Path(
            os.environ["ASTROLABE_POSTHOC_PARALLEL_CUT_ROOT"]
        ).resolve()
        _cut_ready_path = Path(
            os.environ["ASTROLABE_POSTHOC_PARALLEL_CUT_READY"]
        ).resolve()
        from astrolabe.posthoc import refold as _posthoc_refold

        _original_fixed_task_list = (
            _posthoc_refold.run_fixed_task_list
        )

        def _completed_receipts():
            records = []
            for path in sorted(_cut_root.rglob("receipt.json")):
                relative = path.relative_to(_cut_root).as_posix()
                if "/.framework/refold/af3/step_" not in relative:
                    continue
                try:
                    value = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError):
                    continue
                if value.get("status") != "completed":
                    continue
                records.append(
                    {
                        "path": str(path),
                        "sha256": hashlib.sha256(
                            path.read_bytes()
                        ).hexdigest(),
                        "identity_sha256": value.get("identity_sha256"),
                        "output_contract_fingerprint": (
                            value.get("outputs", {}).get("fingerprint")
                        ),
                        "step": value.get("step"),
                        "oligo": value.get("oligo"),
                    }
                )
            return records

        def _process_start_identity(pid):
            line = Path(f"/proc/{pid}/stat").read_text(
                encoding="utf-8"
            ).strip()
            closing = line.rfind(")")
            remainder = line[closing + 2 :].split()
            return {
                "starttime_ticks": int(remainder[19]),
                "boot_id": Path(
                    "/proc/sys/kernel/random/boot_id"
                ).read_text(encoding="utf-8").strip(),
            }

        def _observed_fixed_task_list(
            tasks,
            worker,
            *,
            max_in_flight=1,
            worker_execution="threaded",
            logger=None,
            admission=None,
            terminal_observed=None,
            collection_commit=None,
        ):
            fixed_run = _original_fixed_task_list(
                tasks,
                worker,
                max_in_flight=max_in_flight,
                worker_execution=worker_execution,
                logger=logger,
                admission=admission,
                terminal_observed=terminal_observed,
                collection_commit=collection_commit,
            )
            if (
                getattr(worker, "__name__", None) != "_run_shard"
                or int(max_in_flight) <= 1
            ):
                return fixed_run
            ordered_jobs = [
                result.job_id
                for result in fixed_run.results
            ]
            if (
                len(ordered_jobs) != 2
                or not all(
                    job_id.startswith("shard_")
                    for job_id in ordered_jobs
                )
            ):
                return fixed_run
            receipts = _completed_receipts()
            worker_summaries = []
            for path in sorted(
                _cut_root.glob(
                    "_parallel_workers/run_*/worker_*.summary.json"
                )
            ):
                worker_summaries.append(
                    {
                        "path": str(path),
                        "sha256": hashlib.sha256(
                            path.read_bytes()
                        ).hexdigest(),
                    }
                )
            if (
                len(receipts) != len(ordered_jobs)
                or len(worker_summaries) != len(ordered_jobs)
            ):
                return fixed_run
            proof = json.dumps(
                {
                    "ordered_jobs": ordered_jobs,
                    "receipts": receipts,
                    "worker_summaries": worker_summaries,
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
            cut_evidence_id = (
                "fixedlist-" + hashlib.sha256(proof).hexdigest()
            )
            _write_json_atomic(
                _cut_ready_path,
                {
                    "schema": (
                        "astrolabe.posthoc-refold."
                        "parallel-cut-ready/v2"
                    ),
                    "pid": os.getpid(),
                    "parent_pid": os.getpid(),
                    "parent_ppid": os.getppid(),
                    "process_start": _process_start_identity(
                        os.getpid()
                    ),
                    "action_id": cut_evidence_id,
                    "ordered_jobs": sorted(ordered_jobs),
                    "successful_child_waits": len(ordered_jobs),
                    "outer_worker_name": worker.__name__,
                    "max_in_flight": int(max_in_flight),
                    "receipts": receipts,
                    "worker_summaries": worker_summaries,
                    "boundary": (
                        "refold_run_fixed_task_list_after_child_wait_"
                        "and_receipt_commit_before_result_consume"
                    ),
                    "resume_contract": (
                        "per_slot_committed_receipts_with_no_outer_journal"
                    ),
                    "stage": _stage,
                    "time_ns": time.time_ns(),
                },
            )
            while True:
                os.kill(os.getpid(), signal.SIGSTOP)

        _posthoc_refold.run_fixed_task_list = (
            _observed_fixed_task_list
        )
'''


def _install_capture(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "sitecustomize.py"
    path.write_text(_SITECUSTOMIZE, encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass


def _capture_environment(
    *,
    repo_root: Path,
    capture_dir: Path,
    stage: str,
    max_requests: int,
) -> dict[str, str]:
    environment = os.environ.copy()
    python_path = [
        str(capture_dir),
        str(repo_root),
    ]
    existing = environment.get("PYTHONPATH")
    if existing:
        python_path.append(existing)
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(python_path),
            "PYTHONHASHSEED": "0",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "ASTROLABE_POSTHOC_ACCEPTANCE_CAPTURE": "1",
            "ASTROLABE_POSTHOC_ACCEPTANCE_CAPTURE_DIR": str(capture_dir),
            "ASTROLABE_POSTHOC_ACCEPTANCE_STAGE": stage,
            "ASTROLABE_POSTHOC_ACCEPTANCE_MAX_REQUESTS": str(max_requests),
        }
    )
    return environment


def _run_process(
    command: Sequence[str],
    *,
    environment: Mapping[str, str],
    stdout_path: Path,
    stderr_path: Path,
    timeout_s: int,
) -> dict[str, Any]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            env=dict(environment),
            timeout=int(timeout_s),
        )
    except subprocess.TimeoutExpired as error:
        stdout_path.write_text(error.stdout or "", encoding="utf-8")
        stderr_path.write_text(error.stderr or "", encoding="utf-8")
        raise AcceptanceError(
            f"process timed out after {timeout_s}s: {list(command)}"
        ) from error
    duration = time.monotonic() - started
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    record = {
        "command": list(command),
        "returncode": completed.returncode,
        "duration_s": duration,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }
    if completed.returncode != 0:
        raise AcceptanceError(
            f"process failed rc={completed.returncode}: {list(command)}; "
            f"stderr={str(stderr_path)}"
        )
    return record


def _parse_json_stdout(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8").strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise AcceptanceError(f"child stdout is not one JSON document: {path}") from error
    if not isinstance(value, dict):
        raise AcceptanceError(f"child stdout JSON is not an object: {path}")
    return value


def _request_evidence(capture_dir: Path) -> dict[str, Any]:
    events = _read_jsonl(capture_dir / "events.jsonl")
    rpc_delivered = [
        event for event in events if event.get("event") == "model_rpc_delivered"
    ]
    reservations = [
        event for event in events if event.get("event") == "model_request_reserved"
    ]
    started = [
        event
        for event in events
        if event.get("event") == "model_inner_invocation_started"
    ]
    completed = [
        event
        for event in events
        if event.get("event") == "model_inner_invocation_completed"
    ]
    inner_failed = [
        event
        for event in events
        if event.get("event") == "model_inner_invocation_failed"
    ]
    entrypoint_failed = [
        event for event in events if event.get("event") == "model_entrypoint_failed"
    ]
    raw_capture_failed = [
        event for event in events if event.get("event") == "model_raw_capture_failed"
    ]
    budget_exceeded = [
        event for event in events if event.get("event") == "request_budget_exceeded"
    ]
    completed_entrypoints = [
        event for event in events if event.get("event") == "model_entrypoint_completed"
    ]
    failed = [*inner_failed, *entrypoint_failed, *raw_capture_failed, *budget_exceeded]
    rpc_without_inner = []
    started_delivery_ids = {
        int(event["delivery_index"])
        for event in started
        if event.get("delivery_index") is not None
    }
    for event in rpc_delivered:
        delivery_index = event.get("delivery_index")
        if delivery_index is not None and int(delivery_index) not in started_delivery_ids:
            rpc_without_inner.append(event)
    evaluator_started = [
        event for event in events if event.get("event") == "evaluator_started"
    ]
    evaluator_completed = [
        event for event in events if event.get("event") == "evaluator_completed"
    ]
    return {
        "events_path": str(capture_dir / "events.jsonl"),
        "rpc_delivered": len(rpc_delivered),
        "request_reservations": len(reservations),
        "inner_run_inference_invocations": len(started),
        "inner_run_inference_completed": len(completed),
        "inner_run_inference_failed": len(inner_failed),
        "entrypoint_completed": len(completed_entrypoints),
        "rpc_delivered_without_inner": len(rpc_without_inner),
        "observed_model_requests": len(started),
        "completed_model_requests": len(completed),
        "failed_or_denied_model_requests": len(failed),
        "evaluator_invocations": len(evaluator_started),
        "completed_evaluator_invocations": len(evaluator_completed),
        "rpc_delivery_events": rpc_delivered,
        "reservation_events": reservations,
        "inner_invocation_events": started,
        "completed_entrypoint_events": completed_entrypoints,
        "requests": completed,
        "failures": failed,
        "all_events_sha256": (
            _sha256(capture_dir / "events.jsonl")
            if (capture_dir / "events.jsonl").is_file()
            else None
        ),
    }


def _expected_run_dir(caller_root: Path, item: Mapping[str, Any]) -> Path:
    task = re.sub(r"[^A-Za-z0-9_.+-]", "_", str(item["task"])).strip("_")
    run = re.sub(r"[^A-Za-z0-9_.+-]", "_", str(item["run"])).strip("_")
    return caller_root / task / run


def _model_done_marker(run_dir: Path, item: Mapping[str, Any]) -> Path:
    step = int(item["step"])
    return (
        run_dir
        / "raw_model_results"
        / "alphafold3"
        / f"step_{step:03d}_{OLIGO}"
        / f"step_{step:03d}_{OLIGO}_summary_confidences.json"
    )


def _model_structure(run_dir: Path, item: Mapping[str, Any]) -> Path:
    step = int(item["step"])
    return run_dir / "structures" / f"step_{step:03d}_{OLIGO}_model.cif"


def _refold_receipt_evidence(
    run_dir: Path,
    item: Mapping[str, Any],
) -> dict[str, Any]:
    receipt_path = (
        Path(run_dir)
        / ".framework"
        / "refold"
        / "af3"
        / f"step_{int(item['step']):03d}_{str(item['oligo'])}"
        / "receipt.json"
    )
    if not receipt_path.is_file():
        raise AcceptanceError(
            f"refold completion has no committed receipt: {receipt_path}"
        )
    receipt = _read_json(receipt_path)
    if receipt.get("status") != "completed":
        raise AcceptanceError(
            f"refold receipt is not completed: {receipt_path}"
        )
    if (
        receipt.get("backend") != "af3"
        or int(receipt.get("step", -1)) != int(item["step"])
        or str(receipt.get("oligo")) != str(item["oligo"])
    ):
        raise AcceptanceError(
            f"refold receipt belongs to another slot: {receipt_path}"
        )
    identity_sha256 = receipt.get("identity_sha256")
    if (
        not isinstance(identity_sha256, str)
        or len(identity_sha256) != 64
        or any(character not in "0123456789abcdef" for character in identity_sha256)
    ):
        raise AcceptanceError(
            f"refold receipt has an invalid identity hash: {receipt_path}"
        )
    outputs = receipt.get("outputs")
    if (
        not isinstance(outputs, Mapping)
        or outputs.get("schema") != "astrolabe.posthoc-refold.outputs/v1"
        or not isinstance(outputs.get("files"), list)
        or not outputs["files"]
    ):
        raise AcceptanceError(
            f"refold receipt has no output contract: {receipt_path}"
        )
    return {
        "path": str(receipt_path),
        "sha256": _sha256(receipt_path),
        "status": "completed",
        "identity_sha256": identity_sha256,
        "output_contract_fingerprint": outputs.get("fingerprint"),
        "output_file_count": len(outputs["files"]),
    }


def _tree_snapshot(root: Path) -> dict[str, Any]:
    files = {}
    if root.exists():
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            files[str(path.relative_to(root))] = {
                "sha256": _sha256(path),
                "size": path.stat().st_size,
            }
    return {
        "root": str(root),
        "files": files,
        "fingerprint": _hash_value(files),
    }


def _tree_changes(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> list[dict[str, Any]]:
    left = before.get("files", {})
    right = after.get("files", {})
    changes = []
    for key in sorted(set(left) | set(right)):
        if left.get(key) != right.get(key):
            changes.append(
                {
                    "path": key,
                    "before": left.get(key),
                    "after": right.get(key),
                }
            )
    return changes


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise AcceptanceError(f"array evidence is absent: {path}")
    return _shared_validation()._load_tree_arrays(path)


def compare_npz(left: Path, right: Path) -> dict[str, Any]:
    result = _shared_validation()._compare_arrays(left, right)
    if result["array_count"] == 0:
        result["equal"] = False
        result["failures"].append(
            {
                "key": None,
                "reason": "empty-array-archive",
            }
        )
    result.update({"left": str(left), "right": str(right)})
    return result


def _scores_evidence(pickle_path: Path, root: Path) -> tuple[Path, Path]:
    import pickle

    from astrolabe.evaluators.utils import StandardScores

    shared = _shared_validation()
    with pickle_path.open("rb") as handle:
        scores = pickle.load(handle)
    if type(scores) is not StandardScores:
        raise AcceptanceError(
            f"expected StandardScores, got {type(scores).__qualname__}"
        )
    arrays, metadata = shared._score_snapshot(scores)
    npz_path = root / "standard_scores.npz"
    json_path = root / "standard_scores.json"
    shared._write_npz(npz_path, arrays)
    _atomic_json(json_path, metadata)
    return npz_path, json_path


def structure_atoms(path: Path) -> tuple[np.ndarray, dict[str, list[str]]]:
    return _shared_validation()._structure_atoms(path)


def _coordinate_evidence(
    *,
    name: str,
    structure: Path,
    root: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    if not structure.is_file():
        raise AcceptanceError(f"structure is absent: {structure}")
    coordinates, identity = structure_atoms(structure)
    copied = root / "structures" / f"{name}.cif"
    copied.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(structure, copied)
    npz_path = root / f"{name}_coordinates.npz"
    json_path = root / f"{name}_coordinates.json"
    _shared_validation()._write_npz(npz_path, {"coordinates": coordinates})
    metadata = {
        "name": name,
        "source_path": str(structure),
        "snapshot_path": str(copied),
        "source_sha256": _sha256(structure),
        "coordinate_dtype": coordinates.dtype.str,
        "coordinate_shape": list(coordinates.shape),
        "atom_identity": identity,
    }
    _atomic_json(json_path, metadata)
    return npz_path, json_path, metadata


def _baseline_worker(
    *,
    manifest_path: Path,
    config_path: Path,
    out_root: Path,
) -> int:
    from omegaconf import OmegaConf

    from astrolabe.core import global_config
    from astrolabe.evaluators.utils import construct_evaluators
    from astrolabe.posthoc import refold
    from astrolabe.utils.base import construct_standard_input

    item = _manifest_record(manifest_path)
    cfg = refold._load_cfg_for_refold(config_path)
    input_context = refold._load_refold_input_context(cfg, item=item)
    runtime_overrides = refold._load_refold_runtime_overrides(cfg)
    homologies = refold._resolve_homology_value(
        v=item.get("homologies") or {},
        relpath_base=None,
        path_maps=(),
        require_abs=True,
    )
    refold._init_refold_global_config(
        out_dir=out_root,
        **runtime_overrides,
        **input_context,
    )
    refold._build_protomers(
        seqs=item["seqs"],
        types=item["types"],
        homologies=homologies,
    )
    global_config.set_variable("step", int(item["step"]))
    inputs = construct_standard_input([str(item["oligo"])], seed=int(cfg.seed))
    evaluator_conf = OmegaConf.create(
        {
            "evaluator_terms": {"Alphafold3": {"_type": "Alphafold3"}},
            "alphafold3": OmegaConf.to_container(
                cfg.evaluator.alphafold3,
                resolve=True,
            ),
        }
    )
    evals = construct_evaluators(
        evaluator_conf,
        out_root / "raw_model_results",
        oligo_names=[str(item["oligo"])],
    )
    try:
        scores = evals["Alphafold3"].evaluate(inputs, runtime_config={})
    finally:
        evals["Alphafold3"].shutdown()

    import jax
    import pickle

    runtime = {
        "python": sys.version,
        "jax_version": jax.__version__,
        "devices": [
            {
                "platform": str(device.platform),
                "kind": str(getattr(device, "device_kind", "")),
                "id": int(getattr(device, "id", -1)),
            }
            for device in jax.devices()
        ],
    }
    result = {
        "schema": "astrolabe.posthoc-refold-gpu-acceptance.baseline/v1",
        "status": "completed",
        "source": "direct AlphaFold3Evaluator.evaluate -> ModelRunner.run_inference",
        "manifest": str(manifest_path),
        "config": str(config_path),
        "out_root": str(out_root),
        "seed": int(cfg.seed),
        "num_seeds": int(cfg.evaluator.alphafold3.num_seeds),
        "num_samples": int(cfg.evaluator.alphafold3.num_samples),
        "num_recycles": int(cfg.evaluator.alphafold3.num_recycles),
        "structure_paths": {
            str(key): str(value) for key, value in scores.complex_paths.items()
        },
        "runtime": runtime,
    }
    result_path = out_root / "result.json"
    scores_path = out_root / "scores.pkl"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with scores_path.open("wb") as handle:
        pickle.dump(scores, handle, protocol=5)
    _atomic_json(result_path, result)
    return 0


def _update_state(out_root: Path, **values: Any) -> dict[str, Any]:
    path = out_root / "run_state.json"
    state = _read_json(path) if path.exists() else {
        "schema": "astrolabe.posthoc-refold-gpu-acceptance.state/v1",
        "phases": {},
    }
    state.update(values)
    state["updated_unix_s"] = time.time()
    _atomic_json(path, state)
    return state


def _stage_complete_from_evidence(
    *,
    stage: str,
    out_root: Path,
    capture_dir: Path,
    caller_root: Path | None = None,
    item: Mapping[str, Any] | None = None,
) -> bool:
    evidence = _request_evidence(capture_dir)
    expected = 0 if stage == "cache_resume" else 1
    if (
        evidence["observed_model_requests"] != expected
        or evidence["completed_model_requests"] != expected
        or evidence["failed_or_denied_model_requests"] != 0
    ):
        return False
    if stage != "cache_resume":
        if (
            evidence["rpc_delivered"] != 1
            or evidence["request_reservations"] != 1
            or evidence["inner_run_inference_invocations"] != 1
            or evidence["completed_evaluator_invocations"] != 1
        ):
            return False
    elif (
        evidence["rpc_delivered"] != 0
        or evidence["request_reservations"] != 0
        or evidence["inner_run_inference_invocations"] != 0
        or evidence["evaluator_invocations"] != 0
    ):
        return False
    if caller_root is not None and item is not None:
        run_dir = _expected_run_dir(caller_root, item)
        if not (
            _model_done_marker(run_dir, item).is_file()
            and _model_structure(run_dir, item).is_file()
        ):
            return False
        try:
            _refold_receipt_evidence(run_dir, item)
        except AcceptanceError:
            return False
        return True
    if stage == "native_baseline":
        return (out_root / "baseline" / "result.json").is_file()
    return True


def _get_stage_result(
    state: Mapping[str, Any],
    stage: str,
) -> dict[str, Any] | None:
    phase = state.get("phases", {}).get(stage, {})
    if phase.get("status") != "completed":
        return None
    result = phase.get("result")
    return dict(result) if isinstance(result, Mapping) else None


def _gpu_manifest() -> dict[str, Any]:
    return _shared_validation()._nvidia_manifest()


def _runtime_package_directories(module_name: str) -> tuple[Path, ...]:
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ModuleNotFoundError, ValueError) as exc:
        raise AcceptanceError(
            f"runtime package cannot be resolved: {module_name}"
        ) from exc
    locations = tuple(
        Path(item).expanduser().resolve()
        for item in (spec.submodule_search_locations or ())
    ) if spec is not None else ()
    if not locations or any(not location.is_dir() for location in locations):
        raise AcceptanceError(
            f"runtime package has no directory origin: {module_name}"
        )
    return locations


def _runtime_library_record(
    *,
    module_name: str,
    prefix: str,
) -> dict[str, Any]:
    directories = _runtime_package_directories(module_name)
    candidates = sorted(
        {
            path.resolve()
            for directory in directories
            for path in directory.rglob(f"{prefix}*")
            if path.is_file()
        }
    )
    if not candidates:
        raise AcceptanceError(
            f"runtime wheel contains no {prefix} library under {directories}"
        )
    failures = []
    for candidate in candidates:
        try:
            ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
        except OSError as exc:
            failures.append({"path": str(candidate), "error": str(exc)})
            continue
        return {
            "module": module_name,
            "directories": [str(directory) for directory in directories],
            "loaded_path": str(candidate),
            "load_method": "ctypes.CDLL",
            "failed_candidates": failures,
        }
    raise AcceptanceError(
        f"no {prefix} runtime wheel library could be loaded: {failures}"
    )


def _runtime_preflight(prepared: Mapping[str, Any]) -> dict[str, Any]:
    marker_name = None
    marker_value = None
    for candidate in ("APPTAINER_CONTAINER", "SINGULARITY_CONTAINER"):
        value = os.environ.get(candidate)
        if value:
            marker_name = candidate
            marker_value = value
            break
    if marker_name is None or marker_value is None:
        raise AcceptanceError(
            "posthoc model execution must run inside the prepared Apptainer "
            "container; neither APPTAINER_CONTAINER nor SINGULARITY_CONTAINER "
            "is set"
        )

    sif_path = str(prepared.get("sif") or "")
    sif_sha256 = prepared.get("sif_sha256")
    provenance = prepared.get("sif_provenance")
    if (
        not sif_path
        or not isinstance(sif_sha256, str)
        or len(sif_sha256) != 64
        or not isinstance(provenance, Mapping)
        or provenance.get("sha256") != sif_sha256
        or provenance.get("path") != sif_path
    ):
        raise AcceptanceError(
            "container preflight requires host-provided SIF path and SHA256 provenance"
        )
    expected_python = Path(str(prepared.get("sif_python") or "")).resolve()
    actual_python = Path(sys.executable).resolve()
    if actual_python != expected_python:
        raise AcceptanceError(
            f"runtime Python differs from prepared identity: {actual_python} != "
            f"{expected_python}"
        )

    af3_directories = _runtime_package_directories("alphafold3")
    model_modules_loaded = sorted(
        name
        for name in (
            "astrolabe.evaluators.alphafold3",
            "astrolabe.evaluators.wrappers.alphafold3",
        )
        if name in sys.modules
    )
    if model_modules_loaded:
        raise AcceptanceError(
            "runtime preflight loaded model execution modules: "
            + ", ".join(model_modules_loaded)
        )
    nvrtc = _runtime_library_record(
        module_name="nvidia.cuda_nvrtc",
        prefix="libnvrtc.so",
    )
    cudnn = _runtime_library_record(
        module_name="nvidia.cudnn",
        prefix="libcudnn.so",
    )
    return {
        "schema": RUNTIME_PREFLIGHT_SCHEMA,
        "status": "PASS",
        "container": {
            "marker_name": marker_name,
            "marker_value": marker_value,
            "reported_host_path_exists_in_container": Path(marker_value).is_file(),
            "apptainer_container": os.environ.get("APPTAINER_CONTAINER"),
            "singularity_container": os.environ.get("SINGULARITY_CONTAINER"),
        },
        "python": {
            "executable": str(actual_python),
            "version": sys.version,
        },
        "runtime_source_origin": {
            "module": "alphafold3",
            "directories": [str(path) for path in af3_directories],
        },
        "runtime_libraries": {
            "nvrtc": nvrtc,
            "cudnn": cudnn,
            "find_library": {
                name: ctypes.util.find_library(name)
                for name in ("nvrtc", "cudnn")
            },
            "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
        },
        "model_execution_modules_loaded": model_modules_loaded,
        "model_weights_loaded": False,
        "sif": {
            "path": sif_path,
            "sha256": sif_sha256,
            "provenance": dict(provenance),
            "verified_from_host_provenance": True,
        },
    }


def _environment_manifest(
    prepared: Mapping[str, Any],
    *,
    runtime_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    model_file = Path(prepared["model_file"])
    if not model_file.is_file():
        raise AcceptanceError(f"AF3 model file is absent: {model_file}")
    packages = {}
    for name in ("jax", "jaxlib", "numpy", "omegaconf", "jsonschema"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "gpu": _gpu_manifest(),
        "model": {
            "path": str(model_file),
            "size": model_file.stat().st_size,
            "sha256": _sha256(model_file),
        },
        "sif": runtime_preflight["sif"],
        "sif_python": prepared.get("sif_python"),
        "runtime_preflight": dict(runtime_preflight),
    }


def _write_failure_receipt(
    out_root: Path,
    *,
    error: BaseException,
) -> None:
    out_root = Path(out_root).expanduser().resolve()
    request_receipts = {}
    for stage in ("caller", "native_baseline", "cache_resume"):
        capture_dir = out_root / "evidence" / "requests" / stage
        if capture_dir.exists():
            request_receipts[stage] = _request_evidence(capture_dir)
    prepared = (
        _read_json(out_root / "prepared.json")
        if (out_root / "prepared.json").is_file()
        else None
    )
    failure = {
        "schema": SUMMARY_SCHEMA,
        "acceptance_schema": SCHEMA,
        "status": "FAIL",
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
        "prepared": prepared,
        "request_receipts": request_receipts,
        "observed_requests": {
            stage: receipt["observed_model_requests"]
            for stage, receipt in request_receipts.items()
        },
        "note": (
            "No failed GPU stage is automatically retried. Inspect the durable "
            "capture receipts before choosing any recovery action."
        ),
    }
    _atomic_json(out_root / "failure.json", failure)
    try:
        _update_state(
            out_root,
            status="failed",
            failure_type=type(error).__name__,
            failure=str(error),
        )
    except OSError:
        pass


def _invoke_caller(
    *,
    out_root: Path,
    prepared: Mapping[str, Any],
    item: Mapping[str, Any],
    timeout_s: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    capture_dir = out_root / "evidence" / "requests" / "caller"
    _install_capture(capture_dir)
    environment = _capture_environment(
        repo_root=Path(prepared["repo_root"]),
        capture_dir=capture_dir,
        stage="public_caller",
        max_requests=1,
    )
    process = _run_process(
        prepared["commands"]["public_caller"],
        environment=environment,
        stdout_path=out_root / "logs" / "caller.stdout.json",
        stderr_path=out_root / "logs" / "caller.stderr.log",
        timeout_s=timeout_s,
    )
    summary = _parse_json_stdout(out_root / "logs" / "caller.stdout.json")
    requests = _request_evidence(capture_dir)
    if requests["observed_model_requests"] != 1:
        raise AcceptanceError(
            "public caller did not make exactly one observed AF3 request: "
            f"{requests['observed_model_requests']}"
        )
    if requests["rpc_delivered"] != 1:
        raise AcceptanceError(
            "public caller did not receive exactly one AF3 wrapper delivery: "
            f"{requests['rpc_delivered']}"
        )
    if requests["request_reservations"] != 1:
        raise AcceptanceError(
            "public caller did not reserve exactly one pre-call request: "
            f"{requests['request_reservations']}"
        )
    if requests["completed_model_requests"] != 1:
        raise AcceptanceError("public caller AF3 request did not complete")
    if requests["entrypoint_completed"] != 1:
        raise AcceptanceError("public caller capture entrypoint did not complete")
    if requests["failed_or_denied_model_requests"] != 0:
        raise AcceptanceError("public caller had failed or denied AF3 requests")
    run_dir = _expected_run_dir(out_root / "caller" / "runs", item)
    if not _model_done_marker(run_dir, item).is_file():
        raise AcceptanceError("public caller did not produce the AF3 completion marker")
    if not _model_structure(run_dir, item).is_file():
        raise AcceptanceError("public caller did not produce the canonical CIF")
    refold_receipt = _refold_receipt_evidence(run_dir, item)
    return {
        "process": process,
        "summary": summary,
        "requests": requests,
        "capture_dir": str(capture_dir),
        "run_dir": str(run_dir),
        "refold_receipt": refold_receipt,
    }, requests


def _invoke_baseline(
    *,
    out_root: Path,
    prepared: Mapping[str, Any],
    timeout_s: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    capture_dir = out_root / "evidence" / "requests" / "native_baseline"
    _install_capture(capture_dir)
    environment = _capture_environment(
        repo_root=Path(prepared["repo_root"]),
        capture_dir=capture_dir,
        stage="native_baseline",
        max_requests=1,
    )
    process = _run_process(
        prepared["commands"]["native_baseline"],
        environment=environment,
        stdout_path=out_root / "logs" / "baseline.stdout.log",
        stderr_path=out_root / "logs" / "baseline.stderr.log",
        timeout_s=timeout_s,
    )
    requests = _request_evidence(capture_dir)
    if requests["observed_model_requests"] != 1:
        raise AcceptanceError(
            "native baseline did not make exactly one observed AF3 request: "
            f"{requests['observed_model_requests']}"
        )
    if requests["rpc_delivered"] != 1:
        raise AcceptanceError(
            "native baseline did not receive exactly one AF3 wrapper delivery: "
            f"{requests['rpc_delivered']}"
        )
    if requests["request_reservations"] != 1:
        raise AcceptanceError(
            "native baseline did not reserve exactly one pre-call request: "
            f"{requests['request_reservations']}"
        )
    if requests["completed_model_requests"] != 1:
        raise AcceptanceError("native baseline AF3 request did not complete")
    if requests["entrypoint_completed"] != 1:
        raise AcceptanceError("native baseline capture entrypoint did not complete")
    if requests["failed_or_denied_model_requests"] != 0:
        raise AcceptanceError("native baseline had failed or denied AF3 requests")
    result_path = out_root / "baseline" / "result.json"
    if not result_path.is_file():
        raise AcceptanceError("native baseline did not produce result.json")
    return {
        "process": process,
        "result": _read_json(result_path),
        "requests": requests,
        "capture_dir": str(capture_dir),
    }, requests


def _invoke_cache(
    *,
    out_root: Path,
    prepared: Mapping[str, Any],
    timeout_s: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    capture_dir = out_root / "evidence" / "requests" / "cache_resume"
    if capture_dir.exists() and any(capture_dir.iterdir()):
        raise AcceptanceError(
            f"cache request evidence already exists; refusing to replay: {capture_dir}"
        )
    _install_capture(capture_dir)
    environment = _capture_environment(
        repo_root=Path(prepared["repo_root"]),
        capture_dir=capture_dir,
        stage="cache_resume",
        max_requests=0,
    )
    process = _run_process(
        prepared["commands"]["public_caller"],
        environment=environment,
        stdout_path=out_root / "logs" / "cache.stdout.json",
        stderr_path=out_root / "logs" / "cache.stderr.log",
        timeout_s=timeout_s,
    )
    summary = _parse_json_stdout(out_root / "logs" / "cache.stdout.json")
    requests = _request_evidence(capture_dir)
    if requests["observed_model_requests"] != 0:
        raise AcceptanceError(
            "cache/resume made an AF3 request: "
            f"{requests['observed_model_requests']}"
        )
    if requests["rpc_delivered"] != 0:
        raise AcceptanceError(
            "cache/resume delivered an AF3 wrapper call: "
            f"{requests['rpc_delivered']}"
        )
    if requests["request_reservations"] != 0:
        raise AcceptanceError(
            "cache/resume reserved an AF3 request: "
            f"{requests['request_reservations']}"
        )
    if requests["failed_or_denied_model_requests"] != 0:
        raise AcceptanceError(
            "cache/resume attempted a denied AF3 request; cache invariant failed"
        )
    return {
        "process": process,
        "summary": summary,
        "requests": requests,
        "capture_dir": str(capture_dir),
    }, requests


def _scientific_check(
    *,
    out_root: Path,
    prepared: Mapping[str, Any],
    item: Mapping[str, Any],
    caller: Mapping[str, Any],
    baseline: Mapping[str, Any],
    caller_requests: Mapping[str, Any],
    baseline_requests: Mapping[str, Any],
    cache_requests: Mapping[str, Any],
    caller_before_cache: Mapping[str, Any],
    caller_after_cache: Mapping[str, Any],
) -> dict[str, Any]:
    caller_entrypoints = caller_requests.get("completed_entrypoint_events", [])
    baseline_entrypoints = baseline_requests.get(
        "completed_entrypoint_events", []
    )
    if len(caller_entrypoints) != 1 or len(baseline_entrypoints) != 1:
        raise AcceptanceError(
            "scientific comparison requires one completed entrypoint event "
            "for both caller and baseline"
        )
    caller_request = caller_entrypoints[0]
    baseline_request = baseline_entrypoints[0]
    input_compare = {
        "equal": (
            caller_request.get("input_digest")
            == baseline_request.get("input_digest")
        ),
        "caller": caller_request.get("input_digest"),
        "baseline": baseline_request.get("input_digest"),
    }
    rng_compare = compare_npz(
        Path(caller_request["rng_key_path"]),
        Path(baseline_request["rng_key_path"]),
    )
    raw_result_compare = compare_npz(
        Path(caller_request["result_path"]),
        Path(baseline_request["result_path"]),
    )

    caller_scores_event = next(
        (
            event
            for event in _read_jsonl(
                Path(caller["capture_dir"]) / "events.jsonl"
            )
            if event.get("event") == "evaluator_completed"
        ),
        None,
    )
    baseline_scores_event = next(
        (
            event
            for event in _read_jsonl(
                Path(baseline["capture_dir"]) / "events.jsonl"
            )
            if event.get("event") == "evaluator_completed"
        ),
        None,
    )
    if caller_scores_event is None or baseline_scores_event is None:
        raise AcceptanceError("StandardScores evidence is missing")

    evidence = out_root / "evidence"
    caller_score_npz, _caller_score_json = _scores_evidence(
        Path(caller_scores_event["scores_path"]),
        evidence / "caller_scores",
    )
    baseline_score_npz, _baseline_score_json = _scores_evidence(
        Path(baseline_scores_event["scores_path"]),
        evidence / "baseline_scores",
    )
    scores_compare = compare_npz(caller_score_npz, baseline_score_npz)

    caller_run_dir = Path(caller["run_dir"])
    caller_structure = _model_structure(caller_run_dir, item)
    baseline_structure = Path(
        baseline["result"]["structure_paths"][OLIGO]
    )
    caller_coord_npz, _caller_coord_json, caller_coord = _coordinate_evidence(
        name="caller",
        structure=caller_structure,
        root=evidence / "coordinates",
    )
    baseline_coord_npz, _baseline_coord_json, baseline_coord = (
        _coordinate_evidence(
            name="baseline",
            structure=baseline_structure,
            root=evidence / "coordinates",
        )
    )
    coordinate_compare = compare_npz(caller_coord_npz, baseline_coord_npz)
    coordinate_identity = {
        key: caller_coord.get(key) for key in (
            "coordinate_dtype",
            "coordinate_shape",
            "atom_identity",
        )
    } == {
        key: baseline_coord.get(key) for key in (
            "coordinate_dtype",
            "coordinate_shape",
            "atom_identity",
        )
    }

    cache_changes = _tree_changes(caller_before_cache, caller_after_cache)
    first_summary = caller["summary"]
    cache_summary = out_root / "logs" / "cache.stdout.json"
    cache_summary_value = _parse_json_stdout(cache_summary)

    caller_config_arg = prepared["commands"]["public_caller"][
        prepared["commands"]["public_caller"].index("--config-yaml") + 1
    ]
    baseline_config_arg = prepared["commands"]["native_baseline"][
        prepared["commands"]["native_baseline"].index("--config") + 1
    ]
    same_config = (
        Path(caller_config_arg).resolve()
        == Path(baseline_config_arg).resolve()
        == Path(baseline["result"]["config"]).resolve()
        == (out_root / prepared["inputs"]["config"]).resolve()
    )
    caller_raw = _load_npz(Path(caller_request["result_path"]))
    baseline_raw = _load_npz(Path(baseline_request["result_path"]))
    identifier_key = "result____identifier__"
    same_weights = (
        identifier_key in caller_raw
        and identifier_key in baseline_raw
        and np.array_equal(
            caller_raw[identifier_key],
            baseline_raw[identifier_key],
            equal_nan=True,
        )
    )
    refold_receipt = caller.get("refold_receipt")
    committed_refold_receipt = (
        isinstance(refold_receipt, Mapping)
        and refold_receipt.get("status") == "completed"
        and isinstance(refold_receipt.get("identity_sha256"), str)
        and isinstance(refold_receipt.get("output_contract_fingerprint"), str)
        and int(refold_receipt.get("output_file_count", 0)) > 0
    )

    checks = {
        "caller_one_observed_request": (
            caller_requests["observed_model_requests"] == 1
        ),
        "caller_one_rpc_delivery": caller_requests["rpc_delivered"] == 1,
        "caller_one_request_reservation": (
            caller_requests["request_reservations"] == 1
        ),
        "baseline_one_observed_request": (
            baseline_requests["observed_model_requests"] == 1
        ),
        "baseline_one_rpc_delivery": baseline_requests["rpc_delivered"] == 1,
        "baseline_one_request_reservation": (
            baseline_requests["request_reservations"] == 1
        ),
        "cache_zero_observed_requests": (
            cache_requests["observed_model_requests"] == 0
        ),
        "cache_zero_rpc_deliveries": cache_requests["rpc_delivered"] == 0,
        "cache_zero_request_reservations": (
            cache_requests["request_reservations"] == 0
        ),
        "cache_no_evaluator_invocations": (
            cache_requests["evaluator_invocations"] == 0
        ),
        "total_real_gpu_requests_at_most_two": (
            caller_requests["observed_model_requests"]
            + baseline_requests["observed_model_requests"]
            + cache_requests["observed_model_requests"]
            <= MAX_REAL_GPU_REQUESTS
        ),
        "caller_ran_af3_once": int(first_summary.get("ran_af3", -1)) == 1,
        "caller_failed_zero": int(first_summary.get("failed", -1)) == 0,
        "cache_ran_af3_zero": (
            int(cache_summary_value.get("ran_af3", -1)) == 0
        ),
        "cache_skipped_one": (
            int(cache_summary_value.get("skipped", -1)) == 1
        ),
        "cache_receipt_hit_one": (
            int(cache_summary_value.get("receipt_cache_hits_af3", -1)) == 1
        ),
        "cache_failed_zero": int(cache_summary_value.get("failed", -1)) == 0,
        "caller_committed_refold_receipt": committed_refold_receipt,
        "exact_featurised_input_digest": input_compare["equal"],
        "exact_rng_key": rng_compare["equal"],
        "exact_raw_result_arrays_dtype_shape_values_equal_nan": (
            raw_result_compare["equal"]
        ),
        "exact_standard_score_arrays": scores_compare["equal"],
        "exact_atomic_coordinates": coordinate_compare["equal"],
        "exact_atom_identity": coordinate_identity,
        "cache_scientific_artifacts_unchanged": not cache_changes,
        "same_scientific_config": same_config,
        "same_model_identifier": same_weights,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    return {
        "status": status,
        "checks": checks,
        "featurised_input_digest": input_compare,
        "rng_key": rng_compare,
        "raw_model_result_arrays": raw_result_compare,
        "standard_score_arrays": scores_compare,
        "atomic_coordinates": coordinate_compare,
        "atom_identity_equal": coordinate_identity,
        "cache_artifact_fingerprint_before": caller_before_cache["fingerprint"],
        "cache_artifact_fingerprint_after": caller_after_cache["fingerprint"],
        "cache_artifact_changes": cache_changes,
        "same_scientific_config": same_config,
        "same_model_identifier": same_weights,
        "refold_receipt": refold_receipt,
        "caller_summary": first_summary,
        "cache_summary": cache_summary_value,
    }


def run_acceptance(
    out_root: Path,
    *,
    timeout_s: int = 3600,
) -> dict[str, Any]:
    """Run the two-request public caller/baseline pair and the zero-request cache."""

    out_root = out_root.expanduser().resolve()
    frozen = verify_frozen_inputs(out_root)
    prepared = frozen["prepared"]
    item = frozen["manifest_item"]
    runtime_preflight = _runtime_preflight(prepared)
    _atomic_json(out_root / "runtime_preflight.json", runtime_preflight)

    state = _update_state(out_root, status="running")
    environment = _environment_manifest(
        prepared,
        runtime_preflight=runtime_preflight,
    )
    _atomic_json(out_root / "environment.json", environment)

    caller = _get_stage_result(state, "public_caller")
    if caller is None:
        capture_dir = out_root / "evidence" / "requests" / "caller"
        if state.get("phases", {}).get("public_caller", {}).get("status") == "running":
            if _stage_complete_from_evidence(
                stage="public_caller",
                out_root=out_root,
                capture_dir=capture_dir,
                caller_root=out_root / "caller" / "runs",
                item=item,
            ):
                caller = {
                    "process": None,
                    "summary": _parse_json_stdout(
                        out_root / "logs" / "caller.stdout.json"
                    ),
                    "requests": _request_evidence(capture_dir),
                    "capture_dir": str(capture_dir),
                    "run_dir": str(
                        _expected_run_dir(out_root / "caller" / "runs", item)
                    ),
                    "recovered": True,
                }
                caller["refold_receipt"] = _refold_receipt_evidence(
                    Path(caller["run_dir"]),
                    item,
                )
            else:
                raise AcceptanceError(
                    "public caller was already started but has no complete "
                    "one-request receipt; refusing to replay it"
                )
        else:
            if (out_root / "caller").exists() and any(
                (out_root / "caller").iterdir()
            ):
                raise AcceptanceError(
                    f"caller output exists without a completed receipt: {out_root / 'caller'}"
                )
            _update_state(
                out_root,
                phases={
                    **state.get("phases", {}),
                    "public_caller": {
                        "status": "running",
                        "started_unix_s": time.time(),
                    },
                },
            )
            caller, _ = _invoke_caller(
                out_root=out_root,
                prepared=prepared,
                item=item,
                timeout_s=timeout_s,
            )
        state = _update_state(
            out_root,
            phases={
                **state.get("phases", {}),
                "public_caller": {
                    "status": "completed",
                    "completed_unix_s": time.time(),
                    "result": caller,
                },
            },
        )
    verify_frozen_inputs(out_root)

    baseline = _get_stage_result(state, "native_baseline")
    if baseline is None:
        phase = state.get("phases", {}).get("native_baseline", {})
        capture_dir = out_root / "evidence" / "requests" / "native_baseline"
        if phase.get("status") == "running":
            if _stage_complete_from_evidence(
                stage="native_baseline",
                out_root=out_root,
                capture_dir=capture_dir,
            ):
                baseline = {
                    "process": None,
                    "result": _read_json(out_root / "baseline" / "result.json"),
                    "requests": _request_evidence(capture_dir),
                    "capture_dir": str(capture_dir),
                    "recovered": True,
                }
            else:
                raise AcceptanceError(
                    "native baseline was already started but has no complete "
                    "one-request receipt; refusing to replay it"
                )
        else:
            if (out_root / "baseline").exists() and any(
                (out_root / "baseline").iterdir()
            ):
                raise AcceptanceError(
                    f"baseline output exists without a completed receipt: {out_root / 'baseline'}"
                )
            _update_state(
                out_root,
                phases={
                    **state.get("phases", {}),
                    "native_baseline": {
                        "status": "running",
                        "started_unix_s": time.time(),
                    },
                },
            )
            baseline, _ = _invoke_baseline(
                out_root=out_root,
                prepared=prepared,
                timeout_s=timeout_s,
            )
        state = _update_state(
            out_root,
            phases={
                **state.get("phases", {}),
                "native_baseline": {
                    "status": "completed",
                    "completed_unix_s": time.time(),
                    "result": baseline,
                },
            },
        )
    verify_frozen_inputs(out_root)

    caller_root = out_root / "caller" / "runs"
    before_cache = _tree_snapshot(caller_root)
    _atomic_json(out_root / "evidence" / "cache_before.json", before_cache)
    cache = _get_stage_result(state, "cache_resume")
    if cache is None:
        phase = state.get("phases", {}).get("cache_resume", {})
        capture_dir = out_root / "evidence" / "requests" / "cache_resume"
        if phase.get("status") == "running":
            if _stage_complete_from_evidence(
                stage="cache_resume",
                out_root=out_root,
                capture_dir=capture_dir,
            ):
                cache = {
                    "process": None,
                    "summary": _parse_json_stdout(
                        out_root / "logs" / "cache.stdout.json"
                    ),
                    "requests": _request_evidence(capture_dir),
                    "capture_dir": str(capture_dir),
                    "recovered": True,
                }
            else:
                raise AcceptanceError(
                    "cache/resume was already started without a zero-request "
                    "receipt; refusing to replay it"
                )
        else:
            _update_state(
                out_root,
                phases={
                    **state.get("phases", {}),
                    "cache_resume": {
                        "status": "running",
                        "started_unix_s": time.time(),
                    },
                },
            )
            cache, _ = _invoke_cache(
                out_root=out_root,
                prepared=prepared,
                timeout_s=timeout_s,
            )
        state = _update_state(
            out_root,
            phases={
                **state.get("phases", {}),
                "cache_resume": {
                    "status": "completed",
                    "completed_unix_s": time.time(),
                    "result": cache,
                },
            },
        )
    after_cache = _tree_snapshot(caller_root)
    _atomic_json(out_root / "evidence" / "cache_after.json", after_cache)
    final_freeze = verify_frozen_inputs(out_root)

    comparison = _scientific_check(
        out_root=out_root,
        prepared=prepared,
        item=item,
        caller=caller,
        baseline=baseline,
        caller_requests=caller["requests"],
        baseline_requests=baseline["requests"],
        cache_requests=cache["requests"],
        caller_before_cache=before_cache,
        caller_after_cache=after_cache,
    )
    comparison["source_hash_stable"] = (
        final_freeze["source"]["fingerprint"]
        == prepared["source_fingerprint"]
    )
    comparison["source_fingerprint"] = final_freeze["source"]["fingerprint"]
    if not comparison["source_hash_stable"]:
        comparison["status"] = "FAIL"

    request_counts = {
        "source": (
            "inner ModelRunner.run_inference started/completed events; "
            "RPC delivery and pre-call reservation are separate layers"
        ),
        "rpc_delivered": {
            "public_caller": caller["requests"]["rpc_delivered"],
            "native_baseline": baseline["requests"]["rpc_delivered"],
            "cache_resume": cache["requests"]["rpc_delivered"],
        },
        "request_reservations": {
            "public_caller": caller["requests"]["request_reservations"],
            "native_baseline": baseline["requests"]["request_reservations"],
            "cache_resume": cache["requests"]["request_reservations"],
        },
        "inner_run_inference_invocations": {
            "public_caller": caller["requests"]["inner_run_inference_invocations"],
            "native_baseline": baseline[
                "requests"
            ]["inner_run_inference_invocations"],
            "cache_resume": cache["requests"]["inner_run_inference_invocations"],
        },
        "inner_run_inference_completed": {
            "public_caller": caller["requests"]["inner_run_inference_completed"],
            "native_baseline": baseline[
                "requests"
            ]["inner_run_inference_completed"],
            "cache_resume": cache["requests"]["inner_run_inference_completed"],
        },
        "public_caller": caller["requests"]["observed_model_requests"],
        "native_baseline": baseline["requests"]["observed_model_requests"],
        "cache_resume": cache["requests"]["observed_model_requests"],
        "total": (
            caller["requests"]["observed_model_requests"]
            + baseline["requests"]["observed_model_requests"]
            + cache["requests"]["observed_model_requests"]
        ),
        "maximum": MAX_REAL_GPU_REQUESTS,
    }
    summary = {
        "schema": SUMMARY_SCHEMA,
        "acceptance_schema": SCHEMA,
        "status": comparison["status"],
        "kind": "independent-posthoc-refold-public-caller-acceptance",
        "fixture": prepared["fixture"],
        "scientific_settings": prepared["scientific_settings"],
        "public_caller": prepared["commands"]["public_caller"],
        "native_baseline": prepared["commands"]["native_baseline"],
        "observed_requests": request_counts,
        "request_receipts": {
            "caller": caller["requests"],
            "baseline": baseline["requests"],
            "cache": cache["requests"],
        },
        "comparison": comparison,
        "cache": {
            **posthoc_cache_assessment(),
            "prepared_artifact_fingerprint": before_cache["fingerprint"],
            "post_cache_artifact_fingerprint": after_cache["fingerprint"],
            "artifact_changes": comparison["cache_artifact_changes"],
        },
        "source": {
            "manifest": str(
                out_root / prepared["inputs"]["manifest"]
            ),
            "config": str(out_root / prepared["inputs"]["config"]),
            "manifest_sha256": prepared["input_hashes"]["manifest"],
            "config_sha256": prepared["input_hashes"]["config"],
            "runtime_fingerprint": final_freeze["source"]["fingerprint"],
            "model_sha256": environment["model"]["sha256"],
        },
        "environment": environment,
        "artifacts": {
            "prepared": str(out_root / "prepared.json"),
            "source_manifest": str(out_root / "source_manifest.json"),
            "runtime_preflight": str(out_root / "runtime_preflight.json"),
            "environment": str(out_root / "environment.json"),
            "caller_root": str(caller_root),
            "baseline_root": str(out_root / "baseline"),
            "evidence_root": str(out_root / "evidence"),
            "logs_root": str(out_root / "logs"),
        },
        "unresolved_gaps": [],
    }
    _atomic_json(out_root / "summary.json", summary)
    state = _update_state(
        out_root,
        status="completed" if summary["status"] == "PASS" else "failed",
        summary_status=summary["status"],
    )
    if summary["status"] != "PASS":
        raise AcceptanceError(
            f"acceptance checks failed; inspect {out_root / 'summary.json'}"
        )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bounded public posthoc-refold GPU acceptance."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="Freeze one manifest/config and emit source hashes without GPU use.",
    )
    prepare.add_argument("--out-root", required=True)
    prepare.add_argument("--repo-root", default=str(REPO_ROOT_DEFAULT))
    prepare.add_argument("--model-dir", default=str(MODEL_DIR_DEFAULT))
    prepare.add_argument("--sif-provenance")

    verify = subparsers.add_parser(
        "verify",
        help="Verify frozen source and input hashes without GPU use.",
    )
    verify.add_argument("--out-root", required=True)

    preflight = subparsers.add_parser(
        "preflight",
        help="Verify container runtime origin and libraries without model loading.",
    )
    preflight.add_argument("--out-root", required=True)

    run = subparsers.add_parser(
        "run",
        help="Run public caller, one direct baseline, then zero-request cache.",
    )
    run.add_argument("--out-root", required=True)
    run.add_argument("--timeout-s", type=int, default=3600)

    baseline = subparsers.add_parser(
        "_native-baseline",
        help=argparse.SUPPRESS,
    )
    baseline.add_argument("--manifest", required=True)
    baseline.add_argument("--config", required=True)
    baseline.add_argument("--out-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "prepare":
        prepared = prepare_acceptance(
            Path(arguments.out_root),
            repo_root=Path(arguments.repo_root),
            model_dir=Path(arguments.model_dir),
            sif_provenance=(
                Path(arguments.sif_provenance)
                if arguments.sif_provenance
                else None
            ),
        )
        print(json.dumps(prepared, indent=2, sort_keys=True))
        return 0
    if arguments.command == "preflight":
        frozen = verify_frozen_inputs(Path(arguments.out_root))
        runtime_preflight = _runtime_preflight(frozen["prepared"])
        _atomic_json(
            Path(arguments.out_root).expanduser().resolve()
            / "runtime_preflight.json",
            runtime_preflight,
        )
        print(json.dumps(runtime_preflight, indent=2, sort_keys=True))
        return 0
    if arguments.command == "verify":
        frozen = verify_frozen_inputs(Path(arguments.out_root))
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "source_fingerprint": frozen["source"]["fingerprint"],
                    "checked_inputs": frozen["checked_inputs"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if arguments.command == "run":
        try:
            summary = run_acceptance(
                Path(arguments.out_root),
                timeout_s=int(arguments.timeout_s),
            )
        except AcceptanceError as error:
            _write_failure_receipt(Path(arguments.out_root), error=error)
            print(f"posthoc acceptance failed: {error}", file=sys.stderr)
            return 1
        print(json.dumps({
            "status": summary["status"],
            "summary": str(
                Path(arguments.out_root).expanduser().resolve() / "summary.json"
            ),
            "observed_requests": summary["observed_requests"],
        }, indent=2, sort_keys=True))
        return 0
    if arguments.command == "_native-baseline":
        return _baseline_worker(
            manifest_path=Path(arguments.manifest),
            config_path=Path(arguments.config),
            out_root=Path(arguments.out_root),
        )
    raise AssertionError(arguments.command)


if __name__ == "__main__":
    raise SystemExit(main())
