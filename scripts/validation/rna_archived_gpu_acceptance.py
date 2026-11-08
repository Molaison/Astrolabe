#!/usr/bin/env python3
"""Prepare and run the archived bivalent-RNA GPU acceptance.

The acceptance frontdoor is the normal public configuration lifecycle.  This
module only freezes and observes that lifecycle; it does not implement search,
prediction, or measurement logic.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ARCHIVED_CONFIG = "paper/bivalent_rna_archived"
ARCHIVED_SEQUENCE = "UCGGCAGCUGCCAUCUGACU"
ARCHIVED_SEED = 67101
MAX_EVALUATIONS = 2
MAX_NAMPNN_REQUESTS_PER_ARM = 1
MAX_NAMPNN_REQUESTS = 2
CUT_ROUND = "native-refinement/evaluate/1/prediction"
CUT_EXIT_CODE = 97
PRE_MODEL_GUARD_EXIT_CODE = 86
AF3_SITE_PACKAGE = "/alphafold3_venv/lib/python3.11/site-packages/alphafold3"
NA_MPNN_CHECKPOINT = "/ext/na_mpnn/models/design_model/s_19137.pt"
NA_MPNN_RUN_SHA256 = "f83959cc5f365a53dadae93d6e46c60f232832aa21b731ea1a6a9331b28a0f2a"
NA_MPNN_HEAD = "835974f3d420de18b3eb1dc66f25fbd17c53cc09"
CANONICAL_LD_LIBRARY_PATH = (
    "/opt/check-libs:"
    "/usr/local/nvidia/lib:"
    "/usr/local/nvidia/lib64:"
    "/alphafold3_venv/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:"
    "/alphafold3_venv/lib/python3.11/site-packages/nvidia/cudnn/lib:"
    "/.singularity.d/libs"
)
EXPECTED_SAMPLES = 20


class AcceptanceError(RuntimeError):
    """The RNA acceptance cannot produce trustworthy evidence."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_standalone(cfg: Any, path: Path) -> None:
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


def _compose_archived_config(out_dir: Path) -> Any:
    from omegaconf import open_dict

    from astrolabe.cli import _compose_hydra_config

    cfg = _compose_hydra_config(
        config_dir=REPO_ROOT / "config",
        config_name=ARCHIVED_CONFIG,
        overrides=["env=env_container"],
    )
    repo = os.environ.get("GH_NAMPNN_REPO", "/ext/na_mpnn")
    python_bin = os.environ.get(
        "GH_NAMPNN_PYTHON",
        "/alphafold3_venv/bin/python",
    )
    checkpoint = str(Path(repo) / "models" / "design_model" / "s_19137.pt")
    with open_dict(cfg):
        cfg.out_dir = str(out_dir)
        cfg.repeats = 1
        cfg.seed = ARCHIVED_SEED
        cfg.search.max_evaluations = MAX_EVALUATIONS
        cfg.search.max_proposal_requests = MAX_NAMPNN_REQUESTS_PER_ARM
        cfg.search.resume = False
        cfg.subprocess_envs.nampnn.repo_path = repo
        cfg.subprocess_envs.nampnn.python_bin = python_bin
        cfg.mutator.NAMPNN.repo_path = repo
        cfg.mutator.NAMPNN.python_bin = python_bin
        cfg.mutator.NAMPNN.checkpoint = checkpoint
    return cfg


def _archive_profile(cfg: Any) -> dict[str, Any]:
    from omegaconf import OmegaConf

    af3_term = OmegaConf.to_container(
        cfg.evaluator.evaluator_terms.Alphafold3,
        resolve=True,
    )
    af3 = OmegaConf.to_container(cfg.evaluator.alphafold3, resolve=True)
    af3.update(af3_term)
    nampnn_term = OmegaConf.to_container(
        cfg.mutator.mutator_terms.NAMPNN,
        resolve=True,
    )
    nampnn = OmegaConf.to_container(cfg.mutator.NAMPNN, resolve=True)
    nampnn.update(nampnn_term)
    return {
        "task_name": str(cfg.task_name),
        "seed": int(cfg.seed),
        "sequence_a": str(cfg.seqs.A),
        "types": OmegaConf.to_container(cfg.types, resolve=True),
        "oligo": list(cfg.oligo),
        "search": OmegaConf.to_container(cfg.search, resolve=True),
        "archive_start": OmegaConf.to_container(
            cfg.search.archived_starts[0], resolve=True
        ),
        "evaluator": {
            "alias": "Alphafold3",
            "type": str(af3["_type"]),
            "num_samples": int(af3["num_samples"]),
            "num_seeds": int(af3["num_seeds"]),
            "num_recycles": int(af3["num_recycles"]),
            "run_data_pipeline": bool(af3["run_data_pipeline"]),
            "flash_attention_implementation": str(
                af3["flash_attention_implementation"]
            ),
        },
        "mutator": {
            "alias": "NAMPNN",
            "type": str(nampnn["_type"]),
            "temperature": float(nampnn["temperature"]),
            "design_na_only": int(nampnn["design_na_only"]),
            "parse_na_only": int(nampnn["parse_na_only"]),
        },
        "runtime": {
            "af3_model_dir": str(cfg.weight_path.alphafold3),
            "nampnn_repo": str(cfg.subprocess_envs.nampnn.repo_path),
            "nampnn_python": str(cfg.subprocess_envs.nampnn.python_bin),
            "nampnn_checkpoint": NA_MPNN_CHECKPOINT,
        },
        "repeat_plan": OmegaConf.to_container(cfg.repeat_plan, resolve=True),
    }


def _assert_archived_profile(profile: Mapping[str, Any]) -> None:
    failures = []
    if profile["task_name"] != "bivalent_rna_archived":
        failures.append("task_name")
    if profile["seed"] != ARCHIVED_SEED:
        failures.append("seed")
    if profile["sequence_a"] != f"{ARCHIVED_SEQUENCE}@repeat repeat":
        failures.append("sequence_a")
    if profile["oligo"] != ["ABC"]:
        failures.append("oligo")
    if profile["evaluator"] != {
        "alias": "Alphafold3",
        "type": "Alphafold3",
        "num_samples": 5,
        "num_seeds": 1,
        "num_recycles": 10,
        "run_data_pipeline": False,
        "flash_attention_implementation": "cudnn",
    }:
        failures.append("evaluator")
    if profile["mutator"] != {
        "alias": "NAMPNN",
        "type": "NAMPNN",
        "temperature": 0.3,
        "design_na_only": 1,
        "parse_na_only": 0,
    }:
        failures.append("mutator")
    nampnn_python = Path(profile["runtime"]["nampnn_python"])
    if not nampnn_python.is_file() or not os.access(nampnn_python, os.X_OK):
        failures.append("runtime.nampnn_python")
    if profile["search"]["max_evaluations"] != MAX_EVALUATIONS:
        failures.append("max_evaluations")
    if (
        profile["search"].get("max_proposal_requests")
        != MAX_NAMPNN_REQUESTS_PER_ARM
    ):
        failures.append("max_proposal_requests")
    if profile["search"]["algorithm"] != "branching":
        failures.append("algorithm")
    if profile["search"]["owner"] != "archived_refinement":
        failures.append("owner")
    if profile["search"]["initialization"] != "archived":
        failures.append("initialization")
    if profile["archive_start"]["seed"] != ARCHIVED_SEED:
        failures.append("archive_start.seed")
    if profile["archive_start"]["sequence"] != ARCHIVED_SEQUENCE:
        failures.append("archive_start.sequence")
    if failures:
        raise AcceptanceError(
            "archived RNA scientific profile changed: " + ", ".join(failures)
        )


def _prepare(out_root: Path) -> dict[str, Any]:
    path = out_root.expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    cfg = _compose_archived_config(path / "gpu-run-seed")
    profile = _archive_profile(cfg)
    _assert_archived_profile(profile)
    config_path = path / "archived_rna_acceptance.yaml"
    _write_standalone(cfg, config_path)
    result = {
        "schema": "astrolabe.rna-archived-acceptance.prepare/v1",
        "status": "prepared",
        "profile": profile,
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "request_denominators": {
            "af3_evaluator_requests": 4,
            "nampnn_mutation_requests": MAX_NAMPNN_REQUESTS,
            "assemblies_per_request": 1,
            "af3_seeds_per_request": 1,
            "af3_samples_per_seed": 5,
            "structure_samples": EXPECTED_SAMPLES,
        },
        "resource_envelope": {
            "gpu": 1,
            "cpus": 4,
            "memory_gib": 28,
            "time_minutes": 60,
            "partition": "boron",
            "gpu_memory_requirement": "unknown; not estimated or claimed",
        },
        "verification_scope": {
            "cpu_compose": "PASS",
            "cold_nampnn_constructor": "performed separately in licensed SIF CPU",
            "gpu": "prepared only; not launched",
        },
    }
    _atomic_json(path / "prepared.json", result)
    return result


def _public_route_preflight(out_root: Path) -> dict[str, Any]:
    """Run the public CLI until the guard immediately before model construction."""

    from unified_gpu_acceptance import _run_process

    path = out_root.expanduser().resolve()
    path.mkdir(parents=True, exist_ok=False)
    run_root = path / "run"
    config_path = path / "initial.yaml"
    cfg = _compose_archived_config(run_root)
    profile = _archive_profile(cfg)
    _assert_archived_profile(profile)
    _write_standalone(cfg, config_path)

    instrumentation = path / "instrumentation"
    instrumentation.mkdir()
    marker = path / "model-factory-guard.json"
    (instrumentation / "sitecustomize.py").write_text(
        r'''
import json
import os
from pathlib import Path


def _astrolabe_pre_model_guard():
    from astrolabe.design.native_support import LazyNativeOwner

    original_initialize = LazyNativeOwner._initialize

    def guarded_initialize(self, request):
        def guarded_factory():
            marker = Path(os.environ["ASTROLABE_PRE_MODEL_GUARD"])
            payload = {
                "schema": "astrolabe.rna-archived-acceptance.model-factory-guard/v1",
                "event": "before_lazy_native_owner_factory",
                "pid": os.getpid(),
                "model": request.model.digest,
            }
            descriptor = os.open(
                marker,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            try:
                os.write(
                    descriptor,
                    (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"),
                )
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os._exit(86)

        self.factory = guarded_factory
        return original_initialize(self, request)

    LazyNativeOwner._initialize = guarded_initialize


_astrolabe_pre_model_guard()
'''.lstrip(),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        (str(instrumentation), str(REPO_ROOT), env.get("PYTHONPATH", ""))
    )
    env["ASTROLABE_PRE_MODEL_GUARD"] = str(marker)
    env["ASTROLABE_MODEL_EXECUTION"] = "daemon"
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["PYTHONHASHSEED"] = str(ARCHIVED_SEED)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    command = [
        sys.executable,
        "-m",
        "astrolabe.cli",
        "run",
        "--standalone",
        str(config_path),
        "--no-chdir",
        "--overrides",
        "env=env_container",
    ]
    process = _run_process(
        command,
        cwd=REPO_ROOT,
        env=env,
        stdout_log=path / "stdout.log",
        stderr_log=path / "stderr.log",
        timeout_s=300,
    )
    if process.returncode != PRE_MODEL_GUARD_EXIT_CODE:
        raise AcceptanceError(
            "public route did not reach the pre-model factory guard: "
            f"returncode={process.returncode}, stderr={path / 'stderr.log'}"
        )
    if not marker.is_file():
        raise AcceptanceError("pre-model factory guard marker is missing")
    result = {
        "schema": "astrolabe.rna-archived-acceptance.public-preflight/v1",
        "status": "PASS",
        "frontdoor": "astrolabe.cli run",
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "profile": profile,
        "process": {
            "command": command,
            "returncode": process.returncode,
            "timed_out": process.timed_out,
            "stdout": str(path / "stdout.log"),
            "stderr": str(path / "stderr.log"),
        },
        "guard": _load_json(marker),
        "guard_semantics": (
            "The guard replaced LazyNativeOwner.factory immediately before its "
            "first invocation; no native runner or model constructor ran."
        ),
        "execution": "CPU",
        "model_calls": 0,
        "gpu_used": False,
    }
    _atomic_json(path / "preflight.json", result)
    return result


def _source_manifest(root: Path) -> tuple[dict[str, str], str]:
    files = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in {"source_manifest.json", "capsule_metadata.json"}:
            continue
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        files[relative] = _sha256(path)
    fingerprint = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return files, fingerprint


def _verify_source_manifest(expected_path: Path) -> dict[str, Any]:
    expected = _load_json(expected_path)
    files = expected.get("files")
    if not isinstance(files, Mapping):
        raise AcceptanceError("source manifest has no file mapping")
    actual_files, actual_fingerprint = _source_manifest(REPO_ROOT)
    changes = []
    for name in sorted(set(files) | set(actual_files)):
        if files.get(name) != actual_files.get(name):
            changes.append(
                {
                    "path": name,
                    "expected": files.get(name),
                    "actual": actual_files.get(name),
                }
            )
    result = {
        "schema": "astrolabe.rna-archived-acceptance.source-check/v1",
        "expected_fingerprint": expected.get("fingerprint"),
        "actual_fingerprint": actual_fingerprint,
        "matches": not changes,
        "file_count": len(actual_files),
        "changes": changes,
    }
    if changes:
        raise AcceptanceError(
            "frozen source capsule changed: " + json.dumps(changes[:8], sort_keys=True)
        )
    return result


def _runtime_preflight() -> dict[str, Any]:
    import importlib.metadata

    import torch

    spec = importlib.util.find_spec("alphafold3")
    origin = None if spec is None or spec.origin is None else str(Path(spec.origin).resolve())
    if origin is None or not origin.startswith(AF3_SITE_PACKAGE + os.sep):
        raise AcceptanceError(
            "alphafold3 must load from the licensed SIF site package, "
            f"observed {origin!r}"
        )
    af3_paths = [
        str(item)
        for item in sys.path
        if "vendor/alphafold3" in str(item) or str(item).startswith("/models/alphafold3")
    ]
    if af3_paths:
        raise AcceptanceError(
            "host/vendor AlphaFold3 import roots are present: " + repr(af3_paths)
        )
    smi = shutil.which("nvidia-smi")
    if smi is None:
        raise AcceptanceError("nvidia-smi is unavailable")
    gpu = subprocess.run(
        [
            smi,
            "--query-gpu=uuid,name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    devices = []
    for line in gpu.stdout.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) >= 4:
            devices.append(
                {
                    "uuid": fields[0],
                    "name": fields[1],
                    "driver_version": fields[2],
                    "memory_total": fields[3],
                }
            )
    if not devices:
        raise AcceptanceError("nvidia-smi reported no visible GPU")
    return {
        "schema": "astrolabe.rna-archived-acceptance.runtime/v1",
        "python": sys.version,
        "executable": sys.executable,
        "alphafold3_origin": origin,
        "alphafold3_site_package": AF3_SITE_PACKAGE,
        "host_vendor_af3_root_present": False,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "prody": importlib.metadata.version("ProDy"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
        "canonical_ld_library_path": CANONICAL_LD_LIBRARY_PATH,
        "ld_library_path_matches": os.environ.get("LD_LIBRARY_PATH") == CANONICAL_LD_LIBRARY_PATH,
        "devices": devices,
    }


def _verify_nampnn_source() -> dict[str, Any]:
    from astrolabe.mutators.wrappers.na_mpnn_daemon import verify_vendor

    repo = Path("/ext/na_mpnn")
    verify_vendor(repo)
    run_py = repo / "inference" / "run.py"
    checkpoint = Path(NA_MPNN_CHECKPOINT)
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != NA_MPNN_HEAD:
        raise AcceptanceError(f"NA-MPNN HEAD changed: {head}")
    source_hash = _sha256(run_py)
    if source_hash != NA_MPNN_RUN_SHA256:
        raise AcceptanceError(f"NA-MPNN resident source changed: {source_hash}")
    return {
        "repo": str(repo),
        "head": head,
        "run_py": str(run_py),
        "run_py_sha256": source_hash,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "pythonpath_requires_inference": True,
    }


def _install_instrumentation(arm_root: Path, *, crash: bool) -> dict[str, str]:
    from unified_gpu_acceptance import _SITECUSTOMIZE

    instrumentation = arm_root / "instrumentation"
    instrumentation.mkdir(parents=True, exist_ok=True)
    crash_hook = r'''

def _astrolabe_cut_install():
    from collections.abc import Mapping
    import hashlib
    import re
    import numpy as np
    from pathlib import Path
    cut_round = os.environ.get("ASTROLABE_ACCEPTANCE_CRASH_ROUND")
    marker_value = os.environ.get("ASTROLABE_ACCEPTANCE_CRASH_MARKER")
    mutator_capture = os.environ.get("ASTROLABE_ACCEPTANCE_MUTATOR_CAPTURE")
    nampnn_capture = os.environ.get("ASTROLABE_ACCEPTANCE_NAMPNN_CAPTURE")
    nampnn_claims_value = os.environ.get("ASTROLABE_ACCEPTANCE_NAMPNN_CLAIMS")
    nampnn_maximum_value = os.environ.get("ASTROLABE_ACCEPTANCE_NAMPNN_MAX_CALLS")
    if not mutator_capture or not nampnn_capture or not nampnn_claims_value:
        return
    from astrolabe.mutators.wrappers.na_mpnn_runner import NAMPNNRunner
    from astrolabe.mutators.na_mpnn import NAMPNNMutator

    original_design = NAMPNNRunner.design_sequences
    original_compute = NAMPNNMutator.compute_mutation
    nampnn_capture_path = Path(nampnn_capture)
    nampnn_capture_path.mkdir(parents=True, exist_ok=True)
    nampnn_claims = Path(nampnn_claims_value)
    nampnn_claims.mkdir(parents=True, exist_ok=True)
    nampnn_maximum = int(nampnn_maximum_value or "1")
    conversion_index = [0]

    def append_jsonl(path, payload):
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(
                descriptor,
                (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"),
            )
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def claim_nampnn():
        for index in range(nampnn_maximum + 1):
            marker = nampnn_claims / (str(index) + ".claim")
            try:
                descriptor = os.open(
                    marker,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
            except FileExistsError:
                continue
            os.close(descriptor)
            if index >= nampnn_maximum:
                raise RuntimeError(
                    "NA-MPNN acceptance budget exceeded before model request "
                    f"({index + 1} > {nampnn_maximum}); no replay"
                )
            return index
        raise RuntimeError("NA-MPNN acceptance claim space exhausted")

    def capture_array(prefix, value, arrays, manifest, scalars):
        if isinstance(value, np.ndarray):
            array = np.asarray(value)
            if array.dtype.hasobject:
                values = array.tolist()
                scalars[prefix] = values
                manifest.append({
                    "key": prefix,
                    "dtype": str(array.dtype),
                    "shape": list(array.shape),
                    "encoding": "json",
                    "sha256": hashlib.sha256(
                        json.dumps(
                            values,
                            sort_keys=True,
                            separators=(",", ":"),
                            default=str,
                        ).encode("utf-8")
                    ).hexdigest(),
                })
            else:
                array = array.copy()
                arrays[prefix] = array
                manifest.append({
                    "key": prefix,
                    "dtype": array.dtype.str,
                    "shape": list(array.shape),
                    "encoding": "npz",
                })
            return
        if hasattr(value, "detach") and hasattr(value, "cpu"):
            capture_array(
                prefix,
                value.detach().cpu().numpy(),
                arrays,
                manifest,
                scalars,
            )
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                capture_array(
                    prefix + "." + str(key),
                    item,
                    arrays,
                    manifest,
                    scalars,
                )
            return
        if isinstance(value, (list, tuple)):
            scalars[prefix] = [
                item.item() if hasattr(item, "item") else item
                for item in value
            ]
            return
        if value is None or isinstance(value, (bool, int, float, str)):
            scalars[prefix] = (
                value.item() if hasattr(value, "item") else value
            )

    def capture_mutator(method):
        def wrapped(self, *args, **kwargs):
            index = claim_nampnn()
            path = Path(mutator_capture)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "index": index,
                "method": "design_sequences",
                "structure_path": str(kwargs.get("structure_path", "")),
                "job_name": str(kwargs.get("job_name", "")),
                "execution_mode": getattr(self, "execution_mode", None),
            }
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.write(descriptor, (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"))
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            result = method(self, *args, **kwargs)
            fasta_path = Path(result["fasta_path"])
            fasta_copy = nampnn_capture_path / f"request-{index:04d}.fa"
            fasta_copy.write_bytes(fasta_path.read_bytes())
            arrays = {}
            manifest = []
            scalars = {}
            capture_array(
                f"request.{index:04d}.probabilities",
                result.get("probabilities"),
                arrays,
                manifest,
                scalars,
            )
            array_path = nampnn_capture_path / f"request-{index:04d}.npz"
            temporary_array_path = array_path.with_suffix(".npz.tmp")
            with temporary_array_path.open("wb") as handle:
                np.savez(handle, **arrays)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_array_path, array_path)
            record = {
                "schema": "astrolabe.rna-archived-acceptance.nampnn-request/v1",
                "index": index,
                "job_name": str(kwargs.get("job_name", "")),
                "execution_mode": getattr(self, "execution_mode", None),
                "seed": int(self.extra_args.get("seed", self.seed)),
                "extra_args": dict(self.extra_args),
                "chains_to_design": kwargs.get("chains_to_design"),
                "fixed_residues": list(kwargs.get("fixed_residues") or ()),
                "parse_these_chains_only": kwargs.get("parse_these_chains_only"),
                "fasta_sha256": hashlib.sha256(fasta_copy.read_bytes()).hexdigest(),
                "fasta_sequence": result.get("sequence"),
                "fasta_header": result.get("header"),
                "fasta_metadata": result.get("metadata"),
                "probability_type": type(result.get("probabilities")).__name__,
                "probability_top_level_keys": (
                    sorted(str(key) for key in result["probabilities"])
                    if isinstance(result.get("probabilities"), Mapping)
                    else None
                ),
                "probability_manifest": manifest,
                "probability_scalars": scalars,
            }
            append_jsonl(
                nampnn_capture_path / "requests.jsonl",
                record,
            )
            return result
        wrapped._astrolabe_acceptance_wrapped = True
        return wrapped

    NAMPNNRunner.design_sequences = capture_mutator(original_design)

    def capture_conversion(method):
        def wrapped(self, features):
            result = method(self, features)
            index = conversion_index[0]
            conversion_index[0] += 1
            arrays = {}
            manifest = []
            scalars = {}
            converted = {}
            for oligo, output in result.items():
                proto_sequences = output.get("proto_sequences", {})
                converted[oligo] = {}
                for proto, payload in proto_sequences.items():
                    converted[oligo][proto] = {
                        "sequence": payload.get("sequence"),
                        "excluded": sorted(payload.get("excluded") or ()),
                    }
                    capture_array(
                        f"conversion.{index:04d}.{oligo}.{proto}.sampling_probs",
                        payload.get("sampling_probs"),
                        arrays,
                        manifest,
                        scalars,
                    )
            array_path = nampnn_capture_path / f"conversion-{index:04d}.npz"
            temporary_array_path = array_path.with_suffix(".npz.tmp")
            with temporary_array_path.open("wb") as handle:
                np.savez(handle, **arrays)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_array_path, array_path)
            append_jsonl(
                nampnn_capture_path / "conversions.jsonl",
                {
                    "schema": "astrolabe.rna-archived-acceptance.nampnn-conversion/v1",
                    "index": index,
                    "converted": converted,
                    "array_manifest": manifest,
                    "array_scalars": scalars,
                },
            )
            return result
        wrapped._astrolabe_acceptance_wrapped = True
        return wrapped

    NAMPNNMutator.compute_mutation = capture_conversion(original_compute)
    if (
        cut_round
        and marker_value
        and os.environ.get("ASTROLABE_ACCEPTANCE_CRASHED") != "1"
    ):
        from astrolabe.design.runtime import Runtime

        original_call = Runtime.call

        def crash_after_receipt(self, named_round, operation, request, *, work_id=None):
            result = original_call(self, named_round, operation, request, work_id=work_id)
            if named_round == cut_round:
                marker = Path(marker_value)
                if not marker.exists():
                    payload = {
                        "schema": "astrolabe.rna-archived-acceptance.cut/v1",
                        "named_round": named_round,
                        "work_id": work_id,
                        "pid": os.getpid(),
                        "exit_code": 97,
                    }
                    descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                    try:
                        os.write(descriptor, (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"))
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
                    os._exit(97)
            return result

        Runtime.call = crash_after_receipt

_astrolabe_cut_install()
'''.lstrip()
    (instrumentation / "sitecustomize.py").write_text(
        _SITECUSTOMIZE + "\n" + crash_hook,
        encoding="utf-8",
    )
    capture = arm_root / "capture.pickle"
    claims = arm_root / "call-claims"
    claims.mkdir(parents=True, exist_ok=True)
    mutator_capture = arm_root / "mutator-calls.jsonl"
    nampnn_capture = arm_root / "nampnn-results"
    nampnn_capture.mkdir(parents=True, exist_ok=True)
    nampnn_claims = arm_root / "nampnn-call-claims"
    nampnn_claims.mkdir(parents=True, exist_ok=True)
    import astrolabe

    package_root = Path(astrolabe.__file__).resolve().parent.parent
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (
            str(instrumentation),
            str(package_root),
            "/ext/na_mpnn",
            "/ext/na_mpnn/inference",
            existing or "",
        )
        if part
    )
    env["ASTROLABE_ACCEPTANCE_CAPTURE"] = str(capture)
    env["ASTROLABE_ACCEPTANCE_CLAIMS"] = str(claims)
    env["ASTROLABE_ACCEPTANCE_MAX_CALLS"] = str(MAX_EVALUATIONS)
    env["ASTROLABE_ACCEPTANCE_MUTATOR_CAPTURE"] = str(mutator_capture)
    env["ASTROLABE_ACCEPTANCE_NAMPNN_CAPTURE"] = str(nampnn_capture)
    env["ASTROLABE_ACCEPTANCE_NAMPNN_CLAIMS"] = str(nampnn_claims)
    env["ASTROLABE_ACCEPTANCE_NAMPNN_MAX_CALLS"] = str(
        MAX_NAMPNN_REQUESTS_PER_ARM
    )
    env["ASTROLABE_ACCEPTANCE_CRASH_ROUND"] = CUT_ROUND
    env["ASTROLABE_ACCEPTANCE_CRASH_MARKER"] = str(arm_root / "cut.marker")
    env["ASTROLABE_ACCEPTANCE_CRASHED"] = "0"
    env["ASTROLABE_RUNTIME_ACTIVE"] = "1"
    env["PYTHONHASHSEED"] = str(ARCHIVED_SEED)
    if crash:
        (arm_root / "cut.marker").unlink(missing_ok=True)
    else:
        env["ASTROLABE_ACCEPTANCE_CRASHED"] = "1"
    return env


def _run_child(
    config_path: Path,
    *,
    arm_root: Path,
    label: str,
    crash: bool,
    timeout_s: float,
) -> dict[str, Any]:
    from unified_gpu_acceptance import _run_process

    env = _install_instrumentation(arm_root, crash=crash)
    command = [
        sys.executable,
        "-m",
        "astrolabe.cli",
        "run",
        "--standalone",
        str(config_path),
        "--no-chdir",
        "--overrides",
        "env=env_container",
    ]
    started = time.monotonic()
    process = _run_process(
        command,
        cwd=Path("/opt/Astrolabe"),
        env=env,
        stdout_log=arm_root / f"{label}.stdout.log",
        stderr_log=arm_root / f"{label}.stderr.log",
        timeout_s=timeout_s,
    )
    return {
        "label": label,
        "command": command,
        "returncode": process.returncode,
        "timed_out": process.timed_out,
        "duration_s": process.duration_s,
        "started_unix": time.time() - (time.monotonic() - started),
        "stdout": str(process.stdout_log),
        "stderr": str(process.stderr_log),
    }


def _run_roots(arm_root: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(path.parent.parent.resolve() for path in arm_root.rglob(".framework/journal.sqlite"))
    )


def _single_run_root(arm_root: Path) -> Path:
    roots = _run_roots(arm_root)
    if len(roots) != 1:
        raise AcceptanceError(
            f"expected one durable run root under {arm_root}, found {roots!r}"
        )
    return roots[0]


def _decode_attempts(run_root: Path) -> tuple[dict[str, Any], ...]:
    from astrolabe.design.store import DataStore, design_codec
    from astrolabe.design.types import Blob, Key

    database = run_root / ".framework" / "journal.sqlite"
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        rows = connection.execute(
            "SELECT value FROM records WHERE kind='attempt' ORDER BY name"
        ).fetchall()
    store = DataStore(run_root / ".framework" / "data")
    store.register(design_codec)
    decoded = []
    for (raw,) in rows:
        value = json.loads(raw)
        key_record = value.get("key")
        if not isinstance(key_record, Mapping):
            continue
        call_key = store.get(
            Blob(Key(key_record["identity"]), key_record["codec"]),
            design_codec,
        )
        decoded.append(
            {
                "named_round": call_key.named_round,
                "work_id": call_key.work_id,
                "status": value.get("status"),
                "result": value.get("result"),
                "rng": value.get("rng"),
            }
        )
    return tuple(decoded)


def _assert_raw_prediction_cut(run_root: Path, marker: Path) -> dict[str, Any]:
    if not marker.is_file():
        raise AcceptanceError("os._exit cut marker was not written")
    attempts = _decode_attempts(run_root)
    prediction = [
        item
        for item in attempts
        if item["named_round"] == CUT_ROUND and item["status"] == "completed"
    ]
    measurement = [
        item
        for item in attempts
        if item["named_round"] == "native-refinement/evaluate/1/measurement"
    ]
    if len(prediction) != 1 or prediction[0]["result"] is None or prediction[0]["rng"] is None:
        raise AcceptanceError("cut did not follow a complete durable raw-prediction receipt")
    if measurement:
        raise AcceptanceError("measurement started before the injected process cut")
    return {
        "schema": "astrolabe.rna-archived-acceptance.cut-check/v1",
        "marker": _load_json(marker),
        "prediction": prediction[0],
        "measurement_absent": True,
        "attempts": list(attempts),
    }


def _write_resume_config(run_root: Path, destination: Path) -> Any:
    from omegaconf import OmegaConf, open_dict

    saved_path = run_root / "04_config.yaml"
    if not saved_path.is_file():
        raise AcceptanceError(f"RNA durable config is missing: {saved_path}")
    saved = OmegaConf.load(saved_path)
    with open_dict(saved):
        if saved.get("repeat_plan") is not None:
            del saved.repeat_plan
        saved.out_dir = str(run_root)
        saved.repeats = 1
        saved.search.resume = True
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(OmegaConf.to_yaml(saved, resolve=False), encoding="utf-8")
    return saved


def _normalize_tree(value: Any, roots: Sequence[tuple[str, Path]]) -> Any:
    if isinstance(value, str):
        for label, root in roots:
            text = str(root)
            if value == text:
                return f"${label}"
            if value.startswith(text + os.sep):
                return f"${label}/" + value[len(text) + 1 :].replace(os.sep, "/")
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_tree(item, roots)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_normalize_tree(item, roots) for item in value]
    if isinstance(value, tuple):
        return [_normalize_tree(item, roots) for item in value]
    return value


def _load_events(run_root: Path) -> list[Any]:
    path = run_root / "01_evaluations.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_jsonl(path: Path) -> list[Any]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _nampnn_array_manifests(arm_root: Path) -> dict[str, Any]:
    capture = arm_root / "nampnn-results"
    return {
        path.name: _array_manifest(path)
        for path in sorted(capture.glob("*.npz"))
    }


def _nampnn_capture_complete(arm_root: Path) -> bool:
    capture = arm_root / "nampnn-results"
    requests = _load_jsonl(capture / "requests.jsonl")
    conversions = _load_jsonl(capture / "conversions.jsonl")
    arrays = _nampnn_array_manifests(arm_root)
    return (
        len(requests) == 1
        and len(conversions) == 1
        and bool(requests[0].get("probability_manifest"))
        and bool(conversions[0].get("converted"))
        and "request-0000.npz" in arrays
        and "conversion-0000.npz" in arrays
    )


def _write_arm_evidence(arm_root: Path, run_root: Path) -> dict[str, Any]:
    from unified_gpu_acceptance import (
        _read_capture_records,
        _score_and_coordinate_evidence,
        _write_npz,
    )

    capture_path = arm_root / "capture.pickle"
    records = _read_capture_records(capture_path)
    if not 1 <= len(records) <= MAX_EVALUATIONS:
        raise AcceptanceError(
            f"{arm_root.name} made {len(records)} evaluator calls, "
            f"expected 1 through {MAX_EVALUATIONS}"
        )
    (
        score_arrays,
        score_metadata,
        rng_arrays,
        rng_metadata,
        coordinate_arrays,
        coordinate_metadata,
    ) = _score_and_coordinate_evidence(arm_root, records)
    evidence = arm_root / "evidence"
    _write_npz(evidence / "standard_scores.npz", score_arrays)
    _write_npz(evidence / "rng_trace.npz", rng_arrays)
    _write_npz(evidence / "structure_coordinates.npz", coordinate_arrays)
    _atomic_json(evidence / "standard_scores.json", score_metadata)
    _atomic_json(evidence / "rng_trace.json", rng_metadata)
    _atomic_json(evidence / "structure_coordinates.json", coordinate_metadata)
    final_beam = _load_json(run_root / "03_final_beam.json")
    events = _load_events(run_root)
    mutator_rows = []
    mutator_path = arm_root / "mutator-calls.jsonl"
    if mutator_path.is_file():
        mutator_rows = [
            json.loads(line)
            for line in mutator_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    result = {
        "schema": "astrolabe.rna-archived-acceptance.arm/v1",
        "arm": arm_root.name,
        "run_root": str(run_root),
        "evaluator_requests": len(records),
        "early_stop": len(records) < MAX_EVALUATIONS,
        "mutator_requests": len(mutator_rows),
        "nampnn_result_requests": _load_jsonl(
            arm_root / "nampnn-results" / "requests.jsonl"
        ),
        "nampnn_conversions": _load_jsonl(
            arm_root / "nampnn-results" / "conversions.jsonl"
        ),
        "nampnn_array_manifests": _nampnn_array_manifests(arm_root),
        "score_array_count": len(score_arrays),
        "rng_array_count": len(rng_arrays),
        "coordinate_array_count": len(coordinate_arrays),
        "final_beam": final_beam,
        "events": events,
        "mutator_requests_detail": mutator_rows,
        "score_metadata": score_metadata,
        "rng_metadata": rng_metadata,
        "coordinate_metadata": coordinate_metadata,
    }
    _atomic_json(evidence / "arm.json", result)
    return result


def _array_manifest(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        values = {}
        for key in sorted(payload.files):
            array = np.asarray(payload[key])
            values[key] = {
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
            }
        return values


def _compare_arms(reference: Path, recovered: Path) -> dict[str, Any]:
    checks = {}
    for name in ("standard_scores.npz", "rng_trace.npz", "structure_coordinates.npz"):
        left = _array_manifest(reference / "evidence" / name)
        right = _array_manifest(recovered / "evidence" / name)
        checks[name] = {
            "equal": left == right,
            "reference": left,
            "recovered": right,
        }
    reference_roots = (
        ("RUN", Path(_load_json(reference / "evidence" / "arm.json")["run_root"])),
        ("ARM", reference),
    )
    recovered_roots = (
        ("RUN", Path(_load_json(recovered / "evidence" / "arm.json")["run_root"])),
        ("ARM", recovered),
    )
    for key, filename in (
        ("nampnn_request_metadata", "requests.jsonl"),
        ("nampnn_native_to_rna", "conversions.jsonl"),
    ):
        left = _normalize_tree(
            _load_jsonl(reference / "nampnn-results" / filename),
            reference_roots,
        )
        right = _normalize_tree(
            _load_jsonl(recovered / "nampnn-results" / filename),
            recovered_roots,
        )
        checks[key] = {
            "equal": left == right,
            "reference": left,
            "recovered": right,
        }
    left_arrays = _nampnn_array_manifests(reference)
    right_arrays = _nampnn_array_manifests(recovered)
    checks["nampnn_probability_arrays"] = {
        "equal": left_arrays == right_arrays,
        "reference": left_arrays,
        "recovered": right_arrays,
    }

    reference_arm = _load_json(reference / "evidence" / "arm.json")
    recovered_arm = _load_json(recovered / "evidence" / "arm.json")
    reference_run = Path(reference_arm["run_root"])
    recovered_run = Path(recovered_arm["run_root"])
    scientific_fields = (
        "final_beam",
        "events",
        "score_metadata",
        "rng_metadata",
        "coordinate_metadata",
    )
    normalized_reference = {
        key: _normalize_tree(reference_arm[key], reference_roots)
        for key in scientific_fields
    }
    normalized_recovered = {
        key: _normalize_tree(recovered_arm[key], recovered_roots)
        for key in scientific_fields
    }
    checks["proposal_event_and_selection_projection"] = {
        "equal": normalized_reference == normalized_recovered,
    }
    if normalized_reference != normalized_recovered:
        differing = [
            key
            for key in scientific_fields
            if normalized_reference[key] != normalized_recovered[key]
        ]
        checks["proposal_event_and_selection_projection"]["differing_fields"] = differing
    result = {
        "schema": "astrolabe.rna-archived-acceptance.comparison/v1",
        "status": "PASS" if all(item["equal"] for item in checks.values()) else "ISSUES",
        "comparison": "exact",
        "tolerance": 0,
        "checks": checks,
    }
    return result


def _run(out_root: Path, *, timeout_s: float, source_manifest: Path) -> int:
    path = out_root.expanduser().resolve()
    if path.exists():
        raise AcceptanceError(f"GPU output root must not exist before launch: {path}")
    path.mkdir(parents=True)
    deadline = time.monotonic() + timeout_s

    def remaining_timeout() -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AcceptanceError("acceptance wall-clock budget exhausted")
        return remaining

    try:
        source_check = _verify_source_manifest(source_manifest)
        environment = _runtime_preflight()
        nampnn = _verify_nampnn_source()
        if not environment["ld_library_path_matches"]:
            raise AcceptanceError(
                "LD_LIBRARY_PATH differs from the frozen Hegel-validated order"
            )
        _atomic_json(path / "source_check.json", source_check)
        _atomic_json(path / "runtime_environment.json", environment)
        _atomic_json(path / "nampnn_source.json", nampnn)

        baseline = path / "baseline"
        baseline.mkdir()
        baseline_config = baseline / "initial.yaml"
        _write_standalone(
            _compose_archived_config(baseline / "configured-output"),
            baseline_config,
        )
        baseline_process = _run_child(
            baseline_config,
            arm_root=baseline,
            label="initial",
            crash=False,
            timeout_s=remaining_timeout(),
        )
        if baseline_process["returncode"] != 0 or baseline_process["timed_out"]:
            raise AcceptanceError(f"baseline failed: {baseline_process!r}")
        baseline_run = _single_run_root(baseline)

        recovery = path / "recovery"
        recovery.mkdir()
        recovery_config = recovery / "initial.yaml"
        _write_standalone(
            _compose_archived_config(recovery / "configured-output"),
            recovery_config,
        )
        crash_process = _run_child(
            recovery_config,
            arm_root=recovery,
            label="crash",
            crash=True,
            timeout_s=remaining_timeout(),
        )
        root_only_early_stop = crash_process["returncode"] == 0
        if (
            crash_process["returncode"] != CUT_EXIT_CODE
            and not root_only_early_stop
        ):
            raise AcceptanceError(
                "recovery arm did not stop at the durable prediction receipt: "
                f"{crash_process!r}"
            )
        recovery_run = _single_run_root(recovery)
        if root_only_early_stop:
            cut = {
                "schema": "astrolabe.rna-archived-acceptance.cut-check/v1",
                "status": "INCOMPLETE",
                "reason": "root_only_early_stop; no child prediction receipt",
                "marker": None,
            }
            _atomic_json(recovery / "cut.json", cut)
            resume_process = None
        else:
            cut = _assert_raw_prediction_cut(recovery_run, recovery / "cut.marker")
            _atomic_json(recovery / "cut.json", cut)

            resume_config = recovery / "resume.yaml"
            _write_resume_config(recovery_run, resume_config)
            resume_process = _run_child(
                resume_config,
                arm_root=recovery,
                label="resume",
                crash=False,
                timeout_s=remaining_timeout(),
            )
            if resume_process["returncode"] != 0 or resume_process["timed_out"]:
                raise AcceptanceError(f"resume failed: {resume_process!r}")

        baseline_evidence = _write_arm_evidence(baseline, baseline_run)
        recovery_evidence = _write_arm_evidence(recovery, recovery_run)
        total_af3 = (
            baseline_evidence["evaluator_requests"]
            + recovery_evidence["evaluator_requests"]
        )
        total_nampnn = (
            baseline_evidence["mutator_requests"]
            + recovery_evidence["mutator_requests"]
        )
        if total_af3 > 4 or total_nampnn > 2:
            raise AcceptanceError(
                f"request budget exceeded: AF3={total_af3}, NAMPNN={total_nampnn}"
            )
        comparison = _compare_arms(baseline, recovery)
        _atomic_json(path / "comparison.json", comparison)
        capture_complete = (
            _nampnn_capture_complete(baseline)
            and _nampnn_capture_complete(recovery)
        )
        status = (
            "INCOMPLETE"
            if root_only_early_stop or not capture_complete
            else comparison["status"]
        )
        result = {
            "schema": "astrolabe.rna-archived-acceptance.summary/v1",
            "status": status,
            "baseline_process": baseline_process,
            "crash_process": crash_process,
            "resume_process": resume_process,
            "baseline_run_root": str(baseline_run),
            "recovery_run_root": str(recovery_run),
            "cut": cut,
            "complete_child_recovery": not root_only_early_stop,
            "complete_nampnn_capture": capture_complete,
            "request_counts": {
                "af3": {
                    "baseline": baseline_evidence["evaluator_requests"],
                    "recovery": recovery_evidence["evaluator_requests"],
                    "total": total_af3,
                    "maximum": 4,
                    "early_stop": total_af3 < 4,
                },
                "nampnn": {
                    "baseline": baseline_evidence["mutator_requests"],
                    "recovery": recovery_evidence["mutator_requests"],
                    "total": total_nampnn,
                    "maximum": 2,
                    "early_stop": total_nampnn < MAX_NAMPNN_REQUESTS,
                },
                "structure_samples_formula": (
                    "observed AF3 requests * 1 assembly/evaluation * "
                    "1 seed * 5 samples"
                ),
                "structure_samples_expected": EXPECTED_SAMPLES,
                "structure_samples_actual": total_af3 * 5,
                "early_stop": (
                    total_af3 < 4
                    or total_nampnn < MAX_NAMPNN_REQUESTS
                    or not capture_complete
                ),
            },
            "comparison": str(path / "comparison.json"),
            "limitations": {
                "gpu_memory_requirement": "unknown; not claimed",
                "gpu_performance": "not claimed",
                "gpu_numerical_result": (
                    "PASS only after this run completes; otherwise not established"
                ),
            },
        }
        _atomic_json(path / "summary.json", result)
        print(json.dumps({"status": result["status"], "summary": str(path / "summary.json")}))
        return 0 if result["status"] == "PASS" else 1
    except BaseException as error:
        _atomic_json(
            path / "failure.json",
            {
                "schema": "astrolabe.rna-archived-acceptance.failure/v1",
                "status": "ISSUES",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Compose and validate without model execution")
    prepare.add_argument("--out-root", required=True, type=Path)
    preflight = commands.add_parser(
        "public-preflight",
        help="Run the public CLI on CPU until the pre-model factory guard",
    )
    preflight.add_argument("--out-root", required=True, type=Path)
    run = commands.add_parser("run", help="Run baseline and durable-cut recovery acceptance")
    run.add_argument("--out-root", required=True, type=Path)
    run.add_argument("--timeout-s", type=float, default=3300)
    run.add_argument(
        "--source-manifest",
        type=Path,
        default=REPO_ROOT / "source_manifest.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "prepare":
        result = _prepare(arguments.out_root)
        print(json.dumps({"status": result["status"], "prepared": str(arguments.out_root)}))
        return 0
    if arguments.command == "public-preflight":
        result = _public_route_preflight(arguments.out_root)
        print(json.dumps({"status": result["status"], "preflight": str(arguments.out_root)}))
        return 0
    if arguments.timeout_s <= 0:
        raise ValueError("--timeout-s must be positive")
    return _run(
        arguments.out_root,
        timeout_s=arguments.timeout_s,
        source_manifest=arguments.source_manifest,
    )


if __name__ == "__main__":
    raise SystemExit(main())
