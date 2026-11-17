"""Configuration integration, kept separate from the dependency-free collector."""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import warnings

from .core import ProfileConfig, child_environment, profile_session, span


def _notice(message):
    try:
        print(message, file=sys.stderr, flush=True)
    except Exception:
        pass


def _source_identity():
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    paths = sorted((root / "astrolabe").rglob("*.py"))
    paths.append(root / "run.py")
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=3, check=False,
        )
        commit = result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        commit = None
    return {"root": str(root), "commit": commit, "python_source_sha256": digest.hexdigest()}


def _metadata(cfg):
    from omegaconf import OmegaConf

    raw = OmegaConf.to_container(cfg, resolve=False) if OmegaConf.is_config(cfg) else dict(cfg)
    # Environment profiles may contain credentials. Their contents are not a
    # profiling artifact; only a small explicit list of numerical/thread knobs is.
    raw.pop("env", None)
    raw.pop("profiling", None)
    safe_environment = {
        name: os.environ[name]
        for name in (
            "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", "JAX_PLATFORMS", "XLA_FLAGS",
            "XLA_PYTHON_CLIENT_PREALLOCATE", "PYTHONHASHSEED",
        )
        if name in os.environ
    }
    encoded = json.dumps(raw, sort_keys=True, default=str).encode()
    return {
        "configuration": raw,
        "configuration_sha256": hashlib.sha256(encoded).hexdigest(),
        "configuration_resolution": "unresolved; effective repeat seed is recorded in run.repeat",
        "source": _source_identity(),
        "environment": safe_environment,
        "measurement": "diagnostic; no device synchronization added",
    }


@contextmanager
def configured_profile(cfg):
    """Own one profile across CLI, API and repeat scopes, including cleanup."""
    if not isinstance(cfg, Mapping):
        yield None
        return
    from omegaconf import OmegaConf

    requested = cfg.get("profiling")
    if OmegaConf.is_config(requested):
        requested = OmegaConf.to_container(requested, resolve=True)
    options = ProfileConfig.from_mapping(requested)
    if not options.enabled or child_environment():
        yield None
        return
    try:
        metadata = _metadata(cfg)
    except Exception as exc:
        metadata = {"metadata_error": f"{type(exc).__name__}: {exc}"}
    output = cfg.get("out_dir") or "outputs"
    directory = None
    try:
        with profile_session(options, output_dir=output, metadata=metadata) as session:
            directory = session.directory if session is not None else None
            if directory is not None:
                _notice(f"Profiling artifacts: {directory}")
            with span("run.total"):
                yield session
    finally:
        if directory is not None:
            try:
                from .report import report_session
                report_session(directory)
                _notice(f"Profiling report: {Path(directory) / 'report.html'}")
            except Exception as exc:
                try:
                    warnings.warn(f"Profiling report unavailable: {exc}", RuntimeWarning)
                except Exception:
                    pass
