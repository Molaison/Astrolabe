#!/usr/bin/env python3
"""Run the frozen posthoc public-caller acceptance in prepare-then-run order."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping


CAPSULE_ROOT = Path(__file__).resolve().parents[2]
HARNESS = CAPSULE_ROOT / "scripts" / "validation" / "posthoc_gpu_acceptance.py"
SOURCE_MANIFEST = CAPSULE_ROOT / "source_manifest.json"
CAPSULE_METADATA = CAPSULE_ROOT / "capsule_metadata.json"
IDENTITY_PREFLIGHT_SCHEMA = (
    "astrolabe.posthoc-refold-acceptance.identity-preflight/v1"
)


class RunnerError(RuntimeError):
    """The frozen acceptance runner cannot prove a safe execution boundary."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_value(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _verify_capsule() -> dict[str, Any]:
    if not SOURCE_MANIFEST.is_file():
        raise RunnerError(f"source manifest is absent: {SOURCE_MANIFEST}")
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise RunnerError(f"invalid source manifest: {SOURCE_MANIFEST}")
    mismatches = []
    for relative, expected in files.items():
        path = CAPSULE_ROOT / relative
        if not path.is_file() or _sha256(path) != expected:
            mismatches.append(relative)
    if mismatches:
        raise RunnerError(
            "source capsule hash mismatch: " + ", ".join(sorted(mismatches))
        )
    excluded = {SOURCE_MANIFEST.name, CAPSULE_METADATA.name}
    actual = {
        path.relative_to(CAPSULE_ROOT).as_posix()
        for path in CAPSULE_ROOT.rglob("*")
        if path.is_file()
        and path.relative_to(CAPSULE_ROOT).as_posix() not in excluded
        and "__pycache__" not in path.parts
        and ".pytest_cache" not in path.parts
    }
    unlisted = actual.difference(files)
    if unlisted:
        raise RunnerError(
            "source capsule has unlisted files: " + ", ".join(sorted(unlisted))
        )
    return {
        "files": len(files),
        "manifest": str(SOURCE_MANIFEST),
        "manifest_sha256": _sha256(SOURCE_MANIFEST),
    }


def _run_command(
    command: list[str],
    *,
    deadline: float,
    stage: str,
) -> None:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise RunnerError(f"wall budget exhausted before {stage}")
    completed = subprocess.run(
        command,
        check=False,
        timeout=remaining,
    )
    if completed.returncode != 0:
        raise RunnerError(
            f"{stage} returned {completed.returncode}: {command}"
        )


def _identity_preflight(
    *,
    out_root: Path,
    repo_root: Path,
    capsule: Mapping[str, Any],
) -> dict[str, Any]:
    from omegaconf import OmegaConf

    from astrolabe.posthoc import refold_receipts

    prepared = json.loads((out_root / "prepared.json").read_text(encoding="utf-8"))
    config_path = out_root / prepared["inputs"]["config"]
    config = OmegaConf.load(str(config_path))
    selected_config = OmegaConf.to_container(
        config.evaluator.alphafold3,
        resolve=True,
    )
    if not isinstance(selected_config, dict):
        raise RunnerError("frozen evaluator.alphafold3 config is not a mapping")
    assets = refold_receipts.selected_asset_record(
        backend="af3",
        selected_config=selected_config,
    )
    numeric_source = refold_receipts.numeric_source_record(
        "af3",
        repo_root=repo_root,
    )
    payload = {
        "schema": IDENTITY_PREFLIGHT_SCHEMA,
        "capsule_source_manifest_sha256": capsule["manifest_sha256"],
        "required_runtime_origin": (
            "/alphafold3_venv/lib/python3.11/site-packages/alphafold3"
        ),
        "config_sha256": prepared["input_hashes"]["config"],
        "model_dir": prepared["model_dir"],
        "asset_record": assets,
        "asset_fingerprint": _hash_value(assets),
        "numeric_source": numeric_source,
        "numeric_source_fingerprint": _hash_value(numeric_source),
    }
    path = out_root / "preflight" / "af3_identity_inputs.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != payload:
            raise RunnerError(
                f"identity preflight changed under the frozen capsule: {path}"
            )
    else:
        _atomic_json(path, payload)
    return {
        "asset_fingerprint": payload["asset_fingerprint"],
        "numeric_source_fingerprint": payload["numeric_source_fingerprint"],
        "path": str(path),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    deadline = started + int(args.wall_seconds)
    out_root = Path(args.out_root).expanduser().resolve()
    repo_root = Path(args.repo_root).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()
    sif_provenance = Path(args.sif_provenance).expanduser().resolve()
    capsule = _verify_capsule()

    _run_command(
        [
            sys.executable,
            str(HARNESS),
            "prepare",
            "--out-root",
            str(out_root),
            "--repo-root",
            str(repo_root),
            "--model-dir",
            str(model_dir),
            "--sif-provenance",
            str(sif_provenance),
        ],
        deadline=deadline,
        stage="prepare",
    )
    identity = _identity_preflight(
        out_root=out_root,
        repo_root=repo_root,
        capsule=capsule,
    )
    _run_command(
        [
            sys.executable,
            str(HARNESS),
            "preflight",
            "--out-root",
            str(out_root),
        ],
        deadline=deadline,
        stage="runtime-preflight",
    )
    runtime_preflight = json.loads(
        (out_root / "runtime_preflight.json").read_text(encoding="utf-8")
    )
    result: dict[str, Any] = {
        "capsule": capsule,
        "identity_preflight": identity,
        "out_root": str(out_root),
        "runtime_preflight": runtime_preflight,
        "prepare_only": bool(args.prepare_only),
    }
    if args.prepare_only:
        result["status"] = "prepared"
        return result

    remaining = max(1, int(deadline - time.monotonic()))
    _run_command(
        [
            sys.executable,
            str(HARNESS),
            "run",
            "--out-root",
            str(out_root),
            "--timeout-s",
            str(remaining),
        ],
        deadline=deadline,
        stage="run",
    )
    result["status"] = "complete"
    result["elapsed_seconds"] = time.monotonic() - started
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-root",
        default="/evidence/posthoc_public/attempt_002",
    )
    parser.add_argument("--repo-root", default="/opt/Astrolabe")
    parser.add_argument("--model-dir", default="/models/alphafold3")
    parser.add_argument(
        "--sif-provenance",
        default="/evidence/posthoc_public/sif_provenance.json",
    )
    parser.add_argument("--wall-seconds", type=int, default=3480)
    parser.add_argument("--prepare-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    result = run(_parser().parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
