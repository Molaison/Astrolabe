#!/usr/bin/env python3
"""Run the isolated acceptance with an explicit current-runtime root-pair profile."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Any, Mapping, Sequence


PROFILE_NAME = "current_runtime_paired_v1"
PROFILE_SCHEMA = "astrolabe.current-runtime-paired-profile/v1"
REPO_ROOT = Path(__file__).resolve().parents[2]


class ProfileError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _ordered_unique(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def _load_harness() -> ModuleType:
    candidates = []
    configured = os.environ.get("ASTROLABE_ISOLATED_HARNESS")
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        [
            Path("/opt/Astrolabe/scripts/validation/isolated_gpu_acceptance.py"),
            REPO_ROOT / "scripts" / "validation" / "isolated_gpu_acceptance.py",
        ]
    )
    path = next((item for item in candidates if item.is_file()), None)
    if path is None:
        raise ProfileError("isolated_gpu_acceptance.py was not found")
    name = "astrolabe_isolated_acceptance_current_pair"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ProfileError(f"cannot load isolated acceptance harness: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _omega_container(path: Path) -> Any:
    from omegaconf import OmegaConf

    return OmegaConf.to_container(OmegaConf.load(path), resolve=False)


def _leaf_diffs(
    left: Any,
    right: Any,
    prefix: tuple[str, ...] = (),
) -> list[tuple[str, Any, Any]]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        result = []
        for key in sorted(set(left) | set(right), key=str):
            if key not in left:
                result.append((".".join((*prefix, str(key))), "<missing>", right[key]))
            elif key not in right:
                result.append((".".join((*prefix, str(key))), left[key], "<missing>"))
            else:
                result.extend(_leaf_diffs(left[key], right[key], (*prefix, str(key))))
        return result
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [(".".join(prefix), left, right)]
        result = []
        for index, (a, b) in enumerate(zip(left, right)):
            result.extend(_leaf_diffs(a, b, (*prefix, str(index))))
        return result
    if left != right:
        return [(".".join(prefix), left, right)]
    return []


def _require_sha(profile: Mapping[str, Any], key: str, path: Path) -> dict[str, str]:
    entry = profile.get("assets", {}).get(key)
    if not isinstance(entry, Mapping):
        raise ProfileError(f"profile assets.{key} is missing")
    expected_path = Path(str(entry.get("path", ""))).expanduser().resolve()
    actual_path = path.expanduser().resolve()
    if expected_path != actual_path:
        raise ProfileError(f"assets.{key} path changed: {actual_path} != {expected_path}")
    if not actual_path.is_file():
        raise ProfileError(f"assets.{key} is unavailable: {actual_path}")
    actual = _sha256(actual_path)
    if actual != entry.get("sha256"):
        raise ProfileError(
            f"assets.{key} hash changed: {actual} != {entry.get('sha256')}"
        )
    return {"path": str(actual_path), "sha256": actual}


def _validate_static(
    profile_path: Path,
    *,
    out_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    profile_path = profile_path.expanduser().resolve()
    profile = _read_json(profile_path)
    if profile.get("schema") != PROFILE_SCHEMA:
        raise ProfileError("profile schema mismatch")
    if profile.get("profile") != PROFILE_NAME:
        raise ProfileError(f"profile must be {PROFILE_NAME}")

    reference = profile.get("fresh_pair", {}).get("first_half")
    if not isinstance(reference, Mapping):
        raise ProfileError("fresh_pair.first_half is missing")
    root_result_path = Path(str(reference["root_result"])).expanduser().resolve()
    root_result = _read_json(root_result_path)
    if (
        root_result.get("id") != 0
        or root_result.get("parent_id") is not None
        or root_result.get("status") != "done"
        or root_result.get("evaluation_origin") != "fresh_af3"
    ):
        raise ProfileError("first pair half is not a completed fresh root")
    if root_result_path == out_root.expanduser().resolve():
        raise ProfileError("fresh pair cannot compare a root to itself")
    if out_root.expanduser().resolve() in root_result_path.parents:
        raise ProfileError("fresh pair reference cannot reside under the new execution root")

    asset_paths = {
        "root_result": root_result_path,
        "root_structure": Path(str(root_result["structure_path"])),
        "root_snapshot": Path(str(root_result["snapshot"])),
        "root_confidence": Path(str(reference["confidence"])),
        "root_owner_log": Path(str(reference["owner_log"])),
        "root_call_ledger": Path(str(reference["call_ledger"])),
        "root_environment": Path(str(reference["environment"])),
        "root_source_manifest": Path(str(reference["source_manifest"])),
        "weights": Path(str(profile["runtime"]["weights_path"])),
        "target_structure": Path(str(profile["inputs"]["target_structure"])),
        "target_homology": Path(str(profile["inputs"]["target_homology"])),
        "scientific_template_config": Path(
            str(profile["scientific_template"]["config"])
        ),
        "profile_case_config": Path(str(profile["scientific_template"]["profile_case"])),
        "validation_driver": Path(str(profile["assets"]["validation_driver"]["path"])),
    }
    verified_assets = {
        key: _require_sha(profile, key, path)
        for key, path in asset_paths.items()
    }
    owner_events = []
    for line in asset_paths["root_owner_log"].read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            owner_events.append(value)
    owner_counts = {
        event: sum(1 for item in owner_events if item.get("event") == event)
        for event in ("request_received", "gpu_start", "gpu_end")
    }
    if owner_counts != profile["fresh_pair"]["expected_owner_events"]:
        raise ProfileError(
            f"fresh root owner event counts changed: {owner_counts} != "
            f"{profile['fresh_pair']['expected_owner_events']}"
        )
    if not any(
        item.get("event") == "gpu_end"
        and item.get("ok") is True
        and item.get("request_id") == 1
        for item in owner_events
    ):
        raise ProfileError("fresh root lacks successful GPU terminal evidence")
    ledger = _read_json(asset_paths["root_call_ledger"])
    entries = ledger.get("entries") or {}
    if len(entries) != 1 or next(iter(entries.values())).get("status") != "issued":
        raise ProfileError("fresh root ledger must retain exactly one issued request")

    profile_case = _omega_container(asset_paths["profile_case_config"])
    template = _omega_container(asset_paths["scientific_template_config"])
    differences = _leaf_diffs(template, profile_case)
    expected_path = "protein_beam.root_parity_reference"
    if len(differences) != 1 or differences[0][0] != expected_path:
        raise ProfileError(
            "profile case must differ from the current scientific template only at "
            f"{expected_path}; got {differences!r}"
        )
    configured_reference = (
        profile_case.get("protein_beam", {}).get("root_parity_reference")
    )
    if configured_reference != str(root_result_path):
        raise ProfileError(
            "profile root_parity_reference does not name the fresh first half"
        )
    historical = Path(str(profile["historical_reference"]["root_result"])).resolve()
    if configured_reference == str(historical):
        raise ProfileError("profile must not reuse the historical root reference")

    return profile, {
        "profile_manifest": {
            "path": str(profile_path),
            "sha256": _sha256(profile_path),
        },
        "fresh_pair_root": {
            "root_result": str(root_result_path),
            "sequence_sha256": hashlib.sha256(
                str(root_result["sequence"]).encode()
            ).hexdigest(),
            "seed": int(root_result["seed"]),
            "af3_call_token": root_result.get("af3_call_token"),
            "metrics": root_result["metrics"],
            "owner_event_counts": owner_counts,
            "call_ledger_entries": len(entries),
        },
        "assets": verified_assets,
        "scientific_config_differences": [
            {"path": path, "template": old, "profile": new}
            for path, old, new in differences
        ],
        "historical_reference": {
            **profile["historical_reference"],
            "strict_parity_passed": False,
            "excluded_from_current_profile": True,
        },
        "self_comparison": False,
    }


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as error:
        raise ProfileError(f"required package is missing: {name}") from error


def _load_library(name: str) -> str:
    try:
        library = ctypes.CDLL(name)
    except OSError as error:
        raise ProfileError(f"loader cannot resolve {name}: {error}") from error
    return getattr(library, "_name", name)


def _gpu_identity() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=uuid,name,driver_version,memory.total",
        "--format=csv,noheader",
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise ProfileError(f"nvidia-smi failed: {result.stderr.strip()}")
    values = [item.strip() for item in result.stdout.strip().split(",")]
    if len(values) != 4:
        raise ProfileError(f"unexpected nvidia-smi output: {result.stdout!r}")
    return {
        "uuid": values[0],
        "name": values[1],
        "driver_version": values[2],
        "memory_total": values[3],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _validate_runtime(
    profile: Mapping[str, Any],
    harness: ModuleType,
    *,
    check_gpu: bool,
) -> dict[str, Any]:
    runtime = profile["runtime"]
    expected_ld = [
        str(item) for item in runtime["ld_library_path"]["required_entries"]
    ]
    actual_ld = _ordered_unique(os.environ.get("LD_LIBRARY_PATH", "").split(":"))
    positions = [actual_ld.index(item) if item in actual_ld else -1 for item in expected_ld]
    if -1 in positions or positions != sorted(positions):
        raise ProfileError(
            f"LD_LIBRARY_PATH does not preserve required overlay order: {actual_ld!r}"
        )

    image_name = Path(str(os.environ.get("APPTAINER_NAME", ""))).name
    if image_name != runtime["image_name"]:
        raise ProfileError(
            f"container image mismatch: {image_name} != {runtime['image_name']}"
        )
    image_path = str(os.environ.get("APPTAINER_CONTAINER", ""))
    if image_path != runtime["image_path"]:
        raise ProfileError(
            f"container image path mismatch: {image_path!r} != {runtime['image_path']!r}"
        )
    if Path(sys.executable).resolve() != Path(runtime["python"]).resolve():
        raise ProfileError(f"python executable mismatch: {sys.executable}")

    package_versions = {
        package: _package_version(package)
        for package in runtime["python_packages"]
    }
    nvrtc_name = _load_library("libnvrtc.so.12")
    try:
        nvrtc = ctypes.CDLL("libnvrtc.so.12")
        major = ctypes.c_int()
        minor = ctypes.c_int()
        status = nvrtc.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor))
        if status != 0:
            raise ProfileError(f"nvrtcVersion failed with status {status}")
        observed_nvrtc = f"{major.value}.{minor.value}"
    except OSError as error:
        raise ProfileError(f"loader cannot resolve libnvrtc.so.12: {error}") from error
    cudnn_name = _load_library("libcudnn.so.9")
    cudnn = ctypes.CDLL("libcudnn.so.9")
    cudnn.cudnnGetVersion.restype = ctypes.c_size_t
    cudnn_version = int(cudnn.cudnnGetVersion())
    _load_library("libnvrtc-builtins.so.12.6")
    _load_library("libcudnn_engines_runtime_compiled.so.9")
    if observed_nvrtc != runtime["libraries"]["nvrtc"]:
        raise ProfileError(
            f"NVTRC version mismatch: {observed_nvrtc} != {runtime['libraries']['nvrtc']}"
        )
    encoded_cudnn = int(runtime["libraries"]["cudnn_encoded"])
    if cudnn_version != encoded_cudnn:
        raise ProfileError(
            f"cuDNN version mismatch: {cudnn_version} != {encoded_cudnn}"
        )

    fingerprint = harness._source_fingerprint(harness._source_manifest())
    if fingerprint != profile["source"]["fingerprint"]:
        raise ProfileError(
            f"frozen source fingerprint changed: {fingerprint} != "
            f"{profile['source']['fingerprint']}"
        )
    gpu = _gpu_identity() if check_gpu else None
    if gpu is not None:
        expected_gpu = runtime["gpu"]
        for key in ("uuid", "name", "driver_version", "memory_total"):
            if gpu[key] != expected_gpu[key]:
                raise ProfileError(
                    f"GPU {key} mismatch: {gpu[key]!r} != {expected_gpu[key]!r}"
                )
    return {
        "image_name": image_name,
        "python": str(Path(sys.executable).resolve()),
        "ld_library_path": actual_ld,
        "python_packages": package_versions,
        "libraries": {
            "nvrtc": observed_nvrtc,
            "nvrtc_library": nvrtc_name,
            "cudnn_encoded": cudnn_version,
            "cudnn_library": cudnn_name,
        },
        "source_fingerprint": fingerprint,
        "gpu": gpu,
    }


def _profile_record(
    *,
    profile_path: Path,
    phase: str,
    static: Mapping[str, Any],
    runtime: Mapping[str, Any],
    prepared: Any,
) -> dict[str, Any]:
    return {
        "schema": "astrolabe.current-runtime-paired-evidence/v1",
        "profile": PROFILE_NAME,
        "phase": phase,
        "profile_manifest": str(profile_path),
        "case_config": str(prepared.case_config),
        "source_fingerprint": prepared.source_fingerprint,
        "static": dict(static),
        "runtime": dict(runtime),
        "strict_parity_is_bitwise": True,
        "historical_reference_claimed_passed": False,
        "reference_files_mutable_by_runner": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-manifest", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=10800.0)
    arguments = parser.parse_args(argv)

    harness = _load_harness()
    profile_path = arguments.profile_manifest.expanduser().resolve()
    out_root = arguments.out_root.expanduser().resolve()
    profile, static = _validate_static(profile_path, out_root=out_root)
    root_key = "preflight_root" if arguments.prepare_only else "execution_root"
    expected_root = Path(str(profile.get(root_key, ""))).expanduser().resolve()
    if expected_root != out_root:
        raise ProfileError(
            f"{root_key} mismatch: {out_root} != {expected_root}"
        )
    runtime = _validate_runtime(profile, harness, check_gpu=not arguments.prepare_only)
    prepared = harness._prepare_acceptance(
        Path(str(profile["scientific_template"]["profile_case"])),
        out_root,
    )
    record = _profile_record(
        profile_path=profile_path,
        phase="preflight" if arguments.prepare_only else "execution",
        static=static,
        runtime=runtime,
        prepared=prepared,
    )
    harness._atomic_json(out_root / "verification_profile.json", record)
    if arguments.prepare_only:
        print(json.dumps({"status": "prepared", "profile": str(out_root / "verification_profile.json")}))
        return 0
    try:
        summary = harness._execute_acceptance(
            prepared,
            timeout_s=arguments.timeout_s,
        )
    except BaseException as error:
        try:
            harness._record_failure(out_root, error)
        except BaseException as evidence_error:
            error.add_note(
                f"Current-paired failure evidence could not be written: {evidence_error!r}"
            )
        raise
    print(
        json.dumps(
            {
                "status": summary["status"],
                "summary": str(out_root / "summary.json"),
                "verification_profile": str(out_root / "verification_profile.json"),
            }
        )
    )
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
