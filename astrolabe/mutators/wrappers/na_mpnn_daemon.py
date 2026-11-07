from __future__ import annotations

from astrolabe.model_execution import KernelStages
from astrolabe.mutators.runtime_stages import (guarded_device, mutator_kernel_handler, device_resource, protocol_infer, torch_load_device)

import argparse
import hashlib
import importlib
import os
import re
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping


NA_MPNN_PIN = "835974f3d420de18b3eb1dc66f25fbd17c53cc09"
NA_MPNN_PATCH_SHA256 = (
    "218e01f2bc73b55f3ff6eedba268847febaf108485d8d5a1aa2f6423195ee668"
)
NA_MPNN_PATCHED_RUN_SHA256 = (
    "f83959cc5f365a53dadae93d6e46c60f232832aa21b731ea1a6a9331b28a0f2a"
)
_PATCH_PATH = (
    Path(__file__).resolve().parents[3]
    / "patches"
    / "vendor"
    / "na_mpnn"
    / "resident_api.patch"
)

_DEFAULTS: dict[str, Any] = {
    "mode": "design",
    "model_type": "na_mpnn",
    "file_ending": "",
    "fixed_pos_by_pdb": "",
    "zero_indexed": 0,
    "seed": 0,
    "save_stats": 0,
    "omit_AA": "X",
    "redesigned_residues": "",
    "bias_AA": "",
    "pair_bias_AA": "",
    "symmetry_residues": "",
    "symmetry_weights": "",
    "k_neighbors": None,
    "catch_failed_inferences": 0,
    "output_sequences": 1,
    "output_specificity": 1,
    "load_residues_with_missing_atoms": 0,
}

_INT_OPTIONS = {
    "zero_indexed",
    "seed",
    "save_stats",
    "na_shared_tokens",
    "parse_na_only",
    "design_na_only",
    "k_neighbors",
    "catch_failed_inferences",
    "output_pdbs",
    "output_sequences",
    "output_specificity",
    "load_residues_with_missing_atoms",
    "batch_size",
    "number_of_batches",
}
_FLOAT_OPTIONS = {"temperature"}
_OPTION_NAMES = {
    name
    for name in (
        *_DEFAULTS,
        "checkpoint_na_mpnn",
        "out_folder",
        "pdb_path",
        "chains_to_design",
        "fixed_residues",
        "parse_these_chains_only",
        "na_shared_tokens",
        "parse_na_only",
        "design_na_only",
        "output_pdbs",
        "batch_size",
        "number_of_batches",
        "temperature",
    )
}
_OPTION_LOOKUP = {name.lower(): name for name in _OPTION_NAMES}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_patch_file() -> None:
    if not _PATCH_PATH.is_file():
        raise FileNotFoundError(f"NA-MPNN resident patch is missing: {_PATCH_PATH}")
    actual = _sha256(_PATCH_PATH)
    if actual != NA_MPNN_PATCH_SHA256:
        raise RuntimeError(
            f"NA-MPNN resident patch SHA-256 is {actual}, "
            f"expected {NA_MPNN_PATCH_SHA256}."
        )


def _normalize_chain_list(value: Any) -> str | None:
    if not value:
        return None
    if isinstance(value, str):
        return value
    return ",".join(str(item).strip() for item in value if str(item).strip())


def _normalize_parse_chain_spec(value: Any) -> str | None:
    if not value:
        return None
    if isinstance(value, str):
        return value
    return "".join(str(item).strip() for item in value if str(item).strip())


def _normalize_fixed_residues(value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, str):
        return value
    return " ".join(str(item).strip() for item in value if str(item).strip())


def _safe_job_name(value: Any) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    if not name:
        raise ValueError("NA-MPNN job_name must not be empty.")
    return name


def _option_name(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    try:
        return _OPTION_LOOKUP[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported NA-MPNN option {value!r}.") from exc


def build_upstream_args(
    settings: Mapping[str, Any],
    payload: Mapping[str, Any] | None = None,
) -> argparse.Namespace:
    """Translate one daemon payload to the pinned upstream argument shape."""

    payload = payload or {}
    values = dict(_DEFAULTS)
    values.update(
        {
            "checkpoint_na_mpnn": str(settings.get("checkpoint") or ""),
            "out_folder": str(payload.get("job_dir") or settings.get("output_root") or ""),
            "pdb_path": str(payload.get("structure_path") or ""),
            "batch_size": int(settings.get("batch_size", 1)),
            "number_of_batches": int(settings.get("number_of_batches", 1)),
            "temperature": float(settings.get("temperature", 0.1)),
            "na_shared_tokens": int(settings.get("na_shared_tokens", 0)),
            "parse_na_only": int(settings.get("parse_na_only", 0)),
            "design_na_only": int(settings.get("design_na_only", 0)),
            "output_pdbs": int(settings.get("output_pdbs", 0)),
        }
    )

    for extra_args in (
        settings.get("extra_args"),
        payload.get("extra_args"),
    ):
        if extra_args is None:
            continue
        if not isinstance(extra_args, Mapping):
            raise TypeError("NA-MPNN extra_args must be a mapping")
        for raw_name, raw_value in dict(extra_args).items():
            if raw_value is None:
                continue
            name = _option_name(str(raw_name))
            if name in {
                "checkpoint_na_mpnn",
                "out_folder",
                "pdb_path",
                "mode",
                "model_type",
            }:
                raise ValueError(f"NA-MPNN option {raw_name!r} is managed by the daemon.")
            if name in _INT_OPTIONS:
                value: Any = int(raw_value)
            elif name in _FLOAT_OPTIONS:
                value = float(raw_value)
            else:
                value = str(raw_value)
            values[name] = value

    values.update(
        {
            "seed": int(payload.get("seed", settings.get("seed", 0))),
            "chains_to_design": _normalize_chain_list(payload.get("chains_to_design")),
            "fixed_residues": _normalize_fixed_residues(payload.get("fixed_residues")),
            "parse_these_chains_only": _normalize_parse_chain_spec(
                payload.get("parse_these_chains_only")
            ),
        }
    )

    return argparse.Namespace(**values)


def _git_head(repo_path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_path), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"NA-MPNN checkout is not a Git repository: {repo_path}")
    top_level = Path(result.stdout.strip()).resolve()
    if top_level != repo_path:
        raise RuntimeError(
            f"NA-MPNN path is not a Git root: {repo_path}; resolved to {top_level}"
        )
    head = subprocess.run(
        ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return head.stdout.strip()


def verify_vendor(repo_path: str | os.PathLike[str]) -> Path:
    repo = Path(repo_path).expanduser().resolve()
    source = repo / "inference" / "run.py"
    if not source.is_file():
        raise FileNotFoundError(f"NA-MPNN source is missing: {source}")
    _verify_patch_file()
    head = _git_head(repo)
    if head != NA_MPNN_PIN:
        raise RuntimeError(
            f"NA-MPNN checkout is {head}, expected {NA_MPNN_PIN}."
        )
    text = source.read_text(encoding="utf-8")
    source_sha256 = _sha256(source)
    if source_sha256 != NA_MPNN_PATCHED_RUN_SHA256:
        raise RuntimeError(
            f"NA-MPNN inference/run.py SHA-256 is {source_sha256}, "
            f"expected {NA_MPNN_PATCHED_RUN_SHA256}."
        )
    missing = [
        marker
        for marker in ("def load_model(", "def run_design(", "_resident_model")
        if marker not in text
    ]
    if missing:
        raise RuntimeError(
            "NA-MPNN resident API is not installed. Apply "
            f"{_PATCH_PATH} to the pinned checkout."
        )
    return repo


def apply_vendor_patch(
    repo_path: str | os.PathLike[str],
    *,
    check_only: bool = False,
) -> None:
    repo = Path(repo_path).expanduser().resolve()
    _verify_patch_file()
    head = _git_head(repo)
    if head != NA_MPNN_PIN:
        raise RuntimeError(
            f"NA-MPNN checkout is {head}, expected {NA_MPNN_PIN}."
        )
    forward_check = subprocess.run(
        ["git", "-C", str(repo), "apply", "--check", str(_PATCH_PATH)],
        check=False,
        capture_output=True,
        text=True,
    )
    if forward_check.returncode == 0:
        if check_only:
            return
        subprocess.run(
            ["git", "-C", str(repo), "apply", str(_PATCH_PATH)],
            check=True,
        )
        verify_vendor(repo)
        return
    reverse_check = subprocess.run(
        ["git", "-C", str(repo), "apply", "--reverse", "--check", str(_PATCH_PATH)],
        check=False,
        capture_output=True,
        text=True,
    )
    if reverse_check.returncode == 0:
        verify_vendor(repo)
        return
    detail = (forward_check.stderr or reverse_check.stderr).strip()
    raise RuntimeError(f"NA-MPNN resident patch does not apply: {detail}")


class NAMPNNHandler:
    @guarded_device("na_mpnn", phase="load")
    def __init__(
        self,
        settings: Mapping[str, Any],
        *,
        upstream: Any | None = None,
    ) -> None:
        self.settings = dict(settings)
        self.upstream = upstream or self._load_upstream()
        self._initial_output = tempfile.TemporaryDirectory(
            prefix="astrolabe-na-mpnn-"
        )
        load_args = build_upstream_args(
            self.settings,
            {"job_dir": self._initial_output.name},
        )
        self.resident = self.upstream.load_model(load_args)
        self.device = "unknown"
        if isinstance(self.resident, Mapping) and self.resident.get("model") is not None:
            parameter = next(iter(self.resident["model"].parameters()), None)
            if parameter is not None:
                self.device = str(parameter.device)

    def _load_upstream(self) -> Any:
        verify_vendor(self.settings["repo_path"])
        return importlib.import_module("inference.run")

    def prepare_design(self, payload):
        return build_upstream_args(self.settings, payload)

    @guarded_device("na_mpnn")
    def execute_design(self, args):
        # The vendor run_design includes device featurization and seeded native
        # decoding. Its initializer RNG draws are intentional and unchanged.
        self.upstream.run_design(args, self.resident)
        return args

    @staticmethod
    def finalize_design(args):
        return {"job_dir": args.out_folder, "structure_path": args.pdb_path}

    def __call__(self, payload: Mapping[str, Any]) -> dict[str, str]:
        return self.finalize_design(self.execute_design(self.prepare_design(payload)))


def _load_handler(settings):
    runner = NAMPNNHandler(settings)
    return KernelStages(runner.prepare_design, runner.execute_design, runner.finalize_design,
                        resource=device_resource(runner.device)), {
        "backend": "na_mpnn", "device": runner.device,
        "checkpoint": str(settings.get("checkpoint", "")),
    }


def create_handler(settings: Mapping[str, Any]):
    handler = mutator_kernel_handler(dict(settings), _load_handler,
                                     backend="na_mpnn", torch_device=torch_load_device(cpu_fallback=True))
    def infer(payload):
        if payload.get("operation") in {"describe", "infer"}:
            return handler(payload)
        result = protocol_infer(handler, payload)
        if payload.get("_runtime_report"):
            return {**result, "runtime": handler.last_runtime,
                    "metadata": handler({"operation": "describe"})["metadata"]}
        return result
    return infer


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_path")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.apply:
        apply_vendor_patch(args.repo_path, check_only=args.check)
    else:
        verify_vendor(args.repo_path)


if __name__ == "__main__":
    _main()
