#!/usr/bin/env python3
"""Measure real, seeded ProteinMPNN residency and retain parity evidence.

This harness intentionally performs no CPU stand-in for model execution.  It
constructs one in-process model and one daemon-owned model with the same
checkpoint, input, seed, and request order, then compares their raw
``design_sequences`` outputs exactly.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = Path("/models/ligandmpnn/proteinmpnn_v_48_020.pt")
DEFAULT_INPUT = ROOT / "examples" / "input" / "9TP_cid.cif"
CHECKPOINT_NAME = "proteinmpnn_v_48_020.pt"
SCHEMA = "astrolabe.proteinmpnn-gpu-throughput/v1"
REPORT_NAME = "throughput.json"
DAEMON_FACTORY = "scripts.validation.model_gpu_throughput:daemon_factory"


class HarnessError(RuntimeError):
    """The requested GPU evidence cannot be produced safely."""


@dataclass(frozen=True)
class TimingSummary:
    model_init_s: float
    request_s: tuple[float, ...]
    complete_cold_start_s: float
    warm_total_s: float
    warm_requests_per_s: float
    first_three_warm_total_s: float | None
    first_three_warm_requests_per_s: float | None


def _sha256(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise HarnessError("object-dtype evidence cannot be serialized")
        return value.tolist()
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


def resolve_checkpoint(weights: Path) -> Path:
    path = weights.expanduser().resolve()
    checkpoint = path / CHECKPOINT_NAME if path.is_dir() else path
    if not checkpoint.is_file():
        raise HarnessError(f"ProteinMPNN checkpoint is unavailable: {checkpoint}")
    return checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Absent or empty directory for the report and raw comparison arrays.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help=f"Checkpoint file or directory containing {CHECKPOINT_NAME}.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--chain", default="A")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--requests",
        type=int,
        default=4,
        help="Sequential requests per resident owner; must be at least 2.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.requests < 2:
        parser.error("--requests must be at least 2")
    if args.seed == 0:
        parser.error("--seed must be nonzero for exact RNG continuation evidence")
    if not args.chain:
        parser.error("--chain must not be empty")
    args.input = args.input.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    if not args.input.is_file():
        parser.error(f"Input structure is unavailable: {args.input}")
    try:
        args.weights = resolve_checkpoint(args.weights)
    except HarnessError as error:
        parser.error(str(error))
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error(f"--out-dir must be absent or empty: {args.out_dir}")
    return args


def aggregate_timings(
    model_init_s: float,
    request_s: Sequence[float],
) -> TimingSummary:
    if len(request_s) < 2:
        raise ValueError("at least two requests are required for cold and warm timing")
    values = tuple(float(value) for value in request_s)
    init = float(model_init_s)
    if init < 0 or any(value < 0 for value in values):
        raise ValueError("timings must be nonnegative")
    warm_total = math.fsum(values[1:])
    if warm_total <= 0:
        raise ValueError("warm-call duration must be positive")
    first_three_total: float | None = None
    first_three_rate: float | None = None
    if len(values) >= 4:
        first_three_total = math.fsum(values[1:4])
        if first_three_total <= 0:
            raise ValueError("three-warm duration must be positive")
        first_three_rate = 3 / first_three_total
    return TimingSummary(
        model_init_s=init,
        request_s=values,
        complete_cold_start_s=math.fsum((init, *values)),
        warm_total_s=warm_total,
        warm_requests_per_s=(len(values) - 1) / warm_total,
        first_three_warm_total_s=first_three_total,
        first_three_warm_requests_per_s=first_three_rate,
    )


def _measure(
    call: Callable[[], Any],
    *,
    synchronize: Callable[[], None],
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[Any, float]:
    """Time one blocking boundary after forcing pending CUDA work to finish."""
    synchronize()
    started = clock()
    try:
        value = call()
    finally:
        synchronize()
    return value, clock() - started


@contextmanager
def counted_torch_load(torch_module: Any, reads: list[str]):
    """Wrap the real loader, preserving its arguments and return value."""
    original = torch_module.load

    def loader(path: Any, *args: Any, **kwargs: Any) -> Any:
        reads.append(os.fspath(path))
        return original(path, *args, **kwargs)

    torch_module.load = loader
    try:
        yield reads
    finally:
        torch_module.load = original


def _daemon_factory(settings: Mapping[str, Any]) -> Callable[[dict[str, Any]], Any]:
    """Load one real ProteinMPNN owner in the daemon process."""
    import torch
    from astrolabe.model_execution import kernel_handler
    from astrolabe.mutators.native_daemons import _load_ligandmpnn

    if not torch.cuda.is_available():
        raise HarnessError("daemon ProteinMPNN owner requires CUDA")
    checkpoint_reads: list[str] = []
    torch.cuda.reset_peak_memory_stats()

    def loader(config: Mapping[str, Any]):
        with counted_torch_load(torch, checkpoint_reads):
            infer, metadata = _load_ligandmpnn(config)
        runner = inspect.getclosurevars(infer).nonlocals["runner"]
        torch.cuda.synchronize()
        context = {
            "device": str(runner.device),
            "pid": os.getpid(),
            "model_token": id(runner.model),
            "checkpoint_reads": list(checkpoint_reads),
            "cuda_peak_after_init": {
                "allocated_bytes": torch.cuda.max_memory_allocated(),
                "reserved_bytes": torch.cuda.max_memory_reserved(),
            },
        }

        def predict(payload: dict[str, Any]) -> dict[str, Any]:
            output = infer(payload)
            torch.cuda.synchronize()
            return {
                "output": output,
                "device": str(runner.device),
                "pid": os.getpid(),
                "model_token": id(runner.model),
                "checkpoint_reads": list(checkpoint_reads),
                "load_count": len(checkpoint_reads),
                "cuda_peak": {
                    "allocated_bytes": torch.cuda.max_memory_allocated(),
                    "reserved_bytes": torch.cuda.max_memory_reserved(),
                },
            }

        metadata.update(context)
        return predict, metadata

    return kernel_handler(settings, loader, torch_device="cuda")


def daemon_factory(settings: dict[str, Any]) -> Callable[[dict[str, Any]], Any]:
    """Stable import target used by NativeModelClient's child process."""
    return _daemon_factory(settings)


def _reset_rng(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _random_state_digests(states: Sequence[Mapping[str, Any]]) -> list[str]:
    return [
        hashlib.sha256(
            json.dumps(
                state,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        for state in states
    ]


def _prepare_pdb(*, source: Path, chain: str, destination: Path) -> Path:
    from Bio.PDB import MMCIFParser, PDBIO, PDBParser, Select

    parser = MMCIFParser(QUIET=True) if source.suffix.lower() == ".cif" else PDBParser(QUIET=True)
    structure = parser.get_structure("throughput", str(source))
    if chain not in {item.id for item in structure.get_chains()}:
        raise HarnessError(f"Input {source} has no chain {chain!r}")

    class ChainSelection(Select):
        def accept_chain(self, selected_chain) -> bool:
            return selected_chain.id == chain

    writer = PDBIO()
    writer.set_structure(structure)
    writer.save(str(destination), select=ChainSelection())
    if not any(
        line.startswith(("ATOM", "HETATM"))
        for line in destination.read_text(encoding="utf-8").splitlines()
    ):
        raise HarnessError(f"No atoms were written for chain {chain!r}")
    return destination


def _model_config(*, seed: int, checkpoint: Path, output_dir: Path) -> dict[str, Any]:
    from omegaconf import OmegaConf

    cfg = OmegaConf.merge(
        OmegaConf.load(ROOT / "config" / "env" / "env.yaml"),
        OmegaConf.load(ROOT / "config" / "base.yaml"),
    )
    cfg.seed = seed
    cfg.exclude_AA = ""
    cfg.weight_path.ligandmpnn = str(checkpoint.parent)
    cfg.mutator.LigandMPNN.model_type = "protein_mpnn"
    cfg.mutator.LigandMPNN.out_folder = str(output_dir)
    config = OmegaConf.to_container(cfg.mutator.LigandMPNN, resolve=True)
    if not isinstance(config, dict):
        raise HarnessError("ProteinMPNN configuration did not resolve to a mapping")
    config["checkpoint_protein_mpnn"] = str(checkpoint)
    return config


def _source_identity() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        ROOT / "config" / "env" / "env.yaml",
        ROOT / "config" / "base.yaml",
        ROOT / "astrolabe" / "model_execution.py",
        ROOT / "astrolabe" / "model_daemon.py",
        ROOT / "astrolabe" / "mutators" / "native_daemons.py",
        ROOT / "astrolabe" / "mutators" / "wrappers" / "ligandmpnn_run.py",
        ROOT / "astrolabe" / "mutators" / "wrappers" / "ligandmpnn_utils.py",
    )
    return {
        str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path):
        _sha256(path)
        for path in paths
    }


def _driver_version() -> str | None:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    completed = subprocess.run(
        [executable, "--query-gpu=driver_version", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    values = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return values[0] if len(set(values)) == 1 and values else None


def _gpu_facts(torch_module: Any) -> dict[str, Any]:
    properties = torch_module.cuda.get_device_properties(torch_module.cuda.current_device())
    return {
        "torch_version": torch_module.__version__,
        "torch_cuda_version": torch_module.version.cuda,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "current_device": torch_module.cuda.current_device(),
        "name": properties.name,
        "major": properties.major,
        "minor": properties.minor,
        "total_memory_bytes": properties.total_memory,
        "driver_version": _driver_version(),
    }


def _comparable(value: Any) -> tuple[Any, Any]:
    """Convert output values to an exact, detached evidence representation."""
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy().copy()
        return "array", array
    if isinstance(value, np.ndarray):
        return "array", value.copy()
    if isinstance(value, np.generic):
        return "numpy_scalar", value.copy()
    if isinstance(value, Mapping):
        return "mapping", sorted((str(key), _comparable(item)) for key, item in value.items())
    if isinstance(value, tuple):
        return "tuple", tuple(_comparable(item) for item in value)
    if isinstance(value, list):
        return "list", tuple(_comparable(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return type(value).__name__, value
    raise HarnessError(
        f"Unsupported ProteinMPNN output value {type(value).__module__}.{type(value).__name__}"
    )


def _array_equal(expected: np.ndarray, actual: np.ndarray) -> bool:
    if expected.dtype != actual.dtype or expected.shape != actual.shape:
        return False
    if expected.dtype.kind == "O":
        return all(
            _object_value_equal(left, right)
            for left, right in zip(expected.ravel(), actual.ravel())
        )
    if expected.dtype.kind in {"f", "c"}:
        return np.array_equal(expected, actual, equal_nan=True)
    return np.array_equal(expected, actual)


def _object_value_equal(expected: Any, actual: Any) -> bool:
    """Compare explicit Python values stored inside an object ndarray."""
    if isinstance(expected, np.ndarray) or isinstance(actual, np.ndarray):
        return (
            isinstance(expected, np.ndarray)
            and isinstance(actual, np.ndarray)
            and _array_equal(expected, actual)
        )
    if isinstance(expected, np.generic) or isinstance(actual, np.generic):
        if not (
            isinstance(expected, np.generic)
            and isinstance(actual, np.generic)
            and expected.dtype == actual.dtype
        ):
            return False
        if expected.dtype.kind in {"f", "c"}:
            return bool(np.array_equal(expected, actual, equal_nan=True))
        return bool(np.array_equal(expected, actual))
    if isinstance(expected, Mapping) or isinstance(actual, Mapping):
        if not isinstance(expected, Mapping) or not isinstance(actual, Mapping):
            return False
        if expected.keys() != actual.keys():
            return False
        return all(
            _object_value_equal(expected[key], actual[key])
            for key in expected
        )
    if isinstance(expected, (tuple, list)) or isinstance(actual, (tuple, list)):
        if type(expected) is not type(actual):
            return False
        return len(expected) == len(actual) and all(
            _object_value_equal(left, right)
            for left, right in zip(expected, actual)
        )
    if isinstance(expected, complex) or isinstance(actual, complex):
        if not isinstance(expected, complex) or not isinstance(actual, complex):
            return False
        return _complex_equal(expected, actual)
    if isinstance(expected, float) or isinstance(actual, float):
        if not isinstance(expected, float) or not isinstance(actual, float):
            return False
        return expected == actual or (math.isnan(expected) and math.isnan(actual))
    if expected is None or isinstance(expected, (str, int, bool)):
        return type(expected) is type(actual) and expected == actual
    raise HarnessError(
        f"Unsupported object-array value {type(expected).__module__}."
        f"{type(expected).__name__}"
    )


def _complex_equal(expected: complex, actual: complex) -> bool:
    return (
        expected == actual
        or (
            (expected.real == actual.real or (
                math.isnan(expected.real) and math.isnan(actual.real)
            ))
            and (expected.imag == actual.imag or (
                math.isnan(expected.imag) and math.isnan(actual.imag)
            ))
        )
    )


def _value_digest(value: Any) -> str:
    kind, payload = _comparable(value)
    digest = hashlib.sha256()
    digest.update(kind.encode("ascii"))
    digest.update(b"\0")
    if kind == "array":
        digest.update(payload.dtype.str.encode("ascii"))
        digest.update(json.dumps(payload.shape).encode("ascii"))
        if payload.dtype.kind == "O":
            digest.update(
                json.dumps(
                    _plain_exact_value(payload),
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=True,
                ).encode("utf-8")
            )
        else:
            digest.update(np.ascontiguousarray(payload).tobytes())
    elif kind == "numpy_scalar":
        digest.update(payload.dtype.str.encode("ascii"))
        digest.update(payload.tobytes())
    else:
        digest.update(
            json.dumps(
                _plain_exact_value(value),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=True,
            ).encode("utf-8")
        )
    return digest.hexdigest()


def _plain_exact_value(value: Any) -> Any:
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
        return {
            "$array": True,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
        }
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "O":
            return {
                "$array": True,
                "dtype": value.dtype.str,
                "shape": list(value.shape),
                "values": [
                    _plain_exact_value(item)
                    for item in value.ravel()
                ],
            }
        return {
            "$array": True,
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
        }
    if isinstance(value, np.generic):
        return {
            "$numpy_scalar": True,
            "dtype": value.dtype.str,
            "value": value.item(),
        }
    if isinstance(value, Mapping):
        return {
            str(key): _plain_exact_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_plain_exact_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise HarnessError(
        f"Unsupported ProteinMPNN output value {type(value).__module__}.{type(value).__name__}"
    )


def compare_outputs(expected: Any, actual: Any) -> dict[str, Any]:
    differences: list[dict[str, str]] = []
    expected_digests: dict[str, str] = {}
    actual_digests: dict[str, str] = {}

    def visit(left: Any, right: Any, path: str) -> None:
        expected_digests[path] = _value_digest(left)
        actual_digests[path] = _value_digest(right)
        left_kind, left_value = _comparable(left)
        right_kind, right_value = _comparable(right)
        if left_kind != right_kind:
            differences.append({"path": path, "reason": "type"})
            return
        if left_kind in {"array", "numpy_scalar"}:
            if left_value.dtype != right_value.dtype:
                differences.append({"path": path, "reason": "dtype"})
            elif left_value.shape != right_value.shape:
                differences.append({"path": path, "reason": "shape"})
            elif left_kind == "array" and not _array_equal(left_value, right_value):
                differences.append({"path": path, "reason": "value"})
            elif left_kind == "numpy_scalar" and not (
                np.array_equal(left_value, right_value, equal_nan=True)
            ):
                differences.append({"path": path, "reason": "value"})
            return
        if left_kind == "mapping":
            left_items = {
                str(key): item
                for key, item in sorted(left.items(), key=lambda pair: str(pair[0]))
            }
            right_items = {
                str(key): item
                for key, item in sorted(right.items(), key=lambda pair: str(pair[0]))
            }
            left_keys = list(left_items)
            right_keys = list(right_items)
            if left_keys != right_keys:
                differences.append({"path": path, "reason": "keys"})
                return
            for key in left_keys:
                visit(left_items[key], right_items[key], f"{path}.{key}")
            return
        if left_kind in {"tuple", "list"}:
            if len(left) != len(right):
                differences.append({"path": path, "reason": "length"})
                return
            for index, (left_item, right_item) in enumerate(zip(left, right)):
                visit(left_item, right_item, f"{path}[{index}]")
            return
        if isinstance(left_value, float) and isinstance(right_value, float):
            if not (
                left_value == right_value
                or (math.isnan(left_value) and math.isnan(right_value))
            ):
                differences.append({"path": path, "reason": "value"})
        elif left_value != right_value:
            differences.append({"path": path, "reason": "value"})

    visit(expected, actual, "$")
    return {
        "equal": not differences,
        "differences": differences,
        "expected_sha256": expected_digests,
        "actual_sha256": actual_digests,
    }


def _flatten_arrays(result: Any) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}

    def visit(value: Any, path: str) -> None:
        kind, payload = _comparable(value)
        if kind in {"array", "numpy_scalar"}:
            arrays[path] = payload
        elif kind == "mapping":
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
                visit(item, f"{path}.{key}")
        elif kind in {"tuple", "list"}:
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")

    visit(result, "output")
    return arrays


def _timing_report(summary: TimingSummary) -> dict[str, Any]:
    return {
        "model_init_s": summary.model_init_s,
        "request_s": list(summary.request_s),
        "cold_first_request_s": summary.request_s[0],
        "warm_request_s": list(summary.request_s[1:]),
        "complete_cold_start_s": summary.complete_cold_start_s,
        "warm_total_s": summary.warm_total_s,
        "warm_requests_per_s": summary.warm_requests_per_s,
        "first_three_warm_total_s": summary.first_three_warm_total_s,
        "first_three_warm_requests_per_s": (
            summary.first_three_warm_requests_per_s
        ),
    }


def _assert_exact_trace(
    reads: Sequence[str],
    checkpoint: Path,
    *,
    lane: str,
) -> None:
    normalized = [Path(path).expanduser().resolve() for path in reads]
    if normalized != [checkpoint]:
        raise HarnessError(
            f"{lane} checkpoint load trace must contain exactly {checkpoint}; "
            f"observed {normalized}"
        )


def _run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from omegaconf import OmegaConf

    from astrolabe.model_execution import (
        NativeModelClient,
        capture_random_state,
    )
    from astrolabe.mutators.wrappers.ligandmpnn_run import LigandMPNN

    if not torch.cuda.is_available():
        raise HarnessError("The throughput harness requires a visible CUDA device")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.weights
    source_sha256 = _sha256(args.input)
    checkpoint_sha256 = _sha256(checkpoint)
    source_identity_before = _source_identity()
    pdb_path = _prepare_pdb(
        source=args.input,
        chain=args.chain,
        destination=args.out_dir / "input.pdb",
    )
    features = {
        "pdb_path": str(pdb_path),
        "chains_to_design": args.chain,
        "parse_these_chains_only": args.chain,
        "fixed_residues": "",
    }

    direct_config = _model_config(
        seed=args.seed,
        checkpoint=checkpoint,
        output_dir=args.out_dir / "direct",
    )
    daemon_config = _model_config(
        seed=args.seed,
        checkpoint=checkpoint,
        output_dir=args.out_dir / "daemon",
    )
    direct_scientific_config = dict(direct_config)
    daemon_scientific_config = dict(daemon_config)
    direct_scientific_config.pop("out_folder")
    daemon_scientific_config.pop("out_folder")
    if direct_scientific_config != daemon_scientific_config:
        raise HarnessError("Direct and daemon scientific configurations differ")
    torch.cuda.reset_peak_memory_stats()
    _reset_rng(args.seed)
    direct_reads: list[str] = []
    with counted_torch_load(torch, direct_reads):
        direct, direct_init_s = _measure(
            lambda: LigandMPNN(OmegaConf.create(direct_config)),
            synchronize=torch.cuda.synchronize,
        )
    direct_states = [capture_random_state()]
    direct_outputs: list[dict[str, Any]] = []
    direct_request_s: list[float] = []
    direct_peak = [{
        "phase": "after_init",
        "allocated_bytes": torch.cuda.max_memory_allocated(),
        "reserved_bytes": torch.cuda.max_memory_reserved(),
    }]
    for index in range(args.requests):
        output, elapsed = _measure(
            lambda: direct.design_sequences(**features),
            synchronize=torch.cuda.synchronize,
        )
        direct_outputs.append(output)
        direct_request_s.append(elapsed)
        direct_states.append(capture_random_state())
        direct_peak.append({
            "phase": f"after_request_{index + 1}",
            "allocated_bytes": torch.cuda.max_memory_allocated(),
            "reserved_bytes": torch.cuda.max_memory_reserved(),
        })
    direct_owner = {
        "pid": os.getpid(),
        "model_token": id(direct.model),
        "device": str(direct.device),
        "checkpoint_reads": list(direct_reads),
    }
    direct_timing = aggregate_timings(direct_init_s, direct_request_s)
    del direct
    gc.collect()
    torch.cuda.empty_cache()

    _reset_rng(args.seed)
    daemon, daemon_init_s = _measure(
        lambda: NativeModelClient(
            "ligandmpnn",
            DAEMON_FACTORY,
            {"runner_config": daemon_config},
            cwd=str(ROOT),
        ),
        synchronize=torch.cuda.synchronize,
    )
    daemon_states = [capture_random_state()]
    daemon_outputs: list[dict[str, Any]] = []
    daemon_request_s: list[float] = []
    daemon_records: list[dict[str, Any]] = []
    try:
        for index in range(args.requests):
            record, elapsed = _measure(
                lambda: daemon.request({"features": features}),
                synchronize=torch.cuda.synchronize,
            )
            daemon_records.append(record)
            daemon_outputs.append(record["output"])
            daemon_request_s.append(elapsed)
            daemon_states.append(capture_random_state())
        owner_status = daemon.owner.status()
    finally:
        daemon.shutdown()
    daemon_timing = aggregate_timings(daemon_init_s, daemon_request_s)

    _assert_exact_trace(direct_reads, checkpoint, lane="direct")
    daemon_metadata = dict(daemon.metadata)
    daemon_metadata_reads = list(daemon_metadata.get("checkpoint_reads", []))
    _assert_exact_trace(daemon_metadata_reads, checkpoint, lane="daemon")
    for index, record in enumerate(daemon_records):
        _assert_exact_trace(
            list(record.get("checkpoint_reads", [])),
            checkpoint,
            lane=f"daemon request {index + 1}",
        )
        if int(record.get("load_count", -1)) != 1:
            raise HarnessError("daemon did not retain exactly one loaded checkpoint")

    daemon_pids = {int(record["pid"]) for record in daemon_records}
    daemon_tokens = {int(record["model_token"]) for record in daemon_records}
    if len(daemon_pids) != 1 or len(daemon_tokens) != 1:
        raise HarnessError("daemon model owner changed between requests")

    comparisons = []
    direct_arrays: dict[str, np.ndarray] = {}
    daemon_arrays: dict[str, np.ndarray] = {}
    for index, (expected, actual) in enumerate(zip(direct_outputs, daemon_outputs)):
        comparison = compare_outputs(expected, actual)
        comparison["request_index"] = index
        comparisons.append(comparison)
        for key, value in _flatten_arrays(expected).items():
            direct_arrays[f"request_{index}.{key}"] = value
        for key, value in _flatten_arrays(actual).items():
            daemon_arrays[f"request_{index}.{key}"] = value
    if not all(item["equal"] for item in comparisons):
        failed = [
            {
                "request_index": item["request_index"],
                "differences": item["differences"][:20],
            }
            for item in comparisons
            if not item["equal"]
        ]
        raise HarnessError(f"Direct and daemon outputs differ exactly: {failed}")

    direct_rng = _random_state_digests(direct_states)
    daemon_rng = _random_state_digests(daemon_states)
    if direct_rng != daemon_rng:
        mismatched = [
            index
            for index, (left, right) in enumerate(zip(direct_rng, daemon_rng))
            if left != right
        ]
        raise HarnessError(f"Caller RNG continuation differs at boundaries {mismatched}")

    direct_array_path = args.out_dir / "direct_outputs.npz"
    daemon_array_path = args.out_dir / "daemon_outputs.npz"
    np.savez_compressed(direct_array_path, **direct_arrays)
    np.savez_compressed(daemon_array_path, **daemon_arrays)

    if _source_identity() != source_identity_before:
        raise HarnessError("Source changed while GPU evidence was being collected")
    if _sha256(checkpoint) != checkpoint_sha256 or _sha256(args.input) != source_sha256:
        raise HarnessError("Checkpoint or input changed during the GPU run")

    daemon_peak = daemon_records[-1]["cuda_peak"]
    report = {
        "schema": SCHEMA,
        "verified": True,
        "scope": (
            "Exact within-run direct-vs-resident-daemon ProteinMPNN output and "
            "caller RNG continuation; no performance threshold is asserted."
        ),
        "created_unix_s": time.time(),
        "requests": args.requests,
        "seed": args.seed,
        "chain": args.chain,
        "device_requested": "cuda",
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_sha256,
            "load_count": {
                "direct": len(direct_reads),
                "daemon": int(daemon_records[-1]["load_count"]),
            },
        },
        "input": {
            "path": str(args.input),
            "sha256": source_sha256,
            "prepared_pdb": str(pdb_path),
            "prepared_pdb_sha256": _sha256(pdb_path),
        },
        "gpu": _gpu_facts(torch),
        "source_sha256": source_identity_before,
        "timing": {
            "direct": _timing_report(direct_timing),
            "daemon_owner": _timing_report(daemon_timing),
        },
        "owner": {
            "direct": direct_owner,
            "daemon": {
                "pid": next(iter(daemon_pids)),
                "model_token": next(iter(daemon_tokens)),
                "device": str(daemon_metadata.get("device")),
                "identity": owner_status.get("identity"),
                "checkpoint_reads": daemon_metadata_reads,
                "load_count": int(daemon_records[-1]["load_count"]),
            },
        },
        "cuda_peak_bytes": {
            "direct": direct_peak,
            "daemon_init": daemon_metadata.get("cuda_peak_after_init"),
            "daemon_final": daemon_peak,
        },
        "exact_comparison": comparisons,
        "caller_rng": {
            "equal": True,
            "direct_boundary_sha256": direct_rng,
            "daemon_boundary_sha256": daemon_rng,
            "boundary_order": [
                "after_model_init",
                *[f"after_request_{index + 1}" for index in range(args.requests)],
            ],
        },
        "raw_arrays": {
            "direct": str(direct_array_path),
            "daemon": str(daemon_array_path),
        },
        "performance": {
            "measured": True,
            "threshold_used": False,
            "speedup_claim": None,
            "warm_rate_definition": (
                "all requests after the first divided by their summed wall time; "
                "first_three_warm_requests_per_s is also reported when available"
            ),
        },
    }
    report_path = args.out_dir / REPORT_NAME
    _atomic_json(report_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = _run(args)
    print(args.out_dir / REPORT_NAME)
    if not report["verified"]:
        raise HarnessError("GPU throughput report was not verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
