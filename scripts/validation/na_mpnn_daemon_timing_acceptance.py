#!/usr/bin/env python3
"""Replay two resident NA-MPNN requests with durable timing and GPU identity."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPO = Path("/ext/na_mpnn")
DEFAULT_CHECKPOINT = Path("/ext/na_mpnn/models/design_model/s_19137.pt")
DEFAULT_INPUT = (
    ROOT
    / "outputs"
    / "rna-archived-gpu-acceptance"
    / "gpu_v4"
    / "baseline"
    / "configured-output"
    / "001_seed_67101"
    / "structures"
    / "step_000_ABC_model.cif"
)

SCHEMA = "astrolabe.na-mpnn-daemon-timing/v1"
PREFLIGHT_SCHEMA = f"{SCHEMA}.preflight"
PROFILE = "na_mpnn_daemon_timing_v1"
PREFLIGHT_NAME = "preflight.json"
REPORT_NAME = "acceptance.json"
REFERENCE_MANIFEST_NAME = "manifest.json"
DAEMON_FACTORY = (
    "scripts.validation.na_mpnn_daemon_timing_acceptance:daemon_factory"
)
SEEDS = (267850363, 267850364)
HARD_REQUEST_LIMIT = 2
REQUEST_TIMEOUT_S = 300

from scripts.validation.model_gpu_throughput import (  # noqa: E402
    HarnessError,
    _flatten_arrays,
    _json_default,
    _sha256,
    compare_outputs,
    counted_torch_load,
)
from scripts.validation import na_mpnn_gpu_acceptance as base  # noqa: E402


class TimingAcceptanceError(HarnessError):
    """The bounded resident timing replay cannot be proven."""


@dataclass
class RequestBudget:
    maximum: int = HARD_REQUEST_LIMIT
    claimed: int = 0

    def claim(self, arm: str) -> int:
        if arm != "daemon":
            raise TimingAcceptanceError(
                "na_mpnn_daemon_timing_v1 permits resident daemon requests only"
            )
        if self.claimed >= self.maximum:
            raise TimingAcceptanceError("NA-MPNN daemon hard budget exhausted")
        self.claimed += 1
        return self.claimed


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clock() -> dict[str, int | str]:
    return {
        "utc": _utc_now(),
        "unix_ns": time.time_ns(),
        "monotonic_ns": time.monotonic_ns(),
        "perf_counter_ns": time.perf_counter_ns(),
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _durable_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                allow_nan=False,
                default=_json_default,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _durable_npz(path: Path, values: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **values)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _append_json_line(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                value,
                sort_keys=True,
                allow_nan=False,
                default=_json_default,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def _state_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=_json_default,
        ).encode("utf-8")
    ).hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise TimingAcceptanceError(f"Required evidence file is missing: {path}")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def request_plans() -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "arm": "daemon",
            "request_index": index,
            "seed": seed,
            "timing_label": "first_with_daemon_startup" if index == 0 else "warm_second",
            "temperature": base.TEMPERATURE,
            "batch_size": base.BATCH_SIZE,
            "number_of_batches": base.NUMBER_OF_BATCHES,
            "chains_to_design": "A",
            "parse_these_chains_only": "ABC",
            "design_na_only": 1,
            "parse_na_only": 0,
            "na_shared_tokens": 0,
            "fixed_residues": list(base.FIXED_RESIDUES),
            "design_mask": list(base.EXPECTED_DESIGN_MASK),
        }
        for index, seed in enumerate(SEEDS)
    )


def _reference_relative_paths(seed: int, index: int) -> dict[str, str]:
    directory = f"subprocess_{index:02d}_seed_{seed}"
    return {
        "fasta": f"{directory}/seqs/step_000_ABC_model.fa",
        "specificity_npz": f"{directory}/specificity/step_000_ABC_model.npz",
    }


def reference_facts(reference_root: Path) -> dict[str, Any]:
    root = reference_root.expanduser().resolve()
    manifest_path = root / REFERENCE_MANIFEST_NAME
    if not manifest_path.is_file():
        raise TimingAcceptanceError(
            f"Old-native reference manifest is missing: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != f"{SCHEMA}.native-reference":
        raise TimingAcceptanceError("Old-native reference manifest schema changed")
    if tuple(manifest.get("seeds", ())) != SEEDS:
        raise TimingAcceptanceError("Old-native reference seeds changed")
    expected: set[str] = set()
    for index, seed in enumerate(SEEDS):
        expected.update(_reference_relative_paths(seed, index).values())
    declared = manifest.get("files")
    if not isinstance(declared, Mapping) or set(declared) != expected:
        raise TimingAcceptanceError("Old-native reference file set changed")
    files: dict[str, Any] = {}
    for relative in sorted(expected):
        path = root / relative
        observed = _file_record(path)
        if observed["sha256"] != declared[relative]:
            raise TimingAcceptanceError(
                f"Old-native reference content changed: {relative}"
            )
        files[relative] = observed
    return {
        "root": str(root),
        "manifest": _file_record(manifest_path),
        "files": files,
    }


def _load_reference_bundle(path: Path) -> dict[str, Any]:
    from astrolabe.mutators.wrappers.na_mpnn_runner import NAMPNNRunner

    loader = NAMPNNRunner.__new__(NAMPNNRunner)
    bundle = loader._load_specificity_matrix(
        path.expanduser().resolve().parent.parent,
        Path("step_000_ABC_model.cif"),
    )
    if not isinstance(bundle, Mapping) or set(bundle) != base._EXPECTED_BUNDLE_KEYS:
        raise TimingAcceptanceError(
            f"Old-native probability fields changed: {sorted(bundle or ())}"
        )
    return dict(bundle)


def _assert_gpu_identity(facts: Mapping[str, Any]) -> None:
    if not str(facts.get("uuid") or ""):
        raise TimingAcceptanceError(
            "Actual GPU UUID was not captured; refusing the paid request"
        )
    if not str(facts.get("current_device") or "") and facts.get("current_device") != 0:
        raise TimingAcceptanceError("Actual CUDA device index was not captured")


class TimingResidentHandler:
    """Resident owner wrapper that durably journals before and after inference."""

    def __init__(
        self,
        handler: Callable[[Mapping[str, Any]], Any],
        *,
        resident_model: Any,
        capture_path: Path,
        request_root: Path,
        checkpoint_path: Path,
        checkpoint_reads: Sequence[str],
        gpu_facts: Callable[[], Mapping[str, Any]],
        maximum_requests: int = HARD_REQUEST_LIMIT,
    ) -> None:
        self.handler = handler
        self.pid = os.getpid()
        self.model_token = id(resident_model)
        self.capture_path = capture_path.expanduser().resolve()
        self.request_root = request_root.expanduser().resolve()
        self.checkpoint_path = checkpoint_path.expanduser().resolve()
        self.checkpoint_reads = tuple(
            str(Path(path).expanduser().resolve()) for path in checkpoint_reads
        )
        if self.checkpoint_reads != (str(self.checkpoint_path),):
            raise TimingAcceptanceError(
                "Resident NA-MPNN checkpoint load trace must contain one exact read"
            )
        self.gpu_facts = gpu_facts
        self.maximum_requests = int(maximum_requests)
        self.request_count = 0
        self._lock = threading.Lock()
        try:
            self.device = str(next(resident_model.parameters()).device)
        except (AttributeError, StopIteration):
            self.device = str(getattr(resident_model, "device", "unknown"))

    def infer(self, payload: Mapping[str, Any]) -> Any:
        from astrolabe.model_execution import capture_random_state

        with self._lock:
            if self.request_count >= self.maximum_requests:
                raise TimingAcceptanceError(
                    "Resident NA-MPNN daemon request budget exhausted"
                )
            self.request_count += 1
            request_index = self.request_count - 1
        requests_dir = self.request_root / "daemon"
        before_path = requests_dir / f"request_{request_index:02d}_before.json"
        after_path = requests_dir / f"request_{request_index:02d}_after.json"
        arrays_path = requests_dir / f"request_{request_index:02d}_probabilities.npz"
        gpu = dict(self.gpu_facts())
        _assert_gpu_identity(gpu)
        daemon_rng_before = capture_random_state()
        clock_start = _clock()
        started = time.perf_counter()
        before = {
            "schema": f"{SCHEMA}.daemon-before",
            "status": "starting",
            "request_index": request_index,
            "request_count": self.request_count,
            "request_delta": 1,
            "request": {
                "job_name": payload.get("job_name"),
                "job_dir": payload.get("job_dir"),
                "seed": payload.get("seed"),
            },
            "owner": {
                "pid": self.pid,
                "model_token": self.model_token,
                "device": self.device,
                "gpu": gpu,
                "checkpoint_reads": list(self.checkpoint_reads),
                "model_load_count": len(self.checkpoint_reads),
            },
            "daemon_rng_before_sha256": _state_digest(daemon_rng_before),
            "daemon_rng_before": daemon_rng_before,
            "clock": clock_start,
        }
        _durable_json(before_path, before)

        status = "completed"
        error = None
        try:
            result = self.handler(payload)
        except BaseException as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            result = None

        elapsed_s = time.perf_counter() - started
        clock_after = _clock()
        daemon_rng_after = capture_random_state()
        raw_result: dict[str, Any] = {}
        bundle = result.get("probabilities") if isinstance(result, Mapping) else None
        if isinstance(bundle, Mapping):
            _durable_npz(
                arrays_path,
                {
                    str(key): np.asarray(value)
                    for key, value in bundle.items()
                },
            )
            raw_result["probabilities"] = _file_record(arrays_path)
        if isinstance(result, Mapping):
            for name in ("job_dir", "fasta_path", "sequence"):
                if name in result:
                    raw_result[name] = result[name]
        after = {
            "schema": f"{SCHEMA}.daemon-after",
            "status": status,
            "error": error,
            "request_index": request_index,
            "request_count": self.request_count,
            "request_delta": 1,
            "request": before["request"],
            "owner": {
                "pid": self.pid,
                "model_token": self.model_token,
                "device": self.device,
                "gpu": gpu,
                "checkpoint_reads": list(self.checkpoint_reads),
                "model_load_count": len(self.checkpoint_reads),
            },
            "clock_start": clock_start,
            "clock_after": clock_after,
            "duration_s": elapsed_s,
            "raw_result": raw_result,
            "daemon_rng_after_sha256": _state_digest(daemon_rng_after),
            "daemon_rng_after": daemon_rng_after,
        }
        _durable_json(after_path, after)
        record = {
            "pid": self.pid,
            "model_token": self.model_token,
            "device": self.device,
            "gpu_uuid": gpu["uuid"],
            "model_load_count": len(self.checkpoint_reads),
            "checkpoint_reads": list(self.checkpoint_reads),
            "request_count": self.request_count,
            "request_delta": 1,
            "status": status,
            "error": error,
            "job_name": payload.get("job_name"),
            "job_dir": payload.get("job_dir"),
            "duration_s": elapsed_s,
            "clock_start": clock_start,
            "clock_after": clock_after,
            **raw_result,
        }
        _append_json_line(self.capture_path, record)
        if error is not None:
            raise TimingAcceptanceError(error)
        return result


def daemon_factory(settings: Mapping[str, Any]) -> TimingResidentHandler:
    import torch
    from astrolabe.mutators.wrappers.na_mpnn_daemon import NAMPNNHandler

    settings = dict(settings)
    capture_path = Path(settings.pop("_acceptance_capture_path"))
    request_root = Path(settings.pop("_acceptance_request_root"))
    checkpoint_path = Path(settings["checkpoint"])
    checkpoint_reads: list[str] = []
    with counted_torch_load(torch, checkpoint_reads):
        handler = NAMPNNHandler(settings)
    resident = handler.resident.get("model")
    if resident is None:
        raise TimingAcceptanceError("NA-MPNN daemon handler has no resident model")
    return TimingResidentHandler(
        handler,
        resident_model=resident,
        capture_path=capture_path,
        request_root=request_root,
        checkpoint_path=checkpoint_path,
        checkpoint_reads=checkpoint_reads,
        gpu_facts=lambda: base._gpu_facts(torch),
    )


def _daemon_runner_class():
    from astrolabe.model_daemon import DaemonSpec, OwnedModelDaemon
    from astrolabe.mutators.wrappers.na_mpnn_runner import NAMPNNRunner

    class TimingDaemonRunner(NAMPNNRunner):
        def __init__(
            self,
            args,
            *,
            output_root: Path,
            capture_path: Path,
            request_root: Path,
        ) -> None:
            super().__init__(args, output_root=output_root)
            if self.execution_mode != "daemon":
                raise ValueError("TimingDaemonRunner requires execution_mode=daemon")
            self.acceptance_capture_path = capture_path.expanduser().resolve()
            self.acceptance_request_root = request_root.expanduser().resolve()
            self.records: list[dict[str, Any]] = []

        def _request_daemon(self, payload) -> None:
            with self._daemon_lock:
                if self._owned_daemon is None:
                    settings = dict(self._daemon_settings())
                    settings["_acceptance_capture_path"] = str(
                        self.acceptance_capture_path
                    )
                    settings["_acceptance_request_root"] = str(
                        self.acceptance_request_root
                    )
                    self._owned_daemon = OwnedModelDaemon(
                        DaemonSpec(
                            backend="na_mpnn",
                            factory=DAEMON_FACTORY,
                            settings=settings,
                            python_bin=self.python,
                            cwd=str(self.repo_path),
                            env=self._patched_env(),
                        ),
                        startup_timeout=self.startup_timeout,
                        request_timeout=REQUEST_TIMEOUT_S,
                        shutdown_timeout=self.shutdown_timeout,
                        max_pending=self.max_pending,
                    )
                daemon = self._owned_daemon
            record = daemon.request(payload)
            self.records.append(dict(record))

        def status(self) -> dict[str, Any]:
            if self._owned_daemon is None:
                raise TimingAcceptanceError("NA-MPNN daemon has not started")
            return self._owned_daemon.status()

    return TimingDaemonRunner


def _execution_source_manifest() -> dict[str, str]:
    records = dict(base._execution_source_manifest())
    records[str(Path(__file__).resolve().relative_to(ROOT))] = _sha256(Path(__file__))
    if DAEMON_FACTORY.split(":", 1)[0].replace(".", "/") + ".py" not in records:
        raise TimingAcceptanceError("Timing profile is absent from execution source")
    return records


def _manifest_digest(records: Mapping[str, str]) -> str:
    return hashlib.sha256(
        json.dumps(
            records,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def preflight(
    *,
    repo: Path,
    checkpoint: Path,
    input_path: Path,
    python_bin: str,
    reference_root: Path,
    probe_entrypoint: bool = True,
) -> dict[str, Any]:
    reference_before = reference_facts(reference_root)
    execution_before = _execution_source_manifest()
    legacy = base.preflight(
        repo=repo,
        checkpoint=checkpoint,
        input_path=input_path,
        python_bin=python_bin,
        probe_entrypoint=probe_entrypoint,
    )
    execution_after = _execution_source_manifest()
    reference_after = reference_facts(reference_root)
    if execution_before != execution_after:
        raise TimingAcceptanceError("Timing execution source changed during preflight")
    if reference_before != reference_after:
        raise TimingAcceptanceError("Old-native reference changed during preflight")
    report = dict(legacy)
    report.update(
        {
            "schema": PREFLIGHT_SCHEMA,
            "profile": PROFILE,
            "execution_source": {
                "manifest": execution_after,
                "manifest_sha256": _manifest_digest(execution_after),
            },
            "reference": reference_after,
            "request_budget": {
                "hard_total": HARD_REQUEST_LIMIT,
                "per_arm": {"subprocess": 0, "daemon": HARD_REQUEST_LIMIT},
                "native_reference_requests": 0,
                "af3_requests": 0,
            },
            "request_plans": list(request_plans()),
        }
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Absent or empty directory for raw evidence and acceptance.",
    )
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument(
        "--reference-root",
        type=Path,
        required=True,
        help="Frozen directory containing the old-native FASTA/NPZ manifest.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate inputs/source/reference without CUDA or model inference.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.out_dir = args.out_dir.expanduser().resolve()
    args.repo = args.repo.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.input = args.input.expanduser().resolve()
    args.reference_root = args.reference_root.expanduser().resolve()
    args.python_bin = str(args.python_bin)
    if args.out_dir.exists():
        if not args.out_dir.is_dir():
            parser.error(f"--out-dir is not a directory: {args.out_dir}")
        if any(args.out_dir.iterdir()) and not args.preflight_only:
            parser.error("--out-dir must be absent or empty")
    if not args.input.is_file():
        parser.error(f"Input structure is unavailable: {args.input}")
    if not Path(args.python_bin).is_file():
        parser.error(f"Python executable is unavailable: {args.python_bin}")
    if not args.reference_root.is_dir():
        parser.error(f"Old-native reference is unavailable: {args.reference_root}")
    return args


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = preflight(
        repo=args.repo,
        checkpoint=args.checkpoint,
        input_path=args.input,
        python_bin=args.python_bin,
        reference_root=args.reference_root,
        probe_entrypoint=True,
    )
    _durable_json(args.out_dir / PREFLIGHT_NAME, report)
    return report


def _request_features() -> dict[str, Any]:
    return {
        "chains_to_design": "A",
        "fixed_residues": list(base.FIXED_RESIDUES),
        "parse_these_chains_only": "ABC",
    }


def _raw_output_snapshot(
    path: Path,
    captured: Mapping[str, Any],
) -> dict[str, Any]:
    arrays: dict[str, np.ndarray] = {}
    for key, value in _flatten_arrays(captured["deterministic"]).items():
        arrays[f"deterministic.{key}"] = value
    _durable_npz(path, arrays)
    return _file_record(path)


def _compare_reference(
    *,
    seed: int,
    index: int,
    reference_root: Path,
    captured: Mapping[str, Any],
) -> dict[str, Any]:
    paths = _reference_relative_paths(seed, index)
    reference_fasta = reference_root / paths["fasta"]
    reference_npz = reference_root / paths["specificity_npz"]
    live_fasta = Path(captured["fasta_path"])
    fasta_equal = reference_fasta.read_bytes() == live_fasta.read_bytes()
    array_comparison = compare_outputs(
        _load_reference_bundle(reference_npz),
        captured["deterministic"]["probability_bundle"],
    )
    return {
        "fasta_equal": fasta_equal,
        "reference_fasta": _file_record(reference_fasta),
        "live_fasta": _file_record(live_fasta),
        "reference_npz": _file_record(reference_npz),
        "probability_bundle_equal": bool(array_comparison["equal"]),
        "probability_comparison": array_comparison,
        "selected_sequence": captured["deterministic"]["selected_fasta_entries"][0][
            "sequence"
        ],
        "fixed_mask": captured["deterministic"]["fixed_mask"],
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from astrolabe.model_execution import capture_random_state

    if not torch.cuda.is_available():
        raise TimingAcceptanceError(
            "The NA-MPNN daemon timing profile requires a visible CUDA device"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    preflight_report = preflight(
        repo=args.repo,
        checkpoint=args.checkpoint,
        input_path=args.input,
        python_bin=args.python_bin,
        reference_root=args.reference_root,
        probe_entrypoint=True,
    )
    _durable_json(args.out_dir / PREFLIGHT_NAME, preflight_report)
    source_before = preflight_report["repo"]["source_manifest"]
    execution_before = preflight_report["execution_source"]["manifest"]
    reference_before = preflight_report["reference"]
    input_sha_before = preflight_report["input"]["sha256"]
    checkpoint_sha_before = preflight_report["checkpoint"]["sha256"]
    budget = RequestBudget()
    accepted = {"subprocess": 0, "daemon": 0}

    torch.cuda.current_device()
    torch.cuda.synchronize()
    capture_path = args.out_dir / "resident-requests.jsonl"
    request_root = args.out_dir / "requests"
    TimingDaemonRunner = _daemon_runner_class()
    runner = TimingDaemonRunner(
        base._runner_config(
            repo=args.repo,
            checkpoint=args.checkpoint,
            python_bin=args.python_bin,
            execution_mode="daemon",
        ),
        output_root=args.out_dir / "daemon",
        capture_path=capture_path,
        request_root=request_root,
    )
    request_results: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []
    parent_times: list[float] = []
    caller_rng_checks: list[dict[str, Any]] = []
    try:
        for index, plan in enumerate(request_plans()):
            accepted["daemon"] = budget.claim("daemon")
            runner.extra_args["seed"] = int(plan["seed"])
            caller_before = capture_random_state()
            caller_before_sha = _state_digest(caller_before)
            gpu_before = base._gpu_facts(torch)
            _assert_gpu_identity(gpu_before)
            before_path = request_root / f"request_{index:02d}_caller_before.json"
            clock_start = _clock()
            perf_start = time.perf_counter()
            _durable_json(
                before_path,
                {
                    "schema": f"{SCHEMA}.caller-before",
                    "request_index": index,
                    "request": plan,
                    "gpu": gpu_before,
                    "caller_rng_before_sha256": caller_before_sha,
                    "caller_rng_before": caller_before,
                    "clock": clock_start,
                },
            )
            result = runner.design_sequences(
                structure_path=args.input,
                job_name=f"daemon_{index:02d}_seed_{plan['seed']}",
                **_request_features(),
            )
            torch.cuda.synchronize()
            elapsed_s = time.perf_counter() - perf_start
            parent_times.append(elapsed_s)
            clock_after = _clock()
            caller_after = capture_random_state()
            caller_after_sha = _state_digest(caller_after)
            owner_status = runner.status()
            statuses.append(owner_status)
            captured = base.capture_runner_result(
                runner=runner,
                result=result,
                plan=plan,
                input_data=preflight_report["input"],
            )
            raw_npz_path = request_root / f"request_{index:02d}_live_arrays.npz"
            raw_npz = _raw_output_snapshot(raw_npz_path, captured)
            raw_path = request_root / f"request_{index:02d}_raw_after.json"
            raw_record = {
                "schema": f"{SCHEMA}.caller-after-raw",
                "request_index": index,
                "request": plan,
                "duration_s": elapsed_s,
                "clock_start": clock_start,
                "clock_after": clock_after,
                "caller_rng_before_sha256": caller_before_sha,
                "caller_rng_after_sha256": caller_after_sha,
                "caller_rng_before": caller_before,
                "caller_rng_after": caller_after,
                "gpu_before": gpu_before,
                "owner_status": owner_status,
                "daemon_owner_record": runner.records[index],
                "job_dir": captured["job_dir"],
                "fasta_path": captured["fasta_path"],
                "fasta": _file_record(Path(captured["fasta_path"])),
                "raw_arrays": raw_npz,
                "selected_raw_sha256": hashlib.sha256(
                    str(captured["selected_raw"]).encode("utf-8")
                ).hexdigest(),
            }
            _durable_json(raw_path, raw_record)

            comparison = _compare_reference(
                seed=int(plan["seed"]),
                index=index,
                reference_root=args.reference_root,
                captured=captured,
            )
            comparison_path = request_root / f"request_{index:02d}_comparison.json"
            _durable_json(
                comparison_path,
                {
                    "schema": f"{SCHEMA}.request-comparison",
                    "request_index": index,
                    "seed": plan["seed"],
                    "reference_content_bound": True,
                    **comparison,
                },
            )
            caller_equal = (
                caller_before_sha == caller_after_sha
                and caller_before == caller_after
            )
            caller_rng_checks.append(
                {
                    "request_index": index,
                    "equal": caller_equal,
                    "before_sha256": caller_before_sha,
                    "after_sha256": caller_after_sha,
                }
            )
            request_results.append(
                {
                    **plan,
                    "duration_parent_call_s": elapsed_s,
                    "duration_daemon_infer_s": float(
                        runner.records[index].get("duration_s", 0.0)
                    ),
                    "caller_before": _file_record(before_path),
                    "raw_after": _file_record(raw_path),
                    "comparison": _file_record(comparison_path),
                    "comparison_verified": (
                        comparison["fasta_equal"]
                        and comparison["probability_bundle_equal"]
                        and caller_equal
                    ),
                    "owner_status": owner_status,
                    "daemon_owner_record": runner.records[index],
                }
            )
            if not comparison["fasta_equal"]:
                raise TimingAcceptanceError(
                    f"Seed {plan['seed']} daemon FASTA differs from old native reference"
                )
            if not comparison["probability_bundle_equal"]:
                raise TimingAcceptanceError(
                    f"Seed {plan['seed']} daemon probability bundle differs from reference"
                )
            if not caller_equal:
                raise TimingAcceptanceError(
                    f"Seed {plan['seed']} changed caller RNG state"
                )
    finally:
        runner.shutdown()

    resident_records = [
        json.loads(line)
        for line in capture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    resident_summary = base.validate_resident_evidence(
        resident_records,
        statuses,
    )
    if accepted != {"subprocess": 0, "daemon": 2}:
        raise TimingAcceptanceError(f"Request budget mismatch: {accepted}")
    if len(parent_times) != HARD_REQUEST_LIMIT:
        raise TimingAcceptanceError("Timing profile did not complete two requests")
    if _sha256(args.input) != input_sha_before:
        raise TimingAcceptanceError("NA-MPNN root input changed during inference")
    if _sha256(args.checkpoint) != checkpoint_sha_before:
        raise TimingAcceptanceError("NA-MPNN checkpoint changed during inference")
    if base._repo_source_manifest(args.repo) != source_before:
        raise TimingAcceptanceError("Frozen NA-MPNN repository source changed")
    if _execution_source_manifest() != execution_before:
        raise TimingAcceptanceError("Astrolabe timing execution source changed")
    if reference_facts(args.reference_root) != reference_before:
        raise TimingAcceptanceError("Old-native reference changed during inference")

    report = {
        "schema": SCHEMA,
        "profile": PROFILE,
        "verified": True,
        "created_unix_s": time.time(),
        "scope": (
            "Two seeded resident NA-MPNN requests compared with the preserved "
            "old-native FASTA and ten-field probability reference. Timing is "
            "descriptive only; no significance, speedup, or native-speed claim."
        ),
        "request_budget": {
            "hard_total": HARD_REQUEST_LIMIT,
            "accepted": accepted,
            "native_reference_requests": 0,
            "af3_requests": 0,
        },
        "reference": reference_before,
        "input": preflight_report["input"],
        "repo": preflight_report["repo"],
        "checkpoint": preflight_report["checkpoint"],
        "execution_source": preflight_report["execution_source"],
        "public_entrypoint": preflight_report["public_entrypoint"],
        "gpu": base._gpu_facts(torch),
        "requests": request_results,
        "resident_daemon": resident_summary,
        "caller_rng": {
            "requests": caller_rng_checks,
            "all_unchanged": all(item["equal"] for item in caller_rng_checks),
        },
        "performance": {
            "labels": ["first_with_daemon_startup", "warm_second"],
            "parent_call_s": parent_times,
            "daemon_infer_s": [
                float(item["duration_daemon_infer_s"])
                for item in request_results
            ],
            "descriptive_only": True,
            "speedup_claim": None,
            "statistical_claim": None,
            "native_speed_comparison": None,
        },
        "frozen_after_run": {
            "repo_source_manifest_sha256": _manifest_digest(
                base._repo_source_manifest(args.repo)
            ),
            "execution_source_manifest_sha256": _manifest_digest(
                _execution_source_manifest()
            ),
            "reference_manifest_sha256": _sha256(
                args.reference_root / REFERENCE_MANIFEST_NAME
            ),
            "input_sha256": _sha256(args.input),
            "checkpoint_sha256": _sha256(args.checkpoint),
        },
    }
    _durable_json(args.out_dir / REPORT_NAME, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.preflight_only:
        run_preflight(args)
        print(args.out_dir / PREFLIGHT_NAME)
        return 0
    report = _run(args)
    print(args.out_dir / REPORT_NAME)
    if not report["verified"]:
        raise TimingAcceptanceError("NA-MPNN daemon timing report is not verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
