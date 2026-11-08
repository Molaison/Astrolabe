from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from omegaconf import OmegaConf, open_dict

from astrolabe.experimental.candidate_batch_policy import (
    CandidateCollection,
    CandidateExecutionError,
    CandidatePoolConfig,
    CandidatePoolScheduler,
    CandidateResult,
    CandidateSelector,
    CandidateTask,
    select_highest_score_candidate,
)
from astrolabe.design.collection_search import (
    CollectionAction,
    CollectionEvent,
    CollectionPolicy,
    CollectionRun,
    CollectionState,
    CollectionTrial,
)
from astrolabe.design.collection import Job
from astrolabe.design.collection_receipts import (
    DurableCollectionTrial,
    RehydratedCollection,
)
from astrolabe.design.runtime import ContractConflict, OutcomeUnknown, Runtime
from astrolabe.design.store import DataclassCodec, JSONCodec
from astrolabe.design.types import Blob


CandidateBatchRun = CollectionRun


class CandidateBatchConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ProcessCandidateRunnerConfig:
    config_yaml: Path
    out_root: Path
    runner: Path
    num_candidates: int = 4
    min_completed: int = 2
    max_wait_s: float = 30.0
    max_in_flight: int | None = None
    decision_rounds: int = 1
    candidate_steps: int = 2
    candidate_timeout_s: float | None = None
    dry_run: bool = False
    resume: bool = False


@dataclass(frozen=True)
class ProcessCandidatePayload:
    config_yaml: str
    out_dir: str
    task_name: str
    stdout_log: str
    stderr_log: str
    sequences: dict[str, str] = field(default_factory=dict)

    def as_summary(self) -> dict[str, object]:
        return {
            "config_yaml": self.config_yaml,
            "out_dir": self.out_dir,
            "task_name": self.task_name,
            "stdout_log": self.stdout_log,
            "stderr_log": self.stderr_log,
            "sequences": dict(self.sequences),
        }


@dataclass(frozen=True)
class ProcessCandidateScore:
    weighted_score: float
    sequences: dict[str, str]


CHILD_RECEIPT_SCHEMA = "astrolabe.process-candidate-receipt/v1"
COLLECTION_RECEIPT_SCHEMA = "astrolabe.process-collection-receipt/v1"


@dataclass(frozen=True)
class CandidateExecutionSpec:
    job_id: str
    generation: int
    attempt: int
    actual_seed: int
    task_name: str
    out_dir: str
    work_root: str
    config_yaml: str
    stdout_log: str
    stderr_log: str
    receipt_path: str
    sequences: dict[str, str]
    final_config: dict[str, Any]
    scientific_config: dict[str, Any]
    source_manifest: tuple[dict[str, Any], ...]
    input_assets_manifest: dict[str, Any]
    content_id: str


@dataclass(frozen=True)
class ProcessCandidateReceipt:
    schema: str
    status: str
    job_id: str
    generation: int
    attempt: int
    actual_seed: int
    content_id: str
    task_name: str
    config_yaml: str
    out_dir: str
    stdout_log: str
    stderr_log: str
    sequences: dict[str, str]
    final_config: dict[str, Any]
    scientific_config: dict[str, Any]
    source_manifest: tuple[dict[str, Any], ...]
    input_assets_manifest: dict[str, Any]
    issued_process: dict[str, Any] | None = None
    cli_provenance: dict[str, Any] | None = None
    output_manifest: tuple[dict[str, Any], ...] = ()
    result: CandidateResult | None = None
    error: str | None = None


@dataclass(frozen=True)
class ProcessCollectionReceipt:
    schema: str
    action_id: str
    collection_index: int
    trigger: str
    ordered_job_ids: tuple[str, ...]
    selected_job_id: str | None
    consumed_job_ids: tuple[str, ...]
    carryforward_job_ids: tuple[str, ...]
    rng_before: Blob
    rng_after: Blob
    collection: CandidateCollection


_PAYLOAD_CODEC = DataclassCodec(
    ProcessCandidatePayload,
    "astrolabe.process-candidate-payload/v1",
)
_RECEIPT_CODEC = DataclassCodec(
    ProcessCandidateReceipt,
    "astrolabe.process-candidate-receipt/v1",
    nested_types=(CandidateResult, CandidateTask, ProcessCandidatePayload),
)
_COLLECTION_CODEC = DataclassCodec(
    ProcessCollectionReceipt,
    "astrolabe.process-collection-receipt/v1",
    nested_types=(
        CollectionAction,
        CandidateCollection,
        CandidateResult,
        CandidateTask,
        ProcessCandidatePayload,
    ),
)
_STATE_CODEC = DataclassCodec(
    CollectionState,
    "astrolabe.collection-state/v1",
    nested_types=(
        CandidateCollection,
        CandidateResult,
        CandidateTask,
        ProcessCandidatePayload,
    ),
)
_ACTION_CODEC = DataclassCodec(
    CollectionAction,
    "astrolabe.collection-action/v1",
    nested_types=(CandidateTask, ProcessCandidatePayload),
)
_EVENT_CODEC = DataclassCodec(
    CollectionEvent,
    "astrolabe.collection-event/v1",
    nested_types=(
        CollectionAction,
        CandidateCollection,
        CandidateResult,
        CandidateTask,
        ProcessCandidatePayload,
    ),
)
_JSON_CODEC = JSONCodec("astrolabe.process-candidate-run-identity/v1")


def _write_candidate_receipt(path: Path, receipt: ProcessCandidateReceipt) -> None:
    encoded = _RECEIPT_CODEC.encode(receipt)
    _write_json_atomic(path, json.loads(encoded.decode("utf-8")))


def _read_candidate_receipt(path: Path) -> ProcessCandidateReceipt:
    return _RECEIPT_CODEC.decode(path.read_bytes())


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    """Durably replace one JSON receipt before its state is acknowledged."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp"
    )
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _process_identity(pid: int | None = None) -> dict[str, Any] | None:
    target = os.getpid() if pid is None else int(pid)
    proc = Path("/proc") / str(target)
    try:
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="utf-8"
        ).strip()
        stat_fields = (proc / "stat").read_text(encoding="utf-8").split()
        start_ticks = int(stat_fields[21])
    except (FileNotFoundError, IndexError, OSError, ValueError):
        return None
    return {
        "pid": target,
        "boot_id": boot_id,
        "start_ticks": start_ticks,
    }


def _owned_process_alive(identity: dict[str, Any] | None) -> bool:
    if not isinstance(identity, dict):
        return False
    current = _process_identity(identity.get("pid"))
    return current == identity


def _source_manifest() -> tuple[dict[str, Any], ...]:
    root = _repo_root()
    paths = set((root / "astrolabe").rglob("*.py"))
    paths.update({root / "run.py", root / "scripts" / "astrolabe"})
    files: list[dict[str, Any]] = []
    for path in sorted(item for item in paths if item.is_file()):
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return tuple(files)


def _asset_manifest(cfg: Any) -> dict[str, Any]:
    from astrolabe.design.configured_refinement import (
        _asset_manifest_record,
        _selected_asset_manifest,
    )

    record = _asset_manifest_record(_selected_asset_manifest(cfg))
    return dict(record)


def _scientific_config(cfg: Any) -> dict[str, Any]:
    from astrolabe.design.configured_refinement import (
        _scientific_config_payload,
    )

    return dict(_scientific_config_payload(cfg))


def _candidate_content_id(
    *,
    job_id: str,
    generation: int,
    attempt: int,
    actual_seed: int,
    task_name: str,
    out_dir: str,
    final_config: dict[str, Any],
    scientific_config: dict[str, Any],
    source_manifest: tuple[dict[str, Any], ...],
    input_assets_manifest: dict[str, Any],
) -> str:
    return _digest(
        {
            "job_id": job_id,
            "generation": int(generation),
            "attempt": int(attempt),
            "actual_seed": int(actual_seed),
            "task_name": task_name,
            "out_dir": out_dir,
            "final_config": final_config,
            "scientific_config": scientific_config,
            "source_manifest": list(source_manifest),
            "input_assets_manifest": input_assets_manifest,
        }
    )


def _load_base_cfg(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"candidate batch config not found: {path}")
    try:
        return OmegaConf.load(path)
    except Exception as exc:
        raise CandidateBatchConfigError(
            f"failed to load candidate batch config {path}: {exc}"
        ) from exc


def _find_scores_csv(out_dir: Path, task_name: str) -> Path:
    task_root = out_dir / task_name
    candidates = sorted(task_root.rglob("scores.csv"))
    if not candidates:
        raise FileNotFoundError(f"No scores.csv found under {task_root}")
    return candidates[-1]


def _best_score_row(scores_csv: Path) -> dict[str, str]:
    with scores_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in scores.csv: {scores_csv}")

    def _score(row: dict[str, str]) -> float:
        value = row.get("weighted_score")
        if value in (None, ""):
            return float("-inf")
        return float(value)

    accepted = [
        row
        for row in rows
        if str(row.get("accepted", "")).strip().lower() in {"true", "1", "yes"}
    ]
    return max(accepted or rows, key=_score)


def _extract_sequences(row: dict[str, str]) -> dict[str, str]:
    sequences: dict[str, str] = {}
    for key, value in row.items():
        if not key.endswith("_sequence") or not value:
            continue
        chain_key = key[: -len("_sequence")]
        chain = chain_key[1:] if chain_key[:1] in {"p", "r", "d", "s", "c"} else chain_key
        if chain:
            sequences[chain] = value
    return sequences


class CandidateDecisionRun:
    """Compatibility adapter from candidate payloads to generic collection."""

    def __init__(
        self,
        *,
        pool_config: CandidatePoolConfig,
        worker: Callable[[CandidateTask], CandidateResult | float],
        decision_rounds: int,
        initial_sequences: dict[str, str] | None = None,
        selector: CandidateSelector | None = None,
        logger: logging.Logger | None = None,
        runtime: Runtime | None = None,
        receipts: ProcessCandidateReceipts | None = None,
        resumed: bool = False,
    ) -> None:
        self.pool_config = pool_config
        self.worker = worker
        self.num_candidates = int(pool_config.num_candidates)
        self.decision_rounds = int(decision_rounds)
        self.current_sequences = dict(initial_sequences or {})
        self.next_candidate = 1
        self.selector = selector
        self.logger = logger
        self.runtime = runtime
        self.receipts = receipts
        self.resumed = bool(resumed)

    def run(self) -> CandidateBatchRun:
        from astrolabe.app import Astrolabe
        from astrolabe.design.composition import ControlRun

        policy = CollectionPolicy(
            config=self.pool_config,
            decision_rounds=self.decision_rounds,
            initial_payload=dict(self.current_sequences),
            selector=self.selector,
            job_id_factory=lambda number: f"cand_{int(number):04d}",
            payload_factory=lambda payload: {
                "sequences": dict(payload or {})
            },
            adopt_selected=self._adopt_selected_payload,
        )
        if self.runtime is not None:
            if self.receipts is None:
                raise TypeError("Durable candidate collection requires receipts")
            trial_context = DurableCollectionTrial(
                runtime=self.runtime,
                config=self.pool_config,
                worker=self.worker,
                selector=policy.selector,
                receipts=self.receipts,
                action_codec=_ACTION_CODEC,
                result_codec=_EVENT_CODEC,
                logger=self.logger,
                resumed=self.resumed,
            )
            control_runtime = self.runtime
            state_codec = _STATE_CODEC
            request_codec = _ACTION_CODEC
            result_codec = _EVENT_CODEC
        else:
            trial_context = CollectionTrial(
                config=self.pool_config,
                worker=self.worker,
                selector=policy.selector,
                logger=self.logger,
            )
            control_runtime = None
            state_codec = None
            request_codec = None
            result_codec = None
        with trial_context as trial:
            control = ControlRun(
                policy=policy,
                trial=trial,
                initial_state=policy.initial_state(),
                runtime=control_runtime,
                state_codec=state_codec,
                request_codec=request_codec,
                result_codec=result_codec,
            )
            return Astrolabe(control).run()

    @staticmethod
    def _adopt_selected_payload(current_payload, selected):
        payload = selected.payload
        if isinstance(payload, ProcessCandidatePayload):
            return dict(payload.sequences)
        return current_payload


class ProcessCandidateOutputReader:
    def __init__(self, out_root: Path) -> None:
        self.out_root = Path(out_root)

    def read_best_score(self, task_name: str) -> ProcessCandidateScore:
        row = _best_score_row(_find_scores_csv(self.out_root, task_name))
        return ProcessCandidateScore(
            weighted_score=float(row["weighted_score"]),
            sequences=_extract_sequences(row),
        )


class ProcessCandidateWorker:
    def __init__(self, config: ProcessCandidateRunnerConfig, base_cfg: Any) -> None:
        self.config = config
        self.base_cfg = base_cfg
        self.output_reader = ProcessCandidateOutputReader(config.out_root)
        self._source_manifest: tuple[dict[str, Any], ...] | None = None
        self._input_assets_manifest: dict[str, Any] | None = None

    def execution_spec(self, task: CandidateTask) -> CandidateExecutionSpec:
        payload = dict(task.payload or {})
        sequences = dict(payload.get("sequences") or {})
        task_name = (
            f"{self.config.config_yaml.stem}__g{task.generation:03d}__"
            f"{task.candidate_id}"
        )
        work_root = self.config.out_root / "candidate_pool" / task.candidate_id
        cfg_path = work_root / "config.yaml"
        stdout_log = work_root / "stdout.log"
        stderr_log = work_root / "stderr.log"
        receipt_path = work_root / "candidate_receipt.json"
        seed_base = int(getattr(self.base_cfg, "seed", 0) or 0)
        actual_seed = seed_base + _candidate_seed_offset(task.candidate_id)
        final_cfg = OmegaConf.create(
            OmegaConf.to_container(self.base_cfg, resolve=False)
        )
        with open_dict(final_cfg):
            final_cfg.task_name = task_name
            final_cfg.out_dir = str(self.config.out_root)
            final_cfg.steps = int(self.config.candidate_steps)
            final_cfg.seed = int(actual_seed)
            if sequences:
                if "seqs" not in final_cfg or final_cfg.seqs is None:
                    final_cfg.seqs = {}
                for chain, sequence in sequences.items():
                    final_cfg.seqs[str(chain)] = str(sequence)
        final_config = OmegaConf.to_container(final_cfg, resolve=False)
        if not isinstance(final_config, dict):
            raise CandidateBatchConfigError("candidate config is not a mapping")
        if self.config.dry_run:
            self._source_manifest = self._source_manifest or ()
            self._input_assets_manifest = (
                self._input_assets_manifest
                or {"schema": "astrolabe.process-candidate-dry-run/v1"}
            )
            scientific_config = final_config
        else:
            if self._source_manifest is None:
                self._source_manifest = _source_manifest()
            if self._input_assets_manifest is None:
                self._input_assets_manifest = _asset_manifest(final_cfg)
            scientific_config = _scientific_config(final_cfg)
        content_id = _candidate_content_id(
            job_id=task.job_id,
            generation=int(task.generation),
            attempt=1,
            actual_seed=actual_seed,
            task_name=task_name,
            out_dir=str(self.config.out_root),
            final_config=final_config,
            scientific_config=scientific_config,
            source_manifest=self._source_manifest,
            input_assets_manifest=self._input_assets_manifest,
        )
        return CandidateExecutionSpec(
            job_id=task.job_id,
            generation=int(task.generation),
            attempt=1,
            actual_seed=actual_seed,
            task_name=task_name,
            out_dir=str(self.config.out_root),
            work_root=str(work_root),
            config_yaml=str(cfg_path),
            stdout_log=str(stdout_log),
            stderr_log=str(stderr_log),
            receipt_path=str(receipt_path),
            sequences=sequences,
            final_config=final_config,
            scientific_config=scientific_config,
            source_manifest=self._source_manifest,
            input_assets_manifest=self._input_assets_manifest,
            content_id=content_id,
        )

    def prepare(self, task: CandidateTask) -> ProcessCandidateReceipt:
        spec = self.execution_spec(task)
        path = Path(spec.receipt_path)
        if path.is_file():
            receipt = _read_candidate_receipt(path)
            self._validate_receipt(task, spec, receipt)
            config_path = Path(spec.config_yaml)
            if receipt.status == "admitted" and not config_path.is_file():
                final_cfg = OmegaConf.create(spec.final_config)
                config_path.parent.mkdir(parents=True, exist_ok=True)
                config_path.write_text(
                    OmegaConf.to_yaml(final_cfg, resolve=False),
                    encoding="utf-8",
                )
            return receipt
        final_cfg = OmegaConf.create(spec.final_config)
        Path(spec.config_yaml).parent.mkdir(parents=True, exist_ok=True)
        Path(spec.config_yaml).write_text(
            OmegaConf.to_yaml(final_cfg, resolve=False),
            encoding="utf-8",
        )
        receipt = ProcessCandidateReceipt(
            schema=CHILD_RECEIPT_SCHEMA,
            status="admitted",
            job_id=spec.job_id,
            generation=spec.generation,
            attempt=spec.attempt,
            actual_seed=spec.actual_seed,
            content_id=spec.content_id,
            task_name=spec.task_name,
            config_yaml=spec.config_yaml,
            out_dir=spec.out_dir,
            stdout_log=spec.stdout_log,
            stderr_log=spec.stderr_log,
            sequences=spec.sequences,
            final_config=spec.final_config,
            scientific_config=spec.scientific_config,
            source_manifest=spec.source_manifest,
            input_assets_manifest=spec.input_assets_manifest,
        )
        _write_candidate_receipt(path, receipt)
        return receipt

    def receipt(self, task: CandidateTask) -> ProcessCandidateReceipt | None:
        receipt_path = (
            self.config.out_root
            / "candidate_pool"
            / task.job_id
            / "candidate_receipt.json"
        )
        return (
            _read_candidate_receipt(receipt_path)
            if receipt_path.is_file()
            else None
        )

    @staticmethod
    def _claim_attempt(receipt: ProcessCandidateReceipt) -> Path:
        """Take the one exclusive outer claim before any wrapper is launched."""

        work_root = Path(receipt.config_yaml).parent
        path = work_root / f"candidate_claim_{receipt.attempt}.json"
        claim = {
            "schema": "astrolabe.process-candidate-claim/v1",
            "job_id": receipt.job_id,
            "generation": receipt.generation,
            "attempt": receipt.attempt,
            "actual_seed": receipt.actual_seed,
            "content_id": receipt.content_id,
            "claimant": _process_identity(),
            "claimed_unix_s": time.time(),
        }
        try:
            descriptor = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError as exc:
            raise OutcomeUnknown(
                f"Candidate {receipt.job_id} attempt {receipt.attempt} "
                "already has an exclusive outer claim"
            ) from exc
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(
                    claim,
                    handle,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            directory = os.open(work_root, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except BaseException:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            raise
        return path

    def _validate_receipt(
        self,
        task: CandidateTask,
        spec: CandidateExecutionSpec,
        receipt: ProcessCandidateReceipt,
    ) -> None:
        if receipt.schema != CHILD_RECEIPT_SCHEMA:
            raise ContractConflict("Unsupported candidate receipt schema")
        if (
            receipt.job_id != task.job_id
            or receipt.generation != task.generation
            or receipt.attempt != spec.attempt
            or receipt.actual_seed != spec.actual_seed
        ):
            raise ContractConflict(
                "Candidate receipt job, attempt or seed changed on resume"
            )
        if receipt.content_id != spec.content_id:
            raise ContractConflict(
                "Candidate config, source, weights or input assets changed "
                f"at {spec.work_root}"
            )
        if (
            receipt.final_config != spec.final_config
            or receipt.scientific_config != spec.scientific_config
            or receipt.source_manifest != spec.source_manifest
            or receipt.input_assets_manifest != spec.input_assets_manifest
        ):
            raise ContractConflict(
                "Candidate receipt content does not match its content identity"
            )

    def __call__(self, task: CandidateTask) -> CandidateResult:
        if self.config.dry_run:
            spec = self.execution_spec(task)
            config_path = Path(spec.config_yaml)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                OmegaConf.to_yaml(
                    OmegaConf.create(spec.final_config),
                    resolve=False,
                ),
                encoding="utf-8",
            )
            return CandidateResult(
                task=task,
                status="completed",
                score=0.0,
                payload=ProcessCandidatePayload(
                    config_yaml=spec.config_yaml,
                    out_dir=spec.out_dir,
                    task_name=spec.task_name,
                    stdout_log=spec.stdout_log,
                    stderr_log=spec.stderr_log,
                    sequences=spec.sequences,
                ),
            )
        receipt = self.prepare(task)
        path = Path(receipt.config_yaml).parent / "candidate_receipt.json"
        if receipt.status == "completed":
            if receipt.result is None:
                raise ContractConflict("Completed candidate receipt has no result")
            return receipt.result
        if receipt.status == "failed":
            if receipt.result is None:
                raise ContractConflict("Failed candidate receipt has no result")
            return receipt.result
        if receipt.status in {"issued", "unknown"}:
            raise OutcomeUnknown(
                f"Candidate {task.job_id} has no terminal owner proof"
            )
        if receipt.status != "admitted":
            raise ContractConflict(
                f"Unknown candidate receipt status {receipt.status!r}"
            )
        self._claim_attempt(receipt)
        wrapper_log = path.parent / "candidate_wrapper.stderr.log"
        cmd = [
            sys.executable,
            "-m",
            "astrolabe.experimental.candidate_child",
            "--receipt",
            str(path),
            "--runner",
            str(self.config.runner),
            "--timeout-s",
            (
                "none"
                if self.config.candidate_timeout_s is None
                else str(float(self.config.candidate_timeout_s))
            ),
        ]
        try:
            with wrapper_log.open("w", encoding="utf-8") as stderr:
                proc = subprocess.run(
                    cmd,
                    cwd=str(_repo_root()),
                    stdout=subprocess.DEVNULL,
                    stderr=stderr,
                    text=True,
                    check=False,
                    start_new_session=True,
                )
        except OSError as exc:
            failed = replace(
                receipt,
                status="failed",
                error=f"candidate wrapper launch failed: {exc}",
                result=CandidateResult(
                    task=task,
                    status="failed",
                    error=f"candidate wrapper launch failed: {exc}",
                    payload=ProcessCandidatePayload(
                        config_yaml=receipt.config_yaml,
                        out_dir=receipt.out_dir,
                        task_name=receipt.task_name,
                        stdout_log=receipt.stdout_log,
                        stderr_log=receipt.stderr_log,
                        sequences=receipt.sequences,
                    ),
                ),
            )
            _write_candidate_receipt(path, failed)
            return failed.result
        terminal = _read_candidate_receipt(path)
        self._validate_terminal_receipt(task, receipt, terminal)
        if terminal.status == "completed":
            if terminal.result is None:
                raise ContractConflict("Candidate wrapper completed without result")
            return terminal.result
        if terminal.status == "failed":
            if terminal.result is None:
                raise ContractConflict("Candidate wrapper failed without result")
            return terminal.result
        if terminal.status == "unknown":
            raise OutcomeUnknown(
                terminal.error
                or f"Candidate {task.job_id} has uncertain terminal outcome"
            )
        if proc.returncode != 0:
            failed = replace(
                receipt,
                status="failed",
                error=(
                    "candidate wrapper exited before issue "
                    f"rc={proc.returncode}; stderr={wrapper_log}"
                ),
                result=CandidateResult(
                    task=task,
                    status="failed",
                    error=(
                        "candidate wrapper exited before issue "
                        f"rc={proc.returncode}; stderr={wrapper_log}"
                    ),
                    payload=ProcessCandidatePayload(
                        config_yaml=receipt.config_yaml,
                        out_dir=receipt.out_dir,
                        task_name=receipt.task_name,
                        stdout_log=receipt.stdout_log,
                        stderr_log=receipt.stderr_log,
                        sequences=receipt.sequences,
                    ),
                ),
            )
            _write_candidate_receipt(path, failed)
            return failed.result
        raise OutcomeUnknown(
            f"Candidate wrapper returned no terminal receipt: {path}"
        )

    @staticmethod
    def _validate_terminal_receipt(
        task: CandidateTask,
        admitted: ProcessCandidateReceipt,
        terminal: ProcessCandidateReceipt,
    ) -> None:
        if terminal.content_id != admitted.content_id:
            raise ContractConflict("Candidate wrapper changed admitted content")
        if (
            terminal.job_id != task.job_id
            or terminal.generation != task.generation
            or terminal.actual_seed != admitted.actual_seed
            or terminal.attempt != admitted.attempt
        ):
            raise ContractConflict(
                "Candidate wrapper returned another job or attempt"
            )
        if terminal.result is not None:
            result_payload = dict(terminal.result.job.payload or {})
            if (
                terminal.result.job.job_id != task.job_id
                or terminal.result.job.generation != task.generation
                or dict(result_payload.get("sequences") or {})
                != dict(admitted.sequences)
            ):
                raise ContractConflict("Candidate wrapper returned another task")


class ProcessCandidateReceipts:
    """Candidate-specific single-writer receipts used by durable collection."""

    def __init__(
        self,
        runtime: Runtime,
        worker: ProcessCandidateWorker,
        *,
        resumed: bool,
    ) -> None:
        self.runtime = runtime
        self.worker = worker
        self.resumed = bool(resumed)
        for codec in (
            _RECEIPT_CODEC,
            _COLLECTION_CODEC,
            _JSON_CODEC,
            _PAYLOAD_CODEC,
        ):
            runtime.data.register(codec)
        source = _source_manifest()
        assets = _asset_manifest(worker.base_cfg)
        identity = {
            "schema": "astrolabe.process-candidate-run-identity/v1",
            "config_yaml": str(worker.config.config_yaml),
            "config_sha256": _sha256_file(worker.config.config_yaml),
            "source_manifest": list(source),
            "input_assets_manifest": assets,
        }
        identity["identity"] = _digest(identity)
        prior = runtime.load("collection/run-identity", _JSON_CODEC)
        if prior is not None and prior != identity:
            raise ContractConflict(
                "Candidate base config, source or selected input assets "
                "changed on resume"
            )
        if prior is None:
            runtime.save("collection/run-identity", identity, _JSON_CODEC)
        self.identity = identity

    @staticmethod
    def _job_name(job: Job) -> str:
        return "collection/job/" + job.job_id

    @staticmethod
    def _commit_name(action_id: str) -> str:
        return "collection/commit/" + action_id

    def _save_receipt(self, receipt: ProcessCandidateReceipt) -> None:
        task = CandidateTask(
            candidate_id=receipt.job_id,
            generation=receipt.generation,
            payload={"sequences": dict(receipt.sequences)},
        )
        self.runtime.save(
            self._job_name(task),
            receipt,
            _RECEIPT_CODEC,
        )

    def admission(self, job: Job) -> CandidateResult | None:
        spec = self.worker.execution_spec(job)
        file_receipt = self.worker.receipt(job)
        if file_receipt is not None:
            self.worker._validate_receipt(job, spec, file_receipt)
            self._save_receipt(file_receipt)
            if file_receipt.status in {"completed", "failed"}:
                if file_receipt.result is None:
                    raise ContractConflict(
                        f"Terminal receipt {job.job_id} has no result"
                    )
                return file_receipt.result
            if file_receipt.status in {"issued", "unknown"}:
                raise OutcomeUnknown(
                    f"Candidate {job.job_id} is issued without terminal proof"
                )
        durable = self.runtime.load(self._job_name(job), _RECEIPT_CODEC)
        if durable is not None:
            self.worker._validate_receipt(job, spec, durable)
            if durable.status in {"completed", "failed"}:
                if durable.result is None:
                    raise ContractConflict(
                        f"Terminal receipt {job.job_id} has no result"
                    )
                return durable.result
            if durable.status in {"issued", "unknown"}:
                raise OutcomeUnknown(
                    f"Candidate {job.job_id} is issued without terminal proof"
                )
            if durable.status != "admitted":
                raise ContractConflict(
                    f"Unsupported candidate receipt status {durable.status!r}"
                )
        receipt = self.worker.prepare(job)
        self.worker._validate_receipt(job, spec, receipt)
        self._save_receipt(receipt)
        if receipt.status in {"completed", "failed"}:
            if receipt.result is None:
                raise ContractConflict(
                    f"Terminal receipt {job.job_id} has no result"
                )
            return receipt.result
        if receipt.status in {"issued", "unknown"}:
            raise OutcomeUnknown(
                f"Candidate {job.job_id} is issued without terminal proof"
            )
        if receipt.status != "admitted":
            raise ContractConflict(
                f"Unsupported candidate receipt status {receipt.status!r}"
            )
        return None

    def terminal_observed(self, job: Job, result: CandidateResult) -> None:
        receipt = self.worker.receipt(job)
        if receipt is None:
            receipt = self.runtime.load(self._job_name(job), _RECEIPT_CODEC)
        if receipt is None:
            raise ContractConflict(
                f"Candidate {job.job_id} has no admitted receipt"
            )
        if receipt.status in {"completed", "failed"}:
            if receipt.result != result:
                raise ContractConflict(
                    f"Candidate {job.job_id} result differs from its receipt"
                )
            self._save_receipt(receipt)
            return
        unknown = replace(
            receipt,
            status="unknown",
            result=None,
            error=(
                "pool observed a result without a completed/failed child "
                f"receipt: {result.status}"
            ),
        )
        self._save_receipt(unknown)
        raise OutcomeUnknown(unknown.error)

    def collection_commit(self, commit) -> None:
        receipt = ProcessCollectionReceipt(
            schema=COLLECTION_RECEIPT_SCHEMA,
            action_id=commit.action_id,
            collection_index=commit.collection_index,
            trigger=commit.trigger,
            ordered_job_ids=commit.ordered_job_ids,
            selected_job_id=commit.selected_job_id,
            consumed_job_ids=commit.consumed_job_ids,
            carryforward_job_ids=commit.carryforward_job_ids,
            rng_before=commit.rng_before,
            rng_after=commit.rng_after,
            collection=commit.collection,
        )
        self.runtime.save(
            self._commit_name(commit.action_id),
            receipt,
            _COLLECTION_CODEC,
        )

    def committed_collection(
        self,
        action_id: str,
    ) -> tuple[CandidateCollection, Blob, Blob] | None:
        receipt = self.runtime.load(
            self._commit_name(action_id),
            _COLLECTION_CODEC,
        )
        if receipt is None:
            return None
        return receipt.collection, receipt.rng_before, receipt.rng_after

    def rehydrate(self) -> RehydratedCollection:
        consumed: set[str] = set()
        collection_index = 0
        records = self.runtime._records("checkpoint")
        for name in sorted(records):
            if not name.startswith("collection/commit/"):
                continue
            receipt = self.runtime.load(name, _COLLECTION_CODEC)
            if receipt is not None:
                consumed.update(receipt.consumed_job_ids)
                collection_index = max(
                    collection_index,
                    int(receipt.collection_index),
                )
        ready: list[CandidateResult] = []
        issued: list[CandidateTask] = []
        uncommitted_generations: set[int] = set()
        for name in sorted(records):
            if not name.startswith("collection/job/"):
                continue
            receipt = self.runtime.load(name, _RECEIPT_CODEC)
            if receipt is None or receipt.job_id in consumed:
                continue
            uncommitted_generations.add(int(receipt.generation))
            task = CandidateTask(
                candidate_id=receipt.job_id,
                generation=receipt.generation,
                payload={"sequences": dict(receipt.sequences)},
            )
            file_receipt = self.worker.receipt(task)
            if file_receipt is not None:
                if file_receipt.content_id != receipt.content_id:
                    raise ContractConflict(
                        f"Candidate {receipt.job_id} file receipt changed content"
                    )
                if file_receipt.status in {
                    "completed",
                    "failed",
                    "issued",
                    "unknown",
                }:
                    receipt = file_receipt
                    self._save_receipt(receipt)
            if receipt.status in {"completed", "failed"}:
                if receipt.result is None:
                    raise ContractConflict(
                        f"Terminal receipt {receipt.job_id} has no result"
                    )
                ready.append(receipt.result)
            elif receipt.status in {"issued", "unknown"}:
                issued.append(task)
            elif receipt.status == "admitted":
                # A later exact submit action may execute an admitted job.
                # A collect/drain action without a committed boundary stops.
                continue
            else:
                raise ContractConflict(
                    f"Unsupported candidate receipt status {receipt.status!r}"
                )
        return RehydratedCollection(
            ready_results=tuple(ready),
            issued_jobs=tuple(issued),
            uncommitted_generations=tuple(sorted(uncommitted_generations)),
            collection_index=collection_index,
        )


ProcessCandidateWorkerFactory = Callable[
    [ProcessCandidateRunnerConfig, Any],
    Callable[[CandidateTask], CandidateResult | float],
]


@dataclass(frozen=True)
class ProcessCandidateBatchAdapters:
    worker_factory: ProcessCandidateWorkerFactory = ProcessCandidateWorker
    selector: CandidateSelector = select_highest_score_candidate


def _candidate_seed_offset(candidate_id: str) -> int:
    suffix = str(candidate_id).rsplit("_", 1)[-1]
    if suffix.isdigit():
        return int(suffix)
    digest = hashlib.blake2b(str(candidate_id).encode("utf-8"), digest_size=4)
    return int.from_bytes(digest.digest(), "big")


def run_process_candidate_batch(
    config: ProcessCandidateRunnerConfig,
    *,
    adapters: ProcessCandidateBatchAdapters | None = None,
) -> dict[str, Any]:
    active_adapters = adapters or ProcessCandidateBatchAdapters()
    base_cfg = _load_base_cfg(config.config_yaml)
    pool_cfg = CandidatePoolConfig(
        num_candidates=config.num_candidates,
        min_completed=config.min_completed,
        max_wait_s=config.max_wait_s,
        max_in_flight=config.max_in_flight,
    )
    worker = active_adapters.worker_factory(config, base_cfg)
    durable = isinstance(worker, ProcessCandidateWorker) and not config.dry_run
    if config.resume and not durable:
        raise CandidateBatchConfigError(
            "Durable candidate resume requires ProcessCandidateWorker"
        )
    if durable:
        runtime_root = config.out_root / ".framework"
        journal_path = runtime_root / "journal.sqlite"
        if config.resume and not journal_path.is_file():
            raise FileNotFoundError(
                f"candidate resume journal not found: {journal_path}"
            )
        if not config.resume and journal_path.exists():
            raise CandidateBatchConfigError(
                "candidate output already contains a durable run; use --resume"
            )
        with Runtime(root=runtime_root) as runtime:
            receipts = ProcessCandidateReceipts(
                runtime,
                worker,
                resumed=config.resume,
            )
            run = CandidateDecisionRun(
                pool_config=pool_cfg,
                worker=worker,
                decision_rounds=config.decision_rounds,
                selector=active_adapters.selector,
                runtime=runtime,
                receipts=receipts,
                resumed=config.resume,
            ).run()
    else:
        run = CandidateDecisionRun(
            pool_config=pool_cfg,
            worker=worker,
            decision_rounds=config.decision_rounds,
            selector=active_adapters.selector,
        ).run()

    return {
        "config_yaml": str(config.config_yaml),
        "out_root": str(config.out_root),
        "runner": str(config.runner),
        "dry_run": bool(config.dry_run),
        "resume": bool(config.resume),
        "num_candidates": int(config.num_candidates),
        "min_completed": int(config.min_completed),
        "max_wait_s": float(config.max_wait_s),
        "max_in_flight": int(config.max_in_flight or config.num_candidates),
        "candidate_timeout_s": (
            None
            if config.candidate_timeout_s is None
            else float(config.candidate_timeout_s)
        ),
        "decision_rounds": int(config.decision_rounds),
        "collections": [collection.as_row() for collection in run.collections],
        "selected_candidate_ids": [
            collection.selected.candidate_id
            for collection in run.decision_collections
            if collection.selected is not None
        ],
        "drain_selected_candidate_ids": [
            collection.selected.candidate_id
            for collection in run.drain_collections
            if collection.selected is not None
        ],
        "failed_candidates": [
            failed
            for collection in run.collections
            for failed in collection.as_row()["failed_candidates"]
        ],
    }


def add_candidate_batch_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config-yaml", required=True, help="Base standalone GH YAML.")
    parser.add_argument("--out-root", required=True, help="Output root for candidate runs.")
    parser.add_argument(
        "--runner",
        default=str(_repo_root() / "scripts" / "gh_dev"),
        help="GH CLI runner executable (default: scripts/astrolabe).",
    )
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--min-completed", type=int, default=2)
    parser.add_argument("--max-wait-s", type=float, default=30.0)
    parser.add_argument("--max-in-flight", type=int, default=None)
    parser.add_argument("--decision-rounds", type=int, default=1)
    parser.add_argument("--candidate-steps", type=int, default=2)
    parser.add_argument(
        "--candidate-timeout-s",
        type=float,
        default=None,
        help=(
            "Optional per-candidate subprocess hard timeout. "
            "Unset by default so slow candidates can carry forward past collection windows."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Verify and continue an existing durable candidate batch.",
    )
    parser.add_argument("--format", choices=["text", "json"], default="text")
    return parser


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gh_candidate_batch_run",
        description="Experimental candidate-pool runner for process-isolated GH candidates.",
    )
    return add_candidate_batch_arguments(parser)


def _config_from_namespace(args: argparse.Namespace) -> ProcessCandidateRunnerConfig:
    return ProcessCandidateRunnerConfig(
        config_yaml=Path(args.config_yaml).expanduser().resolve(),
        out_root=Path(args.out_root).expanduser().resolve(),
        runner=Path(args.runner).expanduser().resolve(),
        num_candidates=int(args.num_candidates),
        min_completed=int(args.min_completed),
        max_wait_s=float(args.max_wait_s),
        max_in_flight=args.max_in_flight,
        decision_rounds=int(args.decision_rounds),
        candidate_steps=int(args.candidate_steps),
        candidate_timeout_s=args.candidate_timeout_s,
        dry_run=bool(args.dry_run),
        resume=bool(args.resume),
    )


def _print_summary(summary: dict[str, Any], output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    print(f"config_yaml: {summary['config_yaml']}")
    print(f"out_root: {summary['out_root']}")
    print(f"dry_run: {str(summary['dry_run']).lower()}")
    print(f"resume: {str(summary['resume']).lower()}")
    print(
        "pool: "
        f"num_candidates={summary['num_candidates']} "
        f"min_completed={summary['min_completed']} "
        f"max_wait_s={summary['max_wait_s']} "
        f"max_in_flight={summary['max_in_flight']} "
        f"candidate_timeout_s={summary['candidate_timeout_s']}"
    )
    for row in summary["collections"]:
        print(
            "collection: "
            f"index={row['collection_index']} trigger={row['trigger']} "
            f"selected={row['selected_candidate_id']} "
            f"completed={row['completed']} failed={row['failed']} "
            f"carried_forward={row['carried_forward']}"
        )
        for failed in row.get("failed_candidates", []):
            print(
                "failed_candidate: "
                f"collection={row['collection_index']} "
                f"candidate={failed['candidate_id']} "
                f"error={failed['error']}"
            )


def run_from_namespace(
    args: argparse.Namespace,
    *,
    parser: argparse.ArgumentParser | None = None,
) -> int:
    try:
        summary = run_process_candidate_batch(_config_from_namespace(args))
    except (CandidateExecutionError, CandidateBatchConfigError, FileNotFoundError) as exc:
        if parser is not None:
            parser.exit(1, f"error: {exc}\n")
        raise
    _print_summary(summary, str(args.format))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    return run_from_namespace(args, parser=parser)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
