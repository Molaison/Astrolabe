"""Spawn-pool entry points for prediction and whole-candidate CPU Score.

The parent owns proposals, member commits and search state. Each spawned process
owns one evaluator session and inherits the parent's GPU broker environment.
An issued request without a complete receipt is never automatically replayed.
"""

from __future__ import annotations

import atexit
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
import hashlib
import fcntl
import json
import logging
from multiprocessing.util import Finalize
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
import threading
import time
from typing import Any

from .legacy import LEGACY_REFERENCE_CODECS, LegacyContext
from .native_refinement import REFINEMENT_REFERENCE_CODECS
from .native_scores import NATIVE_SCORES_ENVELOPE_CODEC, restore_scores, snapshot_scores
from .native_refinement_runtime import ScoreReceipt, _SCORE_RECEIPT_CODEC
from .runtime import ContractConflict, OutcomeUnknown
from .store import ArrayCodec, DataStore, JSONCodec, design_codec
from .types import Blob, Candidate, Key


_SCHEMA = "astrolabe.refinement-worker.receipt/v1"
_WORKER: _Worker | None = None
_RESULT_KEYS = {
    "raw_scores", "worker_pid", "constructor_count", "timing", "request", "schema",
    "prediction_wall_s", "job_raw_dir", "sidecars",
    "prediction_after_rng",
    "score_receipt", "score_worker_pid", "score_sidecars", "score_raw_dir",
}


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _blob(record: dict[str, Any]) -> Blob:
    if not isinstance(record, dict) or set(record) != {"identity", "codec"}:
        raise ValueError("expected a dataclasses.asdict Blob")
    identity = record["identity"]
    if not isinstance(identity, dict) or set(identity) != {"digest"}:
        raise ValueError("expected a dataclasses.asdict Key")
    if not isinstance(identity["digest"], str) or not isinstance(record["codec"], str):
        raise ValueError("Blob digest and codec must be strings")
    return Blob(Key(identity["digest"]), record["codec"])


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _complete(path: Path, receipt: dict[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".receipt-")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(_json(receipt))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_dir(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sidecars(root: Path, *, sync: bool = False) -> list[dict[str, Any]]:
    manifest = []
    directories = {root}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ContractConflict(f"Prediction sidecar must be a private regular file: {path}")
        if not stat.S_ISREG(path.lstat().st_mode):
            continue
        digest, size = hashlib.sha256(), 0
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                size += len(chunk)
                digest.update(chunk)
            if sync:
                os.fsync(handle.fileno())
        manifest.append({"path": str(path), "sha256": digest.hexdigest(), "size_bytes": size})
        directories.update(path.parents)
    if sync:
        for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
            if directory == root or root in directory.parents:
                _fsync_dir(directory)
    return manifest


def _private_score_paths(value, raw_root, score_root):
    """Rebind sidecar references while retaining their native container types."""
    from astrolabe.evaluators.utils import StandardScores

    if isinstance(value, (str, Path)):
        path = Path(value)
        if path.is_absolute() and path.is_relative_to(raw_root):
            target = score_root / path.relative_to(raw_root)
            return str(target) if isinstance(value, str) else target
        return value
    if isinstance(value, dict):
        return {key: _private_score_paths(item, raw_root, score_root) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_private_score_paths(item, raw_root, score_root) for item in value)
    if isinstance(value, StandardScores):
        for name, item in vars(value).items():
            setattr(value, name, _private_score_paths(item, raw_root, score_root))
    return value


class _Worker:
    def __init__(self, config_yaml, data_root, assemblies, state, output_root):
        from omegaconf import OmegaConf, open_dict
        from astrolabe.app import _load_run_module
        from astrolabe.core import global_config

        started = time.perf_counter()
        self.pid = os.getpid()
        self.lock = threading.Lock()
        self.closed = False
        self.evaluators = {}
        self.constructor_count = 0
        self.finalizer = Finalize(self, self.shutdown, exitpriority=10)
        atexit.register(self.shutdown)
        self.root = Path(output_root).expanduser().resolve()
        self.receipts = self.root / "receipts"
        self.receipts.mkdir(parents=True, exist_ok=True)
        self.assemblies = tuple(assemblies)
        self.data = DataStore(Path(data_root))
        self.codecs = {
            codec.identity: codec
            for codec in (
                *REFINEMENT_REFERENCE_CODECS, *LEGACY_REFERENCE_CODECS,
                design_codec, ArrayCodec(), JSONCodec(),
                _SCORE_RECEIPT_CODEC,
            )
        }
        for codec in self.codecs.values():
            self.data.register(codec)
        self.run = _load_run_module()
        global_config.reset()
        self.session_root = self.root / "workers" / str(self.pid)
        self.session_root.mkdir(parents=True, exist_ok=True)
        cfg = OmegaConf.create(config_yaml)
        cfg._set_flag("allow_objects", True)
        with open_dict(cfg):
            cfg.out_dir = self.session_root
            cfg.raw_model_results_dir = self.session_root / "raw_model_results"
        try:
            self.args = self.run.parse_arguments(
                cfg, existing_run_dir=self.session_root, persist_config=False
            )
            prepared = self.run.prepare_pipeline_state(self.args, apply_initialization=False)
            # requires_mutators initializers report pending even when replay was
            # disabled. Deliberately bypass load_pipeline_components/initializer.
            if prepared.initializer_pending:
                global_config.init_const("initializer_result", None)
            self.loss, self.stages = self.run._initialize_pipeline_math(self.args, prepared)
            if tuple(self.stages.current_oligo) != self.assemblies:
                raise ContractConflict("worker assemblies differ from configured stage")
            alias = self.stages.execution_setting.evaluator
            self.evaluators = self.run.construct_evaluators(
                self.args.evaluator,
                self.args.raw_model_results_dir,
                self.stages.all_oligos,
                enabled_aliases=[alias],
            )
            self.constructor_count += 1
            global_config.init_const("evaluators", self.evaluators)
            self.evaluator = self.stages.current_evaluator
            self.run._initialize_pipeline_inputs(self.args, self.loss, self.stages)
            self.context = LegacyContext()
            self.translation = self.context.capture_candidate(
                assemblies=self.assemblies, state=state, data=self.data
            )
            self.runtime_config = {
                "mutator_stack_pre_update": False,
                "state_conflict_diagnostics": getattr(
                    getattr(self.args, "mutator", None), "state_conflict_diagnostics", None,
                ),
            }
            self.initialization_seconds = time.perf_counter() - started
        except BaseException:
            self.shutdown()
            raise

    def shutdown(self):
        if self.closed:
            return
        self.closed = True
        seen = set()
        for evaluator in self.evaluators.values():
            if id(evaluator) in seen:
                continue
            seen.add(id(evaluator))
            try:
                close = getattr(evaluator, "shutdown", None)
                if close is not None:
                    close()
            except Exception:
                logging.exception("refinement worker evaluator shutdown failed")

    @contextmanager
    def _output(self, root: Path, seed: int):
        """Redirect global and evaluator-held output paths without model reload."""
        from omegaconf import open_dict
        from astrolabe.core import global_config

        raw = root / "raw_model_results"
        for directory in (raw, root / "structures", root / "sequences"):
            directory.mkdir(parents=True, exist_ok=True)
        previous = (self.args.out_dir, self.args.raw_model_results_dir, self.args.seed)
        global_previous = (global_config.out_dir, global_config.raw_model_results_dir)
        paths = []
        try:
            with open_dict(self.args):
                self.args.out_dir, self.args.raw_model_results_dir, self.args.seed = root, raw, seed
            global_config.update_const("out_dir", root)
            global_config.update_const("raw_model_results_dir", raw)
            # Native evaluator implementations retain their constructor output_dir.
            # Preserve backend-specific suffixes inside the request directory.
            for evaluator in self.evaluators.values():
                for attribute in (
                    "output_dir", "raw_model_results_dir",
                    "_raw_model_results_dir", "_structure_dir",
                ):
                    value = getattr(evaluator, attribute, None)
                    if value is not None:
                        relative = Path(value).relative_to(Path(previous[1]))
                        paths.append((evaluator, attribute, value))
                        destination = raw / relative
                        setattr(evaluator, attribute, destination)
                        destination.mkdir(parents=True, exist_ok=True)
            yield raw
        finally:
            for evaluator, attribute, value in paths:
                setattr(evaluator, attribute, value)
            with open_dict(self.args):
                self.args.out_dir, self.args.raw_model_results_dir, self.args.seed = previous
            global_config.update_const("out_dir", global_previous[0])
            global_config.update_const("raw_model_results_dir", global_previous[1])

    def _read(self, path, task):
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
            if _json(receipt["request"]) != _json(task):
                raise ContractConflict("worker request hash has conflicting request identity")
            if any(_json(receipt[key]) != _json(value) for key, value in task.items()):
                raise ContractConflict("worker receipt task identity changed")
            if receipt["schema"] != _SCHEMA:
                raise ValueError("unknown worker receipt schema")
            scores = _blob(receipt["raw_scores"])
            if scores.codec != NATIVE_SCORES_ENVELOPE_CODEC.identity:
                raise ValueError("receipt does not contain native raw scores")
            # Validate every referenced blob from disk, not the process cache.
            data = DataStore(self.data.root)
            for codec in self.codecs.values():
                data.register(codec)
            after_rng = _blob(receipt["prediction_after_rng"])
            if after_rng.codec != design_codec.identity:
                raise ValueError("invalid prediction RNG codec")
            pending, seen = [scores, after_rng], set()
            if "score_receipt" in receipt:
                reference = _blob(receipt["score_receipt"])
                score = data.get(reference, _SCORE_RECEIPT_CODEC)
                if (score.request_identity != Key(task["_request_hash"])
                        or score.raw_scores != scores
                        or score.scoring_identity != Key(task["scoring_identity"])):
                    raise ContractConflict("Score receipt identity changed")
                score_raw = self.root / "measurements" / task["_request_hash"] / "raw_model_results"
                if receipt["score_raw_dir"] != str(score_raw) or not score_raw.is_dir():
                    raise ValueError("invalid Score output directory")
                if _json(receipt["score_sidecars"]) != _json(_sidecars(score_raw)):
                    raise ValueError("Score sidecars changed after completion")
                pending.append(reference)
            while pending:
                reference = pending.pop()
                if reference in seen:
                    continue
                seen.add(reference)
                value = data.get(reference)
                pending.extend(self.codecs[reference.codec].references(value))
            if type(receipt["worker_pid"]) is not int or receipt["constructor_count"] != 1:
                raise ValueError("incomplete worker construction metadata")
            if set(receipt["timing"]) != {"initialization_seconds", "evaluation_seconds", "persistence_seconds", "total_seconds"}:
                raise ValueError("incomplete worker timing metadata")
            raw_dir = self.root / "requests" / task["_request_hash"] / "raw_model_results"
            if receipt["job_raw_dir"] != str(raw_dir) or not raw_dir.is_dir():
                raise ValueError("invalid job raw directory")
            if _json(receipt["sidecars"]) != _json(_sidecars(raw_dir)):
                raise ValueError("worker sidecars changed after completion")
            if receipt["prediction_wall_s"] != receipt["timing"]["evaluation_seconds"]:
                raise ValueError("inconsistent prediction timing")
            _json(receipt)
            return receipt
        except ContractConflict:
            raise
        except Exception as error:
            raise OutcomeUnknown(f"incomplete worker receipt: {path}") from error

    def _score(self, task, raw_receipt, complete):
        with complete.with_suffix(".lock").open("a") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise OutcomeUnknown("CPU Score still has a live owner") from error
            return self._compute_score(task, raw_receipt, complete)

    def _compute_score(self, task, raw_receipt, complete):
        from astrolabe.core import global_config
        from astrolabe.model_execution import capture_random_state, restore_random_state
        from .configured_refinement import measure_configured_scores

        if complete.exists():
            return self._read(complete, task)
        candidate = self.data.get(_blob(task["candidate_blob"]), design_codec)
        root = self.root / "measurements" / task["_request_hash"]
        # A retry starts from immutable raw evidence, never a half-written view.
        if root.exists():
            shutil.rmtree(root)
        raw = root / "raw_model_results"
        shutil.copytree(raw_receipt["job_raw_dir"], raw)
        scores = restore_scores(_blob(raw_receipt["raw_scores"]), self.data, root / "structures")
        _private_score_paths(scores, Path(raw_receipt["job_raw_dir"]), raw)
        runtime_config = dict(self.runtime_config)
        if task.get("score_context") is not None:
            envelope = restore_scores(_blob(task["score_context"]), self.data, root / "context")
            runtime_config = envelope.configured_runtime_config
            runtime_config = _private_score_paths(runtime_config, Path(raw_receipt["job_raw_dir"]), raw)
        order = task.get("loss_order")
        if order is not None:
            if (not isinstance(order, list) or len(order) != len(set(order))
                    or set(order) != set(self.loss.loss_specs)):
                raise ContractConflict("Worker loss terms differ from the parent registry order")
            self.loss.loss_specs = {name: self.loss.loss_specs[name] for name in order}
            self.loss.loss_names = list(order)
        self.loss._clear_score_cache()
        lease = nullcontext()
        if os.environ.get("ASTROLABE_RESOURCE_SOCKET"):
            from astrolabe.runtime.admission import ResourceClient
            lease = ResourceClient.from_env().lease(
                resource="cpu", backend="configured-score", phase="score",
                identity=task["_request_hash"],
            )
        with lease:
            with self.context.activate(candidate, translation=self.translation, data=self.data):
                with self._output(root, task["seed"]):
                    restore_random_state(self.data.get(_blob(raw_receipt["prediction_after_rng"]), design_codec))
                    global_config.set_variable("step", task["step"])
                    self.run._ensure_evaluation_complex_poses(self.args, scores, self.loss)
                    try:
                        measurement = measure_configured_scores(
                            self.loss, scores, runtime_config, self.assemblies,
                        )
                    finally:
                        # Poses are process-local scratch, not transport state.
                        scores.complex_poses = None
                    measured = snapshot_scores(scores, self.data)
                    after_rng = self.data.put(capture_random_state(), design_codec)
                    reference = self.data.put(ScoreReceipt(
                        Key(task["_request_hash"]), _blob(raw_receipt["raw_scores"]),
                        Key(task["scoring_identity"]), measured, measurement, after_rng,
                    ), _SCORE_RECEIPT_CODEC)
                    self.data.persist((reference,))
                    sidecars = _sidecars(raw, sync=True)
        receipt = {
            **raw_receipt, "score_receipt": asdict(reference),
            "score_worker_pid": self.pid, "score_raw_dir": str(raw),
            "score_sidecars": sidecars,
        }
        _complete(complete, receipt)
        return receipt

    def execute(self, task):
        from astrolabe.argparser import set_random_seeds
        from astrolabe.core import global_config
        from astrolabe.model_execution import capture_random_state

        started = time.perf_counter()
        if self.closed or os.getpid() != self.pid:
            raise RuntimeError("worker must be initialized in this spawned process")
        # Take an independent, JSON-only request snapshot before any side effects.
        task = json.loads(_json(task))
        required = {"id", "kind", "generation", "parent_id", "seed", "sequence", "step", "_request_hash", "candidate_blob", "candidate_identity", "system_identity", "scoring_identity"}
        if not isinstance(task, dict) or required - task.keys() or _RESULT_KEYS & task.keys():
            raise ValueError("invalid or incomplete worker task")
        request_hash = task["_request_hash"]
        if not isinstance(request_hash, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", request_hash):
            raise ValueError("invalid worker request hash")
        if task["kind"] != "evaluate":
            raise ValueError("worker only accepts evaluate tasks")
        if type(task["step"]) is not int or task["step"] < 0:
            raise ValueError("worker step must be a nonnegative integer")
        if type(task["seed"]) is not int or not 1 <= task["seed"] <= 2**32 - 1:
            raise ValueError("worker seed must be in 1..2**32-1")
        with self.lock:
            complete = self.receipts / f"{request_hash}.json"
            issued = self.receipts / f"{request_hash}.issued.json"
            scored = self.receipts / f"{request_hash}.score.json"
            if complete.exists():
                raw_receipt = self._read(complete, task)
                return self._score(task, raw_receipt, scored)
            if issued.exists():
                try:
                    prior = json.loads(issued.read_text(encoding="utf-8"))
                    if not isinstance(prior, dict) or "request" not in prior:
                        raise ValueError("incomplete issued receipt")
                except Exception as error:
                    raise OutcomeUnknown(str(issued)) from error
                if _json(prior.get("request")) != _json(task):
                    raise ContractConflict("issued worker request identity changed")
                raise OutcomeUnknown(str(issued))
            candidate = self.data.get(_blob(task["candidate_blob"]), design_codec)
            if not isinstance(candidate, Candidate):
                raise TypeError("worker candidate_blob does not contain a Candidate")
            for name, key in (("candidate_identity", candidate.identity), ("system_identity", candidate.system)):
                if task[name] not in (key.digest, asdict(key)):
                    raise ContractConflict(f"worker {name} mismatch")
            self.translation.system.edit(candidate, (), data=self.data)
            try:
                descriptor = os.open(issued, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError as error:
                raise OutcomeUnknown(str(issued)) from error
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(_json({"request": task, "worker_pid": self.pid}))
                handle.flush()
                os.fsync(handle.fileno())
            _fsync_dir(self.receipts)
            try:
                with self.context.activate(candidate, translation=self.translation, data=self.data):
                    with self._output(self.root / "requests" / request_hash, task["seed"]) as raw:
                        set_random_seeds(task["seed"])
                        global_config.set_variable("step", task["step"])
                        evaluation_started = time.perf_counter()
                        skip_poses = self.args.skip_complex_poses
                        try:
                            self.args.skip_complex_poses = True
                            scores = self.run.perform_evaluation(
                                self.args, self.evaluator,
                                {"step": task["step"], "raw_model_results_dir": str(raw)},
                                task["step"], list(self.assemblies), loss_computer=self.loss,
                            )
                        finally:
                            self.args.skip_complex_poses = skip_poses
                        after_rng = self.data.put(capture_random_state(), design_codec)
                        evaluation_seconds = time.perf_counter() - evaluation_started
                        persistence_started = time.perf_counter()
                        scores_blob = snapshot_scores(scores, self.data)
                        self.data.persist((scores_blob, after_rng))
                        sidecars = _sidecars(raw, sync=True)
                        persistence_seconds = time.perf_counter() - persistence_started
                receipt = {
                    **task, "request": task, "schema": _SCHEMA,
                    "raw_scores": asdict(scores_blob), "worker_pid": self.pid,
                    "prediction_after_rng": asdict(after_rng),
                    "constructor_count": self.constructor_count,
                    "prediction_wall_s": evaluation_seconds,
                    "job_raw_dir": str(raw), "sidecars": sidecars,
                    "timing": {
                        "initialization_seconds": self.initialization_seconds,
                        "evaluation_seconds": evaluation_seconds,
                        "persistence_seconds": persistence_seconds,
                        "total_seconds": time.perf_counter() - started,
                    },
                }
                _complete(complete, receipt)
            except Exception as error:
                raise OutcomeUnknown(f"worker evaluation outcome unknown: {issued}") from error
            return self._score(task, receipt, scored)


def initializer(config_yaml: str, data_root: str, assemblies: tuple[str, ...], state: str, output_root: str) -> None:
    """Initialize once through a multiprocessing ``spawn`` pool initializer."""
    global _WORKER
    if _WORKER is not None and not _WORKER.closed:
        raise RuntimeError("refinement worker already initialized")
    _WORKER = _Worker(config_yaml, data_root, assemblies, state, output_root)


def execute(task: dict) -> dict:
    """Evaluate and score, or resume only Score from verified raw evidence."""
    if _WORKER is None:
        raise RuntimeError("refinement worker initializer has not run")
    return _WORKER.execute(task)


# The pool owner uses this spelling; keep initializer for direct callers.
initialize = initializer
