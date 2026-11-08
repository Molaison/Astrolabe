"""Shared state/file boundary for native scientific adapters.

This is a compatibility owner. Search and Runtime never know native component
names or read mutable StandardScores objects.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import importlib
import threading
from typing import Callable

from astrolabe.model_execution import capture_random_state, restore_random_state, register_model_owner

from .legacy import LEGACY_REFERENCE_CODECS, LegacyContext, LegacyTranslation
from .native_scores import (
    NATIVE_SCORES_ENVELOPE_CODEC, NATIVE_SCORES_REFERENCE_CODECS, restore_scores,
)
from .store import DataStore, design_codec
from .types import Candidate, Evidence, Key, Blob, Sample, System, make_candidate, key_for


@dataclass(frozen=True)
class ModelInitializationRequest:
    model: Key


@dataclass(frozen=True)
class ModelReady:
    model: Key
    constructor_rng: Blob


class LazyNativeOwner:
    """A restartable resident model owner, separate from inference replay.

    Initialization is an explicit unpaid operation. It preserves original
    constructor order/RNG. On recovery only the readiness record is rebound;
    loading weights is deferred until a new, admitted inference needs them.
    """
    attached = False

    def __init__(self, factory: Callable, data: DataStore, identity: Key, *,
                 alias: str = "model", rng_modules=("numpy", "torch")):
        from .operations import operation
        self.factory, self.data, self.identity = factory, data, identity
        self._rng_modules = tuple(rng_modules)
        self._value = None
        self._cleanup_pending = False
        self._cleanup_running = False
        self._cleanup_error = None
        self._cleanup_lock = threading.RLock()
        self._constructor_rng = None
        self.initialization_request = ModelInitializationRequest(identity)
        self.initialization = operation(alias + "/initialize", ModelInitializationRequest,
            ModelReady, self._initialize, version="native-model-initialization/v1",
            model_identity=key_for((identity, self._rng_modules)), on_result=self._bind_ready)

    @property
    def ready(self):
        return self._constructor_rng is not None

    def _bind_ready(self, ready: ModelReady):
        if ready.model != self.identity:
            raise ValueError("Model initialization belongs to another implementation")
        self._constructor_rng = ready.constructor_rng

    def _prepare_rng(self):
        for module in self._rng_modules:
            importlib.import_module(module)

    def _initialize(self, request: ModelInitializationRequest):
        if self._cleanup_pending:
            raise RuntimeError("Native model cleanup is unresolved")
        if request.model != self.identity:
            raise ValueError("Initialization request belongs to another model")
        self._prepare_rng()
        before = self.data.put(capture_random_state(), design_codec)
        self._value = self.factory()
        register_model_owner(self)
        return ModelReady(self.identity, before)

    def get(self):
        if self._cleanup_pending:
            raise RuntimeError("Native model cleanup is unresolved")
        if self._value is not None:
            return self._value
        if self._constructor_rng is None:
            raise RuntimeError("Native model initialization must run before inference")
        self._prepare_rng()
        active = capture_random_state()
        try:
            restore_random_state(self.data.get(self._constructor_rng, design_codec))
            self._value = self.factory()
        finally:
            restore_random_state(active)
        register_model_owner(self)
        return self._value

    def shutdown(self):
        from astrolabe.runtime_support import cleanup_retry_allowed
        with self._cleanup_lock:
            value = self._value
            if value is None:
                return
            if self._cleanup_running:
                raise RuntimeError("Native model cleanup is already in progress")
            if self._cleanup_error is not None and not cleanup_retry_allowed():
                raise self._cleanup_error
            self._cleanup_pending = True
            self._cleanup_running = True
        try:
            close = getattr(value, "shutdown", None)
            if close is not None:
                close()
        except BaseException as error:
            with self._cleanup_lock:
                self._cleanup_running = False
                self._cleanup_error = error
            raise
        with self._cleanup_lock:
            # Retain the only owner reference until shutdown actually succeeds.
            self._value = None
            self._cleanup_pending = self._cleanup_running = False
            self._cleanup_error = None


def native_score_artifact(evidence: Evidence):
    snapshots = tuple(artifact for artifact in evidence.artifacts
                      if artifact.content.codec == NATIVE_SCORES_ENVELOPE_CODEC.identity)
    if len(snapshots) != 1:
        raise ValueError("Native compatibility requires exactly one native-score snapshot artifact")
    snapshot, = snapshots
    if snapshot.candidate != evidence.evaluated_candidate:
        raise ValueError("Native-score snapshot belongs to another evaluated candidate")
    return snapshot


def numeric_step(sample: Sample) -> int:
    """Numeric native job index; scientific schedules may supply their own step."""
    if not sample.index.isdecimal():
        raise ValueError("Native compatibility requires a numeric Sample.index or an explicit owner step mapping")
    return int(sample.index)


@dataclass(frozen=True)
class NativeBindings:
    context: LegacyContext
    translation: LegacyTranslation
    data: DataStore
    work_root: Path

    def __post_init__(self):
        for codec in LEGACY_REFERENCE_CODECS + NATIVE_SCORES_REFERENCE_CODECS:
            self.data.register(codec)

    def validate(self, system: System, candidate: Candidate):
        if system != self.translation.system or candidate.system != system.identity:
            raise ValueError("Native component is bound to another molecular System")
        if make_candidate(system, candidate.assignments) != candidate:
            raise ValueError("Candidate identity does not match its complete assignments")
        system.edit(self.translation.candidate, candidate.assignments, data=self.data)

    def activate(self, candidate: Candidate):
        self.validate(self.translation.system, candidate)
        return self.context.activate(candidate, translation=self.translation, data=self.data)

    def scores_for(self, evidence: Evidence, destination: Path | str | None = None):
        snapshot = native_score_artifact(evidence)
        target = (Path(destination) if destination is not None else
                  self.work_root / "native-inputs" / evidence.identity.digest)
        return restore_scores(snapshot.content, self.data, target)
