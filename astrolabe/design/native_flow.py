"""The default native scientific composition, driven by the common coordinator.

The existing scientific owners retain their stage, constraint and acceptance
semantics. Live native state has process lifetime: it is never pickled, silently
dropped, or advertised as a restartable checkpoint. Portable compositions use
the same coordinator with the durable native operation adapters instead.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import hashlib
import inspect
import json
import logging
import math
from pathlib import Path

from . import coordinator
from .legacy import LEGACY_REFERENCE_CODECS, standard_scores_to_evidence, translate_legacy_state
from .operations import Components, operation
from .owner import SCIENTIFIC_OWNER
from .runtime import ContractConflict, Runtime
from .store import DataStore
from .types import (
    Assessment, Blob, Continuation, Dependency, Scored, TrialOutcome, key_for, manifest,
)


_ACTIVE = ContextVar("astrolabe_native_scientific_composition", default=None)
_BOUNDARIES = {
    "initial": "",
    "prediction": "native_prediction",
    "mutation": "native_proposal",
    "prepare": "",
    "decision": "",
    "stage": "",
}


def native_call(phase, function, *args, **kwargs):
    """Record a bounded native scientific operation in the active composition."""
    owner = _ACTIVE.get()
    if owner is None:
        return function(*args, **kwargs)
    return owner.call(phase, function, args, kwargs)


def native_phase(phase):
    """Keep standalone owner functions callable outside a design run."""
    if phase not in _BOUNDARIES:
        raise ValueError(f"Unknown scientific phase {phase!r}")
    def decorate(function):
        @wraps(function)
        def invoke(*args, **kwargs):
            return native_call(phase, function, *args, **kwargs)
        return invoke
    return decorate


@dataclass(frozen=True)
class NativeInvocation:
    arguments: Blob
    revision: int


@dataclass(frozen=True)
class NativeReceipt:
    result: Blob
    revision: int


def _translate(stage_manager, data):
    return translate_legacy_state(assemblies=stage_manager.all_oligos, data=data)


def _project(scores, translation, data, resources, decisions, stage_manager, revision, history):
    candidate = translation.candidate
    resource = resources.retain(scores, label="native-scores", candidate=candidate.identity,
                                revision=revision)
    evidence = standard_scores_to_evidence(scores, translation.system, candidate, data,
        state=translation.state, process_resource=resource)
    # Decisions already computed this value under the active objective. A
    # projection must not call the loss or draw acceptance RNG a second time.
    value = float(decisions.scores[decisions.last_accepted])
    if not math.isfinite(value):
        raise ValueError(
            "the optimized weighted objective must be finite; missing auxiliary "
            "measurements cannot mask a non-finite decision score"
        )
    stage = getattr(stage_manager, "current_stage", "default")
    from omegaconf import OmegaConf
    losses = getattr(getattr(stage_manager, "loss_computer", None), "defined_losses", {})
    if OmegaConf.is_config(losses):
        losses = OmegaConf.to_container(losses, resolve=True)
    effective = {name: getattr(stage_manager, name, None) for name in (
        "current_weighted_score_expression", "current_acceptance_gate",
        "current_acceptance_threshold", "current_early_stopping",
        "current_prefix_early_stopping", "current_ignored_complexes",
        "current_negative_design_complexes", "current_transition_consistency_gate",
    )}
    objective = key_for(("native-stage-objective/v2", stage, effective, losses))
    inputs = manifest((Dependency("evidence", evidence.identity),
                       Dependency("objective", objective), Dependency("history", history.revision)))
    return Scored(candidate, evidence, Assessment(evidence.identity, objective, history.revision,
                                                 (value,), (), inputs))


class NativeFlow:
    """One session owner; root and a bounded advance contain no search loop."""

    fail_fast = True

    def __init__(self, args, loss_computer, stage_manager, runtime_config, *,
                 initial_evaluation_result, initial_loader, initialize, advance, data):
        from .process_resources import ProcessResources
        self.args, self.loss = args, loss_computer
        self.stages, self.config = stage_manager, runtime_config
        self.initial_result, self.initial_loader = initial_evaluation_result, initial_loader
        self.initialize, self.advance = initialize, advance
        self.data = data
        self.resources = ProcessResources(data)
        self.revision = 0
        self._delivered = set()
        self._counts = {}
        self.runtime = None
        self.decisions = self.current = None
        sources = []
        for function in (initialize, advance, type(loss_computer), type(stage_manager)):
            source = inspect.getsourcefile(function)
            if source is not None and Path(source).is_file():
                sources.append((str(Path(source).resolve()),
                                hashlib.sha256(Path(source).read_bytes()).hexdigest()))
        self.identity = key_for(("native-flow/process-lifetime/v1", self.resources.nonce,
            int(args.steps), int(getattr(args, "seed", 0)), tuple(sources)))
        self.operations = {
            phase: operation(phase, NativeInvocation, NativeReceipt, self._invoke,
                version="native-flow/v1", model_identity=self.identity,
                paid_boundary=boundary, reference_codecs=(self.resources.codec,),
                on_result=self._on_result)
            for phase, boundary in _BOUNDARIES.items()
        }
        self.parts = Components(self.operations)
        self.requirements = self.parts.requirements(tuple(_BOUNDARIES))
        self.translation = _translate(self.stages, data)
        self.system, self.initial = self.translation.system, self.translation.candidate

    def _invoke(self, request):
        if request.revision != self.revision:
            raise ContractConflict("Native scientific state changed before its operation")
        function, args, kwargs = self.resources.resolve(request.arguments)
        self.resources.expire_scores()
        result = function(*args, **kwargs)
        self.revision += 1
        reference = self.resources.retain(result, label="operation-result", revision=self.revision)
        return NativeReceipt(reference, self.revision)

    def _on_result(self, receipt):
        # Restoring RNG cannot restore a live stage/pose graph. This profile
        # deliberately rejects replay, including within the same process.
        if receipt.revision != self.revision or receipt.result in self._delivered:
            raise ContractConflict("Process-lifetime scientific operations cannot be replayed")
        self.resources.resolve(receipt.result, revision=receipt.revision)
        self._delivered.add(receipt.result)

    def call(self, phase, function, args, kwargs):
        if self.runtime is None or phase not in self.operations:
            raise RuntimeError("Native scientific operation has no active run owner")
        work = self.runtime._work
        slot = (work, phase)
        ordinal = self._counts.get(slot, 0)
        self._counts[slot] = ordinal + 1
        arguments = self.resources.retain((function, args, kwargs), label="operation-arguments",
                                           revision=self.revision)
        receipt = self.runtime.call(f"{phase}/{ordinal}", self.operations[phase],
                                    NativeInvocation(arguments, self.revision))
        return self.resources.resolve(receipt.result, revision=receipt.revision)

    @contextmanager
    def active(self, runtime):
        if runtime.root is not None or runtime.data is not self.data:
            raise ValueError("The native compatibility composition requires its memory-only runtime")
        previous = self.runtime
        self.runtime = runtime
        token = _ACTIVE.set(self)
        try:
            yield
        finally:
            _ACTIVE.reset(token)
            self.runtime = previous

    def scored(self, runtime):
        translated = _translate(self.stages, self.data)
        if translated.system != self.system:
            raise ContractConflict("A native operation changed the declared molecular topology")
        return _project(self.current, translated, self.data, self.resources, self.decisions,
                        self.stages, self.revision, runtime.history)

    def root(self, candidate, runtime):
        with self.active(runtime):
            if self.initial_result is None:
                if self.initial_loader is None:
                    raise ValueError("A native composition requires initial evidence or its loader")
                self.initial_result = native_call("initial", self.initial_loader)
            self.decisions, self.current = native_call("prepare", self.initialize,
                self.args, self.loss, self.initial_result, self.stages, self.config)
            return self.scored(runtime)

    def perform(self, work, runtime):
        if work.stage != "advance" or len(work.parents) != 1:
            raise ValueError("Native continuation work requires one accepted predecessor")
        current = work.parents[0]
        actual = _translate(self.stages, self.data)
        if actual.system != self.system or current.candidate != actual.candidate:
            raise ContractConflict("Native work predecessor differs from the committed scientific state")
        with self.active(runtime):
            self.current, accepted, stopped = self.advance(self.args, self.loss, self.current,
                self.stages, self.config, self.decisions, int(work.sample.index))
            successor = self.scored(runtime)
        return TrialOutcome(work.identity, "completed", (), (successor,), (),
                            Continuation(successor, "scientific_early_stop" if stopped else None))

    def close(self):
        self.resources.close()


def run_native_optimization(args, loss_computer, stage_manager, runtime_config, *,
                            initial_evaluation_result=None, initial_loader=None,
                            initialize, advance):
    """Default native tasks share the same runner as all typed compositions."""
    from .successor_search import AcceptedSuccessorPolicy
    data = DataStore()
    for codec in LEGACY_REFERENCE_CODECS:
        data.register(codec)
    with SCIENTIFIC_OWNER:
        flow = NativeFlow(args, loss_computer, stage_manager, runtime_config,
            initial_evaluation_result=initial_evaluation_result, initial_loader=initial_loader,
            initialize=initialize, advance=advance, data=data)
        try:
            with Runtime(flow.parts, data=data) as runtime:
                state = "failed"
                try:
                    coordinator.run(flow.system, flow.initial, flow,
                        AcceptedSuccessorPolicy(rounds=max(0, int(args.steps) - 1),
                                                seed=int(getattr(args, "seed", 0))), runtime)
                    state = "completed"
                finally:
                    try:
                        _write_execution_summary(args, runtime, state=state)
                    except Exception:
                        logging.exception("Could not write the native execution summary")
            return flow.current
        finally:
            flow.close()


def _write_execution_summary(args, runtime, *, state):
    out_dir = getattr(args, "out_dir", None)
    if out_dir is None:
        return
    path = Path(out_dir) / "execution.json"
    attempts = []
    for row in runtime._records("attempt").values():
        attempts.append({"component": row["component"], "boundary": row["boundary"],
                         "status": row["status"]})
    checkpoint = runtime.load("run/state")
    history = checkpoint["history"] if checkpoint is not None else runtime.history
    payload = {"runner": "astrolabe.design.coordinator", "recovery": "process_lifetime",
               "state": state, "advances": len(history.records), "attempts": attempts}
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)
