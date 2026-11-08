"""One search coordinator; scientific composition lives in Trial and policy.

Policy transitions are deterministic and checkpointed before work is dispatched
and after outcomes are delivered. Completed work is recovered independently of
the policy checkpoint, closing the result-before-tell crash window.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import inspect
import json
from pathlib import Path

from astrolabe.model_execution import model_run_scope

from .runtime import BudgetExhausted, ContractConflict, OutcomeUnknown
from .store import design_codec
from .types import History, RunManifest, RunResult, TrialOutcome, key_for, make_candidate


def drive(policy, trial, state, *, transition=None):
    """The sole control loop for typed scientific and collection compositions.

    Policies own decisions; trials execute one bounded request. A request may
    issue isolated jobs and collect an explicit window, but never contains a
    second search loop. Domain owners declare their concrete request/result
    types, rather than passing an untyped universal command dictionary.
    """
    def change(function, *args):
        if transition is not None:
            return transition(function, *args)
        from astrolabe.model_execution import capture_random_state, restore_random_state
        before = capture_random_state()
        try:
            result = function(*args)
        except BaseException:
            restore_random_state(before)
            raise
        if key_for(capture_random_state()) != key_for(before):
            restore_random_state(before)
            raise ValueError("Control policy transitions must be pure; stochastic work belongs to the trial")
        return result

    while not change(policy.finished, state):
        request = change(policy.ask, state)
        if not isinstance(request, policy.request_type):
            raise TypeError("Control policy returned an undeclared request type")
        admit = getattr(policy, "admit", None)
        if admit is not None:
            state = change(admit, state, request)
        checkpoint = getattr(trial, "checkpoint", None)
        if checkpoint is not None:
            checkpoint(state, request=request)
        result = trial.perform(request)
        if not isinstance(result, policy.result_type):
            raise TypeError("Control trial returned an undeclared result type")
        state = change(policy.tell, state, result)
        if checkpoint is not None:
            checkpoint(state, request=None)
    return change(policy.selected, state)


@dataclass(frozen=True)
class _MolecularBatch:
    checkpoint: dict


@dataclass(frozen=True)
class _MolecularOutcomes:
    outcomes: tuple[TrialOutcome, ...]


def _transition(runtime, function, *arguments):
    before = runtime._capture_rng()
    try:
        value = function(*arguments)
    except BaseException:
        runtime._restore_rng(before)
        raise
    if runtime._capture_rng() != before:
        runtime._restore_rng(before)
        raise ValueError("SearchPolicy transitions must be pure; put stochastic decisions in a named operation")
    return value


def _implementation(owner):
    identity = getattr(owner, "identity", None)
    if identity is not None:
        return identity
    try:
        source = inspect.getsource(type(owner))
    except (OSError, TypeError):
        raise ValueError("A custom Trial/SearchPolicy must declare a stable identity") from None
    # Instance options cannot be inferred from mutable closures or __dict__.
    if vars(owner):
        raise ValueError("A configured Trial/SearchPolicy must declare a stable identity")
    return key_for((type(owner).__module__, source))


def _validate_scored(scored, system, baseline, runtime):
    if make_candidate(system, scored.candidate.assignments) != scored.candidate:
        raise ValueError("Candidate identity does not match its complete assignments")
    system.edit(baseline, scored.candidate.assignments, data=runtime.data)
    if scored.evidence.evaluated_candidate != scored.candidate.identity:
        raise ValueError("Evidence belongs to another candidate")
    if scored.assessment.evidence != scored.evidence.identity:
        raise ValueError("Assessment belongs to another evidence record")
    if any(a.candidate != scored.candidate.identity for a in scored.evidence.artifacts):
        raise ValueError("Current evidence contains another candidate's artifact")


def _verified_attempts(runtime, work_id):
    attempts = runtime.attempts(work_id)
    for attempt in attempts:
        if attempt.status in ("unknown", "issued"):
            raise OutcomeUnknown(attempt.identity)
    return tuple(attempt.identity for attempt in attempts)


def run(system, initial, trial, policy, runtime, *, capacity=1):
    """Run a composition and release only the model owners created by it."""
    with model_run_scope():
        return _run(system, initial, trial, policy, runtime, capacity=capacity)


def _run(system, initial, trial, policy, runtime, *, capacity):
    if type(capacity) is not int or capacity < 1:
        raise ValueError("capacity must be a positive integer")
    if initial.system != system.identity:
        raise ValueError("Initial candidate belongs to another System")
    if make_candidate(system, initial.assignments) != initial:
        raise ValueError("Initial candidate identity does not match its assignments")
    system.edit(initial, initial.assignments, data=runtime.data)
    # No operation/model construction occurs before this cold preflight.
    preflight = runtime.preflight(trial.requirements)
    runtime.data.register(policy.codec)
    contract = key_for((system, initial, _implementation(trial), _implementation(policy),
                        policy.codec.identity, preflight, capacity))
    prior = runtime.load("run/contract")
    if prior is not None and prior != contract:
        raise ContractConflict("System, initial candidate, Trial or SearchPolicy changed on resume")
    runtime.save("run/contract", contract)
    bootstrap = getattr(trial, "bootstrap", None)
    if bootstrap is not None:
        empty = History(key_for(()), ())
        with runtime.bind("bootstrap", empty, entry=(system, initial, contract)):
            bootstrap(runtime)
            _verified_attempts(runtime, "bootstrap")
    checkpoint = runtime.load("run/state")
    if checkpoint is None:
        empty = History(key_for(()), ())
        root = runtime._load_completion("run/root")
        if root is None:
            with runtime.bind("root", empty, entry=(system, initial, contract)):
                root = trial.root(initial, runtime)
                if root.candidate.identity != initial.identity:
                    raise ValueError("Trial.root changed its input candidate")
                _validate_scored(root, system, initial, runtime)
                _verified_attempts(runtime, "root")
                runtime._save_completion("run/root", root)
        state = _transition(runtime, policy.start, (root,))
        checkpoint = dict(state=runtime.data.put(state, policy.codec), history=empty,
                          pending=(), finished=False, rng=runtime._capture_rng())
        runtime.save("run/state", checkpoint)
    runtime._restore_rng(checkpoint["rng"])
    class MolecularSteps:
        request_type = _MolecularBatch
        result_type = _MolecularOutcomes

        def finished(self, current):
            if current["finished"]:
                return True
            state = runtime.data.get(current["state"], policy.codec)
            return _transition(runtime, policy.finished, state)

        def ask(self, current):
            state = runtime.data.get(current["state"], policy.codec)
            history = current["history"]
            pending = current["pending"]
            if not pending:
                pending = _transition(runtime, lambda: tuple(policy.ask(state, history, capacity)))
                if not pending:
                    raise RuntimeError("SearchPolicy neither finished nor produced work")
                if len({work.identity for work in pending}) != len(pending):
                    raise ValueError("SearchPolicy produced duplicate work IDs")
                delivered = {record.work for record in history.records}
                if any(work.identity in delivered for work in pending):
                    raise ContractConflict("SearchPolicy reissued already delivered work")
                if runtime.data.put(state, policy.codec) != current["state"]:
                    raise ValueError("SearchPolicy.ask mutated its checkpoint")
            return _MolecularBatch(dict(current, pending=pending))

        def admit(self, current, batch):
            return batch.checkpoint

        def perform(self, batch):
            current = batch.checkpoint
            pending, history = current["pending"], current["history"]
            outcomes = []
            for work in pending:
                outcome = runtime._load_completion(f"outcome/{work.identity}")
                if outcome is None:
                    with runtime.bind(work.identity, history, entry=work):
                        try:
                            outcome = trial.perform(work, runtime)
                        except (OutcomeUnknown, ContractConflict, KeyboardInterrupt, SystemExit):
                            raise
                        except BudgetExhausted:
                            outcome = TrialOutcome(work.identity, "refused", (), (), ())
                        except Exception:
                            if getattr(trial, "fail_fast", True):
                                raise
                            outcome = TrialOutcome(work.identity, "failed", (), (), ())
                        if outcome.work != work.identity:
                            raise ValueError("Trial returned another work's outcome")
                        if outcome.status == "unknown":
                            raise OutcomeUnknown(work.identity)
                        for child in outcome.children:
                            _validate_scored(child, system, initial, runtime)
                        if outcome.continuation is not None:
                            _validate_scored(outcome.continuation.successor, system, initial, runtime)
                        outcome = replace(outcome, attempts=_verified_attempts(runtime, work.identity))
                        runtime._save_completion(f"outcome/{work.identity}", outcome)
                outcomes.append(outcome)
            return _MolecularOutcomes(tuple(outcomes))

        def tell(self, current, delivery):
            state = runtime.data.get(current["state"], policy.codec)
            state = _transition(runtime, policy.tell, state, delivery.outcomes)
            records = current["history"].records + delivery.outcomes
            history = History(key_for(records), records)
            return dict(state=runtime.data.put(state, policy.codec), history=history,
                        pending=(), finished=_transition(runtime, policy.finished, state),
                        rng=runtime._capture_rng())

        def checkpoint(self, current, *, request):
            runtime.save("run/state", current)

        def selected(self, current):
            return current

    steps = MolecularSteps()
    checkpoint = drive(steps, steps, checkpoint)
    if not checkpoint["finished"]:
        checkpoint = dict(checkpoint, finished=True)
        runtime.save("run/state", checkpoint)
    runtime.history = checkpoint["history"]
    selected = _transition(runtime, lambda: tuple(policy.selected(
        runtime.data.get(checkpoint["state"], policy.codec))))
    summary = runtime.root / "selected.json" if runtime.root else Path("<memory>")
    if runtime.root:
        # Compact projection. Natural metrics/arrays remain in typed evidence.
        rows = [{"candidate": item.candidate.identity.digest,
                 "objective": list(item.assessment.values),
                 "gates": dict(item.assessment.gates),
                 "evidence": item.evidence.identity.digest} for item in selected]
        temporary = summary.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        temporary.replace(summary)
    manifest = RunManifest(contract, preflight.identity, policy.codec.identity,
        design_codec.identity, runtime.root / "journal.sqlite" if runtime.root else Path("<memory>"),
        runtime.data.put(checkpoint, design_codec), summary)
    runtime.save("run/manifest", manifest)
    return RunResult(selected, manifest)
