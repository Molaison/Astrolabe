"""Checkpoint and delivery authority for typed control compositions.

The state and pending request are one atomic checkpoint. Completed bounded
requests have an independent result/RNG receipt, closing the result-before-tell
window. An issued request without a receipt requires its owner's explicit
reconciliation; the coordinator never assumes external jobs have stopped.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import coordinator
from .runtime import ContractConflict, OutcomeUnknown
from .store import DataclassCodec, design_codec
from .types import Blob, History, Key, RNGReceipt, key_for


@dataclass(frozen=True)
class ControlCheckpoint:
    state: Blob
    request: Blob | None = None
    rng: Blob | None = None


_CHECKPOINT = DataclassCodec(ControlCheckpoint, "astrolabe.control-checkpoint/v2")


@dataclass(frozen=True)
class ControlRecovery:
    """An owner-verified typed result and its original scientific RNG receipt."""
    execution: Key
    request: Blob
    result: Blob
    rng_receipt: Blob


def run_recorded_control(composition):
    runtime, policy, trial = composition.runtime, composition.policy, composition.trial
    state_codec, request_codec, result_codec = (
        composition.state_codec, composition.request_codec, composition.result_codec)
    if any(codec is None for codec in (state_codec, request_codec, result_codec)):
        raise TypeError("Recorded control requires explicit state, request and result codecs")
    for codec in (state_codec, request_codec, result_codec, _CHECKPOINT):
        runtime.data.register(codec)
    preflight = runtime.preflight(getattr(trial, "requirements", ()))
    contract = key_for(("typed-control/v1", coordinator._implementation(policy),
        coordinator._implementation(trial), state_codec.identity, request_codec.identity,
        result_codec.identity, preflight.identity))
    prior = runtime.load("control/contract")
    if prior is not None and prior != contract:
        raise ContractConflict("Control policy, scientific trial or codecs changed on resume")
    runtime.save("control/contract", contract)
    checkpoint = runtime.load("control/state", _CHECKPOINT)
    state = (composition.initial_state if checkpoint is None else
             runtime.data.get(checkpoint.state, state_codec))
    if checkpoint is not None:
        if checkpoint.rng is None:
            raise ContractConflict("Control checkpoint has no scientific RNG state")
        runtime._restore_rng(checkpoint.rng)

    class RecordedTrial:
        def checkpoint(self, current, *, request):
            state_ref = runtime.data.put(current, state_codec)
            request_ref = None if request is None else runtime.data.put(request, request_codec)
            runtime.save("control/state", ControlCheckpoint(state_ref, request_ref,
                                                           runtime._capture_rng()), _CHECKPOINT)
            report = getattr(trial, "checkpoint", None)
            if report is not None:
                before = runtime._capture_rng()
                try:
                    report(current, request=request)
                    if runtime._capture_rng() != before:
                        raise ValueError("Control checkpoint reporting consumed scientific RNG")
                finally:
                    runtime._restore_rng(before)

        def perform(self, request):
            request_ref = runtime.data.put(request, request_codec)
            identity = key_for((contract, request_ref)).digest
            result_name = "control/result/" + identity
            issued_name = "control/issued/" + identity
            completed = runtime._load_completion(result_name, result_codec)
            if completed is not None:
                return completed
            issued = runtime.load(issued_name)
            if issued is not None:
                reconcile = getattr(trial, "recover", None)
                if reconcile is None:
                    raise OutcomeUnknown(identity)
                # The owner must reconstruct or verify outstanding native jobs.
                # This is not permission to repeat an unverified external call.
                recovered = reconcile(request)
                if recovered is not None:
                    if not isinstance(recovered, ControlRecovery):
                        raise TypeError("Control reconciliation requires the original typed result/RNG receipt")
                    if recovered.execution != Key(identity) or recovered.request != request_ref:
                        raise ContractConflict("Recovered result/RNG receipt belongs to another control request")
                    receipt = runtime.data.get(recovered.rng_receipt, design_codec)
                    if not isinstance(receipt, RNGReceipt) or receipt.protocol != runtime.scientific_contract:
                        raise ContractConflict("Recovered RNG receipt belongs to another scientific execution")
                    if receipt.before != issued.get("before"):
                        raise ContractConflict("Recovered RNG receipt belongs to another control request")
                    result = runtime.data.get(recovered.result, result_codec)
                    if not isinstance(result, policy.result_type):
                        raise TypeError("Control reconciliation returned an undeclared result")
                    runtime._restore_rng(receipt.after)
                    runtime._save_completion(result_name, result, result_codec)
                    return result
            empty = History(key_for(()), ())
            with runtime.bind("control/" + identity, empty, entry=request_ref):
                runtime.save(issued_name, {"request": request_ref, "contract": contract,
                                          "before": runtime._capture_rng()})
                result = trial.perform(request)
                if not isinstance(result, policy.result_type):
                    raise TypeError("Control trial returned an undeclared result")
                runtime._save_completion(result_name, result, result_codec)
                return result

    recorded = RecordedTrial()
    transition = lambda function, *args: coordinator._transition(runtime, function, *args)
    # Re-deliver the exact admitted request after an interrupted transition.
    # State+pending-request commit atomically, so this is delivered once.
    if checkpoint is not None and checkpoint.request is not None:
        request = runtime.data.get(checkpoint.request, request_codec)
        result = recorded.perform(request)
        state = transition(policy.tell, state, result)
        recorded.checkpoint(state, request=None)
    elif checkpoint is None:
        recorded.checkpoint(state, request=None)
    return coordinator.drive(policy, recorded, state, transition=transition)
