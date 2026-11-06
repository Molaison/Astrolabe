"""Operation recovery tests: these do not stand in for model numerical tests."""
from dataclasses import dataclass
import random

import pytest

from astrolabe.design.operations import INVOKE, operation
from astrolabe.design.runtime import (
    BudgetExhausted, ContractConflict, OutcomeUnknown, RecordedFailure, Runtime,
)
from astrolabe.design.types import Capability, History, Requirement, Reconciliation, key_for
from astrolabe.design.store import JSONCodec
from astrolabe.model_daemon import (
    DaemonDisconnected,
    DaemonRequestFailed,
)


@dataclass(frozen=True)
class Number:
    value: float


@dataclass(frozen=True)
class NativeValues:
    values: list[int]


def history():
    return History(key_for(()), ())


def requirements(alias="draw"):
    return (Requirement(alias, INVOKE, "required"),)


def test_durable_named_call_reuses_output_and_rng_receipt(tmp_path):
    calls = []
    def draw(request):
        calls.append(request)
        return Number(random.random())
    component = operation("draw", Number, Number, draw, version="1", paid_boundary="predict")
    random.seed(17)
    with Runtime((component,), root=tmp_path, budgets={"predict": 1}) as runtime:
        runtime.preflight(requirements())
        with runtime.bind("work/0", history(), entry=(0,)):
            first = runtime.call("round/0", component, Number(1))
            continuation = random.random()
    random.seed(923)
    with Runtime((component,), root=tmp_path, budgets={"predict": 1}) as runtime:
        runtime.preflight(requirements())
        with runtime.bind("work/0", history(), entry=(0,)):
            assert runtime.call("round/0", component, Number(1)) == first
            assert random.random() == continuation
        assert len(runtime.attempts()) == 1
        assert runtime.attempts()[0].status == "completed"
    assert len(calls) == 1


def test_changed_request_at_same_slot_conflicts_before_execution():
    calls = []
    component = operation("draw", Number, Number, lambda x: calls.append(x) or x, version="1")
    with Runtime((component,)) as runtime:
        runtime.preflight(requirements())
        with runtime.bind("work", history()):
            runtime.call("round", component, Number(1))
            with pytest.raises(ContractConflict):
                runtime.call("round", component, Number(2))
    assert calls == [Number(1)]


def test_issued_unknown_never_replays_or_refunds_budget(tmp_path):
    calls = []
    def interrupted(request):
        calls.append(request)
        raise KeyboardInterrupt()
    component = operation("draw", Number, Number, interrupted, version="1", paid_boundary="predict")
    with Runtime((component,), root=tmp_path, budgets={"predict": 1}) as runtime:
        runtime.preflight(requirements())
        with runtime.bind("work", history()):
            with pytest.raises(KeyboardInterrupt):
                runtime.call("round", component, Number(1))
    with Runtime((component,), root=tmp_path, budgets={"predict": 1}) as runtime:
        runtime.preflight(requirements())
        with runtime.bind("work", history()):
            with pytest.raises(OutcomeUnknown):
                runtime.call("round", component, Number(1))
        with runtime.bind("other", history()):
            with pytest.raises(BudgetExhausted):
                runtime.call("round", component, Number(1))
        assert runtime.attempts()[0].status == "unknown"
    assert len(calls) == 1


def test_terminal_failure_is_a_recorded_outcome_not_a_retry():
    calls = []
    def fail(request):
        calls.append(request)
        raise ValueError("verified local failure")
    component = operation("draw", Number, Number, fail, version="1")
    with Runtime((component,)) as runtime:
        runtime.preflight(requirements())
        with runtime.bind("work", history()):
            with pytest.raises(ValueError, match="verified local"):
                runtime.call("round", component, Number(1))
        with runtime.bind("work", history()):
            with pytest.raises(RecordedFailure, match="verified local"):
                runtime.call("round", component, Number(1))
    assert len(calls) == 1


def test_preflight_and_work_contract_refuse_changes():
    component = operation("draw", Number, Number, lambda x: x, version="1")
    with Runtime((component,)) as runtime:
        with pytest.raises(ValueError, match="Missing required"):
            runtime.preflight((Requirement("draw", Capability("test", "gradient", "1"), "required"),))
        runtime.preflight(requirements())
        with runtime.bind("work", history(), entry=(0,)):
            pass
        with pytest.raises(ContractConflict):
            with runtime.bind("work", history(), entry=(1,)):
                pass


def test_single_owner_prevents_two_writers(tmp_path):
    with Runtime(root=tmp_path):
        with pytest.raises(BlockingIOError):
            Runtime(root=tmp_path)


def test_paid_transport_error_is_unknown_without_owner_terminal_proof():
    def disconnected(request):
        raise ConnectionError("reply lost")
    component = operation("draw", Number, Number, disconnected, version="1", paid_boundary="predict")
    with Runtime((component,)) as runtime:
        runtime.preflight(requirements())
        with runtime.bind("work", history()):
            with pytest.raises(OutcomeUnknown):
                runtime.call("round", component, Number(1))
        assert runtime.attempts()[0].status == "unknown"
        with runtime.bind("work", history()):
            with pytest.raises(OutcomeUnknown):
                runtime.call("round", component, Number(1))


def test_paid_daemon_uncertainty_overrides_broad_terminal_verifier(
    tmp_path,
    monkeypatch,
):
    calls = []
    verifier_calls = []

    def disconnected(request):
        calls.append(request)
        raise RuntimeError("outer transport failure") from DaemonDisconnected(
            "reply EOF after request send"
        )

    component = operation(
        "draw",
        Number,
        Number,
        disconnected,
        version="1",
        paid_boundary="predict",
    )
    monkeypatch.setattr(
        component,
        "failure_is_terminal",
        lambda error, *, paid_boundary: (
            verifier_calls.append((error, paid_boundary)) or True
        ),
    )
    with Runtime((component,), root=tmp_path, budgets={"predict": 1}) as runtime:
        runtime.preflight(requirements())
        with runtime.bind("work", history()):
            with pytest.raises(OutcomeUnknown):
                runtime.call("round", component, Number(1))
        assert runtime.attempts()[0].status == "unknown"
        with runtime.bind("work", history()):
            with pytest.raises(OutcomeUnknown):
                runtime.call("round", component, Number(1))
        with runtime.bind("other", history()):
            with pytest.raises(BudgetExhausted):
                runtime.call("round", component, Number(2))
    assert calls == [Number(1)]
    assert verifier_calls == []


def test_explicit_daemon_request_failure_can_be_terminal_for_paid_call():
    calls = []

    def rejected(request):
        calls.append(request)
        raise DaemonRequestFailed(
            "daemon adapter rejected request",
            remote_type="ValueError",
        )

    component = operation(
        "draw",
        Number,
        Number,
        rejected,
        version="1",
        paid_boundary="predict",
        terminal_errors=(DaemonRequestFailed,),
    )
    with Runtime((component,), budgets={"predict": 1}) as runtime:
        runtime.preflight(requirements())
        with runtime.bind("work", history()):
            with pytest.raises(DaemonRequestFailed, match="adapter rejected"):
                runtime.call("round", component, Number(1))
        assert runtime.attempts()[0].status == "failed"
        with runtime.bind("work", history()):
            with pytest.raises(RecordedFailure, match="adapter rejected"):
                runtime.call("round", component, Number(1))
    assert calls == [Number(1)]


def test_exception_group_and_context_chain_are_fully_classified():
    from astrolabe.design.runtime import exception_chain, outcome_is_unknown

    child = DaemonDisconnected("child disconnected")
    eof = EOFError("reply stream ended after request")

    class Outer(RuntimeError):
        pass

    try:
        raise child
    except DaemonDisconnected:
        try:
            raise Outer("context wrapper")
        except Outer as error:
            outer_error = error
    group = ExceptionGroup("request group", [outer_error, eof])
    assert child in tuple(exception_chain(outer_error))
    assert outcome_is_unknown(outer_error)
    assert outcome_is_unknown(group)
    assert isinstance(
        next(item for item in exception_chain(group) if item is child),
        DaemonDisconnected,
    )


def test_terminal_commit_failure_blocks_replay(tmp_path, monkeypatch):
    calls = []
    component = operation("draw", Number, Number, lambda x: calls.append(x) or x,
                          version="1", paid_boundary="predict")
    with Runtime((component,), root=tmp_path) as runtime:
        runtime.preflight(requirements())
        original = runtime._write
        def fail_completion(kind, name, value):
            if kind == "attempt" and value["status"] == "completed":
                raise OSError("terminal commit failed")
            return original(kind, name, value)
        monkeypatch.setattr(runtime, "_write", fail_completion)
        with runtime.bind("work", history()):
            with pytest.raises(OutcomeUnknown):
                runtime.call("round", component, Number(1))
        assert runtime.attempts()[0].status == "unknown"
    with Runtime((component,), root=tmp_path) as runtime:
        runtime.preflight(requirements())
        with runtime.bind("work", history()):
            with pytest.raises(OutcomeUnknown):
                runtime.call("round", component, Number(1))
    assert calls == [Number(1)]


def test_only_owner_verified_matching_terminal_evidence_resolves_unknown():
    calls = []
    correct_binding = [False]
    def lost_reply(request):
        calls.append(request)
        raise ConnectionError("reply lost")
    def reconcile(attempt, record, data):
        result = data.put(Number(42), component.descriptor.result_codec)
        proof = data.put({
            "binding": record["binding"] if correct_binding[0] else "another-request",
            "result": {"identity": result.identity.digest, "codec": result.codec},
            "rng": record["rng"],
        }, JSONCodec())
        return Reconciliation(attempt, "completed", proof)
    component = operation("draw", Number, Number, lost_reply, version="1",
                          paid_boundary="predict", reconciler=reconcile)
    with Runtime((component,), budgets={"predict": 1}) as runtime:
        runtime.preflight(requirements())
        with runtime.bind("work", history()):
            with pytest.raises(OutcomeUnknown):
                runtime.call("round", component, Number(1))
        attempt, = runtime.attempts()
        with pytest.raises(ContractConflict):
            runtime.reconcile(attempt.identity)
        assert runtime.attempts()[0].status == "unknown"
        correct_binding[0] = True
        assert runtime.reconcile(attempt.identity).status == "completed"
        with runtime.bind("work", history()):
            assert runtime.call("round", component, Number(1)) == Number(42)
    assert calls == [Number(1)]


def test_first_result_has_no_mutable_alias_to_adapter_or_replayed_content():
    owned = NativeValues([1])
    component = operation("draw", Number, NativeValues, lambda x: owned, version="1")
    with Runtime((component,)) as runtime:
        runtime.preflight(requirements())
        with runtime.bind("work", history()):
            first = runtime.call("round", component, Number(1))
            first.values.append(2)
            assert owned.values == [1]
            assert runtime.call("round", component, Number(1)).values == [1]
