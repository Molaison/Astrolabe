from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import pytest

from astrolabe.app import Astrolabe
from astrolabe.design.composition import ControlRun
from astrolabe.design.control_journal import ControlCheckpoint, ControlRecovery
from astrolabe.design.runtime import ContractConflict, OutcomeUnknown, Runtime
from astrolabe.design.store import DataclassCodec
from astrolabe.design.types import Blob, Key, key_for


@dataclass(frozen=True)
class CounterState:
    value: int


@dataclass(frozen=True)
class CounterRequest:
    target: int


@dataclass(frozen=True)
class CounterEvent:
    value: int
    python_draw: float
    numpy_draw: tuple[float, ...]


STATE_CODEC = DataclassCodec(CounterState, "test.control-journal.state/v1")
REQUEST_CODEC = DataclassCodec(CounterRequest, "test.control-journal.request/v1")
RESULT_CODEC = DataclassCodec(CounterEvent, "test.control-journal.result/v1")
CHECKPOINT_CODEC = DataclassCodec(
    ControlCheckpoint,
    "astrolabe.control-checkpoint/v2",
)


class CounterPolicy:
    request_type = CounterRequest
    result_type = CounterEvent

    def __init__(self, steps: int, *, identity: Key | None = None) -> None:
        self.steps = steps
        self.identity = identity or key_for(
            ("test.control-journal.counter-policy/v1", steps)
        )
        self.tell_events: list[CounterEvent] = []

    def start(self, initial: CounterState) -> CounterState:
        return initial

    def ask(self, state: CounterState) -> CounterRequest:
        if self.finished(state):
            raise ValueError("counter policy already finished")
        return CounterRequest(state.value + 1)

    def admit(
        self,
        state: CounterState,
        request: CounterRequest,
    ) -> CounterState:
        if request.target != state.value + 1:
            raise ValueError("counter request is out of order")
        return state

    def tell(
        self,
        state: CounterState,
        event: CounterEvent,
    ) -> CounterState:
        if event.value != state.value + 1:
            raise ValueError("counter event is out of order")
        self.tell_events.append(event)
        return CounterState(event.value)

    def finished(self, state: CounterState) -> bool:
        return state.value >= self.steps

    def selected(self, state: CounterState) -> CounterState:
        return state


class CounterTrial:
    requirements = ()

    def __init__(
        self,
        *,
        identity: Key | None = None,
        interrupt_after_issue: bool = False,
        recover=None,
    ) -> None:
        self.identity = identity or key_for(
            ("test.control-journal.counter-trial/v1",)
        )
        self.interrupt_after_issue = interrupt_after_issue
        self.recover = recover
        self.perform_count = 0
        self.events: list[CounterEvent] = []

    def perform(self, request: CounterRequest) -> CounterEvent:
        self.perform_count += 1
        if self.interrupt_after_issue:
            raise KeyboardInterrupt("simulated process loss after issue")
        event = CounterEvent(
            value=request.target,
            python_draw=random.random(),
            numpy_draw=tuple(float(value) for value in np.random.random(2)),
        )
        self.events.append(event)
        return event


def _run_control(
    root: Path,
    policy: CounterPolicy,
    trial: CounterTrial,
    initial: CounterState = CounterState(0),
) -> CounterState:
    with Runtime(root=root) as runtime:
        run = ControlRun(
            policy=policy,
            trial=trial,
            initial_state=initial,
            runtime=runtime,
            state_codec=STATE_CODEC,
            request_codec=REQUEST_CODEC,
            result_codec=RESULT_CODEC,
        )
        result = Astrolabe(run).run()
    assert isinstance(result, CounterState)
    return result


def _rng_snapshot():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    return (
        python_state,
        (
            numpy_state[0],
            numpy_state[1].tolist(),
            numpy_state[2],
            numpy_state[3],
            float(numpy_state[4]),
        ),
    )


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def test_policy_state_identity_is_explicit_and_config_sensitive() -> None:
    first = CounterPolicy(1)
    changed = CounterPolicy(2)

    assert isinstance(first.identity, Key)
    assert first.identity != changed.identity


def test_result_receipt_before_state_commit_delivers_event_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = key_for(("same-policy-and-trial/v1",))
    policy = CounterPolicy(1, identity=identity)
    first_trial = CounterTrial()

    with Runtime(root=tmp_path) as runtime:
        original_save = runtime.save

        def crash_after_result(name, value, codec=None):
            reference = (
                original_save(name, value)
                if codec is None
                else original_save(name, value, codec)
            )
            if name.startswith("control/result/"):
                raise RuntimeError("crash after result receipt")
            return reference

        monkeypatch.setattr(runtime, "save", crash_after_result)
        run = ControlRun(
            policy=policy,
            trial=first_trial,
            initial_state=CounterState(0),
            runtime=runtime,
            state_codec=STATE_CODEC,
            request_codec=REQUEST_CODEC,
            result_codec=RESULT_CODEC,
        )
        with pytest.raises(RuntimeError, match="after result receipt"):
            Astrolabe(run).run()

    assert first_trial.perform_count == 1
    assert len(first_trial.events) == 1
    assert policy.tell_events == []

    resumed_trial = CounterTrial()
    final = _run_control(tmp_path, policy, resumed_trial)

    assert final == CounterState(1)
    assert resumed_trial.perform_count == 0
    assert policy.tell_events == first_trial.events
    assert len(policy.tell_events) == 1


def test_state_commit_crash_resumes_finished_without_reapplying(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = CounterPolicy(1)
    first_trial = CounterTrial()

    with Runtime(root=tmp_path) as runtime:
        original_save = runtime.save

        def crash_after_state_commit(name, value, codec=None):
            reference = (
                original_save(name, value)
                if codec is None
                else original_save(name, value, codec)
            )
            if (
                name == "control/state"
                and isinstance(value, ControlCheckpoint)
                and value.request is None
                and runtime.data.get(value.state, STATE_CODEC)
                == CounterState(1)
            ):
                raise RuntimeError("crash after state commit")
            return reference

        monkeypatch.setattr(runtime, "save", crash_after_state_commit)
        run = ControlRun(
            policy=policy,
            trial=first_trial,
            initial_state=CounterState(0),
            runtime=runtime,
            state_codec=STATE_CODEC,
            request_codec=REQUEST_CODEC,
            result_codec=RESULT_CODEC,
        )
        with pytest.raises(RuntimeError, match="after state commit"):
            Astrolabe(run).run()

    assert first_trial.perform_count == 1
    assert len(policy.tell_events) == 1

    with Runtime(root=tmp_path) as runtime:
        runtime.data.register(CHECKPOINT_CODEC)
        checkpoint = runtime.load("control/state", CHECKPOINT_CODEC)
    assert checkpoint is not None
    assert checkpoint.request is None
    assert isinstance(checkpoint.state, Blob)
    assert isinstance(checkpoint.rng, Blob)

    resumed_trial = CounterTrial()
    final = _run_control(tmp_path, policy, resumed_trial)

    assert final == CounterState(1)
    assert resumed_trial.perform_count == 0
    assert len(policy.tell_events) == 1


def test_issued_without_recovery_raises_outcome_unknown_and_never_executes(
    tmp_path: Path,
) -> None:
    policy = CounterPolicy(1)
    interrupted = CounterTrial(interrupt_after_issue=True)

    with Runtime(root=tmp_path) as runtime:
        run = ControlRun(
            policy=policy,
            trial=interrupted,
            initial_state=CounterState(0),
            runtime=runtime,
            state_codec=STATE_CODEC,
            request_codec=REQUEST_CODEC,
            result_codec=RESULT_CODEC,
        )
        with pytest.raises(KeyboardInterrupt):
            Astrolabe(run).run()
    assert interrupted.perform_count == 1

    resumed = CounterTrial()
    with Runtime(root=tmp_path) as runtime:
        run = ControlRun(
            policy=policy,
            trial=resumed,
            initial_state=CounterState(0),
            runtime=runtime,
            state_codec=STATE_CODEC,
            request_codec=REQUEST_CODEC,
            result_codec=RESULT_CODEC,
        )
        with pytest.raises(OutcomeUnknown):
            Astrolabe(run).run()

    assert resumed.perform_count == 0
    assert policy.tell_events == []


def test_plain_event_is_not_accepted_as_owner_recovery(
    tmp_path: Path,
) -> None:
    policy = CounterPolicy(1)
    interrupted = CounterTrial(interrupt_after_issue=True)

    with Runtime(root=tmp_path) as runtime:
        run = ControlRun(
            policy=policy,
            trial=interrupted,
            initial_state=CounterState(0),
            runtime=runtime,
            state_codec=STATE_CODEC,
            request_codec=REQUEST_CODEC,
            result_codec=RESULT_CODEC,
        )
        with pytest.raises(KeyboardInterrupt):
            Astrolabe(run).run()

    def plain_event_recovery(request):
        return CounterEvent(request.target, 0.25, (0.5,))

    recovered = CounterTrial(recover=plain_event_recovery)
    with Runtime(root=tmp_path) as runtime:
        run = ControlRun(
            policy=policy,
            trial=recovered,
            initial_state=CounterState(0),
            runtime=runtime,
            state_codec=STATE_CODEC,
            request_codec=REQUEST_CODEC,
            result_codec=RESULT_CODEC,
        )
        with pytest.raises(TypeError, match="original typed result/RNG receipt"):
            Astrolabe(run).run()

    assert recovered.perform_count == 0
    assert policy.tell_events == []


@pytest.mark.parametrize("wrong_field", ["execution", "request"])
def test_recovery_must_match_exact_admission_and_request(
    tmp_path: Path,
    wrong_field: str,
) -> None:
    policy = CounterPolicy(1)
    interrupted = CounterTrial(interrupt_after_issue=True)

    with Runtime(root=tmp_path) as runtime:
        run = ControlRun(
            policy=policy,
            trial=interrupted,
            initial_state=CounterState(0),
            runtime=runtime,
            state_codec=STATE_CODEC,
            request_codec=REQUEST_CODEC,
            result_codec=RESULT_CODEC,
        )
        with pytest.raises(KeyboardInterrupt):
            Astrolabe(run).run()

    runtime_holder: dict[str, Runtime] = {}
    wrong_blob = Blob(
        key_for(f"wrong-{wrong_field}"),
        REQUEST_CODEC.identity,
    )
    dummy_result = Blob(key_for("unused-result"), RESULT_CODEC.identity)
    dummy_receipt = Blob(key_for("unused-rng"), "test/unused-rng/v1")

    def mismatched_recovery(request):
        runtime = runtime_holder["runtime"]
        request_ref = runtime.data.put(request, REQUEST_CODEC)
        contract = runtime.load("control/contract")
        execution = Key(key_for((contract, request_ref)).digest)
        if wrong_field == "execution":
            execution = Key(key_for("wrong-execution").digest)
        else:
            request_ref = wrong_blob
        return ControlRecovery(
            execution=execution,
            request=request_ref,
            result=dummy_result,
            rng_receipt=dummy_receipt,
        )

    recovered = CounterTrial(recover=mismatched_recovery)
    with Runtime(root=tmp_path) as runtime:
        runtime_holder["runtime"] = runtime
        run = ControlRun(
            policy=policy,
            trial=recovered,
            initial_state=CounterState(0),
            runtime=runtime,
            state_codec=STATE_CODEC,
            request_codec=REQUEST_CODEC,
            result_codec=RESULT_CODEC,
        )
        with pytest.raises(
            ContractConflict,
            match="belongs to another control request",
        ):
            Astrolabe(run).run()

    assert recovered.perform_count == 0
    assert policy.tell_events == []


def test_changed_trial_identity_conflicts_before_execution(
    tmp_path: Path,
) -> None:
    policy = CounterPolicy(1)
    initial_trial = CounterTrial()
    assert _run_control(tmp_path, policy, initial_trial) == CounterState(1)

    changed = CounterTrial(identity=key_for(("different-trial/v1",)))
    with Runtime(root=tmp_path) as runtime:
        run = ControlRun(
            policy=policy,
            trial=changed,
            initial_state=CounterState(0),
            runtime=runtime,
            state_codec=STATE_CODEC,
            request_codec=REQUEST_CODEC,
            result_codec=RESULT_CODEC,
        )
        with pytest.raises(ContractConflict, match="changed on resume"):
            Astrolabe(run).run()

    assert changed.perform_count == 0


def test_interrupted_python_and_numpy_rng_matches_uninterrupted_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = CounterPolicy(2)
    _seed_all(911)
    first_trial = CounterTrial()

    with Runtime(root=tmp_path) as runtime:
        original_save = runtime.save

        def crash_after_first_result(name, value, codec=None):
            reference = (
                original_save(name, value)
                if codec is None
                else original_save(name, value, codec)
            )
            if name.startswith("control/result/"):
                raise RuntimeError("first result receipt committed")
            return reference

        monkeypatch.setattr(runtime, "save", crash_after_first_result)
        run = ControlRun(
            policy=policy,
            trial=first_trial,
            initial_state=CounterState(0),
            runtime=runtime,
            state_codec=STATE_CODEC,
            request_codec=REQUEST_CODEC,
            result_codec=RESULT_CODEC,
        )
        with pytest.raises(RuntimeError, match="receipt committed"):
            Astrolabe(run).run()

    resumed_trial = CounterTrial()
    resumed_state = _run_control(tmp_path, policy, resumed_trial)
    resumed_events = first_trial.events + resumed_trial.events
    resumed_rng = _rng_snapshot()

    _seed_all(911)
    baseline_policy = CounterPolicy(2)
    baseline_trial = CounterTrial()
    baseline_root = tmp_path / "baseline"
    baseline_state = _run_control(
        baseline_root,
        baseline_policy,
        baseline_trial,
    )
    baseline_rng = _rng_snapshot()

    assert resumed_state == baseline_state == CounterState(2)
    assert resumed_events == baseline_trial.events
    assert resumed_trial.perform_count == 1
    assert len(resumed_events) == 2
    assert resumed_rng == baseline_rng
    assert policy.tell_events == baseline_policy.tell_events
