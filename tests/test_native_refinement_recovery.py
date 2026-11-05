from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import random
import shutil
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import Protomer, SegmentManager
from astrolabe.design.branch_types import (
    BranchAction,
    BranchEvent,
    BranchJob,
    BranchProposal,
    BranchTerminal,
    BranchingState,
    branch_codec,
)
from astrolabe.design.branching import BranchingPolicy
from astrolabe.design.control_journal import _CHECKPOINT
from astrolabe.design.legacy import LegacyContext
from astrolabe.design.native_refinement import (
    NativeMeasurement,
    NativeProposal,
    NativeRefinementSpec,
    run_native_refinement,
)
from astrolabe.design.native_refinement_runtime import (
    DurableNativeRefinementOwner,
    RawPredictionReceipt,
)
from astrolabe.design.native_scores import restore_scores
from astrolabe.design.runtime import OutcomeUnknown, Runtime
from astrolabe.design.store import DataStore, design_codec
from astrolabe.design.types import Blob, History, Key, key_for
from astrolabe.evaluators.utils import StandardScores
from astrolabe.model_execution import capture_random_state


PROTOCOL = Key("a" * 64)
CONTRACT = Key("b" * 64)
ROOT_MEASUREMENT = "native-refinement/evaluate/0/measurement"
PROPOSAL = "native-refinement/propose/1/proposal"
CHILD_PREDICTION = "native-refinement/evaluate/1/prediction"
CHILD_MEASUREMENT = "native-refinement/evaluate/1/measurement"


@pytest.fixture(autouse=True)
def clean_global_config():
    global_config.reset()
    yield
    global_config.reset()


@dataclass
class Recorder:
    calls: list[tuple]
    loads: list[tuple]


class FakeMutator:
    def __init__(self) -> None:
        self.shutdown_count = 0

    def shutdown(self) -> None:
        self.shutdown_count += 1


class FakeEvaluator:
    def __init__(self) -> None:
        self.shutdown_count = 0

    def shutdown(self) -> None:
        self.shutdown_count += 1


@dataclass
class Case:
    root: Path
    data: DataStore
    runtime: Runtime
    spec: NativeRefinementSpec
    recorder: Recorder
    mutator: FakeMutator
    evaluator: FakeEvaluator

    def close(self) -> None:
        self.runtime.close()

    def crash_after(self, named_round: str) -> None:
        original = self.runtime.call
        hit = False

        def call(round_name, operation, request, **kwargs):
            nonlocal hit
            result = original(round_name, operation, request, **kwargs)
            if round_name == named_round and not hit:
                hit = True
                raise RuntimeError(f"injected crash after {named_round}")
            return result

        self.runtime.call = call


def _draw(recorder: Recorder, stage: str, **fields) -> tuple[float, float]:
    value = (random.random(), float(np.random.random()))
    recorder.calls.append((stage, *value, fields))
    return value


def _label(candidate) -> str:
    from astrolabe.design.types import Polymer

    tokens = []
    for assignment in candidate.assignments:
        if isinstance(assignment.value, Polymer):
            tokens.extend(assignment.value.tokens)
    return "".join(tokens)


def _seed(seed: int, step: int) -> int:
    return (seed * 100003 + step) % 2147483647 or 1


def _case(
    root: Path,
    recorder: Recorder | None = None,
    *,
    max_evaluations: int = 2,
) -> Case:
    recorder = recorder or Recorder(calls=[], loads=[])
    global_config.reset()
    global_config.init_const("segments", {})
    global_config.init_const("protomers", {})
    global_config.init_const("mutable_protomers", {})
    global_config.init_const("named_files", {})
    global_config.init_const("segment_manager", SegmentManager())

    protomer = Protomer("A", "AC@shared", "p")
    global_config.protomers["A"] = protomer
    global_config.mutable_protomers["A"] = protomer

    output = root / "output"
    framework = root / "framework"
    data = DataStore(framework / "data")
    context = LegacyContext()
    translation = context.capture_candidate(assemblies=("A",), data=data)
    mutator = FakeMutator()
    evaluator = FakeEvaluator()

    def load_mutator():
        value = _draw(recorder, "load_mutator")
        recorder.loads.append(("mutator", *value))
        return mutator

    def load_evaluator():
        value = _draw(recorder, "load_evaluator")
        recorder.loads.append(("evaluator", *value))
        return evaluator

    def propose(
        model,
        candidate,
        scores,
        *,
        seed,
        step,
        generation,
        parent_id,
        proposal_index,
        output,
    ):
        assert model is mutator
        _draw(
            recorder,
            "propose",
            seed=seed,
            step=step,
            generation=generation,
            parent_id=parent_id,
            proposal_index=proposal_index,
            output=str(output),
            score=float(np.asarray(scores.plddt["A"][0]).mean()),
        )
        segment = global_config.segments[protomer.components[0]]
        segment.sequence = "GC" if step % 2 else "GU"
        captured = context.capture_candidate(
            assemblies=("A",),
            state=translation.state,
            data=data,
        )
        return NativeProposal(captured.candidate, _label(captured.candidate))

    def evaluate(
        model,
        candidate,
        *,
        seed,
        step,
        assemblies,
        output,
        proposal_id,
    ):
        assert model is evaluator
        value = _draw(
            recorder,
            "evaluate",
            seed=seed,
            step=step,
            proposal_id=proposal_id,
            label=_label(candidate),
            assemblies=tuple(assemblies),
        )
        raw = Path(output) / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        structure = raw / f"prediction_{proposal_id}.cif"
        structure.write_text(f"data_{proposal_id}\n# {_label(candidate)}\n")
        base = np.float32(0.25 + value[1] / 100.0)
        plddt = np.asarray([base, base + np.float32(0.25)], dtype=np.float32)
        return StandardScores(
            names=["A"],
            plddt={"A": [plddt]},
            complex_paths={"A": structure},
            per_proto_scores={"A": [plddt.astype(np.float64)]},
            pae={"A": [np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)]},
        )

    def measure(candidate, scores, job):
        value = np.asarray(scores.plddt["A"][0], dtype=np.float32)
        expected = _label(candidate)
        active_protomer = global_config.protomers["A"]
        primary = global_config.segments[active_protomer.components[0]]
        shared = global_config.segments["shared"]
        assert global_config.mutable_protomers["A"] is active_protomer
        assert shared is primary
        assert active_protomer.mutated_sequence == expected
        assert shared.mutated_sequence == expected
        _draw(
            recorder,
            "measure",
            evaluation_id=job.id,
            label=expected,
            active_sequence=active_protomer.mutated_sequence,
            shared_sequence=shared.mutated_sequence,
        )
        return NativeMeasurement(
            score=float(value.mean()),
            metrics={
                "evaluation_id": job.id,
                "label": expected,
                "active_sequence": active_protomer.mutated_sequence,
                "shared_sequence": shared.mutated_sequence,
                "plddt": value.tolist(),
            },
            structure_path=str(scores.complex_paths["A"]),
        )

    spec = NativeRefinementSpec(
        output=output,
        seed=71,
        width=1,
        proposals_per_parent=1,
        max_evaluations=max_evaluations,
        evaluator_alias="fake-fold",
        mutator_alias="fake-mutate",
        assemblies=("A",),
        initial_candidate=translation.candidate,
        context=context,
        translation=translation,
        data=data,
        load_mutator=load_mutator,
        load_evaluator=load_evaluator,
        propose=propose,
        evaluate=evaluate,
        measure=measure,
        label=_label,
        proposal_seed=_seed,
    )
    runtime = Runtime(
        root=framework,
        data=data,
        scientific_contract=CONTRACT,
    )
    return Case(root, data, runtime, spec, recorder, mutator, evaluator)


def _policy(*, max_evaluations: int = 2) -> BranchingPolicy:
    return BranchingPolicy(
        width=1,
        proposals_per_parent=1,
        max_evaluations=max_evaluations,
        proposal_cap=2,
        seed=71,
        evaluation_seed=71,
        quota_mode="attempts",
        expansion="round_robin",
        proposal_batch_size=1,
        evaluation_width=1,
        pipeline=False,
        immediate_evaluation=True,
        cap_error=True,
    )


def _run(case: Case, policy: BranchingPolicy | None = None):
    random.seed(991)
    np.random.seed(991)
    result = run_native_refinement(
        case.spec,
        policy=policy or _policy(),
        protocol=PROTOCOL,
        runtime=case.runtime,
    )
    return result, capture_random_state()


def _drive_manual(
    case: Case,
    *,
    wait_none_first: bool,
) -> tuple[tuple[str, ...], dict, tuple[tuple, ...], tuple[tuple[str, str, str], ...]]:
    random.seed(991)
    np.random.seed(991)
    owner = DurableNativeRefinementOwner(
        case.spec,
        case.runtime,
        protocol=PROTOCOL,
    )
    owner.bootstrap()
    policy = _policy()
    state = policy.start(owner.root_job(protocol=PROTOCOL))
    empty = History(key_for(()), ())
    waits = []
    step = 0
    while not policy.finished(state):
        action = policy.ask(state)
        state = policy.admit(state, action)
        if wait_none_first and step == 0:
            action = replace(action, wait="none")
        waits.append(action.wait)
        with case.runtime.bind(
            f"manual-control/{step}",
            empty,
            entry=f"{policy.identity.digest}:{step}",
        ):
            event = owner.perform(action)
        state = policy.tell(state, event)
        step += 1
        if step > 20:
            raise AssertionError("manual branch drive did not finish")
    owner.shutdown()
    calls = []
    for item in case.recorder.calls:
        if item[0].startswith("load_"):
            continue
        fields = dict(item[-1])
        if "output" in fields:
            fields["output"] = "<output>"
        calls.append((*item[:-1], fields))
    calls = tuple(calls)
    attempts = tuple(
        (
            attempt.key.named_round,
            attempt.key.work_id,
            attempt.status,
        )
        for attempt in case.runtime.attempts()
        if attempt.key.named_round.startswith("native-refinement/")
    )
    return tuple(waits), capture_random_state(), calls, attempts


def _summary(result, root: Path):
    values = []
    for identifier, record in result.member_items():
        relative = Path(record.structure_path).relative_to(root)
        content = Path(record.structure_path).read_bytes()
        values.append(
            {
                "id": identifier,
                "candidate": record.candidate.identity,
                "scores": record.scores,
                "score": record.score,
                "structure": relative,
                "structure_sha256": hashlib.sha256(content).hexdigest(),
                "metrics": result.metrics(record),
                "label": record.label,
                "generation": record.generation,
                "parent_id": record.parent_id,
                "proposal_step": record.proposal_step,
            }
        )
    return {
        "records": values,
        "selected": tuple((item.id, item.key, item.score) for item in result.selected),
        "state_records": result.state.records,
        "evaluations": result.evaluations,
        "proposals": result.proposals,
    }


@pytest.mark.parametrize(
    "boundary",
    (ROOT_MEASUREMENT, PROPOSAL, CHILD_PREDICTION, CHILD_MEASUREMENT),
)
def test_real_control_recovers_each_native_refinement_boundary(
    tmp_path: Path,
    boundary: str,
):
    root = tmp_path / "run"

    baseline_case = _case(root)
    baseline, baseline_rng = _run(baseline_case)
    expected = _summary(baseline, root)
    baseline_control_calls = [
        item for item in baseline_case.recorder.calls
        if not item[0].startswith("load_")
    ]
    assert [name for name, *_ in baseline_case.recorder.loads] == [
        "mutator",
        "evaluator",
    ]
    baseline_case.data.root = Path(baseline_case.data.root)
    baseline.owner.shutdown()
    baseline_case.close()
    shutil.rmtree(root)

    recorder = Recorder(calls=[], loads=[])
    first_case = _case(root, recorder)
    first_case.crash_after(boundary)
    with pytest.raises(RuntimeError, match="injected crash"):
        _run(first_case)
    loads_before_resume = len(recorder.loads)
    first_case.close()

    resumed_case = _case(root, recorder)
    resumed, resumed_rng = _run(resumed_case)
    resumed_summary = _summary(resumed, root)

    assert resumed_summary == expected
    assert resumed_rng == baseline_rng
    assert [
        item for item in recorder.calls if not item[0].startswith("load_")
    ] == baseline_control_calls
    assert sum(item[0] == "propose" for item in recorder.calls) == 1
    assert sum(item[0] == "evaluate" for item in recorder.calls) == 2
    assert sum(item[0] == "measure" for item in recorder.calls) == 2
    resumed_loads = [
        name for name, *_ in recorder.loads[loads_before_resume:]
    ]
    assert resumed_loads in ([], ["mutator", "evaluator"])
    resumed.owner.member_blobs.clear()
    assert tuple(identifier for identifier, _ in resumed.member_items()) == (0, 1)
    resumed.owner.shutdown()
    resumed_case.close()


def test_wait_none_reuses_job_receipts_and_rng_under_next_control_action(
    tmp_path: Path,
):
    baseline_root = tmp_path / "baseline"
    baseline_case = _case(baseline_root)
    _, baseline_rng, baseline_calls, baseline_attempts = _drive_manual(
        baseline_case,
        wait_none_first=False,
    )
    baseline_case.close()

    split_root = tmp_path / "split"
    split_case = _case(split_root)
    waits, split_rng, split_calls, split_attempts = _drive_manual(
        split_case,
        wait_none_first=True,
    )
    split_case.close()

    assert waits[0] == "none"
    assert "all" in waits[1:]
    assert split_rng == baseline_rng
    assert split_calls == baseline_calls
    assert split_attempts == baseline_attempts
    assert len(split_attempts) == 5
    assert {status for _, _, status in split_attempts} == {"completed"}
    assert {
        work_id
        for named_round, work_id, _ in split_attempts
        if named_round.startswith("native-refinement/evaluate/0/")
    } == {"native-refinement/job/evaluate/0"}
    assert {
        work_id
        for named_round, work_id, _ in split_attempts
        if named_round.startswith("native-refinement/evaluate/1/")
    } == {"native-refinement/job/evaluate/1"}
    assert {
        work_id
        for named_round, work_id, _ in split_attempts
        if named_round.startswith("native-refinement/propose/1/")
    } == {"native-refinement/job/propose/1"}
    assert sum(call[0] == "propose" for call in split_calls) == 1
    assert sum(call[0] == "evaluate" for call in split_calls) == 2
    assert sum(call[0] == "measure" for call in split_calls) == 2


def test_same_action_reentry_restores_buffered_after_rng_without_replay(
    tmp_path: Path,
):
    root = tmp_path / "same-action"
    case = _case(root)
    random.seed(991)
    np.random.seed(991)
    owner = DurableNativeRefinementOwner(
        case.spec,
        case.runtime,
        protocol=PROTOCOL,
    )
    owner.bootstrap()
    policy = _policy()
    state = policy.start(owner.root_job(protocol=PROTOCOL))
    action = policy.ask(state)
    state = policy.admit(state, action)
    deferred = replace(action, wait="none")
    history = History(key_for(()), ())
    entry = f"{policy.identity.digest}:0"

    with case.runtime.bind(
        "same-action/control/0",
        history,
        entry=entry,
    ):
        first = owner.perform(deferred)
    assert first.terminals == ()
    first_rng = capture_random_state()
    first_calls = tuple(case.recorder.calls)
    first_attempts = tuple(
        (attempt.identity, attempt.status)
        for attempt in case.runtime.attempts()
    )
    buffered = owner._load_terminal_buffer()
    assert tuple(
        entry.terminal.job.identity for entry in buffered.entries
    ) == ("evaluate/0",)
    case.runtime.close()

    resumed_case = _case(root, case.recorder)
    resumed_owner = DurableNativeRefinementOwner(
        resumed_case.spec,
        resumed_case.runtime,
        protocol=PROTOCOL,
    )
    resumed_owner.bootstrap()
    with resumed_case.runtime.bind(
        "same-action/control/0",
        history,
        entry=entry,
    ):
        recovered = resumed_owner.perform(deferred)
    assert recovered.terminals == ()
    assert capture_random_state() == first_rng
    assert _normalized_control_calls(resumed_case.recorder) == (
        _normalized_control_calls(case.recorder)
    )
    assert tuple(
        (attempt.identity, attempt.status)
        for attempt in resumed_case.runtime.attempts()
    ) == first_attempts
    assert resumed_owner._load_terminal_buffer() == buffered

    state = policy.tell(state, recovered)
    assert sum(call[0] == "evaluate" for call in first_calls) == 1
    assert sum(call[0] == "measure" for call in first_calls) == 1
    resumed_owner.shutdown()
    resumed_case.close()


def _mixed_pipeline_prefix(case: Case):
    random.seed(991)
    np.random.seed(991)
    owner = DurableNativeRefinementOwner(
        case.spec,
        case.runtime,
        protocol=PROTOCOL,
    )
    owner.bootstrap()
    policy = BranchingPolicy(
        width=1,
        proposals_per_parent=4,
        max_evaluations=6,
        proposal_cap=8,
        seed=71,
        evaluation_seed=71,
        quota_mode="attempts",
        expansion="round_robin",
        proposal_batch_size=2,
        evaluation_width=1,
        pipeline=True,
        immediate_evaluation=False,
        cap_error=True,
    )
    state = policy.start(owner.root_job(protocol=PROTOCOL))
    empty = History(key_for(()), ())

    def perform(step: int, action: BranchAction) -> BranchEvent:
        with case.runtime.bind(
            f"mixed-control/{step}",
            empty,
            entry=f"{policy.identity.digest}:{step}",
        ):
            return owner.perform(action)

    action = policy.ask(state)
    state = policy.admit(state, action)
    state = policy.tell(state, perform(0, action))

    action = policy.ask(state)
    state = policy.admit(state, action)
    state = policy.tell(state, perform(1, action))

    proposals = policy.ask(state)
    state = policy.admit(state, proposals)
    state = policy.tell(state, perform(2, proposals))

    evaluation = policy.ask(state)
    assert tuple(job.identity for job in evaluation.jobs) == ("evaluate/1",)
    state = policy.admit(state, evaluation)
    deferred_evaluation = replace(evaluation, wait="none")
    state = policy.tell(state, perform(3, deferred_evaluation))

    later_proposals = policy.ask(state)
    assert tuple(job.identity for job in later_proposals.jobs) == (
        "propose/3",
        "propose/4",
    )
    state = policy.admit(state, later_proposals)
    deferred_proposals = replace(later_proposals, wait="none")
    state = policy.tell(state, perform(4, deferred_proposals))

    buffer = owner._load_terminal_buffer().entries
    assert tuple(
        (entry.terminal.job.identity, entry.position)
        for entry in buffer
    ) == (
        ("evaluate/1", 0),
        ("propose/3", 1),
        ("propose/4", 2),
    )
    assert {
        entry.origin_action for entry in buffer
    } == {
        owner._action_identity(deferred_evaluation),
        owner._action_identity(deferred_proposals),
    }
    assert owner._action_identity(deferred_proposals) != owner._action_identity(
        replace(deferred_proposals, wait="all")
    )

    evaluation_terminal = buffer[0].terminal
    owner._save_terminal_buffer(buffer[1:])
    state = policy.tell(
        state,
        BranchEvent((evaluation_terminal,)),
    )
    assert tuple(job.identity for job in state.pending) == ()
    assert tuple(job.identity for job in state.batch) == (
        "propose/3",
        "propose/4",
    )
    return owner, policy, state, empty


def _normalized_control_calls(recorder: Recorder) -> tuple[tuple, ...]:
    calls = []
    for item in recorder.calls:
        if item[0].startswith("load_"):
            continue
        fields = dict(item[-1])
        if "output" in fields:
            fields["output"] = "<output>"
        calls.append((*item[:-1], fields))
    return tuple(calls)


def _branching_state_semantics(state) -> tuple:
    def member(item):
        return (
            item.id,
            item.key,
            item.score,
            item.eligible,
            item.comparison_context,
        )

    def terminal(item):
        value = item.value
        value_identity = None
        if value is not None:
            value_identity = (
                value.id,
                value.key,
                getattr(value, "score", None),
            )
        return item.job.identity, item.status, value_identity

    return (
        state.generation,
        state.proposals,
        state.next_evaluation,
        tuple(member(item) for item in state.records),
        tuple(member(item) for item in state.frontier),
        tuple(member(item) for item in state.parents),
        state.counts,
        tuple(job.identity for job in state.batch),
        tuple(terminal(item) for item in state.batch_results),
        tuple(job.identity for job in state.queued),
        tuple(job.identity for job in state.pending),
        state.seen,
        state.duplicates,
        state.rejected,
        tuple(job.identity for job in state.refused),
        state.stop_reason,
        state.finished,
    )


def test_mixed_outstanding_collection_does_not_rewind_rng(
    tmp_path: Path,
):
    mixed_case = _case(tmp_path / "mixed", max_evaluations=6)
    mixed_owner, mixed_policy, mixed_state, mixed_empty = _mixed_pipeline_prefix(
        mixed_case
    )
    action = mixed_policy.ask(mixed_state)
    assert tuple(job.identity for job in action.jobs) == ("evaluate/2",)
    assert tuple(job.identity for job in action.state.batch) == (
        "propose/3",
        "propose/4",
    )
    mixed_state = mixed_policy.admit(mixed_state, action)
    with mixed_case.runtime.bind(
        "mixed-control/5",
        mixed_empty,
        entry=f"{mixed_policy.identity.digest}:5",
    ):
        mixed_event = mixed_owner.perform(action)
    delivered_rng = capture_random_state()
    delivered_calls = tuple(mixed_case.recorder.calls)
    delivered_attempts = tuple(
        (attempt.identity, attempt.status)
        for attempt in mixed_case.runtime.attempts()
    )
    with mixed_case.runtime.bind(
        "mixed-control/5",
        mixed_empty,
        entry=f"{mixed_policy.identity.digest}:5",
    ):
        replayed_event = mixed_owner.perform(action)
    assert replayed_event == mixed_event
    assert capture_random_state() == delivered_rng
    assert tuple(mixed_case.recorder.calls) == delivered_calls
    assert tuple(
        (attempt.identity, attempt.status)
        for attempt in mixed_case.runtime.attempts()
    ) == delivered_attempts
    mixed_state = mixed_policy.tell(mixed_state, mixed_event)
    mixed_rng = capture_random_state()
    mixed_calls = _normalized_control_calls(mixed_case.recorder)
    mixed_attempts = tuple(
        (
            attempt.key.named_round,
            attempt.key.work_id,
            attempt.status,
        )
        for attempt in mixed_case.runtime.attempts()
        if attempt.key.named_round.startswith("native-refinement/")
    )
    mixed_owner.shutdown()
    mixed_case.close()

    sequential_case = _case(tmp_path / "sequential", max_evaluations=6)
    sequential_owner, sequential_policy, sequential_state, sequential_empty = (
        _mixed_pipeline_prefix(sequential_case)
    )
    with sequential_case.runtime.bind(
        "sequential-control/collect",
        sequential_empty,
        entry=f"{sequential_policy.identity.digest}:collect",
    ):
        collected = sequential_owner.perform(
            BranchAction(state=sequential_state, wait="all")
        )
    assert tuple(
        terminal.job.identity for terminal in collected.terminals
    ) == ("propose/3", "propose/4")
    sequential_state = sequential_policy.tell(sequential_state, collected)

    action = sequential_policy.ask(sequential_state)
    assert tuple(job.identity for job in action.jobs) == ("evaluate/2",)
    sequential_state = sequential_policy.admit(sequential_state, action)
    with sequential_case.runtime.bind(
        "sequential-control/5",
        sequential_empty,
        entry=f"{sequential_policy.identity.digest}:5",
    ):
        sequential_event = sequential_owner.perform(action)
    sequential_state = sequential_policy.tell(sequential_state, sequential_event)
    sequential_rng = capture_random_state()
    sequential_calls = _normalized_control_calls(sequential_case.recorder)
    sequential_attempts = tuple(
        (
            attempt.key.named_round,
            attempt.key.work_id,
            attempt.status,
        )
        for attempt in sequential_case.runtime.attempts()
        if attempt.key.named_round.startswith("native-refinement/")
    )
    sequential_owner.shutdown()
    sequential_case.close()

    assert _branching_state_semantics(mixed_state) == _branching_state_semantics(
        sequential_state
    )
    assert mixed_rng == sequential_rng
    assert mixed_calls == sequential_calls
    assert mixed_attempts == sequential_attempts
    assert sum(call[0] == "propose" for call in mixed_calls) == 4
    assert sum(call[0] == "evaluate" for call in mixed_calls) == 3
    assert sum(call[0] == "measure" for call in mixed_calls) == 3
    assert {
        named_round
        for named_round, _, status in mixed_attempts
        if status == "completed"
    } == {
        ROOT_MEASUREMENT,
        "native-refinement/evaluate/0/prediction",
        "native-refinement/propose/1/proposal",
        "native-refinement/propose/2/proposal",
        "native-refinement/propose/3/proposal",
        "native-refinement/propose/4/proposal",
        CHILD_PREDICTION,
        CHILD_MEASUREMENT,
        "native-refinement/evaluate/2/prediction",
        "native-refinement/evaluate/2/measurement",
    }


def test_unknown_child_call_blocks_reentry_without_invoking_gpu_callback(
    tmp_path: Path,
):
    root = tmp_path / "run"
    recorder = Recorder(calls=[], loads=[])
    first_case = _case(root, recorder)
    first_case.crash_after(CHILD_PREDICTION)
    with pytest.raises(RuntimeError, match="injected crash"):
        _run(first_case)

    attempts = tuple(first_case.runtime.attempts())
    prediction_attempt = next(
        item for item in attempts if item.key.named_round == CHILD_PREDICTION
    )
    row = first_case.runtime._read("attempt", prediction_attempt.identity)
    row["status"] = "unknown"
    first_case.runtime._write("attempt", prediction_attempt.identity, row)
    first_case.close()

    calls_before = tuple(recorder.calls)
    loads_before = tuple(recorder.loads)
    resumed_case = _case(root, recorder)
    with pytest.raises(OutcomeUnknown):
        _run(resumed_case)

    assert tuple(recorder.calls) == calls_before
    assert tuple(recorder.loads) == loads_before
    resumed_case.close()


def test_raw_prediction_receipt_survives_before_measurement(
    tmp_path: Path,
):
    root = tmp_path / "run"
    recorder = Recorder(calls=[], loads=[])
    case = _case(root, recorder)
    case.crash_after(CHILD_PREDICTION)
    with pytest.raises(RuntimeError, match="injected crash"):
        _run(case)

    attempt = next(
        item
        for item in case.runtime.attempts()
        if item.key.named_round == CHILD_PREDICTION
    )
    assert attempt.status == "completed"
    assert attempt.result is not None
    receipt = case.data.get(attempt.result, case.spec.data._codecs[
        "astrolabe.design.native-refinement.raw-prediction/v1"
    ])
    assert isinstance(receipt, RawPredictionReceipt)
    restored = restore_scores(
        receipt.scores,
        case.data,
        case.root / "restored-raw-scores",
    )
    plddt = restored.plddt["A"][0]
    assert plddt.shape == (2,)
    assert plddt.dtype == np.float32
    assert restored.complex_paths["A"].read_text() == "data_1\n# GC\n"
    assert sum(item[0] == "measure" for item in recorder.calls) == 1
    case.close()


def _extended_score_spec(case: Case):
    original_measure = case.spec.measure

    def propose(
        model,
        candidate,
        scores,
        *,
        seed,
        step,
        generation,
        parent_id,
        proposal_index,
        output,
    ):
        assert model is case.mutator
        expected = _label(candidate)
        assert scores.candidate_scores == {
            "label": expected,
            "evaluation_id": parent_id,
        }
        assert scores.posthoc_metric == [parent_id]
        _draw(
            case.recorder,
            "propose",
            seed=seed,
            step=step,
            generation=generation,
            parent_id=parent_id,
            proposal_index=proposal_index,
            output=str(output),
            candidate_scores=scores.candidate_scores,
            posthoc_metric=scores.posthoc_metric,
        )
        segment = global_config.segments["shared"]
        segment.sequence = "GC" if step == 1 else "GD"
        captured = case.spec.context.capture_candidate(
            assemblies=("A",),
            state=case.spec.translation.state,
            data=case.data,
        )
        return NativeProposal(captured.candidate, _label(captured.candidate))

    def measure(candidate, scores, job):
        measurement = original_measure(candidate, scores, job)
        label = _label(candidate)
        measurement = replace(
            measurement,
            score=2.0 if label == "GC" else 1.0,
        )
        scores.candidate_scores = {
            "label": label,
            "evaluation_id": job.id,
        }
        scores.posthoc_metric = [job.id]
        return measurement

    return replace(
        case.spec,
        max_evaluations=3,
        propose=propose,
        measure=measure,
    )


def test_measure_extensions_survive_raw_cut_and_are_visible_to_next_proposal(
    tmp_path: Path,
):
    root = tmp_path / "run"
    recorder = Recorder(calls=[], loads=[])
    first = _case(root, recorder)
    first_spec = _extended_score_spec(first)
    first = replace(first, spec=first_spec)
    first.crash_after(CHILD_MEASUREMENT)
    with pytest.raises(RuntimeError, match=f"injected crash after {CHILD_MEASUREMENT}"):
        _run(first, _policy(max_evaluations=3))
    first.close()

    resumed = _case(root, recorder)
    resumed_spec = _extended_score_spec(resumed)
    resumed = replace(resumed, spec=resumed_spec)
    result, _ = _run(resumed, _policy(max_evaluations=3))

    proposal_calls = [
        item for item in recorder.calls if item[0] == "propose"
    ]
    assert len(proposal_calls) == 2
    assert proposal_calls[0][-1]["candidate_scores"]["evaluation_id"] == 0
    assert proposal_calls[1][-1]["candidate_scores"]["evaluation_id"] == 1

    prediction_attempts = tuple(
        item
        for item in resumed.runtime.attempts()
        if item.key.named_round
        in (
            "native-refinement/evaluate/0/prediction",
            "native-refinement/evaluate/1/prediction",
        )
    )
    assert len(prediction_attempts) == 2
    raw_receipts = {}
    for attempt in prediction_attempts:
        raw_receipt = resumed.data.get(
            attempt.result,
            resumed.data._codecs[
                "astrolabe.design.native-refinement.raw-prediction/v1"
            ],
        )
        raw_scores = restore_scores(
            raw_receipt.scores,
            resumed.data,
            root / "raw-check" / attempt.identity[:12],
        )
        raw_receipts[attempt.key.named_round] = raw_receipt
        assert not hasattr(raw_scores, "candidate_scores")
        assert not hasattr(raw_scores, "posthoc_metric")

    members = {
        identifier: record
        for identifier, record in result.member_items()
    }
    member_scores = restore_scores(
        members[1].scores,
        resumed.data,
        root / "member-score-check",
    )
    assert member_scores.candidate_scores == {
        "label": "GC",
        "evaluation_id": 1,
    }
    assert member_scores.posthoc_metric == [1]
    assert members[1].scores != raw_receipts[CHILD_PREDICTION].scores
    result.owner.shutdown()
    resumed.close()


def test_gpu_v5_persistent_checkpoint_rehydrates_proposal_report_ids() -> None:
    repository = Path(__file__).resolve().parents[1]
    framework = (
        repository
        / "outputs/rna-archived-gpu-acceptance/gpu_v5/recovery"
        / "configured-output/001_seed_67101/.framework"
    )
    if not framework.is_dir():
        pytest.skip("real gpu_v5 recovery checkpoint is unavailable")

    journal_path = framework / "journal.sqlite"
    journal_sha256 = hashlib.sha256(journal_path.read_bytes()).hexdigest()
    data = DataStore(framework / "data")
    data.register(design_codec)
    data.register(_CHECKPOINT)
    state_codec = branch_codec(BranchingState)
    data.register(state_codec)

    with sqlite3.connect(f"file:{journal_path}?mode=ro", uri=True) as connection:
        row = connection.execute(
            "select value from records "
            "where kind='checkpoint' and name='control/state'"
        ).fetchone()
    assert row is not None
    record = json.loads(row[0])
    checkpoint = data.get(
        Blob(Key(record["identity"]), record["codec"]),
        _CHECKPOINT,
    )
    state = data.get(checkpoint.state, state_codec)
    assert state.finished is True
    assert tuple(
        (terminal.job.identity, terminal.job.phase, terminal.status)
        for terminal in state.batch_results
    ) == (("propose/1", "propose", "completed"),)

    events: list[tuple[str, dict]] = []
    owner = object.__new__(DurableNativeRefinementOwner)
    owner.data = data
    owner.spec = SimpleNamespace(
        event=lambda name, **fields: events.append((name, fields))
    )
    owner.controller_state = None
    owner._state = None
    owner._request = None
    owner._observed_proposals = set()

    state_before = state
    state_identity = key_for(state)
    budget_before = (
        state.generation,
        state.proposals,
        state.next_evaluation,
        state.duplicates,
        state.rejected,
        state.refused,
        state.stop_reason,
        state.finished,
    )
    rng_before = key_for(capture_random_state())

    owner.checkpoint(state, request=None, report=False)
    assert owner._observed_proposals == {1}
    owner.checkpoint(state, request=None)
    assert events == []
    assert state == state_before
    assert key_for(state) == state_identity
    assert (
        state.generation,
        state.proposals,
        state.next_evaluation,
        state.duplicates,
        state.rejected,
        state.refused,
        state.stop_reason,
        state.finished,
    ) == budget_before
    assert key_for(capture_random_state()) == rng_before

    proposal = state.batch_results[0].value
    assert isinstance(proposal, BranchProposal)

    def terminal(
        identifier: int,
        *,
        valid: bool = True,
        status: str = "completed",
    ) -> BranchTerminal:
        job = BranchJob(
            phase="propose",
            id=identifier,
            generation=state.generation,
            seed=proposal.id,
        )
        return BranchTerminal(
            job,
            status,
            replace(proposal, id=identifier, valid=valid),
        )

    unique = terminal(101)
    owner.checkpoint(
        replace(
            state,
            batch_results=(unique,),
            queued=(
                BranchJob(
                    phase="evaluate",
                    id=101,
                    generation=state.generation,
                    seed=71,
                    proposal=unique.value,
                ),
            ),
            pending=(),
        ),
        request=None,
    )
    assert events == []

    duplicate = terminal(102)
    duplicate_state = replace(
        state,
        batch_results=(duplicate,),
        queued=(),
        pending=(),
    )
    owner.checkpoint(duplicate_state, request=None)
    owner.checkpoint(duplicate_state, request=None)

    rejected = terminal(103, valid=False)
    rejected_state = replace(
        state,
        batch_results=(rejected,),
        queued=(),
        pending=(),
    )
    owner.checkpoint(rejected_state, request=None)
    owner.checkpoint(rejected_state, request=None)

    refused = terminal(104, status="refused")
    refused_state = replace(
        state,
        batch_results=(refused,),
        queued=(),
        pending=(),
    )
    owner.checkpoint(refused_state, request=None)
    owner.checkpoint(refused_state, request=None)

    assert [name for name, _ in events] == [
        "duplicate_proposal",
        "rejected_proposal",
        "refused_proposal",
    ]
    assert state == state_before
    assert key_for(state) == state_identity
    assert key_for(capture_random_state()) == rng_before
    assert hashlib.sha256(journal_path.read_bytes()).hexdigest() == journal_sha256


def test_resume_does_not_reproject_reported_proposal_terminal(
    tmp_path: Path,
) -> None:
    baseline_events: list[tuple[str, dict]] = []
    baseline_root = tmp_path / "baseline"
    baseline = _case(baseline_root)
    baseline = replace(
        baseline,
        spec=replace(
            baseline.spec,
            on_event=lambda name, fields: baseline_events.append((name, fields)),
        ),
    )
    baseline_result, _ = _run(baseline)
    baseline_result.owner.shutdown()
    baseline.close()

    recovery_root = tmp_path / "recovery"
    first_events: list[tuple[str, dict]] = []
    first = _case(recovery_root)
    first = replace(
        first,
        spec=replace(
            first.spec,
            on_event=lambda name, fields: first_events.append((name, fields)),
        ),
    )
    first.crash_after(CHILD_PREDICTION)
    with pytest.raises(RuntimeError, match="injected crash"):
        _run(first)
    first.close()

    resumed_events: list[tuple[str, dict]] = []
    resumed = _case(recovery_root)
    resumed = replace(
        resumed,
        spec=replace(
            resumed.spec,
            on_event=lambda name, fields: resumed_events.append((name, fields)),
        ),
    )
    result, _ = _run(resumed)

    def normalized(events, root: Path):
        value = json.loads(
            json.dumps(events, sort_keys=True).replace(str(root), "$RUN")
        )
        return value

    recovered_events = first_events + resumed_events
    assert normalized(recovered_events, recovery_root) == normalized(
        baseline_events,
        baseline_root,
    )
    assert sum(name == "proposal_completed" for name, _ in recovered_events) == 1
    assert all(name != "duplicate_proposal" for name, _ in recovered_events)
    result.owner.shutdown()
    resumed.close()

