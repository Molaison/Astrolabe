from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import Protomer, SegmentManager
from astrolabe.design.branch_types import (
    BranchAction,
    BranchEvent,
    BranchJob,
    BranchMember,
)
from astrolabe.design.branching import BranchingPolicy
from astrolabe.design.legacy import LegacyContext, capture_legacy_candidate
from astrolabe.design.native_refinement import (
    NativeMeasurement,
    NativeProposal,
    NativeRefinementOwner,
    NativeRefinementSpec,
    run_native_refinement,
)
from astrolabe.design.native_scores import restore_scores
from astrolabe.design.store import DataStore
from astrolabe.design.types import Key
from astrolabe.evaluators.utils import StandardScores


@pytest.fixture(autouse=True)
def clean_legacy_globals():
    global_config.reset()
    global_config.init_const("segments", {})
    global_config.init_const("protomers", {})
    global_config.init_const("mutable_protomers", {})
    global_config.init_const("named_files", {})
    global_config.init_const("segment_manager", SegmentManager())
    yield
    global_config.reset()


class FakeMutator:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.calls = []
        self.shutdown_calls = 0

    def shutdown(self) -> None:
        self.shutdown_calls += 1


class FakeEvaluator:
    def __init__(self, output: Path, events: list[str]) -> None:
        self.output = output
        self.events = events
        self.calls = []
        self.shutdown_calls = 0

    def shutdown(self) -> None:
        self.shutdown_calls += 1


def _candidate_label(candidate) -> str:
    from astrolabe.design.types import Polymer

    tokens = []
    for assignment in candidate.assignments:
        value = assignment.value
        if isinstance(value, Polymer):
            tokens.extend(value.tokens)
    return "".join(tokens)


def _fixture(tmp_path):
    protomer = Protomer("A", "AC@shared", "p")
    global_config.protomers["A"] = protomer
    global_config.mutable_protomers["A"] = protomer
    data = DataStore(tmp_path / "data")
    context = LegacyContext()
    translation = context.capture_candidate(assemblies=("A",), data=data)
    events = []
    instances = {"mutator": [], "evaluator": []}

    def load_mutator():
        events.append("mutator")
        value = FakeMutator(events)
        instances["mutator"].append(value)
        return value

    def load_evaluator():
        events.append("evaluator")
        value = FakeEvaluator(tmp_path / "prediction.cif", events)
        instances["evaluator"].append(value)
        return value

    prediction_seeds = []
    proposal_seeds = []

    def propose(
        mutator,
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
        mutator.calls.append(
            {
                "seed": seed,
                "step": step,
                "sequence": _candidate_label(candidate),
                "score": float(scores.plddt["A"][0].mean()),
            }
        )
        proposal_seeds.append(seed)
        segment = global_config.segments[protomer.components[0]]
        segment.sequence = "GC"
        captured = capture_legacy_candidate(assemblies=("A",), data=data)
        return NativeProposal(captured.candidate, _candidate_label(captured.candidate))

    def evaluate(
        evaluator,
        candidate,
        *,
        seed,
        step,
        assemblies,
        output,
        proposal_id,
    ):
        prediction_seeds.append(seed)
        sequence = _candidate_label(candidate)
        evaluator.calls.append((sequence, seed, step, tuple(assemblies), proposal_id))
        evaluator.output.write_bytes(f"model:{sequence}\n".encode("ascii"))
        values = np.asarray(
            [0.25, 0.75] if sequence == "AC" else [0.5, 0.9],
            dtype=np.float32,
        )
        return StandardScores(
            names=["A"],
            plddt={"A": [values]},
            complex_paths={"A": evaluator.output},
            per_proto_scores={"A": [values.astype(np.float64)]},
        )

    def measure(candidate, scores, job):
        values = np.asarray(scores.plddt["A"][0], dtype=float)
        return NativeMeasurement(
            score=float(values.mean()),
            metrics={
                "tokens": _candidate_label(candidate),
                "evaluation_id": job.id,
                "numpy_scalar": np.float64(values.mean()),
            },
            structure_path=str(scores.complex_paths["A"]),
        )

    spec = NativeRefinementSpec(
        output=tmp_path,
        seed=71,
        width=1,
        proposals_per_parent=1,
        max_evaluations=3,
        evaluator_alias="fake_fold",
        mutator_alias="fake_mutator",
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
        label=_candidate_label,
        proposal_seed=lambda seed, step: (seed * 100003 + step) % 2147483647 or 1,
    )
    return spec, events, instances, prediction_seeds, proposal_seeds


def test_protein_owner_constructs_mutator_then_evaluator_once_and_round_trips_scores(
    tmp_path,
):
    spec, events, instances, prediction_seeds, proposal_seeds = _fixture(tmp_path)
    owner = NativeRefinementOwner(spec)
    assert events == ["mutator", "evaluator"]

    root_job = owner.root_job(protocol=Key("1" * 64))
    root_terminal = owner.execute(root_job)
    assert root_terminal.status == "completed"
    root = root_terminal.value

    proposal_job = BranchJob(
        phase="propose",
        id=1,
        generation=1,
        seed=71 * 100003 + 1,
        parent=root,
        slot=1,
    )
    proposal_terminal = owner.execute(proposal_job)
    assert proposal_terminal.status == "completed"
    proposal = proposal_terminal.value
    assert proposal.key == "GC"

    evaluation_job = BranchJob(
        phase="evaluate",
        id=1,
        generation=1,
        seed=71,
        parent=root,
        proposal=proposal,
        slot=1,
    )
    child_terminal = owner.execute(evaluation_job)
    child = child_terminal.value
    assert child.key == "GC"

    # IDs are job identities; the predictor always receives the fixed run seed.
    assert prediction_seeds == [71, 71]
    assert proposal_seeds == [71 * 100003 + 1]
    assert instances["evaluator"][0].calls == [
        ("AC", 71, 0, ("A",), 0),
        ("GC", 71, 1, ("A",), 1),
    ]
    assert len(instances["mutator"]) == 1
    assert len(instances["evaluator"]) == 1

    child_record = owner.member(child)
    restored = restore_scores(
        child_record.scores,
        spec.data,
        tmp_path / "restored-child",
    )
    np.testing.assert_array_equal(
        restored.plddt["A"][0],
        np.asarray([0.5, 0.9], dtype=np.float32),
    )
    assert restored.complex_paths["A"].read_bytes() == b"model:GC\n"
    assert _candidate_label(child_record.candidate) == "GC"
    metrics = spec.data.get(
        child_record.metrics,
        spec.data._codecs[
            "astrolabe.design.native-refinement-metrics/v1"
        ],
    )
    assert metrics["evaluation_id"] == 1
    assert metrics["tokens"] == "GC"
    assert metrics["numpy_scalar"] == pytest.approx(0.7)

    # Native activation is scoped. The proposal mutation does not leak into the
    # owner after either call.
    assert _candidate_label(spec.translation.candidate) == "AC"
    assert global_config.segments["shared"].sequence == "AC"

    owner.shutdown()
    owner.shutdown()
    assert instances["mutator"][0].shutdown_calls == 1
    assert instances["evaluator"][0].shutdown_calls == 1


def test_nonfinite_measurement_fails_without_inventing_a_member(tmp_path):
    spec, _, instances, _, _ = _fixture(tmp_path)

    def invalid_measurement(candidate, scores, job):
        values = np.asarray(scores.plddt["A"][0], dtype=float)
        score = float(values.mean())
        return NativeMeasurement(
            score=float("nan") if job.id else score,
            metrics={"job": job.id},
            structure_path=str(scores.complex_paths["A"]),
        )

    invalid = NativeRefinementSpec(
        output=spec.output,
        seed=spec.seed,
        width=spec.width,
        proposals_per_parent=spec.proposals_per_parent,
        max_evaluations=spec.max_evaluations,
        evaluator_alias=spec.evaluator_alias,
        mutator_alias=spec.mutator_alias,
        assemblies=spec.assemblies,
        initial_candidate=spec.initial_candidate,
        context=spec.context,
        translation=spec.translation,
        data=spec.data,
        load_mutator=spec.load_mutator,
        load_evaluator=spec.load_evaluator,
        propose=spec.propose,
        evaluate=spec.evaluate,
        measure=invalid_measurement,
        label=spec.label,
        proposal_seed=spec.proposal_seed,
    )
    owner = NativeRefinementOwner(invalid)
    root = owner.execute(owner.root_job(protocol=Key("2" * 64))).value
    proposal = owner.execute(
        BranchJob(
            phase="propose",
            id=1,
            generation=1,
            seed=71 * 100003 + 1,
            parent=root,
            slot=1,
        )
    ).value
    with pytest.raises(ValueError, match="finite"):
        owner.execute(
            BranchJob(
                phase="evaluate",
                id=1,
                generation=1,
                seed=71,
                parent=root,
                proposal=proposal,
                slot=1,
            )
        )
    assert owner.evaluations == 1
    owner.shutdown()
    assert len(instances["mutator"]) == 1


class OneBranchPolicy:
    """Test-only policy proving the trial uses the generic control boundary."""

    request_type = BranchAction
    result_type = BranchEvent

    def start(self, root_job):
        return {
            "jobs": (root_job,),
            "index": 0,
            "root": None,
            "proposal": None,
            "members": (),
            "trace": (),
        }

    def finished(self, state):
        return state["index"] >= len(state["jobs"])

    def ask(self, state):
        return BranchAction(state, jobs=(state["jobs"][state["index"]],), wait="all")

    def tell(self, state, event):
        (terminal,) = event.terminals
        job = state["jobs"][state["index"]]
        trace = state["trace"] + (job.phase,)
        root, proposal = state["root"], state["proposal"]
        members = state["members"]
        if job.phase == "evaluate" and proposal is None:
            root = terminal.value
            members = (root,)
            evaluate_job = BranchJob(
                phase="propose",
                id=1,
                generation=1,
                seed=71 * 100003 + 1,
                parent=root,
                slot=1,
            )
            jobs = state["jobs"] + (evaluate_job,)
        elif job.phase == "propose":
            proposal = terminal.value
            # A proposal ordinal and an evaluation ordinal are different job
            # identities even when their integer IDs coincide.
            evaluate_job = BranchJob(
                phase="evaluate",
                id=1,
                generation=1,
                seed=71,
                parent=root,
                proposal=proposal,
                slot=1,
            )
            jobs = state["jobs"] + (evaluate_job,)
        else:
            members += (terminal.value,)
            jobs = state["jobs"]
        return {
            **state,
            "index": state["index"] + 1,
            "jobs": jobs,
            "root": root,
            "proposal": proposal,
            "members": members,
            "trace": trace,
        }

    def selected(self, state):
        return SimpleNamespace(
            frontier=(
                max(state["members"], key=lambda member: (member.score, -member.id)),
            ),
            source=state,
        )


def test_control_run_preserves_serial_evaluate_propose_evaluate_order(tmp_path):
    spec, _, _, prediction_seeds, proposal_seeds = _fixture(tmp_path)
    result = run_native_refinement(
        spec,
        policy=OneBranchPolicy(),
        protocol=Key("4" * 64),
    )
    assert result.state.source["trace"] == ("evaluate", "propose", "evaluate")
    assert prediction_seeds == [71, 71]
    assert proposal_seeds == [71 * 100003 + 1]
    assert result.evaluations == 2
    assert result.proposals == 1
    selected = result.members()
    assert len(selected) == 1
    assert selected[0].label == "GC"
    result.owner.shutdown()


def test_generic_branching_policy_interleaves_each_proposal_with_its_evaluation(
    tmp_path,
):
    spec, _, _, prediction_seeds, proposal_seeds = _fixture(tmp_path)
    events = []
    spec = NativeRefinementSpec(
        output=spec.output,
        seed=spec.seed,
        width=1,
        proposals_per_parent=1,
        max_evaluations=2,
        evaluator_alias=spec.evaluator_alias,
        mutator_alias=spec.mutator_alias,
        assemblies=spec.assemblies,
        initial_candidate=spec.initial_candidate,
        context=spec.context,
        translation=spec.translation,
        data=spec.data,
        load_mutator=spec.load_mutator,
        load_evaluator=spec.load_evaluator,
        propose=spec.propose,
        evaluate=spec.evaluate,
        measure=spec.measure,
        label=spec.label,
        proposal_seed=spec.proposal_seed,
        on_event=lambda name, fields: events.append(name),
    )
    policy = BranchingPolicy(
        width=1,
        proposals_per_parent=1,
        max_evaluations=2,
        proposal_cap=20,
        seed=71,
        evaluation_seed=71,
        quota_mode="attempts",
        expansion="round_robin",
        proposal_batch_size=1,
        evaluation_width=1,
        pipeline=False,
        immediate_evaluation=True,
    )
    result = run_native_refinement(
        spec,
        policy=policy,
        protocol=Key("5" * 64),
    )
    assert events == [
        "evaluation_started",
        "evaluation_completed",
        "proposal_started",
        "proposal_completed",
        "evaluation_started",
        "evaluation_completed",
    ]
    assert prediction_seeds == [71, 71]
    assert proposal_seeds == [71 * 100003 + 1]
    assert result.state.finished
    assert result.state.records
    assert result.members()[0].label == "GC"
    result.owner.shutdown()


def test_duplicate_proposals_consume_attempt_quota_without_replacement(tmp_path):
    spec, _, _, prediction_seeds, _ = _fixture(tmp_path)
    events = []

    def propose_duplicate(
        mutator,
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
        return NativeProposal(candidate, spec.label(candidate))

    spec = replace(
        spec,
        max_evaluations=2,
        proposals_per_parent=2,
        propose=propose_duplicate,
        on_event=lambda name, fields: events.append(name),
    )
    policy = BranchingPolicy(
        width=1,
        proposals_per_parent=2,
        max_evaluations=2,
        proposal_cap=20,
        seed=71,
        evaluation_seed=71,
        quota_mode="attempts",
        expansion="round_robin",
        proposal_batch_size=1,
        evaluation_width=1,
        pipeline=False,
        immediate_evaluation=True,
        max_generations=1,
    )
    result = run_native_refinement(
        spec,
        policy=policy,
        protocol=Key("6" * 64),
    )
    assert prediction_seeds == [71]
    assert result.proposals == 2
    assert result.state.proposals == 2
    assert result.state.duplicates == 2
    assert len(result.state.records) == 1
    assert events.count("duplicate_proposal") == 2
    result.owner.shutdown()


def test_control_run_failure_shuts_down_evaluator_and_mutator(tmp_path):
    spec, _, instances, _, _ = _fixture(tmp_path)
    original_measure = spec.measure

    def fail_child_measurement(candidate, scores, job):
        result = original_measure(candidate, scores, job)
        if job.id == 0:
            return result
        return replace(result, score=float("nan"))

    spec = replace(
        spec,
        max_evaluations=2,
        measure=fail_child_measurement,
    )
    policy = BranchingPolicy(
        width=1,
        proposals_per_parent=1,
        max_evaluations=2,
        proposal_cap=20,
        seed=71,
        evaluation_seed=71,
        quota_mode="attempts",
        immediate_evaluation=True,
    )
    with pytest.raises(ValueError, match="finite"):
        run_native_refinement(
            spec,
            policy=policy,
            protocol=Key("7" * 64),
        )
    assert instances["mutator"][0].shutdown_calls == 1
    assert instances["evaluator"][0].shutdown_calls == 1


def test_evaluator_construction_failure_cleans_up_mutator(tmp_path):
    spec, _, instances, _, _ = _fixture(tmp_path)

    def fail_evaluator():
        raise RuntimeError("evaluator startup failed")

    broken = replace(spec, load_evaluator=fail_evaluator)
    with pytest.raises(RuntimeError, match="evaluator startup"):
        NativeRefinementOwner(broken)
    assert instances["mutator"][0].shutdown_calls == 1
