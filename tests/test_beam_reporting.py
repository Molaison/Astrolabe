from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from astrolabe.beam_reporting import BeamCandidateView, build_beam_report
from astrolabe.design.phased_search import PhasedBeamSearchPolicy
from astrolabe.design.store import DataStore
from astrolabe.design.types import (
    Assessment,
    Assignment,
    Attempt,
    Blob,
    CallKey,
    Candidate,
    Domain,
    Evidence,
    History,
    InputManifest,
    Polymer,
    Proposal,
    Scored,
    TrialOutcome,
    Work,
    key_for,
    make_candidate,
    make_system,
)


INPUTS = InputManifest(key_for(("beam-report-input/v1",)), ())
TOPOLOGY = Blob(key_for(("beam-report-topology/v1",)), "beam-report-topology/v1")
SYSTEM = make_system(
    variables=("chain",),
    occurrences=(),
    topology=TOPOLOGY,
    domains=(Domain("chain", (0,), (), None),),
)
PREDICTION = "predictor"
PREDICTION_IDENTITY = key_for(("beam-report-predictor/v1",))


def candidate(*tokens: str) -> Candidate:
    return make_candidate(
        SYSTEM,
        (Assignment("chain", Polymer("protein", tuple(tokens), ())),),
    )


def sequence_of(value: Candidate) -> str:
    polymer = value.assignments[0].value
    assert isinstance(polymer, Polymer)
    return "".join(polymer.tokens)


ROOT_CANDIDATE = candidate("A", "C", "D")


def scored(
    value: Candidate,
    score: float,
    *,
    values: tuple[float, ...] | None = None,
) -> Scored:
    evidence = Evidence(
        identity=key_for(("beam-report-evidence/v1", value.identity)),
        evaluated_candidate=value.identity,
        artifacts=(),
        measurements=(),
        inputs=INPUTS,
    )
    assessment = Assessment(
        evidence=evidence.identity,
        objective_revision=key_for(("beam-report-objective/v1",)),
        history_revision=key_for(("beam-report-history/v1",)),
        values=(float(score),) if values is None else values,
        gates=(),
        inputs=INPUTS,
    )
    return Scored(value, evidence, assessment)


ROOT = scored(ROOT_CANDIDATE, 1.0)


def proposal(parent: Scored, child: Candidate) -> Proposal:
    return Proposal(
        identity=key_for(
            ("beam-report-proposal/v1", parent.candidate.identity, child.identity)
        ),
        parents=(parent.candidate.identity,),
        base=parent.candidate.identity,
        candidate=child,
        used_evidence=(parent.evidence.identity,),
        protocol=key_for(("beam-report-proposal-protocol/v1",)),
    )


def outcome(
    work: Work,
    *,
    proposals: tuple[Proposal, ...] = (),
    children: tuple[Scored, ...] = (),
    status: str = "completed",
) -> TrialOutcome:
    return TrialOutcome(work.identity, status, proposals, children, ())


@dataclass
class FakeRuntime:
    data: DataStore
    root_completion: Scored | None
    checkpoint: dict | None
    work_entries: dict[str, Work]
    attempt_rows: dict[str, tuple[Attempt, ...]]

    def __post_init__(self) -> None:
        self.operations = {
            PREDICTION: SimpleNamespace(
                descriptor=SimpleNamespace(identity=PREDICTION_IDENTITY)
            )
        }

    def load(self, name, codec=None):
        assert codec is None
        return self.checkpoint if name == "run/state" else None

    def peek_completion(self, name, codec=None):
        assert codec is None and name == "run/root"
        return self.root_completion

    def work_entry(self, work_id):
        return self.work_entries[work_id]

    def attempts(self, work_id=None):
        if work_id is None:
            return tuple(
                attempt
                for attempts in self.attempt_rows.values()
                for attempt in attempts
            )
        return self.attempt_rows.get(work_id, ())


def attempt(work_id: str, status: str, index: int = 0) -> Attempt:
    return Attempt(
        identity=f"attempt-{index}",
        key=CallKey(
            work_id,
            "predict",
            PREDICTION_IDENTITY,
            key_for(("beam-report-input-manifest/v1", work_id)),
        ),
        status=status,
        result=None,
        rng=None,
    )


def runtime_for(
    policy: PhasedBeamSearchPolicy,
    state,
    *,
    root: Scored | None = ROOT,
    records: tuple[TrialOutcome, ...] = (),
    pending: tuple[Work, ...] = (),
    work_entries: dict[str, Work] | None = None,
    attempt_rows: dict[str, tuple[Attempt, ...]] | None = None,
) -> FakeRuntime:
    data = DataStore()
    checkpoint = {
        "state": data.put(state, policy.codec),
        "history": History(key_for(("beam-report-history/v1", records)), records),
        "pending": pending,
        "finished": policy.finished(state),
        "rng": None,
    }
    return FakeRuntime(
        data=data,
        root_completion=root,
        checkpoint=checkpoint,
        work_entries=dict(work_entries or {}),
        attempt_rows=dict(attempt_rows or {}),
    )


def details_of(value: Scored) -> tuple[str, dict]:
    sequence = sequence_of(value.candidate)
    return f"/case/structures/{sequence}.cif", {
        "raw_metric": value.assessment.values[0]
    }


def report_for(runtime: FakeRuntime, policy: PhasedBeamSearchPolicy, initial=ROOT_CANDIDATE):
    return build_beam_report(
        runtime,
        policy,
        initial=initial,
        sequence_of=sequence_of,
        details_of=details_of,
        prediction_alias=PREDICTION,
    )


def test_root_projection_uses_legacy_view_and_events() -> None:
    policy = PhasedBeamSearchPolicy(
        width=1,
        proposals_per_parent=1,
        max_evaluations=1,
        proposal_cap=1,
        seed=7,
        evaluation_seed=11,
        directions=("max",),
    )
    state = policy.start((ROOT,))
    report = report_for(runtime_for(policy, state), policy)

    assert isinstance(report.beam[0], BeamCandidateView)
    assert report.beam[0] == BeamCandidateView(
        evaluation_id=0,
        sequence="ACD",
        score=1.0,
        structure_path="/case/structures/ACD.cif",
        generation=0,
        parent_id=None,
        metrics={"raw_metric": 1.0},
    )
    assert report.records == [
        {
            "evaluation_id": 0,
            "proposal_step": 0,
            "sequence": "ACD",
            "generation": 0,
            "parent_id": None,
            "status": "completed",
            "score": 1.0,
            "structure_path": "/case/structures/ACD.cif",
            "entered_beam": True,
            "in_final_beam": True,
            "raw_metric": 1.0,
            "metrics": {"raw_metric": 1.0},
        }
    ]
    assert [event["event"] for event in report.events] == [
        "evaluation_started",
        "evaluation_completed",
        "beam_selected",
    ]
    assert report.events[0]["metrics"] == {}
    assert report.events[1]["metrics"] == {"raw_metric": 1.0}
    assert report.proposals == 0
    assert report.evaluations == 1
    assert report.duplicates == 0
    assert report.generation == 1
    assert report.budget_complete


def test_duplicate_proposal_has_no_evaluation_row_or_id() -> None:
    policy = PhasedBeamSearchPolicy(
        width=1,
        proposals_per_parent=2,
        max_evaluations=4,
        proposal_cap=4,
        seed=13,
        evaluation_seed=17,
        directions=("max",),
    )
    state = policy.start((ROOT,))
    proposal_work = policy.ask(
        state, History(key_for(("empty/v1",)), ()), capacity=1
    )[0]
    duplicate_outcome = outcome(
        proposal_work,
        proposals=(proposal(ROOT, ROOT_CANDIDATE),),
    )
    state = policy.tell(state, (duplicate_outcome,))
    runtime = runtime_for(
        policy,
        state,
        records=(duplicate_outcome,),
        work_entries={proposal_work.identity: proposal_work},
    )
    report = report_for(runtime, policy)

    assert report.evaluations == 1
    assert report.proposals == 1
    assert report.duplicates == 1
    assert [row["evaluation_id"] for row in report.records] == [0]
    duplicate = next(
        event for event in report.events if event["event"] == "duplicate_proposal"
    )
    assert duplicate["generation"] == 1
    assert duplicate["parent_id"] == 0
    assert duplicate["proposal_index"] == 0
    assert duplicate["sequence"] == "ACD"
    assert duplicate["seed"] == proposal_work.sample.seed


def test_partial_generation_budget_updates_entered_beam_and_selected_event() -> None:
    policy = PhasedBeamSearchPolicy(
        width=2,
        proposals_per_parent=3,
        max_evaluations=2,
        proposal_cap=6,
        seed=19,
        evaluation_seed=23,
        directions=("max",),
    )
    state = policy.start((ROOT,))
    proposal_work = policy.ask(
        state, History(key_for(("empty/v1",)), ()), capacity=1
    )[0]
    child = scored(candidate("A", "C", "E"), 2.0)
    proposal_outcome = outcome(
        proposal_work,
        proposals=(proposal(ROOT, child.candidate),),
    )
    state = policy.tell(state, (proposal_outcome,))
    evaluation_work = policy.ask(
        state, History(key_for(("proposal/v1",)), (proposal_outcome,)), capacity=1
    )[0]
    evaluation_outcome = outcome(evaluation_work, children=(child,))
    state = policy.tell(state, (evaluation_outcome,))
    runtime = runtime_for(
        policy,
        state,
        records=(proposal_outcome, evaluation_outcome),
        work_entries={
            proposal_work.identity: proposal_work,
            evaluation_work.identity: evaluation_work,
        },
    )
    report = report_for(runtime, policy)

    assert report.budget_complete
    assert report.generation == 1
    assert [row["evaluation_id"] for row in report.records] == [0, 1]
    assert report.records[1]["proposal_step"] == 1
    assert report.records[1]["parent_id"] == 0
    assert report.records[1]["entered_beam"] is True
    selected = [event for event in report.events if event["event"] == "beam_selected"]
    assert selected[-1]["generation"] == 1
    assert selected[-1]["candidates"] == [1]
    assert [item["evaluation_id"] for item in selected[-1]["beam"]] == [1, 0]


def test_pending_unknown_prediction_is_visible_without_fabricated_score() -> None:
    policy = PhasedBeamSearchPolicy(
        width=1,
        proposals_per_parent=2,
        max_evaluations=4,
        proposal_cap=4,
        seed=29,
        evaluation_seed=31,
        directions=("max",),
    )
    state = policy.start((ROOT,))
    proposal_work = policy.ask(
        state, History(key_for(("empty/v1",)), ()), capacity=1
    )[0]
    child = candidate("A", "C", "E")
    proposal_outcome = outcome(
        proposal_work,
        proposals=(proposal(ROOT, child),),
    )
    state = policy.tell(state, (proposal_outcome,))
    evaluation_work = policy.ask(
        state, History(key_for(("proposal/v1",)), (proposal_outcome,)), capacity=1
    )[0]
    runtime = runtime_for(
        policy,
        state,
        records=(proposal_outcome,),
        pending=(evaluation_work,),
        work_entries={proposal_work.identity: proposal_work},
        attempt_rows={evaluation_work.identity: (attempt(evaluation_work.identity, "unknown"),)},
    )
    report = report_for(runtime, policy)

    failed = report.records[1]
    assert failed["evaluation_id"] == 1
    assert failed["status"] == "unknown"
    assert failed["score"] is None
    assert failed["structure_path"] == ""
    assert failed["metrics"] == {}
    assert report.evaluations == 2
    assert any(event["event"] == "evaluation_failed" for event in report.events)


def test_stable_equal_scores_keep_parent_before_child_in_view() -> None:
    policy = PhasedBeamSearchPolicy(
        width=2,
        proposals_per_parent=3,
        max_evaluations=2,
        proposal_cap=6,
        seed=37,
        evaluation_seed=41,
        directions=("max",),
    )
    state = policy.start((ROOT,))
    proposal_work = policy.ask(
        state, History(key_for(("empty/v1",)), ()), capacity=1
    )[0]
    child = scored(candidate("A", "C", "E"), 1.0)
    proposal_outcome = outcome(
        proposal_work,
        proposals=(proposal(ROOT, child.candidate),),
    )
    state = policy.tell(state, (proposal_outcome,))
    evaluation_work = policy.ask(
        state, History(key_for(("proposal/v1",)), (proposal_outcome,)), capacity=1
    )[0]
    evaluation_outcome = outcome(evaluation_work, children=(child,))
    state = policy.tell(state, (evaluation_outcome,))
    runtime = runtime_for(
        policy,
        state,
        records=(proposal_outcome, evaluation_outcome),
        work_entries={
            proposal_work.identity: proposal_work,
            evaluation_work.identity: evaluation_work,
        },
    )
    report = report_for(runtime, policy)

    assert [view.evaluation_id for view in report.beam] == [0, 1]
    assert [row["in_final_beam"] for row in report.records] == [True, True]


def test_failed_proposal_adds_no_evaluation_projection() -> None:
    policy = PhasedBeamSearchPolicy(
        width=1,
        proposals_per_parent=1,
        max_evaluations=3,
        proposal_cap=3,
        seed=43,
        evaluation_seed=47,
        directions=("max",),
    )
    state = policy.start((ROOT,))
    proposal_work = policy.ask(
        state, History(key_for(("empty/v1",)), ()), capacity=1
    )[0]
    failed_outcome = outcome(proposal_work, status="failed")
    state = policy.tell(state, (failed_outcome,))
    runtime = runtime_for(
        policy,
        state,
        records=(failed_outcome,),
        work_entries={proposal_work.identity: proposal_work},
    )
    report = report_for(runtime, policy)

    assert report.proposals == 1
    assert report.evaluations == 1
    assert [row["evaluation_id"] for row in report.records] == [0]


def test_root_failure_requires_consumed_prediction_attempt() -> None:
    policy = PhasedBeamSearchPolicy(
        width=1,
        proposals_per_parent=1,
        max_evaluations=2,
        proposal_cap=2,
        seed=53,
        evaluation_seed=59,
        directions=("max",),
    )
    consumed = attempt("root", "completed", index=0)
    runtime = FakeRuntime(
        data=DataStore(),
        root_completion=None,
        checkpoint=None,
        work_entries={},
        attempt_rows={"root": (consumed,)},
    )
    report = report_for(runtime, policy)

    assert report.evaluations == 1
    assert report.records[0]["evaluation_id"] == 0
    assert report.records[0]["status"] == "failed"
    assert report.records[0]["score"] is None
    assert report.beam == []
    assert [event["event"] for event in report.events] == [
        "evaluation_started",
        "evaluation_failed",
    ]

    absent = FakeRuntime(DataStore(), None, None, {}, {})
    with pytest.raises(ValueError, match="consumed prediction attempt"):
        report_for(absent, policy)


def test_completed_child_requires_one_scalar_score() -> None:
    policy = PhasedBeamSearchPolicy(
        width=1,
        proposals_per_parent=2,
        max_evaluations=3,
        proposal_cap=4,
        seed=61,
        evaluation_seed=67,
        directions=("max",),
    )
    state = policy.start((ROOT,))
    proposal_work = policy.ask(
        state, History(key_for(("empty/v1",)), ()), capacity=1
    )[0]
    child = scored(candidate("A", "C", "E"), 0.0, values=(2.0, 1.0))
    proposal_outcome = outcome(
        proposal_work,
        proposals=(proposal(ROOT, child.candidate),),
    )
    state = policy.tell(state, (proposal_outcome,))
    evaluation_work = policy.ask(
        state, History(key_for(("proposal/v1",)), (proposal_outcome,)), capacity=1
    )[0]
    evaluation_outcome = outcome(evaluation_work, children=(child,))
    state = policy.tell(state, (evaluation_outcome,))
    runtime = runtime_for(
        policy,
        state,
        records=(proposal_outcome, evaluation_outcome),
        work_entries={
            proposal_work.identity: proposal_work,
            evaluation_work.identity: evaluation_work,
        },
    )

    with pytest.raises(ValueError, match="exactly one assessment value"):
        report_for(runtime, policy)
