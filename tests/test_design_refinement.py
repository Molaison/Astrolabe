from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jsonschema
import pytest

from astrolabe.design.coordinator import run
from astrolabe.design.native_support import numeric_step
from astrolabe.design.native_trial import RefinementTrial
from astrolabe.design.operations import operation
from astrolabe.design.phased_search import PhasedBeamSearchPolicy
from astrolabe.design.runtime import Runtime
from astrolabe.design.search import (
    BeamSearchPolicy,
    PopulationSearchPolicy,
    TrajectorySearchPolicy,
)
from astrolabe.design.store import DataStore, JSONCodec
from astrolabe.design.types import (
    AssessRequest,
    Assessment,
    Assignment,
    Candidate,
    Dependency,
    Domain,
    Evidence,
    EvidenceUse,
    History,
    Occurrence,
    Polymer,
    PredictRequest,
    Proposal,
    Proposals,
    ProposeRequest,
    Sample,
    SamplingPolicy,
    Scored,
    System,
    Work,
    key_for,
    make_candidate,
    make_system,
    manifest,
)


class ProposalAdapter:
    """Typed deterministic proposer used to exercise policy composition."""

    def __init__(
        self,
        system: System,
        *,
        count: int,
        duplicate: bool = False,
    ) -> None:
        self.system = system
        self.count = count
        self.duplicate = duplicate
        self.calls: list[ProposeRequest] = []
        self.batch_sizes: list[int] = []

    def __call__(self, request: ProposeRequest) -> Proposals:
        self.calls.append(request)
        if self.duplicate:
            children = (request.parents[0],)
        else:
            token_pairs = (("G", "A"), ("A", "G"), ("G", "G"))
            children = tuple(
                self.system.edit(
                    request.parents[0],
                    (
                        Assignment(
                            "shared",
                            Polymer("protein", tokens, ()),
                        ),
                    ),
                )
                for tokens in token_pairs[: self.count]
            )
        proposals = tuple(
            Proposal(
                identity=key_for(
                    (
                        "refinement-test/proposal/v1",
                        request.sample,
                        child.identity,
                    )
                ),
                parents=tuple(parent.identity for parent in request.parents),
                base=request.working.identity,
                candidate=child,
                used_evidence=tuple(
                    item.evidence.identity for item in request.evidence
                ),
                protocol=request.sampling.identity,
            )
            for child in children
        )
        self.batch_sizes.append(len(proposals))
        return Proposals(proposals)


def _prediction_evidence(
    request: PredictRequest,
    *,
    evaluated_candidate: Candidate,
) -> Evidence:
    dependencies = [
        Dependency("system", request.system.identity),
        Dependency("candidate", evaluated_candidate.identity),
    ]
    if request.samples:
        dependencies.append(
            Dependency("sample", key_for(request.samples[0]))
        )
    inputs = manifest(tuple(dependencies))
    artifacts = ()
    measurements = ()
    return Evidence(
        identity=key_for(
            (
                "refinement-test/evidence/v1",
                evaluated_candidate.identity,
                artifacts,
                measurements,
                inputs,
            )
        ),
        evaluated_candidate=evaluated_candidate.identity,
        artifacts=artifacts,
        measurements=measurements,
        inputs=inputs,
    )


class PredictionAdapter:
    def __init__(
        self,
        root: Candidate,
        *,
        mismatch_children: bool = False,
    ) -> None:
        self.root = root
        self.mismatch_children = mismatch_children
        self.calls: list[PredictRequest] = []
        self.sample_steps: list[tuple[Candidate, int]] = []

    def __call__(self, request: PredictRequest) -> Evidence:
        self.calls.append(request)
        sample = request.samples[0]
        self.sample_steps.append((request.candidate, numeric_step(sample)))
        evaluated_candidate = request.candidate
        if (
            self.mismatch_children
            and request.candidate.identity != self.root.identity
        ):
            evaluated_candidate = self.root
        return _prediction_evidence(
            request,
            evaluated_candidate=evaluated_candidate,
        )


def _objective_value(candidate: Candidate) -> float:
    tokens = candidate.assignments[0].value.tokens
    return {
        ("A", "A"): 1.0,
        ("A", "G"): 2.0,
        ("G", "A"): 3.0,
        ("G", "G"): 4.0,
    }.get(tokens, 0.5)


class AssessmentAdapter:
    def __init__(self) -> None:
        self.calls: list[AssessRequest] = []

    def __call__(self, request: AssessRequest) -> Assessment:
        self.calls.append(request)
        inputs = manifest(
            (
                Dependency("evidence", request.evidence.identity),
                Dependency("history", request.history.revision),
            )
        )
        return Assessment(
            evidence=request.evidence.identity,
            objective_revision=request.objective_revision,
            history_revision=request.history.revision,
            values=(_objective_value(request.candidate),),
            gates=(),
            inputs=inputs,
        )


@dataclass(frozen=True)
class _Fixture:
    trial: RefinementTrial
    system: System
    root: Candidate
    proposer: ProposalAdapter
    predictor: PredictionAdapter
    assessor: AssessmentAdapter
    sample: Sample


def _fixture(
    *,
    proposal_count: int = 1,
    duplicate: bool = False,
    mismatch_children: bool = False,
) -> _Fixture:
    store = DataStore()
    json_codec = JSONCodec("refinement-test/object/v1")
    topology = store.put({"nodes": ["shared"]}, json_codec)
    first_index = store.put({"positions": [0, 1]}, json_codec)
    second_index = store.put({"positions": [0, 1]}, json_codec)
    occurrences = (
        Occurrence(
            identity=key_for("refinement-test/occurrence/v1/first"),
            state="state",
            assembly="assembly",
            chain="A",
            component="component.shared",
            variable="shared",
            index_map=first_index,
        ),
        Occurrence(
            identity=key_for("refinement-test/occurrence/v1/second"),
            state="state",
            assembly="assembly",
            chain="B",
            component="component.shared",
            variable="shared",
            index_map=second_index,
        ),
    )
    system = make_system(
        ("shared",),
        occurrences,
        topology,
        (Domain("shared", (), (), None),),
    )
    root = make_candidate(
        system,
        (
            Assignment(
                "shared",
                Polymer("protein", ("A", "A"), ()),
            ),
        ),
    )
    sample = Sample(
        protocol=key_for("refinement-test/sampling/v1"),
        replicate="root",
        seed=17,
        index="0",
    )
    sampling = SamplingPolicy(
        key_for("refinement-test/sampling/v1"),
        store.put({"temperature": 1.0}, json_codec),
    )
    proposer = ProposalAdapter(
        system,
        count=proposal_count,
        duplicate=duplicate,
    )
    predictor = PredictionAdapter(
        root,
        mismatch_children=mismatch_children,
    )
    assessor = AssessmentAdapter()
    options_schema = {
        "type": "object",
        "properties": {"count": {"type": "integer", "minimum": 1}},
        "required": ["count"],
        "additionalProperties": False,
    }
    propose = operation(
        "propose",
        ProposeRequest,
        Proposals,
        proposer,
        version="refinement-test/v1",
        options={"count": proposal_count},
        schema=options_schema,
    )
    predict = operation(
        "predict",
        PredictRequest,
        Evidence,
        predictor,
        version="refinement-test/v1",
        paid_boundary="prediction",
        options={"kind": "current"},
        schema={
            "type": "object",
            "properties": {"kind": {"const": "current"}},
            "required": ["kind"],
            "additionalProperties": False,
        },
    )
    assess = operation(
        "assess",
        AssessRequest,
        Assessment,
        assessor,
        version="refinement-test/v1",
        options={"rank": "evidence"},
        schema={
            "type": "object",
            "properties": {"rank": {"const": "evidence"}},
            "required": ["rank"],
            "additionalProperties": False,
        },
    )
    trial = RefinementTrial(
        system,
        predict=predict,
        propose=propose,
        assess=assess,
        sampling=sampling,
        assemblies=("assembly",),
        root_sample=sample,
    )
    return _Fixture(
        trial=trial,
        system=system,
        root=root,
        proposer=proposer,
        predictor=predictor,
        assessor=assessor,
        sample=sample,
    )


def _common_policy(kind: str, *, width: int = 1):
    policy_type = {
        "beam": BeamSearchPolicy,
        "population": PopulationSearchPolicy,
        "trajectory": TrajectorySearchPolicy,
    }[kind]
    return policy_type(
        rounds=1,
        width=width,
        offspring=1,
        max_work=2,
        seed=29,
        directions=("max",),
    )


def _all_policy_families():
    return (
        ("beam", _common_policy("beam")),
        ("population", _common_policy("population")),
        ("trajectory", _common_policy("trajectory")),
        (
            "phased_beam",
            PhasedBeamSearchPolicy(
                width=1,
                proposals_per_parent=1,
                max_evaluations=2,
                proposal_cap=2,
                seed=29,
                evaluation_seed=31,
            ),
        ),
    )


def _run_fixture(fixture: _Fixture, policy, *, prediction_budget: int):
    history = History(key_for(("refinement-test/history/v1", policy.kind)), ())
    runtime = Runtime(
        fixture.trial.parts,
        budgets={"prediction": prediction_budget},
    )
    result = run(
        fixture.system,
        fixture.root,
        fixture.trial,
        policy,
        runtime,
        capacity=8,
    )
    return result, runtime, history


def _synthetic_scored(fixture: _Fixture, candidate: Candidate) -> Scored:
    request = PredictRequest(
        fixture.system,
        candidate,
        ("assembly",),
        (),
        (),
        (fixture.sample,),
    )
    evidence = _prediction_evidence(
        request,
        evaluated_candidate=candidate,
    )
    inputs = manifest(
        (
            Dependency("evidence", evidence.identity),
            Dependency(
                "history",
                key_for("refinement-test/history/v1"),
            ),
        )
    )
    assessment = Assessment(
        evidence=evidence.identity,
        objective_revision=fixture.trial.assess.descriptor.identity,
        history_revision=key_for("refinement-test/history/v1"),
        values=(_objective_value(candidate),),
        gates=(),
        inputs=inputs,
    )
    return Scored(candidate, evidence, assessment)


def test_fake_typed_operations_use_installed_jsonschema() -> None:
    assert jsonschema.__file__ is not None
    assert Path(jsonschema.__file__).is_file()
    with pytest.raises(jsonschema.ValidationError):
        operation(
            "propose",
            ProposeRequest,
            Proposals,
            lambda _: Proposals(()),
            version="refinement-test/v1",
            options={"count": 0},
            schema={
                "type": "object",
                "properties": {
                    "count": {"type": "integer", "minimum": 1}
                },
                "required": ["count"],
                "additionalProperties": False,
            },
        )


def test_same_refinement_trial_is_interchangeable_across_policy_families() -> None:
    fixture = _fixture()
    assert fixture.system.variables == ("shared",)
    assert {
        occurrence.variable for occurrence in fixture.system.occurrences
    } == {"shared"}
    assert tuple(
        component.descriptor.alias for component in fixture.trial.parts
    ) == ("propose", "predict", "assess")
    assert not hasattr(fixture.trial.propose, "initialization")
    assert not hasattr(fixture.trial.predict, "initialization")

    observed_rounds: dict[str, tuple[str, ...]] = {}
    for name, policy in _all_policy_families():
        result, runtime, _ = _run_fixture(
            fixture,
            policy,
            prediction_budget=2,
        )
        try:
            assert result.selected
            assert all(
                scored.candidate.identity != fixture.root.identity
                for scored in result.selected
            )
            rounds = tuple(
                attempt.key.named_round
                for attempt in runtime.attempts()
                if attempt.key.named_round.endswith("/predict")
            )
            assert all(
                attempt.status == "completed"
                for attempt in runtime.attempts()
            )
            observed_rounds[name] = rounds
        finally:
            runtime.close()

    for name in ("beam", "population", "trajectory"):
        assert observed_rounds[name] == (
            "root/predict",
            "child/0/predict",
        )
    assert observed_rounds["phased_beam"] == (
        "root/predict",
        "evaluate/predict",
    )
    assert len(fixture.proposer.calls) == 4


def test_common_policy_samples_are_positive_native_numeric_steps() -> None:
    fixture = _fixture()
    for kind in ("beam", "population", "trajectory"):
        prior_steps = len(fixture.predictor.sample_steps)
        prior_calls = len(fixture.predictor.calls)
        _, runtime, _ = _run_fixture(
            fixture,
            _common_policy(kind),
            prediction_budget=2,
        )
        try:
            child_steps = tuple(
                step
                for candidate, step in fixture.predictor.sample_steps[
                    prior_steps:
                ]
                if candidate.identity != fixture.root.identity
            )
            assert child_steps == (1,), kind
            child_indices = tuple(
                request.samples[0].index
                for request in fixture.predictor.calls[prior_calls:]
                if request.candidate.identity != fixture.root.identity
            )
            assert child_indices == ("1",), kind
        finally:
            runtime.close()


@pytest.mark.parametrize("kind", ("beam", "population", "trajectory"))
def test_multiple_proposals_get_distinct_paid_child_predictions(kind: str) -> None:
    fixture = _fixture(proposal_count=2)
    width = 1 if kind == "trajectory" else 2
    result, runtime, _ = _run_fixture(
        fixture,
        _common_policy(kind, width=width),
        prediction_budget=3,
    )
    try:
        assert fixture.proposer.batch_sizes == [2]
        assert len(fixture.predictor.calls) == 3
        child_calls = tuple(
            request
            for request in fixture.predictor.calls
            if request.candidate.identity != fixture.root.identity
        )
        assert len(child_calls) == 2
        assert len(
            {request.candidate.identity for request in child_calls}
        ) == 2

        prediction_rounds = tuple(
            attempt.key.named_round
            for attempt in runtime.attempts()
            if attempt.key.named_round.endswith("/predict")
        )
        assert prediction_rounds == (
            "root/predict",
            "child/0/predict",
            "child/1/predict",
        )
        assert len(prediction_rounds) == runtime.budgets["prediction"] == 3
        assert result.selected
    finally:
        runtime.close()


def test_malformed_stages_and_multi_parent_work_are_rejected_before_calls() -> None:
    fixture = _fixture()
    runtime = Runtime(fixture.trial.parts)
    try:
        root = _synthetic_scored(fixture, fixture.root)
        other_candidate = fixture.system.edit(
            fixture.root,
            (
                Assignment(
                    "shared",
                    Polymer("protein", ("A", "G"), ()),
                ),
            ),
        )
        other = _synthetic_scored(fixture, other_candidate)

        with pytest.raises(
            ValueError,
            match="Unsupported refinement stage",
        ):
            fixture.trial.perform(
                Work(
                    identity="unsupported",
                    parents=(root,),
                    stage="unsupported",
                    sample=fixture.sample,
                ),
                runtime,
            )
        with pytest.raises(
            ValueError,
            match="declared parents",
        ):
            fixture.trial.perform(
                Work(
                    identity="no-parent",
                    parents=(),
                    stage="propose",
                    sample=fixture.sample,
                ),
                runtime,
            )
        with pytest.raises(
            ValueError,
            match="explicit working candidate",
        ):
            fixture.trial.perform(
                Work(
                    identity="multi-parent",
                    parents=(root, other),
                    stage="search",
                    sample=fixture.sample,
                ),
                runtime,
            )
        with pytest.raises(
            ValueError,
            match="Current evidence must belong",
        ):
            EvidenceUse(
                other.evidence,
                "current",
                root.candidate.identity,
            )
        assert runtime.attempts() == ()
    finally:
        runtime.close()


def test_child_prediction_evidence_must_match_current_candidate() -> None:
    fixture = _fixture(mismatch_children=True)
    runtime = Runtime(
        fixture.trial.parts,
        budgets={"prediction": 2},
    )
    try:
        with pytest.raises(
            ValueError,
            match="Evidence belongs to another candidate",
        ):
            run(
                fixture.system,
                fixture.root,
                fixture.trial,
                _common_policy("beam"),
                runtime,
                capacity=8,
            )
        rounds = tuple(
            attempt.key.named_round
            for attempt in runtime.attempts()
        )
        assert rounds == (
            "root/predict",
            "root/assess",
            "propose",
            "child/0/predict",
            "child/0/assess",
        )
        assert rounds.count("child/0/predict") == 1
    finally:
        runtime.close()
