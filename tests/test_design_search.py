from __future__ import annotations

import random

import pytest

from astrolabe.design.coordinator import run as run_coordinator
from astrolabe.design.runtime import Runtime
from astrolabe.design.search import (
    BeamSearchPolicy,
    PopulationSearchPolicy,
    TrajectorySearchPolicy,
)
from astrolabe.design.store import DataStore
from astrolabe.design.types import (
    Assessment,
    Assignment,
    Blob,
    Candidate,
    Domain,
    Evidence,
    History,
    InputManifest,
    Polymer,
    Scored,
    TrialOutcome,
    key_for,
    make_candidate,
    make_system,
)


INPUTS = InputManifest(key_for(("test-input/v1",)), ())
HISTORY = History(key_for(("empty-history/v1",)), ())


def _scored(
    label: str,
    values: tuple[float, ...],
    *,
    gates: tuple[tuple[str, bool], ...] = (),
) -> Scored:
    system = key_for(("system/v1", "search-test"))
    candidate = Candidate(
        identity=key_for(("candidate/v1", system, label)),
        system=system,
        assignments=(),
    )
    evidence = Evidence(
        identity=key_for(("evidence/v1", candidate.identity)),
        evaluated_candidate=candidate.identity,
        artifacts=(),
        measurements=(),
        inputs=INPUTS,
    )
    assessment = Assessment(
        evidence=evidence.identity,
        objective_revision=key_for(("objective/v1", len(values))),
        history_revision=HISTORY.revision,
        values=tuple(float(value) for value in values),
        gates=tuple(gates),
        inputs=INPUTS,
    )
    return Scored(candidate, evidence, assessment)


def _outcome(work_id: str, *children: Scored, status: str = "completed") -> TrialOutcome:
    return TrialOutcome(work_id, status, (), tuple(children), ())


def test_start_counts_roots_ask_is_stable_and_samples_are_distinct() -> None:
    policy = BeamSearchPolicy(
        rounds=2,
        width=2,
        offspring=2,
        max_work=5,
        seed=11,
        directions=("max",),
        expand_parents="round_robin",
    )
    root = _scored("root", (0.5,))
    state = policy.start((root,))

    assert state.used == 1
    assert state.max_work == 5
    before_random = random.getstate()
    first = policy.ask(state, HISTORY, capacity=10)
    second = policy.ask(state, HISTORY, capacity=10)

    assert first == second
    assert len(first) == 2
    assert len({work.identity for work in first}) == 2
    assert len({work.sample.replicate for work in first}) == 2
    assert len({work.sample.index for work in first}) == 2
    assert random.getstate() == before_random
    assert state.used == 1


def test_explicit_directions_gate_rejection_and_stable_ties() -> None:
    policy = PopulationSearchPolicy(
        rounds=1,
        width=3,
        offspring=1,
        max_work=10,
        seed=7,
        directions=("max", "min"),
    )
    low_first = _scored("low-first", (1.0, 9.0))
    better_second = _scored("better-second", (2.0, 9.0))
    tied = _scored("tied", (3.0, 8.0))
    best_second = _scored("best-second", (3.0, 8.0))
    gated_high = _scored("gated-high", (99.0, 0.0), gates=(("pose", False),))

    selected = policy.start((low_first, gated_high, tied, best_second, better_second)).frontier

    assert selected == (tied, best_second, better_second)
    reversed_state = policy.start((best_second, tied))
    assert reversed_state.frontier[:2] == (best_second, tied)


@pytest.mark.parametrize(
    ("mode", "expected_labels"),
    [
        ("each", ("left", "left", "right", "right")),
        ("round_robin", ("left", "right", "left", "right")),
    ],
)
def test_expansion_mode_controls_parent_order(
    mode: str, expected_labels: tuple[str, ...]
) -> None:
    policy = BeamSearchPolicy(
        rounds=1,
        width=2,
        offspring=2,
        max_work=4,
        seed=3,
        directions=("max",),
        expand_parents=mode,
        root_counts_budget=False,
    )
    left = _scored("left", (2.0,))
    right = _scored("right", (1.0,))
    state = policy.start((right, left))

    works = policy.ask(state, HISTORY, capacity=10)
    labels = tuple(
        work.parents[0].candidate.identity.digest
        for work in works
    )
    by_digest = {
        left.candidate.identity.digest: "left",
        right.candidate.identity.digest: "right",
    }
    assert tuple(by_digest[label] for label in labels) == expected_labels


def test_partial_tell_keeps_frozen_batch_and_recharges_each_work_once() -> None:
    policy = TrajectorySearchPolicy(
        rounds=2,
        offspring=4,
        max_work=6,
        seed=5,
        directions=("max",),
    )
    state = policy.start((_scored("root", (0.0,)),))
    first_two = policy.ask(state, HISTORY, capacity=2)
    assert state.round_target == 4

    state = policy.tell(
        state,
        (
            _outcome(first_two[0].identity, _scored("one", (1.0,))),
            _outcome(first_two[1].identity, _scored("two", (2.0,))),
        ),
    )
    assert state.used == 3
    assert state.round == 0
    remaining = policy.ask(state, HISTORY, capacity=10)
    assert len(remaining) == 2
    assert {work.identity for work in remaining}.isdisjoint(
        {work.identity for work in first_two}
    )
    with pytest.raises(ValueError, match="more than once"):
        policy.tell(
            state,
            (_outcome(first_two[0].identity, _scored("again", (9.0,))),),
        )


def test_failure_and_duplicate_children_consume_work_not_prediction_budget() -> None:
    policy = BeamSearchPolicy(
        rounds=1,
        width=2,
        offspring=2,
        max_work=4,
        seed=13,
        directions=("max",),
        root_counts_budget=False,
    )
    root = _scored("root", (1.0,))
    state = policy.start((root,))
    works = policy.ask(state, HISTORY, capacity=10)

    state = policy.tell(
        state,
        (
            _outcome(works[0].identity, _scored("duplicate", (2.0,))),
            _outcome(works[1].identity, _scored("duplicate", (2.0,))),
        ),
    )

    assert state.used == 2
    assert state.deduplicated_children == 1
    assert len(state.frontier) == 2  # retained root plus one unique child
    assert state.status_counts == (
        ("completed", 2),
        ("failed", 0),
        ("unknown", 0),
        ("refused", 0),
    )
    failed_policy = TrajectorySearchPolicy(
        rounds=1,
        offspring=1,
        max_work=1,
        seed=17,
        directions=("max",),
        root_counts_budget=False,
    )
    failed_state = failed_policy.start((root,))
    work = failed_policy.ask(failed_state, HISTORY, capacity=1)[0]
    failed_state = failed_policy.tell(
        failed_state,
        (_outcome(work.identity, status="failed"),),
    )
    assert failed_state.used == 1
    assert failed_policy.finished(failed_state)
    assert dict(failed_state.status_counts)["failed"] == 1


def test_retained_parent_precedes_an_equal_child_in_stable_order() -> None:
    policy = BeamSearchPolicy(
        rounds=1,
        width=2,
        offspring=1,
        max_work=1,
        seed=23,
        directions=("max",),
        root_counts_budget=False,
    )
    root = _scored("root", (1.0,))
    child = _scored("child", (1.0,))
    state = policy.start((root,))
    work = policy.ask(state, HISTORY, capacity=1)[0]
    state = policy.tell(state, (_outcome(work.identity, child),))

    assert state.frontier == (root, child)


def test_parent_limit_and_offspring_only_retention_are_explicit_policy_controls() -> None:
    policy = PopulationSearchPolicy(
        rounds=1,
        width=3,
        offspring=1,
        max_work=1,
        seed=31,
        directions=("max",),
        parents_per_round=1,
        retain_parents=False,
        root_counts_budget=False,
    )
    best = _scored("best-parent", (3.0,))
    state = policy.start(
        (
            _scored("third-parent", (1.0,)),
            best,
            _scored("second-parent", (2.0,)),
        )
    )
    work = policy.ask(state, HISTORY, capacity=1)[0]
    assert work.parents == (best,)

    child = _scored("only-child", (4.0,))
    state = policy.tell(state, (_outcome(work.identity, child),))
    assert state.frontier == (child,)


def test_unknown_is_terminal_for_the_policy_and_never_auto_retried() -> None:
    policy = BeamSearchPolicy(
        rounds=2,
        width=1,
        offspring=1,
        max_work=2,
        seed=19,
        directions=("max",),
        root_counts_budget=False,
    )
    state = policy.start((_scored("root", (1.0,)),))
    work = policy.ask(state, HISTORY, capacity=1)[0]
    state = policy.tell(state, (_outcome(work.identity, status="unknown"),))
    next_work = policy.ask(state, HISTORY, capacity=1)[0]

    assert state.used == 1
    assert state.round == 1
    assert next_work.identity != work.identity
    assert dict(state.status_counts)["unknown"] == 1


def test_policy_checkpoint_round_trip_and_identity() -> None:
    rank = lambda scored: scored.assessment.values  # noqa: E731
    first = BeamSearchPolicy(
        rounds=1,
        width=1,
        offspring=1,
        max_work=2,
        seed=1,
        directions=("max",),
        rank=rank,
        rank_identity="test.values/v1",
    )
    second = BeamSearchPolicy(
        rounds=1,
        width=1,
        offspring=1,
        max_work=2,
        seed=1,
        directions=("max",),
        rank=rank,
        rank_identity="test.values/v2",
    )
    state = first.start((_scored("root", (1.0,)),))
    store = DataStore()
    reference = store.put(state, first.codec)

    assert first.identity != second.identity
    assert store.get(reference, first.codec) == state


def test_capturing_rank_requires_explicit_identity() -> None:
    threshold = 1.0

    def rank(scored: Scored) -> tuple[float, ...]:
        return (scored.assessment.values[0] + threshold,)

    with pytest.raises(ValueError, match="rank_identity"):
        BeamSearchPolicy(
            rounds=1,
            width=1,
            offspring=1,
            max_work=2,
            seed=1,
            directions=("max",),
            rank=rank,
        )

    policy = BeamSearchPolicy(
        rounds=1,
        width=1,
        offspring=1,
        max_work=2,
        seed=1,
        directions=("max",),
        rank=rank,
        rank_identity="test.capturing-rank/v1",
    )
    assert policy.rank_identity == "test.capturing-rank/v1"


def test_coordinator_uses_standard_policy_protocol_end_to_end() -> None:
    topology = Blob(key_for(("test-topology/v1",)), "test-topology/v1")
    domain = Domain(
        variable="shared",
        fixed_tokens=(0,),
        fixed_atoms=(),
        allowed_values=None,
    )
    system = make_system(
        variables=("shared",),
        occurrences=(),
        topology=topology,
        domains=(domain,),
    )
    root_candidate = make_candidate(
        system,
        (Assignment("shared", Polymer("protein", ("A", "C", "D"), ())),),
    )
    child_candidate = make_candidate(
        system,
        (Assignment("shared", Polymer("protein", ("A", "C", "E"), ())),),
    )

    def evidence_for(candidate: Candidate) -> Evidence:
        evidence_identity = key_for(
            ("evidence/v1", candidate.identity, (), (), INPUTS)
        )
        return Evidence(
            identity=evidence_identity,
            evaluated_candidate=candidate.identity,
            artifacts=(),
            measurements=(),
            inputs=INPUTS,
        )

    def scored(candidate: Candidate, value: float) -> Scored:
        evidence = evidence_for(candidate)
        assessment = Assessment(
            evidence=evidence.identity,
            objective_revision=key_for(("objective/v1", "test")),
            history_revision=HISTORY.revision,
            values=(value,),
            gates=(),
            inputs=INPUTS,
        )
        return Scored(candidate, evidence, assessment)

    root = scored(root_candidate, 0.0)
    child = scored(child_candidate, 1.0)

    class Trial:
        identity = key_for(("test-trial/v1",))
        requirements = ()
        fail_fast = True

        def root(self, initial: Candidate, runtime: Runtime) -> Scored:
            assert initial == root_candidate
            return root

        def perform(self, work, runtime: Runtime) -> TrialOutcome:
            return TrialOutcome(work.identity, "completed", (), (child,), ())

    policy = BeamSearchPolicy(
        rounds=1,
        width=1,
        offspring=1,
        max_work=1,
        seed=29,
        directions=("max",),
        root_counts_budget=False,
    )
    result = run_coordinator(
        system,
        root_candidate,
        Trial(),
        policy,
        Runtime(()),
        capacity=1,
    )

    assert result.selected == (child,)
    assert result.manifest.policy_codec == policy.codec.identity
