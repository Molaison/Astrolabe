from __future__ import annotations

from dataclasses import replace
import random

import pytest

from astrolabe.design.phased_search import PhasedBeamSearchPolicy
from astrolabe.design.store import DataStore
from astrolabe.design.types import (
    Assessment,
    Assignment,
    Blob,
    Domain,
    Evidence,
    History,
    InputManifest,
    Polymer,
    Proposal,
    Scored,
    TrialOutcome,
    key_for,
    make_candidate,
    make_system,
)


INPUTS = InputManifest(key_for(("phased-test-input/v1",)), ())
HISTORY = History(key_for(("phased-test-history/v1",)), ())
TOPOLOGY = Blob(key_for(("phased-test-topology/v1",)), "phased-test-topology/v1")
SYSTEM = make_system(
    variables=("chain",),
    occurrences=(),
    topology=TOPOLOGY,
    domains=(Domain("chain", (0,), (), None),),
)


def candidate(*tokens: str):
    return make_candidate(
        SYSTEM,
        (Assignment("chain", Polymer("protein", tuple(tokens), ())),),
    )


def scored(value: candidate, objective: float, *, gate: bool = True) -> Scored:
    evidence = Evidence(
        identity=key_for(("phased-test-evidence/v1", value.identity)),
        evaluated_candidate=value.identity,
        artifacts=(),
        measurements=(),
        inputs=INPUTS,
    )
    assessment = Assessment(
        evidence=evidence.identity,
        objective_revision=key_for(("phased-test-objective/v1",)),
        history_revision=HISTORY.revision,
        values=(float(objective),),
        gates=(("valid", gate),),
        inputs=INPUTS,
    )
    return Scored(value, evidence, assessment)


ROOT_CANDIDATE = candidate("A", "C", "D")
ROOT = scored(ROOT_CANDIDATE, 1.0)


def proposal(parent: Scored, child) -> Proposal:
    return Proposal(
        identity=key_for(
            ("phased-test-proposal/v1", parent.candidate.identity, child.identity)
        ),
        parents=(parent.candidate.identity,),
        base=parent.candidate.identity,
        candidate=child,
        used_evidence=(parent.evidence.identity,),
        protocol=key_for(("phased-test-proposal-protocol/v1",)),
    )


def outcome(work, *, proposals=(), children=(), status="completed") -> TrialOutcome:
    return TrialOutcome(
        work.identity,
        status,
        tuple(proposals),
        tuple(children),
        (),
    )


def policy(**overrides) -> PhasedBeamSearchPolicy:
    options = dict(
        width=2,
        proposals_per_parent=2,
        max_evaluations=8,
        proposal_cap=8,
        seed=71,
        evaluation_seed=19,
        directions=("max",),
        expansion="round_robin",
    )
    options.update(overrides)
    return PhasedBeamSearchPolicy(**options)


def test_duplicate_is_rejected_before_evaluation_and_consumes_its_slot() -> None:
    current = policy()
    state = current.start((ROOT,))
    first = current.ask(state, HISTORY, capacity=1)[0]
    assert first.stage == "propose"
    assert first.candidate is None

    state = current.tell(
        state,
        (outcome(first, proposals=(proposal(ROOT, ROOT_CANDIDATE),)),),
    )

    assert state.proposals == 1
    assert state.duplicates == 1
    assert state.evaluations == 1
    assert state.pending_proposal is None
    second = current.ask(state, HISTORY, capacity=1)[0]
    assert second.stage == "propose"
    assert second.sample.index == "2"
    assert second.identity != first.identity


@pytest.mark.parametrize(
    ("expansion", "expected_parent_values"),
    [
        ("round_robin", (3.0, 2.0, 3.0, 2.0)),
        ("each", (3.0, 3.0, 2.0, 2.0)),
    ],
)
def test_frozen_expansion_order_and_single_phase_work(
    expansion: str, expected_parent_values: tuple[float, ...]
) -> None:
    left = scored(candidate("A", "C", "D"), 3.0)
    right = scored(candidate("A", "C", "E"), 2.0)
    children = iter(
        (
            candidate("A", "D", "E"),
            candidate("A", "E", "D"),
            candidate("C", "A", "E"),
            candidate("C", "D", "A"),
        )
    )
    current = policy(
        width=2,
        proposals_per_parent=2,
        max_evaluations=9,
        proposal_cap=8,
        expansion=expansion,
    )
    state = current.start((left, right))
    observed = []

    for _ in range(4):
        work = current.ask(state, HISTORY, capacity=100)
        assert len(work) == 1
        assert work[0].stage == "propose"
        observed.append(work[0].parents[0].assessment.values[0])
        child = next(children)
        state = current.tell(
            state,
            (outcome(work[0], proposals=(proposal(work[0].parents[0], child),)),),
        )
        evaluation = current.ask(state, HISTORY, capacity=100)
        assert len(evaluation) == 1
        assert evaluation[0].stage == "evaluate"
        assert evaluation[0].candidate == child
        state = current.tell(
            state,
            (outcome(evaluation[0], children=(scored(child, 1.0),)),),
        )

    assert tuple(observed) == expected_parent_values


def test_duplicate_slots_are_not_refilled_and_cap_fails_before_next_attempt() -> None:
    current = policy(
        width=1,
        proposals_per_parent=4,
        max_evaluations=8,
        proposal_cap=2,
    )
    state = current.start((ROOT,))
    for expected in (1, 2):
        work = current.ask(state, HISTORY, capacity=1)[0]
        state = current.tell(
            state,
            (outcome(work, proposals=(proposal(ROOT, ROOT_CANDIDATE),)),),
        )
        assert state.proposals == expected
        assert state.next_slot == expected
        assert state.pending_proposal is None

    assert state.duplicates == 2
    with pytest.raises(RuntimeError, match="proposal attempt cap"):
        current.ask(state, HISTORY, capacity=1)
    assert state.proposals == 2


def test_cold_codec_round_trip_preserves_pending_evaluation_and_work() -> None:
    current = policy()
    state = current.start((ROOT,))
    proposal_work = current.ask(state, HISTORY, capacity=1)[0]
    child = candidate("A", "C", "E")
    state = current.tell(
        state,
        (outcome(proposal_work, proposals=(proposal(ROOT, child),)),),
    )
    expected = current.ask(state, HISTORY, capacity=1)[0]

    store = DataStore()
    reference = store.put(current.snapshot(state), policy().codec)
    restored = store.get(reference, policy().codec)

    assert restored == state
    assert current.ask(restored, HISTORY, capacity=1) == (expected,)


def test_root_counts_toward_evaluation_budget_and_finishes_before_proposal() -> None:
    current = policy(max_evaluations=1, proposal_cap=1)
    state = current.start((ROOT,))

    assert state.root_evaluations == 1
    assert state.evaluations == 1
    assert state.finished
    assert state.phase == "finished"
    assert current.finished(state)
    assert current.ask(state, HISTORY, capacity=1) == ()
    assert current.selected(state) == (ROOT,)


def test_budget_end_mid_generation_keeps_parent_before_equal_first_child() -> None:
    current = policy(
        width=2,
        proposals_per_parent=3,
        max_evaluations=2,
        proposal_cap=3,
    )
    state = current.start((ROOT,))
    proposal_work = current.ask(state, HISTORY, capacity=1)[0]
    child_candidate = candidate("A", "C", "E")
    child = scored(child_candidate, 1.0)
    state = current.tell(
        state,
        (outcome(proposal_work, proposals=(proposal(ROOT, child_candidate),)),),
    )
    evaluation_work = current.ask(state, HISTORY, capacity=1)[0]
    state = current.tell(
        state,
        (outcome(evaluation_work, children=(child,)),),
    )

    assert state.finished
    assert state.evaluations == 2
    assert state.frontier == (ROOT, child)
    # Two proposal slots remain but cannot start because the live budget ended.
    assert state.next_slot == 1


def test_generation_selection_orders_equal_children_by_admission_slot() -> None:
    current = policy(
        width=3,
        proposals_per_parent=2,
        max_evaluations=5,
        proposal_cap=4,
    )
    state = current.start((ROOT,))
    children = (
        scored(candidate("A", "D", "E"), 1.0),
        scored(candidate("C", "A", "E"), 1.0),
    )
    for child in children:
        proposal_work = current.ask(state, HISTORY, capacity=1)[0]
        state = current.tell(
            state,
            (
                outcome(
                    proposal_work,
                    proposals=(proposal(ROOT, child.candidate),),
                ),
            ),
        )
        evaluation_work = current.ask(state, HISTORY, capacity=1)[0]
        state = current.tell(
            state,
            (outcome(evaluation_work, children=(child,)),),
        )

    assert state.generation == 2
    assert state.frontier == (ROOT, children[0], children[1])


def test_replicate_parent_label_uses_global_admission_not_frontier_index() -> None:
    current = policy(
        width=2,
        proposals_per_parent=2,
        max_evaluations=4,
        proposal_cap=20,
    )
    state = current.start((ROOT,))
    children = (
        scored(candidate("A", "C", "E"), 1.0),
        scored(candidate("A", "D", "E"), 2.0),
    )

    for child in children:
        proposal_work = current.ask(state, HISTORY, capacity=1)[0]
        state = current.tell(
            state,
            (
                outcome(
                    proposal_work,
                    proposals=(proposal(proposal_work.parents[0], child.candidate),),
                ),
            ),
        )
        evaluation_work = current.ask(state, HISTORY, capacity=1)[0]
        state = current.tell(
            state,
            (outcome(evaluation_work, children=(child,)),),
        )

    assert state.generation == 2
    assert state.frontier[0] == children[1]
    assert state.admission_ordinals == (
        (ROOT_CANDIDATE.identity.digest, 0),
        (children[0].candidate.identity.digest, 1),
        (children[1].candidate.identity.digest, 2),
    )

    proposal_work = current.ask(state, HISTORY, capacity=1)[0]
    assert proposal_work.parents[0] == children[1]
    assert proposal_work.sample.replicate == (
        "generation_0002_parent_0002_proposal_00"
    )
    next_candidate = candidate("C", "D", "E")
    state = current.tell(
        state,
        (
            outcome(
                proposal_work,
                proposals=(proposal(children[1], next_candidate),),
            ),
        ),
    )
    evaluation_work = current.ask(state, HISTORY, capacity=1)[0]
    assert evaluation_work.parents[0] == children[1]
    assert evaluation_work.sample.replicate == (
        "generation_0002_parent_0002_proposal_00_evaluate"
    )

    store = DataStore()
    restored = store.get(store.put(state, current.codec), current.codec)
    assert restored.admission_ordinals == state.admission_ordinals


def test_changed_candidate_changes_evaluation_work_and_replay_is_stable() -> None:
    current = policy()
    state = current.start((ROOT,))
    proposal_work = current.ask(state, HISTORY, capacity=1)[0]
    first_candidate = candidate("A", "C", "E")
    state = current.tell(
        state,
        (outcome(proposal_work, proposals=(proposal(ROOT, first_candidate),)),),
    )
    first_work = current.ask(state, HISTORY, capacity=1)[0]
    assert current.ask(state, HISTORY, capacity=1) == (first_work,)

    second_candidate = candidate("A", "D", "E")
    changed = replace(
        state,
        pending_proposal=proposal(ROOT, second_candidate),
    )
    second_work = current.ask(changed, HISTORY, capacity=1)[0]

    assert first_work.candidate == first_candidate
    assert second_work.candidate == second_candidate
    assert first_work.identity != second_work.identity


def test_interrupted_evaluation_resumes_same_work_and_does_not_reissue() -> None:
    current = policy()
    state = current.start((ROOT,))
    proposal_work = current.ask(state, HISTORY, capacity=1)[0]
    child_candidate = candidate("A", "C", "E")
    state = current.tell(
        state,
        (outcome(proposal_work, proposals=(proposal(ROOT, child_candidate),)),),
    )

    store = DataStore()
    checkpoint = store.put(state, current.codec)
    pending = current.ask(store.get(checkpoint, current.codec), HISTORY, capacity=1)[0]
    resumed_state = store.get(checkpoint, current.codec)
    assert current.ask(resumed_state, HISTORY, capacity=1) == (pending,)

    resumed_state = current.tell(
        resumed_state,
        (outcome(pending, children=(scored(child_candidate, 2.0),)),),
    )
    next_work = current.ask(resumed_state, HISTORY, capacity=1)[0]
    assert next_work.identity != pending.identity
    assert next_work.stage == "propose"


def test_wrong_stale_and_duplicate_outcome_ids_are_rejected() -> None:
    current = policy()
    state = current.start((ROOT,))
    proposal_work = current.ask(state, HISTORY, capacity=1)[0]
    wrong = TrialOutcome("wrong", "completed", (), (), ())
    with pytest.raises(ValueError, match="outside"):
        current.tell(state, (wrong,))

    child_candidate = candidate("A", "C", "E")
    state = current.tell(
        state,
        (outcome(proposal_work, proposals=(proposal(ROOT, child_candidate),)),),
    )
    evaluation_work = current.ask(state, HISTORY, capacity=1)[0]
    state = current.tell(
        state,
        (outcome(evaluation_work, children=(scored(child_candidate, 2.0),)),),
    )
    with pytest.raises(ValueError, match="more than once"):
        current.tell(
            state,
            (outcome(proposal_work, proposals=(proposal(ROOT, child_candidate),)),),
        )


def test_failed_evaluation_does_not_increment_completed_count_or_retry() -> None:
    current = policy()
    state = current.start((ROOT,))
    proposal_work = current.ask(state, HISTORY, capacity=1)[0]
    child_candidate = candidate("A", "C", "E")
    state = current.tell(
        state,
        (outcome(proposal_work, proposals=(proposal(ROOT, child_candidate),)),),
    )
    evaluation_work = current.ask(state, HISTORY, capacity=1)[0]
    before = state
    state = current.tell(
        state,
        (outcome(evaluation_work, status="failed"),),
    )

    assert state.evaluations == before.evaluations == 1
    next_work = current.ask(state, HISTORY, capacity=1)[0]
    assert next_work.stage == "propose"
    assert next_work.identity != evaluation_work.identity
    with pytest.raises(ValueError, match="more than once"):
        current.tell(state, (outcome(evaluation_work, status="failed"),))


def test_policy_transitions_do_not_mutate_global_random_state() -> None:
    current = policy()
    before = random.getstate()
    state = current.start((ROOT,))
    proposal_work = current.ask(state, HISTORY, capacity=1)[0]
    child_candidate = candidate("A", "C", "E")
    state = current.tell(
        state,
        (outcome(proposal_work, proposals=(proposal(ROOT, child_candidate),)),),
    )
    evaluation_work = current.ask(state, HISTORY, capacity=1)[0]
    state = current.tell(
        state,
        (outcome(evaluation_work, children=(scored(child_candidate, 2.0),)),),
    )
    current.finished(state)
    current.selected(state)

    assert random.getstate() == before


def test_callable_seed_projection_and_capturing_rank_require_identities() -> None:
    with pytest.raises(ValueError, match="seed_identity"):
        policy(proposal_seed=lambda index: index)
    with pytest.raises(ValueError, match="projection_identity"):
        policy(projection=lambda value: value.assignments)

    offset = 0.5

    def rank(value: Scored):
        return (value.assessment.values[0] + offset,)

    with pytest.raises(ValueError, match="rank_identity"):
        policy(rank=rank)

    configured = policy(
        proposal_seed=lambda index: 1000 + index,
        seed_identity="phased-test-seed/v1",
        projection=lambda value: value.assignments,
        projection_identity="phased-test-projection/v1",
        rank=rank,
        rank_identity="phased-test-rank/v1",
    )
    state = configured.start((ROOT,))
    work = configured.ask(state, HISTORY, capacity=1)[0]
    assert work.sample.seed == 1001
