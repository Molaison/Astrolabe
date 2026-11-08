from __future__ import annotations

import random

import pytest

from astrolabe.design.store import DataStore
from astrolabe.design.successor_search import (
    ACCEPTED_SUCCESSOR_STATE_CODEC,
    AcceptedSuccessorPolicy,
)
from astrolabe.design.types import (
    Assessment,
    Candidate,
    Continuation,
    Evidence,
    History,
    InputManifest,
    Scored,
    TrialOutcome,
    key_for,
)


INPUTS = InputManifest(key_for(("successor-input/v1",)), ())
HISTORY = History(key_for(("empty-history/v1",)), ())


def _scored(
    label: str,
    value: float,
    *,
    gates: tuple[tuple[str, bool], ...] = (),
) -> Scored:
    system = key_for(("successor-system/v1",))
    candidate = Candidate(
        identity=key_for(("successor-candidate/v1", label)),
        system=system,
        assignments=(),
    )
    evidence = Evidence(
        identity=key_for(("successor-evidence/v1", candidate.identity)),
        evaluated_candidate=candidate.identity,
        artifacts=(),
        measurements=(),
        inputs=INPUTS,
    )
    assessment = Assessment(
        evidence=evidence.identity,
        objective_revision=key_for(("successor-objective/v1",)),
        history_revision=HISTORY.revision,
        values=(float(value),),
        gates=gates,
        inputs=INPUTS,
    )
    return Scored(candidate, evidence, assessment)


def _outcome(
    work: str,
    continuation: Continuation | None,
    *,
    status: str = "completed",
) -> TrialOutcome:
    return TrialOutcome(
        work=work,
        status=status,
        proposals=(),
        children=(),
        attempts=(),
        continuation=continuation,
    )


def test_ask_is_singleton_stable_and_uses_parent_step_sample_and_seed():
    policy = AcceptedSuccessorPolicy(rounds=2, seed=17)
    root = _scored("root", 10.0)
    state = policy.start((root,))

    first = policy.ask(state, HISTORY, capacity=1)
    second = policy.ask(state, HISTORY, capacity=99)

    assert first == second
    assert len(first) == 1
    work = first[0]
    assert work.stage == "advance"
    assert work.parents == (root,)
    assert work.candidate is None
    assert work.sample.protocol == policy.identity
    assert work.sample.replicate == "1"
    assert work.sample.seed == 17
    assert work.sample.index == "1"
    assert state.used == 0


def test_accepted_lower_score_and_failed_gate_are_kept_verbatim():
    policy = AcceptedSuccessorPolicy(rounds=2, seed=23)
    root = _scored("root", 10.0)
    state = policy.start((root,))
    work = policy.ask(state, HISTORY, capacity=1)[0]
    successor = _scored(
        "accepted-worse",
        -100.0,
        gates=(("objective_gate", False),),
    )

    next_state = policy.tell(
        state,
        (
            _outcome(
                work.identity,
                Continuation(successor),
            ),
        ),
    )

    assert next_state.frontier == (successor,)
    assert policy.selected(next_state) == (successor,)
    assert next_state.used == 1
    next_work = policy.ask(next_state, HISTORY, capacity=1)[0]
    assert next_work.parents == (successor,)
    assert next_work.sample.index == "2"
    assert next_work.sample.seed == 23


def test_rejection_continues_from_the_same_predecessor_as_a_new_step():
    policy = AcceptedSuccessorPolicy(rounds=2, seed=29)
    root = _scored("root", 4.0)
    state = policy.start((root,))
    work = policy.ask(state, HISTORY, capacity=1)[0]

    state = policy.tell(
        state,
        (_outcome(work.identity, Continuation(root)),),
    )
    next_work = policy.ask(state, HISTORY, capacity=1)[0]

    assert state.frontier == (root,)
    assert next_work.parents == (root,)
    assert next_work.sample.index == "2"
    assert next_work.identity != work.identity


@pytest.mark.parametrize("failure", ("missing", "failed", "out_of_order", "multiple"))
def test_invalid_outcome_leaves_predecessor_and_expected_work_unchanged(failure):
    policy = AcceptedSuccessorPolicy(rounds=3, seed=5)
    root = _scored("root", 1.0)
    state = policy.start((root,))
    work = policy.ask(state, HISTORY, capacity=1)[0]
    successor = Continuation(_scored("successor", 2.0))
    outcomes = (_outcome(work.identity, successor),)
    if failure == "missing":
        outcomes = (_outcome(work.identity, None),)
    elif failure == "failed":
        outcomes = (_outcome(work.identity, successor, status="failed"),)
    elif failure == "out_of_order":
        outcomes = (_outcome("some-other-work", successor),)
    elif failure == "multiple":
        outcomes = (
            _outcome(work.identity, successor),
            _outcome(work.identity, successor),
        )

    with pytest.raises((TypeError, ValueError)):
        policy.tell(state, outcomes)

    assert state.frontier == (root,)
    assert state.used == 0
    assert policy.ask(state, HISTORY, capacity=1) == (work,)
    with pytest.raises(ValueError, match="out of order|already consumed"):
        policy.tell(state, (_outcome("different", successor),))


def test_stop_reason_freezes_selected_successor_and_rejects_more_work():
    policy = AcceptedSuccessorPolicy(rounds=10, seed=2)
    state = policy.start((_scored("root", 3.0),))
    work = policy.ask(state, HISTORY, capacity=1)[0]
    successor = _scored("stopped", 1.0)
    state = policy.tell(
        state,
        (
            _outcome(
                work.identity,
                Continuation(successor, stop_reason="owner converged"),
            ),
        ),
    )

    assert state.stop_reason == "owner converged"
    assert policy.finished(state)
    assert policy.ask(state, HISTORY, capacity=1) == ()
    assert policy.selected(state) == (successor,)
    with pytest.raises(ValueError, match="after the policy finished"):
        policy.tell(
            state,
            (_outcome(work.identity, Continuation(successor)),),
        )


def test_round_exhaustion_allows_exactly_rounds_after_root():
    policy = AcceptedSuccessorPolicy(rounds=2, seed=7)
    state = policy.start((_scored("root", 0.0),))
    successors = []
    for step in (1, 2):
        work = policy.ask(state, HISTORY, capacity=1)[0]
        assert work.sample.index == str(step)
        successor = _scored(f"successor-{step}", float(step))
        successors.append(successor)
        state = policy.tell(
            state,
            (_outcome(work.identity, Continuation(successor)),),
        )

    assert state.used == 2
    assert policy.finished(state)
    assert policy.ask(state, HISTORY, capacity=100) == ()
    assert policy.selected(state) == (successors[-1],)


def test_duplicate_work_is_rejected_after_it_has_advanced():
    policy = AcceptedSuccessorPolicy(rounds=3, seed=11)
    state = policy.start((_scored("root", 0.0),))
    first = policy.ask(state, HISTORY, capacity=1)[0]
    state = policy.tell(
        state,
        (_outcome(first.identity, Continuation(_scored("one", 1.0))),),
    )
    with pytest.raises(ValueError, match="out of order|already consumed"):
        policy.tell(
            state,
            (_outcome(first.identity, Continuation(_scored("again", 2.0))),),
        )


def test_transitions_do_not_consume_rng():
    policy = AcceptedSuccessorPolicy(rounds=1, seed=13)
    random.seed(999)
    before = random.getstate()
    state = policy.start((_scored("root", 0.0),))
    work = policy.ask(state, HISTORY, capacity=1)[0]
    state = policy.tell(
        state,
        (_outcome(work.identity, Continuation(_scored("next", 1.0))),),
    )
    policy.finished(state)
    policy.selected(state)
    assert random.getstate() == before


def test_identity_depends_on_source_rounds_and_seed():
    first = AcceptedSuccessorPolicy(rounds=1, seed=4)
    same = AcceptedSuccessorPolicy(rounds=1, seed=4)
    more_rounds = AcceptedSuccessorPolicy(rounds=2, seed=4)
    other_seed = AcceptedSuccessorPolicy(rounds=1, seed=5)

    assert first.identity == same.identity
    assert first.identity != more_rounds.identity
    assert first.identity != other_seed.identity


def test_immutable_checkpoint_reopens_with_same_next_work(tmp_path):
    policy = AcceptedSuccessorPolicy(rounds=2, seed=19)
    root = _scored("root", 0.0)
    state = policy.start((root,))
    work = policy.ask(state, HISTORY, capacity=1)[0]
    successor = _scored("persisted", 0.5)
    state = policy.tell(
        state,
        (_outcome(work.identity, Continuation(successor)),),
    )
    expected_work = policy.ask(state, HISTORY, capacity=1)[0]

    store = DataStore(tmp_path / "store")
    blob = store.put(state, ACCEPTED_SUCCESSOR_STATE_CODEC)
    store.persist(
        (
            blob,
            *ACCEPTED_SUCCESSOR_STATE_CODEC.references(state),
        )
    )

    reopened = DataStore(tmp_path / "store")
    restored = reopened.get(blob, ACCEPTED_SUCCESSOR_STATE_CODEC)
    restored_policy = AcceptedSuccessorPolicy(rounds=2, seed=19)

    assert restored == state
    assert restored.frontier == (successor,)
    assert restored_policy.ask(restored, HISTORY, capacity=1) == (expected_work,)
