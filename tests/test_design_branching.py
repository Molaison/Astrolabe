from __future__ import annotations

from concurrent.futures import Future
from dataclasses import replace
import random
import threading

import pytest

from astrolabe.design.branch_execution import BranchExecutor
from astrolabe.design.branch_types import (
    BranchEvent,
    BranchJob,
    BranchMember,
    BranchProposal,
    BranchTerminal,
    BranchingState,
    branch_codec,
)
from astrolabe.design.branching import BranchingPolicy
from astrolabe.design.coordinator import drive
from astrolabe.design.types import Blob, Key, key_for


def _blob(name: str) -> Blob:
    return Blob(Key(key_for(name).digest), "test/branching/v1")


def _member(
    identifier: int,
    *,
    score: float | None = None,
    key: str | None = None,
    context: Key | None = None,
) -> BranchMember:
    member_key = key or f"member-{identifier}"
    return BranchMember(
        id=identifier,
        key=member_key,
        value=_blob(member_key),
        score=float(identifier if score is None else score),
        comparison_context=context,
    )


def _proposal_result(
    job: BranchJob,
    key: str,
    *,
    valid: bool = True,
) -> BranchProposal:
    return BranchProposal(
        id=job.id,
        parent_id=job.parent.id,
        key=key,
        value=_blob(key),
        valid=valid,
        reasons=() if valid else ("invalid-test-proposal",),
    )


def _proposal_terminal(
    job: BranchJob,
    key: str,
    *,
    valid: bool = True,
) -> BranchTerminal:
    return BranchTerminal(
        job=job,
        status="completed",
        value=_proposal_result(job, key, valid=valid),
    )


def _evaluation_terminal(
    job: BranchJob,
    *,
    score: float | None = None,
    context: Key | None = None,
) -> BranchTerminal:
    key = job.proposal.key if job.proposal is not None else "root"
    return BranchTerminal(
        job=job,
        status="completed",
        value=_member(
            job.id,
            score=score,
            key=key,
            context=context,
        ),
    )


def _root_job(policy: BranchingPolicy) -> BranchJob:
    return BranchJob(
        phase="evaluate",
        id=0,
        generation=0,
        seed=policy.evaluation_seed,
    )


def _started(policy: BranchingPolicy) -> BranchingState:
    root = _root_job(policy)
    state = policy.start(root)
    action = policy.ask(state)
    assert action.jobs == (root,)
    state = policy.admit(state, action)
    return policy.tell(state, BranchEvent((_evaluation_terminal(root),)))


def _advance_empty(
    policy: BranchingPolicy,
    state: BranchingState,
) -> BranchingState:
    action = policy.ask(state)
    assert action.jobs == ()
    state = policy.admit(state, action)
    assert policy.tell(state, BranchEvent(())) == state
    return state


def _ready_generation(
    policy: BranchingPolicy,
    *,
    parents: tuple[BranchMember, ...] | None = None,
) -> BranchingState:
    state = _advance_empty(policy, _started(policy))
    if parents is not None:
        state = replace(
            state,
            generation=1,
            parents=parents,
            frontier=parents,
        )
    return state


def _policy(**overrides) -> BranchingPolicy:
    options = dict(
        width=2,
        proposals_per_parent=2,
        max_evaluations=12,
        proposal_cap=20,
        seed=7,
        evaluation_seed=13,
        proposal_batch_size=2,
        evaluation_width=2,
    )
    options.update(overrides)
    return BranchingPolicy(**options)


def test_proposal_batch_commits_out_of_order_only_at_complete_barrier() -> None:
    policy = _policy(quota_mode="attempts")
    state = _ready_generation(policy)

    action = policy.ask(state)
    assert tuple(job.id for job in action.jobs) == (1, 2)
    state = policy.admit(state, action)
    assert state.batch == action.jobs

    state = policy.tell(
        state,
        BranchEvent((_proposal_terminal(action.jobs[1], "child-2"),)),
    )
    assert state.batch == action.jobs
    assert state.queued == ()
    assert state.batch_results[0].job.id == 2

    state = policy.tell(
        state,
        BranchEvent((_proposal_terminal(action.jobs[0], "child-1"),)),
    )
    assert state.batch == ()
    assert tuple(terminal.job.id for terminal in state.batch_results) == (1, 2)
    assert tuple(job.proposal.key for job in state.queued) == (
        "child-1",
        "child-2",
    )
    assert tuple(job.id for job in state.queued) == (1, 2)


def test_admitted_quota_refills_invalid_and_duplicate_attempts() -> None:
    policy = _policy(quota_mode="admitted")
    state = _ready_generation(policy)

    first = policy.ask(state)
    state = policy.tell(
        policy.admit(state, first),
        BranchEvent(
            (
                _proposal_terminal(first.jobs[1], "root"),
                _proposal_terminal(first.jobs[0], "invalid", valid=False),
            )
        ),
    )

    assert state.counts == ()
    assert state.rejected == 1
    assert state.duplicates == 1
    assert state.queued == ()
    assert state.proposals == 2

    retry = policy.ask(state)
    assert tuple(job.id for job in retry.jobs) == (3, 4)
    assert tuple(job.slot for job in retry.jobs) == (0, 1)
    state = policy.tell(
        policy.admit(state, retry),
        BranchEvent(
            (
                _proposal_terminal(retry.jobs[0], "accepted-1"),
                _proposal_terminal(retry.jobs[1], "accepted-2"),
            )
        ),
    )
    assert state.counts == ((0, 2),)
    assert tuple(job.proposal.key for job in state.queued) == (
        "accepted-1",
        "accepted-2",
    )


def test_attempt_quota_consumes_invalid_and_duplicate_slots_without_refill() -> None:
    policy = _policy(quota_mode="attempts")
    state = _ready_generation(policy)

    action = policy.ask(state)
    state = policy.tell(
        policy.admit(state, action),
        BranchEvent(
            (
                _proposal_terminal(action.jobs[0], "invalid", valid=False),
                _proposal_terminal(action.jobs[1], "root"),
            )
        ),
    )

    assert state.counts == ((0, 2),)
    assert state.proposals == 2
    assert state.queued == ()
    assert state.duplicates == 1
    assert state.rejected == 1

    action = policy.ask(state)
    assert action.jobs == ()
    advanced = policy.admit(state, action)
    assert advanced.generation == 2
    next_attempt = policy.ask(advanced)
    assert tuple(job.id for job in next_attempt.jobs) == (3, 4)


@pytest.mark.parametrize("pipeline", [False, True])
def test_nonpipeline_holds_evaluation_until_proposal_phase_while_pipeline_overlaps(
    pipeline: bool,
) -> None:
    policy = _policy(
        proposals_per_parent=2,
        proposal_batch_size=1,
        evaluation_width=1,
        quota_mode="attempts",
        pipeline=pipeline,
        cap_error=False,
    )
    state = _ready_generation(policy)

    first = policy.ask(state)
    state = policy.tell(
        policy.admit(state, first),
        BranchEvent((_proposal_terminal(first.jobs[0], "child-1"),)),
    )
    assert len(state.queued) == 1

    next_action = policy.ask(state)
    if pipeline:
        assert tuple(job.phase for job in next_action.jobs) == ("evaluate",)
        state = policy.admit(state, next_action)
        proposal_action = policy.ask(state)
        assert tuple(job.phase for job in proposal_action.jobs) == ("propose",)
        assert state.pending
    else:
        assert tuple(job.phase for job in next_action.jobs) == ("propose",)
        assert state.pending == ()


def test_immediate_evaluation_serializes_propose_evaluate_and_next_proposal() -> None:
    with pytest.raises(ValueError, match="requires proposal_batch_size=1"):
        _policy(
            immediate_evaluation=True,
            proposal_batch_size=2,
            evaluation_width=1,
            pipeline=False,
        )
    with pytest.raises(ValueError, match="separate execution modes"):
        _policy(
            immediate_evaluation=True,
            proposal_batch_size=1,
            evaluation_width=1,
            pipeline=True,
        )

    policy = _policy(
        proposals_per_parent=2,
        proposal_batch_size=1,
        evaluation_width=1,
        quota_mode="attempts",
        immediate_evaluation=True,
    )
    state = _ready_generation(policy)

    proposal = policy.ask(state)
    assert proposal.jobs[0].id == 1
    state = policy.tell(
        policy.admit(state, proposal),
        BranchEvent((_proposal_terminal(proposal.jobs[0], "child-1"),)),
    )

    evaluation = policy.ask(state)
    assert evaluation.jobs[0].phase == "evaluate"
    state = policy.tell(
        policy.admit(state, evaluation),
        BranchEvent((_evaluation_terminal(evaluation.jobs[0]),)),
    )

    next_proposal = policy.ask(state)
    assert next_proposal.jobs[0].phase == "propose"
    assert next_proposal.jobs[0].id == 2


@pytest.mark.parametrize("expansion", ["round_robin", "balanced"])
def test_expansion_slots_and_stable_ids_preserve_original_parent_order(
    expansion: str,
) -> None:
    first = _member(0)
    second = _member(10, key="second-parent")
    policy = _policy(
        proposals_per_parent=2,
        proposal_batch_size=4,
        quota_mode="attempts",
        expansion=expansion,
    )
    state = _ready_generation(policy, parents=(first, second))

    action = policy.ask(state)
    assert tuple((job.id, job.parent.id, job.slot) for job in action.jobs) == (
        (1, 0, 0),
        (2, 10, 0),
        (3, 0, 1),
        (4, 10, 1),
    )
    assert (
        action.jobs[0].seed
        == (policy.seed * 100003 + 1) % 2147483647
    )


def test_global_dedup_precedes_evaluation_and_seen_survives_generation() -> None:
    policy = _policy(
        proposals_per_parent=2,
        proposal_batch_size=2,
        quota_mode="attempts",
    )
    state = _ready_generation(policy)

    proposals = policy.ask(state)
    state = policy.tell(
        policy.admit(state, proposals),
        BranchEvent(
            (
                _proposal_terminal(proposals.jobs[0], "shared-child"),
                _proposal_terminal(proposals.jobs[1], "shared-child"),
            )
        ),
    )
    assert state.duplicates == 1
    assert len(state.queued) == 1
    assert state.seen == ("root", "shared-child")


def test_out_of_order_evaluation_terminals_store_records_in_stable_id_order() -> None:
    policy = _policy(quota_mode="attempts")
    state = _ready_generation(policy)
    proposals = policy.ask(state)
    state = policy.tell(
        policy.admit(state, proposals),
        BranchEvent(
            (
                _proposal_terminal(proposals.jobs[0], "child-1"),
                _proposal_terminal(proposals.jobs[1], "child-2"),
            )
        ),
    )
    dispatch = policy.ask(state)
    state = policy.tell(
        policy.admit(state, dispatch),
        BranchEvent(
            (
                _evaluation_terminal(dispatch.jobs[1]),
                _evaluation_terminal(dispatch.jobs[0]),
            )
        ),
    )

    assert tuple(member.id for member in state.records) == (0, 1, 2)
    assert tuple(member.id for member in state.frontier) == (0, 1, 2)


def test_root_consumes_budget_and_finished_selection_returns_full_state() -> None:
    policy = _policy(max_evaluations=1)
    state = _started(policy)
    assert tuple(member.key for member in state.frontier) == ("root",)

    action = policy.ask(state)
    assert action.jobs == ()
    state = policy.admit(state, action)
    state = policy.tell(state, BranchEvent(()))

    assert policy.finished(state)
    assert state.stop_reason == "evaluation_budget"
    assert state.proposals == 0
    assert policy.selected(state) is state
    assert policy.selected(state).records == (_member(0, key="root"),)


@pytest.mark.parametrize("quota_mode", ["admitted", "attempts"])
def test_evaluation_room_clips_batch_and_refills_one_attempt_at_a_time(
    quota_mode: str,
) -> None:
    policy = _policy(
        max_evaluations=2,
        proposals_per_parent=3,
        proposal_batch_size=3,
        quota_mode=quota_mode,
    )
    state = _ready_generation(policy)

    first = policy.ask(state)
    assert tuple(job.id for job in first.jobs) == (1,)
    assert first.jobs[0].slot == 0
    state = policy.tell(
        policy.admit(state, first),
        BranchEvent(
            (_proposal_terminal(first.jobs[0], "invalid", valid=False),)
        ),
    )
    assert state.next_evaluation == 1
    assert state.counts == (((0, 1),) if quota_mode == "attempts" else ())

    retry = policy.ask(state)
    assert tuple(job.id for job in retry.jobs) == (2,)
    assert retry.jobs[0].slot == (1 if quota_mode == "attempts" else 0)
    state = policy.tell(
        policy.admit(state, retry),
        BranchEvent((_proposal_terminal(retry.jobs[0], "unique-child"),)),
    )
    assert state.next_evaluation == 2
    assert state.proposals == 2
    assert len(state.queued) == 1

    dispatch = policy.ask(state)
    assert tuple(job.id for job in dispatch.jobs) == (1,)
    state = policy.tell(
        policy.admit(state, dispatch),
        BranchEvent((_evaluation_terminal(dispatch.jobs[0]),)),
    )
    state = policy.admit(state, policy.ask(state))

    assert state.finished
    assert state.stop_reason == "evaluation_budget"
    assert state.proposals == 2


def test_negative_thresholds_do_not_apply_but_zero_generations_is_root_only() -> None:
    policy = _policy(max_generations=0)
    state = _advance_empty(policy, _started(policy))
    assert state.finished
    assert state.stop_reason == "generation_limit"
    with pytest.raises(ValueError, match="non-negative"):
        _policy(max_generations=-1)


def test_mixed_comparison_contexts_require_reassessment() -> None:
    policy = _policy()
    first = _member(1, context=Key("context-a"))
    second = _member(2, context=Key("context-b"))
    state = BranchingState(
        generation=1,
        next_evaluation=3,
        records=(first, second),
        frontier=(first, second),
        parents=(first, second),
        stop_reason="proposal_cap",
    )

    with pytest.raises(ValueError, match="comparison_context mismatch"):
        policy.ask(state)


def test_refusal_drops_queued_work_and_drains_already_pending_calls() -> None:
    policy = _policy(
        proposals_per_parent=3,
        proposal_batch_size=1,
        evaluation_width=2,
        quota_mode="attempts",
        pipeline=True,
    )
    state = _ready_generation(policy)

    first = policy.ask(state)
    state = policy.tell(
        policy.admit(state, first),
        BranchEvent((_proposal_terminal(first.jobs[0], "child-1"),)),
    )
    dispatch_one = policy.ask(state)
    state = policy.admit(state, dispatch_one)

    second = policy.ask(state)
    state = policy.tell(
        policy.admit(state, second),
        BranchEvent((_proposal_terminal(second.jobs[0], "child-2"),)),
    )
    dispatch_two = policy.ask(state)
    state = policy.admit(state, dispatch_two)

    third = policy.ask(state)
    state = policy.tell(
        policy.admit(state, third),
        BranchEvent((_proposal_terminal(third.jobs[0], "child-3"),)),
    )
    assert len(state.queued) == 1
    queued_evaluation = state.queued[0]
    first_evaluation = dispatch_one.jobs[0]
    second_evaluation = dispatch_two.jobs[0]

    state = policy.tell(
        state,
        BranchEvent(
            (BranchTerminal(first_evaluation, "refused"),)
        ),
    )
    assert state.stop_reason == "native_call_budget"
    assert state.refused == (first_evaluation,)
    assert state.pending == (second_evaluation,)
    assert state.queued == ()
    assert queued_evaluation not in state.queued

    drain = policy.ask(state)
    assert drain.jobs == ()
    state = policy.tell(
        policy.admit(state, drain),
        BranchEvent((_evaluation_terminal(second_evaluation),)),
    )
    state = policy.admit(state, policy.ask(state))
    assert state.finished
    assert state.stop_reason == "native_call_budget"
    assert tuple(member.key for member in state.frontier) == (
        "child-2",
        "root",
    )


def test_refused_proposal_is_typed_and_stops_without_dispatching_child() -> None:
    policy = _policy(quota_mode="attempts")
    state = _ready_generation(policy)
    proposals = policy.ask(state)
    state = policy.tell(
        policy.admit(state, proposals),
        BranchEvent(
            (
                _proposal_terminal(proposals.jobs[0], "child-1"),
                BranchTerminal(proposals.jobs[1], "refused"),
            )
        ),
    )
    assert state.refused == (proposals.jobs[1],)
    assert state.queued == ()
    assert state.stop_reason == "native_call_budget"

    state = policy.admit(state, policy.ask(state))
    assert state.finished
    assert state.stop_reason == "native_call_budget"
    assert tuple(member.key for member in state.frontier) == ("root",)


def test_proposal_cap_stops_partial_generation_and_cap_error_is_explicit() -> None:
    policy = _policy(
        proposals_per_parent=5,
        proposal_cap=2,
        proposal_batch_size=2,
        quota_mode="attempts",
    )
    state = _ready_generation(policy)
    proposals = policy.ask(state)
    state = policy.tell(
        policy.admit(state, proposals),
        BranchEvent(
            (
                _proposal_terminal(proposals.jobs[0], "child-1"),
                _proposal_terminal(proposals.jobs[1], "child-2"),
            )
        ),
    )
    dispatch = policy.ask(state)
    state = policy.tell(
        policy.admit(state, dispatch),
        BranchEvent(
            (
                _evaluation_terminal(dispatch.jobs[0]),
                _evaluation_terminal(dispatch.jobs[1]),
            )
        ),
    )
    state = policy.admit(state, policy.ask(state))
    assert state.finished
    assert state.stop_reason == "proposal_cap"
    assert tuple(member.id for member in state.frontier) == (2, 1)

    failing = _policy(
        proposals_per_parent=5,
        proposal_cap=2,
        proposal_batch_size=2,
        quota_mode="attempts",
        cap_error=True,
    )
    state = _ready_generation(failing)
    proposals = failing.ask(state)
    state = failing.admit(state, proposals)
    state = failing.tell(
        state,
        BranchEvent(
            (
                _proposal_terminal(proposals.jobs[0], "child-1", valid=False),
                _proposal_terminal(proposals.jobs[1], "child-2", valid=False),
            )
        ),
    )
    with pytest.raises(ValueError, match="proposal cap 2"):
        failing.ask(state)


def test_proposal_request_budget_counts_duplicate_attempts_before_scientific_cap() -> None:
    policy = _policy(
        proposals_per_parent=4,
        proposal_cap=20,
        proposal_request_budget=2,
        proposal_batch_size=2,
        quota_mode="attempts",
    )
    state = _ready_generation(policy)

    proposals = policy.ask(state)
    state = policy.tell(
        policy.admit(state, proposals),
        BranchEvent(
            (
                _proposal_terminal(proposals.jobs[0], "root"),
                _proposal_terminal(proposals.jobs[1], "root"),
            )
        ),
    )

    final = policy.admit(state, policy.ask(state))
    assert final.finished
    assert final.stop_reason == "native_call_budget"
    assert final.proposals == 2
    assert final.duplicates == 2
    assert final.queued == ()
    assert tuple(member.key for member in final.frontier) == ("root",)


def test_one_request_native_budget_stops_after_a_duplicate() -> None:
    policy = _policy(
        proposals_per_parent=4,
        proposal_cap=20,
        proposal_request_budget=1,
        proposal_batch_size=1,
        evaluation_width=1,
        quota_mode="attempts",
        immediate_evaluation=True,
    )
    state = _ready_generation(policy)

    dispatch = policy.ask(state)
    state = policy.tell(
        policy.admit(state, dispatch),
        BranchEvent((_proposal_terminal(dispatch.jobs[0], "root"),)),
    )
    final = policy.admit(state, policy.ask(state))

    assert final.finished
    assert final.stop_reason == "native_call_budget"
    assert final.proposals == 1
    assert final.duplicates == 1
    assert tuple(member.key for member in final.frontier) == ("root",)


def test_default_selection_uses_score_descending_and_stable_id_ties() -> None:
    policy = _policy(width=3)
    state = BranchingState(
        generation=1,
        next_evaluation=4,
        records=(
            _member(3, score=2),
            _member(2, score=1),
            _member(1, score=1),
        ),
        frontier=(
            _member(3, score=2),
            _member(2, score=1),
            _member(1, score=1),
        ),
        parents=(_member(1, score=1),),
        stop_reason="proposal_cap",
    )

    action = policy.ask(state)
    final = policy.admit(state, action)
    assert tuple(member.id for member in final.frontier) == (3, 1, 2)


def test_archive_selection_selects_from_all_records_and_keeps_archive() -> None:
    policy = _policy(width=2, archive_selection=True)
    old = _member(1, score=0)
    high = _member(2, score=5)
    low = _member(3, score=1)
    state = BranchingState(
        generation=1,
        next_evaluation=4,
        records=(old, high, low),
        frontier=(old,),
        parents=(old,),
        stop_reason="proposal_cap",
    )

    action = policy.ask(state)
    final = policy.admit(state, action)
    assert final.records == (old, high, low)
    assert final.frontier == (high, low)


def test_policy_runs_through_the_shared_coordinator_drive_loop() -> None:
    policy = _policy(
        width=1,
        proposals_per_parent=1,
        max_evaluations=5,
        proposal_cap=5,
        proposal_batch_size=1,
        evaluation_width=1,
        quota_mode="attempts",
        max_generations=1,
    )

    class Trial:
        request_type = policy.request_type
        result_type = policy.result_type

        def perform(self, action):
            terminals = []
            for job in action.jobs:
                if job.phase == "propose":
                    terminals.append(
                        _proposal_terminal(job, f"generated-{job.id}")
                    )
                else:
                    terminals.append(
                        _evaluation_terminal(job, score=float(job.id))
                    )
            return BranchEvent(tuple(terminals))

    root = _root_job(policy)
    final = drive(policy, Trial(), policy.start(root))

    assert isinstance(final, BranchingState)
    assert final.finished
    assert final.stop_reason == "generation_limit"
    assert final.proposals == 1
    assert final.records == (
        _member(0, key="root"),
        _member(1, key="generated-1", score=1.0),
    )


def test_pipeline_evaluation_can_unblock_a_later_proposal_batch() -> None:
    policy = _policy(
        width=2,
        proposals_per_parent=4,
        max_evaluations=6,
        proposal_cap=4,
        proposal_batch_size=3,
        evaluation_width=3,
        quota_mode="attempts",
        pipeline=True,
        max_generations=1,
    )
    initial = _ready_generation(policy)

    def terminal_for(job: BranchJob) -> BranchTerminal:
        if job.phase == "propose":
            return _proposal_terminal(job, f"generated-{job.id}")
        return _evaluation_terminal(job, score=float(job.id))

    class Pool:
        def __init__(self) -> None:
            self.calls: list[BranchJob] = []
            self.worker = terminal_for
            self.first_evaluation: Future[object] = Future()
            self.first_evaluation_job: BranchJob | None = None

        def submit(self, worker, job):
            self.calls.append(job)
            if job.phase == "evaluate" and job.id == 1:
                self.first_evaluation_job = job
                return self.first_evaluation
            if job.phase == "propose" and job.id == 4:
                assert self.first_evaluation_job is not None
                self.first_evaluation.set_result(
                    worker(self.first_evaluation_job)
                )
            future = Future()
            future.set_result(worker(job))
            return future

    pool = Pool()
    executor = BranchExecutor(pool, worker=terminal_for)
    outcome: dict[str, object] = {}

    def run_drive() -> None:
        try:
            outcome["value"] = drive(policy, executor, initial)
        except BaseException as error:
            outcome["error"] = error

    thread = threading.Thread(target=run_drive, daemon=True)
    thread.start()
    thread.join(3)
    if thread.is_alive():
        pytest.fail("pipeline deadlocked waiting across proposal/evaluation lanes")
    if "error" in outcome:
        raise outcome["error"]  # type: ignore[misc]

    final = outcome["value"]
    assert isinstance(final, BranchingState)
    assert final.finished
    assert final.stop_reason == "generation_limit"
    assert tuple(member.id for member in final.records) == (0, 1, 2, 3, 4)
    assert final.pending == ()
    assert pool.calls.index(pool.first_evaluation_job) < next(
        index
        for index, job in enumerate(pool.calls)
        if job.phase == "propose" and job.id == 4
    )


def test_identity_tracks_all_options_and_callback_identities() -> None:
    first = _policy()
    same = _policy()
    changed = _policy(width=3)
    selected_source = _policy(
        selection=_reverse_selection,
        selection_identity="reverse-selection/v1",
    )
    selected_source_same = _policy(
        selection=_reverse_selection,
        selection_identity="reverse-selection/v1",
    )
    selected_source_changed = _policy(
        selection=_reverse_selection,
        selection_identity="reverse-selection/v2",
    )
    inspected_source = _policy(selection=_reverse_selection)

    assert first.identity == same.identity
    assert first.identity != changed.identity
    assert selected_source.identity == selected_source_same.identity
    assert selected_source.identity != selected_source_changed.identity
    assert inspected_source.selection_identity != (
        "reverse-selection/v1"
    )

    captured = 2
    with pytest.raises(ValueError, match="selection_identity is required"):
        _policy(
            selection=lambda members, width: members[:captured],
        )

    with pytest.raises(TypeError, match="must return a tuple"):
        bad = _policy(
            selection=lambda members, width: list(members[:width]),
        )
        leaves = (_member(1),)
        bad.ask(
            BranchingState(
                generation=1,
                next_evaluation=2,
                records=leaves,
                frontier=leaves,
                parents=leaves,
                stop_reason="proposal_cap",
            )
        )


def _reverse_selection(
    members: tuple[BranchMember, ...],
    width: int,
) -> tuple[BranchMember, ...]:
    return tuple(reversed(members))[:width]


def test_tell_is_pure_and_state_survives_cold_codec_round_trip() -> None:
    policy = _policy(quota_mode="attempts")
    state = _ready_generation(policy)
    proposal = policy.ask(state)
    checkpoint = branch_codec(BranchingState)
    restored = checkpoint.decode(checkpoint.encode(policy.admit(state, proposal)))
    assert restored == policy.admit(state, proposal)

    before = random.getstate()
    result = policy.tell(
        restored,
        BranchEvent((_proposal_terminal(proposal.jobs[0], "cold-child"),)),
    )
    after = random.getstate()
    assert before == after
    assert result.batch == restored.batch
