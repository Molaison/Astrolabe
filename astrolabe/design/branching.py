"""Deterministic proposal/evaluation branching control.

The policy owns admission, parent quotas, stable seeds, deduplication and beam
selection. A trial supplies proposal and evaluation outcomes; it never owns
search control.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
import inspect
import math
from typing import Literal

from .branch_types import (
    BranchAction,
    BranchEvent,
    BranchJob,
    BranchMember,
    BranchProposal,
    BranchTerminal,
    BranchingState,
)
from .types import Blob, Key, key_for


QuotaMode = Literal["admitted", "attempts"]
Expansion = Literal["balanced", "round_robin"]
ProposalSeed = Callable[[int, int], int]
Selection = Callable[
    [tuple[BranchMember, ...], int], tuple[BranchMember, ...]
]

_CONTROL_SCHEMA = "astrolabe.design.branching.Control/v1"
_DEFAULT_SELECTION_IDENTITY = "score-desc-member-id/v1"
_DEFAULT_PROPOSAL_SEED_IDENTITY = "linear-proposal-seed/v1"


def _positive_int(name: str, value: object) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(name: str, value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _seed(name: str, value: object) -> int:
    if type(value) is not int or not 0 <= value <= 2**32 - 1:
        raise ValueError(f"{name} must be an integer in 0..2**32-1")
    return value


def _callback_identity(
    name: str,
    callback: Callable[..., object] | None,
    explicit: str | Key | None,
    *,
    default: str,
) -> str:
    if callback is None:
        if explicit is not None:
            raise ValueError(f"{name}_identity does not apply to the default")
        return default
    if explicit is not None:
        if isinstance(explicit, Key):
            identity = explicit.digest
        elif isinstance(explicit, str):
            identity = explicit
        else:
            raise TypeError(f"{name}_identity must be a string or Key")
        if not identity:
            raise ValueError(f"{name}_identity must not be empty")
        return identity
    declared = getattr(callback, "identity", None)
    if isinstance(declared, Key):
        return declared.digest
    if isinstance(declared, str) and declared:
        return declared
    try:
        closure = inspect.getclosurevars(callback)
        source = inspect.getsource(callback)
    except (OSError, TypeError, ValueError):
        raise ValueError(
            f"{name}_identity is required for a non-inspectable callable"
        ) from None
    if closure.nonlocals or closure.globals or vars(callback):
        raise ValueError(
            f"{name}_identity is required for a capturing callable"
        )
    return key_for(
        (
            f"{name}-callable/v1",
            getattr(callback, "__module__", type(callback).__module__),
            getattr(callback, "__qualname__", type(callback).__qualname__),
            source,
        )
    ).digest


def _default_proposal_seed(seed: int, proposal_id: int) -> int:
    return (seed * 100003 + proposal_id) % 2147483647 or 1


def _default_selection(
    members: tuple[BranchMember, ...],
    width: int,
) -> tuple[BranchMember, ...]:
    return tuple(sorted(members, key=lambda item: (-item.score, item.id))[:width])


def _proposal_seed(value: object) -> int:
    if type(value) is not int or not 0 <= value <= 2**32 - 1:
        raise ValueError("proposal seed must be an integer in 0..2**32-1")
    return value


def _check_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _check_blob(name: str, value: object) -> Blob:
    if not isinstance(value, Blob):
        raise TypeError(f"{name} must be a Blob")
    _check_text(f"{name}.codec", value.codec)
    if not isinstance(value.identity, Key):
        raise TypeError(f"{name}.identity must be a Key")
    _check_text(f"{name}.identity.digest", value.identity.digest)
    return value


def _check_member(value: object, *, label: str = "BranchMember") -> BranchMember:
    if not isinstance(value, BranchMember):
        raise TypeError(f"{label} must be a BranchMember")
    if type(value.id) is not int or value.id < 0:
        raise ValueError(f"{label}.id must be a non-negative integer")
    _check_text(f"{label}.key", value.key)
    _check_blob(f"{label}.value", value.value)
    if type(value.score) is not float and type(value.score) is not int:
        raise TypeError(f"{label}.score must be numeric")
    if not math.isfinite(float(value.score)):
        raise ValueError(f"{label}.score must be finite")
    if type(value.eligible) is not bool:
        raise TypeError(f"{label}.eligible must be boolean")
    if value.comparison_context is not None:
        if not isinstance(value.comparison_context, Key):
            raise TypeError(f"{label}.comparison_context must be a Key or None")
        _check_text(
            f"{label}.comparison_context.digest",
            value.comparison_context.digest,
        )
    return value


def _check_proposal(value: object) -> BranchProposal:
    if not isinstance(value, BranchProposal):
        raise TypeError("completed proposal terminal requires BranchProposal")
    if type(value.id) is not int or value.id < 1:
        raise ValueError("BranchProposal.id must be a positive integer")
    if type(value.parent_id) is not int or value.parent_id < 0:
        raise ValueError("BranchProposal.parent_id must be a non-negative integer")
    _check_text("BranchProposal.key", value.key)
    _check_blob("BranchProposal.value", value.value)
    if type(value.valid) is not bool:
        raise TypeError("BranchProposal.valid must be boolean")
    if not isinstance(value.reasons, tuple) or any(
        not isinstance(reason, str) or not reason for reason in value.reasons
    ):
        raise TypeError("BranchProposal.reasons must contain non-empty strings")
    return value


def _check_job(value: object) -> BranchJob:
    if not isinstance(value, BranchJob):
        raise TypeError("BranchJob expected")
    if value.phase not in ("propose", "evaluate"):
        raise ValueError("BranchJob.phase must be 'propose' or 'evaluate'")
    if type(value.id) is not int or value.id < 0:
        raise ValueError("BranchJob.id must be a non-negative integer")
    if type(value.generation) is not int or value.generation < 0:
        raise ValueError("BranchJob.generation must be a non-negative integer")
    if type(value.seed) is not int or not 0 <= value.seed <= 2**32 - 1:
        raise ValueError("BranchJob.seed must be in 0..2**32-1")
    if type(value.slot) is not int or value.slot < 0:
        raise ValueError("BranchJob.slot must be a non-negative integer")
    if value.phase == "propose":
        if not isinstance(value.parent, BranchMember):
            raise TypeError("proposal jobs require a BranchMember parent")
        _check_member(value.parent, label="BranchJob.parent")
        if value.proposal is not None:
            raise ValueError("proposal jobs cannot carry a proposal result")
    else:
        if value.parent is not None:
            _check_member(value.parent, label="BranchJob.parent")
        if value.proposal is not None:
            _check_proposal(value.proposal)
    if value.source is not None:
        _check_blob("BranchJob.source", value.source)
    return value


def _check_terminal(value: object) -> BranchTerminal:
    if not isinstance(value, BranchTerminal):
        raise TypeError("BranchEvent.terminals must contain BranchTerminal values")
    _check_job(value.job)
    if value.status not in ("completed", "refused"):
        raise ValueError("terminal status must be 'completed' or 'refused'")
    if value.status == "refused":
        if value.value is not None:
            raise ValueError("refused terminals cannot carry a result")
    elif value.job.phase == "propose":
        _check_proposal(value.value)
    else:
        _check_member(value.value, label="evaluation terminal value")
    return value


@dataclass(frozen=True)
class _ProposalPlan:
    jobs: tuple[BranchJob, ...]
    state: BranchingState


class BranchingPolicy:
    """One generic state machine for beam, population and serial branching."""

    request_type = BranchAction
    result_type = BranchEvent

    def __init__(
        self,
        *,
        width: int,
        proposals_per_parent: int,
        max_evaluations: int,
        proposal_cap: int,
        proposal_request_budget: int | None = None,
        seed: int,
        evaluation_seed: int,
        proposal_seed: ProposalSeed | None = None,
        proposal_seed_identity: str | Key | None = None,
        quota_mode: QuotaMode = "admitted",
        expansion: Expansion = "round_robin",
        proposal_batch_size: int = 1,
        evaluation_width: int = 1,
        pipeline: bool = False,
        immediate_evaluation: bool = False,
        selection: Selection | None = None,
        selection_identity: str | Key | None = None,
        archive_selection: bool = False,
        max_generations: int | None = None,
        cap_error: bool = False,
    ) -> None:
        self.width = _positive_int("width", width)
        self.proposals_per_parent = _positive_int(
            "proposals_per_parent", proposals_per_parent
        )
        self.max_evaluations = _positive_int(
            "max_evaluations", max_evaluations
        )
        self.proposal_cap = _positive_int("proposal_cap", proposal_cap)
        if proposal_request_budget is not None:
            proposal_request_budget = _positive_int(
                "proposal_request_budget",
                proposal_request_budget,
            )
        self.proposal_request_budget = proposal_request_budget
        self.seed = _seed("seed", seed)
        self.evaluation_seed = _seed("evaluation_seed", evaluation_seed)
        self.proposal_batch_size = _positive_int(
            "proposal_batch_size", proposal_batch_size
        )
        self.evaluation_width = _positive_int(
            "evaluation_width", evaluation_width
        )
        if quota_mode not in ("admitted", "attempts"):
            raise ValueError("quota_mode must be 'admitted' or 'attempts'")
        if expansion not in ("balanced", "round_robin"):
            raise ValueError("expansion must be 'balanced' or 'round_robin'")
        if type(pipeline) is not bool:
            raise TypeError("pipeline must be boolean")
        if type(immediate_evaluation) is not bool:
            raise TypeError("immediate_evaluation must be boolean")
        if immediate_evaluation and pipeline:
            raise ValueError(
                "immediate_evaluation and pipeline are separate execution modes"
            )
        if immediate_evaluation and (
            self.proposal_batch_size != 1 or self.evaluation_width != 1
        ):
            raise ValueError(
                "immediate_evaluation requires proposal_batch_size=1 "
                "and evaluation_width=1"
            )
        if type(archive_selection) is not bool:
            raise TypeError("archive_selection must be boolean")
        if type(cap_error) is not bool:
            raise TypeError("cap_error must be boolean")
        if max_generations is not None:
            max_generations = _nonnegative_int(
                "max_generations", max_generations
            )

        self.proposal_seed = proposal_seed or _default_proposal_seed
        self.proposal_seed_identity = _callback_identity(
            "proposal_seed",
            proposal_seed,
            proposal_seed_identity,
            default=_DEFAULT_PROPOSAL_SEED_IDENTITY,
        )
        self.selection = selection or _default_selection
        self.selection_identity = _callback_identity(
            "selection",
            selection,
            selection_identity,
            default=_DEFAULT_SELECTION_IDENTITY,
        )
        self.quota_mode = quota_mode
        self.expansion = expansion
        self.pipeline = pipeline
        self.immediate_evaluation = immediate_evaluation
        self.archive_selection = archive_selection
        self.max_generations = max_generations
        self.cap_error = cap_error
        self.identity = key_for(
            (
                _CONTROL_SCHEMA,
                inspect.getsource(BranchingPolicy),
                inspect.getsource(type(self)),
                self.width,
                self.proposals_per_parent,
                self.max_evaluations,
                self.proposal_cap,
                self.proposal_request_budget,
                self.seed,
                self.evaluation_seed,
                self.proposal_seed_identity,
                self.quota_mode,
                self.expansion,
                self.proposal_batch_size,
                self.evaluation_width,
                self.pipeline,
                self.immediate_evaluation,
                self.selection_identity,
                self.archive_selection,
                self.max_generations,
                self.cap_error,
            )
        )

    def _validate_state(self, state: BranchingState) -> BranchingState:
        if not isinstance(state, BranchingState):
            raise TypeError("BranchingPolicy requires BranchingState")
        for name in (
            "generation",
            "proposals",
            "next_evaluation",
            "duplicates",
            "rejected",
        ):
            if type(getattr(state, name)) is not int or getattr(state, name) < 0:
                raise ValueError(f"BranchingState.{name} must be non-negative")
        if state.next_evaluation < 1:
            raise ValueError("BranchingState.next_evaluation must be positive")
        for name in ("records", "frontier", "parents"):
            values = getattr(state, name)
            if not isinstance(values, tuple):
                raise TypeError(f"BranchingState.{name} must be a tuple")
            for value in values:
                _check_member(value, label=f"BranchingState.{name} item")
            if len({value.id for value in values}) != len(values):
                raise ValueError(f"BranchingState.{name} has duplicate IDs")
        if not isinstance(state.counts, tuple):
            raise TypeError("BranchingState.counts must be a tuple")
        parent_ids: set[int] = set()
        for item in state.counts:
            if (
                not isinstance(item, tuple)
                or len(item) != 2
                or type(item[0]) is not int
                or item[0] < 0
                or type(item[1]) is not int
                or item[1] < 0
            ):
                raise ValueError(
                    "BranchingState.counts must contain (parent_id, count) pairs"
                )
            if item[0] in parent_ids:
                raise ValueError("BranchingState.counts has duplicate parent IDs")
            parent_ids.add(item[0])
        for name, phase in (
            ("batch", "propose"),
            ("queued", "evaluate"),
            ("pending", "evaluate"),
        ):
            values = getattr(state, name)
            if not isinstance(values, tuple):
                raise TypeError(f"BranchingState.{name} must be a tuple")
            for job in values:
                _check_job(job)
                if job.phase != phase:
                    raise ValueError(
                        f"BranchingState.{name} must contain {phase} jobs"
                    )
            if len({job.identity for job in values}) != len(values):
                raise ValueError(f"BranchingState.{name} has duplicate jobs")
        if set(job.identity for job in state.queued) & set(
            job.identity for job in state.pending
        ):
            raise ValueError("an evaluation job cannot be queued and pending")
        if not isinstance(state.batch_results, tuple):
            raise TypeError("BranchingState.batch_results must be a tuple")
        proposal_ids: set[int] = set()
        for terminal in state.batch_results:
            _check_terminal(terminal)
            if terminal.job.phase != "propose":
                raise ValueError("batch_results must contain proposal terminals")
            if terminal.job.id in proposal_ids:
                raise ValueError("batch_results has duplicate proposal IDs")
            proposal_ids.add(terminal.job.id)
        if not isinstance(state.seen, tuple) or any(
            not isinstance(key, str) or not key for key in state.seen
        ):
            raise TypeError("BranchingState.seen must contain non-empty strings")
        if len(set(state.seen)) != len(state.seen):
            raise ValueError("BranchingState.seen contains duplicates")
        if not isinstance(state.refused, tuple):
            raise TypeError("BranchingState.refused must be a tuple")
        seen_refusals: set[str] = set()
        for job in state.refused:
            _check_job(job)
            if job.identity in seen_refusals:
                raise ValueError("BranchingState.refused contains duplicate jobs")
            seen_refusals.add(job.identity)
        if state.stop_reason is not None:
            _check_text("BranchingState.stop_reason", state.stop_reason)
        if type(state.finished) is not bool:
            raise TypeError("BranchingState.finished must be boolean")
        if state.finished and state.stop_reason is None:
            raise ValueError("a finished branching state requires a stop reason")
        return state

    @staticmethod
    def _counts(state: BranchingState) -> dict[int, int]:
        return dict(state.counts)

    def _evaluation_load(self, state: BranchingState) -> int:
        return len(state.records) + len(state.queued) + len(state.pending)

    def _evaluation_room(self, state: BranchingState) -> int:
        return max(0, self.max_evaluations - state.next_evaluation)

    def _proposal_request_limit(self) -> int:
        if self.proposal_request_budget is None:
            return self.proposal_cap
        return min(self.proposal_cap, self.proposal_request_budget)

    def _proposal_request_budget_reached(self, state: BranchingState) -> bool:
        return (
            self.proposal_request_budget is not None
            and state.proposals >= self._proposal_request_limit()
        )

    def _proposal_phase_complete(self, state: BranchingState) -> bool:
        if state.generation == 0:
            return bool(state.records or state.refused)
        if not state.parents:
            return True
        if state.stop_reason is not None:
            return True
        if self._proposal_request_budget_reached(state):
            return True
        if state.proposals >= self.proposal_cap:
            return True
        if self._evaluation_room(state) == 0:
            return True
        counts = self._counts(state)
        return all(
            counts.get(parent.id, 0) >= self.proposals_per_parent
            for parent in state.parents
        )

    def _can_dispatch_evaluations(self, state: BranchingState) -> bool:
        if not state.queued:
            return False
        if state.generation == 0:
            return True
        return (
            self.immediate_evaluation
            or self.pipeline
            or self._proposal_phase_complete(state)
        )

    def _can_issue_proposals(self, state: BranchingState) -> bool:
        if (
            state.batch
            or state.generation == 0
            or not state.parents
            or self._evaluation_room(state) == 0
            or self._proposal_request_budget_reached(state)
            or state.proposals >= self.proposal_cap
        ):
            return False
        counts = self._counts(state)
        return any(
            counts.get(parent.id, 0) < self.proposals_per_parent
            for parent in state.parents
        )

    def _proposal_plan(self, state: BranchingState) -> _ProposalPlan:
        if (
            state.batch
            or state.generation == 0
            or not state.parents
            or self._proposal_phase_complete(state)
        ):
            return _ProposalPlan((), state)
        if self._evaluation_room(state) == 0:
            return _ProposalPlan(
                (), replace(state, stop_reason="evaluation_budget")
            )
        if self._proposal_request_budget_reached(state):
            return _ProposalPlan(
                (), replace(state, stop_reason="native_call_budget")
            )
        if state.proposals >= self.proposal_cap:
            if self.cap_error:
                raise ValueError(
                    f"proposal cap {self.proposal_cap} would be exceeded"
                )
            return _ProposalPlan(
                (), replace(state, stop_reason="proposal_cap")
            )

        planned = self._counts(state)
        jobs: list[BranchJob] = []
        parent_order = {parent.id: index for index, parent in enumerate(state.parents)}
        remaining_cap = min(
            self.proposal_cap,
            self.proposal_request_budget or self.proposal_cap,
        ) - state.proposals
        target = min(
            self.proposal_batch_size,
            remaining_cap,
            self._evaluation_room(state),
        )
        while len(jobs) < target:
            feasible = [
                parent
                for parent in state.parents
                if planned.get(parent.id, 0) < self.proposals_per_parent
            ]
            if not feasible:
                break
            parent = min(
                feasible,
                key=lambda item: (planned.get(item.id, 0), parent_order[item.id]),
            )
            slot = planned.get(parent.id, 0)
            proposal_id = state.proposals + len(jobs) + 1
            seed = _proposal_seed(self.proposal_seed(self.seed, proposal_id))
            jobs.append(
                BranchJob(
                    phase="propose",
                    id=proposal_id,
                    generation=state.generation,
                    seed=seed,
                    parent=parent,
                    slot=slot,
                    source=parent.value,
                )
            )
            planned[parent.id] = slot + 1

        if not jobs:
            return _ProposalPlan((), state)
        frozen_jobs = tuple(jobs)
        counts = state.counts
        if self.quota_mode == "attempts":
            updated = self._counts(state)
            for job in frozen_jobs:
                updated[job.parent.id] = updated.get(job.parent.id, 0) + 1
            counts = tuple(sorted(updated.items()))
        reserved = replace(
            state,
            proposals=state.proposals + len(frozen_jobs),
            counts=counts,
            batch=frozen_jobs,
        )
        return _ProposalPlan(frozen_jobs, reserved)

    def _dispatch_plan(
        self,
        state: BranchingState,
    ) -> tuple[tuple[BranchJob, ...], BranchingState]:
        if not self._can_dispatch_evaluations(state):
            return (), state
        available = self.evaluation_width - len(state.pending)
        if available <= 0:
            return (), state
        jobs = state.queued[:available]
        if not jobs:
            return (), state
        return jobs, replace(
            state,
            queued=state.queued[len(jobs):],
            pending=state.pending + jobs,
        )

    def _wait_action(self, state: BranchingState) -> BranchAction:
        if state.pending:
            wait: Literal["any", "all"] = (
                "any" if self.pipeline or state.batch else "all"
            )
        elif state.batch:
            wait = "any" if self.pipeline else "all"
        else:
            raise RuntimeError("branching policy has no outstanding work")
        return BranchAction(state=state, wait=wait)

    def start(self, root: BranchJob) -> BranchingState:
        _check_job(root)
        if (
            root.phase != "evaluate"
            or root.id != 0
            or root.generation != 0
            or root.parent is not None
            or root.proposal is not None
            or root.slot != 0
            or root.seed != self.evaluation_seed
        ):
            raise ValueError(
                "root must be evaluation job 0 in generation 0 "
                "with the configured evaluation seed"
            )
        return self._validate_state(
            BranchingState(next_evaluation=1, queued=(root,))
        )

    def ask(self, state: BranchingState) -> BranchAction:
        self._validate_state(state)
        if state.finished:
            raise ValueError("cannot ask a finished branching policy")

        jobs, reserved = self._dispatch_plan(state)
        if jobs:
            wait: Literal["any", "all", "none"]
            if self.pipeline and self._can_issue_proposals(reserved):
                # Submit this lane without consuming the complete ready set so
                # the next control turn can fill the independent proposal lane.
                wait = "none"
            elif self.pipeline or state.pending or state.batch:
                wait = "any"
            else:
                wait = "all"
            return BranchAction(state=reserved, jobs=jobs, wait=wait)

        if state.batch:
            return self._wait_action(state)

        if self.pipeline or not self._proposal_phase_complete(state):
            plan = self._proposal_plan(state)
            if plan.jobs:
                return BranchAction(
                    state=plan.state,
                    jobs=plan.jobs,
                    wait="any" if state.pending or self.pipeline else "all",
                )
            if plan.state.stop_reason != state.stop_reason:
                return BranchAction(state=plan.state, wait="none")

        if state.pending or state.batch:
            return self._wait_action(state)
        if state.queued:
            # Evaluations are held explicitly until the non-pipeline proposal
            # phase has committed its complete generation.
            if self._can_dispatch_evaluations(state):
                raise RuntimeError("dispatchable evaluation jobs were not returned")
            return BranchAction(state=state, wait="any")
        if (
            self.cap_error
            and state.generation > 0
            and state.proposals >= self.proposal_cap
            and len(state.records) < self.max_evaluations
            and any(
                self._counts(state).get(parent.id, 0)
                < self.proposals_per_parent
                for parent in state.parents
            )
        ):
            raise ValueError(
                f"proposal cap {self.proposal_cap} would be exceeded"
            )
        return BranchAction(state=self._advance_generation(state), wait="none")

    def admit(
        self,
        state: BranchingState,
        action: BranchAction,
    ) -> BranchingState:
        self._validate_state(state)
        if not isinstance(action, BranchAction):
            raise TypeError("BranchingPolicy.admit requires BranchAction")
        self._validate_state(action.state)
        if state.finished:
            raise ValueError("cannot admit work after the policy finished")

        jobs = action.jobs
        if not jobs:
            return action.state
        if len({job.identity for job in jobs}) != len(jobs):
            raise ValueError("BranchAction contains duplicate job identities")
        phases = {job.phase for job in jobs}
        if len(phases) != 1:
            raise ValueError("BranchAction cannot mix proposal and evaluation jobs")
        phase = next(iter(phases))

        if phase == "propose":
            if state.batch or action.state.batch != jobs:
                raise ValueError("proposal action did not reserve its exact batch")
            if action.state.generation != state.generation:
                raise ValueError("proposal action changed generation")
            expected_ids = tuple(
                range(state.proposals + 1, state.proposals + len(jobs) + 1)
            )
            if tuple(job.id for job in jobs) != expected_ids:
                raise ValueError("proposal IDs are not contiguous and ordered")
            if any(
                job.generation != state.generation
                or job.parent not in state.parents
                for job in jobs
            ):
                raise ValueError("proposal job belongs to another branch state")
            if action.state.proposals != state.proposals + len(jobs):
                raise ValueError("proposal action has invalid proposal accounting")
            if self.quota_mode == "attempts":
                counts = self._counts(state)
                for job in jobs:
                    counts[job.parent.id] = counts.get(job.parent.id, 0) + 1
                if action.state.counts != tuple(sorted(counts.items())):
                    raise ValueError("attempt quota was not reserved exactly")
            elif action.state.counts != state.counts:
                raise ValueError("admitted quota changed before evaluation")
        else:
            jobs = tuple(jobs)
            if action.state.queued != state.queued[len(jobs):]:
                raise ValueError("evaluation action removed another queued job")
            if jobs != state.queued[:len(jobs)]:
                raise ValueError("evaluation jobs are not the next queued jobs")
            if action.state.pending != state.pending + jobs:
                raise ValueError("evaluation dispatch accounting changed")
            if len(action.state.pending) > self.evaluation_width:
                raise ValueError("evaluation dispatch exceeds configured width")
        return action.state

    def _proposal_terminal_state(
        self,
        state: BranchingState,
        terminals: tuple[BranchTerminal, ...],
    ) -> BranchingState:
        active = {job.identity: job for job in state.batch}
        completed = {
            terminal.job.id: terminal for terminal in state.batch_results
        }
        if not active:
            raise ValueError("no proposal batch is awaiting terminals")
        for terminal in terminals:
            job = active.get(terminal.job.identity)
            if job is None:
                raise ValueError(
                    f"proposal terminal {terminal.job.identity!r} "
                    "does not belong to the active batch"
                )
            if terminal.job != job:
                raise ValueError("proposal terminal does not match its job")
            if terminal.job.id in completed:
                raise ValueError("proposal terminal was already delivered")
            completed[terminal.job.id] = terminal

        batch_results = tuple(
            sorted(completed.values(), key=lambda item: item.job.id)
        )
        batch = tuple(sorted(state.batch, key=lambda item: item.id))
        if not self.pipeline and any(job.id not in completed for job in batch):
            return replace(
                state, batch_results=batch_results
            )

        # Only the remaining batch participates: results include earlier
        # committed batches as well as terminals buffered beyond a gap.
        ordered: list[BranchTerminal] = []
        for job in batch:
            if job.id not in completed:
                break
            ordered.append(completed[job.id])
        if not ordered:
            return replace(state, batch_results=batch_results)
        seen = list(state.seen)
        counts = self._counts(state)
        queued = list(state.queued)
        refused = list(state.refused)
        next_evaluation = state.next_evaluation
        # A refusal may have committed in an earlier prefix (or evaluation).
        # Drain admitted work, but never restart the cleared evaluation queue.
        stopped = state.stop_reason == "native_call_budget"
        rejected = state.rejected
        duplicates = state.duplicates
        load = self._evaluation_load(state)
        stop_reason = state.stop_reason

        for terminal in ordered:
            job = terminal.job
            if terminal.status == "refused":
                refused.append(job)
                stopped = True
                continue
            proposal = _check_proposal(terminal.value)
            if (
                proposal.id != job.id
                or proposal.parent_id != job.parent.id
            ):
                raise ValueError(
                    "proposal result identity does not match its proposal job"
                )
            if not proposal.valid:
                rejected += 1
                continue
            if proposal.key in seen:
                duplicates += 1
                continue
            if load >= self.max_evaluations:
                stop_reason = "evaluation_budget"
                continue
            evaluation = BranchJob(
                phase="evaluate",
                id=next_evaluation,
                generation=state.generation,
                seed=self.evaluation_seed,
                parent=job.parent,
                proposal=proposal,
                slot=job.slot,
                source=proposal.value,
            )
            queued.append(evaluation)
            seen.append(proposal.key)
            next_evaluation += 1
            load += 1
            if self.quota_mode == "admitted":
                counts[job.parent.id] = counts.get(job.parent.id, 0) + 1

        if stopped:
            stop_reason = "native_call_budget"
            queued.clear()
        return replace(
            state,
            next_evaluation=next_evaluation,
            counts=tuple(sorted(counts.items())),
            batch=batch[len(ordered):],
            batch_results=batch_results,
            queued=tuple(queued),
            refused=tuple(refused),
            seen=tuple(seen),
            duplicates=duplicates,
            rejected=rejected,
            stop_reason=stop_reason,
        )

    def _evaluation_terminal_state(
        self,
        state: BranchingState,
        terminals: tuple[BranchTerminal, ...],
    ) -> BranchingState:
        pending = {job.identity: job for job in state.pending}
        observed: dict[str, BranchTerminal] = {}
        for terminal in terminals:
            job = pending.get(terminal.job.identity)
            if job is None:
                raise ValueError(
                    f"evaluation terminal {terminal.job.identity!r} "
                    "is not pending"
                )
            if terminal.job != job:
                raise ValueError("evaluation terminal does not match its job")
            if terminal.job.identity in observed:
                raise ValueError("evaluation terminal was delivered twice")
            observed[terminal.job.identity] = terminal

        ordered = tuple(
            sorted(observed.values(), key=lambda item: item.job.id)
        )
        remain = tuple(
            job for job in state.pending if job.identity not in observed
        )
        records = list(state.records)
        frontier = list(state.frontier)
        refused = list(state.refused)
        seen = list(state.seen)
        completed_ids = {member.id for member in records}
        stop_reason = state.stop_reason
        stopped = False

        for terminal in ordered:
            job = terminal.job
            if terminal.status == "refused":
                refused.append(job)
                stopped = True
                continue
            member = _check_member(
                terminal.value, label="completed evaluation value"
            )
            if member.id != job.id:
                raise ValueError(
                    "evaluation result ID does not match its evaluation job"
                )
            if job.proposal is not None:
                if member.key != job.proposal.key:
                    raise ValueError(
                        "evaluation result key differs from its admitted proposal"
                    )
                if member.key not in state.seen:
                    raise ValueError("evaluation result was not admitted")
            else:
                if job.id != 0 or member.key in state.seen:
                    raise ValueError("invalid root evaluation identity")
                seen.append(member.key)
            if member.id in completed_ids:
                raise ValueError("evaluation result ID was already recorded")
            records.append(member)
            frontier.append(member)
            completed_ids.add(member.id)

        if stopped:
            stop_reason = "native_call_budget"
        return replace(
            state,
            records=tuple(sorted(records, key=lambda member: member.id)),
            frontier=tuple(sorted(frontier, key=lambda member: member.id)),
            pending=remain,
            queued=() if stopped else state.queued,
            seen=tuple(seen),
            refused=tuple(refused),
            stop_reason=stop_reason,
        )

    def tell(
        self,
        state: BranchingState,
        event: BranchEvent,
    ) -> BranchingState:
        self._validate_state(state)
        if not isinstance(event, BranchEvent):
            raise TypeError("BranchingPolicy.tell requires BranchEvent")
        terminals = tuple(event.terminals)
        for terminal in terminals:
            _check_terminal(terminal)
        if not terminals:
            return state
        if len({item.job.identity for item in terminals}) != len(terminals):
            raise ValueError("BranchEvent contains duplicate job identities")
        proposals = tuple(
            sorted(
                (
                    terminal
                    for terminal in terminals
                    if terminal.job.phase == "propose"
                ),
                key=lambda terminal: terminal.job.id,
            )
        )
        evaluations = tuple(
            sorted(
                (
                    terminal
                    for terminal in terminals
                    if terminal.job.phase == "evaluate"
                ),
                key=lambda terminal: terminal.job.id,
            )
        )
        if proposals:
            state = self._proposal_terminal_state(state, proposals)
        if evaluations:
            if not state.pending:
                raise ValueError("no evaluation jobs are pending")
            state = self._evaluation_terminal_state(state, evaluations)
        return state

    def _select(self, members: tuple[BranchMember, ...]) -> tuple[BranchMember, ...]:
        eligible = tuple(member for member in members if member.eligible)
        if not eligible:
            return ()
        contexts = {
            member.comparison_context
            for member in eligible
            if member.comparison_context is not None
        }
        if len(contexts) > 1:
            raise ValueError(
                "comparison_context mismatch requires explicit reassessment"
            )
        selected = self.selection(eligible, self.width)
        if not isinstance(selected, tuple):
            raise TypeError("selection callback must return a tuple")
        by_id = {member.id: member for member in eligible}
        if not selected:
            raise ValueError("selection callback returned no members")
        if len({member.id for member in selected}) != len(selected):
            raise ValueError("selection callback returned duplicate member IDs")
        if len(selected) > self.width:
            raise ValueError("selection callback exceeded beam width")
        if any(
            not isinstance(member, BranchMember)
            or by_id.get(member.id) != member
            for member in selected
        ):
            raise ValueError(
                "selection callback returned a member outside its candidate set"
            )
        return tuple(selected)

    def _advance_generation(self, state: BranchingState) -> BranchingState:
        candidates = (
            state.records if self.archive_selection else state.frontier
        )
        selected = self._select(candidates)
        reason = state.stop_reason
        if not selected:
            reason = reason or "no_feasible_parent"
        elif reason is None:
            if (
                self.max_generations is not None
                and state.generation >= self.max_generations
            ):
                reason = "generation_limit"
            elif self._evaluation_load(state) >= self.max_evaluations:
                reason = "evaluation_budget"
            elif self._proposal_request_budget_reached(state):
                reason = "native_call_budget"
            elif state.proposals >= self.proposal_cap:
                reason = "proposal_cap"
        if reason is not None:
            return replace(
                state,
                frontier=selected,
                stop_reason=reason,
                finished=True,
            )
        return replace(
            state,
            generation=state.generation + 1,
            records=state.records,
            frontier=selected,
            parents=selected,
            counts=(),
            batch=(),
            batch_results=(),
            queued=(),
            pending=(),
            duplicates=state.duplicates,
            rejected=state.rejected,
            stop_reason=None,
            finished=False,
        )

    def finished(self, state: BranchingState) -> bool:
        self._validate_state(state)
        return state.finished

    def selected(self, state: BranchingState) -> BranchingState:
        self._validate_state(state)
        if not state.finished:
            raise ValueError("branching policy has not finished")
        return state


__all__ = ["BranchingPolicy"]
