"""Deterministic two-phase beam search over typed proposals and evaluations.

The policy separates proposal attempts from completed evaluations. Proposal
work consumes a frozen parent/slot attempt; only an accepted, non-duplicate
candidate receives evaluation work. Runtime remains responsible for enforcing
actual paid-operation budgets.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
import hashlib
import inspect
import math
from typing import Literal

from .store import DataclassCodec
from .types import (
    Candidate,
    History,
    Key,
    Proposal,
    Sample,
    Scored,
    TrialOutcome,
    Work,
    key_for,
)


Direction = Literal["max", "min", 1, -1]
ExpansionMode = Literal["each", "round_robin"]
Phase = Literal["propose", "evaluate", "finished"]
RankCallable = Callable[[Scored], Sequence[float]]
ProposalSeed = Callable[[int], int]
Projection = Callable[[Candidate], object]

_STATE_SCHEMA = "astrolabe.design.phased-search.State/v1"
_CONTROL_SCHEMA = "astrolabe.design.phased-search.Control/v1"
_PROTOCOL_SCHEMA = "astrolabe.design.phased-search.Protocol/v1"
_WORK_SCHEMA = "astrolabe.design.phased-search.Work/v1"
_STATUSES = ("completed", "failed", "unknown", "refused")


@dataclass(frozen=True)
class PhasedBeamState:
    """Immutable checkpoint for a proposal/evaluation beam generation."""

    schema: str
    policy: Key
    protocol: Key
    generation: int
    root_evaluations: int
    evaluations: int
    proposals: int
    duplicates: int
    proposal_cap: int
    max_evaluations: int
    frontier: tuple[Scored, ...]
    generation_parents: tuple[Scored, ...]
    admission_ordinals: tuple[tuple[str, int], ...]
    next_slot: int
    slot_count: int
    pending_slot: int | None
    pending_proposal_number: int | None
    pending_parent: Scored | None
    pending_proposal: Proposal | None
    currentgen_children: tuple[Scored, ...]
    child_admission_slots: tuple[int, ...]
    seen_candidates: tuple[str, ...]
    told_work: tuple[str, ...]
    proposal_status_counts: tuple[tuple[str, int], ...]
    evaluation_status_counts: tuple[tuple[str, int], ...]
    finished: bool

    @property
    def phase(self) -> Phase:
        if self.finished:
            return "finished"
        if self.pending_proposal is not None:
            return "evaluate"
        return "propose"

    @property
    def completed_evaluations(self) -> int:
        return self.evaluations

    @property
    def selected_frontier(self) -> tuple[Scored, ...]:
        return self.frontier


class PhasedBeamStateCodec:
    """Versioned JSON-safe checkpoint codec for :class:`PhasedBeamState`."""

    identity = _STATE_SCHEMA

    def __init__(self) -> None:
        self._codec = DataclassCodec(PhasedBeamState, self.identity)

    def encode(self, value: PhasedBeamState) -> bytes:
        if not isinstance(value, PhasedBeamState):
            raise TypeError("PhasedBeamStateCodec expects PhasedBeamState")
        return self._codec.encode(value)

    def decode(self, data: bytes) -> PhasedBeamState:
        value = self._codec.decode(data)
        if not isinstance(value, PhasedBeamState):
            raise TypeError("PhasedBeamStateCodec decoded another type")
        return value

    def references(self, value: PhasedBeamState):
        return self._codec.references(value)


PHASED_BEAM_STATE_CODEC = PhasedBeamStateCodec()


def _positive_int(name: str, value: object) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _seed(name: str, value: object) -> int:
    if type(value) is not int or not 0 <= value <= 2**32 - 1:
        raise ValueError(f"{name} must be an integer in 0..2**32-1")
    return value


def _normalize_directions(
    direction: Direction | None,
    directions: Sequence[Direction] | None,
) -> tuple[int, ...]:
    if direction is not None and directions is not None:
        raise ValueError("specify direction or directions, not both")
    selected: Sequence[Direction]
    if directions is None:
        selected = (direction if direction is not None else "max",)
    else:
        selected = directions
    if not selected:
        raise ValueError("directions must contain at least one maximize/minimize value")
    result: list[int] = []
    for value in selected:
        if value in (1, "max"):
            result.append(1)
        elif value in (-1, "min"):
            result.append(-1)
        else:
            raise ValueError("directions allow only 'max'/'min' or 1/-1")
    return tuple(result)


def _required_callable_identity(
    name: str,
    value: Callable[..., object] | None,
    identity: str | None,
) -> str | None:
    if value is None:
        if identity is not None:
            raise ValueError(f"{name}_identity does not apply without {name}")
        return None
    if not isinstance(identity, str) or not identity:
        raise ValueError(f"{name}_identity is required when {name} is callable")
    return identity


def _rank_identity(rank: RankCallable | None, explicit: str | None) -> str:
    if rank is None:
        if explicit is not None:
            raise ValueError("rank_identity does not apply to the default rank")
        return "assessment.values/v1"
    if explicit is not None:
        if not isinstance(explicit, str) or not explicit:
            raise ValueError("rank_identity must be a non-empty string")
        return explicit
    try:
        closure = inspect.getclosurevars(rank)
        defaults = tuple(inspect.signature(rank).parameters.values())
        attributes = vars(rank)
    except (TypeError, AttributeError):
        raise ValueError(
            "rank_identity is required for a non-function or introspectable rank callable"
        ) from None
    captures_defaults = any(
        parameter.default is not inspect.Parameter.empty
        and not callable(parameter.default)
        for parameter in defaults[1:]
    )
    if closure.nonlocals or attributes or captures_defaults:
        raise ValueError("rank_identity is required for a capturing rank callable")
    try:
        source = inspect.getsource(rank)
    except (OSError, TypeError):
        source = ""
    return key_for(
        (
            "rank-callable/v1",
            getattr(rank, "__module__", type(rank).__module__),
            getattr(rank, "__qualname__", type(rank).__qualname__),
            source,
        )
    ).digest


class PhasedBeamSearchPolicy:
    """Generic elitist beam with explicit proposal and evaluation phases.

    One ``ask`` returns one proposal or one evaluation, regardless of capacity.
    Slots are frozen per generation. Each slot is attempted at most once;
    duplicates and failed proposals are not refilled. A completed evaluation
    contributes one child. The root evaluations count toward
    ``max_evaluations``.
    """

    kind = "phased_beam"
    codec = PHASED_BEAM_STATE_CODEC

    def __init__(
        self,
        *,
        width: int,
        proposals_per_parent: int,
        max_evaluations: int,
        proposal_cap: int,
        seed: int,
        evaluation_seed: int,
        proposal_seed: ProposalSeed | None = None,
        seed_identity: str | None = None,
        direction: Direction | None = None,
        directions: Sequence[Direction] | None = None,
        rank: RankCallable | None = None,
        rank_identity: str | None = None,
        expansion: ExpansionMode = "round_robin",
        projection: Projection | None = None,
        projection_identity: str | None = None,
    ) -> None:
        self.width = _positive_int("width", width)
        self.proposals_per_parent = _positive_int(
            "proposals_per_parent", proposals_per_parent
        )
        self.max_evaluations = _positive_int(
            "max_evaluations", max_evaluations
        )
        self.proposal_cap = _positive_int("proposal_cap", proposal_cap)
        self.seed = _seed("seed", seed)
        self.evaluation_seed = _seed("evaluation_seed", evaluation_seed)
        if expansion not in ("each", "round_robin"):
            raise ValueError("expansion must be 'each' or 'round_robin'")
        self.expansion = expansion
        self.directions = _normalize_directions(direction, directions)
        self.rank = rank
        self.rank_identity = _rank_identity(rank, rank_identity)
        self.proposal_seed = proposal_seed
        self.seed_identity = _required_callable_identity(
            "seed", proposal_seed, seed_identity
        )
        self.projection = projection
        self.projection_identity = _required_callable_identity(
            "projection", projection, projection_identity
        )
        self.protocol = key_for((_PROTOCOL_SCHEMA, self.kind))
        self.identity = key_for(
            (
                _CONTROL_SCHEMA,
                self.kind,
                self.width,
                self.proposals_per_parent,
                self.max_evaluations,
                self.proposal_cap,
                self.seed,
                self.seed_identity,
                self.evaluation_seed,
                self.directions,
                self.rank_identity,
                self.expansion,
                self.projection_identity,
                self.codec.identity,
                inspect.getsource(PhasedBeamSearchPolicy),
                inspect.getsource(type(self)),
            )
        )

    def _rank_values(self, scored: Scored) -> tuple[float, ...]:
        raw = scored.assessment.values if self.rank is None else self.rank(scored)
        values = tuple(float(value) for value in raw)
        if len(values) != len(self.directions):
            raise ValueError(
                f"rank returned {len(values)} values for "
                f"{len(self.directions)} directions"
            )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("rank values must be finite")
        return values

    @staticmethod
    def _gate_passed(scored: Scored) -> bool:
        return all(bool(value) for _, value in scored.assessment.gates)

    def _rank_key(self, scored: Scored) -> tuple[object, ...]:
        gate_rank = 0 if self._gate_passed(scored) else 1
        values = self._rank_values(scored)
        return (
            gate_rank,
            tuple(
                -direction * value
                for direction, value in zip(self.directions, values)
            ),
        )

    def _select(self, values: Sequence[Scored]) -> tuple[Scored, ...]:
        seen: set[str] = set()
        unique: list[Scored] = []
        for scored in values:
            identity = scored.candidate.identity.digest
            if identity in seen:
                continue
            seen.add(identity)
            unique.append(scored)
        eligible = (value for value in unique if self._gate_passed(value))
        return tuple(sorted(eligible, key=self._rank_key)[: self.width])

    def _projection_token(self, candidate: Candidate) -> str:
        if self.projection is None:
            value = candidate.identity
        else:
            value = self.projection(candidate)
        return key_for(("candidate-projection/v1", self.projection_identity, value)).digest

    def _default_proposal_seed(self, index: int) -> int:
        encoded = f"{self.seed}:{index}".encode("ascii")
        return int.from_bytes(hashlib.sha256(encoded).digest()[:4], "big") % (
            2**32 - 1
        ) + 1

    def _proposal_seed(self, index: int) -> int:
        value = (
            self._default_proposal_seed(index)
            if self.proposal_seed is None
            else self.proposal_seed(index)
        )
        return _seed("proposal seed result", value)

    def _slot_details(
        self,
        parents: tuple[Scored, ...],
        slot: int,
    ) -> tuple[int, int, Scored]:
        if not 0 <= slot < len(parents) * self.proposals_per_parent:
            raise IndexError("proposal slot is outside the frozen generation")
        if self.expansion == "round_robin":
            parent_index = slot % len(parents)
            proposal_index = slot // len(parents)
        else:
            parent_index = slot // self.proposals_per_parent
            proposal_index = slot % self.proposals_per_parent
        return parent_index, proposal_index, parents[parent_index]

    def _admission_ordinal(
        self,
        state: PhasedBeamState,
        parent: Scored,
    ) -> int:
        digest = parent.candidate.identity.digest
        matches = tuple(
            ordinal
            for candidate_digest, ordinal in state.admission_ordinals
            if candidate_digest == digest
        )
        if len(matches) != 1:
            raise ValueError("parent lacks one global admission ordinal")
        return matches[0]

    def _proposal_sample(
        self,
        *,
        generation: int,
        parent_admission_ordinal: int,
        proposal_index: int,
        proposal_number: int,
    ) -> Sample:
        return Sample(
            protocol=self.protocol,
            replicate=(
                f"generation_{generation:04d}_parent_{parent_admission_ordinal:04d}"
                f"_proposal_{proposal_index:02d}"
            ),
            seed=self._proposal_seed(proposal_number),
            index=str(proposal_number),
        )

    def _proposal_work(self, state: PhasedBeamState) -> Work:
        _, proposal_index, parent = self._slot_details(
            state.generation_parents, state.next_slot
        )
        proposal_number = state.proposals + 1
        sample = self._proposal_sample(
            generation=state.generation,
            parent_admission_ordinal=self._admission_ordinal(state, parent),
            proposal_index=proposal_index,
            proposal_number=proposal_number,
        )
        identity = key_for(
            (
                _WORK_SCHEMA,
                self.identity,
                state.generation,
                state.next_slot,
                parent.candidate.identity,
                sample,
                "propose",
            )
        ).digest
        return Work(
            identity=identity,
            parents=(parent,),
            stage="propose",
            sample=sample,
            candidate=None,
        )

    def _evaluation_work(self, state: PhasedBeamState) -> Work:
        if (
            state.pending_parent is None
            or state.pending_proposal is None
            or state.pending_proposal_number is None
            or state.pending_slot is None
        ):
            raise ValueError("evaluation phase requires a pending proposal")
        _, proposal_index, parent = self._slot_details(
            state.generation_parents, state.pending_slot
        )
        if parent.candidate.identity != state.pending_parent.candidate.identity:
            raise ValueError("pending parent does not match its frozen slot")
        sample = Sample(
            protocol=self.protocol,
            replicate=(
                f"generation_{state.generation:04d}_parent_"
                f"{self._admission_ordinal(state, parent):04d}"
                f"_proposal_{proposal_index:02d}_evaluate"
            ),
            seed=self.evaluation_seed,
            index=str(state.pending_proposal_number),
        )
        identity = key_for(
            (
                _WORK_SCHEMA,
                self.identity,
                state.generation,
                state.pending_slot,
                state.pending_proposal_number,
                state.pending_parent.candidate.identity,
                state.pending_proposal.identity,
                state.pending_proposal.candidate.identity,
                sample,
                "evaluate",
            )
        ).digest
        return Work(
            identity=identity,
            parents=(state.pending_parent,),
            stage="evaluate",
            sample=sample,
            candidate=state.pending_proposal.candidate,
        )

    def _expected_work(self, state: PhasedBeamState) -> Work:
        if state.phase == "evaluate":
            return self._evaluation_work(state)
        if state.phase == "propose":
            return self._proposal_work(state)
        raise ValueError("finished policy has no current work")

    def start(self, initial: tuple[Scored, ...]) -> PhasedBeamState:
        if not initial:
            raise ValueError("PhasedBeamSearchPolicy.start requires initial Scored values")
        if any(not isinstance(value, Scored) for value in initial):
            raise TypeError("initial must contain Scored values")
        if len(initial) > self.max_evaluations:
            raise ValueError("initial roots exceed the evaluation budget")
        if len({value.candidate.identity.digest for value in initial}) != len(initial):
            raise ValueError("initial roots must have distinct candidate identities")
        frontier = self._select(initial)
        seen = tuple(
            dict.fromkeys(
                self._projection_token(value.candidate) for value in initial
            )
        )
        slot_count = len(frontier) * self.proposals_per_parent
        finished = (
            len(initial) >= self.max_evaluations
            or not frontier
            or slot_count == 0
        )
        return PhasedBeamState(
            schema=_STATE_SCHEMA,
            policy=self.identity,
            protocol=self.protocol,
            generation=1,
            root_evaluations=len(initial),
            evaluations=len(initial),
            proposals=0,
            duplicates=0,
            proposal_cap=self.proposal_cap,
            max_evaluations=self.max_evaluations,
            frontier=frontier,
            generation_parents=frontier,
            admission_ordinals=tuple(
                (value.candidate.identity.digest, ordinal)
                for ordinal, value in enumerate(initial)
            ),
            next_slot=0,
            slot_count=slot_count,
            pending_slot=None,
            pending_proposal_number=None,
            pending_parent=None,
            pending_proposal=None,
            currentgen_children=(),
            child_admission_slots=(),
            seen_candidates=seen,
            told_work=(),
            proposal_status_counts=tuple((status, 0) for status in _STATUSES),
            evaluation_status_counts=tuple((status, 0) for status in _STATUSES),
            finished=finished,
        )

    def _validate_state(self, state: PhasedBeamState) -> None:
        if not isinstance(state, PhasedBeamState):
            raise TypeError("PhasedBeamSearchPolicy state has the wrong type")
        if state.schema != _STATE_SCHEMA or state.policy != self.identity:
            raise ValueError("phased beam checkpoint belongs to another policy")
        if (
            state.proposal_cap != self.proposal_cap
            or state.max_evaluations != self.max_evaluations
        ):
            raise ValueError("phased beam checkpoint changed its configured budgets")
        if not 0 <= state.next_slot <= state.slot_count:
            raise ValueError("phased beam checkpoint has an invalid next slot")
        pending = (
            state.pending_slot,
            state.pending_proposal_number,
            state.pending_parent,
            state.pending_proposal,
        )
        if any(value is None for value in pending) and any(
            value is not None for value in pending
        ):
            raise ValueError("phased beam checkpoint has a partial pending proposal")
        if len(state.currentgen_children) != len(state.child_admission_slots):
            raise ValueError("phased beam child admission slots are inconsistent")
        admission_digests = tuple(
            digest for digest, _ in state.admission_ordinals
        )
        admission_values = tuple(
            ordinal for _, ordinal in state.admission_ordinals
        )
        if len(set(admission_digests)) != len(admission_digests):
            raise ValueError("phased beam candidate admissions must be unique")
        if admission_values != tuple(range(len(state.admission_ordinals))):
            raise ValueError("phased beam admission ordinals are not contiguous")

    def ask(
        self,
        state: PhasedBeamState,
        history: History,
        capacity: int,
    ) -> tuple[Work, ...]:
        self._validate_state(state)
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        if self.finished(state):
            return ()
        if state.phase == "propose" and state.proposals >= state.proposal_cap:
            raise RuntimeError(
                "proposal attempt cap reached before the next proposal could start"
            )
        return (self._expected_work(state),)

    def _status_counts(
        self,
        current: tuple[tuple[str, int], ...],
        status: str,
    ) -> tuple[tuple[str, int], ...]:
        counts = dict(current)
        counts[status] += 1
        return tuple((name, counts[name]) for name in _STATUSES)

    def _validate_proposal(
        self,
        proposal: Proposal,
        parent: Scored,
    ) -> None:
        if not isinstance(proposal, Proposal):
            raise TypeError("completed proposal work must contain one Proposal")
        if proposal.parents != (parent.candidate.identity,):
            raise ValueError("proposal parents must be the work's frozen parent")
        if proposal.candidate.system != parent.candidate.system:
            raise ValueError("proposal candidate belongs to another System")

    def _complete_generation(
        self,
        state: PhasedBeamState,
        *,
        exhausted: bool,
    ) -> PhasedBeamState:
        ordered_children = tuple(
            child
            for _, child in sorted(
                zip(state.child_admission_slots, state.currentgen_children),
                key=lambda item: item[0],
            )
        )
        frontier = self._select((*state.generation_parents, *ordered_children))
        if exhausted or not frontier:
            return replace(
                state,
                frontier=frontier,
                pending_slot=None,
                pending_proposal_number=None,
                pending_parent=None,
                pending_proposal=None,
                finished=True,
            )
        generation = state.generation + 1
        slot_count = len(frontier) * self.proposals_per_parent
        return replace(
            state,
            generation=generation,
            frontier=frontier,
            generation_parents=frontier,
            next_slot=0,
            slot_count=slot_count,
            pending_slot=None,
            pending_proposal_number=None,
            pending_parent=None,
            pending_proposal=None,
            currentgen_children=(),
            child_admission_slots=(),
            finished=slot_count == 0,
        )

    def _tell_proposal(
        self,
        state: PhasedBeamState,
        work: Work,
        outcome: TrialOutcome,
    ) -> PhasedBeamState:
        if outcome.children:
            raise ValueError("proposal outcomes cannot contain Scored children")
        proposals = state.proposals + 1
        next_slot = state.next_slot + 1
        told = state.told_work + (outcome.work,)
        counts = self._status_counts(
            state.proposal_status_counts, outcome.status
        )
        base = replace(
            state,
            proposals=proposals,
            next_slot=next_slot,
            told_work=told,
            proposal_status_counts=counts,
        )
        if outcome.status != "completed":
            if outcome.proposals:
                raise ValueError(
                    "failed/unknown/refused proposal outcomes cannot contain proposals"
                )
            if next_slot == state.slot_count:
                return self._complete_generation(base, exhausted=False)
            return base

        if len(outcome.proposals) != 1:
            raise ValueError(
                "completed proposal work must return exactly one Proposal"
            )
        parent = work.parents[0]
        proposal = outcome.proposals[0]
        self._validate_proposal(proposal, parent)
        projection = self._projection_token(proposal.candidate)
        if projection in state.seen_candidates:
            duplicate = replace(base, duplicates=state.duplicates + 1)
            if next_slot == state.slot_count:
                return self._complete_generation(duplicate, exhausted=False)
            return duplicate
        return replace(
            base,
            pending_slot=state.next_slot,
            pending_proposal_number=proposals,
            pending_parent=parent,
            pending_proposal=proposal,
            seen_candidates=state.seen_candidates + (projection,),
        )

    def _tell_evaluation(
        self,
        state: PhasedBeamState,
        work: Work,
        outcome: TrialOutcome,
    ) -> PhasedBeamState:
        if outcome.proposals:
            raise ValueError("evaluation outcomes cannot contain Proposals")
        told = state.told_work + (outcome.work,)
        counts = self._status_counts(
            state.evaluation_status_counts, outcome.status
        )
        completed = outcome.status == "completed"
        if completed:
            if len(outcome.children) != 1:
                raise ValueError(
                    "completed evaluation work must return exactly one Scored child"
                )
            child = outcome.children[0]
            if child.candidate != work.candidate:
                raise ValueError("evaluation child changed the accepted proposal candidate")
        else:
            if outcome.children:
                raise ValueError(
                    "failed/unknown/refused evaluation outcomes cannot contain children"
                )
            child = None

        evaluations = state.evaluations + (1 if completed else 0)
        admission_ordinals = (
            state.admission_ordinals
            + ((child.candidate.identity.digest, state.evaluations),)
            if child is not None
            else state.admission_ordinals
        )
        children = (
            state.currentgen_children + (child,)
            if child is not None
            else state.currentgen_children
        )
        slots = (
            state.child_admission_slots + (state.pending_slot,)
            if child is not None
            else state.child_admission_slots
        )
        base = replace(
            state,
            evaluations=evaluations,
            pending_slot=None,
            pending_proposal_number=None,
            pending_parent=None,
            pending_proposal=None,
            currentgen_children=children,
            child_admission_slots=slots,
            admission_ordinals=admission_ordinals,
            told_work=told,
            evaluation_status_counts=counts,
        )
        exhausted = evaluations >= state.max_evaluations
        if exhausted or state.next_slot == state.slot_count:
            return self._complete_generation(base, exhausted=exhausted)
        return base

    def tell(
        self,
        state: PhasedBeamState,
        outcomes: tuple[TrialOutcome, ...],
    ) -> PhasedBeamState:
        self._validate_state(state)
        if len(outcomes) != 1:
            raise ValueError("phased beam policy accepts exactly one outcome per tell")
        outcome = outcomes[0]
        if outcome.status not in _STATUSES:
            raise ValueError(f"unknown TrialOutcome status {outcome.status!r}")
        if outcome.work in state.told_work:
            raise ValueError(f"tell received work {outcome.work!r} more than once")
        if self.finished(state):
            raise ValueError("tell received work after the policy finished")
        work = self._expected_work(state)
        if outcome.work != work.identity:
            raise ValueError("tell received work outside the current proposal/evaluation phase")
        if work.stage == "propose":
            return self._tell_proposal(state, work, outcome)
        return self._tell_evaluation(state, work, outcome)

    def finished(self, state: PhasedBeamState) -> bool:
        self._validate_state(state)
        return state.finished

    def selected(self, state: PhasedBeamState) -> tuple[Scored, ...]:
        self._validate_state(state)
        return state.frontier

    def snapshot(self, state: PhasedBeamState) -> PhasedBeamState:
        """Validate and return the immutable state for durable checkpointing."""

        self._validate_state(state)
        return state


def phased_beam_policy(**kwargs) -> PhasedBeamSearchPolicy:
    return PhasedBeamSearchPolicy(**kwargs)


__all__ = [
    "PHASED_BEAM_STATE_CODEC",
    "PhasedBeamSearchPolicy",
    "PhasedBeamState",
    "PhasedBeamStateCodec",
    "phased_beam_policy",
]
