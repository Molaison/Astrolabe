"""Deterministic search policies over typed design work.

The policy owns parent retention, stable work identities, replicate derivation,
work accounting, candidate deduplication and lexical ranking. It neither runs
an operation nor enforces prediction budgets. Runtime's paid boundaries are the
authority for actual model/tool calls; ``max_work`` only caps Trial work items.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import hashlib
import inspect
import math
from typing import Literal

from .store import DataclassCodec
from .types import (
    History,
    Key,
    Sample,
    Scored,
    TrialOutcome,
    Work,
    key_for,
)


ExpansionMode = Literal["each", "round_robin"]
Direction = Literal["max", "min", 1, -1]
TieMode = Literal["stable_input_order", "candidate_identity"]
RankCallable = Callable[[Scored], Sequence[float]]

_STATE_SCHEMA = "astrolabe.design.search.PolicyState/v1"
_WORK_SCHEMA = "astrolabe.design.search.Work/v1"
_CONTROL_SCHEMA = "astrolabe.design.search.Control/v1"
_STATUSES = ("completed", "failed", "unknown", "refused")


@dataclass(frozen=True)
class PolicyState:
    """Serializable, immutable search state consumed by the coordinator."""

    schema: str
    policy: Key
    round: int
    used: int
    max_work: int
    round_target: int
    frontier: tuple[Scored, ...]
    round_children: tuple[Scored, ...]
    seen_candidates: tuple[str, ...]
    told_work: tuple[str, ...]
    status_counts: tuple[tuple[str, int], ...]
    deduplicated_children: int
    gated_children: int


class PolicyStateCodec:
    """Versioned checkpoint codec for ``PolicyState`` and nested typed values."""

    identity = _STATE_SCHEMA

    def __init__(self) -> None:
        self._codec = DataclassCodec(PolicyState, self.identity)

    def encode(self, value: PolicyState) -> bytes:
        if not isinstance(value, PolicyState):
            raise TypeError("PolicyStateCodec expects PolicyState")
        return self._codec.encode(value)

    def decode(self, data: bytes) -> PolicyState:
        value = self._codec.decode(data)
        if not isinstance(value, PolicyState):
            raise TypeError("PolicyStateCodec decoded another type")
        return value

    def references(self, value: PolicyState):
        return self._codec.references(value)


POLICY_STATE_CODEC = PolicyStateCodec()


def _positive_int(name: str, value: object) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(name: str, value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _candidate_identity(value: Scored) -> str:
    identity = value.candidate.identity
    digest = getattr(identity, "digest", None)
    if not isinstance(digest, str) or not digest:
        raise TypeError("Candidate identity must expose a non-empty digest")
    return digest


def _callable_identity(rank: RankCallable | None, explicit: str | None) -> str:
    if rank is None:
        if explicit is not None:
            raise ValueError("rank_identity does not apply to the default rank")
        return "assessment.values/v1"
    if explicit is not None:
        if not isinstance(explicit, str) or not explicit:
            raise ValueError("rank_identity must be a non-empty string")
        return explicit
    declared = getattr(rank, "identity", None)
    if isinstance(declared, str) and declared:
        return declared
    closure = inspect.getclosurevars(rank)
    defaults = tuple(inspect.signature(rank).parameters.values())
    captures_defaults = any(
        parameter.default is not inspect.Parameter.empty
        and not callable(parameter.default)
        for parameter in defaults[1:]
    )
    if closure.nonlocals or vars(rank) or captures_defaults:
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


def _normalize_directions(directions: Sequence[Direction]) -> tuple[int, ...]:
    if not directions:
        raise ValueError("directions must contain at least one maximize/minimize value")
    normalized: list[int] = []
    for value in directions:
        if value in (1, "max"):
            normalized.append(1)
        elif value in (-1, "min"):
            normalized.append(-1)
        else:
            raise ValueError("directions allow only 'max'/'min' or 1/-1")
    return tuple(normalized)


class CommonSearchPolicy:
    """Shared trajectory, beam and population mechanics.

    ``rounds`` counts expansion rounds after the supplied root. Work IDs are
    stable for ``(policy, round, parent, offspring slot)``.
    """

    kind = "common"
    codec = POLICY_STATE_CODEC

    def __init__(
        self,
        *,
        rounds: int,
        width: int,
        offspring: int,
        max_work: int,
        seed: int,
        directions: Sequence[Direction],
        rank: RankCallable | None = None,
        rank_identity: str | None = None,
        expand_parents: ExpansionMode = "each",
        tie_mode: TieMode = "stable_input_order",
        retain_parents: bool = True,
        root_counts_budget: bool = True,
        parents_per_round: int | None = None,
    ) -> None:
        self.rounds = _nonnegative_int("rounds", rounds)
        self.width = _positive_int("width", width)
        self.offspring = _positive_int("offspring", offspring)
        self.max_work = _positive_int("max_work", max_work)
        if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise ValueError("seed must be an integer in 0..2**32-1")
        self.seed = seed
        if expand_parents not in ("each", "round_robin"):
            raise ValueError("expand_parents must be 'each' or 'round_robin'")
        if tie_mode not in ("stable_input_order", "candidate_identity"):
            raise ValueError("invalid tie_mode")
        if type(retain_parents) is not bool or type(root_counts_budget) is not bool:
            raise ValueError("retain_parents and root_counts_budget must be booleans")
        if parents_per_round is not None:
            parents_per_round = _positive_int("parents_per_round", parents_per_round)
        self.expand_parents = expand_parents
        self.tie_mode = tie_mode
        self.retain_parents = retain_parents
        self.root_counts_budget = root_counts_budget
        self.parents_per_round = parents_per_round
        self.directions = _normalize_directions(directions)
        self.rank = rank
        self.rank_identity = _callable_identity(rank, rank_identity)
        self.identity = key_for(
            (
                _CONTROL_SCHEMA,
                self.kind,
                self.rounds,
                self.width,
                self.offspring,
                self.max_work,
                self.seed,
                self.directions,
                self.rank_identity,
                self.expand_parents,
                self.tie_mode,
                self.retain_parents,
                self.root_counts_budget,
                self.parents_per_round,
                self.codec.identity,
                inspect.getsource(CommonSearchPolicy),
                inspect.getsource(type(self)),
            )
        )

    def _rank_values(self, scored: Scored) -> tuple[float, ...]:
        raw = scored.assessment.values if self.rank is None else self.rank(scored)
        values = tuple(float(value) for value in raw)
        if len(values) != len(self.directions):
            raise ValueError(
                f"rank returned {len(values)} values for {len(self.directions)} directions"
            )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("rank values must be finite")
        return values

    def _gate_passed(self, scored: Scored) -> bool:
        return all(bool(value) for _, value in scored.assessment.gates)

    def _rank_key(self, scored: Scored) -> tuple[object, ...]:
        gate_rank = 0 if self._gate_passed(scored) else 1
        values = self._rank_values(scored)
        oriented = tuple(
            -direction * value
            for direction, value in zip(self.directions, values)
        )
        if self.tie_mode == "candidate_identity":
            return gate_rank, oriented, _candidate_identity(scored)
        return gate_rank, oriented

    def _deduplicate(self, values: Sequence[Scored]) -> tuple[Scored, ...]:
        seen: set[str] = set()
        result: list[Scored] = []
        for value in values:
            identity = _candidate_identity(value)
            if identity in seen:
                continue
            seen.add(identity)
            result.append(value)
        return tuple(result)

    def _select(self, values: Sequence[Scored]) -> tuple[Scored, ...]:
        unique = self._deduplicate(values)
        eligible = tuple(value for value in unique if self._gate_passed(value))
        return tuple(sorted(eligible, key=self._rank_key)[: self.width])

    def _parents(self, state: PolicyState) -> tuple[Scored, ...]:
        ordered = sorted(state.frontier, key=self._rank_key)
        if self.parents_per_round is not None:
            ordered = ordered[: self.parents_per_round]
        return tuple(ordered)

    def _sample(self, state: PolicyState, parent: Scored, slot: int) -> Sample:
        replicate = f"{self.kind}/round-{state.round}/slot-{slot}"
        seed_key = key_for(
            (
                _WORK_SCHEMA,
                self.identity,
                state.round,
                slot,
                _candidate_identity(parent),
            )
        )
        seed = int(seed_key.digest[:16], 16) % (2**31 - 1) or 1
        return Sample(
            protocol=self.identity,
            replicate=replicate,
            seed=seed,
            index=str(state.round * self.width * self.offspring + slot + 1),
        )

    def _works(self, state: PolicyState) -> tuple[Work, ...]:
        parents = self._parents(state)
        if not parents or state.round_target == 0:
            return ()
        pairs: list[tuple[Scored, int]]
        if self.expand_parents == "round_robin":
            pairs = [
                (parent, slot)
                for slot in range(self.offspring)
                for parent in parents
            ]
        else:
            pairs = [
                (parent, slot)
                for parent in parents
                for slot in range(self.offspring)
            ]
        works: list[Work] = []
        for slot, (parent, _) in enumerate(pairs[: state.round_target]):
            sample = self._sample(state, parent, slot)
            identity = key_for(
                (
                    _WORK_SCHEMA,
                    self.identity,
                    state.round,
                    slot,
                    _candidate_identity(parent),
                    sample,
                )
            ).digest
            works.append(
                Work(
                    identity=identity,
                    parents=(parent,),
                    stage="search",
                    sample=sample,
                )
            )
        return tuple(works)

    def start(self, initial: tuple[Scored, ...]) -> PolicyState:
        if not initial:
            raise ValueError("SearchPolicy.start requires at least one initial Scored")
        if self.root_counts_budget and len(initial) > self.max_work:
            raise ValueError("Initial roots exceed the search work budget")
        frontier = self._select(initial)
        seen = tuple(dict.fromkeys(_candidate_identity(value) for value in initial))
        used = len(initial) if self.root_counts_budget else 0
        target = (
            min(len(frontier) * self.offspring, self.max_work - used)
            if self.rounds and frontier
            else 0
        )
        return PolicyState(
            schema=_STATE_SCHEMA,
            policy=self.identity,
            round=0,
            used=used,
            max_work=self.max_work,
            round_target=target,
            frontier=frontier,
            round_children=(),
            seen_candidates=seen,
            told_work=(),
            status_counts=tuple((status, 0) for status in _STATUSES),
            deduplicated_children=0,
            gated_children=0,
        )

    def _validate_state(self, state: PolicyState) -> None:
        if not isinstance(state, PolicyState):
            raise TypeError("SearchPolicy state has the wrong type")
        if state.schema != _STATE_SCHEMA or state.policy != self.identity:
            raise ValueError("SearchPolicy checkpoint belongs to another policy")
        if state.max_work != self.max_work:
            raise ValueError("SearchPolicy checkpoint changed its work budget")

    def ask(
        self,
        state: PolicyState,
        history: History,
        capacity: int,
    ) -> tuple[Work, ...]:
        self._validate_state(state)
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        if self.finished(state):
            return ()
        told = set(state.told_work)
        outstanding = tuple(
            work for work in self._works(state) if work.identity not in told
        )
        return outstanding[:capacity]

    def tell(
        self,
        state: PolicyState,
        outcomes: tuple[TrialOutcome, ...],
    ) -> PolicyState:
        self._validate_state(state)
        if not outcomes:
            raise ValueError("tell requires at least one TrialOutcome")
        if len({outcome.work for outcome in outcomes}) != len(outcomes):
            raise ValueError("tell received duplicate work outcomes")
        expected = {
            work.identity
            for work in self._expected_works(state)
        }
        told_before = set(state.told_work)
        for outcome in outcomes:
            if outcome.work not in expected:
                raise ValueError(f"tell received work outside round {state.round}")
            if outcome.work in told_before:
                raise ValueError(f"tell received work {outcome.work!r} more than once")
            if outcome.status not in _STATUSES:
                raise ValueError(f"unknown TrialOutcome status {outcome.status!r}")
            if outcome.status != "completed" and outcome.children:
                raise ValueError("failed/unknown/refused outcomes cannot contain children")
        used = state.used + len(outcomes)
        if used > self.max_work:
            raise ValueError("tell exceeds the configured work budget")

        status_counts = dict(state.status_counts)
        for outcome in outcomes:
            status_counts[outcome.status] += 1

        seen = list(state.seen_candidates)
        seen_set = set(seen)
        round_children = list(state.round_children)
        deduplicated = state.deduplicated_children
        gated = state.gated_children
        for outcome in outcomes:
            for child in outcome.children:
                identity = _candidate_identity(child)
                if identity in seen_set:
                    deduplicated += 1
                    continue
                seen.append(identity)
                seen_set.add(identity)
                if self._gate_passed(child):
                    round_children.append(child)
                else:
                    gated += 1

        told = state.told_work + tuple(outcome.work for outcome in outcomes)
        round_complete = expected.issubset(told)
        round_index = state.round
        target = state.round_target
        frontier = state.frontier
        children = tuple(round_children)
        if round_complete:
            pool: list[Scored] = []
            if self.retain_parents:
                pool.extend(state.frontier)
            pool.extend(children)
            frontier = self._select(pool)
            round_index += 1
            children = ()
            target = (
                min(len(frontier) * self.offspring, self.max_work - used)
                if round_index < self.rounds and frontier
                else 0
            )
        return PolicyState(
            schema=state.schema,
            policy=state.policy,
            round=round_index,
            used=used,
            max_work=state.max_work,
            round_target=target,
            frontier=frontier,
            round_children=children,
            seen_candidates=tuple(seen),
            told_work=told,
            status_counts=tuple((status, status_counts[status]) for status in _STATUSES),
            deduplicated_children=deduplicated,
            gated_children=gated,
        )

    def finished(self, state: PolicyState) -> bool:
        self._validate_state(state)
        return (
            state.round >= self.rounds
            or state.used >= state.max_work
            or not state.frontier
            or state.round_target == 0
        )

    def selected(self, state: PolicyState) -> tuple[Scored, ...]:
        self._validate_state(state)
        return state.frontier

    def _expected_works(self, state: PolicyState) -> tuple[Work, ...]:
        """Rebuild the current round's frozen work IDs without mutable history."""

        if state.round_target == 0:
            return ()
        parents = self._parents(state)
        if not parents:
            return ()
        pairs: list[tuple[Scored, int]]
        if self.expand_parents == "round_robin":
            pairs = [
                (parent, slot)
                for slot in range(self.offspring)
                for parent in parents
            ]
        else:
            pairs = [
                (parent, slot)
                for parent in parents
                for slot in range(self.offspring)
            ]
        works: list[Work] = []
        for slot, (parent, _) in enumerate(pairs[: state.round_target]):
            sample = Sample(
                protocol=self.identity,
                replicate=f"{self.kind}/round-{state.round}/slot-{slot}",
                seed=self._sample(state, parent, slot).seed,
                index=str(state.round * self.width * self.offspring + slot + 1),
            )
            works.append(
                Work(
                    identity=key_for(
                        (
                            _WORK_SCHEMA,
                            self.identity,
                            state.round,
                            slot,
                            _candidate_identity(parent),
                            sample,
                        )
                    ).digest,
                    parents=(parent,),
                    stage="search",
                    sample=sample,
                )
            )
        return tuple(works)


class TrajectorySearchPolicy(CommonSearchPolicy):
    kind = "trajectory"

    def __init__(self, *, width: int = 1, **kwargs) -> None:
        if width != 1:
            raise ValueError("trajectory search width must be 1")
        super().__init__(width=width, **kwargs)


class BeamSearchPolicy(CommonSearchPolicy):
    kind = "beam"


class PopulationSearchPolicy(CommonSearchPolicy):
    kind = "population"


def trajectory_policy(**kwargs) -> TrajectorySearchPolicy:
    return TrajectorySearchPolicy(**kwargs)


def beam_policy(**kwargs) -> BeamSearchPolicy:
    return BeamSearchPolicy(**kwargs)


def population_policy(**kwargs) -> PopulationSearchPolicy:
    return PopulationSearchPolicy(**kwargs)


__all__ = [
    "BeamSearchPolicy",
    "CommonSearchPolicy",
    "POLICY_STATE_CODEC",
    "PolicyState",
    "PolicyStateCodec",
    "PopulationSearchPolicy",
    "TrajectorySearchPolicy",
    "beam_policy",
    "population_policy",
    "trajectory_policy",
]
