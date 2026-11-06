"""Accepted-successor trajectory policy.

Scientific acceptance belongs to a typed decision operation. This policy only
continues from the successor that operation accepted, including successors that
rank worse by the policy's own metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect

from .store import DataclassCodec
from .types import (
    Continuation,
    History,
    Key,
    Sample,
    Scored,
    TrialOutcome,
    Work,
    key_for,
)


_STATE_SCHEMA = "astrolabe.design.successor-search.AcceptedSuccessorState/v1"
_WORK_SCHEMA = "astrolabe.design.successor-search.Work/v1"


def _rounds(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("rounds must be a non-negative integer")
    return value


def _seed(value: object) -> int:
    if type(value) is not int or not 0 <= value <= 2**32 - 1:
        raise ValueError("seed must be an integer in 0..2**32-1")
    return value


def _one_scored(label: str, values: tuple[Scored, ...]) -> Scored:
    if len(values) != 1:
        raise ValueError(f"{label} requires exactly one Scored candidate")
    value = values[0]
    if not isinstance(value, Scored):
        raise TypeError(f"{label} requires Scored values")
    return value


@dataclass(frozen=True)
class AcceptedSuccessorState:
    """Immutable checkpoint advanced only by accepted continuations."""

    schema: str
    policy: Key
    rounds: int
    seed: int
    used: int
    frontier: tuple[Scored, ...]
    stop_reason: str | None = None

    @property
    def round(self) -> int:
        return self.used


class AcceptedSuccessorStateCodec:
    identity = _STATE_SCHEMA

    def __init__(self) -> None:
        self._codec = DataclassCodec(AcceptedSuccessorState, self.identity)

    def encode(self, value: AcceptedSuccessorState) -> bytes:
        if not isinstance(value, AcceptedSuccessorState):
            raise TypeError(
                "AcceptedSuccessorStateCodec expects AcceptedSuccessorState"
            )
        return self._codec.encode(value)

    def decode(self, data: bytes) -> AcceptedSuccessorState:
        value = self._codec.decode(data)
        if not isinstance(value, AcceptedSuccessorState):
            raise TypeError(
                "AcceptedSuccessorStateCodec decoded another type"
            )
        return value

    def references(self, value: AcceptedSuccessorState):
        if not isinstance(value, AcceptedSuccessorState):
            raise TypeError(
                "AcceptedSuccessorStateCodec expects AcceptedSuccessorState"
            )
        return self._codec.references(value)


ACCEPTED_SUCCESSOR_STATE_CODEC = AcceptedSuccessorStateCodec()
POLICY_STATE_CODEC = ACCEPTED_SUCCESSOR_STATE_CODEC


class AcceptedSuccessorPolicy:
    """Advance one accepted successor per round without reranking it."""

    kind = "accepted_successor"
    codec = ACCEPTED_SUCCESSOR_STATE_CODEC

    def __init__(self, *, rounds: int, seed: int) -> None:
        self.rounds = _rounds(rounds)
        self.seed = _seed(seed)
        source = inspect.getsource(type(self))
        self.identity = key_for(
            (
                "accepted-successor-control/v1",
                type(self).__module__,
                type(self).__qualname__,
                source,
                self.rounds,
                self.seed,
            )
        )

    def _validate_state(self, state: AcceptedSuccessorState) -> None:
        if not isinstance(state, AcceptedSuccessorState):
            raise TypeError("AcceptedSuccessorPolicy state has the wrong type")
        if state.schema != _STATE_SCHEMA or state.policy != self.identity:
            raise ValueError(
                "AcceptedSuccessorPolicy checkpoint belongs to another policy"
            )
        if state.rounds != self.rounds or state.seed != self.seed:
            raise ValueError(
                "AcceptedSuccessorPolicy checkpoint changed rounds or seed"
            )
        if type(state.used) is not int or not 0 <= state.used <= self.rounds:
            raise ValueError("AcceptedSuccessorPolicy checkpoint has invalid usage")
        _one_scored("AcceptedSuccessorPolicy checkpoint", state.frontier)
        if state.stop_reason is not None and not isinstance(state.stop_reason, str):
            raise TypeError("stop_reason must be a string or None")

    def _work(self, state: AcceptedSuccessorState) -> Work:
        self._validate_state(state)
        step = str(state.used + 1)
        sample = Sample(
            protocol=self.identity,
            replicate=step,
            seed=self.seed,
            index=step,
        )
        parent = state.frontier[0]
        return Work(
            identity=key_for(
                (
                    _WORK_SCHEMA,
                    self.identity,
                    state.used + 1,
                    parent,
                    sample,
                )
            ).digest,
            parents=(parent,),
            stage="advance",
            sample=sample,
        )

    def start(self, initial: tuple[Scored, ...]) -> AcceptedSuccessorState:
        roots = tuple(initial)
        _one_scored("AcceptedSuccessorPolicy.start", roots)
        return AcceptedSuccessorState(
            schema=_STATE_SCHEMA,
            policy=self.identity,
            rounds=self.rounds,
            seed=self.seed,
            used=0,
            frontier=roots,
            stop_reason=None,
        )

    def ask(
        self,
        state: AcceptedSuccessorState,
        history: History,
        capacity: int,
    ) -> tuple[Work, ...]:
        self._validate_state(state)
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        if self.finished(state):
            return ()
        return (self._work(state),)

    def tell(
        self,
        state: AcceptedSuccessorState,
        outcomes: tuple[TrialOutcome, ...],
    ) -> AcceptedSuccessorState:
        self._validate_state(state)
        outcomes = tuple(outcomes)
        if len(outcomes) != 1:
            raise ValueError("AcceptedSuccessorPolicy.tell requires exactly one outcome")
        outcome = outcomes[0]
        if not isinstance(outcome, TrialOutcome):
            raise TypeError("outcomes must contain TrialOutcome values")
        if outcome.status != "completed":
            raise ValueError("AcceptedSuccessorPolicy requires a completed outcome")
        continuation = outcome.continuation
        if not isinstance(continuation, Continuation):
            raise ValueError("completed outcome must include a Continuation")
        if not isinstance(continuation.successor, Scored):
            raise TypeError("Continuation.successor must be Scored")
        if self.finished(state):
            raise ValueError("Received an outcome after the policy finished")
        expected = self._work(state)
        if outcome.work != expected.identity:
            raise ValueError(
                "outcome work is out of order or was already consumed"
            )
        return AcceptedSuccessorState(
            schema=_STATE_SCHEMA,
            policy=self.identity,
            rounds=self.rounds,
            seed=self.seed,
            used=state.used + 1,
            frontier=(continuation.successor,),
            stop_reason=continuation.stop_reason,
        )

    def finished(self, state: AcceptedSuccessorState) -> bool:
        self._validate_state(state)
        return state.stop_reason is not None or state.used >= self.rounds

    def selected(self, state: AcceptedSuccessorState) -> tuple[Scored, ...]:
        self._validate_state(state)
        return state.frontier


def accepted_successor_policy(**kwargs) -> AcceptedSuccessorPolicy:
    return AcceptedSuccessorPolicy(**kwargs)


__all__ = [
    "ACCEPTED_SUCCESSOR_STATE_CODEC",
    "AcceptedSuccessorPolicy",
    "AcceptedSuccessorState",
    "AcceptedSuccessorStateCodec",
    "POLICY_STATE_CODEC",
    "accepted_successor_policy",
]
