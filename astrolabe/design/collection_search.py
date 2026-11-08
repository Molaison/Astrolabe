"""Typed policy/trial composition for generation-wise completion collection."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import logging
from typing import Any, Callable, Literal

from .collection import (
    AdmissionHook,
    CollectionConfig,
    CollectionCommitHook,
    CollectionExecutionError,
    CompletionPool,
    ExecutorFactory,
    Job,
    JobResult,
    RngSnapshotter,
    ResultCollection,
    ResultSelector,
    TerminalObservedHook,
    select_highest_score,
)
from .types import key_for


Phase = Literal["submit", "collect", "drain", "finished"]
ActionKind = Literal["submit", "collect", "drain"]
JobIdFactory = Callable[[int], str]
PayloadFactory = Callable[[Any], Any]
SelectedAdopter = Callable[[Any, JobResult], Any]


def _default_job_id(number: int) -> str:
    return f"job_{int(number):04d}"


def _default_payload(payload: Any) -> Any:
    return deepcopy(payload)


def _keep_payload(payload: Any, _selected: JobResult) -> Any:
    return payload


def _callable_identity(value: Callable[..., Any]) -> str:
    module = getattr(value, "__module__", type(value).__module__)
    qualname = getattr(value, "__qualname__", type(value).__qualname__)
    return f"{module}:{qualname}"


def collection_action_id(action: CollectionAction) -> str:
    """Return a stable logical id that does not depend on object identity."""

    record = {
        "kind": action.kind,
        "generation": action.generation,
        "jobs": [
            {
                "job_id": job.job_id,
                "generation": job.generation,
            }
            for job in action.jobs
        ],
    }
    payload = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return "collection-" + hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class CollectionState:
    """Pure generation/selection state for one collection search."""

    phase: Phase
    generation: int
    decision_rounds: int
    next_job_number: int
    current_payload: Any
    collections: tuple[ResultCollection, ...]
    decision_collection_count: int


@dataclass(frozen=True)
class CollectionAction:
    """Typed request from policy state to the CompletionPool trial."""

    kind: ActionKind
    generation: int | None = None
    jobs: tuple[Job, ...] = ()


@dataclass(frozen=True)
class CollectionEvent:
    """Typed completion returned by the CollectionTrial."""

    action: CollectionAction
    collections: tuple[ResultCollection, ...] = ()


@dataclass(frozen=True)
class CollectionRun:
    """Selected result after decision generations and explicit drain."""

    collections: tuple[ResultCollection, ...]
    decision_collection_count: int

    @property
    def decision_collections(self) -> tuple[ResultCollection, ...]:
        return self.collections[: self.decision_collection_count]

    @property
    def drain_collections(self) -> tuple[ResultCollection, ...]:
        return self.collections[self.decision_collection_count :]


class CollectionPolicy:
    """Own generation count, job numbering, selection adoption and drain phase."""

    request_type = CollectionAction
    result_type = CollectionEvent

    def __init__(
        self,
        *,
        config: CollectionConfig,
        decision_rounds: int,
        initial_payload: Any = None,
        selector: ResultSelector | None = None,
        job_id_factory: JobIdFactory | None = None,
        payload_factory: PayloadFactory | None = None,
        adopt_selected: SelectedAdopter | None = None,
    ) -> None:
        self.config = config
        self.decision_rounds = int(decision_rounds)
        self.initial_payload = deepcopy(initial_payload)
        self.selector = selector or select_highest_score
        self.job_id_factory = job_id_factory or _default_job_id
        self.payload_factory = payload_factory or _default_payload
        self.adopt_selected = adopt_selected or _keep_payload
        self.identity = key_for(
            (
                "astrolabe.collection-policy/v1",
                self.config,
                self.decision_rounds,
                _callable_identity(self.selector),
                _callable_identity(self.job_id_factory),
                _callable_identity(self.payload_factory),
                _callable_identity(self.adopt_selected),
                self.initial_payload,
            )
        )

    def initial_state(self) -> CollectionState:
        return CollectionState(
            phase="submit",
            generation=1,
            decision_rounds=self.decision_rounds,
            next_job_number=1,
            current_payload=deepcopy(self.initial_payload),
            collections=(),
            decision_collection_count=0,
        )

    def _submit_action(self, state: CollectionState) -> CollectionAction:
        jobs = tuple(
            Job(
                job_id=self.job_id_factory(number),
                generation=state.generation,
                payload=self.payload_factory(state.current_payload),
            )
            for number in range(
                state.next_job_number,
                state.next_job_number + self.config.num_jobs,
            )
        )
        return CollectionAction(
            kind="submit",
            generation=state.generation,
            jobs=jobs,
        )

    def ask(self, state: CollectionState) -> CollectionAction:
        self._validate_state(state)
        if state.phase == "submit":
            return self._submit_action(state)
        if state.phase == "collect":
            return CollectionAction(kind="collect", generation=state.generation)
        if state.phase == "drain":
            return CollectionAction(kind="drain")
        raise ValueError("finished CollectionPolicy has no action")

    def tell(
        self,
        state: CollectionState,
        event: CollectionEvent,
    ) -> CollectionState:
        self._validate_state(state)
        if not isinstance(event, CollectionEvent):
            raise TypeError("CollectionPolicy.tell requires CollectionEvent")
        expected = self.ask(state)
        if event.action != expected:
            raise ValueError("CollectionEvent does not match the current action")

        if state.phase == "submit":
            if event.collections:
                raise ValueError("submit event cannot contain collections")
            next_number = state.next_job_number + len(event.action.jobs)
            next_phase: Phase = (
                "collect" if state.generation <= state.decision_rounds else "drain"
            )
            return CollectionState(
                phase=next_phase,
                generation=state.generation,
                decision_rounds=state.decision_rounds,
                next_job_number=next_number,
                current_payload=state.current_payload,
                collections=state.collections,
                decision_collection_count=state.decision_collection_count,
            )

        if state.phase == "collect":
            if not event.collections:
                return CollectionState(
                    phase="drain",
                    generation=state.generation,
                    decision_rounds=state.decision_rounds,
                    next_job_number=state.next_job_number,
                    current_payload=state.current_payload,
                    collections=state.collections,
                    decision_collection_count=state.decision_collection_count,
                )
            if len(event.collections) != 1:
                raise ValueError(
                    "collect event must contain exactly one ResultCollection"
                )
            collection = event.collections[0]
            if collection.selected is None:
                raise CollectionExecutionError(
                    "No successful candidate was available for decision "
                    f"round {state.generation}; failed={len(collection.failed)}"
                )
            payload = self.adopt_selected(state.current_payload, collection.selected)
            generation = state.generation + 1
            next_phase = (
                "submit" if state.generation < state.decision_rounds else "drain"
            )
            return CollectionState(
                phase=next_phase,
                generation=generation,
                decision_rounds=state.decision_rounds,
                next_job_number=state.next_job_number,
                current_payload=payload,
                collections=state.collections + (collection,),
                decision_collection_count=state.decision_collection_count + 1,
            )

        if state.phase == "drain":
            return CollectionState(
                phase="finished",
                generation=state.generation,
                decision_rounds=state.decision_rounds,
                next_job_number=state.next_job_number,
                current_payload=state.current_payload,
                collections=state.collections + event.collections,
                decision_collection_count=state.decision_collection_count,
            )
        raise ValueError("cannot tell a finished CollectionPolicy")

    def finished(self, state: CollectionState) -> bool:
        self._validate_state(state)
        return state.phase == "finished"

    def selected(self, state: CollectionState) -> CollectionRun:
        self._validate_state(state)
        return CollectionRun(
            collections=state.collections,
            decision_collection_count=state.decision_collection_count,
        )

    def _validate_state(self, state: CollectionState) -> None:
        if not isinstance(state, CollectionState):
            raise TypeError("CollectionPolicy state must be CollectionState")
        if state.phase not in {"submit", "collect", "drain", "finished"}:
            raise ValueError("invalid CollectionState phase")
        if state.decision_rounds != self.decision_rounds:
            raise ValueError("CollectionState changed decision_rounds")
        if state.next_job_number < 1:
            raise ValueError("CollectionState next_job_number must be positive")
        if state.decision_collection_count > len(state.collections):
            raise ValueError("decision collection count exceeds collections")


class CollectionTrial:
    """Translate typed collection actions to one live CompletionPool."""

    fail_fast = True

    def __init__(
        self,
        *,
        config: CollectionConfig,
        worker: Callable[[Job], JobResult | float],
        selector: ResultSelector | None = None,
        logger: logging.Logger | None = None,
        admission: AdmissionHook | None = None,
        terminal_observed: TerminalObservedHook | None = None,
        collection_commit: CollectionCommitHook | None = None,
        rng_snapshot: RngSnapshotter | None = None,
        executor_factory: ExecutorFactory | None = None,
    ) -> None:
        self.pool = CompletionPool(
            config,
            worker,
            selector=selector,
            logger=logger,
            admission=admission,
            terminal_observed=terminal_observed,
            collection_commit=collection_commit,
            rng_snapshot=rng_snapshot,
            executor_factory=executor_factory,
        )
        self._closed = False
        self.identity = key_for(
            (
                "astrolabe.collection-trial/v1",
                config,
                _callable_identity(self.pool.selector),
                f"{type(worker).__module__}:{type(worker).__qualname__}",
            )
        )

    def perform(self, action: CollectionAction) -> CollectionEvent:
        if self._closed:
            raise RuntimeError("CollectionTrial is closed")
        if not isinstance(action, CollectionAction):
            raise TypeError("CollectionTrial requires CollectionAction")
        if action.kind == "submit":
            if action.generation is None or not action.jobs:
                raise ValueError("submit action requires generation and jobs")
            self.pool.submit_generation(action.generation, action.jobs)
            return CollectionEvent(action=action)
        if action.kind == "collect":
            if action.generation is None:
                raise ValueError("collect action requires generation")
            collection = self.pool.collect_until_ready(
                target_generation=action.generation,
                action_id=collection_action_id(action),
            )
            return CollectionEvent(
                action=action,
                collections=() if collection is None else (collection,),
            )
        if action.kind == "drain":
            return CollectionEvent(
                action=action,
                collections=tuple(
                    self.pool.drain(action_id=collection_action_id(action))
                ),
            )
        raise ValueError(f"unknown CollectionAction kind {action.kind!r}")

    def rehydrate(
        self,
        *,
        ready_results: tuple[JobResult, ...] = (),
        issued_jobs: tuple[Job, ...] = (),
        collection_index: int = 0,
    ) -> None:
        self.pool.rehydrate(
            ready_results=ready_results,
            issued_jobs=issued_jobs,
            collection_index=collection_index,
        )

    def close(self, *, wait_for_pending: bool = True) -> None:
        if self._closed:
            return
        self._closed = True
        self.pool.close(wait_for_pending=wait_for_pending)

    def __enter__(self) -> CollectionTrial:
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> bool:
        self.close(wait_for_pending=_exc_type is None)
        return False


def run_collection_search(
    policy: CollectionPolicy,
    trial: CollectionTrial,
    initial_state: CollectionState | None = None,
) -> CollectionRun:
    """Run collection control through the public application lifecycle."""

    state = policy.initial_state() if initial_state is None else initial_state
    from astrolabe.app import Astrolabe

    from .composition import ControlRun

    return Astrolabe(
        ControlRun(
            policy=policy,
            trial=trial,
            initial_state=state,
            runtime=None,
        )
    ).run()


__all__ = [
    "CollectionAction",
    "CollectionEvent",
    "CollectionPolicy",
    "CollectionRun",
    "CollectionState",
    "CollectionTrial",
    "JobIdFactory",
    "PayloadFactory",
    "SelectedAdopter",
    "collection_action_id",
    "run_collection_search",
]
