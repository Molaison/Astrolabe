"""Deterministic fixed-task-list execution over the shared collection pool.

This module contains no search or selection policy.  Callers provide an
ordered list of jobs and a worker; each job represents one bounded unit of
work, and batches retain submission order regardless of completion order.

P0 execution is process-lifetime only: it uses no durable policy checkpoint.
Refold recovery is delegated to its per-slot receipts at each job admission.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from typing import Any, Callable, Iterable, Literal

from .collection import (
    CollectionConfig,
    CollectionExecutionError,
    InlineExecutor,
    Job,
    JobResult,
    ResultCollection,
)
from .collection_search import (
    CollectionAction,
    CollectionEvent,
    CollectionTrial,
)
from .composition import ControlRun


FixedListPhase = Literal["submit", "drain", "finished"]
WorkerExecution = Literal["threaded", "synchronous"]


@dataclass(frozen=True)
class FixedTask:
    """One stable item in a fixed ordered task list."""

    job_id: str
    payload: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.job_id, str) or not self.job_id:
            raise ValueError("fixed task job_id must be a non-empty string")


@dataclass(frozen=True)
class FixedTaskListState:
    """Cursor and committed collections for an ordered fixed task list."""

    phase: FixedListPhase
    next_index: int
    generation: int
    batch_size: int
    pending_job_ids: tuple[str, ...]
    collections: tuple[ResultCollection, ...]


@dataclass(frozen=True)
class FixedTaskListRun:
    """Ordered completion batches from a fixed task list."""

    collections: tuple[ResultCollection, ...]

    @property
    def results(self) -> tuple[JobResult, ...]:
        return tuple(
            result
            for collection in self.collections
            for result in collection.results
        )


def no_selection(_results: Iterable[JobResult]) -> None:
    """CompletionPool hook for policies that never select a winner."""

    return None


class FixedTaskListPolicy:
    """Advance an ordered task list without search or selection."""

    request_type = CollectionAction
    result_type = CollectionEvent

    def __init__(
        self,
        tasks: Iterable[FixedTask],
        *,
        batch_size: int = 1,
    ) -> None:
        self.tasks = tuple(tasks)
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("fixed-list batch_size must be a positive integer")
        if len({task.job_id for task in self.tasks}) != len(self.tasks):
            raise ValueError("fixed-list job IDs must be unique")
        self.batch_size = batch_size
        payload = json.dumps(
            {
                "batch_size": self.batch_size,
                "job_ids": [task.job_id for task in self.tasks],
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        # P0 is process-lifetime control.  This digest covers scheduling shape
        # only; it is neither a scientific identity nor a durable checkpoint.
        self.process_lifetime_schedule_digest = hashlib.sha256(payload).hexdigest()

    def initial_state(self) -> FixedTaskListState:
        phase: FixedListPhase = "submit" if self.tasks else "finished"
        return FixedTaskListState(
            phase=phase,
            next_index=0,
            generation=1,
            batch_size=self.batch_size,
            pending_job_ids=(),
            collections=(),
        )

    def ask(self, state: FixedTaskListState) -> CollectionAction:
        self._validate_state(state)
        if state.phase == "submit":
            stop = min(len(self.tasks), state.next_index + state.batch_size)
            jobs = tuple(
                Job(
                    job_id=task.job_id,
                    generation=state.generation,
                    payload=task.payload,
                )
                for task in self.tasks[state.next_index:stop]
            )
            if not jobs:
                return CollectionAction(kind="drain")
            return CollectionAction(
                kind="submit",
                generation=state.generation,
                jobs=jobs,
            )
        if state.phase == "drain":
            return CollectionAction(kind="drain")
        raise ValueError("finished fixed task list has no action")

    def tell(
        self,
        state: FixedTaskListState,
        event: CollectionEvent,
    ) -> FixedTaskListState:
        self._validate_state(state)
        if not isinstance(event, CollectionEvent):
            raise TypeError("FixedTaskListPolicy requires CollectionEvent")
        expected = self.ask(state)
        if event.action != expected:
            raise ValueError("CollectionEvent does not match the fixed-list action")

        if state.phase == "submit":
            if event.collections:
                raise ValueError("fixed-list submit event cannot contain collections")
            return replace(
                state,
                phase="drain",
                pending_job_ids=tuple(job.job_id for job in event.action.jobs),
            )

        if state.phase == "drain":
            if not event.collections:
                raise CollectionExecutionError(
                    "fixed-list drain returned no terminal batch"
                )
            if len(event.collections) != 1:
                raise ValueError(
                    "fixed-list drain requires exactly one ResultCollection"
                )
            collection = _ordered_collection(
                event.collections[0],
                state.pending_job_ids,
            )
            return replace(
                state,
                phase=(
                    "submit"
                    if state.next_index + len(state.pending_job_ids) < len(self.tasks)
                    else "finished"
                ),
                next_index=state.next_index + len(state.pending_job_ids),
                generation=state.generation + 1,
                pending_job_ids=(),
                collections=state.collections + (collection,),
            )

        raise ValueError("cannot tell a finished fixed task list")

    def finished(self, state: FixedTaskListState) -> bool:
        self._validate_state(state)
        return state.phase == "finished"

    def selected(self, state: FixedTaskListState) -> FixedTaskListRun:
        self._validate_state(state)
        return FixedTaskListRun(collections=state.collections)

    def _validate_state(self, state: FixedTaskListState) -> None:
        if not isinstance(state, FixedTaskListState):
            raise TypeError("FixedTaskListPolicy state has the wrong type")
        if state.phase not in {"submit", "drain", "finished"}:
            raise ValueError("invalid fixed task list phase")
        if state.next_index < 0 or state.next_index > len(self.tasks):
            raise ValueError("fixed task list cursor is out of range")
        if state.batch_size != self.batch_size:
            raise ValueError("fixed task list batch_size changed")
        expected = set(self.tasks[i].job_id for i in range(
            state.next_index,
            min(len(self.tasks), state.next_index + state.batch_size),
        ))
        if state.phase == "drain" and set(state.pending_job_ids) != expected:
            raise ValueError("fixed task list pending jobs changed")
        if state.phase != "drain" and state.pending_job_ids:
            raise ValueError("fixed task list has pending jobs outside drain")


def run_fixed_task_list(
    tasks: Iterable[FixedTask],
    worker: Callable[[Job], JobResult | float],
    *,
    max_in_flight: int = 1,
    worker_execution: WorkerExecution = "threaded",
    logger=None,
    admission=None,
    terminal_observed=None,
    collection_commit=None,
) -> FixedTaskListRun:
    """Run one ordered fixed-task list through ControlRun and CompletionPool."""

    tasks = tuple(tasks)
    if type(max_in_flight) is not int or max_in_flight < 1:
        raise ValueError("max_in_flight must be a positive integer")
    if worker_execution not in {"threaded", "synchronous"}:
        raise ValueError(
            "worker_execution must be 'threaded' or 'synchronous'"
        )
    if worker_execution == "synchronous" and max_in_flight != 1:
        raise ValueError("synchronous fixed-list execution requires max_in_flight=1")

    batch_size = max(1, min(max_in_flight, len(tasks) or 1))
    policy = FixedTaskListPolicy(tasks, batch_size=batch_size)
    config = CollectionConfig(
        num_jobs=batch_size,
        min_completed=batch_size,
        max_in_flight=batch_size,
        fail_policy="record-and-continue",
    )
    with CollectionTrial(
        config=config,
        worker=worker,
        selector=no_selection,
        logger=logger,
        admission=admission,
        terminal_observed=terminal_observed,
        collection_commit=collection_commit,
        executor_factory=(
            InlineExecutor if worker_execution == "synchronous" else None
        ),
    ) as trial:
        result = ControlRun(
            policy=policy,
            trial=trial,
            initial_state=policy.initial_state(),
        ).run()
    if not isinstance(result, FixedTaskListRun):
        raise TypeError("fixed task list returned an unexpected run type")
    return result


def _ordered_collection(
    collection: ResultCollection,
    expected_job_ids: tuple[str, ...],
) -> ResultCollection:
    by_id = {result.job_id: result for result in collection.results}
    if len(by_id) != len(collection.results):
        raise ValueError("fixed-list collection contains duplicate job results")
    if set(by_id) != set(expected_job_ids):
        missing = sorted(set(expected_job_ids) - set(by_id))
        extra = sorted(set(by_id) - set(expected_job_ids))
        raise CollectionExecutionError(
            "fixed-list collection has incomplete job proof: "
            f"missing={missing} extra={extra}"
        )
    return replace(
        collection,
        results=tuple(by_id[job_id] for job_id in expected_job_ids),
    )
__all__ = [
    "FixedTask",
    "FixedTaskListPolicy",
    "FixedTaskListRun",
    "FixedTaskListState",
    "no_selection",
    "run_fixed_task_list",
]
