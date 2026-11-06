"""Bounded concrete-Future execution for typed branch requests.

The executor owns only observations and one pending-future map. Its caller owns
the process pool, job lifecycle, collection policy, and checkpointing.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ALL_COMPLETED, FIRST_COMPLETED, Future, wait
from typing import Any, TypeVar, cast

from .branch_types import BranchAction, BranchEvent, BranchJob, BranchTerminal


T = TypeVar("T")
R = TypeVar("R")
I = TypeVar("I")


def _observe_futures(
    futures: Iterable[Future[R]],
) -> tuple[dict[Future[R], object], tuple[tuple[Future[R], BaseException], ...]]:
    """Read every result in a ready group without stopping at the first error."""

    values: dict[Future[R], object] = {}
    errors: list[tuple[Future[R], BaseException]] = []
    for future in futures:
        try:
            values[future] = future.result()
        except BaseException as error:
            errors.append((future, error))
    return values, tuple(errors)


def _submitted_future(
    pool: object,
    worker: Callable[[T], R],
    task: T,
) -> Future[R]:
    future = pool.submit(worker, task)  # type: ignore[attr-defined]
    if not isinstance(future, Future):
        raise TypeError("pool.submit() must return a concrete Future")
    return future


def _job_order(job: BranchJob) -> tuple[str, int]:
    return job.phase, job.id


class BranchExecutor:
    """Execute one bounded branch action over a caller-owned pool."""

    request_type = BranchAction
    result_type = BranchEvent

    def __init__(
        self,
        pool: object,
        *,
        worker: Callable[[BranchJob], object] | None = None,
        submit: Callable[[object, BranchJob], Future[Any]] | None = None,
        resolve: Callable[[BranchJob, Any], BranchTerminal] | None = None,
        on_error: Callable[[BranchJob, BaseException], None] | None = None,
        on_observed: Callable[[BranchJob, Any], None] | None = None,
    ) -> None:
        if submit is None and worker is None:
            raise TypeError("worker is required when submit is omitted")
        if submit is not None and not callable(submit):
            raise TypeError("submit must be callable")
        if resolve is not None and not callable(resolve):
            raise TypeError("resolve must be callable")
        if on_error is not None and not callable(on_error):
            raise TypeError("on_error must be callable")
        if on_observed is not None and not callable(on_observed):
            raise TypeError("on_observed must be callable")
        self.pool = pool
        self.worker = worker
        self._submit = submit
        self._resolve = resolve
        self._on_error = on_error
        self._on_observed = on_observed
        self.one_pending: dict[Future[Any], BranchJob] = {}
        self.observed: dict[str, BranchTerminal] = {}
        self._admitted: dict[str, BranchJob] = {}

    def _submit_job(self, job: BranchJob) -> Future[Any]:
        if self._submit is not None:
            future = self._submit(self.pool, job)
        else:
            future = _submitted_future(
                self.pool,
                cast(Callable[[BranchJob], object], self.worker),
                job,
            )
        if not isinstance(future, Future):
            raise TypeError("submit must return a concrete Future")
        return future

    def restore(self, job: BranchJob, future: Future[Any]) -> None:
        """Install one already reserved owner Future without submitting it."""

        if not isinstance(job, BranchJob):
            raise TypeError("restore expects BranchJob")
        if not isinstance(future, Future):
            raise TypeError("restore expects a concrete Future")
        previous = self._admitted.get(job.identity)
        if previous is not None:
            pending_future = next(
                (
                    pending
                    for pending, pending_job in self.one_pending.items()
                    if pending_job.identity == job.identity
                ),
                None,
            )
            if previous == job and pending_future is future:
                return
            raise ValueError(
                f"restored BranchJob {job.identity!r} conflicts with "
                "the admitted job"
            )
        if future in self.one_pending:
            raise ValueError("restored Future is already pending")
        self._admitted[job.identity] = job
        self.one_pending[future] = job

    def _admit(self, jobs: tuple[BranchJob, ...]) -> None:
        fresh: list[BranchJob] = []
        identities: set[str] = set()
        for job in jobs:
            if not isinstance(job, BranchJob):
                raise TypeError("BranchAction.jobs must contain BranchJob values")
            identity = job.identity
            admitted = self._admitted.get(identity)
            if admitted is not None:
                if admitted != job:
                    raise ValueError(
                        f"BranchJob {identity!r} conflicts with the admitted job"
                    )
                continue
            if identity in identities:
                raise ValueError(f"duplicate BranchJob identity {identity!r}")
            identities.add(identity)
            fresh.append(job)

        for job in fresh:
            future = self._submit_job(job)
            if future in self.one_pending:
                raise ValueError("submit returned a Future already pending")
            self._admitted[job.identity] = job
            self.one_pending[future] = job

    def _resolve_terminal(
        self,
        job: BranchJob,
        raw: object,
    ) -> BranchTerminal:
        terminal = raw if self._resolve is None else self._resolve(job, raw)
        if not isinstance(terminal, BranchTerminal):
            raise TypeError(
                "worker result resolver must return BranchTerminal"
            )
        if terminal.job != job:
            raise ValueError(
                f"BranchTerminal job {terminal.job.identity!r} does not match "
                f"admitted job {job.identity!r}"
            )
        return terminal

    def perform(self, action: BranchAction) -> BranchEvent:
        if type(action) is not BranchAction:
            raise TypeError("BranchExecutor expects BranchAction")
        if action.wait not in ("any", "all", "none"):
            raise ValueError(f"unsupported wait mode {action.wait!r}")

        self._admit(action.jobs)
        if action.wait == "none" or not self.one_pending:
            return BranchEvent(())

        return_when = (
            FIRST_COMPLETED if action.wait == "any" else ALL_COMPLETED
        )
        ready, _ = wait(
            tuple(self.one_pending),
            return_when=return_when,
        )
        ordered = sorted(ready, key=lambda future: _job_order(self.one_pending[future]))
        # Future.result() is the only background-result observation phase. Raw
        # owner adapter records may therefore be native DTOs or JSON projections.
        raw_results, raw_errors = _observe_futures(ordered)
        observation_errors = []
        if self._on_observed is not None:
            for future, raw in raw_results.items():
                try:
                    self._on_observed(self.one_pending[future], raw)
                except BaseException as error:
                    observation_errors.append((future, error))
        raw_errors += tuple(observation_errors)
        if raw_errors:
            callback_error: BaseException | None = None
            if self._on_error is not None:
                for future, error in raw_errors:
                    try:
                        self._on_error(self.one_pending[future], error)
                    except BaseException as error_callback_error:
                        if callback_error is None:
                            callback_error = error_callback_error
            if callback_error is not None:
                raise callback_error
            raise raw_errors[0][1]
        terminals = tuple(
            self._resolve_terminal(
                self.one_pending[future],
                raw_results[future],
            )
            for future in ordered
        )

        # Futures leave pending only after every result and full job identity in
        # the ready group has been validated.
        for future, terminal in zip(ordered, terminals):
            self.observed[self.one_pending[future].identity] = terminal
            self.one_pending.pop(future)
        return BranchEvent(terminals)


def gather_ordered(
    pool: object,
    tasks: Sequence[T],
    width: int,
    worker: Callable[[T], R],
    identity: Callable[[T], I],
) -> list[R]:
    """Run tasks with bounded free-slot refill and ordered result collection.

    ``identity`` is supplied by the actual owning records and must identify both
    each input task and its corresponding worker result. The complete ready
    group is observed before completed slots are released for refill.
    """

    if type(width) is not int or width < 1:
        raise ValueError("Dispatch width must be a positive integer")
    frozen = tuple(tasks)
    identities: list[I] = []
    for task in frozen:
        task_identity = identity(task)
        if task_identity in identities:
            raise ValueError(f"duplicate task identity {task_identity!r}")
        identities.append(task_identity)
    if not frozen:
        return []

    results: list[R | None] = [None] * len(frozen)
    pending: dict[Future[R], int] = {}
    next_index = 0
    while next_index < len(frozen) or pending:
        while next_index < len(frozen) and len(pending) < width:
            future = _submitted_future(pool, worker, frozen[next_index])
            if future in pending:
                raise ValueError("pool.submit() returned a Future already pending")
            pending[future] = next_index
            next_index += 1

        ready, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
        ordered = sorted(ready, key=pending.__getitem__)

        raw_results, errors = _observe_futures(ordered)
        if errors:
            raise errors[0][1]
        observed: dict[Future[R], tuple[int, R]] = {}
        for future in ordered:
            index = pending[future]
            value = raw_results[future]
            if identity(value) != identities[index]:
                raise ValueError("Worker result identity differs from dispatched task")
            observed[future] = (index, cast(R, value))

        # No completed slot is released until the entire ready group has passed
        # result and identity validation.
        for future, (index, value) in observed.items():
            pending.pop(future)
            results[index] = value

    if any(value is None for value in results):
        raise RuntimeError("ordered dispatch completed with missing results")
    return cast(list[R], results)


__all__ = ["BranchExecutor", "gather_ordered"]
