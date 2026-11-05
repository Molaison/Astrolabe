"""Generic asynchronous completion collection primitives.

The pool owns worker admission and result collection only. Search generation
state belongs to ``collection_search`` and the shared coordinator.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import (
    FIRST_COMPLETED,
    Executor,
    Future,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Literal

from astrolabe.design.runtime import OutcomeUnknown
from astrolabe.utils.log_events import event_span


FailurePolicy = Literal["record-and-continue", "fail-fast"]
JobStatus = Literal["completed", "failed"]
CollectionTrigger = Literal["min_completed", "max_wait_s", "drain"]


@dataclass(frozen=True, init=False)
class CollectionConfig:
    """Pool sizing and completion-window policy."""

    num_jobs: int
    min_completed: int
    max_wait_s: float
    max_in_flight: int | None
    fail_policy: FailurePolicy

    def __init__(
        self,
        num_jobs: int = 4,
        min_completed: int = 2,
        max_wait_s: float = 30.0,
        max_in_flight: int | None = None,
        fail_policy: FailurePolicy = "record-and-continue",
        *,
        num_candidates: int | None = None,
    ) -> None:
        if num_candidates is not None:
            candidate_count = int(num_candidates)
            if num_jobs != 4 and int(num_jobs) != candidate_count:
                raise ValueError("num_jobs and num_candidates disagree")
            num_jobs = candidate_count
        num_jobs = int(num_jobs)
        min_completed = int(min_completed)
        max_wait_s = float(max_wait_s)
        max_in_flight = None if max_in_flight is None else int(max_in_flight)
        if num_jobs < 1:
            raise ValueError("num_jobs must be >= 1")
        if min_completed < 1:
            raise ValueError("min_completed must be >= 1")
        if max_wait_s <= 0:
            raise ValueError("max_wait_s must be > 0")
        if max_in_flight is not None and max_in_flight < 1:
            raise ValueError("max_in_flight must be >= 1 when set")
        if fail_policy not in {"record-and-continue", "fail-fast"}:
            raise ValueError(
                "fail_policy must be 'record-and-continue' or 'fail-fast'"
            )
        object.__setattr__(self, "num_jobs", num_jobs)
        object.__setattr__(self, "min_completed", min_completed)
        object.__setattr__(self, "max_wait_s", max_wait_s)
        object.__setattr__(self, "max_in_flight", max_in_flight)
        object.__setattr__(self, "fail_policy", fail_policy)

    @property
    def num_candidates(self) -> int:
        """Compatibility spelling for the former candidate-specific DTO."""

        return self.num_jobs

    @property
    def worker_count(self) -> int:
        return int(self.max_in_flight or self.num_jobs)


@dataclass(frozen=True, init=False)
class Job:
    """One deferred completion request."""

    job_id: str
    generation: int
    payload: Any

    def __init__(
        self,
        job_id: str | None = None,
        generation: int = 0,
        payload: Any = None,
        *,
        candidate_id: str | None = None,
    ) -> None:
        if candidate_id is not None:
            if job_id is not None and str(job_id) != str(candidate_id):
                raise ValueError("job_id and candidate_id disagree")
            job_id = candidate_id
        if job_id is None:
            raise TypeError("job_id is required")
        object.__setattr__(self, "job_id", str(job_id))
        object.__setattr__(self, "generation", int(generation))
        object.__setattr__(self, "payload", payload)

    @property
    def candidate_id(self) -> str:
        return self.job_id


@dataclass(frozen=True, init=False)
class JobResult:
    """One worker outcome, including failures."""

    job: Job
    status: JobStatus
    score: float | None
    payload: Any
    error: str | None
    duration_s: float

    def __init__(
        self,
        job: Job | None = None,
        status: JobStatus = "completed",
        score: float | None = None,
        payload: Any = None,
        error: str | None = None,
        duration_s: float = 0.0,
        *,
        task: Job | None = None,
    ) -> None:
        if task is not None:
            if job is not None and job != task:
                raise ValueError("job and task disagree")
            job = task
        if job is None:
            raise TypeError("job is required")
        object.__setattr__(self, "job", job)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "error", error)
        object.__setattr__(self, "duration_s", float(duration_s))

    @property
    def task(self) -> Job:
        return self.job

    @property
    def job_id(self) -> str:
        return self.job.job_id

    @property
    def candidate_id(self) -> str:
        return self.job_id


ResultSelector = Callable[[Iterable[JobResult]], JobResult | None]
AdmissionHook = Callable[[Job], JobResult | None]
TerminalObservedHook = Callable[[Job, JobResult], None]
RngSnapshotter = Callable[[], Any]
CollectionCommitHook = Callable[["CollectionCommit"], None]
ExecutorFactory = Callable[[int], Executor]


class InlineExecutor(Executor):
    """Run one submitted function on the caller thread and return a Future."""

    def __init__(self, max_workers: int = 1) -> None:
        if max_workers != 1:
            raise ValueError("InlineExecutor requires max_workers=1")
        self._closed = False

    def submit(self, function, /, *args, **kwargs) -> Future:
        if self._closed:
            raise RuntimeError("cannot schedule new futures after shutdown")
        future: Future = Future()
        if not future.set_running_or_notify_cancel():
            return future
        try:
            result = function(*args, **kwargs)
        except BaseException as exc:
            future.set_exception(exc)
        else:
            future.set_result(result)
        return future

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        self._closed = True


def select_highest_score(results: Iterable[JobResult]) -> JobResult | None:
    completed = [result for result in results if result.status == "completed"]
    if not completed:
        return None
    return sorted(
        completed,
        key=lambda result: (-float(result.score), result.job_id),
    )[0]


def payload_to_summary(payload: Any) -> object:
    if payload is None:
        return None
    if hasattr(payload, "as_summary"):
        return payload.as_summary()
    if isinstance(payload, (str, int, float, bool)):
        return payload
    if isinstance(payload, dict):
        return dict(payload)
    return repr(payload)


@dataclass(frozen=True)
class ResultCollection:
    collection_index: int
    trigger: CollectionTrigger
    results: tuple[JobResult, ...]
    selected: JobResult | None
    carried_forward_job_ids: tuple[str, ...]
    started_at_s: float
    finished_at_s: float

    @property
    def failed(self) -> tuple[JobResult, ...]:
        return tuple(result for result in self.results if result.status == "failed")

    @property
    def completed(self) -> tuple[JobResult, ...]:
        return tuple(result for result in self.results if result.status == "completed")

    @property
    def carried_forward_candidate_ids(self) -> tuple[str, ...]:
        return self.carried_forward_job_ids

    def as_row(self) -> dict[str, object]:
        failed_jobs = [
            {
                "job_id": result.job_id,
                "candidate_id": result.job_id,
                "error": result.error or "",
                "duration_s": result.duration_s,
                "payload": payload_to_summary(result.payload),
            }
            for result in self.failed
        ]
        carried_forward = ",".join(self.carried_forward_job_ids)
        return {
            "collection_index": self.collection_index,
            "trigger": self.trigger,
            "selected_job_id": (
                self.selected.job_id if self.selected is not None else ""
            ),
            "selected_candidate_id": (
                self.selected.job_id if self.selected is not None else ""
            ),
            "selected_score": (
                self.selected.score if self.selected is not None else None
            ),
            "completed": len(self.completed),
            "failed": len(self.failed),
            "failed_jobs": failed_jobs,
            "failed_candidates": failed_jobs,
            "carried_forward_jobs": carried_forward,
            "carried_forward": carried_forward,
            "duration_s": self.finished_at_s - self.started_at_s,
        }


@dataclass(frozen=True)
class CollectionCommit:
    """The durable selection boundary committed before ready results are consumed."""

    action_id: str
    collection_index: int
    trigger: CollectionTrigger
    ordered_job_ids: tuple[str, ...]
    selected_job_id: str | None
    consumed_job_ids: tuple[str, ...]
    carryforward_job_ids: tuple[str, ...]
    rng_before: Any
    rng_after: Any
    collection: ResultCollection


class CollectionExecutionError(RuntimeError):
    pass


class CompletionPool:
    """Own asynchronous completions without owning generation search state."""

    def __init__(
        self,
        config: CollectionConfig,
        worker: Callable[[Job], JobResult | float],
        *,
        selector: ResultSelector | None = None,
        logger: logging.Logger | None = None,
        admission: AdmissionHook | None = None,
        terminal_observed: TerminalObservedHook | None = None,
        collection_commit: CollectionCommitHook | None = None,
        rng_snapshot: RngSnapshotter | None = None,
        executor_factory: ExecutorFactory | None = None,
    ) -> None:
        self.config = config
        self.worker = worker
        self.selector = selector or select_highest_score
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.admission = admission
        self.terminal_observed = terminal_observed
        self.collection_commit = collection_commit
        self.rng_snapshot = rng_snapshot
        factory = executor_factory or ThreadPoolExecutor
        self._executor = factory(config.worker_count)
        self._pending: dict[Future[JobResult], Job] = {}
        self._ready: list[JobResult] = []
        self._issued: dict[str, Job] = {}
        self._known_jobs: dict[str, Job] = {}
        self._collection_index = 0
        self._window_started = time.monotonic()
        self._generation_started: dict[int, float] = {}
        self._closed = False

    def rehydrate(
        self,
        *,
        ready_results: Iterable[JobResult] = (),
        issued_jobs: Iterable[Job] = (),
        collection_index: int = 0,
    ) -> None:
        """Restore durable terminal/issued state without starting a worker."""

        if self._pending or self._ready or self._issued or self._known_jobs:
            raise RuntimeError("CompletionPool rehydrate requires an empty pool")
        for result in ready_results:
            self._remember_job(result.job)
            self._append_ready(result)
        for job in issued_jobs:
            self._remember_job(job)
            self._issued[job.job_id] = job
        if type(collection_index) is not int or collection_index < 0:
            raise ValueError("collection_index must be a nonnegative integer")
        self._collection_index = collection_index

    def submit_generation(self, generation: int, jobs: Iterable[Job]) -> None:
        if self._closed:
            raise RuntimeError("CompletionPool is closed")
        submitted = 0
        generation_int = int(generation)
        for job in jobs:
            if int(job.generation) != generation_int:
                raise ValueError(
                    f"Job {job.job_id} generation does not match submit generation"
                )
            self._remember_job(job)
            admitted_result = None
            if self.admission is not None:
                admitted_result = self.admission(job)
            if admitted_result is not None:
                if admitted_result.job != job:
                    raise CollectionExecutionError(
                        "Admission returned a result for another job"
                    )
                if (
                    admitted_result.status == "failed"
                    and self.config.fail_policy == "fail-fast"
                ):
                    raise CollectionExecutionError(
                        f"Job {job.job_id} failed: {admitted_result.error}"
                    )
                self._append_ready(admitted_result)
                continue
            future = self._executor.submit(self._run_job, job)
            self._pending[future] = job
            submitted += 1
        if submitted:
            started = time.monotonic()
            self._generation_started[generation_int] = started
            if not self._ready:
                self._window_started = started

    def collect_until_ready(
        self,
        *,
        target_generation: int | None = None,
        action_id: str = "collect",
    ) -> ResultCollection | None:
        generation = None if target_generation is None else int(target_generation)
        while True:
            ready = self._ready_for_generation(generation)
            if len(ready) >= self.config.min_completed:
                return self._finalize_collection(
                    "min_completed",
                    target_generation=generation,
                    action_id=action_id,
                )
            pending = self._pending_for_generation(generation)
            if not pending:
                if ready:
                    return self._finalize_collection(
                        "drain",
                        target_generation=generation,
                        action_id=action_id,
                    )
                unresolved = self._issued_for_generation(generation)
                if unresolved:
                    raise OutcomeUnknown(
                        "Issued collection jobs have no owner terminal proof: "
                        + ",".join(job.job_id for job in unresolved)
                    )
                return None

            window_started = self._collection_started_at(generation)
            elapsed = time.monotonic() - window_started
            remaining = max(0.0, float(self.config.max_wait_s) - elapsed)
            if remaining <= 0:
                if ready:
                    return self._finalize_collection(
                        "max_wait_s",
                        target_generation=generation,
                        action_id=action_id,
                    )
                raise CollectionExecutionError(
                    "No candidate finished within "
                    f"max_wait_s={self.config.max_wait_s}"
                )

            done, _not_done = wait(
                list(self._pending.keys()),
                timeout=remaining,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                ready = self._ready_for_generation(generation)
                if ready:
                    return self._finalize_collection(
                        "max_wait_s",
                        target_generation=generation,
                        action_id=action_id,
                    )
                raise CollectionExecutionError(
                    "No candidate finished within "
                    f"max_wait_s={self.config.max_wait_s}"
                )
            for future in done:
                job = self._pending.pop(future)
                result = future.result()
                if self.terminal_observed is not None:
                    self.terminal_observed(job, result)
                if (
                    result.status == "failed"
                    and self.config.fail_policy == "fail-fast"
                ):
                    raise CollectionExecutionError(
                        f"Job {job.job_id} failed: {result.error}"
                    )
                self._append_ready(result)
            if not self._pending_for_generation(generation):
                unresolved = self._issued_for_generation(generation)
                ready = self._ready_for_generation(generation)
                if unresolved and not ready:
                    raise OutcomeUnknown(
                        "Issued collection jobs have no owner terminal proof: "
                        + ",".join(job.job_id for job in unresolved)
                    )

    def drain(self, *, action_id: str = "drain") -> list[ResultCollection]:
        collections: list[ResultCollection] = []
        if self._issued:
            raise OutcomeUnknown(
                "Issued collection jobs have no owner terminal proof: "
                + ",".join(job.job_id for job in self._issued.values())
            )
        if self._pending:
            done, _not_done = wait(list(self._pending.keys()))
            for future in done:
                job = self._pending.pop(future)
                result = future.result()
                if self.terminal_observed is not None:
                    self.terminal_observed(job, result)
                if result.status == "failed" and self.config.fail_policy == "fail-fast":
                    raise CollectionExecutionError(
                        f"Job {job.job_id} failed: {result.error}"
                    )
                self._append_ready(result)
        if self._ready:
            collections.append(self._finalize_collection("drain", action_id=action_id))
        return collections

    def close(self, *, wait_for_pending: bool = True) -> None:
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(
            wait=wait_for_pending,
            cancel_futures=not wait_for_pending,
        )

    def __enter__(self) -> CompletionPool:
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> bool:
        self.close(wait_for_pending=_exc_type is None)
        return False

    def _run_job(self, job: Job) -> JobResult:
        started = time.monotonic()
        with event_span(
            logger=self.logger,
            component="completion",
            alias=job.job_id,
            detail=f"generation={job.generation}",
        ) as span:
            try:
                raw = self.worker(job)
                duration = time.monotonic() - started
                if isinstance(raw, JobResult):
                    result = raw
                else:
                    result = JobResult(
                        job=job,
                        status="completed",
                        score=float(raw),
                        duration_s=duration,
                    )
                if result.duration_s <= 0:
                    result = replace(result, duration_s=duration)
                if result.status == "failed" and not result.error:
                    result = replace(result, error="job failed")
                if result.status == "completed" and result.score is None:
                    raise ValueError("completed job result must include a score")
                if result.status == "failed":
                    span.finish(
                        event="span_error",
                        status="failed",
                        artifact=job.job_id,
                        error=result.error,
                        duration_s=result.duration_s,
                    )
                else:
                    span.finish(
                        artifact=job.job_id,
                        score=result.score,
                        duration_s=result.duration_s,
                    )
                return result
            except OutcomeUnknown:
                duration = time.monotonic() - started
                span.finish(
                    event="span_error",
                    status="unknown",
                    artifact=job.job_id,
                    duration_s=duration,
                )
                raise
            except Exception as exc:
                duration = time.monotonic() - started
                span.error(exc, duration_s=duration)
                return JobResult(
                    job=job,
                    status="failed",
                    error=str(exc) or exc.__class__.__name__,
                    duration_s=duration,
                )

    def _finalize_collection(
        self,
        trigger: CollectionTrigger,
        *,
        target_generation: int | None = None,
        action_id: str = "",
    ) -> ResultCollection:
        finished = time.monotonic()
        results = tuple(self._ready_for_generation(target_generation))
        rng_before = self.rng_snapshot() if self.rng_snapshot is not None else None
        selected = self.selector(results)
        rng_after = self.rng_snapshot() if self.rng_snapshot is not None else None
        if selected is not None:
            if selected not in results:
                raise CollectionExecutionError(
                    "Result selector returned a result outside the collection"
                )
            if selected.status != "completed":
                raise CollectionExecutionError(
                    "Result selector returned a non-completed result"
                )
        collection_index = self._collection_index + 1
        if target_generation is None:
            pending = tuple(self._pending.values())
        else:
            pending = tuple(self._pending_for_generation(target_generation).values())
        unresolved = tuple(
            job
            for job in self._issued.values()
            if target_generation is None
            or int(job.generation) == int(target_generation)
        )
        consumed = tuple(result.job_id for result in results)
        carried = tuple(job.job_id for job in pending + unresolved)
        collection = ResultCollection(
            collection_index=collection_index,
            trigger=trigger,
            results=results,
            selected=selected,
            carried_forward_job_ids=carried,
            started_at_s=self._collection_started_at(target_generation),
            finished_at_s=finished,
        )
        if self.collection_commit is not None:
            self.collection_commit(
                CollectionCommit(
                    action_id=action_id,
                    collection_index=collection_index,
                    trigger=trigger,
                    ordered_job_ids=consumed,
                    selected_job_id=(
                        selected.job_id if selected is not None else None
                    ),
                    consumed_job_ids=consumed,
                    carryforward_job_ids=carried,
                    rng_before=rng_before,
                    rng_after=rng_after,
                    collection=collection,
                )
            )
        self._collection_index = collection_index
        for job_id in consumed:
            self._issued.pop(job_id, None)
        if target_generation is None:
            self._ready = []
        else:
            self._ready = [
                result
                for result in self._ready
                if result.job.generation != target_generation
            ]
        self._window_started = finished
        if target_generation is not None:
            self._generation_started.pop(target_generation, None)
        self.logger.info(
            "completion_collection index=%s trigger=%s completed=%s failed=%s "
            "selected=%s carried_forward=%s",
            collection.collection_index,
            collection.trigger,
            len(collection.completed),
            len(collection.failed),
            collection.selected.job_id if collection.selected else "",
            ",".join(collection.carried_forward_job_ids),
        )
        return collection

    def _remember_job(self, job: Job) -> None:
        previous = self._known_jobs.get(job.job_id)
        if previous is not None and previous != job:
            raise CollectionExecutionError(
                f"Job id {job.job_id!r} changed its admitted content"
            )
        self._known_jobs[job.job_id] = job

    def _append_ready(self, result: JobResult) -> None:
        for existing in self._ready:
            if existing.job_id != result.job_id:
                continue
            if existing != result:
                raise CollectionExecutionError(
                    f"Job id {result.job_id!r} changed its terminal result"
                )
            return
        self._ready.append(result)

    def _issued_for_generation(
        self,
        target_generation: int | None,
    ) -> tuple[Job, ...]:
        if target_generation is None:
            return tuple(self._issued.values())
        return tuple(
            job
            for job in self._issued.values()
            if int(job.generation) == int(target_generation)
        )

    def _ready_for_generation(
        self,
        target_generation: int | None,
    ) -> list[JobResult]:
        if target_generation is None:
            return list(self._ready)
        return [
            result
            for result in self._ready
            if int(result.job.generation) == int(target_generation)
        ]

    def _pending_for_generation(
        self,
        target_generation: int | None,
    ) -> dict[Future[JobResult], Job]:
        if target_generation is None:
            return dict(self._pending)
        return {
            future: job
            for future, job in self._pending.items()
            if int(job.generation) == int(target_generation)
        }

    def _collection_started_at(self, target_generation: int | None) -> float:
        if target_generation is None:
            return self._window_started
        return self._generation_started.get(
            int(target_generation),
            self._window_started,
        )


__all__ = [
    "CollectionConfig",
    "CollectionCommit",
    "CollectionCommitHook",
    "CollectionExecutionError",
    "CollectionTrigger",
    "CompletionPool",
    "ExecutorFactory",
    "FailurePolicy",
    "InlineExecutor",
    "Job",
    "JobResult",
    "JobStatus",
    "ResultCollection",
    "ResultSelector",
    "AdmissionHook",
    "RngSnapshotter",
    "TerminalObservedHook",
    "payload_to_summary",
    "select_highest_score",
]
