"""Small runtime lifecycle, ordered overlap, and atomic output primitives."""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import copy_context
from dataclasses import dataclass, replace
import logging
import os
from pathlib import Path
import secrets
import stat
import threading
from typing import Callable, Iterable, Iterator, Literal, TextIO, TypeVar


_T = TypeVar("_T")
_U = TypeVar("_U")
_V = TypeVar("_V")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CleanupRecord:
    """An adapter-contract receipt, not proof of OS/device resource release."""

    resource: object
    label: str
    state: Literal["pending", "closing", "closed", "failed", "skipped"] = "pending"
    attempts: int = 0
    operation: str | None = None
    error: BaseException | None = None
    errors: tuple[BaseException, ...] = ()


@dataclass(frozen=True)
class CloseReport:
    """Immutable snapshot; failed and in-flight resources remain inspectable."""

    records: tuple[CleanupRecord, ...]
    draining: bool = False

    @property
    def failed(self) -> tuple[CleanupRecord, ...]:
        return tuple(record for record in self.records if record.state == "failed")

    @property
    def complete(self) -> bool:
        return not self.draining and all(
            record.state in {"closed", "skipped"} for record in self.records
        )


class ResourceOwner:
    """Identity-deduplicated, nonblocking cleanup with retained failure receipts.

    Concurrent/reentrant close returns an in-flight snapshot, never waits for a
    callback (which may itself be waiting on that caller). Inspect ``report``
    after the draining caller finishes. Late registrations are drained too.
    Failed operations are not automatically repeated. ``retry_failed=True`` is
    an explicit caller assertion that these operations are safe to retry, e.g.
    a documented idempotent shutdown; it must not be used for arbitrary
    destructors. A retry attempts each previously failed operation at most once.
    """

    def __init__(self) -> None:
        self._resources: dict[int, CleanupRecord] = {}
        self._lock = threading.RLock()
        self._draining = False
        self._close_requested = False

    @property
    def report(self) -> CloseReport:
        with self._lock:
            return CloseReport(tuple(self._resources.values()), self._draining)

    def add(self, resource: object, label: str) -> None:
        if resource is not None:
            with self._lock:
                self._resources.setdefault(id(resource), CleanupRecord(resource, label))
                drain = self._close_requested and not self._draining
            if drain:
                self.close()

    def close(self, *, retry_failed: bool = False) -> CloseReport:
        with self._lock:
            self._close_requested = True
            if self._draining:
                return self.report
            self._draining = True
            retry = {identity for identity, record in self._resources.items()
                     if retry_failed and record.state == "failed"}
        owns_drain = True
        try:
            while True:
                with self._lock:
                    candidate = next((
                        (identity, record) for identity, record in self._resources.items()
                        if record.state == "pending" or identity in retry
                    ), None)
                    if candidate is None:
                        # Atomically finish the drain relative to add().
                        self._draining = False
                        owns_drain = False
                        return self.report
                    identity, record = candidate
                    retry.discard(identity)
                    record = replace(record, state="closing", attempts=record.attempts + 1,
                                     error=None)
                    self._resources[identity] = record
                operation = record.operation
                try:
                    # A retry uses the same adapter operation, not a fallback.
                    if operation is None:
                        operation = "shutdown"
                        method = getattr(record.resource, operation, None)
                        if not callable(method):
                            operation = "close"
                            method = getattr(record.resource, operation, None)
                        if not callable(method):
                            operation = None
                    else:
                        method = getattr(record.resource, operation)
                    if operation is not None:
                        method()
                    result = replace(record, state="closed" if operation else "skipped",
                                     operation=operation)
                except BaseException as error:
                    result = replace(record, state="failed", operation=operation, error=error,
                                     errors=(*record.errors, error))
                    logger.exception("Failed to close resource %s", record.label)
                with self._lock:
                    self._resources[identity] = result
        finally:
            if owns_drain:
                with self._lock:
                    self._draining = False


def ordered_cpu_overlap(
    items: Iterable[_T],
    evaluate: Callable[[_T], _U],
    finalize: Callable[[_U], _V],
    *,
    max_workers: int,
    max_in_flight: int | None = None,
) -> list[_V]:
    """Run evaluate serially and finalize concurrently, consuming in input order.

    A producer exception is deferred until earlier submitted work is consumed,
    so an earlier postprocessing error takes precedence. No further work is
    evaluated after an observed failure.
    """
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
        raise ValueError("max_workers must be a positive integer")
    if max_in_flight is None:
        max_in_flight = max_workers + 1
    if isinstance(max_in_flight, bool) or not isinstance(max_in_flight, int) or max_in_flight < 1:
        raise ValueError("max_in_flight must be a positive integer")

    results: list[_V] = []
    pending = deque()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        try:
            iterator = iter(items)
            while True:
                if len(pending) >= max_in_flight:
                    results.append(pending.popleft().result())
                try:
                    item = next(iterator)
                except StopIteration:
                    break
                except Exception:
                    while pending:
                        results.append(pending.popleft().result())
                    raise
                try:
                    value = evaluate(item)
                except Exception:
                    while pending:
                        results.append(pending.popleft().result())
                    raise
                context = copy_context()
                pending.append(executor.submit(context.run, finalize, value))
            while pending:
                results.append(pending.popleft().result())
        finally:
            for future in pending:
                future.cancel()
    return results


@contextmanager
def atomic_text_writer(
    path: str | os.PathLike[str], *, encoding: str = "utf-8", newline: str | None = None
) -> Iterator[TextIO]:
    """Replace a file after writing; follow destination symlinks as open does.

    Atomic replacement creates a new inode and is not a durability guarantee.
    """
    destination = Path(path).resolve()
    # Match ordinary open(..., 'w') permissions for a new destination, without
    # reading or changing the process-wide umask in this multithreaded program.
    while True:
        temporary = destination.parent / f".{destination.name}.{secrets.token_hex(8)}.tmp"
        try:
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
            break
        except FileExistsError:
            continue
    try:
        try:
            mode = stat.S_IMODE(destination.stat().st_mode)
        except FileNotFoundError:
            mode = None
        if mode is not None:
            os.fchmod(fd, mode)
        with os.fdopen(fd, "w", encoding=encoding, newline=newline) as handle:
            fd = -1
            yield handle
        os.replace(temporary, destination)
    finally:
        if fd != -1:
            os.close(fd)
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
