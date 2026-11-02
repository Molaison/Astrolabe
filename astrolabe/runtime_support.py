"""Small runtime lifecycle, ordered overlap, and atomic output primitives."""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import copy_context
import logging
import os
from pathlib import Path
import secrets
import stat
from typing import Callable, Iterable, Iterator, TextIO, TypeVar


_T = TypeVar("_T")
_U = TypeVar("_U")
_V = TypeVar("_V")
logger = logging.getLogger(__name__)


class ResourceOwner:
    """Close explicitly registered objects once, even when aliases repeat."""

    def __init__(self) -> None:
        self._resources: dict[int, tuple[object, str]] = {}
        self._closed: set[int] = set()

    def add(self, resource: object, label: str) -> None:
        if resource is not None:
            self._resources.setdefault(id(resource), (resource, label))

    def close(self) -> None:
        for identity, (resource, label) in list(self._resources.items()):
            if identity in self._closed:
                continue
            self._closed.add(identity)
            try:
                method = getattr(resource, "shutdown", None)
                if not callable(method):
                    method = getattr(resource, "close", None)
                if callable(method):
                    method()
            except Exception:
                logger.exception("Failed to close resource %s", label)


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
    """Replace one text file only after the complete write and close succeed."""
    destination = Path(path)
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
