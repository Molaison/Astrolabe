"""Cooperative continuations: only an explicitly frozen transport may leave main.

Continuations are live, single-writer objects, not recoverable checkpoints. An
interrupted issued Runtime call remains unknown, just like its synchronous form.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class TransportWait:
    transport: Callable[[], object]


def complete(iterator):
    """Drive the identical protocol synchronously for existing callers."""
    value, error = None, None
    while True:
        try:
            pending = iterator.throw(error) if error is not None else iterator.send(value)
        except StopIteration as done:
            return done.value
        if not isinstance(pending, TransportWait):
            raise TypeError("A continuation may yield only a frozen TransportWait")
        try:
            value, error = pending.transport(), None
        except BaseException as failure:
            value, error = None, failure


def activated(iterator, activate, context):
    """Restore the exact live bridge projection per main-thread resume.

    Never suspend inside LegacyContext.activate: its lock, active marker and
    parent projection are released before the caller sees a transport wait.
    """
    snapshot = None
    value, error = None, None
    while True:
        with activate():
            if snapshot is not None:
                context.restore_state(snapshot)
            try:
                pending = iterator.throw(error) if error is not None else iterator.send(value)
            except StopIteration as done:
                return done.value
            if not isinstance(pending, TransportWait):
                raise TypeError("Scientific continuation yielded non-transport work")
            snapshot = context.capture_state()
        try:
            value, error = (yield pending), None
        except BaseException as failure:
            value, error = None, failure
