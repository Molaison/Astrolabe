"""CPU execution slots that can be yielded without releasing request memory."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from enum import Enum, auto
import os
import threading


_CURRENT = ContextVar("astrolabe_cpu_work", default=None)


class _State(Enum):
    YIELDED = auto()
    ACTIVE = auto()
    LOST = auto()
    CLOSED = auto()


class _CpuWork:
    def __init__(self, client, request, capacity):
        self.client = client
        self.request = request
        self.capacity = capacity
        self.owner = (os.getpid(), threading.get_ident())
        self.slot = None
        self.state = _State.YIELDED
        self.depth = 0

    def check_owner(self, *, cleanup=False):
        if self.owner != (os.getpid(), threading.get_ident()):
            raise RuntimeError("CPU scope belongs to a different process or thread")
        if self.state is _State.CLOSED or (self.state is _State.LOST and not cleanup):
            raise RuntimeError("CPU scope is no longer admitted")

    def acquire(self):
        self.check_owner()
        slot = self.client.lease(**self.request, reservation_token=self.capacity.token)
        self.slot = slot
        try:
            slot.__enter__()
        except BaseException as error:
            self.state = _State.LOST
            try:
                slot.cancel()
            except BaseException as cleanup:
                _retain(error, cleanup, self)
            else:
                self.slot = None
            raise
        self.state = _State.ACTIVE

    def release(self):
        self.check_owner()
        try:
            self.slot.finish()
        except BaseException:
            self.state = _State.LOST
            raise
        self.slot = None
        self.state = _State.YIELDED

    def close(self):
        if self.owner != (os.getpid(), threading.get_ident()):
            raise RuntimeError("CPU scope belongs to a different process or thread")
        if self.slot is not None:
            slot = self.slot
            if self.state is _State.LOST:
                slot.cancel()
            else:
                try:
                    slot.finish()
                except BaseException as error:
                    self.state = _State.LOST
                    try:
                        slot.cancel()
                    except BaseException as cleanup:
                        _retain(error, cleanup, self)
                    else:
                        self.slot = None
                    raise
            self.slot = None
        self.state = _State.CLOSED


def _retain(primary, secondary, scope):
    primary.add_note(f"CPU admission restoration/cleanup failed: {secondary!r}")
    primary.cpu_work_scope = scope


@contextmanager
def cpu_work_scope(client, *, backend, identity, phase, shape="", ram_bytes=0,
                   output_bytes=0, timeout=None):
    if client is None:
        yield None
        return
    if _CURRENT.get() is not None:
        raise RuntimeError("CPU work scopes cannot be nested")
    request = dict(resource="cpu", backend=backend, identity=identity,
                   phase=phase, shape=shape, timeout=timeout)
    capacity = client.lease(**request, ram_bytes=ram_bytes,
                            output_bytes=output_bytes, capacity_only=True)
    scope = None
    token = None
    primary = None
    try:
        capacity.__enter__()
        scope = _CpuWork(client, request, capacity)
        scope.acquire()
        token = _CURRENT.set(scope)
        yield scope
        scope.check_owner()
    except BaseException as error:
        primary = error
        raise
    finally:
        if token is not None:
            _CURRENT.reset(token)
        cleanup_error = None
        if scope is not None:
            try:
                scope.close()
            except BaseException as error:
                cleanup_error = error
        try:
            if scope is None:
                capacity.cancel()
            else:
                capacity.finish()
        except BaseException as error:
            try:
                capacity.cancel()
            except BaseException as cancellation:
                _retain(error, cancellation, scope)
            if cleanup_error is None:
                cleanup_error = error
            else:
                _retain(cleanup_error, error, scope)
        if cleanup_error is not None:
            if primary is not None:
                _retain(primary, cleanup_error, scope)
            else:
                raise cleanup_error


@contextmanager
def yield_cpu_slot(*, cleanup=False):
    """Yield execution for a wait, allowing cleanup after admission is lost."""
    scope = _CURRENT.get()
    if scope is None:
        yield
        return
    scope.check_owner(cleanup=cleanup)
    if scope.state is _State.LOST:
        yield
        return
    if scope.depth == 0:
        try:
            scope.release()
        except BaseException as release_error:
            if not cleanup:
                raise
            try:
                yield
            except BaseException as error:
                _retain(error, release_error, scope)
                raise
            raise
    scope.depth += 1
    primary = None
    try:
        yield
    except BaseException as error:
        primary = error
        raise
    finally:
        scope.depth -= 1
        if scope.depth == 0:
            try:
                scope.acquire()
            except BaseException as error:
                if primary is None:
                    raise
                _retain(primary, error, scope)


@contextmanager
def cpu_lock_scope(lock):
    """Yield CPU while acquiring a lock; restore admission before its body."""
    with ExitStack() as stack:
        with yield_cpu_slot():
            acquired = stack.enter_context(lock)
        yield acquired
