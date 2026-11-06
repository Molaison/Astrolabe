"""Bounded, ordered inference stages across explicitly owned executors.

Resources are supplied by the caller, for example one CPU pool, one single-thread
GPU owner and an I/O pool. A stage's Python return is not device completion:
return Pending to wait off the resource's dispatch thread before advancing.
"""
from __future__ import annotations

from concurrent.futures import Executor, Future, ThreadPoolExecutor
from dataclasses import dataclass
import json
import os
import threading
import time
from typing import Any, Callable, Mapping, Sequence


@dataclass(frozen=True)
class Stage:
    name: str
    executor: str
    run: Callable[[Any], Any]


@dataclass(frozen=True)
class Pending:
    value: Any
    completion: Future | Callable[[], Any]
    keepalive: Any = None


def log_event(record):
    """One append per stage event; completion timestamps are host observations."""
    os.write(1, (json.dumps(record) + '\n').encode())


class PipelineExecutor:
    """Run each request's stage chain; admission lasts through finalization.

    ``finalize(value, error)`` runs even after a failed stage and may handle
    transport/cleanup. Its return value is ignored; the request Future retains
    the stage result or error. Caller-owned resource executors must remain alive
    until shutdown returns. Accepted requests cannot be cancelled midway.
    """

    def __init__(self, executors: Mapping[str, Executor], max_pending: int = 4, *,
                 on_event=log_event):
        if max_pending < 1:
            raise ValueError('max_pending must be positive')
        self.executors = dict(executors)
        self.on_event = on_event
        self.telemetry_errors = 0
        self._slots = threading.BoundedSemaphore(max_pending)
        self._condition = threading.Condition()
        self._active = 0
        self._closed = False
        self._waiters = ThreadPoolExecutor(max_workers=max_pending,
                                           thread_name_prefix='inference-completion')

    def submit(self, stages: Sequence[Stage], value: Any, *,
               identity=None,
               finalize: Callable[[Any, BaseException | None], Any] | None = None,
               finalizer_executor: str = 'io') -> Future:
        stages = tuple(stages)
        for stage in stages:
            if stage.executor not in self.executors:
                raise KeyError(stage.executor)
        if finalize is not None and finalizer_executor not in self.executors:
            raise KeyError(finalizer_executor)
        self._slots.acquire()
        with self._condition:
            if self._closed:
                self._slots.release()
                raise RuntimeError('Pipeline is closed')
            self._active += 1
        result = Future()
        result.set_running_or_notify_cancel()

        def release(_):
            self._slots.release()
            with self._condition:
                self._active -= 1
                self._condition.notify_all()

        result.add_done_callback(release)

        def emit(event, stage, **fields):
            if self.on_event is not None:
                try:
                    self.on_event(dict(event=event, ts=time.time(), identity=identity,
                                       stage=stage.name, executor=stage.executor, **fields))
                except Exception:
                    # Telemetry must never lose a Pending or strand admission.
                    with self._condition:
                        self.telemetry_errors += 1

        def resolve(value, error):
            if error is None:
                result.set_result(value)
            else:
                result.set_exception(error)

        def finish(value, error):
            if finalize is None:
                resolve(value, error)
                return

            def finalized(future):
                try:
                    future.result()
                except BaseException as final_error:
                    resolve(value, final_error)
                else:
                    resolve(value, error)

            try:
                future = self.executors[finalizer_executor].submit(finalize, value, error)
            except BaseException as final_error:
                resolve(value, final_error)
            else:
                future.add_done_callback(finalized)

        def advance(index, value):
            if index == len(stages):
                finish(value, None)
                return
            stage = stages[index]

            def run():
                emit('stage_start', stage)
                try:
                    output = stage.run(value)
                except BaseException:
                    emit('stage_return', stage, ok=False)
                    raise
                emit('stage_return', stage, ok=True, pending=isinstance(output, Pending))
                return output

            def completed(future):
                try:
                    output = future.result()
                    if isinstance(output, Pending):
                        wait = (output.completion if isinstance(output.completion, Future)
                                else self._waiters.submit(output.completion))

                        def ready(completion, held=[output]):
                            # The closure retains backend-owned buffers until the
                            # asynchronous operation really finishes. Empty it so
                            # an externally retained Future does not retain VRAM.
                            pending = held.pop()
                            try:
                                completion.result()
                            except BaseException as error:
                                emit('stage_end', stage, ok=False)
                                finish(pending.value, error)
                            else:
                                emit('stage_end', stage, ok=True)
                                advance(index + 1, pending.value)

                        wait.add_done_callback(ready)
                    else:
                        emit('stage_end', stage, ok=True)
                        advance(index + 1, output)
                except BaseException as error:
                    emit('stage_end', stage, ok=False)
                    finish(value, error)

            try:
                future = self.executors[stage.executor].submit(run)
            except BaseException as error:
                finish(value, error)
            else:
                future.add_done_callback(completed)

        advance(0, value)
        return result

    def shutdown(self):
        """Stop admission and drain every accepted request, including finalizers."""
        with self._condition:
            self._closed = True
            self._condition.wait_for(lambda: self._active == 0)
        self._waiters.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.shutdown()
