"""CPU-only regression for resident AF3 compile and host-submission locking."""

from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from types import SimpleNamespace
import threading

import numpy as np
import pytest


TIMEOUT = 5


class ObservedLock:
    """Keep the runner's real lock and signal a failed nonblocking acquisition."""

    def __init__(self, lock):
        self.lock = lock
        self.contended = threading.Event()

    def __enter__(self):
        if not self.lock.acquire(blocking=False):
            self.contended.set()
            assert self.lock.acquire(timeout=TIMEOUT), "runner lock did not release"
        return self

    def __exit__(self, *exc):
        self.lock.release()


class AsyncArray:
    """An enqueued result whose device completion is independently controlled."""

    def __init__(self):
        self.completion = Future()

    def block_until_ready(self):
        self.completion.result(timeout=TIMEOUT)
        return self

    def __array__(self, dtype=None, copy=None):
        self.block_until_ready()
        return np.asarray([1], dtype=dtype)


@pytest.fixture
def resident_race(monkeypatch):
    # Match the existing AF3 pipeline tests: import the actual wrapper, replacing
    # only model weights/JIT and using NumPy leaves (no devices or model files).
    from astrolabe.evaluators.wrappers.alphafold3 import ModelRunner
    from astrolabe.evaluators import inference_service

    runner = ModelRunner(SimpleNamespace(), SimpleNamespace(platform="cpu"), "unused")
    runner.enable_resident_execution()
    runner._compile_lock = ObservedLock(runner._compile_lock)
    runner._launch_lock = ObservedLock(runner._launch_lock)
    state = SimpleNamespace(
        runner=runner,
        compiling=threading.Event(),
        release_compile=threading.Event(),
        first_launch=threading.Event(),
        release_launch=threading.Event(),
        second_launch=threading.Event(),
        lower_calls=[],
        compile_calls=[],
        submissions=[],
        markers=[],
        results=[AsyncArray(), AsyncArray()],
        active=0,
        peak_active=0,
    )
    guard = threading.Lock()
    weights = object()

    def executable(params, key, batch):
        assert params is weights
        with guard:
            index = len(state.submissions)
            state.submissions.append((key, batch))
            state.active += 1
            state.peak_active = max(state.peak_active, state.active)
        try:
            if index == 0:
                state.first_launch.set()
                assert state.release_launch.wait(TIMEOUT), "first submission not released"
            else:
                assert not state.results[0].completion.done(), (
                    "second submission must precede first device completion"
                )
                state.second_launch.set()
            return state.results[index]
        finally:
            with guard:
                state.active -= 1

    def compile_model():
        state.compile_calls.append(threading.get_ident())
        state.compiling.set()
        assert state.release_compile.wait(TIMEOUT), "cold compile not released"
        return executable

    def lower(*args):
        state.lower_calls.append(args)
        return SimpleNamespace(compile=compile_model)

    def uncached_model(*args):
        raise AssertionError("resident execution bypassed the cached executable")

    uncached_model.lower = lower
    # run_model reads the cached partial's .func/.args; never override run_model.
    runner.__dict__["_model"] = partial(uncached_model, weights)
    monkeypatch.setattr(inference_service, "mark", lambda event, **fields:
                        state.markers.append((event, fields)))
    state.pool = ThreadPoolExecutor(max_workers=2)
    try:
        yield state
    finally:
        # Every blocking fixture operation is bounded; release gates even when
        # an assertion fails, before joining the two worker threads.
        state.release_compile.set()
        state.release_launch.set()
        for result in state.results:
            if not result.completion.done():
                result.completion.set_result(None)
        state.pool.shutdown(wait=True)


def test_resident_serializes_host_launch_not_device_completion(resident_race):
    race = resident_race
    runner = race.runner
    keys = [np.asarray([0, seed], dtype=np.uint32) for seed in (7, 11)]
    batches = [{"tokens": np.asarray([seed, 2], dtype=np.int32)} for seed in (7, 11)]

    first = race.pool.submit(runner.run_model, batches[0], keys[0])
    assert race.compiling.wait(TIMEOUT), "first request did not start cold compilation"
    second = race.pool.submit(runner.run_model, batches[1], keys[1])
    assert runner._compile_lock.contended.wait(TIMEOUT), (
        "second request did not contend on the in-progress signature compile"
    )
    race.release_compile.set()

    assert race.first_launch.wait(TIMEOUT)
    assert runner._launch_lock.contended.wait(TIMEOUT), (
        "second host submission did not contend while the first was inside executable"
    )
    assert len(race.submissions) == 1
    assert race.active == race.peak_active == 1
    assert not race.second_launch.is_set()
    assert not first.done() and not second.done()

    race.release_launch.set()
    assert race.second_launch.wait(TIMEOUT), (
        "host launch lock incorrectly waits for device completion"
    )
    outputs = [first.result(timeout=TIMEOUT), second.result(timeout=TIMEOUT)]
    assert {id(output) for output in outputs} == {id(result) for result in race.results}
    assert not any(output.completion.done() for output in outputs)
    assert race.peak_active == 1 and race.active == 0

    assert len(race.lower_calls) == len(race.compile_calls) == 1
    assert len(runner._resident_executables) == 1
    assert sum(event == "resident_executable_loaded" for event, _ in race.markers) == 1
    dispatch_ids = [fields["executable_id"] for event, fields in race.markers
                    if event == "resident_executable_dispatch"]
    assert len(dispatch_ids) == 2 and len(set(dispatch_ids)) == 1
    for key, batch in race.submissions:
        index = next(i for i, original in enumerate(keys) if key is original)
        assert batch is batches[index]
        np.testing.assert_array_equal(key, [0, (7, 11)[index]])
        np.testing.assert_array_equal(batch["tokens"], [(7, 11)[index], 2])
