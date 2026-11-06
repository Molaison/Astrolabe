"""CPU contracts for real thread-pool scheduling, completion and RPC transport.

These tests do not emulate a GPU or qualify either model's numerical outputs.
"""
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack
import json
import gc
from pathlib import Path
import subprocess
import sys
import threading
import time
import weakref

import pytest

from astrolabe.inference_pipeline import Pending, PipelineExecutor, Stage


def resources(stack):
    return {name: stack.enter_context(ThreadPoolExecutor(max_workers=count,
                                                        thread_name_prefix=name))
            for name, count in (('cpu', 2), ('owner', 1), ('io', 2))}


def test_repeated_cpu_owner_dependencies_and_thread_ownership():
    seen = []
    events = []

    def operation(name, fn):
        def run(value):
            seen.append((name, threading.get_ident(), value))
            return fn(value)
        return run

    with ExitStack() as stack:
        pools = resources(stack)
        with PipelineExecutor(pools, max_pending=2, on_event=events.append) as pipeline:
            stages = [Stage('parse', 'cpu', operation('parse', json.loads)),
                      Stage('square', 'owner', operation('square', lambda xs: [x*x for x in xs])),
                      Stage('reduce', 'cpu', operation('reduce', sum)),
                      Stage('double', 'owner', operation('double', lambda x: x*2)),
                      Stage('encode', 'io', operation('encode', json.dumps))]
            assert pipeline.submit(stages, '[1, 2, 3]', identity='trajectory-1').result(timeout=3) == '28'
    assert [name for name, _, _ in seen] == ['parse', 'square', 'reduce', 'double', 'encode']
    assert seen[1][1] == seen[3][1]
    assert seen[0][1] != seen[1][1]
    assert all(event['identity'] == 'trajectory-1' for event in events)
    assert [event['stage'] for event in events if event['event'] == 'stage_end'] == [
        'parse', 'square', 'reduce', 'double', 'encode']


def test_async_completion_does_not_hold_owner_or_advance_early():
    completion = Future()
    first_dispatched = threading.Event()
    first_advanced = threading.Event()

    def dispatch(value):
        if value == 1:
            first_dispatched.set()
            return Pending(value, completion, keepalive={'input': value})
        return value

    def after(value):
        if value == 1:
            first_advanced.set()
        return value * 10

    with ExitStack() as stack:
        pools = resources(stack)
        with PipelineExecutor(pools, max_pending=2) as pipeline:
            stages = [Stage('dispatch', 'owner', dispatch), Stage('after', 'cpu', after)]
            first = pipeline.submit(stages, 1)
            try:
                assert first_dispatched.wait(2)
                second = pipeline.submit(stages, 2)
                assert second.result(timeout=2) == 20
                assert not first_advanced.is_set()
                assert not first.cancel()
            finally:
                completion.set_result(None)
            assert first.result(timeout=2) == 10


def test_pending_lifetime_ends_even_when_completion_future_is_retained():
    class Buffer:
        pass

    completion = Future()
    dispatched = threading.Event()
    refs = []

    def dispatch(value):
        buffer = Buffer()
        refs.append(weakref.ref(buffer))
        dispatched.set()
        return Pending(value, completion, keepalive=buffer)

    with ExitStack() as stack:
        pools = resources(stack)
        with PipelineExecutor(pools, max_pending=1) as pipeline:
            result = pipeline.submit([Stage('dispatch', 'owner', dispatch)], 3)
            assert dispatched.wait(2)
            assert refs[0]() is not None
            completion.set_result(None)
            assert result.result(timeout=2) == 3
    gc.collect()
    assert refs[0]() is None


def test_admission_is_held_until_finalizer_finishes():
    entered = threading.Event()
    release = threading.Event()
    submitted = threading.Event()

    def finalize(value, error):
        assert error is None
        entered.set()
        assert release.wait(3)

    with ExitStack() as stack:
        pools = resources(stack)
        submitter = stack.enter_context(ThreadPoolExecutor(max_workers=1))
        with PipelineExecutor(pools, max_pending=1) as pipeline:
            first = pipeline.submit([Stage('sum', 'cpu', sum)], [1, 2], finalize=finalize)
            assert entered.wait(2)

            def second_submit():
                future = pipeline.submit([Stage('sum', 'cpu', sum)], [3, 4])
                submitted.set()
                return future

            second = submitter.submit(second_submit)
            try:
                assert not submitted.wait(0.05)
            finally:
                release.set()
            assert first.result(timeout=2) == 3
            assert second.result(timeout=2).result(timeout=2) == 7


@pytest.mark.parametrize('asynchronous', [False, True])
def test_errors_finalize_and_shutdown_drains(asynchronous):
    finalized = threading.Event()
    release = threading.Event()
    shutdown_done = threading.Event()
    caught = []

    def fail(value):
        def operation():
            return int(value)
        return Pending(value, operation) if asynchronous else operation()

    def finalize(value, error):
        caught.append(error)
        finalized.set()
        assert release.wait(3)

    with ExitStack() as stack:
        pools = resources(stack)
        supervisor = stack.enter_context(ThreadPoolExecutor(max_workers=1))
        pipeline = PipelineExecutor(pools, max_pending=2)
        failed = pipeline.submit([Stage('parse', 'owner', fail)], 'invalid', finalize=finalize)
        assert finalized.wait(2)
        valid = pipeline.submit([Stage('sum', 'cpu', sum)], [3, 5])

        def shutdown():
            pipeline.shutdown()
            shutdown_done.set()

        stopped = supervisor.submit(shutdown)
        try:
            assert valid.result(timeout=2) == 8
            assert not shutdown_done.wait(0.05)
        finally:
            release.set()
        with pytest.raises(ValueError):
            failed.result(timeout=2)
        stopped.result(timeout=2)
        assert isinstance(caught[0], ValueError)
        with pytest.raises(RuntimeError, match='closed'):
            pipeline.submit([], None)


def test_staged_rpc_preserves_response_and_error_protocol(tmp_path):
    from astrolabe.evaluators.inference_service import call, serve_staged
    socket = tmp_path / 'service.sock'

    def calculate(item):
        item['value'] = sum(item['payload']['values'])
        return item

    with ExitStack() as stack:
        pools = resources(stack)
        service = stack.enter_context(ThreadPoolExecutor(max_workers=1))
        pipeline = PipelineExecutor(pools, max_pending=2)
        future = service.submit(serve_staged, lambda request: [Stage('calculate', 'cpu', calculate)],
                                socket, pipeline, max_pending=2)
        deadline = time.monotonic() + 3
        while not socket.exists() and not future.done() and time.monotonic() < deadline:
            time.sleep(0.005)
        assert socket.exists()
        try:
            assert call(socket, dict(values=[2, 3], identity='good')) == 5
            with pytest.raises(RuntimeError, match='TypeError'):
                call(socket, dict(values=['invalid'], identity='bad'))
        finally:
            assert call(socket, dict(type='shutdown')) is None
        future.result(timeout=3)


def test_generic_and_rpc_imports_do_not_load_model_runtimes():
    code = '''
import sys
import astrolabe.inference_pipeline
import astrolabe.inference_backends
import astrolabe.evaluators.inference_service
import astrolabe.evaluators.af3_pipeline
assert "torch" not in sys.modules
assert "jax" not in sys.modules
assert "numpy" not in sys.modules
assert "scipy" not in sys.modules
'''
    subprocess.run([sys.executable, '-c', code], check=True,
                   cwd=Path(__file__).resolve().parents[1], timeout=5)


def test_telemetry_failure_does_not_lose_pending_or_strand_shutdown():
    completion = Future()
    dispatched = threading.Event()
    def broken_event(record):
        raise OSError('injected telemetry failure')
    def dispatch(value):
        dispatched.set()
        return Pending(value, completion)
    with ExitStack() as stack:
        pools = resources(stack)
        pipeline = PipelineExecutor(pools, max_pending=1, on_event=broken_event)
        future = pipeline.submit([Stage('dispatch','owner',dispatch)], 42)
        assert dispatched.wait(2) and not future.done()
        completion.set_result(None)
        assert future.result(timeout=2) == 42
        pipeline.shutdown()
        assert pipeline.telemetry_errors >= 3
