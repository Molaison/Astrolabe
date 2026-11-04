"""Real CPU transports and mocked physical-device labels, never GPU evidence."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import threading
import time

import pytest

from astrolabe.runtime.broker_state import State
from astrolabe.runtime.client import AdmissionRejected, ApplicationError, OutcomeUnknown
from astrolabe.runtime.contracts import ResourceLimits

FIXTURE = Path(__file__).with_name('runtime_request_isolation_a_fixture.py')


@pytest.fixture
def owner_state(monkeypatch):
    monkeypatch.delenv('ASTROLABE_EXECUTION_ENDPOINT', raising=False)
    monkeypatch.delenv('ASTROLABE_EXECUTION_TOKEN', raising=False)
    monkeypatch.delenv('ASTROLABE_RUNTIME_TRACE', raising=False)
    state = State('request-a-test', 8, 8, limits=ResourceLimits(
        cpu_slots=2, ram_bytes=1024**3, prepared_bytes=1024**2,
        output_bytes=1024**2, pending_items=8))
    yield state
    state.close()
    assert not state.requests
    assert not state.ledger.pending
    assert not state.ledger.active


def register(state, name='one', epoch='a'):
    identity = {'environment': 'request-a', 'code': name, 'weights': 'none', 'device': 'cpu'}
    return state.register({'command': [sys.executable, str(FIXTURE)], 'identity': identity,
        'env': {'REQUEST_ISOLATION_A_IDENTITY': json.dumps(identity),
                'PYTHONPATH': str(FIXTURE.parents[1]) + os.pathsep + os.environ.get('PYTHONPATH', '')},
        'limit': 1048576, 'resources': {}, 'owner_pid': os.getpid(),
        'client_epoch': epoch, 'startup_timeout': 10})['handle']


def req(handle, rid, *, epoch='a', timeout=3, runtime=None, **payload):
    return {'handle': handle, 'epoch': epoch, 'request_id': rid, 'payload': payload,
            'timeout': timeout, 'runtime': runtime or {'resources': {'output_bytes': 0, 'ram_bytes': 128}}}


def cancelling(state, handle, rid, epoch='a'):
    return state.cancel({'handle': handle, 'epoch': epoch, 'target_request_id': rid})


def started_call(monkeypatch, state, key):
    started = threading.Event()
    client = state.workers[key]['client']
    original = client._read
    def read(*args, **kwargs):
        started.set()  # request bytes are already written; cancellation is now active.
        return original(*args, **kwargs)
    monkeypatch.setattr(client, '_read', read)
    return started


@pytest.mark.parametrize('nested_first', [False, True])
def test_timeout_waiter_preserves_managed_owner(owner_state, monkeypatch, nested_first):
    state = owner_state
    key = register(state)
    register(state, epoch='b')
    started = started_call(monkeypatch, state, key)
    runtime = {'phase': 'managed', 'resources': {'output_bytes': 0, 'ram_bytes': 128}}
    with ThreadPoolExecutor(2) as pool:
        active = pool.submit(state.call, req(key, 'owner', sleep=.35, method='managed', runtime=runtime))
        assert started.wait(2)
        parent = state.workers[key]['delegated']
        nested = {'owner_pid': state.workers[key]['client'].process.pid,
                  'parent_request_id': parent['dispatch_id'],
                  'worker_epoch': state.workers[key]['client'].hello['worker_epoch']}
        if nested_first:
            state.nested(nested)
        with pytest.raises(TimeoutError):
            state.call(req(key, 'waiter', epoch='b', timeout=.03))
        assert state.workers[key]['delegated'] is parent
        if not nested_first:
            state.nested(nested)
        state.nested(nested, finish=True)
        assert active.result(3)['result']['value'] is None
    assert state.metrics['completed'] == 1
    assert state.metrics['waiting'] == 0


def test_cpu_application_error_keeps_other_worker_healthy(owner_state):
    state = owner_state
    bad, good = register(state), register(state, 'two')
    with pytest.raises(ApplicationError):
        state.call(req(bad, 'bad', fail=True))
    assert not state.poisoned
    assert state.call(req(good, 'healthy', value=42))['result']['value'] == 42
    assert state.call(req(bad, 'recovered', value=43))['result']['value'] == 43


@pytest.mark.parametrize('failure', [{'sleep': 1}, {'exit': True}])
@pytest.mark.parametrize('gpu', [False, True])
def test_transport_unknown_scopes_fence(owner_state, failure, gpu):
    state = owner_state
    bad, good = register(state), register(state, 'two')
    if gpu:
        for key in (bad, good):
            state.workers[key]['device'] = 'mock-physical-a'
            state.ledger.residents[key]['device'] = 'mock-physical-a'
        state.ledger.devices['mock-physical-a'] = {'uuid': 'mock-physical-a', 'mock': True, 'total_bytes': None}
    with pytest.raises(OutcomeUnknown):
        state.call(req(bad, 'unknown', timeout=.06, **failure))
    assert state.workers[bad]['fenced']
    assert state.workers[bad]['client'].process.poll() is not None
    with pytest.raises(OutcomeUnknown, match='fenced'):
        state.call(req(bad, 'bad-again'))
    if gpu:
        with pytest.raises(OutcomeUnknown, match='fenced'):
            state.call(req(good, 'blocked'))
        assert state.poisoned == {'mock-physical-a'}
    else:
        assert state.call(req(good, 'healthy', value=99))['result']['value'] == 99
        assert 'cpu' not in state.poisoned


def test_cancel_queued_waiter_does_not_cancel_active(owner_state, monkeypatch):
    state = owner_state
    key = register(state)
    register(state, epoch='b')
    started = started_call(monkeypatch, state, key)
    queued = threading.Event()
    original = state._check_request
    def check(request):
        if request is state.requests.get(('b', 'queued')):
            queued.set()
        return original(request)
    monkeypatch.setattr(state, '_check_request', check)
    with ThreadPoolExecutor(2) as pool:
        active = pool.submit(state.call, req(key, 'owner', sleep=.35, value=1))
        assert started.wait(2)
        parent = state.workers[key]['delegated']
        waiter = pool.submit(state.call, req(key, 'queued', epoch='b', timeout=None))
        assert queued.wait(2)
        ack = cancelling(state, key, 'queued', epoch='b')
        assert ack == {'cancelled': True, 'state': 'queued', 'release_confirmed': False}
        with pytest.raises(AdmissionRejected) as caught:
            waiter.result(1)
        assert caught.value.execution_state == 'not_executed' and caught.value.retryable
        assert state.workers[key]['delegated'] is parent
        assert active.result(3)['result']['value'] == 1
        assert cancelling(state, key, 'owner')['cancelled'] is False
    assert state.call(req(key, 'next', value=2))['result']['value'] == 2
    assert state.call(req(key, 'queued', epoch='b', value=3))['result']['value'] == 3


def test_active_cancel_is_identity_scoped_and_owner_cleans(owner_state, monkeypatch):
    state = owner_state
    key = register(state)
    register(state, epoch='b')
    started = started_call(monkeypatch, state, key)
    with ThreadPoolExecutor(1) as pool:
        active = pool.submit(state.call, req(key, 'owner', sleep=10, timeout=None))
        assert started.wait(2)
        assert cancelling(state, key, 'owner', epoch='b')['cancelled'] is False
        assert cancelling(state, key, 'unrelated')['cancelled'] is False
        ack = cancelling(state, key, 'owner')
        assert ack['cancelled'] and not ack['release_confirmed']
        with pytest.raises(OutcomeUnknown):
            active.result(3)
    assert state.workers[key]['client'].lifecycle['release_confirmed'] is True
    assert state.workers[key]['delegated'] is None
    assert not state.ledger.pending and not state.requests


@pytest.mark.parametrize('stage', ['pending', 'compute'])
def test_cancel_wakes_resource_wait(owner_state, monkeypatch, stage):
    state = owner_state
    key = register(state)
    if stage == 'pending':
        state.ledger.limits = ResourceLimits(cpu_slots=2, ram_bytes=1024**3,
            prepared_bytes=1024**2, output_bytes=1024**2, pending_items=1)
    state.ledger.admit_pending('blocker', {'cpu_slots': 2, 'output_bytes': 0})
    if stage == 'compute':
        state.ledger.acquire_compute('blocker', 'cpu')
    waiting = threading.Event()
    original = state.ledger._wait
    def wait(predicate, deadline, reason, cancel_token=None):
        if not predicate():
            waiting.set()
        return original(predicate, deadline, reason, cancel_token)
    monkeypatch.setattr(state.ledger, '_wait', wait)
    try:
        with ThreadPoolExecutor(1) as pool:
            pending = pool.submit(state.call, req(key, 'pending', timeout=None))
            assert waiting.wait(2)
            ack = cancelling(state, key, 'pending')
            assert ack['state'] == 'queued' and not ack['release_confirmed']
            with pytest.raises(AdmissionRejected):
                pending.result(1)
        assert state.workers[key]['client'].process.poll() is None
    finally:
        state.ledger.release('blocker')


def test_pending_and_compute_share_absolute_deadline(owner_state, monkeypatch):
    state = owner_state
    key = register(state)
    state.ledger.limits = ResourceLimits(cpu_slots=2, ram_bytes=1024**3,
        prepared_bytes=1024**2, output_bytes=1024**2, pending_items=1)
    state.ledger.admit_pending('pending-block', {'output_bytes': 0})
    waiting = threading.Event()
    original = state.ledger._wait
    deadlines = []
    def wait(predicate, deadline, reason, cancel_token=None):
        if cancel_token:
            deadlines.append(deadline)
            if not predicate():
                waiting.set()
        return original(predicate, deadline, reason, cancel_token)
    monkeypatch.setattr(state.ledger, '_wait', wait)
    # Hold both CPU slots independently of the pending blocker.
    state.ledger.cpu_slots = 2
    timer = threading.Timer(.12, state.ledger.release, args=('pending-block',))
    start = time.monotonic()
    timer.start()
    try:
        with pytest.raises(TimeoutError):
            state.call(req(key, 'deadline', timeout=.22))
        elapsed = time.monotonic() - start
        assert len(deadlines) == 2 and deadlines[0] == deadlines[1]
        assert elapsed < .32  # .1 s scheduling tolerance, old two-stage wait was .34 s.
        assert not state.workers[key]['client']._used  # no daemon submission
    finally:
        timer.join()
        state.ledger.cpu_slots = 0
        state.ledger.release('pending-block')


def test_expired_ready_admission_never_dispatches(owner_state, monkeypatch):
    state = owner_state
    key = register(state)
    original = state.ledger.acquire_compute
    def acquire(*args, **kwargs):
        result = original(*args, **kwargs)
        time.sleep(.04)
        return result
    monkeypatch.setattr(state.ledger, 'acquire_compute', acquire)
    with pytest.raises(TimeoutError):
        state.call(req(key, 'expired', timeout=.02))
    assert not state.workers[key]['client']._used


def test_gpu_application_error_still_fences_device(owner_state):
    state = owner_state
    bad, good = register(state), register(state, 'two')
    for key in (bad, good):
        state.workers[key]['device'] = 'mock-physical-a'
        state.ledger.residents[key]['device'] = 'mock-physical-a'
    state.ledger.devices['mock-physical-a'] = {'uuid': 'mock-physical-a', 'mock': True, 'total_bytes': None}
    with pytest.raises(ApplicationError):
        state.call(req(bad, 'application', fail=True))
    assert state.poisoned == {'mock-physical-a'}
    assert state.workers[bad]['client'].lifecycle['release_confirmed'] is True
    with pytest.raises(OutcomeUnknown, match='fenced'):
        state.call(req(good, 'blocked'))


def test_uncertain_cleanup_retains_lease_until_confirmed(owner_state, monkeypatch):
    state = owner_state
    key = register(state)
    client = state.workers[key]['client']
    original = client.close
    def unconfirmed(**kwargs):
        client.lifecycle['release_confirmed'] = False
        raise OutcomeUnknown('synthetic cleanup not confirmed')
    monkeypatch.setattr(client, 'close', unconfirmed)
    try:
        with pytest.raises(OutcomeUnknown):
            state.call(req(key, 'unknown', sleep=1, timeout=.03))
        assert state.ledger.pending
        assert state.ledger.active
        assert state.ledger.residents
        assert state.retain_artifacts
        assert not state.requests
    finally:
        monkeypatch.setattr(client, 'close', original)
    state.detach({'handle': key, 'client_epoch': 'a', 'release': True})
    assert not state.ledger.pending and not state.ledger.residents


def test_request_record_admission_is_bounded(owner_state, monkeypatch):
    state = owner_state
    key = register(state)
    state.request_limit = 1
    started = started_call(monkeypatch, state, key)
    with ThreadPoolExecutor(1) as pool:
        active = pool.submit(state.call, req(key, 'owner', sleep=.15))
        assert started.wait(2)
        with pytest.raises(AdmissionRejected) as rejected:
            state.call(req(key, 'excess'))
        assert rejected.value.execution_state == 'not_executed' and rejected.value.retryable
        assert len(state.requests) == 1
        active.result(2)
    assert not state.requests
    assert state.call(req(key, 'excess', value=4))['result']['value'] == 4
    with pytest.raises(ValueError, match='duplicate'):
        state.call(req(key, 'excess', value=5))


def test_unknown_request_id_is_never_reusable(owner_state):
    state = owner_state
    key = register(state)
    with pytest.raises(OutcomeUnknown):
        state.call(req(key, 'uncertain-id', timeout=.03, sleep=1))
    with pytest.raises(ValueError, match='duplicate'):
        state.call(req(key, 'uncertain-id', value='must not replay'))


def test_cancel_staged_execute_preserves_prepared_owner(owner_state, monkeypatch):
    state = owner_state
    identity = {'device': 'cpu', 'code': 'staged-request-a', 'weights': 'none'}
    fixture = FIXTURE.with_name('runtime_staged_fixture.py')
    key = state.register({'command': [sys.executable, str(fixture)], 'identity': identity,
        'env': {'RUNTIME_TEST_IDENTITY': json.dumps(identity),
                'PYTHONPATH': str(FIXTURE.parents[1]) + os.pathsep + os.environ.get('PYTHONPATH', '')},
        'limit': 1048576, 'startup_timeout': 10, 'owner_pid': os.getpid(), 'client_epoch': 'a'})['handle']
    def stage(method, rid):
        phase = {'prepare': 'cpu_prepare', 'execute': 'composite', 'finalize': 'cpu_finalize'}[method]
        return req(key, rid, timeout=None, method=method, key='science-a', runtime={
            'phase': phase, 'scientific_request_id': 'science-a',
            'resources': {'prepared_bytes': 256, 'output_bytes': 0, 'ram_bytes': 128}})
    state.call(stage('prepare', 'prep'))
    retained = state.workers[key]['retained'][('a', 'science-a')]
    state.ledger.admit_pending('cpu-blocker', {'cpu_slots': 2, 'output_bytes': 0})
    state.ledger.acquire_compute('cpu-blocker', 'cpu')
    waiting = threading.Event()
    original = state.ledger._wait
    def wait(predicate, deadline, reason, cancel_token=None):
        if cancel_token and not predicate():
            waiting.set()
        return original(predicate, deadline, reason, cancel_token)
    monkeypatch.setattr(state.ledger, '_wait', wait)
    try:
        with ThreadPoolExecutor(1) as pool:
            pending = pool.submit(state.call, stage('execute', 'exec'))
            assert waiting.wait(2)
            cancelling(state, key, 'exec')
            with pytest.raises(AdmissionRejected):
                pending.result(1)
        assert state.workers[key]['retained'][('a', 'science-a')] is retained
        assert retained['token'] in state.ledger.pending
        assert retained['token'] not in state.ledger.active
    finally:
        state.ledger.release('cpu-blocker')
    assert state.call(stage('execute', 'exec'))['result'] == {'executed': True}
    assert state.call(stage('finalize', 'final'))['result'] == {'value': 8128}


def test_cancel_wakes_nested_admission_and_drains_owner(owner_state, monkeypatch):
    state = owner_state
    key = register(state)
    state.workers[key]['device'] = 'mock-physical-a'
    state.ledger.residents[key]['device'] = 'mock-physical-a'
    state.ledger.devices['mock-physical-a'] = {'uuid': 'mock-physical-a', 'mock': True, 'total_bytes': None}
    state.ledger.admit_pending('device-blocker', {'output_bytes': 0, 'ram_bytes': 128})
    state.ledger.acquire_compute('device-blocker', 'mock-physical-a', cpu_work=False)
    started = started_call(monkeypatch, state, key)
    waiting = threading.Event()
    original = state.ledger._wait
    def wait(predicate, deadline, reason, cancel_token=None):
        if cancel_token and not predicate():
            waiting.set()
        return original(predicate, deadline, reason, cancel_token)
    monkeypatch.setattr(state.ledger, '_wait', wait)
    runtime = {'phase': 'managed', 'resources': {'output_bytes': 0, 'ram_bytes': 128}}
    try:
        with ThreadPoolExecutor(2) as pool:
            active = pool.submit(state.call, req(key, 'owner', timeout=None, sleep=10,
                                                method='managed', runtime=runtime))
            assert started.wait(2)
            parent = state.workers[key]['delegated']
            nested = {'owner_pid': state.workers[key]['client'].process.pid,
                      'parent_request_id': parent['dispatch_id'],
                      'worker_epoch': state.workers[key]['client'].hello['worker_epoch']}
            lease = pool.submit(state.nested, nested)
            assert waiting.wait(2)
            assert cancelling(state, key, 'owner')['release_confirmed'] is False
            with pytest.raises(AdmissionRejected):
                lease.result(1)
            with pytest.raises(OutcomeUnknown):
                active.result(3)
        assert state.workers[key]['delegated'] is None
        assert set(state.ledger.pending) == {'device-blocker'}
        assert not state.requests
    finally:
        state.ledger.release('device-blocker')


def test_absolute_wire_deadline_overrides_longer_relative_timeout(owner_state):
    state = owner_state
    key = register(state)
    request = req(key, 'expired-wire', timeout=10, value='never dispatch')
    request['deadline'] = time.monotonic() - .01
    with pytest.raises(TimeoutError):
        state.call(request)
    assert not state.workers[key]['client']._used


@pytest.mark.parametrize('kind', ['wrong-id', 'bad-runtime', 'bad-error'])
def test_invalid_response_is_unknown_and_closes_worker(owner_state, monkeypatch, kind):
    state = owner_state
    key = register(state)
    client = state.workers[key]['client']
    def response(*args, **kwargs):
        rid = next(iter(client._used))
        result = {'type': 'result', 'request_id': rid, 'epoch': client.epoch,
                  'result': {}, 'runtime': {'worker_epoch': client.hello['worker_epoch']}}
        if kind == 'wrong-id':
            result['request_id'] = 'incorrect'
        elif kind == 'bad-runtime':
            result['runtime'] = [1]
        else:
            result.update(type='error', error={'invalid': True})
        return result
    monkeypatch.setattr(client, '_read', response)
    with pytest.raises(OutcomeUnknown):
        state.call(req(key, 'bad-response'))
    assert state.workers[key]['fenced']
    assert client.process.poll() is not None
    assert client.lifecycle['release_confirmed'] is True


def test_gpu_application_failure_without_release_keeps_protection(owner_state, monkeypatch):
    state = owner_state
    key, healthy = register(state), register(state, 'two')
    for handle in (key, healthy):
        state.workers[handle]['device'] = 'mock-physical-a'
        state.ledger.residents[handle]['device'] = 'mock-physical-a'
    state.ledger.devices['mock-physical-a'] = {'uuid': 'mock-physical-a', 'mock': True, 'total_bytes': None}
    client = state.workers[key]['client']
    close = client.close
    def unconfirmed(**kwargs):
        client.lifecycle['release_confirmed'] = False
        raise OutcomeUnknown('GPU work quiescence not confirmed')
    monkeypatch.setattr(client, 'close', unconfirmed)
    try:
        with pytest.raises(ApplicationError):
            state.call(req(key, 'known-application-error', fail=True))
        assert state.ledger.pending and state.ledger.active
        assert client.process.poll() is None
        assert state.poisoned == {'mock-physical-a'}
        with pytest.raises(OutcomeUnknown, match='fenced'):
            state.call(req(healthy, 'protected'))
    finally:
        monkeypatch.setattr(client, 'close', close)
    state.detach({'handle': key, 'client_epoch': 'a', 'release': True})
    assert not state.ledger.pending


def test_late_cancel_cannot_kill_successor(owner_state, monkeypatch):
    state = owner_state
    key = register(state)
    state.call(req(key, 'completed', value=1))
    started = started_call(monkeypatch, state, key)
    with ThreadPoolExecutor(1) as pool:
        active = pool.submit(state.call, req(key, 'successor', sleep=.1, value=2))
        assert started.wait(2)
        assert cancelling(state, key, 'completed')['cancelled'] is False
        assert state.workers[key]['client'].process.poll() is None
        assert active.result(2)['result']['value'] == 2


def test_exceptional_nested_end_fences_before_yield_and_owner_cleans(owner_state, monkeypatch):
    state = owner_state
    key = register(state)
    state.workers[key]['device'] = 'mock-physical-review'
    state.ledger.residents[key]['device'] = 'mock-physical-review'
    state.ledger.devices['mock-physical-review'] = {'uuid': 'mock-physical-review', 'mock': True, 'total_bytes': None}
    started = started_call(monkeypatch, state, key)
    client = state.workers[key]['client']
    cleanup_entered, cleanup_allowed = threading.Event(), threading.Event()
    original = client.close
    def slow_close(**kwargs):
        cleanup_entered.set()
        assert cleanup_allowed.wait(3)
        return original(**kwargs)
    monkeypatch.setattr(client, 'close', slow_close)
    runtime = {'phase': 'managed', 'resources': {'output_bytes': 0, 'ram_bytes': 128}}
    with ThreadPoolExecutor(1) as pool:
        active = pool.submit(state.call, req(key, 'exception-owner', timeout=None, sleep=10,
                                            method='managed', runtime=runtime))
        try:
            assert started.wait(2)
            parent = state.workers[key]['delegated']
            nested = {'owner_pid': client.process.pid, 'parent_request_id': parent['dispatch_id'],
                      'worker_epoch': client.hello['worker_epoch']}
            state.nested(nested)
            token = parent['token']
            with pytest.raises(OutcomeUnknown, match='exceptional device subgraph'):
                state.nested({**nested, 'error': 'failure after enqueue'}, finish=True)
            assert cleanup_entered.wait(1)
            assert parent['nested'] and parent['uncertain']
            assert token in state.ledger.active and token in state.ledger.pending
            assert state.poisoned == {'mock-physical-review'}
            assert client.process.poll() is None
            state.ledger.admit_pending('other-review', {'output_bytes': 0, 'ram_bytes': 1})
            with pytest.raises(OutcomeUnknown, match='fenced'):
                state.ledger.acquire_compute('other-review', 'mock-physical-review', timeout=.02)
        finally:
            cleanup_allowed.set()
            state.ledger.release('other-review')
        with pytest.raises(OutcomeUnknown):
            active.result(3)
    assert client.lifecycle['release_confirmed'] is True
    assert not state.ledger.pending and not state.ledger.active


@pytest.mark.parametrize('operation', ['call', 'cancel', 'register', 'probe', 'metrics'])
def test_global_state_lock_wait_obeys_absolute_deadline(owner_state, operation):
    state = owner_state
    key = register(state)
    held, release = threading.Event(), threading.Event()
    def hold():
        with state.lock:
            held.set()
            release.wait(2)
    owner = threading.Thread(target=hold)
    owner.start()
    assert held.wait(1)
    before = time.monotonic()
    deadline = before + .02
    try:
        with pytest.raises(TimeoutError):
            if operation == 'call':
                state.call({**req(key, 'blocked-state', timeout=.02), 'deadline': deadline})
            elif operation == 'cancel':
                state.cancel({'handle': key, 'epoch': 'a', 'target_request_id': 'blocked-state', 'deadline': deadline})
            elif operation == 'probe':
                state.probe({'epoch': 'a', 'request_id': 'blocked-probe', 'timeout': .02,
                             'deadline': deadline, 'device': 'cpu', 'env': {}, 'command': ['unused']})
            elif operation == 'metrics':
                state.snapshot_for_request({'deadline': deadline})
            else:
                identity = {'device': 'cpu', 'code': 'blocked-register'}
                state.register({'identity': identity, 'command': ['unused'], 'env': {}, 'limit': 1048576,
                                'startup_timeout': .02, 'deadline': deadline,
                                'client_epoch': 'a', 'owner_pid': os.getpid()})
        assert time.monotonic() - before < .12  # .1s scheduling allowance
    finally:
        release.set()
        owner.join(1)
    assert not state.requests


def test_slow_detach_cleanup_leaves_unrelated_controls_and_calls_free(owner_state, monkeypatch):
    state = owner_state
    slow, good = register(state), register(state, 'review-good')
    client = state.workers[slow]['client']
    closing, release = threading.Event(), threading.Event()
    original = client.close
    def slow_close(**kwargs):
        closing.set()
        assert release.wait(3)
        return original(**kwargs)
    monkeypatch.setattr(client, 'close', slow_close)
    with ThreadPoolExecutor(1) as pool:
        cleanup = pool.submit(state.detach, {'handle': slow, 'client_epoch': 'a', 'release': True})
        try:
            assert closing.wait(1)
            assert state.workers[slow]['closing']
            assert state.snapshot()['waiting'] == 0
            assert not cancelling(state, good, 'absent')['cancelled']
            assert state.call(req(good, 'during-close', timeout=.2, value=7))['result']['value'] == 7
        finally:
            release.set()
        assert cleanup.result(3)['released']


def test_rejected_startup_identities_leave_no_lock_history(owner_state):
    state = owner_state
    state.max_resident_models = state.max_resident_per_device = 1
    register(state)
    from astrolabe.runtime.resources import ResourceExhausted
    for index in range(12):
        with pytest.raises(ResourceExhausted):
            register(state, 'rejected-review-' + str(index))
        assert not state.start_locks
    assert len(state.workers) == 1


def test_cpu_probe_failure_does_not_fence_healthy_workers(owner_state):
    state = owner_state
    key = register(state)
    with pytest.raises(OutcomeUnknown):
        state.probe({'epoch': 'a', 'request_id': 'failed-review-probe', 'timeout': .5,
                     'device': 'cpu', 'env': {}, 'command': [sys.executable, '-c', 'raise SystemExit(3)']})
    assert 'cpu' not in state.poisoned
    assert state.call(req(key, 'after-probe', value=9))['result']['value'] == 9


@pytest.mark.parametrize('operation', ['nested', 'nested_process'])
def test_nested_gate_wait_uses_wire_deadline(owner_state, monkeypatch, operation):
    state = owner_state
    key = register(state)
    started = started_call(monkeypatch, state, key)
    runtime = {'phase': 'managed', 'resources': {'output_bytes': 0, 'ram_bytes': 128}}
    with ThreadPoolExecutor(1) as pool:
        active = pool.submit(state.call, req(key, 'gate-owner', timeout=None, sleep=.2,
                                            method='managed', runtime=runtime))
        assert started.wait(2)
        parent = state.workers[key]['delegated']
        nested = {'owner_pid': state.workers[key]['client'].process.pid,
                  'worker_epoch': state.workers[key]['client'].hello['worker_epoch'],
                  'parent_request_id': parent['dispatch_id']}
        if operation == 'nested_process':
            state.nested(nested)
        parent['owner']['lease_lock'].acquire()
        before = time.monotonic()
        try:
            with pytest.raises(TimeoutError):
                getattr(state, operation)({**nested, 'pid': os.getpid(), 'deadline': before + .02})
            assert time.monotonic() - before < .12
            assert parent['nested'] is (operation == 'nested_process')
        finally:
            with state.condition:
                parent['owner']['lease_lock'].release()
                state.condition.notify_all()
        if operation == 'nested_process':
            state.nested(nested, finish=True)
        assert active.result(2)['result']['value'] is None


def test_concurrent_same_identity_start_locks_are_retired(owner_state):
    state = owner_state
    with ThreadPoolExecutor(3) as pool:
        futures = [pool.submit(register, state, 'shared-review', epoch) for epoch in ('a', 'b', 'c')]
        keys = [future.result(3) for future in futures]
    assert len(set(keys)) == 1
    assert len(state.workers) == 1
    assert set(state.workers[keys[0]]['clients']) == {'a', 'b', 'c'}
    assert not state.start_locks and not state.starting


def test_socket_metrics_propagates_state_lock_deadline(owner_state):
    import socket
    import uuid
    from astrolabe.runtime.broker import Handler, Server
    state = owner_state
    endpoint = '\0request-review-metrics-' + uuid.uuid4().hex
    server = Server(endpoint, Handler, lane='control', capacity=1, state=state)
    serving = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01})
    serving.start()
    held, release = threading.Event(), threading.Event()
    def hold():
        with state.lock:
            held.set()
            release.wait(2)
    owner = threading.Thread(target=hold)
    owner.start()
    assert held.wait(1)
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(.2)
            sock.connect(endpoint)
            sock.sendall((json.dumps({'op': 'metrics', 'token': state.token,
                                     'allocated_cuda_devices': os.environ.get('CUDA_VISIBLE_DEVICES') or '',
                                     'deadline': time.monotonic() + .02}) + '\n').encode())
            response = json.loads(sock.makefile('rb').readline())
            assert response['error_type'] == 'TimeoutError'
            assert not response['ok'] and not response['unknown']
    finally:
        release.set()
        owner.join(1)
        server.shutdown()
        server.server_close()
        serving.join(1)


@pytest.mark.parametrize('parent_earlier', [False, True])
def test_nested_compute_receives_minimum_wire_parent_deadline(owner_state, monkeypatch, parent_earlier):
    state = owner_state
    key = register(state)
    started = started_call(monkeypatch, state, key)
    runtime = {'phase': 'managed', 'resources': {'output_bytes': 0, 'ram_bytes': 128}}
    with ThreadPoolExecutor(1) as pool:
        active = pool.submit(state.call, req(key, 'compute-deadline-owner', timeout=3, sleep=.2,
                                            method='managed', runtime=runtime))
        assert started.wait(2)
        parent = state.workers[key]['delegated']
        wire = parent['deadline'] + (1 if parent_earlier else -1)
        expected = min(wire, parent['deadline'])
        received = []
        original = state.ledger.acquire_compute
        def acquire(*args, **kwargs):
            received.append(kwargs['deadline'])
            return original(*args, **kwargs)
        monkeypatch.setattr(state.ledger, 'acquire_compute', acquire)
        nested = {'owner_pid': state.workers[key]['client'].process.pid,
                  'worker_epoch': state.workers[key]['client'].hello['worker_epoch'],
                  'parent_request_id': parent['dispatch_id'], 'deadline': wire}
        state.nested(nested)
        state.nested(nested, finish=True)
        assert received == [expected, expected]
        assert active.result(2)['result']['value'] is None
