"""B-owned ingress checks; CPU/local sockets only, no model installation."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import threading
import time
import uuid

import pytest

from astrolabe.runtime.broker import Handler, Server
from astrolabe.runtime.broker_client import BrokerClient
from astrolabe.runtime.broker_transport import (encode_response, endpoint_for, read_line,
                                               rejection, unix_address)
from astrolabe.runtime.client import AdmissionRejected, ApplicationError, OutcomeUnknown


class IngressState:
    token = 'ingress-test'

    def __init__(self):
        self.business_started = threading.Event()
        self.nested_started = threading.Event()
        self.finish = threading.Event()
        self.calls = []

    def snapshot(self):
        return {}

    def probe(self, req):
        self.calls.append(req)
        self.business_started.set()
        assert self.finish.wait(3)
        return {'value': 'business'}

    def nested(self, req, *, finish=False):
        self.calls.append(req)
        self.nested_started.set()
        assert self.finish.wait(3)
        return {'lease': 'nested'}

    def cancel(self, req):
        self.calls.append(req)
        return {'cancel_requested': True}

    def detach(self, req):
        self.calls.append(req)
        return {'released': False}

    def stage(self, req):
        return {}


@contextmanager
def ingress_servers(capacity=2, read_timeout=.4):
    base = '@astrolabe-ingress-test-' + uuid.uuid4().hex
    state = IngressState()
    servers, threads = {}, []
    try:
        for lane, op in [('business', 'call'), ('control', 'metrics'), ('nested', 'nested_begin')]:
            server = Server(unix_address(endpoint_for(base, op)), Handler, lane=lane,
                            capacity=capacity, state=state, read_timeout=read_timeout)
            servers[lane] = server
        for server in servers.values():
            server.lanes = servers
            thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01})
            thread.start()
            threads.append(thread)
        proxy = object.__new__(BrokerClient)
        proxy.endpoint, proxy.token, proxy.limit = base, state.token, 1048576
        yield proxy, servers, state
    finally:
        state.finish.set()
        for server in servers.values():
            server.interrupt()
            server.shutdown()
        for server in servers.values():
            server.server_close()
        for thread in threads:
            thread.join(2)
            assert not thread.is_alive()
        assert all(server.snapshot()['in_flight'] == 0 for server in servers.values())


def until(predicate):
    deadline = time.monotonic() + 2
    while not predicate():
        assert time.monotonic() < deadline
        threading.Event().wait(.005)


@pytest.fixture(autouse=True)
def allocation(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.delenv('ASTROLABE_CPU_OWNER', raising=False)
    monkeypatch.delenv('ASTROLABE_ALLOCATED_CUDA_DEVICES', raising=False)


@pytest.mark.parametrize('op,lane', [('probe', 'business'), ('metrics', 'control'), ('nested_begin', 'nested')])
def test_each_lane_rejects_at_accept_without_executing(op, lane):
    with ingress_servers(capacity=1) as (proxy, servers, state):
        held = socket.socket(socket.AF_UNIX)
        try:
            held.connect(unix_address(endpoint_for(proxy.endpoint, op)))
            until(lambda: servers[lane].snapshot()['in_flight'] == 1)
            with pytest.raises(AdmissionRejected) as error:
                proxy._rpc({'op': op, 'epoch': 'epoch', 'request_id': 'scientific'}, 1, submitted=True)
            assert error.value.execution_state == 'not_executed'
            assert error.value.retryable is True
            assert state.calls == []
            snapshot = servers[lane].snapshot()
            assert snapshot == {'capacity': 1, 'in_flight': 1, 'high_water': 1, 'rejected': 1}
            for other in set(servers) - {lane}:
                assert servers[other].snapshot()['in_flight'] == 0
        finally:
            held.close()


def test_blocked_business_and_nested_leave_cancel_metrics_detach_available():
    with ingress_servers(capacity=1) as (proxy, servers, state), ThreadPoolExecutor(2) as pool:
        business = pool.submit(proxy._rpc, {'op': 'probe'}, 2, submitted=True)
        assert state.business_started.wait(1)
        nested = pool.submit(proxy._rpc, {'op': 'nested_begin', 'owner_pid': os.getpid(),
                                         'worker_epoch': 'worker', 'parent_request_id': 'parent'}, 2, submitted=True)
        assert state.nested_started.wait(1)
        metrics = proxy._rpc({'op': 'metrics'}, 1, submitted=False)['metrics']['ingress']
        assert metrics['business']['in_flight'] == metrics['nested']['in_flight'] == 1
        assert metrics['control']['capacity'] == 1
        assert proxy._rpc({'op': 'cancel', 'target_request_id': 'parent'}, 1, submitted=True)['cancel_requested']
        assert proxy._rpc({'op': 'detach'}, 1, submitted=True)['released'] is False
        state.finish.set()
        assert business.result()['result']['value'] == 'business'
        assert nested.result()['lease'] == 'nested'


def test_same_parent_cannot_consume_two_nested_handlers():
    with ingress_servers(capacity=2) as (proxy, servers, state), ThreadPoolExecutor(1) as pool:
        req = {'op': 'nested_begin', 'owner_pid': os.getpid(), 'worker_epoch': 'worker', 'parent_request_id': 'p'}
        nested = pool.submit(proxy._rpc, dict(req), 2, submitted=True)
        assert state.nested_started.wait(1)
        with pytest.raises(AdmissionRejected):
            proxy._rpc(dict(req), 1, submitted=True)
        assert len(state.calls) == 1
        state.finish.set()
        assert nested.result()['lease'] == 'nested'


def test_business_operation_on_control_socket_is_explicitly_rejected():
    with ingress_servers() as (proxy, servers, state), socket.socket(socket.AF_UNIX) as sock:
        sock.settimeout(1)
        sock.connect(unix_address(endpoint_for(proxy.endpoint, 'metrics')))
        sock.sendall(encode_response({'op': 'call', 'token': state.token, 'allocated_cuda_devices': ''}))
        result = json.loads(read_line(sock, 4096, time.monotonic() + 1))
        assert result['error_type'] == 'AdmissionRejected'
        assert result['execution_state'] == 'not_executed' and result['retryable'] is True
        assert state.calls == []


def test_long_paths_and_abstract_addresses_route_deterministically():
    for base in ['/tmp/' + 'x' * 100, '@' + 'x' * 106, '@normal', '/tmp/normal']:
        endpoints = {endpoint_for(base, op) for op in ['call', 'metrics', 'nested_begin']}
        assert len(endpoints) == 3
        assert all(len(os.fsencode(endpoint)) <= 107 for endpoint in endpoints)
        assert endpoint_for(base, 'call') == endpoint_for(base, 'probe')
        assert endpoint_for(base, 'cancel') == endpoint_for(base, 'detach')
        assert endpoint_for(base, 'nested_begin') == endpoint_for(base, 'nested_end')


def test_read_deadline_releases_slot_and_shutdown_interrupts_partial_messages():
    with ingress_servers(capacity=1, read_timeout=.1) as (proxy, servers, state):
        with socket.socket(socket.AF_UNIX) as sock:
            sock.connect(unix_address(endpoint_for(proxy.endpoint, 'metrics')))
            sock.sendall(b'{')
            until(lambda: servers['control'].snapshot()['high_water'] == 1)
            until(lambda: servers['control'].snapshot()['in_flight'] == 0)
        assert proxy._rpc({'op': 'metrics'}, 1, submitted=False)['ok'] is True
        # Context cleanup must interrupt the incomplete producer and join it.
        held = socket.socket(socket.AF_UNIX)
        held.connect(unix_address(endpoint_for(proxy.endpoint, 'call')))
        until(lambda: servers['business'].snapshot()['in_flight'] == 1)
    held.close()


def test_client_has_no_call_queue_cancel_and_close_bypass_call_lock():
    proxy = object.__new__(BrokerClient)
    proxy._lock, proxy._close_lock = threading.Lock(), threading.Lock()
    proxy._closed, proxy._used = False, set()
    proxy.handle, proxy.epoch, proxy.limit = 'handle', 'epoch', 4096
    proxy.lifecycle, proxy.hello = {}, {}
    entered, finish = threading.Event(), threading.Event()
    seen = []

    def rpc(req, timeout, *, submitted):
        seen.append(req)
        if req['op'] == 'call':
            entered.set()
            assert finish.wait(2)
            return {'result': {}, 'request_id': req['request_id'], 'epoch': req['epoch']}
        return {'released': False}
    proxy._rpc = rpc
    with ThreadPoolExecutor(1) as pool:
        call = pool.submit(proxy.call, {}, request_id='first', timeout=2)
        assert entered.wait(1)
        try:
            with pytest.raises(AdmissionRejected):
                proxy.call({}, request_id='second', timeout=2)
            assert 'second' not in proxy._used
            proxy.cancel('first')
            assert seen[-1] == {'op': 'cancel', 'handle': 'handle', 'epoch': 'epoch', 'target_request_id': 'first'}
            proxy.close()
            assert proxy._release_confirmed is False
        finally:
            finish.set()
        assert call.result() == {}


def test_rejected_request_id_is_reusable_but_transport_loss_is_unknown():
    with ingress_servers(capacity=1) as (proxy, servers, state):
        proxy._lock, proxy._close_lock = threading.Lock(), threading.Lock()
        proxy._closed, proxy._used = False, set()
        proxy.handle, proxy.epoch, proxy.lifecycle, proxy.hello = 'h', 'e', {}, {}
        with socket.socket(socket.AF_UNIX) as held:
            held.connect(unix_address(endpoint_for(proxy.endpoint, 'call')))
            until(lambda: servers['business'].snapshot()['in_flight'] == 1)
            with pytest.raises(AdmissionRejected):
                proxy.call({}, timeout=1, request_id='same-id')
            assert 'same-id' not in proxy._used
    with pytest.raises(OutcomeUnknown):
        proxy._rpc({'op': 'call'}, .1, submitted=True)


def test_application_error_type_survives_wire():
    with ingress_servers() as (proxy, servers, state):
        def fail(req):
            raise ApplicationError('explicit application failure')
        state.probe = fail
        with pytest.raises(ApplicationError, match='explicit application failure'):
            proxy._rpc({'op': 'probe'}, 1, submitted=True)


def test_shutdown_revalidates_group_anchor_before_each_signal(monkeypatch):
    import signal
    from astrolabe.runtime import broker, telemetry
    signalled = []
    monkeypatch.setattr(broker.os, 'killpg', lambda pid, sig: signalled.append((pid, sig)))
    monkeypatch.setattr(broker.os, 'getpgid', lambda pid: pid)
    observed = {'start': 'original'}
    monkeypatch.setattr(telemetry, 'process_identity', lambda pid: (pid, observed['start']))
    anchors = {12345: 'original'}
    broker._signal_groups(anchors, signal.SIGTERM)
    observed['start'] = 'reused-pid'
    broker._signal_groups(anchors, signal.SIGKILL)
    assert signalled == [(12345, signal.SIGTERM)]
    observed['start'] = 'original'
    monkeypatch.setattr(broker.os, 'getpgid', lambda pid: 54321)
    broker._signal_groups(anchors, signal.SIGKILL)
    assert signalled == [(12345, signal.SIGTERM)]


def test_session_exit_interrupts_active_probe_and_joins_servers(tmp_path):
    import sys
    from astrolabe.runtime.session import BrokerSession
    marker = tmp_path / 'probe-started'
    program = "import pathlib,sys,time; pathlib.Path(sys.argv[1]).touch(); time.sleep(60)"
    with ThreadPoolExecutor(1) as pool:
        with BrokerSession(resource_limits={'pending_items': 1}) as session:
            proxy = object.__new__(BrokerClient)
            proxy.endpoint, proxy.token, proxy.limit = session.endpoint, session.token, 1048576
            future = pool.submit(proxy._rpc, {'op': 'probe', 'command': [sys.executable, '-c', program, str(marker)],
                                              'env': {}, 'timeout': 60, 'device': 'cpu',
                                              'epoch': 'probe', 'request_id': 'shutdown'}, 60, submitted=True)
            until(marker.exists)
            stopped = time.monotonic()
        assert time.monotonic() - stopped < 5
        assert session.process.returncode == 0
        with pytest.raises(OutcomeUnknown):
            future.result(timeout=2)
