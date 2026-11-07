"""Portable Boltz model identity, RNG and child-resource contracts (CPU doubles).

Private launcher/telemetry qualification remains in the frozen experiment suite;
these product checks use only the repository and its execution dependencies.
"""
from contextlib import contextmanager
import os
import random
import struct
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from astrolabe import model_daemon as daemon, model_execution as execution

GPU = 'GPU-11111111-2222-3333-4444-555555555555'


@pytest.fixture(autouse=True)
def cpu_guard(monkeypatch):
    monkeypatch.delenv('ASTROLABE_ALLOCATED_GPU_UUID', raising=False)
    monkeypatch.delenv('ASTROLABE_RESOURCE_PROFILES', raising=False)
    monkeypatch.delenv('ASTROLABE_RESOURCE_SOCKET', raising=False)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: False)
    def deny(*args, **kwargs):
        pytest.fail('CPU contract test attempted CUDA or checkpoint load')
    monkeypatch.setattr(torch.cuda, '_lazy_init', deny)
    monkeypatch.setattr(torch, 'load', deny)
    execution.reset_remote_random_state()
    yield
    execution.reset_remote_random_state()


class LeaseClient:
    socket_path = '/tmp/synthetic-broker'
    def __init__(self):
        self.calls = []
    @contextmanager
    def _connect(self):
        yield SimpleNamespace(getsockopt=lambda *a: struct.pack('3i', os.getpid() + 100, os.getuid(), os.getgid()))
    @contextmanager
    def lease(self, **kwargs):
        self.calls.append(kwargs)
        yield SimpleNamespace(device=GPU)
    def register_model(self, *args, **kwargs):
        pass


def lazy_factory(settings):
    initial = dict(execution._KERNEL_CONTEXT.get())
    late = None
    def handle(request):
        nonlocal late
        if request.get('action') == 'raise':
            with execution.model_kernel_context('inner', 'nested'):
                raise ValueError('nested dispatch failure')
        if late is None:
            late = execution.kernel_handler(settings, lambda cfg: (
                lambda payload: (random.random(), float(np.random.random()), float(torch.rand(()))),
                {'factory_context': initial}), torch_device='cpu', lifecycle={'resource': 'cpu'})
        return late(request)
    return handle


@pytest.mark.parametrize('variant', ['thread', 'local'])
def test_lazy_exact_identity_profile_rng_and_error_context(monkeypatch, tmp_path, variant):
    client = LeaseClient()
    monkeypatch.setattr(execution, '_resource_client', lambda: client)
    monkeypatch.setattr(daemon, '_load_factory', lambda name: lazy_factory)
    random.seed(17); np.random.seed(17); torch.manual_seed(17)
    initial = execution.capture_random_state()
    expected = [(random.random(), float(np.random.random()), float(torch.rand(()))) for _ in range(2)]
    stopped_contexts = []
    owners = []
    def owner_for(tag):
        # Profile is inserted after the existing model identity, as NativeModelClient does.
        spec = daemon.DaemonSpec(backend='boltz1', factory='synthetic:factory',
            settings={'tag': tag, '_initial_random_state': initial}, python_bin=sys.executable)
        identity = daemon.model_identity(spec)
        spec.settings['_runtime_profile'] = {'identity': identity}
        if variant == 'local':
            owner = daemon.LocalModelOwner(spec)
            assert owner.identity == identity
            owners.append(owner)
            return identity, owner.request
        with execution.model_kernel_context(spec.backend, identity):
            handle = lazy_factory(spec.settings)
        server = daemon._Server(handle, socket_path=tmp_path / tag, backend=spec.backend,
            identity=identity, max_pending=4, max_frame_bytes=spec.max_frame_bytes, parent_pid=None)
        def worker():
            server._worker()
            stopped_contexts.append(execution._KERNEL_CONTEXT.get())
        thread = threading.Thread(target=worker)
        thread.start()
        owners.append((server, thread))
        def request(payload):
            item = {'request_id': 'cpu-request', 'generation': 'existing-generation',
                    'payload': payload, 'done': threading.Event()}
            server.tasks.put(item)
            assert item['done'].wait(5)
            reply = item['response']
            if reply['type'] == 'error':
                raise ValueError(reply['message'])
            return reply['result']
        return identity, request
    try:
        with execution.model_kernel_context('outer', 'preserve-caller'):
            ids = []
            for tag in ('a', 'b'):
                identity, request = owner_for(tag)
                ids.append(identity)
                state = initial
                results = []
                for ordinal in range(2):
                    response = request({'operation': 'infer', 'payload': {'ordinal': ordinal},
                        'random_state': state, 'runtime_request_id': f'original-{ordinal}'})
                    assert response['ok']
                    assert response['runtime']['lifecycle']['identity'] == identity
                    assert response['runtime']['lifecycle']['resource_profile']['identity'] == identity
                    assert response['runtime']['request_id'] == f'original-{ordinal}'
                    results.append(response['result'])
                    state = response['random_state']
                assert results == expected
                with pytest.raises(ValueError, match='nested dispatch failure'):
                    request({'action': 'raise'})
                description = request({'operation': 'describe'})
                assert description['metadata']['factory_context']['identity'] == identity
                assert description['metadata']['runtime_lifecycle']['identity'] == identity
                assert execution._KERNEL_CONTEXT.get() == {'backend': 'outer', 'identity': 'preserve-caller'}
            assert ids[0] != ids[1]
            assert {call['identity'] for call in client.calls} == set(ids)
        assert execution._KERNEL_CONTEXT.get() is None
    finally:
        for owner in owners:
            if variant == 'local': owner.shutdown()
            else:
                server, thread = owner
                item = {'stop': True, 'done': threading.Event()}
                server.tasks.put(item); thread.join(5)
                assert not thread.is_alive()
        if variant == 'thread': assert stopped_contexts == [None, None]


@pytest.mark.parametrize('variant', ['thread', 'local'])
def test_actual_boltz_lazy_factory_keeps_original_identity(monkeypatch, tmp_path, variant):
    from astrolabe.evaluators import native_daemons
    from astrolabe.evaluators.wrappers import boltz
    calls = []
    class Runner:
        def __init__(self, config):
            calls.append('construct')
        def run_inference(self, data, out_dir):
            self.model_module = getattr(self, 'model_module', object())
            calls.append((id(self.model_module), random.random(), float(np.random.random()), float(torch.rand(()))))
    monkeypatch.setattr(boltz, 'Boltz1Runner', Runner)
    monkeypatch.setattr(daemon, '_load_factory', lambda name: native_daemons._boltz_factory)
    client = LeaseClient()
    monkeypatch.setattr(execution, '_resource_client', lambda: client)
    random.seed(17); np.random.seed(17); torch.manual_seed(17)
    initial = execution.capture_random_state()
    expected = [(random.random(), float(np.random.random()), float(torch.rand(()))) for _ in (1, 2)]
    spec = daemon.DaemonSpec(backend='boltz1', factory='astrolabe.evaluators.native_daemons:_boltz_factory',
        settings={'config': {'accelerator': 'cpu', 'devices': 1}, '_initial_random_state': initial},
        python_bin=sys.executable)
    identity = daemon.model_identity(spec)
    spec.settings['_runtime_profile'] = {'identity': identity}
    if variant == 'local':
        owner = daemon.LocalModelOwner(spec)
        request = owner.request
    else:
        with execution.model_kernel_context('boltz1', identity):
            handle = native_daemons._boltz_factory(spec.settings)
        server = daemon._Server(handle, socket_path=tmp_path / 'cpu', identity=identity,
            backend='boltz1', max_pending=4, max_frame_bytes=spec.max_frame_bytes, parent_pid=None)
        thread = threading.Thread(target=server._worker)
        thread.start()
        def request(payload):
            item = {'request_id': 'original', 'generation': 'original', 'payload': payload,
                    'done': threading.Event()}
            server.tasks.put(item)
            assert item['done'].wait(5)
            assert item['response']['type'] == 'result', item['response']
            return item['response']['result']
    try:
        state = initial
        for ordinal in (1, 2):
            response = request({'operation': 'infer', 'runtime_request_id': f'original-{ordinal}',
                'random_state': state, 'payload': {'operation': 'run_inference', 'data': ordinal,
                                                'out_dir': tmp_path}})
            assert response['ok']
            assert response['runtime']['lifecycle']['identity'] == identity != 'boltz1'
            assert response['runtime']['lifecycle']['resource_profile']['identity'] == identity
            assert response['runtime']['scientific_requests'] == ordinal
            state = response['random_state']
        assert calls[0] == 'construct' and len(calls) == 3
        assert calls[1][0] == calls[2][0]
        assert [row[1:] for row in calls[1:]] == expected
        assert {row['identity'] for row in client.calls} == {identity}
    finally:
        if variant == 'local': owner.shutdown()
        else:
            server.tasks.put({'stop': True, 'done': threading.Event()})
            thread.join(5)
            assert not thread.is_alive()


def test_canonical_uuid_and_broker_environment_reach_daemon_children(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', GPU)
    monkeypatch.setenv('ASTROLABE_ALLOCATED_GPU_UUID', GPU)
    monkeypatch.setenv('ASTROLABE_RESOURCE_SOCKET', '/tmp/original-broker')
    owner = object.__new__(daemon.OwnedModelDaemon)
    owner.spec = daemon.DaemonSpec(backend='boltz1', factory='synthetic:factory', settings={})
    child = owner._child_env()
    assert child['CUDA_VISIBLE_DEVICES'] == child['ASTROLABE_ALLOCATED_GPU_UUID'] == GPU
    assert child['ASTROLABE_RESOURCE_SOCKET'] == '/tmp/original-broker'
    assert child['ASTROLABE_MODEL_PARENT_PID'] == str(os.getpid())
