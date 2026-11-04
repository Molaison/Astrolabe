"""Transport/lifecycle regression checks using actual CPU subprocesses."""
import sys
import threading

import pytest

from astrolabe.runtime.broker_client import BrokerClient
from astrolabe.runtime.client import DaemonClient, OutcomeUnknown, WorkerError
from astrolabe.runtime.contracts import OperationResources


def test_unknown_native_outcome_survives_jsonl_protocol():
    program = """
from astrolabe.runtime.client import OutcomeUnknown
from astrolabe.runtime.server import serve
def handler(payload):
    raise OutcomeUnknown('native command may already have executed')
serve(handler, {}, load_count=None)
"""
    with DaemonClient([sys.executable, '-c', program], {}, _direct=True) as client:
        with pytest.raises(OutcomeUnknown, match='may already have executed'):
            client.call({'scientific_request': 'same-parent'}, timeout=3)
        assert client.process.poll() is not None


def test_private_release_failure_is_not_silently_acknowledged():
    client = object.__new__(BrokerClient)
    client._lock, client._closed = threading.Lock(), False
    client._close_lock = threading.Lock()
    client.handle, client.epoch = 'private', 'client'
    def lost_broker(*args, **kwargs):
        raise WorkerError('broker connection failed')
    client._rpc = lost_broker
    with pytest.raises(OutcomeUnknown, match='release outcome unknown'):
        client.close(release=True)
    assert client.lifecycle['process_alive'] is None
    assert client.lifecycle['gpu_resident'] is None
    # A second cleanup path cannot turn the first uncertain release into a
    # successful no-op. No inference or cleanup RPC is silently replayed.
    with pytest.raises(OutcomeUnknown, match='release outcome unknown'):
        client.close(release=True)


def test_private_release_requires_confirmed_cleanup_response():
    client = object.__new__(BrokerClient)
    client._lock, client._closed = threading.Lock(), False
    client._close_lock = threading.Lock()
    client.handle, client.epoch = 'private', 'client'
    client._rpc = lambda *args, **kwargs: {'released': False}
    with pytest.raises(OutcomeUnknown):
        client.close(release=True)
    with pytest.raises(OutcomeUnknown):
        client.close(release=True)


def test_detached_shared_handle_cannot_pretend_private_release_was_confirmed():
    client = object.__new__(BrokerClient)
    client._lock, client._closed = threading.Lock(), False
    client._close_lock = threading.Lock()
    client.handle, client.epoch = 'shared', 'client'
    client._rpc = lambda *args, **kwargs: {'released': False}
    client.close()
    with pytest.raises(OutcomeUnknown, match='never confirmed'):
        client.close(release=True)


def test_confirmed_private_release_is_idempotent():
    client = object.__new__(BrokerClient)
    client._lock, client._closed = threading.Lock(), False
    client._close_lock = threading.Lock()
    client.handle, client.epoch = 'private', 'client'
    responses = []
    def release(*args, **kwargs):
        responses.append(True)
        return {'released': True}
    client._rpc = release
    client.close(release=True)
    client.close(release=True)
    assert responses == [True]
    assert client.lifecycle['release_confirmed'] is True
    assert client.lifecycle['process_alive'] is False


def test_direct_cleanup_retains_uncertain_descendant_state_on_second_close():
    from types import SimpleNamespace
    from astrolabe.runtime.telemetry import ProcessTree
    client = object.__new__(DaemonClient)
    client.process = SimpleNamespace(pid=123, poll=lambda: 0, stdin=None, stdout=None)
    client._namespace_verified, client._closed = False, False
    client._release_error, client.lifecycle = None, {}
    client._observe_children = lambda: ProcessTree({}, unverified_groups=[123])
    with pytest.raises(OutcomeUnknown, match='descendants'):
        client.close()
    with pytest.raises(OutcomeUnknown, match='descendants'):
        client.close()
    assert client.lifecycle['release_confirmed'] is False


def test_unknown_ram_is_not_an_explicit_zero_byte_estimate():
    assert OperationResources.parse().ram_bytes is None
    assert OperationResources.parse({'ram_bytes': 0}).ram_bytes == 0
    with pytest.raises(ValueError, match='cpu_slots'):
        OperationResources.parse({'cpu_slots': 0})


def test_explicit_cpu_probe_masks_gpu_even_when_parent_has_allocation(monkeypatch):
    from astrolabe.runtime.admission import run_probe
    monkeypatch.delenv('ASTROLABE_EXECUTION_ENDPOINT', raising=False)
    monkeypatch.delenv('ASTROLABE_EXECUTION_TOKEN', raising=False)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'allocated-but-not-used')
    result = run_probe([sys.executable, '-c',
                        'import json,os; print(json.dumps({"visible":os.environ["CUDA_VISIBLE_DEVICES"],'
                        '"jax":os.environ["JAX_PLATFORMS"]}))'], device='cpu')
    assert result == {'visible': '', 'jax': 'cpu'}
