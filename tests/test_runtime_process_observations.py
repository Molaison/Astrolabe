"""Omitted /proc entries cannot prove release of a registered live process."""
import errno
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from astrolabe.runtime import guardian, telemetry
from astrolabe.runtime.client import DaemonClient, OutcomeUnknown
from astrolabe.runtime.contracts import ResourceLimits
from astrolabe.runtime.resources import ResourceLedger
from astrolabe.runtime.telemetry import ProcessObservationError, ProcessTree


def record(pid, start, *, group=101):
    return {'pid': pid, 'start': start, 'ppid': 1, 'pgrp': group,
            'rss': 200, 'cpu': 1, 'state': 'S'}


def limits():
    return ResourceLimits(cpu_slots=1, ram_bytes=4096, prepared_bytes=1024, output_bytes=1024)


def test_direct_close_does_not_release_known_live_child_missing_from_snapshot(monkeypatch):
    client = object.__new__(DaemonClient)
    client.process = SimpleNamespace(pid=101, poll=lambda: 0, stdin=None, stdout=None)
    client._namespace_verified, client._closed = True, False
    client._release_error, client.lifecycle = None, {}
    client._processes = {101: (101, 1), 202: (202, 2)}
    client._groups = {101: 1}
    # Preserve the exact original repro: even if a replacement observer omits
    # all entries, the final registered-identity check must prevent release.
    client._observe_children = lambda: ProcessTree({})
    monkeypatch.setattr(telemetry, '_require_process_namespace', lambda: None)
    monkeypatch.setattr(telemetry, '_process_record', lambda pid: record(202, 2) if pid == 202 else None)
    with patch('os.kill') as kill, patch('os.killpg') as killpg:
        with pytest.raises(OutcomeUnknown, match='descendants'):
            client.close()
        with pytest.raises(OutcomeUnknown, match='descendants'):
            client.close()
    assert client.lifecycle['release_confirmed'] is False
    assert kill.call_count + killpg.call_count == 0


def test_cleanup_unknown_resolves_only_after_an_explicit_verified_exit(monkeypatch):
    client = object.__new__(DaemonClient)
    client.process = SimpleNamespace(pid=101, poll=lambda: 0, stdin=None, stdout=None)
    client._namespace_verified, client._closed = True, False
    client._release_error, client.lifecycle = None, {}
    client._processes, client._groups = {101: (101, 1)}, {101: 1}
    client._observe_children = lambda: (_ for _ in ()).throw(ProcessObservationError('temporary read error'))
    with pytest.raises(OutcomeUnknown, match='state unknown'):
        client.close()
    assert client.lifecycle['release_confirmed'] is False
    client._observe_children = lambda: ProcessTree({})
    monkeypatch.setattr(telemetry, 'process_identity', lambda pid: None)
    monkeypatch.setattr(telemetry, 'known_process_records', lambda identities: {})
    client.close()
    client.close()
    assert client.lifecycle['release_confirmed'] is True


def test_resident_accounting_recovers_known_live_pid_omitted_by_snapshot(monkeypatch):
    class MemoryReader:
        def process_memory(self, device, pids):
            # Explicit injected bookkeeping values, not GPU measurements.
            return 200 * len(pids)
    ledger = ResourceLedger(limits(), telemetry=MemoryReader())
    ledger.devices['device'] = {'uuid': 'bookkeeping-fixture', 'total_bytes': 1000, 'mock': True}
    ledger.residents['model'] = {'device': 'device', 'reserved_bytes': 400, 'measured_bytes': 400,
        'processes': {101: (101, 1), 202: (202, 2)}, 'groups': {101: 1},
        'lifecycle': {'gpu_resident': True}, 'unverified_groups': []}
    monkeypatch.setattr(telemetry, '_require_process_namespace', lambda: None)
    monkeypatch.setattr(telemetry, '_process_record', lambda pid: record(pid, 1 if pid == 101 else 2))
    with patch('astrolabe.runtime.resources.process_tree', return_value=ProcessTree({101: record(101, 1)})):
        ledger.observe_resident('model')
        assert ledger.residents['model']['reserved_bytes'] == 400
        assert ledger.release_resident('model') is False


def test_guardian_eof_checks_registered_root_even_if_every_snapshot_omits_it(monkeypatch):
    registration = (json.dumps({'identity': [101, 1], 'group': True}) + '\n').encode()
    alive = {101: True}
    closed, signals = [], []
    monkeypatch.setattr(telemetry, '_require_process_namespace', lambda: None)
    monkeypatch.setattr(telemetry, '_process_record', lambda pid: record(101, 1) if alive[101] else None)
    monkeypatch.setattr(guardian, 'process_tree', lambda *args, **kwargs: ProcessTree({}))
    def terminate(pid, sig):
        signals.append((pid, sig))
        alive[pid] = False
    def close(fd):
        if fd == 202:
            assert not alive[101], 'guardian returned physical ownership while registered root was live'
        closed.append(fd)
    with patch.object(guardian.select, 'select', return_value=([303], [], [])), \
         patch.object(guardian.os, 'read', side_effect=[registration, b'']), \
         patch.object(guardian.os, 'kill', side_effect=terminate), \
         patch.object(guardian.os, 'close', side_effect=close), \
         patch.object(guardian.time, 'sleep', return_value=None):
        guardian.supervise(303, 202)
    assert signals == [(101, signal.SIGTERM)]
    assert closed == [303, 202]


def test_process_tree_backfills_registered_pid_absent_from_enumeration(monkeypatch):
    monkeypatch.setattr(telemetry, '_require_process_namespace', lambda: None)
    monkeypatch.setattr(Path, 'iterdir', lambda self: iter(()))
    monkeypatch.setattr(telemetry, '_process_record', lambda pid: record(pid, 7))
    tree = telemetry.process_tree({101: (101, 7)}, groups={101: 7})
    assert tree[101]['start'] == 7
    assert not tree.unverified_groups


@pytest.mark.parametrize('failure', [PermissionError(errno.EACCES, 'denied'), OSError(errno.EIO, 'I/O error')])
def test_process_permission_and_io_errors_are_unknown_not_exit(monkeypatch, failure):
    monkeypatch.setattr(telemetry, '_require_process_namespace', lambda: None)
    def read(*args, **kwargs):
        raise failure
    monkeypatch.setattr(Path, 'read_text', read)
    with pytest.raises(ProcessObservationError):
        telemetry.process_identity(101)


def test_malformed_stat_is_unknown_and_missing_pid_is_exit(monkeypatch):
    monkeypatch.setattr(telemetry, '_require_process_namespace', lambda: None)
    monkeypatch.setattr(Path, 'read_text', lambda *args, **kwargs: '101 broken')
    with pytest.raises(ProcessObservationError):
        telemetry.process_identity(101)
    def missing(*args, **kwargs):
        raise FileNotFoundError(errno.ENOENT, 'process exited')
    monkeypatch.setattr(Path, 'read_text', missing)
    assert telemetry.process_identity(101) is None


def test_process_namespace_enumeration_error_cannot_be_an_empty_tree(monkeypatch):
    monkeypatch.setattr(telemetry, '_require_process_namespace', lambda: None)
    def denied(self):
        raise PermissionError(errno.EACCES, 'cannot enumerate /proc')
    monkeypatch.setattr(Path, 'iterdir', denied)
    with pytest.raises(ProcessObservationError):
        telemetry.process_tree({101: (101, 1)})


def test_unassociated_unreadable_pid_is_a_discovery_limit_not_an_owned_process(monkeypatch):
    monkeypatch.setattr(telemetry, '_require_process_namespace', lambda: None)
    monkeypatch.setattr(Path, 'iterdir', lambda self: iter([Path('/proc/101'), Path('/proc/202')]))
    def read(pid):
        if pid == 202:
            raise ProcessObservationError('unassociated PID is not readable')
        return record(101, 1)
    monkeypatch.setattr(telemetry, '_process_record', read)
    tree = telemetry.process_tree({101: (101, 1)}, groups={101: 1})
    assert set(tree) == {101}
    assert tree.unobserved_pids == (202,)
    assert not tree.unverified_groups


def test_registered_unreadable_pid_still_fails_closed_during_enumeration(monkeypatch):
    monkeypatch.setattr(telemetry, '_require_process_namespace', lambda: None)
    monkeypatch.setattr(Path, 'iterdir', lambda self: iter([Path('/proc/101')]))
    def denied(pid):
        raise ProcessObservationError('registered PID is not readable')
    monkeypatch.setattr(telemetry, '_process_record', denied)
    with pytest.raises(ProcessObservationError):
        telemetry.process_tree({101: (101, 1)}, groups={101: 1})


@pytest.mark.skipif(not telemetry.process_namespace_supported(), reason='executor PID API and /proc namespaces disagree')
def test_real_cpu_child_keeps_residency_when_tree_observer_omits_it(monkeypatch):
    process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)'], start_new_session=True)
    ledger = ResourceLedger(limits())
    try:
        ledger.reserve_resident('model', 'cpu', 0)
        ledger.bind_process('model', process.pid, managed_group=True)
        monkeypatch.setattr('astrolabe.runtime.resources.process_tree', lambda *args, **kwargs: ProcessTree({}))
        assert ledger.release_resident('model') is False
        assert process.poll() is None
        process.terminate()
        process.wait(timeout=3)
        assert ledger.release_resident('model') is True
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)
        ledger.close()


@pytest.mark.skipif(not telemetry.process_namespace_supported(), reason='executor PID API and /proc namespaces disagree')
def test_real_direct_worker_cannot_confirm_ignored_term_child_from_empty_snapshot(monkeypatch):
    program = """
import subprocess,sys
from astrolabe.runtime.server import serve
child=subprocess.Popen([sys.executable,'-c',
 'import os,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print(os.getpid(),flush=True); time.sleep(20)'],
 stdout=subprocess.PIPE,text=True)
pid=int(child.stdout.readline())
serve(lambda payload:{'child':pid}, {}, load_count=None)
"""
    client = DaemonClient([sys.executable, '-c', program], {}, _direct=True)
    child = client.call({}, timeout=3)['child']
    identity = telemetry.process_identity(child)
    assert identity is not None and child in client._processes
    monkeypatch.setattr(client, '_observe_children', lambda: ProcessTree({}))
    try:
        with pytest.raises(OutcomeUnknown, match='descendants'):
            client.close()
        assert telemetry.process_identity(child) == identity
        assert client.lifecycle['release_confirmed'] is False
    finally:
        if telemetry.process_identity(child) == identity:
            os.kill(child, signal.SIGKILL)
        deadline = time.monotonic() + 3
        while telemetry.process_identity(child) == identity and time.monotonic() < deadline:
            time.sleep(.01)
