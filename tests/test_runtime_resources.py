"""Exercise ownership/resource invariants. Mock devices are not GPU evidence."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import pytest

from astrolabe.runtime.contracts import ResourceLimits, identity_key, performance_identity
from astrolabe.runtime.resources import DeviceOwned, PhysicalGPUOwner, ResourceExhausted, ResourceLedger
from astrolabe.runtime.telemetry import process_namespace_supported


def limits(**overrides):
    values = dict(cpu_slots=2, ram_bytes=4096, prepared_bytes=1024, output_bytes=1024, pending_items=2)
    values.update(overrides)
    return ResourceLimits(**values)


class MeasuredFixture:
    def resolve(self, token, *, mock=False):
        return {'token': token, 'uuid': 'GPU-resource-fixture', 'total_bytes': 1000, 'mock': True}

    def process_memory(self, device, pids):
        return 200


def test_physical_ownership_survives_tmpdir_aliases_and_is_released(monkeypatch, tmp_path):
    monkeypatch.setenv('ASTROLABE_GPU_OWNER_DIRECTORY', str(tmp_path))
    monkeypatch.setenv('TMPDIR', str(tmp_path / 'session_one'))
    device = {'uuid': 'GPU-shared-physical', 'token': '0'}
    owner = PhysicalGPUOwner(device)
    program = """
from astrolabe.runtime.resources import PhysicalGPUOwner, DeviceOwned
try:
 lease = PhysicalGPUOwner({'uuid':'GPU-shared-physical','token':'GPU-shared-physical'})
except DeviceOwned:
 raise SystemExit(37)
lease.close()
"""
    env = dict(os.environ, TMPDIR=str(tmp_path / 'session_two'))
    try:
        assert subprocess.run([sys.executable, '-c', program], env=env, timeout=5).returncode == 37
    finally:
        owner.close()
    assert subprocess.run([sys.executable, '-c', program], env=env, timeout=5).returncode == 0


def test_resident_and_scratch_budgets_have_distinct_lifetimes(monkeypatch, tmp_path):
    monkeypatch.setenv('ASTROLABE_GPU_OWNER_DIRECTORY', str(tmp_path))
    ledger = ResourceLedger(limits(), telemetry=MeasuredFixture())
    try:
        device = ledger.device('0')
        ledger.reserve_resident('model', device, None)
        assert ledger.snapshot()['resident']['model']['reserved_bytes'] == 900
        with pytest.raises(ResourceExhausted):
            ledger.reserve_resident('other', device, 1)
        ledger.update_resident('model', measured_bytes=200)
        ledger.reserve_resident('other', device, 300)
        with ledger.lease('request', device, {'scratch_bytes': 350, 'output_bytes': 0}):
            snapshot = ledger.snapshot()
            assert snapshot['active']['request']['scratch_bytes'] == 350
            assert sum(r['reserved_bytes'] for r in snapshot['resident'].values()) == 500
        assert not ledger.snapshot()['active']
        assert len(ledger.snapshot()['resident']) == 2
        with pytest.raises(ResourceExhausted, match='scratch'):
            with ledger.lease('too_large', device, {'scratch_bytes': 401, 'output_bytes': 0}):
                pytest.fail('over-budget request entered compute')
    finally:
        ledger.close()


def test_ready_queue_reserves_bytes_before_compute_and_releases_on_timeout():
    ledger = ResourceLedger(limits())
    ledger.admit_pending('first', {'prepared_bytes': 800, 'output_bytes': 0})
    with pytest.raises(TimeoutError):
        ledger.admit_pending('second', {'prepared_bytes': 400, 'output_bytes': 0}, timeout=.01)
    assert ledger.snapshot()['prepared_bytes'] == 800
    assert not ledger.snapshot()['active']  # waiting for GPU uses no CPU slot
    ledger.release('first')
    ledger.admit_pending('second', {'prepared_bytes': 400, 'output_bytes': 0}, timeout=.1)
    ledger.release('second')
    assert ledger.snapshot()['prepared_bytes'] == 0


def test_unknown_outputs_use_bounded_per_request_caps_without_serializing_preparation():
    ledger = ResourceLedger(limits())
    ledger.admit_pending('first', {'prepared_bytes': 128, 'output_bytes': None})
    ledger.admit_pending('second', {'prepared_bytes': 128, 'output_bytes': None}, timeout=.1)
    snapshot = ledger.snapshot()
    assert snapshot['pending'] == 2
    assert snapshot['output_reserved_bytes'] == 1024
    assert snapshot['prepared_bytes'] == 256
    ledger.release('first')
    ledger.release('second')


def test_composite_work_on_two_devices_still_obeys_one_cpu_slot():
    ledger = ResourceLedger(limits(cpu_slots=1))
    ledger.devices = {name: {'uuid': name, 'mock': True, 'total_bytes': 1000} for name in ('a', 'b')}
    for token in ('a', 'b'):
        ledger.admit_pending(token, {'output_bytes': 0})
    try:
        ledger.acquire_compute('a', 'a')
        with pytest.raises(TimeoutError, match='compute'):
            ledger.acquire_compute('b', 'b', timeout=.01)
        assert ledger.snapshot()['cpu_slots'] == 1
        ledger.release_compute('a')
        ledger.acquire_compute('a', 'a', cpu_work=False)
        ledger.acquire_compute('b', 'b', timeout=.01)
        assert ledger.snapshot()['cpu_slots'] == 1
        assert len(ledger.snapshot()['active']) == 2
    finally:
        ledger.release('a')
        ledger.release('b')


def test_actual_memory_pressure_blocks_loading_and_compute_without_fictional_headroom():
    class Observation:
        sample_value = {'available': True, 'used_bytes': 750, 'total_bytes': 1000}
        def sample(self, device):
            return dict(self.sample_value)
    observed = Observation()
    ledger = ResourceLedger(limits(), telemetry=observed)
    ledger.devices['physical'] = {'uuid': 'physical', 'total_bytes': 1000, 'mock': False}
    with pytest.raises(ResourceExhausted, match='headroom'):
        ledger.reserve_resident('oversize', 'physical', 151)
    ledger.reserve_resident('model', 'physical', 100)
    ledger.admit_pending('request', {'scratch_bytes': 151, 'output_bytes': 0})
    with pytest.raises(ResourceExhausted, match='headroom'):
        ledger.acquire_compute('request', 'physical')
    observed.sample_value = {'available': True, 'used_bytes': 901, 'total_bytes': 1000}
    with pytest.raises(ResourceExhausted, match='actual GPU'):
        ledger.acquire_compute('request', 'physical')
    observed.sample_value = {'available': False}
    with pytest.raises(ResourceExhausted, match='unavailable'):
        ledger.reserve_resident('unobserved', 'physical', 1)
    ledger.release('request')


def test_performance_cohort_removes_only_ownership_nonces():
    first = {'device': {'platform': 'gpu', 'visible': '0', 'id': 0, 'name': 'A100'},
             'model_config': {'instance': 'private-a', 'trajectory': 'one', 'seed': 7, 'precision': 'bf16'},
             'weights': 'sha256:weights-one', 'environment': 'exact-runtime'}
    second = json.loads(json.dumps(first))
    second['model_config'].update(instance='private-b', trajectory='two')
    second['device']['visible'] = '1'
    assert identity_key(first) != identity_key(second)  # models are still private
    assert performance_identity(first) == performance_identity(second)
    for path, value in ((('model_config', 'seed'), 8), (('model_config', 'precision'), 'float32'),
                        (('device', 'name'), 'H100'), (('weights',), 'sha256:weights-two')):
        changed = json.loads(json.dumps(second))
        parent = changed
        for name in path[:-1]:
            parent = parent[name]
        parent[path[-1]] = value
        assert performance_identity(first) != performance_identity(changed)


def test_output_roots_account_job_files_and_shared_cache_growth_once(monkeypatch, tmp_path):
    from astrolabe.runtime.broker_state import State
    artifacts, outputs, shared = (tmp_path / name for name in ('artifacts', 'outputs', 'cache'))
    for path in (artifacts, outputs / 'nested', shared):
        path.mkdir(parents=True)
    (artifacts / 'input').write_bytes(b'12')
    (outputs / 'nested' / 'result').write_bytes(b'1234567')
    (shared / 'existing').write_bytes(b'1234567890')
    monkeypatch.setenv('ASTROLABE_EXECUTION_ARTIFACT_ROOT', str(artifacts))
    state = State('test', 1, 1, limits=limits())
    try:
        state._register_output_roots('backend', {'output_roots': [
            {'path': str(outputs), 'scope': 'job'},
            {'path': str(outputs / 'nested'), 'scope': 'job'},
            {'path': str(shared), 'scope': 'shared_cache'}]})
        (shared / 'new').write_bytes(b'12345')
        usage = state.output_usage()
        assert usage['external_job_bytes'] == 7
        assert usage['shared_cache_growth_bytes'] == 5
        assert usage['external_output_roots'][str(shared)]['baseline_bytes'] == 10
        assert (shared / 'existing').read_bytes() == b'1234567890'
    finally:
        state.close()


@pytest.mark.skipif(not process_namespace_supported(), reason='executor PID API and /proc namespaces disagree')
def test_guardian_retains_owner_after_broker_kill_until_close_fds_descendant_exits(monkeypatch, tmp_path):
    from astrolabe.runtime.telemetry import process_identity
    monkeypatch.setenv('ASTROLABE_GPU_OWNER_DIRECTORY', str(tmp_path))
    grandchild = "import os,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print(os.getpid(),flush=True); time.sleep(15)"
    worker = "import subprocess,sys; p=subprocess.Popen([sys.executable,'-c',sys.argv[1]],close_fds=True); p.wait()"
    parent = """
import json,os,signal,subprocess,sys
from astrolabe.runtime.resources import PhysicalGPUOwner
owner=PhysicalGPUOwner({'uuid':'GPU-guardian-fixture','token':'fixture'},guarded=True)
p=subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]],stdout=subprocess.PIPE,text=True,start_new_session=True)
child=int(p.stdout.readline())
owner.register_process(p.pid,group=True)
print(json.dumps({'worker':p.pid,'child':child,'guardian':owner.guardian.pid}),flush=True)
signal.pause()
"""
    broker = subprocess.Popen([sys.executable, '-c', parent, worker, grandchild], stdout=subprocess.PIPE, text=True)
    record = json.loads(broker.stdout.readline())
    try:
        broker.kill()
        broker.wait(timeout=2)
        assert process_identity(record['child']) is not None
        with pytest.raises(DeviceOwned):
            PhysicalGPUOwner({'uuid': 'GPU-guardian-fixture', 'token': 'same-physical'})
        deadline = time.monotonic() + 6
        while process_identity(record['child']) is not None and time.monotonic() < deadline:
            time.sleep(.05)
        assert process_identity(record['child']) is None
        while time.monotonic() < deadline:
            try:
                owner = PhysicalGPUOwner({'uuid': 'GPU-guardian-fixture', 'token': 'same-physical'})
                break
            except DeviceOwned:
                time.sleep(.05)
        else:
            pytest.fail('guardian retained the lock after every child exited')
        owner.close()
    finally:
        for pid in (record['worker'], record['child'], record['guardian']):
            if process_identity(pid) is not None:
                os.kill(pid, signal.SIGKILL)
        broker.stdout.close()


@pytest.mark.skipif(not process_namespace_supported(), reason='executor PID API and /proc namespaces disagree')
def test_guarded_launch_registers_before_exec_without_parent_binding(monkeypatch, tmp_path):
    from astrolabe.runtime.telemetry import process_identity
    monkeypatch.setenv('ASTROLABE_GPU_OWNER_DIRECTORY', str(tmp_path))
    grandchild = "import os,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print(os.getpid(),flush=True); time.sleep(15)"
    worker = "import subprocess,sys; p=subprocess.Popen([sys.executable,'-c',sys.argv[1]],close_fds=True); p.wait()"
    parent = """
import json,os,signal,subprocess,sys
from pathlib import Path
from astrolabe.runtime.resources import PhysicalGPUOwner
import astrolabe.runtime.resources as resources
owner=PhysicalGPUOwner({'uuid':'GPU-bootstrap-fixture','token':'fixture'},guarded=True)
command=[sys.executable,'-c',sys.argv[1],sys.argv[2]]
p=subprocess.Popen([sys.executable,'-S',str(Path(resources.__file__).with_name('guarded_launch.py')),
                    str(owner.control_fd),json.dumps(command)],stdout=subprocess.PIPE,text=True,
                    start_new_session=True,pass_fds=(owner.fd,owner.control_fd))
# Deliberately NO parent register_process/bind_process call. The bootstrap
# must register itself and receive the guardian acknowledgement before exec.
child=int(p.stdout.readline())
print(json.dumps({'worker':p.pid,'child':child,'guardian':owner.guardian.pid}),flush=True)
signal.pause()
"""
    broker = subprocess.Popen([sys.executable, '-c', parent, worker, grandchild], stdout=subprocess.PIPE, text=True)
    record = json.loads(broker.stdout.readline())
    try:
        broker.kill()
        broker.wait(timeout=2)
        with pytest.raises(DeviceOwned):
            PhysicalGPUOwner({'uuid': 'GPU-bootstrap-fixture', 'token': 'same-physical'})
        deadline = time.monotonic() + 6
        while process_identity(record['child']) is not None and time.monotonic() < deadline:
            time.sleep(.05)
        assert process_identity(record['child']) is None
        while time.monotonic() < deadline:
            try:
                owner = PhysicalGPUOwner({'uuid': 'GPU-bootstrap-fixture', 'token': 'same-physical'})
                break
            except DeviceOwned:
                time.sleep(.05)
        else:
            pytest.fail('guarded launch left a live owner after cleanup')
        owner.close()
    finally:
        for pid in (record['worker'], record['child'], record['guardian']):
            if process_identity(pid) is not None:
                os.kill(pid, signal.SIGKILL)
        broker.stdout.close()


@pytest.mark.skipif(not process_namespace_supported(), reason='executor PID API and /proc namespaces disagree')
def test_residency_is_not_released_when_worker_leader_exits_but_child_survives():
    ledger = ResourceLedger(limits())
    program = """
import os, subprocess, sys
p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(15)'])
print(p.pid, flush=True)
sys.stdin.read(1)
"""
    process = subprocess.Popen([sys.executable, '-c', program], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, text=True, start_new_session=True)
    child = int(process.stdout.readline())
    ledger.reserve_resident('model', 'cpu', 0)
    ledger.bind_process('model', process.pid, managed_group=True)
    try:
        process.stdin.close()
        process.wait(timeout=3)
        assert ledger.release_resident('model') is False
        assert child in ledger.snapshot()['resident']['model']['processes']
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 2
        while not ledger.release_resident('model') and time.monotonic() < deadline:
            time.sleep(.01)
        process.stdout.close()
        ledger.close()


@pytest.mark.skipif(not process_namespace_supported(), reason='executor PID API and /proc namespaces disagree')
def test_staged_real_process_retains_prepared_memory_and_rejects_reordered_execute(monkeypatch, tmp_path):
    from astrolabe.runtime.broker_state import State
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv('ASTROLABE_EXECUTION_ARTIFACT_ROOT', str(tmp_path))
    identity = {'device': 'cpu', 'code': 'fixture', 'weights': 'none'}
    state = State('test', 2, 2, limits=limits())
    command = [sys.executable, str(root / 'tests/runtime_staged_fixture.py')]
    register = {'identity': identity, 'command': command, 'env': {'RUNTIME_TEST_IDENTITY': json.dumps(identity)},
                'limit': 1048576, 'startup_timeout': 5, 'owner_pid': os.getpid(), 'client_epoch': 'client'}
    try:
        handle = state.register(register)['handle']
        def invoke(method, rid, phase):
            return state.call({'handle': handle, 'epoch': 'client', 'request_id': rid,
                               'payload': {'method': method, 'key': 'scientific-one'}, 'timeout': 2,
                               'runtime': {'phase': phase, 'shape_key': '128',
                                           'scientific_request_id': 'scientific-one',
                                           'resources': {'prepared_bytes': 256, 'ram_bytes': 512, 'output_bytes': 256}}})
        assert invoke('prepare', 'prepare-dispatch', 'cpu_prepare')['result'] == {'prepared': 128}
        assert state.ledger.snapshot()['prepared_bytes'] == 256
        assert state.ledger.snapshot()['cpu_slots'] == 0
        with pytest.raises(ValueError, match='transition'):
            invoke('finalize', 'too-soon', 'cpu_finalize')
        assert state.ledger.snapshot()['prepared_bytes'] == 256
        invoke('execute', 'execute-dispatch', 'composite')
        assert state.ledger.snapshot()['prepared_bytes'] == 256
        assert invoke('finalize', 'finalize-dispatch', 'cpu_finalize')['result'] == {'value': sum(range(128))}
        assert state.ledger.snapshot()['prepared_bytes'] == 0
        state.detach({'handle': handle, 'client_epoch': 'client', 'release': True})
        assert not state.ledger.snapshot()['resident']
    finally:
        state.close()
