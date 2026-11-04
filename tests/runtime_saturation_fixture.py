"""CPU-only real socket/daemon scenarios; no broker implementation imports.

Run in a disposable process. Every wait has a deadline and cleanup owns only
the broker and daemon PIDs created by this fixture.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import socket
import sys
import threading
import time


IDENTITY = {'environment': 'saturation-test', 'code': 'saturation-v1',
            'weights': 'none', 'device': 'cpu'}


def emit(path, kind, **fields):
    with open(path, 'a') as stream:
        stream.write(json.dumps({'kind': kind, 'time': time.monotonic(),
                                 'pid': os.getpid(), **fields}) + '\n')


def until(predicate, seconds=5, label='condition'):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(.005)
    raise TimeoutError('fixture deadline: ' + label)


def daemon_handler(payload):
    from astrolabe.runtime.admission import operation
    from astrolabe.runtime.server import current_request
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    log = payload['log']
    rid = payload['value']
    emit(log, 'start', value=rid, context=current_request())
    if payload.get('ready'):
        Path(payload['ready']).write_text(str(rid))
    if payload.get('gate'):
        until(lambda: Path(payload['gate']).exists(), 15, 'daemon gate')
    if payload.get('application_error'):
        raise ValueError('deterministic saturation fixture application error')
    if payload.get('method') == 'managed':
        assert current_request()['phase'] == 'managed'
        with operation(IDENTITY, timeout=5, resident_bytes=0, scratch_bytes=0):
            emit(log, 'nested_begin', value=rid)
            Path(payload['nested_ready']).touch()
            until(lambda: Path(payload['nested_gate']).exists(), 10, 'nested gate')
        emit(log, 'nested_end', value=rid)
        if payload.get('finish_gate'):
            until(lambda: Path(payload['finish_gate']).exists(), 10, 'managed finish gate')
    time.sleep(payload.get('delay', 0))
    emit(log, 'end', value=rid)
    return {'value': rid, 'pid': os.getpid(), 'visible': os.environ.get('CUDA_VISIBLE_DEVICES')}


class Harness:
    def __init__(self, directory, pending=32):
        from astrolabe.runtime.admission import available_cpus
        from astrolabe.runtime.session import BrokerSession
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.log = self.directory / 'events.jsonl'
        self.clients = []
        self.futures = []
        self.pool = ThreadPoolExecutor(max_workers=40)
        self.session = BrokerSession(max_resident_models=5, resource_limits={
            'cpu_slots': min(3, available_cpus()), 'pending_items': pending,
            # Three independent Python daemons must fit; this test stresses
            # ingress capacity, not the process-tree RAM guardrail.
            'ram_bytes': 256 * 1024**2, 'prepared_bytes': 4 * 1024**2,
            'output_bytes': 4 * 1024**2, 'request_output_bytes': 16384})

    def __enter__(self):
        self.session.__enter__()
        return self

    def client(self, name):
        from astrolabe.runtime.broker_client import BrokerClient
        identity = {**IDENTITY, 'instance': name}
        client = BrokerClient(self.session.endpoint, self.session.token,
                              [sys.executable, str(Path(__file__).resolve()), '--daemon'], identity,
                              env={'RUNTIME_SATURATION_IDENTITY': json.dumps(identity)},
                              max_message_bytes=1048576, startup_timeout=5,
                              resources={'resident_bytes': 0, 'ram_bytes': 1024 * 1024,
                                         'output_bytes': 16384})
        self.clients.append(client)
        return client

    def submit(self, client, rid, **payload):
        started = time.monotonic()
        def run():
            result = client.call({'log': str(self.log), 'value': rid, **payload},
                                 request_id=rid, timeout=16,
                                 runtime={'phase': 'managed' if payload.get('method') == 'managed' else 'composite',
                                          'resources': {'output_bytes': 16384, 'ram_bytes': 1024 * 1024}})
            return {'result': result, 'latency_seconds': time.monotonic() - started,
                    'finished': time.monotonic()}
        future = self.pool.submit(run)
        self.futures.append(future)
        return future

    def metrics(self):
        return self.clients[0]._rpc({'op': 'metrics'}, 2, submitted=False)['metrics']

    def events(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def wire(self, op, **fields):
        from astrolabe.runtime.broker_transport import endpoint_for
        endpoint = endpoint_for(self.session.endpoint, op)
        request = {'op': op, 'token': self.session.token,
                   'allocated_cuda_devices': '', **fields}
        with socket.socket(socket.AF_UNIX) as sock:
            sock.settimeout(2)
            sock.connect('\0' + endpoint[1:] if endpoint.startswith('@') else endpoint)
            sock.sendall((json.dumps(request) + '\n').encode())
            data = bytearray()
            while not data.endswith(b'\n'):
                part = sock.recv(65536)
                if not part:
                    raise EOFError('wire response closed')
                data.extend(part)
                assert len(data) < 2 * 1024**2
            return json.loads(data)

    def __exit__(self, exc_type, _exc, _traceback):
        # Release file gates even after an assertion; never wait on futures
        # before making it possible for their workers to finish.
        for name in ('gate', 'nested_gate', 'finish_gate', 'producer_gate', 'target_gate', 'survivor_gate'):
            (self.directory / name).touch()
        cleanup_deadline = time.monotonic() + 3
        for future in self.futures:
            try:
                future.result(timeout=max(0, cleanup_deadline - time.monotonic()))
            except Exception:
                pass
        self.pool.shutdown(wait=False, cancel_futures=True)
        # Bound BrokerSession.__exit__, which otherwise waits without timeout.
        def alarm(_signum, _frame):
            raise TimeoutError('fixture cleanup deadline')
        old = signal.signal(signal.SIGALRM, alarm)
        signal.alarm(5)
        forced = []
        try:
            self.session.__exit__(None, None, None)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
            # Daemons are process-group leaders, the broker is not. Only kill
            # fixture-owned still-live processes, never the caller's group.
            for client in self.clients:
                pid = client.hello['pid']
                if Path('/proc', str(pid)).exists():
                    forced.append(pid)
                    try:
                        os.killpg(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            if self.session.process.poll() is None:
                forced.append(self.session.process.pid)
                self.session.process.kill()
                self.session.process.wait(timeout=3)
            self.cleanup = {'broker_exit_code': self.session.process.returncode,
                            'forced_cleanup_pids': sorted(set(forced)),
                            'daemon_pids': sorted({client.hello['pid'] for client in self.clients})}
        if exc_type is None:
            assert not forced, 'production cleanup left live fixture processes: ' + str(forced)
            assert self.session.process.returncode == 0, self.cleanup


def clean(metrics):
    ledger = metrics['ledger']
    assert ledger['pending'] == 0, metrics
    assert not ledger['active'], metrics
    assert ledger['cpu_slots'] == 0, metrics
    for key in ('prepared_bytes', 'output_reserved_bytes', 'ram_reserved_bytes'):
        if key in ledger:
            assert ledger[key] == 0, metrics


def smoke(h):
    client = h.client('smoke')
    result = h.submit(client, 'smoke').result(timeout=5)
    assert result['result']['value'] == 'smoke'
    assert result['result']['visible'] == ''
    clean(h.metrics())
    return result


def ready_consumers(h):
    from astrolabe.runtime.admission import available_cpus
    assert available_cpus() >= 2, 'two CPU slots required for independent producer/evaluators'
    producer = h.client('producer')
    evaluators = [h.client('evaluator-a'), h.client('evaluator-b')]
    # Candidates already exist before producer starts. No evaluator waits on
    # producer output or shares its worker/call lock.
    candidates = []
    for index in range(8):
        path = h.directory / ('ready-' + str(index))
        path.write_text(json.dumps({'candidate': index, 'sequence': 'AAAA'}))
        candidates.append(path)
    ready = h.directory / 'producer-ready'
    slow = h.submit(producer, 'producer', ready=str(ready), gate=str(h.directory / 'producer_gate'))
    until(ready.exists, label='producer entered')
    start = time.monotonic()
    def consume(index):
        results = []
        for path in candidates[index::2]:
            candidate = json.loads(path.read_text())['candidate']
            results.append(h.submit(evaluators[index], 'eval-' + str(candidate), delay=.01).result(timeout=5))
        return results
    # Different feeder threads keep both evaluator streams runnable.
    with ThreadPoolExecutor(max_workers=2) as feeders:
        pending = [feeders.submit(consume, index) for index in range(2)]
        results = [future.result(timeout=8) for future in pending]
    elapsed = time.monotonic() - start
    assert not slow.done(), 'consumers were serialized behind a slow producer'
    snapshot = h.metrics()
    assert snapshot['ledger']['pending'] <= 32
    assert len({item['result']['pid'] for items in results for item in items}) == 2
    (h.directory / 'producer_gate').touch()
    slow.result(timeout=5)
    clean(h.metrics())
    return {'seconds': elapsed, 'completed': 8, 'throughput_per_second': 8 / elapsed,
            'producer_pending_at_consumer_completion': True,
            'evaluators': [{'name': name, 'completed': len(items),
                            'latency_seconds': [item['latency_seconds'] for item in items],
                            'max_latency_seconds': max(item['latency_seconds'] for item in items),
                            'throughput_per_second': len(items) / elapsed}
                           for name, items in zip(('a', 'b'), results)], 'metrics': snapshot}


def saturation(h):
    from astrolabe.runtime.client import AdmissionRejected
    clients = [h.client('shared-managed') for _ in range(32)]
    spare = h.client('release-under-saturation')
    ready = h.directory / 'active-ready'
    active = h.submit(clients[0], 'outer-0', method='managed', ready=str(ready),
                      gate=str(h.directory / 'gate'), nested_ready=str(h.directory / 'nested-ready'),
                      nested_gate=str(h.directory / 'nested_gate'), finish_gate=str(h.directory / 'finish_gate'))
    until(ready.exists, label='managed outer active')
    queued = [h.submit(client, 'outer-' + str(index)) for index, client in enumerate(clients[1:], 1)]
    full = until(lambda: (m if (m := h.metrics())['ingress']['business']['in_flight'] == 32 else None),
                 label='32 accepted real socket calls')
    assert full['ingress']['business']['capacity'] == 32
    assert not active.done() and all(not future.done() for future in queued)
    req = {'handle': clients[0].handle, 'epoch': clients[0].epoch,
           'request_id': 'wire-extra', 'payload': {'log': str(h.log), 'value': 'wire-extra'}, 'timeout': 2}
    rejection = h.wire('call', **req)
    assert rejection['ok'] is False and rejection['execution_state'] == 'not_executed' and rejection['retryable'] is True, rejection
    with_error = None
    try:
        clients[0]._rpc({'op': 'call', **req, 'request_id': 'client-extra'}, 2, submitted=True)
    except AdmissionRejected as exc:
        with_error = type(exc).__name__
    assert with_error == 'AdmissionRejected'
    # Both nested acquire AND nested release must progress while all 32
    # business requests still occupy their accepted ingress slots.
    (h.directory / 'gate').touch()
    until((h.directory / 'nested-ready').exists, label='nested acquired with business full')
    during_nested = h.metrics()
    assert during_nested['ingress']['business']['in_flight'] == 32
    (h.directory / 'nested_gate').touch()
    until(lambda: any(event['kind'] == 'nested_end' for event in h.events()), label='nested released with business full')
    assert h.metrics()['ingress']['business']['in_flight'] == 32
    spare_pid = spare.hello['pid']
    spare.close(release=True)
    until(lambda: not Path('/proc', str(spare_pid)).exists(), label='release on control lane')
    cancel = clients[1].cancel('outer-1')
    try:
        queued[0].result(timeout=3)
    except Exception as exc:
        pending_error = type(exc).__name__
    else:
        raise AssertionError('cancelled pending request ran')
    assert not active.done() and Path('/proc', str(clients[0].hello['pid'])).exists()
    (h.directory / 'finish_gate').touch()
    assert active.result(timeout=5)['result']['value'] == 'outer-0'
    for future in queued[1:]:
        future.result(timeout=6)
    events = h.events()
    values = {event.get('value') for event in events if event['kind'] == 'start'}
    assert not values.intersection({'wire-extra', 'client-extra', 'outer-1'})
    assert any(event['kind'] == 'nested_end' for event in events)
    final = until(lambda: (m if (m := h.metrics())['ingress']['business']['in_flight'] == 0 else None))
    clean(final)
    assert final['ingress']['business']['high_water'] == 32
    assert final['ingress']['business']['rejected'] >= 2
    return {'full': full, 'nested': during_nested, 'final': final,
            'wire_rejection': rejection, 'pending_cancel': cancel, 'pending_error': pending_error,
            'executed': len(values)}


def bounded_control(h):
    from astrolabe.runtime.broker_transport import endpoint_for
    from astrolabe.runtime.client import AdmissionRejected
    h.client('control')
    initial = h.metrics()['ingress']['control']
    capacity = initial['capacity']
    assert isinstance(capacity, int) and 0 < capacity <= 128
    endpoint = endpoint_for(h.session.endpoint, 'metrics')
    held = []
    tasks = Path('/proc', str(h.session.process.pid), 'task')
    before_threads = len(list(tasks.iterdir()))
    try:
        # Partial JSON holds a real handler, independent of private classes.
        for _ in range(capacity):
            sock = socket.socket(socket.AF_UNIX)
            sock.settimeout(2)
            sock.connect('\0' + endpoint[1:] if endpoint.startswith('@') else endpoint)
            sock.sendall(b'{')
            held.append(sock)
        rejected = []
        for _ in range(capacity + 4):
            response = h.wire('metrics')
            assert response.get('execution_state') == 'not_executed' and response.get('retryable') is True, response
            rejected.append(response)
        peak_threads = len(list(tasks.iterdir()))
        assert peak_threads <= before_threads + capacity + 2, (before_threads, peak_threads, capacity)
    finally:
        for sock in held:
            sock.close()
    def drained_metrics():
        try:
            return h.metrics()
        except AdmissionRejected:
            return None  # Partial frames are still being closed by handlers.
    final = until(drained_metrics)['ingress']['control']
    assert final['high_water'] == capacity
    assert final['rejected'] >= len(rejected)
    return {'capacity': capacity, 'rejected_observed': len(rejected), 'threads_before': before_threads,
            'threads_at_capacity': peak_threads, 'metrics': final}


def cancel_isolation(h):
    target = h.client('target')
    waiter = h.client('target')
    survivor = h.client('survivor')
    a_ready, b_ready = h.directory / 'a-ready', h.directory / 'b-ready'
    a = h.submit(target, 'active-target', ready=str(a_ready), gate=str(h.directory / 'target_gate'))
    b = h.submit(survivor, 'active-survivor', ready=str(b_ready), gate=str(h.directory / 'survivor_gate'))
    until(lambda: a_ready.exists() and b_ready.exists(), label='two independent active daemons')
    from astrolabe.runtime.client import WorkerError, OutcomeUnknown
    try:
        survivor._rpc({'op': 'cancel', 'handle': target.handle, 'epoch': survivor.epoch,
                       'target_request_id': 'active-target'}, 2, submitted=False)
    except WorkerError as exc:
        assert not isinstance(exc, OutcomeUnknown), 'ownership rejection must be definitive'
    else:
        raise AssertionError('foreign client epoch cancelled target request')
    assert not a.done() and not b.done()
    pending = h.submit(waiter, 'pending-target')
    until(lambda: h.metrics()['waiting'] >= 1, label='pending target queued')
    pending_response = waiter.cancel('pending-target')
    try:
        pending.result(timeout=3)
    except Exception as exc:
        pending_error = type(exc).__name__
    else:
        raise AssertionError('pending cancellation returned success')
    assert not a.done() and not b.done()
    assert Path('/proc', str(target.hello['pid'])).exists()
    active_response = target.cancel('active-target')
    try:
        a.result(timeout=4)
    except Exception as exc:
        active_error = type(exc).__name__
    else:
        raise AssertionError('active cancellation returned success')
    until(lambda: not Path('/proc', str(target.hello['pid'])).exists(), label='target daemon terminated')
    assert not b.done() and Path('/proc', str(survivor.hello['pid'])).exists()
    (h.directory / 'survivor_gate').touch()
    assert b.result(timeout=4)['result']['value'] == 'active-survivor'
    assert h.submit(survivor, 'survivor-next').result(timeout=4)['result']['value'] == 'survivor-next'
    assert not any(event['kind'] == 'start' and event['value'] == 'pending-target' for event in h.events())
    snapshot = h.metrics()
    clean(snapshot)
    return {'pending_response': pending_response, 'active_response': active_response,
            'pending_error': pending_error, 'active_error': active_error, 'metrics': snapshot}


def application_error(h):
    from astrolabe.runtime.client import ApplicationError
    first, other = h.client('application'), h.client('healthy')
    try:
        h.submit(first, 'bad', application_error=True).result(timeout=5)
    except ApplicationError:
        pass
    else:
        raise AssertionError('deterministic error was not ApplicationError')
    result = h.submit(other, 'healthy').result(timeout=5)
    assert result['result']['value'] == 'healthy'
    clean(h.metrics())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--daemon', action='store_true')
    parser.add_argument('--scenario', choices=['smoke', 'ready_consumers', 'saturation', 'bounded_control',
                                              'cancel_isolation', 'application_error'])
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.daemon:
        from astrolabe.runtime.server import serve
        identity = json.loads(os.environ['RUNTIME_SATURATION_IDENTITY'])
        serve(daemon_handler, identity, load_count=1, lifecycle={'weights_loaded': False, 'gpu_resident': False},
              capabilities={'managed_device': True, 'managed_methods': ['managed']})
        return
    def deadline(_signum, _frame):
        raise TimeoutError('scenario total deadline')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(30)
    report = {'scenario': args.scenario, 'success': False}
    started = time.monotonic()
    try:
        with Harness(args.out) as harness:
            report['result'] = globals()[args.scenario](harness)
        report['cleanup'] = harness.cleanup
        report['success'] = True
    except BaseException as exc:
        report['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        signal.alarm(0)
        report['seconds'] = time.monotonic() - started
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / 'result.json').write_text(json.dumps(report, indent=2, default=str) + '\n')


if __name__ == '__main__':
    main()
