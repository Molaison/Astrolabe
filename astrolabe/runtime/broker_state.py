"""Session execution state. Transport and scientific state have other owners."""
from collections import deque
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import signal
import threading
import time
import uuid

from .admission import _execute_probe, allocated_devices
from .client import (AdmissionRejected, ApplicationError, CancellationToken,
                     DaemonClient, OutcomeUnknown)
from .contracts import identity_key, performance_identity, public_identity, OperationResources, request_metadata
from .resources import ResourceLedger, ResourceExhausted
from .telemetry import (EventLog, ProcessObservationError, memory_allocation,
                        process_identity, process_tree)

MAX_CONTROL = 4 * 1024 * 1024


def directory_bytes(root):
    total = 0
    path = Path(root)
    try:
        if path.is_file():
            return path.stat().st_size
        for file in path.rglob('*'):
            if file.is_file() and not file.is_symlink():
                total += file.stat().st_size
    except OSError:
        pass
    return total


class State:
    def __init__(self, token, max_resident_models, max_resident_per_device, *, limits=None):
        self.token = token
        self.max_resident_models = max_resident_models
        self.max_resident_per_device = max_resident_per_device
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.requests = {}
        self.request_limit = 1024
        self.start_locks = {}
        self.starting = {}
        self.workers = {}
        self.used = set()
        self.poisoned = set()
        self.stopping = False
        self.metrics = {'waiting': 0, 'completed': 0, 'failures': 0, 'oom': 0,
                        'wait_seconds': 0.0, 'service_seconds': 0.0}
        self.events = EventLog(os.environ.get('ASTROLABE_RUNTIME_TRACE'))
        self.ledger = ResourceLedger(limits, events=self.events)
        self.recent = deque(maxlen=256)
        self.observation = {'devices': {}, 'available': False}
        self.closed = threading.Event()
        self.external = {}
        self.owners = {}
        self.output_roots = {}
        self.retain_artifacts = False
        self.cleanup_unknown = False
        self.artifact_root = os.environ.get('ASTROLABE_EXECUTION_ARTIFACT_ROOT')
        self.events.write('broker_started', pid=os.getpid(), limits=self.ledger.limits.as_dict())

    def select_device(self, requested):
        devices = allocated_devices(os.environ.get('CUDA_VISIBLE_DEVICES') or '')
        if requested is None:
            if len(devices) > 1:
                raise ValueError('explicit allocated device required for multiple GPUs')
            return devices[0] if devices else ''
        if not isinstance(requested, str) or requested not in devices:
            raise ValueError('device is not in session allocation')
        return requested

    def identity_device(self, identity):
        device = identity.get('device', 'cpu')
        if device == 'cpu':
            return 'cpu', ''
        if not isinstance(device, dict):
            raise ValueError('device must declare an allocated device')
        platform = device.get('platform', device.get('name', ''))
        visible = device.get('visible')
        if platform == 'cpu':
            if visible != '':
                raise ValueError('CPU workers must declare empty CUDA visibility')
            return 'cpu', ''
        if platform not in ('gpu', 'cuda', 'cuda:0', 'mock') or device.get('id', 0) != 0:
            raise ValueError('unsupported device selection')
        self.select_device(visible)
        if not visible:
            raise ValueError('GPU workers must declare an allocated device')
        return self.ledger.device(visible, mock=platform == 'mock'), visible

    def _launch_environment(self, req, device, visible):
        launch_env = dict(req['env'])
        if device == 'cpu':
            launch_env.update(CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='void',
                              HIP_VISIBLE_DEVICES='', ROCR_VISIBLE_DEVICES='',
                              JAX_PLATFORMS='cpu', GH_CUDA_PREINITIALIZE='0')
        else:
            for name in ('CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES',
                         'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'):
                allowed = (os.environ.get(name), visible) if name == 'CUDA_VISIBLE_DEVICES' else (os.environ.get(name),)
                if name in launch_env and launch_env[name] not in allowed:
                    raise ValueError('worker environment changes allocated device visibility')
            launch_env['CUDA_VISIBLE_DEVICES'] = visible
        # Framework CPU threading is bounded even while the main thread waits
        # for an accelerator. Nested BLAS/OpenMP pools cannot claim the host.
        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                     'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
            launch_env[name] = '1'
        launch_env['ASTROLABE_INNER_BROKER_ENDPOINT'] = os.environ.get('ASTROLABE_BROKER_ENDPOINT', '')
        launch_env['ASTROLABE_INNER_BROKER_TOKEN'] = self.token
        launch_env['ASTROLABE_ALLOCATED_CUDA_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES') or ''
        return launch_env

    @contextmanager
    def _state_lock(self, deadline=None, cancel_token=None):
        # State critical sections only pin/update ownership; never stop a
        # process or poll physical residency while holding this global lock.
        if cancel_token and cancel_token.is_set():
            raise AdmissionRejected('request cancelled before submission')
        wait = 2.0 if deadline is None else self._remaining(deadline)
        if not self.lock.acquire(timeout=wait):
            raise TimeoutError('broker state lock admission timed out')
        try:
            if cancel_token and cancel_token.is_set():
                raise AdmissionRejected('request cancelled before submission')
            if deadline is not None:
                self._remaining(deadline)
            yield
        finally:
            self.lock.release()

    @contextmanager
    def _lease_gate(self, owner, deadline=None):
        with self._state_lock(deadline, owner['cancel']):
            while not owner['lease_lock'].acquire(blocking=False):
                self._check_request(owner)
                self.condition.wait(self._remaining(deadline))
        try:
            self._check_request(owner)
            yield
        finally:
            with self.lock:
                owner['lease_lock'].release()
                self.condition.notify_all()

    def _reap_references(self, deadline=None):
        # Return cleanup candidates; callers must close them outside State.lock.
        abandoned = []
        with self._state_lock(deadline):
            for key, record in self.workers.items():
                record['clients'] = {epoch: identity for epoch, identity in record['clients'].items()
                                     if process_identity(identity[0]) == identity}
                if not record['clients'] and not record['busy'] and record.get('retained'):
                    abandoned.append(key)
            self.owners = {pid: ident for pid, ident in self.owners.items() if process_identity(pid) == ident}
        return abandoned

    def _close_record(self, key, deadline=None):
        with self._state_lock(deadline):
            record = self.workers.get(key)
            if record is None:
                return True
            if record['busy'] or record['clients'] or record.get('closing'):
                return False
            record['closing'] = True  # pin this exact record until cleanup ends
        try:
            if key in self.ledger.residents:
                self.ledger.observe_resident(key)
            record['client'].close()
            if record['client'].lifecycle.get('release_confirmed') is not True:
                raise OutcomeUnknown('worker exit unconfirmed; residency was not released')
            cleanup_deadline = time.monotonic() + 2
            released = self.ledger.release_resident(key)
            while not released and time.monotonic() < cleanup_deadline:
                time.sleep(.01)
                released = self.ledger.release_resident(key)
            if not released:
                raise OutcomeUnknown('worker descendants remain unconfirmed; residency was not released')
            for retained in record.get('retained', {}).values():
                self.ledger.release(retained['token'])
            for token in record.get('uncertain_tokens', ()):
                self.ledger.release(token)
            with self.lock:
                if self.workers.get(key) is record:
                    del self.workers[key]
            self.events.write('model_unloaded', backend=key, device=record['device'],
                              pid=record['client'].process.pid, reason='no_live_clients')
            return True
        except BaseException:
            self.retain_artifacts = True
            self._fence(record)
            raise
        finally:
            with self.lock:
                record['closing'] = False
                self.condition.notify_all()

    def _make_registration_room(self, key, device, deadline):
        for abandoned in self._reap_references(deadline):
            self._close_record(abandoned, deadline)
        while True:
            with self._state_lock(deadline):
                same = sum(r['device'] == device for r in self.workers.values())
                same += sum(d == device for d in self.starting.values())
                full = (len(self.workers) + len(self.starting) >= self.max_resident_models or
                        same >= self.max_resident_per_device)
                if not full:
                    self.starting[key] = device
                    return
                candidates = [(other, record) for other, record in self.workers.items()
                              if not record['clients'] and not record['busy'] and not record.get('closing') and
                              (len(self.workers) + len(self.starting) >= self.max_resident_models or
                               record['device'] == device)]
                if not candidates:
                    reason = 'per-device resident model' if same >= self.max_resident_per_device else 'resident model'
                    raise ResourceExhausted(reason + ' limit reached')
                victim = min(candidates, key=lambda pair: pair[1]['last_used'])[0]
            self._close_record(victim, deadline)

    def register(self, req):
        identity = req['identity']
        key = identity_key(identity)
        if type(req.get('limit')) is not int or not 256 <= req['limit'] <= MAX_CONTROL - 65536:
            raise ValueError('invalid worker message limit')
        if not isinstance(req.get('env'), dict) or any(name in req['env'] for name in (
                'ASTROLABE_EXECUTION_ENDPOINT', 'ASTROLABE_EXECUTION_TOKEN')):
            raise ValueError('invalid worker environment')
        if ('ASTROLABE_EXECUTION_ARTIFACT_ROOT' in req['env'] and
                req['env']['ASTROLABE_EXECUTION_ARTIFACT_ROOT'] != self.artifact_root):
            raise ValueError('worker environment changes artifact root')
        owner = process_identity(req.get('owner_pid', os.getppid()))
        epoch = req.get('client_epoch')
        if owner is None or not isinstance(epoch, str) or not epoch:
            raise ValueError('invalid worker client owner')
        spec = (req['command'], req['env'], req['limit'], OperationResources.parse(req.get('resources')).as_dict())
        timeout = req.get('startup_timeout')
        if not isinstance(timeout, (int, float)) or not 0 < timeout <= 3600:
            raise ValueError('invalid worker startup timeout')
        deadline = self._deadline(req, timeout)
        device, visible = self.identity_device(identity)
        launch_env = self._launch_environment(req, device, visible)
        with self._state_lock(deadline):
            entry = self.start_locks.get(key)
            if entry is None:
                if len(self.start_locks) >= self.request_limit:
                    raise AdmissionRejected('startup admission is full')
                entry = {'lock': threading.Lock(), 'refs': 0}
                self.start_locks[key] = entry
            entry['refs'] += 1
        acquired = False
        reserved = False
        try:
            acquired = entry['lock'].acquire(timeout=self._remaining(deadline))
            if not acquired:
                raise TimeoutError('worker startup admission timed out')
            with self._state_lock(deadline):
                if self.stopping or device in self.poisoned:
                    raise OutcomeUnknown('device fenced or session stopping')
                record = self.workers.get(key)
                if record is not None:
                    if record.get('fenced') or record.get('closing'):
                        raise OutcomeUnknown('worker fenced after uncertain outcome')
                    if record['spec'] != spec:
                        raise ValueError('identity reused with different launch command or environment')
                    record['clients'][epoch] = owner
                    return {'handle': key, 'hello': record['client'].hello}
            self._make_registration_room(key, device, deadline)
            resources = OperationResources.parse(req.get('resources'))
            self.ledger.reserve_resident(key, device, resources.resident_bytes)
            reserved = True
            token = 'load:' + key
            startup_resources = {**resources.as_dict(), 'output_bytes': 0, 'prepared_bytes': 0}
            with self.ledger.lease(token, device, startup_resources,
                                   deadline=deadline):
                client = DaemonClient(self.ledger.guarded_command(device, req['command']), identity, env=launch_env,
                                      max_message_bytes=req['limit'],
                                      startup_timeout=self._remaining(deadline),
                                      resources=resources.as_dict(), _direct=True,
                                      _pass_fds=self.ledger.inherited_fds(device),
                                      _on_start=lambda proc: self.ledger.bind_process(key, proc.pid, managed_group=True))
                try:
                    self.ledger.update_resident(key, lifecycle=client.lifecycle)
                    self.ledger.observe_resident(key)
                    self._register_output_roots(key, client.lifecycle)
                except BaseException:
                    client.close()
                    raise
                record = {'client': client, 'spec': spec, 'device': device, 'identity': identity,
                          'cohort': performance_identity(identity),
                          'clients': {epoch: owner}, 'busy': 0, 'call_lock': threading.Lock(),
                          'last_used': time.monotonic(), 'shapes': set(), 'delegated': None,
                          'retained': {}}
                try:
                    with self._state_lock(deadline):
                        if self.stopping:
                            raise OutcomeUnknown('broker session stopped during model startup')
                        self.workers[key] = record
                except BaseException:
                    client.close()
                    raise
                self.events.write('model_loaded', backend=key, device=device, pid=client.process.pid,
                                  identity=public_identity(identity), performance_cohort=record['cohort'],
                                  worker_epoch=client.hello.get('worker_epoch'),
                                  lifecycle=client.lifecycle, measured=self.ledger.snapshot()['resident'][key])
            return {'handle': key, 'hello': client.hello}
        except BaseException as exc:
            if reserved and key not in self.workers:
                if not self.ledger.release_resident(key) and device != 'cpu':
                    self.poisoned.add(device)
                    self.ledger.fence(device)
            with self.lock:
                self.metrics['failures'] += 1
                if 'out of memory' in str(exc).lower() or 'outofmemory' in str(exc).lower():
                    self.metrics['oom'] += 1
            self.events.write('failure', backend=key, device=device, phase='load', cold_start=True,
                              outcome='outcome_unknown' if isinstance(exc, OutcomeUnknown) else 'failed',
                              error=str(exc))
            raise
        finally:
            with self.lock:
                if acquired:
                    self.starting.pop(key, None)
                    entry['lock'].release()
                entry['refs'] -= 1
                if not entry['refs'] and self.start_locks.get(key) is entry:
                    self.start_locks.pop(key)
                self.condition.notify_all()

    def detach(self, req):
        key = req['handle']
        deadline = self._deadline(req, None)
        close = False
        with self._state_lock(deadline):
            record = self.workers.get(key)
            if record is not None:
                record['clients'].pop(req.get('client_epoch'), None)
                record['last_used'] = time.monotonic()
                close = req.get('release') and not record['busy'] and not record['clients']
        if close:
            self._close_record(key, deadline)
        with self._state_lock():
            return {'released': key not in self.workers}

    def _track_request(self, req, deadline=None):
        ident = (req.get('epoch'), req.get('request_id'))
        if not all(isinstance(item, str) and item for item in ident):
            raise ValueError('invalid request identity')
        with self._state_lock(deadline):
            if self.stopping:
                raise OutcomeUnknown('broker session is stopping')
            if ident in self.used:
                raise ValueError('duplicate request')
            if len(self.used) >= 100000:
                raise ValueError('session request limit reached')
            self.used.add(ident)
        return ident

    def _request_resources(self, req, metadata):
        resources = metadata['resources'].copy()
        # Account the actual prepared artifact in addition to the small JSONL
        # control message. Shape alone is never a feature byte estimate.
        prepared = len(json.dumps(req['payload'], separators=(',', ':'), allow_nan=False).encode())
        if self.artifact_root:
            root = Path(self.artifact_root).resolve()
            for name in ('request_dir', 'directory', 'path', 'request_path'):
                value = req['payload'].get(name)
                if isinstance(value, str):
                    path = Path(value).resolve()
                    if path.is_relative_to(root):
                        prepared += directory_bytes(path)
        resources['prepared_bytes'] = max(resources['prepared_bytes'], prepared)
        if resources['output_bytes'] is None:
            occupied = directory_bytes(self.artifact_root) if self.artifact_root else 0
            remaining = self.ledger.limits.output_bytes - occupied
            if remaining <= 0:
                raise ResourceExhausted('session artifact output byte budget exhausted')
            cap = self.ledger.limits.request_output_bytes or max(1, min(1024 ** 3, self.ledger.limits.output_bytes // 2))
            resources['output_bytes'] = min(cap, remaining)
        return resources

    @staticmethod
    def _deadline(req, timeout, start=None):
        own = None if timeout is None else (time.monotonic() if start is None else start) + timeout
        upstream = req.get('deadline')
        if upstream is not None:
            if type(upstream) not in (int, float) or not math.isfinite(upstream):
                raise ValueError('invalid request deadline')
            return upstream if own is None else min(upstream, own)
        return own

    @staticmethod
    def _remaining(deadline):
        if deadline is None:
            return None
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('request deadline expired before submission')
        return remaining

    def _check_request(self, request):
        if request['cancel'].is_set():
            raise AdmissionRejected('request cancelled before submission')
        if self.stopping:
            raise AdmissionRejected('broker session is stopping')
        self._remaining(request['deadline'])

    def _fence(self, record):
        with self.lock:
            record['fenced'] = True
            if record['device'] != 'cpu':
                self.poisoned.add(record['device'])
                self.ledger.fence(record['device'])

    def cancel(self, req):
        key, epoch, rid = req.get('handle'), req.get('epoch'), req.get('target_request_id')
        if not all(isinstance(item, str) and item for item in (key, epoch, rid)):
            raise ValueError('invalid cancellation identity')
        with self._state_lock(self._deadline(req, None)):
            record = self.workers.get(key)
            if record is None or epoch not in record['clients']:
                raise ValueError('worker handle does not belong to this client')
            request = self.requests.get((epoch, rid))
            matched = request is not None and request['handle'] == key
            state = request['state'] if matched else 'not_active'
            if matched:
                request['cancel'].cancel()
                self.condition.notify_all()
        # Do not hold State's lock while acquiring the ledger condition.
        with self.ledger.condition:
            self.ledger.condition.notify_all()
        return {'cancelled': matched, 'state': state, 'release_confirmed': False}

    def call(self, req):
        key = req['handle']
        metadata = request_metadata(req.get('runtime'))
        timeout = req.get('timeout')
        if timeout is not None and (type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError('invalid timeout')
        start = time.monotonic()
        deadline = self._deadline(req, timeout, start)
        ident = self._track_request(req, deadline)
        with self._state_lock(deadline):
            record = self.workers[key]
            if req['epoch'] not in record['clients']:
                raise ValueError('worker handle does not belong to this client')
            if record.get('closing') or record.get('fenced') or record['device'] in self.poisoned:
                raise OutcomeUnknown('device fenced after uncertain worker outcome')
            capabilities = record['client'].hello.get('capabilities') or {}
            phase = metadata['phase']
            method = req['payload'].get('method')
            cpu_phase = phase in ('cpu_prepare', 'cpu_finalize')
            if cpu_phase and (phase not in capabilities.get('cpu_phases', []) or
                              capabilities.get('phase_methods', {}).get(method) != phase):
                raise ValueError('worker did not certify this method as a CPU phase')
            managed = phase == 'managed'
            if managed and (capabilities.get('managed_device') is not True or
                            method not in capabilities.get('managed_methods', [])):
                raise ValueError('worker did not declare managed device subgraphs for this method')
            staged = capabilities.get('request_staging') is True and method in ('prepare', 'execute', 'finalize')
            scientific = metadata.get('scientific_request_id')
            if staged and not scientific:
                raise ValueError('staged request requires a stable scientific_request_id')
            if len(self.requests) >= self.request_limit:
                self.used.discard(ident)
                raise AdmissionRejected('request record admission is full')
            request = {'handle': key, 'cancel': CancellationToken(), 'state': 'queued',
                       'deadline': deadline, 'lease_lock': threading.Lock()}
            self.requests[ident] = request
            record['busy'] += 1
            self.metrics['waiting'] += 1
        token = 'call:' + identity_key({'epoch': ident[0], 'request_id': ident[1]})
        shape = metadata.get('shape_key', 'unknown')
        cold = metadata.get('cold', (shape, phase) not in record['shapes'])
        identity_fields = {name: metadata[name] for name in ('scientific_request_id', 'request_identity') if name in metadata}
        identity_fields.update(worker_epoch=record['client'].hello.get('worker_epoch'),
                               performance_cohort=record['cohort'])
        self.events.write('request_queued', backend=key, request_id=ident[1], epoch=ident[0],
                          shape_key=shape, cold_start=cold, phase=phase, **identity_fields)
        acquired = False
        delegated = None
        submitted = False
        retry_not_executed = False
        release_safe = True
        admitted = None
        waiting_counted = True
        pending = False
        existing = None
        retained_key = (ident[0], scientific) if staged else None
        outcome = 'not_submitted'
        try:
            with self._state_lock(deadline, request['cancel']):
                while True:
                    self._check_request(request)
                    acquired = record['call_lock'].acquire(blocking=False)
                    if acquired:
                        break
                    self.condition.wait(self._remaining(deadline))
                if record.get('closing') or record.get('fenced') or record['device'] in self.poisoned:
                    raise OutcomeUnknown('worker or device fenced after uncertain worker outcome')
            existing = record['retained'].get(retained_key) if staged else None
            if staged:
                if method == 'prepare' and existing is None and record['retained']:
                    raise ResourceExhausted('worker already holds another staged scientific request')
                expected = {'prepare': None, 'execute': 'prepared', 'finalize': 'executed'}[method]
                actual = existing['state'] if existing else None
                if actual != expected:
                    raise ValueError(f'invalid staged request transition: {actual} -> {method}')
            resources = self._request_resources(req, metadata) if existing is None else existing['resources']
            if existing:
                token = existing['token']
            else:
                self.ledger.admit_pending(token, resources, deadline=deadline, cancel_token=request['cancel'])
            pending = True
            self.ledger.acquire_compute(token, record['device'], deadline=deadline,
                                        cpu=cpu_phase or managed, cpu_work=phase != 'device',
                                        cancel_token=request['cancel'])
            admitted = time.monotonic()
            with self._state_lock(deadline, request['cancel']):
                self.metrics['waiting'] -= 1
                waiting_counted = False
                if record['device'] in self.poisoned or self.stopping:
                    raise OutcomeUnknown('device fenced or session stopping')
            dispatch_id = identity_key({'client_epoch': ident[0], 'request_id': ident[1]})
            metadata['output_limit_bytes'] = resources['output_bytes']
            delegated = {'token': token, 'dispatch_id': dispatch_id,
                                   'managed': managed, 'nested': False, 'request': ident,
                                   'metadata': metadata, 'deadline': deadline, 'owner': request}
            with self._state_lock(deadline, request['cancel']):
                self._check_request(request)
                record['delegated'] = delegated
                request['state'] = 'active'
            event = 'device_start' if phase == 'device' else 'compute_start'
            self.events.write(event, backend=key, request_id=ident[1], shape_key=shape,
                              cold_start=cold, phase=phase, device=record['device'], **identity_fields)
            self._check_request(request)
            remaining = self._remaining(deadline)
            submitted = True
            result = record['client'].call(req['payload'], timeout=remaining, request_id=dispatch_id,
                                           runtime=metadata, cancel_token=request['cancel'],
                                           _on_unknown=lambda: self._fence(record), _deadline=deadline)
            outcome = 'completed'
            # No pending CUDA result may survive a managed subgraph lease.
            if delegated['nested'] or delegated.get('uncertain'):
                raise OutcomeUnknown('worker returned with an open device subgraph')
            self.ledger.update_resident(key, lifecycle=record['client'].lifecycle)
            self.ledger.observe_resident(key)
            self._register_output_roots(key, record['client'].lifecycle)
            with self._state_lock(deadline):
                self.metrics['completed'] += 1
                self.metrics['wait_seconds'] += admitted - start
                self.metrics['service_seconds'] += time.monotonic() - admitted
                record['shapes'].add((shape, phase))
            return {'result': result, 'lifecycle': record['client'].lifecycle,
                    'runtime': {**record['client'].last_metrics,
                                'queue_seconds': admitted - start,
                                'admitted_seconds': time.monotonic() - admitted}}
        except BaseException as exc:
            original_error = exc
            if isinstance(exc, AdmissionRejected):
                # Only an explicit no-dispatch rejection gives the caller its
                # ID back. Keep it reserved until this record is fully retired.
                retry_not_executed = True
                submitted = False
            if (isinstance(exc, ApplicationError) and delegated and delegated['nested'] or
                    isinstance(exc, ProcessObservationError) or submitted and isinstance(exc, TimeoutError)):
                exc = OutcomeUnknown('worker resource state could not be observed after dispatch')
            outcome = 'outcome_unknown' if isinstance(exc, OutcomeUnknown) else 'failed'
            with self.lock:
                if isinstance(exc, OutcomeUnknown):
                    self.retain_artifacts = True
                self.metrics['failures'] += 1
                if 'out of memory' in str(exc).lower() or 'outofmemory' in str(exc).lower():
                    self.metrics['oom'] += 1
                if submitted and (isinstance(exc, OutcomeUnknown) or
                                  isinstance(exc, ApplicationError) and record['device'] != 'cpu'):
                    self._fence(record)
            if submitted and (isinstance(exc, OutcomeUnknown) or staged or
                              isinstance(exc, ApplicationError) and record['device'] != 'cpu'):
                # Only this dispatch's owner stops the worker. A cancel ACK is
                # merely a notification, never evidence of resource release.
                request['cancel'].cancel()
                with self.ledger.condition:
                    self.ledger.condition.notify_all()
                release_safe = False
                try:
                    record['client'].close()
                    with request['lease_lock']:
                        release_safe = (record['client'].lifecycle.get('release_confirmed') is True and
                                        self.ledger.release_resident(key))
                except Exception as cleanup:
                    self.events.write('cleanup_unknown', backend=key, request_id=ident[1], error=str(cleanup))
                if not release_safe:
                    self.retain_artifacts = True
                    self._fence(record)
                elif staged:
                    record['fenced'] = True
            self.events.write('failure', backend=key, request_id=ident[1], shape_key=shape,
                              cold_start=cold, phase=phase, outcome=outcome, error=str(exc), **identity_fields)
            if exc is not original_error:
                raise exc from original_error
            raise
        finally:
            ended = time.monotonic()
            if admitted is not None:
                event = 'device_end' if phase == 'device' else 'compute_end'
                self.events.write(event, backend=key, request_id=ident[1], phase=phase,
                                  shape_key=shape, outcome=outcome, duration=ended - admitted, **identity_fields)
            # Nested admission is cancellable and holds this gate while it
            # changes ledger ownership. Drain it before releasing parent tokens.
            with request['lease_lock']:
                with self.lock:
                    if delegated is not None and record['delegated'] is delegated:
                        record['delegated'] = None
            if pending and not release_safe:
                record.setdefault('uncertain_tokens', set()).add(token)
            if pending and release_safe:
                if existing and not submitted:
                    # An unsubmitted execute/finalize still owns the prepared
                    # scientific request from the preceding successful stage.
                    self.ledger.release_compute(token)
                elif staged and outcome == 'completed' and method != 'finalize':
                    self.ledger.release_compute(token)
                    record['retained'][retained_key] = {'token': token, 'resources': resources,
                                                       'state': 'prepared' if method == 'prepare' else 'executed'}
                else:
                    self.ledger.release(token)
                    if staged:
                        record['retained'].pop(retained_key, None)
            with self.lock:
                if waiting_counted:
                    self.metrics['waiting'] -= 1
                record['busy'] -= 1
                record['last_used'] = ended
                if delegated is not None and record['delegated'] is delegated:
                    record['delegated'] = None
                request['state'] = 'finished'
                self.requests.pop(ident, None)
                if retry_not_executed:
                    self.used.discard(ident)
                request['cancel'].close()
                self.recent.append({'time': ended, 'backend': record['cohort'], 'worker_handle': key, 'shape_key': shape,
                                    'cold_start': cold, 'phase': phase, 'outcome': outcome,
                                    'duration': ended - start, 'queue_seconds': None if admitted is None else admitted - start})
            with self.condition:
                if acquired:
                    record['call_lock'].release()
                self.condition.notify_all()
            self.events.write('request_complete', backend=key, request_id=ident[1], shape_key=shape,
                              cold_start=cold, phase=phase, duration=ended - start, outcome=outcome,
                              lifecycle=record['client'].lifecycle, **identity_fields)

    def nested(self, req, *, finish=False):
        with self._state_lock(self._deadline(req, None)):
            candidates = [(key, record) for key, record in self.workers.items()
                          if record['client'].process.pid == req.get('owner_pid')]
            if len(candidates) != 1:
                raise ValueError('nested lease has no registered worker')
            key, record = candidates[0]
            parent = record['delegated']
            if (not parent or not parent['managed'] or req.get('parent_request_id') != parent['dispatch_id'] or
                    req.get('worker_epoch') != record['client'].hello.get('worker_epoch')):
                raise ValueError('nested lease does not belong to the active managed request')
            if bool(parent['nested']) == (not finish):
                raise ValueError('invalid nested lease transition')
            if finish and req.get('error') is not None:
                # Keep the device lease intact. Error return is not proof that
                # asynchronous device work or traceback-held outputs are gone.
                parent['uncertain'] = True
                self.retain_artifacts = True
                self._fence(record)
                parent['owner']['cancel'].cancel()
                self.condition.notify_all()
                raise OutcomeUnknown('exceptional device subgraph; original dispatch owner must confirm cleanup')
        deadline = self._deadline(req, None)
        if deadline is None or parent['deadline'] is not None and parent['deadline'] < deadline:
            deadline = parent['deadline']
        with self._lease_gate(parent['owner'], deadline):
            with self._state_lock(deadline, parent['owner']['cancel']):
                if record['delegated'] is not parent:
                    raise ValueError('nested lease owner has finished')
                self._check_request(parent['owner'])
                if bool(parent['nested']) == (not finish):
                    raise ValueError('invalid nested lease transition')
                parent['nested'] = not finish
            token = parent['token']
            if not finish:
                subgraph = OperationResources.parse(req.get('resources'))
                with self.ledger.condition:
                    reserved = self.ledger.pending[token]
                    # A subgraph can document a smaller scratch requirement, but
                    # cannot silently widen the scientific request's memory lease.
                    if (subgraph.scratch_bytes is not None and reserved.scratch_bytes is not None and
                            subgraph.scratch_bytes > reserved.scratch_bytes):
                        raise ResourceExhausted('device subgraph exceeds its parent scratch reservation')
                    if subgraph.resident_bytes is not None:
                        resident = self.ledger.residents[key]['reserved_bytes']
                        if resident is not None and subgraph.resident_bytes > resident:
                            raise ResourceExhausted('device subgraph requires an explicit residency expansion')
            if finish:
                for pid in req.get('process_ids') or []:
                    if process_identity(pid) is not None:
                        self.ledger.bind_process(key, pid)
                if req.get('lifecycle'):
                    self.ledger.update_resident(key, lifecycle=req['lifecycle'])
                    self._register_output_roots(key, req['lifecycle'])
                self.ledger.observe_resident(key)
            self.ledger.release_compute(token)
            self.ledger.acquire_compute(token, record['device'], deadline=deadline, cpu=finish, cpu_work=finish,
                                        cancel_token=parent['owner']['cancel'])
            self.events.write('device_end' if finish else 'device_start', backend=key,
                              request_id=parent['request'][1], shape_key=parent['metadata'].get('shape_key', 'unknown'),
                              device=record['device'], nested=True)
            return {'lease': token}

    def nested_process(self, req):
        with self._state_lock(self._deadline(req, None)):
            records = [(key, record) for key, record in self.workers.items()
                       if record['client'].process.pid == req.get('owner_pid')]
            if len(records) != 1:
                raise ValueError('nested child has no registered worker')
            key, record = records[0]
            parent = record['delegated']
            if (not parent or not parent['managed'] or not parent['nested'] or
                    req.get('parent_request_id') != parent['dispatch_id'] or
                    req.get('worker_epoch') != record['client'].hello.get('worker_epoch')):
                raise ValueError('nested child does not belong to an active device subgraph')
        deadline = self._deadline(req, None)
        if deadline is None or parent['deadline'] is not None and parent['deadline'] < deadline:
            deadline = parent['deadline']
        with self._lease_gate(parent['owner'], deadline):
            self.ledger.bind_process(key, req['pid'])
        return {}

    def probe(self, req):
        start = time.monotonic()
        timeout = req.get('timeout')
        if not isinstance(timeout, (int, float)) or not 0 < timeout <= 3600:
            raise ValueError('invalid probe timeout')
        deadline = self._deadline(req, timeout, start)
        ident = self._track_request(req, deadline)
        overrides = req.get('env')
        if not isinstance(overrides, dict):
            raise ValueError('invalid probe environment')
        for name in ('ASTROLABE_EXECUTION_ENDPOINT',
                     'ASTROLABE_EXECUTION_TOKEN', 'ASTROLABE_EXECUTION_ARTIFACT_ROOT'):
            if name in overrides and overrides[name] != os.environ.get(name):
                raise ValueError('probe environment changes allocated resources')
        visible = '' if req.get('device') == 'cpu' else self.select_device(req.get('device'))
        # Mock devices exist only in explicit test fixtures. Actual probe
        # selection must resolve its UUID before any CUDA-capable process runs.
        mock = bool(visible and visible.startswith('mock-'))
        device = self.ledger.device(visible, mock=mock) if visible else 'cpu'
        env = {**os.environ, **self._launch_environment({'env': overrides}, device, visible)}
        with self._state_lock(deadline):
            if self.stopping or device in self.poisoned:
                raise OutcomeUnknown('device fenced or session stopping')
        with self.ledger.lease(':'.join(ident), device, {'output_bytes': 65536}, deadline=deadline):
            try:
                return _execute_probe(self.ledger.guarded_command(device, req['command']), env, self._remaining(deadline),
                                      pass_fds=self.ledger.inherited_fds(device))
            except Exception as exc:
                with self.lock:
                    if device != 'cpu':
                        self.poisoned.add(device)
                        self.ledger.fence(device)
                raise OutcomeUnknown('probe outcome unknown; failed GPU device remains fenced') from exc

    def stage(self, req):
        event = req['event']
        if event.get('phase') not in ('cpu', 'cpu_execute', 'cpu_prepare', 'cpu_finalize',
                                     'load', 'unload', 'device', 'composite'):
            raise ValueError('invalid stage phase')
        duration = event.get('duration')
        if not isinstance(duration, (float, int)) or duration < 0:
            raise ValueError('invalid stage duration')
        if event['phase'] == 'cpu':
            event = {**event, 'phase': 'cpu_execute'}
        self.events.write('stage', **event)
        return {}

    def register_owner(self, req):
        identity = process_identity(req['pid'])
        if identity is None:
            raise ValueError('trajectory owner does not exist')
        with self.lock:
            self.owners[identity[0]] = identity
        return {}

    def _register_output_roots(self, key, lifecycle):
        """Keep configured output paths in the job's accounting after unload.

        Existing shared caches are not charged as newly generated job output.
        Their positive growth is observable but cannot be attributed exactly
        when a separate job writes the same cache. No cache file is deleted.
        """
        roots = lifecycle.get('output_roots') or []
        if not isinstance(roots, list) or len(roots) > 64:
            raise ValueError('lifecycle.output_roots must be a bounded list')
        prepared = []
        for item in roots:
            if (not isinstance(item, dict) or item.get('scope') not in ('job', 'shared_cache') or
                    not isinstance(item.get('path'), str)):
                raise ValueError('invalid lifecycle output root')
            path = Path(item['path'])
            if not path.is_absolute() or path.resolve() == Path('/'):
                raise ValueError('output roots must be absolute, bounded backend directories')
            name = str(path.resolve())
            with self._state_lock():
                known = name in self.output_roots
            baseline = None if known else directory_bytes(path)
            prepared.append((name, item['scope'], baseline))
        with self._state_lock():
            for name, scope, baseline in prepared:
                if name not in self.output_roots:
                    if len(self.output_roots) >= 4096:
                        raise ResourceExhausted('session output-root accounting limit reached')
                    self.output_roots[name] = {'scope': scope, 'backend': key,
                                              'baseline_bytes': baseline,
                                              'registered_at': time.monotonic()}
                elif self.output_roots[name]['scope'] != scope:
                    raise ValueError('output directory changed accounting scope')

    def output_usage(self):
        with self.lock:
            roots = {path: dict(record) for path, record in self.output_roots.items()}
        artifact = Path(self.artifact_root).resolve() if self.artifact_root else None
        job_paths = sorted((Path(path) for path, record in roots.items() if record['scope'] == 'job'),
                           key=lambda path: len(path.parts))
        # Count each inode subtree once, including any job directory that is
        # nested within the broker's artifact root or another reported root.
        counted = [artifact] if artifact else []
        job_bytes = 0
        details = {}
        for path in job_paths:
            if any(path.is_relative_to(parent) for parent in counted):
                continue
            # A configured parent of artifact_root would otherwise double
            # charge the artifact directory. It replaces that nested count.
            size = directory_bytes(path)
            if artifact is not None and artifact.is_relative_to(path):
                size = max(0, size - directory_bytes(artifact))
            job_bytes += size
            counted.append(path)
            details[str(path)] = {'scope': 'job', 'observed_bytes': size, 'attribution': 'configured_job_root'}
        shared_growth = 0
        for name, record in sorted(roots.items(), key=lambda pair: len(Path(pair[0]).parts)):
            if record['scope'] != 'shared_cache' or any(Path(name).is_relative_to(parent) for parent in counted):
                continue
            size = directory_bytes(name)
            growth = max(0, size - record['baseline_bytes'])
            shared_growth += growth
            counted.append(Path(name))
            details[name] = {'scope': 'shared_cache', 'observed_bytes': size,
                             'baseline_bytes': record['baseline_bytes'], 'growth_bytes': growth,
                             'attribution': 'unknown_if_other_jobs_write'}
        return {'external_job_bytes': job_bytes, 'shared_cache_growth_bytes': shared_growth,
                'external_output_roots': details}

    def sample(self):
        snapshot = self.ledger.snapshot()
        devices = {key: self.ledger.telemetry.sample(device) for key, device in snapshot['devices'].items()}
        with self.lock:
            roots = dict(self.owners)
            groups = {pid: identity[1] for pid, identity in self.owners.items()}
            for resident in snapshot['resident'].values():
                roots.update(resident['processes'])
                groups.update(resident['groups'])
        processes = process_tree(roots, groups=groups)
        rss = sum(info['rss'] for info in processes.values())
        artifact_bytes = directory_bytes(self.artifact_root) if self.artifact_root else 0
        outputs = self.output_usage()
        memory = memory_allocation()
        observation = {'time': time.monotonic(), 'devices': devices, 'memory': memory,
                       'rss_bytes': rss, 'artifact_bytes': artifact_bytes, **outputs,
                       'process_tree_unverified_groups': list(processes.unverified_groups),
                       'unassociated_processes_unobserved': processes.unobserved_count,
                       'cpu_ticks': sum(info['cpu'] for info in processes.values()),
                       'clock_ticks': os.sysconf('SC_CLK_TCK'),
                       'available': bool(devices) and all(d['available'] for d in devices.values())}
        with self.lock:
            observation.update(waiting=self.metrics['waiting'], pending=snapshot['pending'],
                               active_compute=snapshot['active'])
        vram_breach = any(sample.get('available') and sample['used_bytes'] >
                         sample['total_bytes'] * self.ledger.limits.vram_fraction for sample in devices.values())
        breach = (bool(processes.unverified_groups) or rss > self.ledger.limits.ram_bytes or vram_breach or
                  artifact_bytes + outputs['external_job_bytes'] + outputs['shared_cache_growth_bytes'] >
                  self.ledger.limits.output_bytes + self.ledger.limits.prepared_bytes)
        with self.lock:
            self.observation = observation
            if processes.unverified_groups:
                self.cleanup_unknown = True
                self.retain_artifacts = True
            if breach and not self.stopping:
                self.events.write('resource_limit_exceeded', observation=observation)
                self.stopping = True
                with self.ledger.condition:
                    self.ledger.stopping = True
                    self.ledger.condition.notify_all()
                for pid, identity in roots.items():
                    if process_identity(pid) != tuple(identity):
                        continue
                    try:
                        # Managed trajectories and workers each own a group.
                        if os.getpgid(pid) == pid:
                            os.killpg(pid, signal.SIGTERM)
                        else:
                            os.kill(pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
        for key in self._reap_references():
            self._close_record(key)
        self.events.write('resource_sample', **observation)
        return observation

    def monitor(self):
        while not self.closed.wait(self.ledger.limits.telemetry_interval):
            try:
                self.sample()
            except Exception as exc:
                self.events.write('telemetry_error', error=f'{type(exc).__name__}: {exc}')
                if isinstance(exc, ProcessObservationError):
                    with self.lock:
                        self.stopping = self.cleanup_unknown = self.retain_artifacts = True
                    with self.ledger.condition:
                        self.ledger.stopping = True
                        self.ledger.condition.notify_all()

    def snapshot(self, *, deadline=None):
        with self._state_lock(deadline):
            snapshot = {**self.metrics, 'observation': dict(self.observation), 'recent': list(self.recent),
                    'stopping': self.stopping,
                    'requests': [{'epoch': ident[0], 'request_id': ident[1], 'handle': request['handle'],
                                  'state': request['state'], 'cancel_requested': request['cancel'].is_set()}
                                 for ident, request in self.requests.items()],
                    'trace_dropped': self.events.dropped}
        snapshot['ledger'] = self.ledger.snapshot(deadline=deadline)
        return snapshot

    def snapshot_for_request(self, req):
        return self.snapshot(deadline=self._deadline(req, None))

    def close(self):
        self.closed.set()
        with self.lock:
            self.stopping = True
            records = list(self.workers.items())
            for request in self.requests.values():
                request['cancel'].cancel()
            self.condition.notify_all()
        with self.ledger.condition:
            self.ledger.stopping = True
            self.ledger.condition.notify_all()
        errors = []
        for key, record in records:
            try:
                record['client'].close()
            except Exception as exc:
                errors.append(f'{key}: {exc}')
        guarded = self.ledger.close()
        # Guardians retain physical ownership while cleaning descendants whose
        # Popen(close_fds=True) did not inherit the original lock descriptor.
        deadline = time.monotonic() + 2
        remaining = list(self.ledger.residents)
        while remaining:
            remaining = [key for key in remaining if not self.ledger.release_resident(key)]
            if not remaining or time.monotonic() >= deadline:
                break
            time.sleep(.01)
        for key, record in records:
            if key not in remaining and record['client'].lifecycle.get('release_confirmed') is True:
                for retained in record.get('retained', {}).values():
                    self.ledger.release(retained['token'])
                for token in record.get('uncertain_tokens', ()):
                    self.ledger.release(token)
        confirmed = guarded and not remaining and not errors and not self.cleanup_unknown
        if (self.retain_artifacts or not confirmed) and self.artifact_root:
            root = Path(self.artifact_root)
            root.mkdir(parents=True, exist_ok=True)
            (root / 'retained.json').write_text(json.dumps(
                {'cleanup_confirmed': confirmed, 'outcome_unknown': self.retain_artifacts,
                 'unreleased': remaining, 'cleanup_errors': errors}, allow_nan=False) + '\n')
        self.events.write('broker_stopped', resources=self.ledger.snapshot(), cleanup_confirmed=confirmed,
                          cleanup_errors=errors, unreleased=remaining)
        if not confirmed:
            raise OutcomeUnknown('broker cleanup not confirmed; live resources/artifacts must be retained')
