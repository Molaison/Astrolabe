"""Admit small device-discovery subprocesses to the session compute slot."""
import json
import os
import select
import signal
import subprocess
import time
import uuid
import pathlib
import re


def allocated_devices(allocation):
    """Validate allocation tokens without discovering devices outside the allocation."""
    devices = allocation.split(',') if allocation else []
    if any(not d or d.strip() != d for d in devices):
        raise ValueError('invalid CUDA allocation')
    if len(set(d.lower() for d in devices)) != len(devices):
        raise ValueError('duplicate CUDA allocation')
    numeric = [d.isdecimal() for d in devices]
    if any(numeric) and not all(numeric):
        raise ValueError('mixed numeric and non-numeric CUDA allocation')
    if any(d.isdecimal() and str(int(d)) != d for d in devices):
        raise ValueError('non-canonical numeric CUDA allocation')
    for device in devices:
        if device.upper().startswith('GPU-') and not re.fullmatch(
                r'GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}', device):
            raise ValueError('GPU UUID must be complete')
    return devices


def resolve_resident_limit(value, allocation=None):
    if value == 'auto':
        if allocation is None:
            allocation = os.environ.get('CUDA_VISIBLE_DEVICES') or ''
        allocated_devices(allocation)
        # This is a registry/process bound, not a memory estimate. One GPU
        # commonly hosts an evaluator and a small mutator. The physical VRAM
        # ledger makes the actual load admission decision.
        return 32
    if type(value) is not int or value < 1:
        raise ValueError('max_resident_models must be a positive integer or auto')
    return value


def _cgroup_v2_dirs():
    """Current cgroup and ancestors, if the unified hierarchy is mounted here."""
    root = pathlib.Path('/sys/fs/cgroup')
    try:
        lines = pathlib.Path('/proc/self/cgroup').read_text().splitlines()
        group = next(line.split(':', 2)[2] for line in lines if line.startswith('0::'))
        path = (root / group.lstrip('/')).resolve()
        if not path.is_relative_to(root):
            return [root]
        return [path, *[p for p in path.parents if p.is_relative_to(root)]]
    except (OSError, StopIteration, ValueError):
        return [root]


def _cgroup_v1_cpu_dirs():
    try:
        lines = pathlib.Path('/proc/self/cgroup').read_text().splitlines()
        group = next(parts[2] for line in lines if len(parts := line.split(':', 2)) == 3
                     and 'cpu' in parts[1].split(','))
        root = next(p for p in (pathlib.Path('/sys/fs/cgroup/cpu'),
                                pathlib.Path('/sys/fs/cgroup/cpu,cpuacct')) if p.is_dir())
        path = (root / group.lstrip('/')).resolve()
        if not path.is_relative_to(root):
            return []
        return [path, *[p for p in path.parents if p.is_relative_to(root)]]
    except (OSError, StopIteration, ValueError):
        return []


def available_cpus():
    """Conservative CPU count from affinity, cgroup-v2 quota, and Slurm task allocation."""
    try:
        limits = [len(os.sched_getaffinity(0))]
    except (AttributeError, OSError):
        limits = [os.cpu_count() or 1]
    for path in _cgroup_v2_dirs():
        try:
            quota, period = (path / 'cpu.max').read_text().split()
            if quota != 'max' and int(quota) > 0 and int(period) > 0:
                limits.append(max(1, int(quota) // int(period)))
        except (OSError, ValueError):
            pass
    for path in _cgroup_v1_cpu_dirs():
        try:
            quota = int((path / 'cpu.cfs_quota_us').read_text())
            period = int((path / 'cpu.cfs_period_us').read_text())
            if quota > 0 and period > 0:
                limits.append(max(1, quota // period))
        except (OSError, ValueError):
            pass
    try:
        cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', '0'))
        if cpus > 0:
            limits.append(cpus)
    except ValueError:
        pass
    return max(1, min(limits))


def resolve_active_limit(value, repeats):
    """Resolve an explicit active bound or auto against this process allocation."""
    if type(repeats) is not int or repeats < 1:
        raise ValueError('repeats must be a positive integer')
    if value == 'auto':
        return min(repeats, available_cpus())
    if type(value) is not int or value < 1:
        raise ValueError('max_active must be a positive integer or auto')
    return min(value, repeats)


def resource_pressure(cpus):
    try:
        if os.getloadavg()[0] >= cpus:
            return True
    except OSError:
        return True
    for path in _cgroup_v2_dirs():
        try:
            maximum = (path / 'memory.max').read_text().strip()
            current = int((path / 'memory.current').read_text())
            if maximum != 'max' and current >= int(maximum) * .9:
                return True
        except (OSError, ValueError):
            pass
    return False


def _execute_probe(command, env, timeout, limit=65536, *, pass_fds=()):
    if timeout <= 0:
        raise TimeoutError('probe deadline expired')
    p = subprocess.Popen(command, stdout=subprocess.PIPE, env=env, start_new_session=True,
                         bufsize=0, pass_fds=tuple(pass_fds))
    deadline = time.monotonic() + timeout
    data = bytearray()
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('device probe timed out')
            ready, _, _ = select.select([p.stdout], [], [], remaining)
            if not ready:
                raise TimeoutError('device probe timed out')
            chunk = os.read(p.stdout.fileno(), min(65536, limit + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
            if len(data) > limit:
                raise ValueError('device probe output exceeds byte limit')
        remaining = max(0, deadline - time.monotonic())
        if p.wait(timeout=remaining) != 0:
            raise RuntimeError('device probe failed')
        result = json.loads(data)
        if not isinstance(result, dict):
            raise ValueError('device probe must return a JSON object')
        return result
    except BaseException:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if p.poll() is None:
            p.wait()
        raise
    finally:
        p.stdout.close()


def run_probe(command, *, env=None, timeout=60, device=None):
    """Return a small JSON device descriptor; never replay uncertain work."""
    endpoint = os.environ.get('ASTROLABE_EXECUTION_ENDPOINT')
    token = os.environ.get('ASTROLABE_EXECUTION_TOKEN')
    if bool(endpoint) != bool(token):
        raise ValueError('incomplete execution broker credentials')
    if not endpoint:
        child_env = os.environ.copy()
        if env:
            child_env.update(env)
        if device == 'cpu':
            child_env.update(CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='void',
                             HIP_VISIBLE_DEVICES='', ROCR_VISIBLE_DEVICES='',
                             JAX_PLATFORMS='cpu', GH_CUDA_PREINITIALIZE='0')
        elif device is not None:
            allocation = (os.environ.get('CUDA_VISIBLE_DEVICES') or '').split(',')
            if not isinstance(device, str) or not device or device not in allocation:
                raise ValueError('device is not in process allocation')
            child_env['CUDA_VISIBLE_DEVICES'] = device
        return _execute_probe(command, child_env, timeout)
    if device is None:
        device = os.environ.get('ASTROLABE_PREFERRED_CUDA_DEVICE') or None
    from .broker_client import BrokerClient, worker_overrides
    proxy = object.__new__(BrokerClient)
    proxy.endpoint, proxy.token, proxy.limit = endpoint, token, 1048576
    overrides = worker_overrides(env)
    return proxy._rpc({'op': 'probe', 'command': command, 'env': overrides, 'timeout': timeout, 'device': device,
                       'epoch': uuid.uuid4().hex, 'request_id': uuid.uuid4().hex},
                      timeout + 5, submitted=True)['result']


def _execution_proxy(*, inner=False):
    from .broker_client import BrokerClient
    from .client import WorkerError
    prefix = 'ASTROLABE_INNER_BROKER_' if inner else 'ASTROLABE_EXECUTION_'
    endpoint, token = os.environ.get(prefix + 'ENDPOINT'), os.environ.get(prefix + 'TOKEN')
    if bool(endpoint) != bool(token):
        raise WorkerError('incomplete execution broker credentials')
    if not endpoint:
        return None
    proxy = object.__new__(BrokerClient)
    proxy.endpoint, proxy.token, proxy.limit = endpoint, token, 1048576
    return proxy


def stage_event(phase, *, duration, identity=None, request_id=None, shape_key=None, **fields):
    """Record an observed CPU stage without claiming a device timing sample."""
    from .contracts import identity_key
    from .telemetry import EventLog
    event = {'phase': phase, 'duration': duration, **fields}
    if identity is not None:
        event['backend'] = identity_key(identity)
    if request_id is not None:
        event['request_id'] = request_id
    if shape_key is not None:
        event['shape_key'] = shape_key
    proxy = _execution_proxy() or _execution_proxy(inner=True)
    if proxy is not None:
        proxy._rpc({'op': 'stage', 'event': event}, 5, submitted=False)
    else:
        EventLog(os.environ.get('ASTROLABE_RUNTIME_TRACE')).write('stage', **event)


class OperationLease:
    """An exact device subgraph inside a declared managed worker request.

    Legacy direct execution keeps its original behavior. A GPU-hidden
    trajectory cannot use this context to change its own CUDA visibility.
    """
    def __init__(self, identity, *, phase='device', device=None, resident_bytes=None,
                 scratch_bytes=None, request_id=None, shape_key=None, timeout=None, process_ids=None):
        self.identity, self.phase = identity, phase
        from .contracts import OperationResources
        self.resources = OperationResources.parse({'resident_bytes': resident_bytes,
                                                    'scratch_bytes': scratch_bytes}).as_dict()
        self.request_id, self.shape_key = request_id, shape_key
        self.timeout = timeout
        self.process_ids = list(process_ids or [])
        self.lifecycle = {}
        self.proxy = None
        self.context = None

    def __enter__(self):
        from .server import current_request
        self.started = time.monotonic()
        self.proxy = _execution_proxy(inner=True)
        if self.proxy is None:
            if _execution_proxy() is not None and self.phase in ('device', 'load', 'unload'):
                raise RuntimeError('GPU-hidden trajectory requires an isolated worker for device operations')
            return self
        self.context = current_request()
        if not self.context:
            # Constructor work is already admitted by the broker's load lease.
            self.proxy = None
            return self
        if self.phase != 'device' or self.context.get('phase') != 'managed':
            raise ValueError('nested device operation requires a managed worker request')
        self.proxy._rpc({'op': 'nested_begin', 'owner_pid': os.getpid(),
                         'parent_request_id': self.context['request_id'],
                         'worker_epoch': self.context['worker_epoch'], 'resources': self.resources},
                        self.timeout, submitted=True)
        return self

    def record_process(self, pid):
        self.process_ids.append(int(pid))
        if self.proxy is not None:
            self.proxy._rpc({'op': 'nested_process', 'owner_pid': os.getpid(), 'pid': int(pid),
                             'parent_request_id': self.context['request_id'],
                             'worker_epoch': self.context['worker_epoch']}, self.timeout, submitted=True)

    def update_lifecycle(self, **fields):
        from .contracts import lifecycle_snapshot
        self.lifecycle.update(fields)
        self.lifecycle = lifecycle_snapshot(self.lifecycle)

    def __exit__(self, exc_type, exc, traceback):
        if self.proxy is not None:
            self.proxy._rpc({'op': 'nested_end', 'owner_pid': os.getpid(),
                             'parent_request_id': self.context['request_id'],
                             'worker_epoch': self.context['worker_epoch'],
                             'process_ids': self.process_ids, 'lifecycle': self.lifecycle,
                             'error': None if exc is None else str(exc)},
                            self.timeout, submitted=True)
        else:
            stage_event(self.phase, duration=time.monotonic() - self.started,
                        identity=self.identity, request_id=self.request_id, shape_key=self.shape_key,
                        admission='legacy_direct', lifecycle=self.lifecycle)


def operation(identity, **kwargs):
    return OperationLease(identity, **kwargs)
