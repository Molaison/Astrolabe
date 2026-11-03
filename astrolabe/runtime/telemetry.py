"""Read-only, allocation-scoped resource observations; never initializes CUDA.

NVIDIA utilization is a sampled activity signal, not useful throughput. A
missing driver/tool/sample stays unavailable rather than becoming zero busy.
"""
from __future__ import annotations

import csv
import errno
import io
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time

MIB = 1024 ** 2


def process_namespace_supported():
    """Some hosted executors virtualize getpid without virtualizing /proc."""
    try:
        return int(Path('/proc/self/stat').read_text().split(' ', 1)[0]) == os.getpid()
    except (OSError, ValueError):
        return False


def _require_process_namespace():
    if not process_namespace_supported():
        raise RuntimeError('process IDs and /proc do not share a namespace; resource ownership cannot be verified')


def memory_allocation():
    from .admission import _cgroup_v2_dirs
    try:
        physical = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    except (ValueError, OSError, AttributeError):
        physical = 1024 ** 3
    bounds = [(physical, None, 'host')]
    for path in _cgroup_v2_dirs():
        try:
            maximum = (path / 'memory.max').read_text().strip()
            if maximum != 'max' and int(maximum) > 0:
                bounds.append((int(maximum), int((path / 'memory.current').read_text()), 'cgroup_v2'))
        except (OSError, ValueError):
            pass
    try:
        groups = Path('/proc/self/cgroup').read_text().splitlines()
        relative = next(parts[2] for line in groups if len(parts := line.split(':', 2)) == 3
                        and 'memory' in parts[1].split(','))
        root = Path('/sys/fs/cgroup/memory')
        leaf = (root / relative.lstrip('/')).resolve()
        if leaf.is_relative_to(root):
            for path in [leaf, *[parent for parent in leaf.parents if parent.is_relative_to(root)]]:
                try:
                    maximum = int((path / 'memory.limit_in_bytes').read_text())
                    current = int((path / 'memory.usage_in_bytes').read_text())
                    if 0 < maximum < physical:
                        bounds.append((maximum, current, 'cgroup_v1'))
                except (OSError, ValueError):
                    pass
    except (OSError, StopIteration, ValueError):
        pass
    # Slurm environment is an additional ceiling, never a reason to widen a
    # cgroup/affinity bound. These variables are MiB, not bytes.
    for key in ('SLURM_MEM_PER_NODE', 'SLURM_MEM_PER_CPU'):
        try:
            quantity = int(os.environ.get(key, '0')) * MIB
            if quantity > 0:
                if key == 'SLURM_MEM_PER_CPU':
                    from .admission import available_cpus
                    quantity *= available_cpus()
                bounds.append((quantity, None, key))
        except ValueError:
            pass
    maximum, current, source = min(bounds, key=lambda entry: entry[0])
    if current is None:
        try:
            fields = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
            available = int(fields['MemAvailable'].split()[0]) * 1024
            # Host available memory is a pressure signal. It is not this job's
            # RSS and must not be presented as such.
            host_pressure = 1 - available / physical
        except (OSError, KeyError, ValueError):
            host_pressure = None
    else:
        host_pressure = None
    return {'limit_bytes': maximum, 'current_bytes': current, 'source': source,
            'pressure': None if current is None else current / maximum,
            'host_pressure': host_pressure}


class ProcessTree(dict):
    """Only identity-anchored processes may be signalled by their owner."""
    def __init__(self, *args, unverified_groups=(), unobserved_pids=(), unobserved_count=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.unverified_groups = tuple(unverified_groups)
        self.unobserved_count = len(unobserved_pids) if unobserved_count is None else unobserved_count
        self.unobserved_pids = tuple(unobserved_pids)[:16]


class ProcessObservationError(RuntimeError):
    """Process state could not be read; this is never evidence of exit."""


def _process_record(pid):
    pid = int(pid)
    try:
        text = Path(f'/proc/{pid}/stat').read_text(errors='surrogateescape')
    except OSError as exc:
        if exc.errno in (errno.ENOENT, errno.ESRCH):
            return None
        raise ProcessObservationError(f'cannot observe process {pid}: {exc}') from exc
    try:
        if int(text.split(' ', 1)[0]) != pid or text.rfind(')') < 0:
            raise ValueError('PID or stat framing mismatch')
        fields = text[text.rfind(')') + 2:].split()
        record = {'pid': pid, 'ppid': int(fields[1]), 'start': int(fields[19]),
                  'pgrp': int(fields[2]), 'rss': int(fields[21]) * os.sysconf('SC_PAGE_SIZE'),
                  'cpu': int(fields[11]) + int(fields[12]), 'state': fields[0]}
        if any(record[name] < 0 for name in ('ppid', 'start', 'pgrp', 'rss', 'cpu')):
            raise ValueError('negative process counter')
    except (ValueError, IndexError) as exc:
        raise ProcessObservationError(f'invalid stat for process {pid}') from exc
    return None if record['state'] in ('Z', 'X', 'x') else record


def process_record(pid):
    """Read one PID directly; only a missing/dead process returns None.

    Permission, I/O and format failures remain unknown and raise. A caller
    retaining an identity must also compare this record's start time.
    """
    _require_process_namespace()
    return _process_record(pid)


def known_process_records(identities):
    """Reconcile registered PID/start pairs independently of /proc listing."""
    _require_process_namespace()
    result = {}
    for pid, identity in identities.items():
        record = _process_record(pid)
        if record is not None and record['start'] == tuple(identity)[1]:
            result[int(pid)] = record
    return result


def process_tree(root_pids, *, groups=None):
    """Return PID and start-time pairs so PID reuse cannot release a lease."""
    _require_process_namespace()
    records = {}
    unobserved = []
    unobserved_count = 0
    strict = set(root_pids) | set(groups or {})
    try:
        paths = Path('/proc').iterdir()
        for path in paths:
            if not path.name.isdecimal():
                continue
            pid = int(path.name)
            try:
                record = _process_record(pid)
            except ProcessObservationError:
                if pid in strict:
                    raise
                # In a shared host namespace an unreadable PID need not
                # belong to this job. Keep discovery limits explicit without
                # capturing it or treating a registered process as exited.
                unobserved_count += 1
                if len(unobserved) < 16:
                    unobserved.append(pid)
                continue
            if record is not None:
                records[pid] = record
    except OSError as exc:
        raise ProcessObservationError('cannot enumerate process namespace') from exc
    if isinstance(root_pids, dict):
        # /proc iteration and task creation/exit are not an atomic snapshot.
        # A previously verified PID may be missing from this enumeration while
        # still alive. Read each such identity directly before declaring exit.
        missing = {pid: identity for pid, identity in root_pids.items()
                   if pid not in records or records[pid]['start'] != tuple(identity)[1]}
        records.update(known_process_records(missing))
        selected = {int(pid) for pid, identity in root_pids.items() if pid in records and
                    records[pid]['start'] == tuple(identity)[1] and records[pid]['state'] != 'Z'}
    else:
        selected = set(int(pid) for pid in root_pids)
    if groups is not None and not isinstance(groups, dict):
        raise ValueError('process groups require original leader start-time identities')
    unverified = []
    for group, leader_start in (groups or {}).items():
        members = {pid for pid, info in records.items() if info['pgrp'] == group and info['state'] != 'Z'}
        leader = records.get(group)
        anchored = (leader is not None and leader['start'] == leader_start and leader['state'] != 'Z')
        anchored |= bool(members & selected)
        if anchored:
            selected.update(members)
        elif members:
            # The old leader exited and every observed member disappeared.
            # A numerical PGID alone can now refer to a different process
            # group. Do not capture or kill it; report uncertain cleanup.
            unverified.append(group)
    changed = True
    while changed:
        children = {pid for pid, info in records.items() if info['ppid'] in selected}
        changed = not children.issubset(selected)
        selected.update(children)
    return ProcessTree({pid: records[pid] for pid in selected if pid in records and records[pid]['state'] != 'Z'},
                       unverified_groups=unverified, unobserved_pids=unobserved,
                       unobserved_count=unobserved_count)


def process_identity(pid):
    record = process_record(pid)
    return None if record is None else (int(pid), record['start'])


def _number(value, multiplier=1):
    try:
        result = float(value.strip()) * multiplier
        if not math.isfinite(result) or result < 0:
            return None
        return result if multiplier == 1 else int(result)
    except ValueError:
        return None


class NvidiaTelemetry:
    """nvidia-smi is queried only for explicitly allocated device tokens."""
    def __init__(self, timeout=2.0):
        self.timeout = timeout
        self._resolved = {}
        self._lock = threading.Lock()

    def _query(self, token, fields):
        command = ['nvidia-smi', '--id=' + token, '--query-gpu=' + ','.join(fields),
                   '--format=csv,noheader,nounits']
        result = subprocess.run(command, capture_output=True, text=True, timeout=self.timeout, check=True)
        rows = list(csv.reader(io.StringIO(result.stdout)))
        if len(rows) != 1 or len(rows[0]) != len(fields):
            raise ValueError('allocated token did not resolve to exactly one physical GPU')
        return [entry.strip() for entry in rows[0]]

    def resolve(self, token, *, mock=False):
        if mock:
            return {'token': token, 'uuid': 'mock:' + token, 'total_bytes': None, 'mock': True}
        with self._lock:
            if token in self._resolved:
                return dict(self._resolved[token])
        # Do not treat a token string as a physical identity: index and UUID
        # aliases must collide on the same owner lock. MIG needs a parent/slice
        # resource model and is deliberately not guessed here.
        if token.startswith('MIG-'):
            raise ValueError('MIG allocation requires an explicit supported parent/slice adapter')
        try:
            uuid, total = self._query(token, ['uuid', 'memory.total'])
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            raise RuntimeError('cannot resolve allocated GPU to a physical UUID; no GPU was admitted') from exc
        if not uuid.startswith('GPU-') or _number(total, MIB) is None:
            raise ValueError('driver returned an incomplete physical GPU identity')
        resolved = {'token': token, 'uuid': uuid, 'total_bytes': _number(total, MIB), 'mock': False}
        with self._lock:
            self._resolved[token] = resolved
        return dict(resolved)

    def sample(self, device):
        stamp = time.monotonic()
        if device.get('mock'):
            return {'timestamp': stamp, 'uuid': device['uuid'], 'available': False, 'reason': 'mock_device'}
        try:
            uuid, total, used, free, busy = self._query(device['uuid'],
                ['uuid', 'memory.total', 'memory.used', 'memory.free', 'utilization.gpu'])
            result = {'timestamp': stamp, 'uuid': uuid, 'available': True,
                      'total_bytes': _number(total, MIB), 'used_bytes': _number(used, MIB),
                      'free_bytes': _number(free, MIB), 'gpu_activity': _number(busy)}
            if None in (result['total_bytes'], result['used_bytes'], result['free_bytes']):
                result.update(available=False, reason='memory_counters_unavailable')
            return result
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            return {'timestamp': stamp, 'uuid': device['uuid'], 'available': False,
                    'reason': type(exc).__name__}

    def process_memory(self, device, pids):
        if device.get('mock'):
            return None
        try:
            output = subprocess.run(['nvidia-smi', '--id=' + device['uuid'],
                                     '--query-compute-apps=pid,used_gpu_memory',
                                     '--format=csv,noheader,nounits'], capture_output=True,
                                    text=True, timeout=self.timeout, check=True).stdout
            aliases = set(pids)
            for pid in pids:
                try:
                    for line in Path(f'/proc/{pid}/status').read_text().splitlines():
                        if line.startswith('NSpid:'):
                            aliases.update(int(value) for value in line.split()[1:])
                except (OSError, ValueError):
                    pass
            total, observed, matched = 0, False, False
            for row in csv.reader(io.StringIO(output)):
                if len(row) != 2 or not row[0].strip().isdecimal():
                    return None
                observed = True
                if int(row[0]) in aliases:
                    matched = True
                    memory = _number(row[1], MIB)
                    if memory is None:
                        return None
                    total += memory
            # NVML may expose host PIDs while /proc exposes only a container
            # namespace. An unmatched driver process is not proof of zero
            # residency. Preserve the unknown reservation in that case.
            return None if observed and not matched else total
        except (OSError, ValueError, subprocess.SubprocessError):
            return None


class EventLog:
    """Bounded JSONL timeline; existing files are never silently overwritten."""
    def __init__(self, path=None, max_bytes=64 * 1024 ** 2):
        self.path = str(path) if path else None
        self.max_bytes = max_bytes
        self.lock = threading.Lock()
        self.dropped = 0

    def write(self, kind, **fields):
        if not self.path:
            return
        record = {'event': kind, 'time': time.monotonic(), 'time_monotonic_ns': time.monotonic_ns(),
                  'wall_time': time.time(), **fields}
        raw = (json.dumps(record, allow_nan=False, separators=(',', ':')) + '\n').encode()
        with self.lock:
            path = Path(self.path)
            path.parent.mkdir(parents=True, exist_ok=True)
            size = path.stat().st_size if path.exists() else 0
            if size + len(raw) > self.max_bytes:
                self.dropped += 1
                return
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.write(fd, raw)
            finally:
                os.close(fd)
