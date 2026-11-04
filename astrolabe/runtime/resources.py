"""The broker's single resource ledger and physical-GPU ownership.

All same-card work is serial until a backend/configuration is actually
certified for overlap. Resident allocations outlive compute leases. Unknown
estimates reserve the remaining budget; absence of telemetry never grants
additional capacity.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import threading
import time

from .contracts import OperationResources, ResourceLimits
from .telemetry import NvidiaTelemetry, process_identity, process_tree, known_process_records


class ResourceExhausted(RuntimeError):
    pass


class DeviceOwned(RuntimeError):
    pass


class PhysicalGPUOwner:
    """An OS lease, shared by all Astrolabe sessions of the current user.

    The lock uses physical UUID, not a CUDA ordinal. Other users and unrelated
    programs remain the batch scheduler's responsibility. Children inherit the
    descriptor so broker death does not immediately advertise a still-live
    resident process as released.
    """
    def __init__(self, device, *, guarded=False):
        self.device = device
        # TMPDIR is commonly private to one batch job. It cannot name the
        # physical-device ownership domain. Containers must bind this same
        # directory or provide an exclusive allocation at their outer layer.
        root = Path(os.environ.get('ASTROLABE_GPU_OWNER_DIRECTORY', '/tmp'))
        if not root.is_absolute():
            raise ValueError('ASTROLABE_GPU_OWNER_DIRECTORY must be absolute')
        directory = root / f'astrolabe-gpu-owners-{os.getuid()}'
        directory.mkdir(mode=0o700, exist_ok=True)
        stat = directory.lstat()
        if directory.is_symlink() or stat.st_uid != os.getuid() or stat.st_mode & 0o077:
            raise PermissionError('GPU owner directory is not private to the current user')
        filename = hashlib.sha256(device['uuid'].encode()).hexdigest() + '.lock'
        self.path = directory / filename
        self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT | getattr(os, 'O_NOFOLLOW', 0), 0o600)
        self.guardian = None
        self.control_fd = None
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(self.fd)
            self.fd = None
            raise DeviceOwned(f"physical GPU {device['uuid']} already has an Astrolabe admission owner") from exc
        record = {'pid': os.getpid(), 'physical_uuid': device['uuid'], 'started': time.time()}
        os.ftruncate(self.fd, 0)
        os.write(self.fd, json.dumps(record).encode())
        if guarded:
            from .telemetry import _require_process_namespace
            _require_process_namespace()
            read_fd, write_fd = os.pipe()
            try:
                self.guardian = subprocess.Popen(
                    [sys.executable, '-m', 'astrolabe.runtime.guardian', str(read_fd), str(self.fd)],
                    pass_fds=(read_fd, self.fd), start_new_session=True,
                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'NVIDIA_VISIBLE_DEVICES': 'void',
                         'HIP_VISIBLE_DEVICES': '', 'ROCR_VISIBLE_DEVICES': '', 'JAX_PLATFORMS': 'cpu',
                         'GH_CUDA_PREINITIALIZE': '0'})
                self.control_fd = write_fd
            except BaseException:
                os.close(write_fd)
                os.close(self.fd)
                self.fd = None
                raise
            finally:
                os.close(read_fd)

    def register_process(self, pid, *, group=False):
        if self.control_fd is not None:
            if self.guardian.poll() is not None:
                raise ResourceExhausted('GPU guardian exited before worker registration')
            identity = process_identity(pid)
            if identity is None:
                raise ValueError('worker exited before guardian registration')
            record = {'identity': identity, 'group': group}
            os.write(self.control_fd, (json.dumps(record) + '\n').encode())

    def close(self):
        if self.control_fd is not None:
            os.close(self.control_fd)
            self.control_fd = None
        if self.fd is not None:
            # Do not unlink: replacing an inode would permit another owner
            # while a worker still holds the old lock descriptor.
            os.close(self.fd)
            self.fd = None
        if self.guardian is not None:
            try:
                self.guardian.wait(timeout=3)
            except subprocess.TimeoutExpired:
                # An unkillable child is OUTCOME_UNKNOWN. The guardian MUST
                # retain the physical lease, not be killed to make shutdown
                # look successful. A new owner will fail closed on the flock.
                return False
        return True


class ResourceLedger:
    def __init__(self, limits=None, *, telemetry=None, events=None):
        self.limits = limits if isinstance(limits, ResourceLimits) else ResourceLimits.from_mapping(limits)
        self.telemetry = telemetry or NvidiaTelemetry(self.limits.telemetry_timeout)
        self.events = events
        self.condition = threading.Condition(threading.RLock())
        self.devices = {}
        self.fenced_devices = set()
        self.owners = {}
        self.residents = {}
        self.pending = {}
        self.active = {}
        self.prepared_bytes = 0
        self.output_bytes = 0
        self.ram_bytes = 0
        self.cpu_slots = 0
        self.stopping = False

    def device(self, token, *, mock=False):
        if not token:
            return 'cpu'
        device = self.telemetry.resolve(token, mock=mock)
        key = device['uuid']
        with self.condition:
            if key not in self.devices:
                owner = PhysicalGPUOwner(device, guarded=not device.get('mock'))
                self.owners[key] = owner
                self.devices[key] = device
                if self.events:
                    self.events.write('device_owned', device=device)
        return key

    def inherited_fds(self, device_key):
        with self.condition:
            owner = self.owners.get(device_key)
            if owner is None:
                return ()
            return (owner.fd,) if owner.control_fd is None else (owner.fd, owner.control_fd)

    def guarded_command(self, device_key, command):
        with self.condition:
            owner = self.owners.get(device_key)
            if owner is None or owner.control_fd is None:
                return command
            return [sys.executable, '-S', str(Path(__file__).with_name('guarded_launch.py')),
                    str(owner.control_fd), json.dumps(command)]

    def _capacity(self, device_key):
        device = self.devices.get(device_key)
        if not device or device['total_bytes'] is None:
            return None
        return int(device['total_bytes'] * self.limits.vram_fraction)

    def reserve_resident(self, key, device, estimate=None, *, owner_pid=None):
        resources = OperationResources.parse({'resident_bytes': estimate})
        with self.condition:
            if key in self.residents:
                raise ValueError('resident instance is already registered')
            capacity = self._capacity(device)
            others = [r for r in self.residents.values() if r['device'] == device]
            used = sum(r['reserved_bytes'] or 0 for r in others)
            # The first unknown load reserves remaining room, not a fictional
            # measured size. Startup itself still owns the entire compute slot.
            reservation = resources.resident_bytes
            observed_free = None
            if device in self.devices and not self.devices[device].get('mock'):
                sample = self.telemetry.sample(self.devices[device])
                if not sample.get('available'):
                    raise ResourceExhausted('GPU memory observation unavailable before model load')
                if sample['used_bytes'] >= capacity:
                    raise ResourceExhausted('actual GPU memory pressure exceeds the job VRAM budget')
                observed_free = max(0, capacity - sample['used_bytes'])
            if reservation is None and capacity is not None:
                reservation = max(0, capacity - used)
                if observed_free is not None:
                    reservation = min(reservation, observed_free)
            if observed_free is not None and reservation > observed_free:
                raise ResourceExhausted('requested model residency exceeds actual GPU headroom')
            if capacity is not None and (used >= capacity or used + (reservation or 0) > capacity):
                raise ResourceExhausted('resident VRAM budget is exhausted')
            if device != 'cpu' and capacity is None and not self.devices[device].get('mock'):
                if others:
                    raise ResourceExhausted('unknown residency prevents another GPU model load')
            record = {'device': device, 'reserved_bytes': reservation,
                      'measured_bytes': None, 'estimate_unknown': estimate is None,
                      'owner': process_identity(owner_pid) if owner_pid else None,
                      'processes': {}, 'groups': {}, 'unverified_groups': [],
                      'lifecycle': {'weights_loaded': None, 'gpu_resident': None}}
            self.residents[key] = record
            return dict(record)

    def bind_process(self, key, pid, *, managed_group=False):
        identity = process_identity(pid)
        if identity is None:
            raise ValueError('cannot bind a nonexistent process')
        with self.condition:
            resident = self.residents[key]
            resident['processes'][pid] = identity
            if managed_group:
                if os.getpgid(pid) != pid:
                    raise ValueError('managed worker must own its process group')
                resident['groups'][pid] = identity[1]
            owner = self.owners.get(resident['device'])
            if owner is not None:
                owner.register_process(pid, group=managed_group)
            if resident['owner'] is None:
                resident['owner'] = identity
            tree = process_tree(resident['processes'], groups=resident['groups'])
            resident['processes'].update({child: (child, info['start']) for child, info in tree.items()})

    def update_resident(self, key, *, lifecycle=None, measured_bytes=None):
        with self.condition:
            record = self.residents[key]
            if lifecycle is not None:
                record['lifecycle'] = dict(lifecycle)
            if measured_bytes is not None:
                OperationResources.parse({'resident_bytes': measured_bytes})
                # Drivers include contexts and allocator pools. This is a
                # conservative physical reservation, not just weight bytes.
                record['measured_bytes'] = measured_bytes
                record['reserved_bytes'] = measured_bytes
                capacity = self._capacity(record['device'])
                occupied = sum(r['reserved_bytes'] or 0 for r in self.residents.values()
                               if r['device'] == record['device'])
                if capacity is not None and occupied > capacity:
                    raise ResourceExhausted('observed resident VRAM exceeds the job budget')
            self.condition.notify_all()

    def observe_resident(self, key):
        with self.condition:
            record = self.residents[key]
            device = self.devices.get(record['device'])
            roots = dict(record['processes'])
        pids = process_tree(roots, groups=record['groups'])
        with self.condition:
            record['processes'].update({pid: (pid, info['start']) for pid, info in pids.items()})
            record['unverified_groups'] = list(pids.unverified_groups)
            record['unassociated_processes_unobserved'] = pids.unobserved_count
            identities = dict(record['processes'])
        if device:
            # A partial enumeration must not turn an already known live CUDA
            # child into apparent free VRAM. Unknown stat data raises before
            # the previous reservation can be reduced.
            live = known_process_records(identities)
            measured = None if pids.unverified_groups else self.telemetry.process_memory(device, live)
            if measured == 0 and record['lifecycle'].get('gpu_resident') is True:
                measured = None
            self.update_resident(key, measured_bytes=measured)
            return measured
        return None

    def release_resident(self, key, *, confirmed=False):
        with self.condition:
            record = self.residents.get(key)
            if record is None:
                return True
            descendants = process_tree(record['processes'], groups=record['groups'])
            record['processes'].update({pid: (pid, info['start']) for pid, info in descendants.items()})
            record['unverified_groups'] = list(descendants.unverified_groups)
            record['unassociated_processes_unobserved'] = descendants.unobserved_count
            alive = [pid for pid, identity in record['processes'].items() if process_identity(pid) == identity]
            if record['unverified_groups'] or alive and not confirmed:
                # Python object deletion/offload is not proof that the device
                # allocator or a descendant released its allocation.
                return False
            self.residents.pop(key)
            self.condition.notify_all()
            if self.events:
                self.events.write('resident_released', resident=key, device=record['device'],
                                  process_exit=not alive, confirmed=confirmed)
            return True

    def _wait(self, predicate, deadline, reason, cancel_token=None):
        from .client import AdmissionRejected
        while True:
            if cancel_token and cancel_token.is_set():
                raise AdmissionRejected('request cancelled before submission')
            if self.stopping:
                raise ResourceExhausted('resource ledger is stopping')
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise TimeoutError(reason)
            if predicate():
                return
            self.condition.wait(remaining)

    def admit_pending(self, token, resources, *, timeout=None, deadline=None, cancel_token=None):
        resources = OperationResources.parse(resources)
        if resources.output_bytes is None:
            cap = self.limits.request_output_bytes or max(1, min(1024 ** 3, self.limits.output_bytes // 2))
            resources = replace(resources, output_bytes=cap)
        if resources.ram_bytes is None:
            # This is conservative admission headroom, not a measurement of
            # backend memory. Live process-tree RSS is independently bounded
            # by the broker and the scheduler's cgroup limit.
            cap = max(resources.prepared_bytes, min(64 * 1024 ** 2, max(1, self.limits.ram_bytes // 2)))
            resources = replace(resources, ram_bytes=cap)
        for value, cap, name in ((resources.prepared_bytes, self.limits.prepared_bytes, 'prepared'),
                                 (resources.output_bytes, self.limits.output_bytes, 'output'),
                                 (resources.ram_bytes, self.limits.ram_bytes, 'RAM')):
            if value > cap:
                raise ResourceExhausted(f'request exceeds {name} byte budget')
        deadline = deadline if deadline is not None else (None if timeout is None else time.monotonic() + timeout)
        with self.condition:
            if token in self.pending:
                raise ValueError('duplicate resource admission')
            def room():
                return (len(self.pending) < self.limits.pending_items and
                        self.prepared_bytes + resources.prepared_bytes <= self.limits.prepared_bytes and
                        self.output_bytes + resources.output_bytes <= self.limits.output_bytes and
                        self.ram_bytes + resources.ram_bytes <= self.limits.ram_bytes)
            self._wait(room, deadline, 'prepared/output/RAM admission timed out', cancel_token)
            self.pending[token] = resources
            self.prepared_bytes += resources.prepared_bytes
            self.output_bytes += resources.output_bytes
            self.ram_bytes += resources.ram_bytes

    def release_compute(self, token):
        """Yield compute while retaining prepared/output and scratch ownership."""
        with self.condition:
            active = self.active.pop(token, None)
            if active:
                self.cpu_slots -= active['cpu_slots']
            self.condition.notify_all()

    def fence(self, device):
        if device != 'cpu':
            with self.condition:
                self.fenced_devices.add(device)
                self.condition.notify_all()

    def acquire_compute(self, token, device, *, timeout=None, cpu=False, cpu_work=True, deadline=None, cancel_token=None):
        deadline = deadline if deadline is not None else (None if timeout is None else time.monotonic() + timeout)
        with self.condition:
            resources = self.pending[token]
            charge = resources.cpu_slots if cpu or device == 'cpu' or cpu_work else 0
            if charge > self.limits.cpu_slots:
                raise ResourceExhausted('request exceeds CPU slot budget')
            capacity = self._capacity(device)
            resident = sum(r['reserved_bytes'] or 0 for r in self.residents.values() if r['device'] == device)
            scratch = resources.scratch_bytes
            if scratch is None and capacity is not None:
                scratch = max(0, capacity - resident)
            if capacity is not None and resources.scratch_bytes is not None and resident + scratch > capacity:
                raise ResourceExhausted('resident plus requested scratch VRAM exceeds budget')
            def room():
                if device in self.fenced_devices:
                    from .client import OutcomeUnknown
                    raise OutcomeUnknown('device fenced after uncertain worker outcome')
                return (self.cpu_slots + charge <= self.limits.cpu_slots and
                        (cpu or device == 'cpu' or
                         not any(a['device'] == device and not a['cpu'] for a in self.active.values())))
            self._wait(room, deadline, 'compute admission timed out', cancel_token)
            if not cpu and device in self.devices and not self.devices[device].get('mock'):
                sample = self.telemetry.sample(self.devices[device])
                if not sample.get('available'):
                    raise ResourceExhausted('GPU memory observation unavailable at compute admission')
                if sample['used_bytes'] > capacity:
                    raise ResourceExhausted('actual GPU memory pressure exceeds the job VRAM budget')
                headroom = max(0, capacity - sample['used_bytes'])
                if resources.scratch_bytes is not None and resources.scratch_bytes > headroom:
                    raise ResourceExhausted('requested scratch exceeds actual GPU headroom')
                if resources.scratch_bytes is None:
                    scratch = min(scratch, headroom)
            self.active[token] = {'device': device, 'scratch_bytes': scratch, 'cpu': cpu or device == 'cpu',
                                  'cpu_slots': charge}
            self.cpu_slots += charge

    def release(self, token):
        with self.condition:
            active = self.active.pop(token, None)
            if active:
                self.cpu_slots -= active['cpu_slots']
            resources = self.pending.pop(token, None)
            if resources:
                self.prepared_bytes -= resources.prepared_bytes
                self.output_bytes -= resources.output_bytes
                self.ram_bytes -= resources.ram_bytes
            self.condition.notify_all()

    @contextmanager
    def lease(self, token, device, resources=None, *, timeout=None, cpu=False, cpu_work=True, deadline=None):
        deadline = deadline if deadline is not None else (None if timeout is None else time.monotonic() + timeout)
        self.admit_pending(token, resources, deadline=deadline)
        try:
            self.acquire_compute(token, device, deadline=deadline, cpu=cpu, cpu_work=cpu_work)
            yield
        finally:
            self.release(token)

    def snapshot(self, *, deadline=None):
        remaining = 2.0 if deadline is None else deadline - time.monotonic()
        if remaining <= 0 or not self.condition.acquire(timeout=remaining):
            raise TimeoutError('resource snapshot deadline expired')
        try:
            return {'limits': asdict(self.limits), 'prepared_bytes': self.prepared_bytes,
                    'output_reserved_bytes': self.output_bytes, 'ram_reserved_bytes': self.ram_bytes,
                    'cpu_slots': self.cpu_slots, 'pending': len(self.pending),
                    'active': {key: dict(value) for key, value in self.active.items()},
                    'resident': {key: {**value, 'processes': dict(value['processes'])}
                                 for key, value in self.residents.items()},
                    'devices': {key: dict(value) for key, value in self.devices.items()}}
        finally:
            self.condition.release()

    def close(self):
        with self.condition:
            self.stopping = True
            self.condition.notify_all()
            owners = list(self.owners.values())
            self.owners.clear()
        confirmed = True
        for owner in owners:
            confirmed = owner.close() and confirmed
        return confirmed
