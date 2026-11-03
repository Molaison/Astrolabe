"""Bounded, synchronous JSONL transport for a resident worker."""
from __future__ import annotations

import json
import os
import select
import signal
import subprocess
import threading
import time
import uuid
from .contracts import OperationResources, request_metadata


class WorkerError(RuntimeError):
    pass


class OutcomeUnknown(WorkerError):
    """The worker may have executed the request; never automatically replay it."""


def _encode(value, limit):
    data = (json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n').encode()
    if len(data) > limit:
        raise ValueError('JSONL message exceeds max_message_bytes')
    return data


class DaemonClient:
    def __new__(cls, command, identity, *, env=None, max_message_bytes=1048576, startup_timeout=60,
                resources=None, _direct=False, _pass_fds=(), _on_start=None):
        endpoint = os.environ.get('ASTROLABE_EXECUTION_ENDPOINT')
        token = os.environ.get('ASTROLABE_EXECUTION_TOKEN')
        if not _direct and (endpoint or token):
            if not endpoint or not token:
                raise WorkerError('incomplete execution broker credentials')
            from .broker_client import BrokerClient
            return BrokerClient(endpoint, token, command, identity, env=env,
                                max_message_bytes=max_message_bytes, startup_timeout=startup_timeout,
                                resources=resources)
        return super().__new__(cls)

    def __init__(self, command, identity, *, env=None, max_message_bytes=1048576, startup_timeout=60,
                 resources=None, _direct=False, _pass_fds=(), _on_start=None):
        if max_message_bytes < 256:
            raise ValueError('max_message_bytes must be at least 256')
        self.limit = max_message_bytes
        self.epoch = uuid.uuid4().hex
        self._lock = threading.Lock()
        self._used = set()
        self._buffer = b''
        self.resources = OperationResources.parse(resources).as_dict()
        self.lifecycle = {}
        self.last_metrics = {}
        child_env = os.environ.copy()
        if env:
            child_env.update(env)
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        env=child_env, bufsize=0, start_new_session=True,
                                        pass_fds=tuple(_pass_fds))
        from .telemetry import process_identity, process_namespace_supported
        self._namespace_verified = process_namespace_supported()
        self._processes = {}
        self._groups = {}
        self._closed = False
        self._release_error = None
        os.set_blocking(self.process.stdin.fileno(), False)
        try:
            initial = process_identity(self.process.pid) if self._namespace_verified else None
            if initial:
                self._processes[initial[0]] = initial
                self._groups[initial[0]] = initial[1]
            if _on_start is not None:
                _on_start(self.process)
            hello = self._read(startup_timeout)
            if (hello.get('type') != 'hello' or hello.get('protocol') != 1 or
                hello.get('identity') != identity or hello.get('pid') != self.process.pid or
                (hello.get('load_count') is not None and
                 (type(hello['load_count']) is not int or hello['load_count'] < 0))):
                raise WorkerError('worker identity or handshake mismatch')
            self.hello = hello
            self.lifecycle = dict(hello.get('lifecycle') or {})
            self._observe_children()
        except BaseException:
            self.close()
            raise

    def _read(self, timeout):
        deadline = None if timeout is None else time.monotonic() + timeout
        while b'\n' not in self._buffer:
            remaining = None if deadline is None else max(0, deadline - time.monotonic())
            ready, _, _ = select.select([self.process.stdout], [], [], remaining)
            if not ready:
                raise TimeoutError('worker response timed out')
            chunk = os.read(self.process.stdout.fileno(), min(65536, self.limit + 1 - len(self._buffer)))
            if not chunk:
                raise EOFError('worker exited')
            self._buffer += chunk
            if len(self._buffer) > self.limit:
                raise WorkerError('worker message exceeds max_message_bytes')
        line, self._buffer = self._buffer.split(b'\n', 1)
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise WorkerError('worker message must be an object')
        return obj

    def call(self, payload, *, timeout=None, request_id=None, runtime=None):
        if not isinstance(payload, dict):
            raise TypeError('payload must be a JSON object')
        if timeout is not None and timeout <= 0:
            raise ValueError('timeout must be positive')
        metadata = request_metadata(runtime)
        with self._lock:
            if self._closed:
                raise WorkerError('worker client is closed')
            self._observe_children()
            rid = request_id or uuid.uuid4().hex
            if not isinstance(rid, str) or not rid or rid in self._used:
                raise ValueError('request_id must be a new nonempty string')
            if len(self._used) >= 100000:
                raise WorkerError('worker request limit reached; start a new session')
            data = _encode(dict(type='call', request_id=rid, epoch=self.epoch, payload=payload,
                                runtime=metadata), self.limit)
            self._used.add(rid)
            try:
                deadline = None if timeout is None else time.monotonic() + timeout
                view = memoryview(data)
                while view:
                    remaining = None if deadline is None else max(0, deadline - time.monotonic())
                    _, ready, _ = select.select([], [self.process.stdin], [], remaining)
                    if not ready:
                        raise TimeoutError('worker request timed out')
                    view = view[os.write(self.process.stdin.fileno(), view):]
                remaining = None if deadline is None else max(0, deadline - time.monotonic())
                result = self._read(remaining)
                if result.get('request_id') != rid or result.get('epoch') != self.epoch:
                    raise WorkerError('response identity mismatch')
                if result.get('type') == 'error':
                    if result.get('error_type') == 'OutcomeUnknown':
                        raise OutcomeUnknown('worker outcome unknown: ' + str(result.get('error')))
                    raise WorkerError('worker error: ' + str(result.get('error')))
                if result.get('type') != 'result' or not isinstance(result.get('result'), dict):
                    raise WorkerError('invalid worker result')
                reported = result.get('runtime') or {}
                for field in ('scientific_request_id', 'request_identity'):
                    if field in metadata and reported.get(field) != metadata[field]:
                        raise WorkerError('scientific response identity mismatch')
                if (self.hello.get('worker_epoch') is not None and
                        reported.get('worker_epoch') != self.hello['worker_epoch']):
                    raise WorkerError('worker epoch mismatch')
                self.lifecycle = dict(result.get('lifecycle') or self.lifecycle)
                self.hello['lifecycle'] = self.lifecycle
                self.last_metrics = dict(reported)
                return result['result']
            except WorkerError as exc:
                self.close()
                if isinstance(exc, OutcomeUnknown):
                    raise
                if str(exc).startswith('worker error:'):
                    raise
                raise OutcomeUnknown('invalid response; request was not replayed') from exc
            except (OSError, EOFError, TimeoutError, ValueError) as exc:
                self.close()
                raise OutcomeUnknown('worker outcome unknown; request was not replayed') from exc

    def _observe_children(self):
        if not self._namespace_verified:
            return None
        from .telemetry import process_tree, known_process_records
        tree = process_tree(self._processes, groups=self._groups)
        tree.update(known_process_records(self._processes))
        self._processes.update({pid: (pid, info['start']) for pid, info in tree.items()})
        return tree

    def close(self, *, release=False):
        from .telemetry import ProcessObservationError
        try:
            self._close_process(release=release)
        except ProcessObservationError as exc:
            self.lifecycle = {**getattr(self, 'lifecycle', {}), 'process_alive': None,
                              'weights_loaded': None, 'gpu_resident': None, 'ready': False,
                              'release_confirmed': False}
            self._release_error = OutcomeUnknown('process state unknown during worker cleanup; resources were not released')
            raise self._release_error from exc

    def _close_process(self, *, release=False):
        if getattr(self, '_closed', False):
            if getattr(self, '_release_error', None) is not None:
                raise self._release_error
            return
        p = self.process
        from .telemetry import process_identity, known_process_records
        tree = self._observe_children()
        # A historical numerical PGID is unsafe after every original member
        # exited: the kernel may reuse it. Only a live, verified leader permits
        # group signalling; otherwise signal observed PID/start-time pairs.
        leader_live = (self._processes.get(p.pid) is not None and process_identity(p.pid) == self._processes[p.pid]
                       if self._namespace_verified else p.poll() is None)
        if leader_live:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        elif tree:
            for pid in tree:
                if process_identity(pid) == self._processes[pid]:
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
        if p.poll() is None:
            try:
                p.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        tree = self._observe_children()
        if tree is not None:
            for pid in tree:
                if process_identity(pid) == self._processes[pid]:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        elif p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if p.poll() is None:
            try:
                p.wait(timeout=2)
            except subprocess.TimeoutExpired as exc:
                self.lifecycle = {**getattr(self, 'lifecycle', {}), 'process_alive': True,
                                  'ready': False, 'release_confirmed': False}
                raise OutcomeUnknown('worker could not be stopped; resource owner must retain its lease') from exc
        for stream in (p.stdin, p.stdout):
            if stream:
                stream.close()
        tree = self._observe_children()
        deadline = time.monotonic() + 2
        while tree and time.monotonic() < deadline:
            time.sleep(.01)
            tree = self._observe_children()
        live_known = known_process_records(self._processes) if self._namespace_verified else {}
        confirmed = tree is not None and not tree and not tree.unverified_groups and not live_known
        self.lifecycle = {**getattr(self, 'lifecycle', {}), 'process_alive': False,
                          'weights_loaded': False if confirmed else None,
                          'gpu_resident': False if confirmed else None, 'ready': False,
                          'release_confirmed': confirmed,
                          'release_scope': 'registered_and_observed_processes',
                          'unassociated_processes_unobserved': 0 if tree is None else tree.unobserved_count}
        if hasattr(self, 'hello'):
            self.hello['lifecycle'] = self.lifecycle
        self._closed = True
        if confirmed:
            # A later explicit cleanup may obtain real exit evidence after an
            # earlier observation error. Only that confirmation resolves it.
            self._release_error = None
        if tree is not None and (tree or tree.unverified_groups or live_known):
            self._release_error = OutcomeUnknown('worker descendants could not be confirmed stopped')
            raise self._release_error

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
