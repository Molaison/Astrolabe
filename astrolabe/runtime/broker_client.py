"""Synchronous broker proxy retaining the DaemonClient API."""
import json
import os
import socket
import threading
import uuid

from .client import OutcomeUnknown, WorkerError, _encode
from .contracts import OperationResources, request_metadata


def worker_overrides(env):
    """Strip owner-only masks; broker supplies the actual allocated devices."""
    result = dict(env or {})
    if os.environ.get('ASTROLABE_CPU_OWNER') == '1':
        for key, hidden in (('CUDA_VISIBLE_DEVICES', ''), ('NVIDIA_VISIBLE_DEVICES', 'void'),
                            ('HIP_VISIBLE_DEVICES', ''), ('ROCR_VISIBLE_DEVICES', ''),
                            ('JAX_PLATFORMS', 'cpu'), ('GH_CUDA_PREINITIALIZE', '0')):
            if result.get(key) == hidden:
                result.pop(key, None)
    for key in ('ASTROLABE_EXECUTION_ENDPOINT', 'ASTROLABE_EXECUTION_TOKEN',
                'ASTROLABE_ALLOCATED_CUDA_DEVICES', 'ASTROLABE_CPU_OWNER'):
        result.pop(key, None)
    return result


class BrokerClient:
    def __init__(self, endpoint, token, command, identity, *, env, max_message_bytes, startup_timeout,
                 resources=None):
        self.endpoint, self.token, self.limit = endpoint, token, max_message_bytes
        self.epoch = uuid.uuid4().hex
        self._lock = threading.Lock()
        self._used = set()
        self._closed = False
        self._release_confirmed = False
        self._release_error = None
        self.resources = OperationResources.parse(resources).as_dict()
        response = self._rpc({'op': 'register', 'command': command, 'identity': identity,
                              'env': worker_overrides(env), 'limit': max_message_bytes,
                              'startup_timeout': startup_timeout, 'resources': self.resources,
                              'client_epoch': self.epoch, 'owner_pid': os.getpid()},
                             startup_timeout + 10, submitted=False)
        self.handle, self.hello = response['handle'], response['hello']
        self.lifecycle = dict(self.hello.get('lifecycle') or {})
        self.last_metrics = {}

    def _rpc(self, request, timeout, *, submitted):
        request['token'] = self.token
        allocation = os.environ.get('ASTROLABE_ALLOCATED_CUDA_DEVICES')
        is_owner = os.environ.get('ASTROLABE_CPU_OWNER') == '1'
        if is_owner and allocation is None:
            raise WorkerError('trajectory owner allocation assertion is missing')
        if allocation is None:
            # Compatibility for explicit broker clients living in the launcher.
            allocation = os.environ.get('CUDA_VISIBLE_DEVICES') or ''
        if is_owner:
            masks = {'CUDA_VISIBLE_DEVICES': '', 'NVIDIA_VISIBLE_DEVICES': 'void',
                     'HIP_VISIBLE_DEVICES': '', 'ROCR_VISIBLE_DEVICES': '',
                     'JAX_PLATFORMS': 'cpu', 'GH_CUDA_PREINITIALIZE': '0'}
            if any(os.environ.get(key) != value for key, value in masks.items()):
                raise WorkerError('trajectory owner device isolation was changed')
        request['allocated_cuda_devices'] = allocation
        data = _encode(request, self.limit + 65536)
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(timeout)
                sock.connect('\0' + self.endpoint[1:] if self.endpoint.startswith('@') else self.endpoint)
                sock.sendall(data)
                chunks = bytearray()
                while not chunks.endswith(b'\n'):
                    part = sock.recv(min(65536, self.limit + 65537 - len(chunks)))
                    if not part:
                        raise EOFError('broker closed connection')
                    chunks.extend(part)
                    if len(chunks) > self.limit + 65536:
                        raise ValueError('broker response too large')
            response = json.loads(chunks)
        except (OSError, EOFError, ValueError) as exc:
            if submitted:
                raise OutcomeUnknown('broker outcome unknown; request was not replayed') from exc
            raise WorkerError('broker connection failed') from exc
        if not isinstance(response, dict):
            if submitted:
                raise OutcomeUnknown('invalid broker response; request was not replayed')
            raise WorkerError('invalid broker response')
        if not response.get('ok'):
            if response.get('unknown'):
                raise OutcomeUnknown(response.get('error', 'broker outcome unknown'))
            raise WorkerError(response.get('error', 'broker request failed'))
        return response

    def call(self, payload, *, timeout=None, request_id=None, runtime=None):
        if not isinstance(payload, dict):
            raise TypeError('payload must be a JSON object')
        _encode(payload, self.limit)
        if timeout is not None and timeout <= 0:
            raise ValueError('timeout must be positive')
        metadata = request_metadata(runtime)
        with self._lock:
            if self._closed:
                raise WorkerError('broker client is closed')
            rid = request_id or uuid.uuid4().hex
            if not isinstance(rid, str) or not rid or rid in self._used:
                raise ValueError('request_id must be a new nonempty string')
            if len(self._used) >= 100000:
                raise WorkerError('client request limit reached; start a new session')
            self._used.add(rid)
            response = self._rpc({'op': 'call', 'handle': self.handle, 'epoch': self.epoch,
                                  'request_id': rid, 'payload': payload, 'timeout': timeout,
                                  'runtime': metadata},
                                 None if timeout is None else timeout + 5, submitted=True)
            result = response.get('result')
            if not isinstance(result, dict):
                raise OutcomeUnknown('invalid broker result; request was not replayed')
            if response.get('request_id') != rid or response.get('epoch') != self.epoch:
                raise OutcomeUnknown('broker result identity mismatch; request was not replayed')
            self.lifecycle = dict(response.get('lifecycle') or self.lifecycle)
            self.hello['lifecycle'] = self.lifecycle
            self.last_metrics = dict(response.get('runtime') or {})
            for field in ('scientific_request_id', 'request_identity'):
                if field in metadata and self.last_metrics.get(field) != metadata[field]:
                    raise OutcomeUnknown('broker scientific identity mismatch; request was not replayed')
            return result

    def close(self, *, release=False):
        # A private trajectory-owned model opts into release. Shared exact-
        # identity models may remain idle in the broker until pressure/exit.
        with self._lock:
            if self._closed:
                if release and not getattr(self, '_release_confirmed', False):
                    error = getattr(self, '_release_error', None)
                    if error is not None:
                        raise error
                    raise OutcomeUnknown('worker handle is detached but resource release was never confirmed')
                return
            try:
                response = self._rpc({'op': 'detach', 'handle': self.handle, 'client_epoch': self.epoch,
                                      'release': bool(release)}, 10, submitted=False)
                self._release_confirmed = response.get('released') is True
                if self._release_confirmed:
                    self.lifecycle = {**getattr(self, 'lifecycle', {}), 'process_alive': False,
                                      'weights_loaded': False, 'gpu_resident': False, 'ready': False,
                                      'release_confirmed': True}
                    if hasattr(self, 'hello'):
                        self.hello['lifecycle'] = self.lifecycle
                if release and response.get('released') is not True:
                    raise OutcomeUnknown('private worker release was not confirmed by its resource owner')
            except WorkerError as exc:
                if release:
                    # Cleanup may proceed in the guardian, but its completion
                    # has not been observed. Do not turn this into success.
                    self._release_error = OutcomeUnknown('worker release outcome unknown; artifacts must be retained')
                    self.lifecycle = {**getattr(self, 'lifecycle', {}), 'process_alive': None,
                                      'weights_loaded': None, 'gpu_resident': None, 'ready': False,
                                      'release_confirmed': False}
                    if hasattr(self, 'hello'):
                        self.hello['lifecycle'] = self.lifecycle
                    raise self._release_error from exc
            finally:
                self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
