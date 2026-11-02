"""Synchronous broker proxy retaining the DaemonClient API."""
import json
import os
import socket
import threading
import uuid

from .client import OutcomeUnknown, WorkerError, _encode


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
    def __init__(self, endpoint, token, command, identity, *, env, max_message_bytes, startup_timeout):
        self.endpoint, self.token, self.limit = endpoint, token, max_message_bytes
        self.epoch = uuid.uuid4().hex
        self._lock = threading.Lock()
        self._used = set()
        self._closed = False
        response = self._rpc({'op': 'register', 'command': command, 'identity': identity,
                              'env': worker_overrides(env), 'limit': max_message_bytes,
                              'startup_timeout': startup_timeout}, startup_timeout + 10, submitted=False)
        self.handle, self.hello = response['handle'], response['hello']

    def _rpc(self, request, timeout, *, submitted):
        request['token'] = self.token
        allocation = os.environ.get('ASTROLABE_ALLOCATED_CUDA_DEVICES')
        if allocation is None:
            # Compatibility for explicit broker clients living in the launcher.
            allocation = os.environ.get('CUDA_VISIBLE_DEVICES') or ''
        if os.environ.get('ASTROLABE_CPU_OWNER') == '1' and os.environ.get('CUDA_VISIBLE_DEVICES') != '':
            raise WorkerError('trajectory owner CUDA visibility is not hidden')
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

    def call(self, payload, *, timeout=None, request_id=None):
        if not isinstance(payload, dict):
            raise TypeError('payload must be a JSON object')
        _encode(payload, self.limit)
        if timeout is not None and timeout <= 0:
            raise ValueError('timeout must be positive')
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
                                  'request_id': rid, 'payload': payload, 'timeout': timeout},
                                 None if timeout is None else timeout + 5, submitted=True)
            result = response.get('result')
            if not isinstance(result, dict):
                raise OutcomeUnknown('invalid broker result; request was not replayed')
            return result

    def close(self):
        # Residency belongs to the broker session, not this handle.
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
