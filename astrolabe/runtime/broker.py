"""Private session broker with independent, bounded ingress bulkheads."""
import json
import math
import os
import select
import signal
import shutil
import socket
import socketserver
import sys
import threading
import time

from .client import OutcomeUnknown
from .broker_transport import (CONTROL_CAPACITY, LANE_OPS, MAX_CONTROL, READ_TIMEOUT,
                               WRITE_TIMEOUT, encode_response, endpoint_for, read_line,
                               rejection, unix_address)
from .broker_state import State


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        req = None
        parent = None
        try:
            raw = read_line(self.request, MAX_CONTROL, time.monotonic() + self.server.read_timeout)
            req = json.loads(raw)
            if not isinstance(req, dict):
                raise ValueError('broker request must be an object')
            if req.get('token') != self.server.state.token:
                raise ValueError('unauthorized')
            if req.get('allocated_cuda_devices') != (os.environ.get('CUDA_VISIBLE_DEVICES') or ''):
                raise ValueError('request allocation differs from broker allocation')
            op = req.get('op')
            if op not in LANE_OPS[self.server.lane]:
                result = rejection('operation is not admitted on this broker lane')
            else:
                deadline = req.get('deadline')
                if deadline is not None:
                    if not isinstance(deadline, (int, float)) or not math.isfinite(deadline):
                        raise ValueError('invalid broker request deadline')
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self._write(rejection('broker request deadline expired before dispatch'))
                        return
                    field = 'startup_timeout' if op == 'register' else 'timeout'
                    if op in ('register', 'call', 'probe'):
                        req[field] = min(req.get(field) or remaining, remaining)
                if self.server.lane == 'nested':
                    parent = self.server.claim_parent(req)
                    if parent is None:
                        self._write(rejection('parent already has a nested RPC in flight'))
                        return
                if op == 'register':
                    result = self.server.state.register(req)
                elif op == 'call':
                    result = self.server.state.call(req)
                elif op == 'probe':
                    result = {'result': self.server.state.probe(req)}
                elif op == 'metrics':
                    snapshot = getattr(self.server.state, 'snapshot_for_request', None)
                    metrics = snapshot(req) if snapshot is not None else self.server.state.snapshot()
                    metrics['ingress'] = {name: server.snapshot() for name, server in self.server.lanes.items()}
                    result = {'metrics': metrics}
                elif op == 'detach':
                    result = self.server.state.detach(req)
                elif op == 'cancel':
                    result = self.server.state.cancel(req)
                elif op in ('nested_begin', 'nested_end'):
                    result = self.server.state.nested(req, finish=op == 'nested_end')
                elif op == 'nested_process':
                    result = self.server.state.nested_process(req)
                elif op == 'stage':
                    result = self.server.state.stage(req)
                elif op == 'owner':
                    result = self.server.state.register_owner(req)
                result['ok'] = True
                if 'request_id' in req:
                    result.update(request_id=req['request_id'], epoch=req.get('epoch'))
        except Exception as exc:
            unknown = isinstance(exc, OutcomeUnknown)
            result = {'ok': False, 'unknown': unknown, 'error_type': type(exc).__name__,
                      'error': f'{type(exc).__name__}: {exc}'}
            for name in ('execution_state', 'retryable'):
                if hasattr(exc, name):
                    result[name] = getattr(exc, name)
            if req is None:
                result.update(execution_state='not_executed', retryable=True)
        try:
            self._write(result)
        finally:
            if parent is not None:
                self.server.release_parent(parent)

    def _write(self, result):
        try:
            self.request.settimeout(WRITE_TIMEOUT)
            self.request.sendall(encode_response(result))
        except OSError:
            pass


class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = False
    block_on_close = True

    def __init__(self, address, handler=Handler, *, capacity=32, lane='business', state=None,
                 read_timeout=READ_TIMEOUT):
        if type(capacity) is not int or capacity < 1 or lane not in LANE_OPS:
            raise ValueError('invalid broker lane capacity')
        self.capacity, self.lane, self.state = capacity, lane, state
        self.read_timeout = read_timeout
        self._ingress_lock = threading.Lock()
        self._active = set()
        self._parents = set()
        self._in_flight = self._high_water = self._rejected = 0
        self._accepting = True
        self.lanes = {lane: self}
        self.request_queue_size = capacity
        super().__init__(address, handler)

    def snapshot(self):
        with self._ingress_lock:
            return dict(capacity=self.capacity, in_flight=self._in_flight,
                        high_water=self._high_water, rejected=self._rejected)

    def process_request(self, request, client_address):
        with self._ingress_lock:
            admitted = self._accepting and self._in_flight < self.capacity
            if admitted:
                self._in_flight += 1
                self._high_water = max(self._high_water, self._in_flight)
                self._active.add(request)
            else:
                self._rejected += 1
        if not admitted:
            # A fresh local socket can take this small frame immediately. Never
            # wait for a slot or a reader in the accept loop, or spawn a reject thread.
            try:
                request.setblocking(False)
                request.send(encode_response(rejection(self.lane + ' ingress capacity exhausted')))
                request.shutdown(socket.SHUT_WR)
            except OSError:
                pass
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._finished(request)
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._finished(request)

    def _finished(self, request):
        with self._ingress_lock:
            self._active.discard(request)
            self._in_flight -= 1

    def claim_parent(self, req):
        parent = (req.get('owner_pid'), req.get('worker_epoch'), req.get('parent_request_id'))
        if (type(parent[0]) is not int or
                any(not isinstance(value, str) or not value for value in parent[1:])):
            raise ValueError('invalid nested parent identity')
        with self._ingress_lock:
            if parent in self._parents:
                self._rejected += 1
                return None
            self._parents.add(parent)
        return parent

    def release_parent(self, parent):
        with self._ingress_lock:
            self._parents.discard(parent)

    def interrupt(self):
        with self._ingress_lock:
            self._accepting = False
            active = list(self._active)
        for request in active:
            try:
                request.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def _signal_groups(anchors, sig):
    """Signal only groups whose observed leader identity is still anchored."""
    from .telemetry import process_identity
    for pid, start in anchors.items():
        # An exited leader is never proof that its old numeric PGID is safe.
        # Guardian/ledger cleanup owns descendants after the anchor disappears.
        if process_identity(pid) != (pid, start):
            continue
        try:
            if os.getpgid(pid) == pid and process_identity(pid) == (pid, start):
                os.killpg(pid, sig)
        except ProcessLookupError:
            pass


def _stop_state(state):
    # Fence the resource owner before joining handlers that may await admission.
    with state.lock:
        state.stopping = True
        for request in getattr(state, 'requests', {}).values():
            request['cancel'].cancel()
        if hasattr(state, 'condition'):
            state.condition.notify_all()
    with state.ledger.condition:
        state.ledger.stopping = True
        state.ledger.condition.notify_all()
        anchors = {pid: start for record in state.ledger.residents.values()
                   for pid, start in record.get('groups', {}).items()}
    _signal_groups(anchors, signal.SIGTERM)
    # A worker that ignores TERM must not hold session exit indefinitely.
    # Every signal independently validates its original process identity.
    if anchors:
        threading.Event().wait(.2)
        _signal_groups(anchors, signal.SIGKILL)


def main():
    endpoint, token_fd, parent_fd, artifact_root = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    limits = json.loads(sys.argv[7]) if len(sys.argv) > 7 else None
    with os.fdopen(token_fd, 'rb') as token_pipe:
        token = token_pipe.read(128).decode()
    if not token:
        raise ValueError('missing broker token')
    os.environ['ASTROLABE_BROKER_ENDPOINT'] = endpoint
    state = State(token, int(sys.argv[5]), int(sys.argv[6]), limits=limits)
    from . import admission
    admission._broker_stopping = state.closed
    servers, threads, paths = {}, [], []
    monitor = threading.Thread(target=state.monitor, name='broker-monitor')
    try:
        # All accepted managed parents occupy a business slot. A separate
        # equally sized nested lane accommodates one RPC per parent even when
        # every business handler is waiting for its managed child.
        pending = state.ledger.limits.pending_items
        for lane, op, capacity in (('business', 'call', pending), ('control', 'metrics', CONTROL_CAPACITY),
                                   ('nested', 'nested_begin', pending)):
            target = endpoint_for(endpoint, op)
            server = Server(unix_address(target), capacity=capacity, lane=lane, state=state)
            servers[lane] = server
            if not target.startswith('@'):
                os.chmod(target, 0o600)
                paths.append(target)
        for server in servers.values():
            server.lanes = servers
            thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .05},
                                      name='broker-' + server.lane)
            thread.start()
            threads.append(thread)
        monitor.start()
        with os.fdopen(parent_fd, 'rb', buffering=0) as parent:
            while True:
                ready, _, _ = select.select([parent], [], [], .1)
                if ready and not parent.read(1):
                    break
    finally:
        state.closed.set()
        _stop_state(state)
        for server in servers.values():
            server.interrupt()
        for server, thread in zip(servers.values(), threads):
            server.shutdown()
            thread.join()
        for server in servers.values():
            server.server_close()
        if monitor.ident is not None:
            monitor.join()
        try:
            state.close()
        finally:
            for path in paths:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
        if not state.retain_artifacts:
            shutil.rmtree(artifact_root, ignore_errors=True)


if __name__ == '__main__':
    main()
