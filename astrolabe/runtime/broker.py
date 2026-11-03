"""Private session broker; model residency and per-device compute are separate."""
import hashlib
import json
import os
import signal
import shutil
import socketserver
import sys
import threading
import time
import uuid
from .client import DaemonClient, OutcomeUnknown, WorkerError
from .admission import _execute_probe, allocated_devices

MAX_CONTROL = 4 * 1024 * 1024


from .broker_state import State


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            self.request.settimeout(30)
            raw = self.rfile.readline(MAX_CONTROL + 1)
            if len(raw) > MAX_CONTROL or not raw.endswith(b'\n'):
                raise ValueError('control message too large or incomplete')
            req = json.loads(raw)
            if req.get('token') != self.server.state.token:
                raise ValueError('unauthorized')
            if req.get('allocated_cuda_devices') != (os.environ.get('CUDA_VISIBLE_DEVICES') or ''):
                raise ValueError('request allocation differs from broker allocation')
            if req.get('op') == 'register':
                result = self.server.state.register(req)
            elif req.get('op') == 'call':
                result = self.server.state.call(req)
            elif req.get('op') == 'probe':
                result = {'result': self.server.state.probe(req)}
            elif req.get('op') == 'metrics':
                result = {'metrics': self.server.state.snapshot()}
            elif req.get('op') == 'detach':
                result = self.server.state.detach(req)
            elif req.get('op') == 'nested_begin':
                result = self.server.state.nested(req)
            elif req.get('op') == 'nested_end':
                result = self.server.state.nested(req, finish=True)
            elif req.get('op') == 'nested_process':
                result = self.server.state.nested_process(req)
            elif req.get('op') == 'stage':
                result = self.server.state.stage(req)
            elif req.get('op') == 'owner':
                result = self.server.state.register_owner(req)
            else:
                raise ValueError('unknown broker operation')
            result['ok'] = True
            if 'request_id' in req:
                result.update(request_id=req['request_id'], epoch=req.get('epoch'))
        except Exception as exc:
            result = {'ok': False, 'unknown': isinstance(exc, OutcomeUnknown),
                      'error': f'{type(exc).__name__}: {exc}'}
        try:
            self.wfile.write((json.dumps(result, allow_nan=False, separators=(',', ':')) + '\n').encode())
        except OSError:
            pass


class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = False
    block_on_close = True
    _slots = threading.BoundedSemaphore(32)

    def process_request(self, request, client_address):
        self._slots.acquire()
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()


def main():
    endpoint, token_fd, parent_fd, artifact_root = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    max_resident_models = int(sys.argv[5])
    limits = json.loads(sys.argv[7]) if len(sys.argv) > 7 else None
    with os.fdopen(token_fd, 'rb') as token_pipe:
        token = token_pipe.read(128).decode()
    if not token:
        raise ValueError('missing broker token')
    address = '\0' + endpoint[1:] if endpoint.startswith('@') else endpoint
    server = Server(address, Handler)
    if not endpoint.startswith('@'):
        os.chmod(endpoint, 0o600)
    os.environ['ASTROLABE_BROKER_ENDPOINT'] = endpoint
    server.state = State(token, max_resident_models, int(sys.argv[6]), limits=limits)
    monitor = threading.Thread(target=server.state.monitor, daemon=True)
    monitor.start()
    def watch():
        # EOF means the owning session exited. In-flight calls retain their
        # leases until complete or the underlying worker has been terminated.
        with os.fdopen(parent_fd, 'rb') as parent:
            while parent.read(1):
                pass
        with server.state.lock:
            server.state.stopping = True
            with server.state.ledger.condition:
                server.state.ledger.stopping = True
                server.state.ledger.condition.notify_all()
            # Interrupt active work, then let the handler wait for worker exit
            # while it still owns the compute lock. No request is replayed.
            for record in server.state.workers.values():
                process = record['client'].process
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
        server.shutdown()
    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        server.serve_forever(poll_interval=0.1)
    finally:
        server.server_close()
        server.state.close()
        if not endpoint.startswith('@'):
            try:
                os.unlink(endpoint)
            except FileNotFoundError:
                pass
        if not server.state.retain_artifacts:
            shutil.rmtree(artifact_root, ignore_errors=True)


if __name__ == '__main__':
    main()
