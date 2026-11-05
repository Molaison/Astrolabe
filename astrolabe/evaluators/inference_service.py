"""Bounded, same-UID AF_UNIX RPC, adapted from GH's local call/serve.

Pickle is only for trusted local trajectory processes. Handlers are serialized;
response transport is not. No network listener, scheduler, or result cache.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from multiprocessing.connection import Client, Listener
import os
from pathlib import Path
import pickle
import socket
import stat
import struct
import threading
import time
import traceback


def mark(event, **fields):
    os.write(1, (json.dumps(dict(event=event, ts=time.time(), pid=os.getpid(),
                               **fields)) + "\n").encode())


def _same_uid(connection):
    with socket.fromfd(connection.fileno(), socket.AF_UNIX, socket.SOCK_STREAM) as peer:
        _, uid, _ = struct.unpack("3i", peer.getsockopt(
            socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")))
    if uid != os.getuid():
        raise PermissionError("Inference RPC requires the same local UID")


def call(socket_path, payload):
    path = Path(socket_path)
    for target, mode in ((path.parent, 0o700), (path, 0o600)):
        info = target.stat()
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != mode:
            raise PermissionError(f"Inference RPC requires owner-only {mode:o}: {target}")
    with Client(str(path), family="AF_UNIX") as connection:
        _same_uid(connection)
        connection.send_bytes(pickle.dumps(payload, protocol=5))
        response = pickle.loads(connection.recv_bytes())
    if not response["ok"]:
        raise RuntimeError(response["error"])
    return response["result"]


def serve(handler, socket_path, max_pending=4):
    """Serialize complete synchronous inference; drain before shutdown ack."""
    if max_pending < 1:
        raise ValueError("max_pending must be positive")
    path = Path(socket_path)
    if path.exists():
        raise FileExistsError(f"Inference socket already exists; recover its owner: {path}")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.parent.stat().st_uid != os.getuid():
        raise PermissionError(f"Inference socket directory must belong to this UID: {path.parent}")
    path.parent.chmod(0o700)
    gpu_lock = threading.Lock()
    slots = threading.BoundedSemaphore(max_pending)

    def execute(connection, payload, queued, request_id):
        fields = dict(request_id=request_id, identity=payload.get("identity"))
        try:
            try:
                with gpu_lock:
                    mark("gpu_start", queue_s=time.time() - queued, **fields)
                    ok = False
                    try:
                        result = handler(payload)
                        ok = True
                    finally:
                        mark("gpu_end", ok=ok, **fields)
                response = dict(ok=True, result=result)
            except Exception:
                response = dict(ok=False, error=traceback.format_exc())
            connection.send_bytes(pickle.dumps(response, protocol=5))
        finally:
            connection.close()
            slots.release()

    shutdown = None
    with Listener(str(path), family="AF_UNIX") as listener:
        path.chmod(0o600)
        try:
            with ThreadPoolExecutor(max_workers=max_pending) as executor:
                mark("service_ready", socket=str(path), max_pending=max_pending)
                request_id = 0
                while True:
                    slots.acquire()
                    connection = None
                    try:
                        connection = listener.accept()
                        _same_uid(connection)
                        payload = pickle.loads(connection.recv_bytes())
                        if payload.get("type") == "shutdown":
                            shutdown = connection
                            slots.release()
                            mark("shutdown_received")
                            break
                        request_id += 1
                        queued = time.time()
                        mark("request_received", request_id=request_id,
                             identity=payload.get("identity"))
                        executor.submit(execute, connection, payload, queued, request_id)
                    except BaseException:
                        if connection is not None:
                            connection.close()
                        slots.release()
                        raise
            mark("shutdown_drained")
            if shutdown is not None:
                shutdown.send_bytes(pickle.dumps(dict(ok=True, result=None), protocol=5))
        finally:
            if shutdown is not None:
                shutdown.close()
    mark("service_stopped", socket=str(path))


def serve_staged(stages_factory, socket_path, pipeline, max_pending=4):
    """GH's staged RPC with the existing same-UID and lifecycle protections.

    Each stage receives a private item. Its ``value`` becomes the RPC result.
    Model settings are validated by the adapter before any device dispatch.
    """
    if max_pending < 1:
        raise ValueError('max_pending must be positive')
    path = Path(socket_path)
    if path.exists():
        raise FileExistsError(f'Inference socket already exists; recover its owner: {path}')
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.parent.stat().st_uid != os.getuid():
        raise PermissionError('Inference socket directory must belong to this UID')
    path.parent.chmod(0o700)
    slots = threading.BoundedSemaphore(max_pending)

    def send(item, error):
        fields = dict(request_id=item['request_id'], identity=item['payload'].get('identity'))
        try:
            response = (dict(ok=True, result=item['value']) if error is None else
                        dict(ok=False, error=''.join(traceback.format_exception(
                            type(error), error, error.__traceback__))))
            mark('transport_start', **fields)
            item['connection'].send_bytes(pickle.dumps(response, protocol=5))
            mark('transport_end', ok=response['ok'], **fields)
        finally:
            item['connection'].close()
            slots.release()

    shutdown = None
    with Listener(str(path), family='AF_UNIX') as listener:
        path.chmod(0o600)
        mark('service_ready', socket=str(path), max_pending=max_pending, mode='staged')
        request_id = 0
        try:
            while True:
                slots.acquire()
                connection = None
                owns_slot = True
                try:
                    connection = listener.accept()
                    _same_uid(connection)
                    payload = pickle.loads(connection.recv_bytes())
                    if payload.get('type') == 'shutdown':
                        shutdown = connection
                        owns_slot = False
                        slots.release()
                        mark('shutdown_received')
                        break
                    request_id += 1
                    item = dict(connection=connection, payload=payload, queued=time.time(),
                                request_id=request_id, value=None)
                    fields = dict(request_id=request_id, identity=payload.get('identity'))
                    mark('request_received', **fields)
                    try:
                        stages = stages_factory(payload)
                    except Exception as error:
                        # Validation errors are request failures, not server death.
                        owns_slot = False
                        try:
                            send(item, error)
                        except (OSError, EOFError):
                            # Request is already closed/released by send's finally.
                            # A disconnected validation client cannot stop the owner.
                            pass
                        continue
                    identity = dict(request_identity=payload.get('identity'), request_id=request_id)
                    pipeline.submit(stages, item, identity=identity, finalize=send)
                    owns_slot = False
                except BaseException:
                    if owns_slot:
                        if connection is not None:
                            connection.close()
                        slots.release()
                    raise
        finally:
            pipeline.shutdown()  # Includes completion waits, materialization and transport.
        mark('shutdown_drained')
        if shutdown is not None:
            try:
                shutdown.send_bytes(pickle.dumps(dict(ok=True, result=None), protocol=5))
            finally:
                shutdown.close()
    mark('service_stopped', socket=str(path))
