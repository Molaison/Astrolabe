"""Bounded, same-UID AF_UNIX RPC, adapted from GH's local call/serve.

Pickle is only for trusted local trajectory processes. Handlers are serialized;
response transport is not. No network listener, scheduler, or result cache.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
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


_OPAQUE = "__astrolabe_opaque_rpc__"


def _check_opaque_request(frame):
    if (type(frame) is not dict or set(frame) !=
            {_OPAQUE, "kind", "model", "operation", "identity", "root_request_id", "body"}
            or type(frame[_OPAQUE]) is not int or frame[_OPAQUE] != 1
            or type(frame["kind"]) is not str or frame["kind"] != "predict"
            or type(frame["body"]) is not bytes
            or (frame["model"] is not None and type(frame["model"]) is not str)
            or (frame["operation"] is not None and type(frame["operation"]) is not str)
            or not (frame["model"] == "alphafold3" or frame["operation"] == "extract_structures")
            or (frame["root_request_id"] is not None and
                (type(frame["root_request_id"]) is not int or frame["root_request_id"] < 1))):
        raise ValueError("Invalid opaque AF3 request envelope")
    identity = frame["identity"]
    if (identity is not None and
            (type(identity) is not dict or
             any(type(k) is not str or
                 (v is not None and type(v) not in (int, str))
                 for k, v in identity.items()))):
        raise ValueError("Invalid opaque AF3 identity")
    return frame


def _check_opaque_reply(frame):
    if (type(frame) is not dict or type(frame.get(_OPAQUE)) is not int or frame[_OPAQUE] != 1 or
            type(frame.get("ok")) is not bool or
            (frame["ok"] and (set(frame) != {_OPAQUE, "ok", "body"} or
                              type(frame["body"]) is not bytes)) or
            (not frame["ok"] and (set(frame) != {_OPAQUE, "ok", "error"} or
                                  type(frame["error"]) is not str))):
        raise ValueError("Invalid opaque AF3 reply envelope")
    return frame


def _opaque_wire(ok, value):
    return pickle.dumps({_OPAQUE: 1, "ok": ok, "body" if ok else "error": value}, protocol=5)


def _checked_path(socket_path):
    path = Path(socket_path)
    for target, mode in ((path.parent, 0o700), (path, 0o600)):
        info = target.stat()
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != mode:
            raise PermissionError(f"Inference RPC requires owner-only {mode:o}: {target}")
    return path


def _opaque_exchange(socket_path, frame):
    _check_opaque_request(frame)
    with Client(str(_checked_path(socket_path)), family="AF_UNIX") as connection:
        _same_uid(connection)
        connection.send_bytes(pickle.dumps(frame, protocol=5))
        wire = connection.recv_bytes()
    reply = _check_opaque_reply(pickle.loads(wire))
    if not reply["ok"]:
        raise RuntimeError(reply["error"])
    return wire


def call(socket_path, payload, *, opaque=False):
    if opaque:
        frame = {_OPAQUE: 1, "kind": "predict", "model": payload.get("model"),
                 "operation": payload.get("operation"), "identity": payload.get("identity"),
                 "root_request_id": None, "body": pickle.dumps(payload, protocol=5)}
        return pickle.loads(_check_opaque_reply(pickle.loads(_opaque_exchange(socket_path, frame)))["body"])
    path = _checked_path(socket_path)
    with Client(str(path), family="AF_UNIX") as connection:
        _same_uid(connection)
        connection.send_bytes(pickle.dumps(payload, protocol=5))
        response = pickle.loads(connection.recv_bytes())
    if not response["ok"]:
        raise RuntimeError(response["error"])
    return response["result"]


_JSON_LIMIT = 1024 * 1024  # Control frames only. Scientific bytes never belong here.


def _json_dump(value):
    wire = json.dumps(value, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(wire) > _JSON_LIMIT:
        raise ValueError("Inference control frame exceeds 1 MiB")
    return wire


def _json_load(wire):
    if len(wire) > _JSON_LIMIT:
        raise ValueError("Inference control frame exceeds 1 MiB")
    def invalid(value):
        raise ValueError(f"Invalid JSON constant: {value}")
    value = json.loads(wire, parse_constant=invalid)
    if type(value) is not dict:
        raise ValueError("Inference control frame must be a mapping")
    return value


def json_call(socket_path, payload):
    """Explicit control-only protocol. Never detects or falls back to pickle."""
    wire = _json_dump(payload)
    with Client(str(_checked_path(socket_path)), family="AF_UNIX") as connection:
        _same_uid(connection)
        connection.send_bytes(wire)
        response = _json_load(connection.recv_bytes(_JSON_LIMIT))
    if type(response.get("ok")) is not bool:
        raise ValueError("Invalid inference control response")
    if not response["ok"]:
        raise RuntimeError(response["error"])
    return response["result"]


def serve(handler, socket_path, max_pending=4, *, transport="pickle"):
    """Drain before shutdown ack; only qualified AF3 handlers bypass the lock."""
    if max_pending < 1:
        raise ValueError("max_pending must be positive")
    if transport not in {"pickle", "json"}:
        raise ValueError("Unknown inference transport")
    dumps = _json_dump if transport == "json" else lambda value: pickle.dumps(value, protocol=5)
    loads = _json_load if transport == "json" else pickle.loads
    path = Path(socket_path)
    if path.exists():
        raise FileExistsError(f"Inference socket already exists; recover its owner: {path}")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.parent.stat().st_uid != os.getuid():
        raise PermissionError(f"Inference socket directory must belong to this UID: {path.parent}")
    path.parent.chmod(0o700)
    gpu_lock = threading.Lock()
    request_local_rng = getattr(handler, "request_local_rng", False) is True
    slots = threading.BoundedSemaphore(max_pending)
    pending_lock = threading.Lock()
    pending = 0

    def execute(connection, payload, queued, request_id):
        nonlocal pending
        fields = dict(request_id=request_id, identity=payload.get("identity"))
        opaque = transport == "pickle" and _OPAQUE in payload
        if opaque and payload.get("root_request_id") is not None:
            fields["root_request_id"] = payload["root_request_id"]
        try:
            try:
                if opaque:
                    _check_opaque_request(payload)
                    if getattr(handler, "opaque_proxy", False):
                        payload = dict(payload, root_request_id=request_id)
                    else:
                        scientific = pickle.loads(payload["body"])
                        if type(scientific) is not dict:
                            raise ValueError("Opaque AF3 header/body mismatch")
                        header_identity = payload["identity"]
                        body_identity = scientific.get("identity")
                        identity_matches = (body_identity is None if header_identity is None else
                                            type(body_identity) is dict and
                                            body_identity.keys() == header_identity.keys() and
                                            all(type(body_identity[key]) is type(value) and
                                                body_identity[key] == value
                                                for key, value in header_identity.items()))
                        if (scientific.get("model") != payload["model"] or
                                scientific.get("operation") != payload["operation"] or
                                not identity_matches):
                            raise ValueError("Opaque AF3 header/body mismatch")
                        payload = dict(scientific, request_id=payload["root_request_id"] or request_id)
                with nullcontext() if request_local_rng else gpu_lock:
                    mark("evaluation_start" if request_local_rng else "gpu_start",
                         queue_s=time.time() - queued, **fields)
                    ok = False
                    try:
                        result = handler(payload if opaque else
                                         dict(payload, request_id=request_id) if request_local_rng else payload)
                        ok = True
                    finally:
                        mark("evaluation_end" if request_local_rng else "gpu_end", ok=ok, **fields)
                response = dict(ok=True, result=result)
            except Exception:
                response = dict(ok=False, error=traceback.format_exc())
            if opaque:
                wire = (result if response["ok"] and getattr(handler, "opaque_proxy", False)
                        else _opaque_wire(True, pickle.dumps(result, protocol=5)) if response["ok"]
                        else _opaque_wire(False, response["error"]))
            else:
                wire = dumps(response)
            connection.send_bytes(wire)
        finally:
            connection.close()
            with pending_lock:
                pending -= 1
            slots.release()

    def control(connection, payload):
        """The listener owns ingress; pending includes transport/finalization."""
        try:
            kind = payload["type"]
            with pending_lock:
                if kind == "describe_snapshot":
                    snapshot = handler.describe_snapshot()
                    snapshot.update(pending=pending, max_pending=max_pending)
                    result = snapshot
                else:
                    if pending:
                        raise RuntimeError("adaptive_control requires pending=0 and all stages drained")
                    if set(payload) - {"type", "enabled", "reset"}:
                        raise ValueError("Unknown adaptive_control fields")
                    result = handler.adaptive_control(
                        enabled=payload["enabled"], reset=payload.get("reset", False))
            response = dict(ok=True, result=result)
        except Exception:
            response = dict(ok=False, error=traceback.format_exc())
            mark("adaptive_control_rejected" if payload["type"] == "adaptive_control"
                 else "describe_snapshot_failed", error=response["error"])
        try:
            connection.send_bytes(dumps(response))
        except (OSError, EOFError):
            pass
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
                        if transport == "json":
                            try:
                                payload = loads(connection.recv_bytes(_JSON_LIMIT))
                            except (ValueError, UnicodeError, RecursionError, OSError, EOFError):
                                try:
                                    connection.send_bytes(_json_dump({
                                        "ok": False, "error": "Invalid inference control frame"}))
                                except (OSError, EOFError):
                                    pass
                                connection.close()
                                slots.release()
                                continue
                        else:
                            payload = loads(connection.recv_bytes())
                        if payload.get("type") in {"adaptive_control", "describe_snapshot"}:
                            control(connection, payload)
                            continue
                        if payload.get("type") == "shutdown":
                            shutdown = connection
                            slots.release()
                            mark("shutdown_received")
                            break
                        request_id += 1
                        queued = time.time()
                        mark("request_received", request_id=request_id,
                             identity=payload.get("identity"))
                        with pending_lock:
                            pending += 1
                        try:
                            executor.submit(execute, connection, payload, queued, request_id)
                        except BaseException:
                            with pending_lock:
                                pending -= 1
                            raise
                    except BaseException:
                        if connection is not None:
                            connection.close()
                        slots.release()
                        raise
            mark("shutdown_drained")
            close = getattr(handler, "close", None)
            if close is not None:
                close()
            if shutdown is not None:
                shutdown.send_bytes(dumps(dict(ok=True, result=None)))
        finally:
            close = getattr(handler, "close", None)
            if close is not None:
                close()
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
