"""Same-UID persistent model owner with bounded pickle5 request transport."""

from __future__ import annotations

import argparse
import atexit
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
import hashlib
import importlib
import json
import logging
import math
import os
from pathlib import Path
import pickle
import queue
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from typing import Any
import uuid


PROTOCOL = "astrolabe-model-daemon/1"
MAX_FRAME_BYTES = 8 * 1024 * 1024
_REPO_ROOT = Path(__file__).resolve().parents[1]


class ModelDaemonError(RuntimeError):
    pass


class DaemonStartupError(ModelDaemonError):
    pass


class DaemonChildExited(ModelDaemonError):
    pass


class DaemonDisconnected(ModelDaemonError):
    pass


class IdentityMismatch(ModelDaemonError):
    pass


class DaemonRequestFailed(ModelDaemonError):
    def __init__(self, message: str, *, remote_type: str | None = None) -> None:
        super().__init__(message)
        self.remote_type = remote_type


class RequestOutcomeUnknown(ModelDaemonError):
    outcome_unknown = True

    def __init__(
        self,
        message: str,
        *,
        request_id: str | None = None,
        generation: str | None = None,
    ) -> None:
        super().__init__(message)
        self.request_id = request_id
        self.generation = generation


class FrameTooLarge(ModelDaemonError):
    pass


class ProtocolError(ModelDaemonError):
    pass


@dataclass(slots=True)
class DaemonSpec:
    backend: str
    factory: str
    settings: dict[str, Any]
    python_bin: str = sys.executable
    cwd: str | None = None
    env: dict[str, str] | None = None
    provider: dict[str, Any] | None = None
    max_frame_bytes: int = MAX_FRAME_BYTES


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    raise TypeError(f"Daemon settings must be JSON-like, got {type(value).__name__}")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _module_file(module_name: str, spec: DaemonSpec) -> Path | None:
    candidates = [_REPO_ROOT]
    if spec.cwd:
        candidates.append(Path(spec.cwd).expanduser())
    pythonpath = (spec.env or {}).get("PYTHONPATH", "")
    candidates.extend(Path(item) for item in pythonpath.split(os.pathsep) if item)
    parts = module_name.split(".")
    for root in candidates:
        root = root.expanduser().resolve()
        module_file = root.joinpath(*parts).with_suffix(".py")
        package_file = root.joinpath(*parts, "__init__.py")
        if module_file.is_file():
            return module_file
        if package_file.is_file():
            return package_file
    return None


def _adapter_hash(spec: DaemonSpec) -> str:
    module_name, separator, _ = spec.factory.partition(":")
    if not separator:
        module_name, _, _ = spec.factory.rpartition(".")
    module_file = _module_file(module_name, spec)
    digest = hashlib.sha256()
    if module_file is None:
        digest.update(spec.factory.encode("utf-8"))
    else:
        with module_file.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def model_identity(spec: DaemonSpec, *, max_pending: int = 4) -> str:
    identity_settings = dict(spec.settings)
    identity_settings.pop("_initial_random_state", None)
    payload = {
        "protocol": PROTOCOL,
        "backend": spec.backend,
        "factory": spec.factory,
        "settings": identity_settings,
        "python": str(spec.python_bin),
        "cwd": str(Path(spec.cwd).expanduser().resolve()) if spec.cwd else None,
        "env": dict(sorted((spec.env or {}).items())),
        "provider": spec.provider,
        "adapter_sha256": _adapter_hash(spec),
        "max_pending": int(max_pending),
        "max_frame_bytes": int(spec.max_frame_bytes),
    }
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _provider_selected(spec: DaemonSpec) -> bool:
    environment = os.environ.copy()
    environment.update(spec.env or {})
    return spec.provider is not None or bool(environment.get("ASTROLABE_MODEL_PROVIDERS"))


def _prepare_provider_spec(spec: DaemonSpec) -> DaemonSpec:
    if not _provider_selected(spec):
        return spec
    from astrolabe.provider_runtime import prepare_provider_spec

    prepared = prepare_provider_spec(spec)
    if not isinstance(prepared, DaemonSpec):
        raise TypeError("prepare_provider_spec must return DaemonSpec")
    return prepared


def _activate_child_provider(spec: DaemonSpec) -> dict[str, Any] | None:
    if not _provider_selected(spec):
        return None
    from astrolabe.provider_runtime import activate_child_provider

    report = activate_child_provider(spec)
    return dict(report) if isinstance(report, Mapping) else None


def _provider_environment_for_spec(spec: DaemonSpec) -> dict[str, str]:
    value = (spec.env or {}).get("ASTROLABE_MODEL_PROVIDERS")
    return {"ASTROLABE_MODEL_PROVIDERS": str(value)} if value is not None else {}


def _signal_owned_group(proc: subprocess.Popen[bytes], sig: int) -> None:
    """Signal only the private session created for this owned child."""
    if hasattr(os, "killpg"):
        if proc.pid == os.getpgrp():
            raise ModelDaemonError("refusing to signal the coordinator process group")
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass
    elif proc.poll() is None:
        proc.send_signal(sig)


def _load_factory(path: str) -> Callable[..., Any]:
    if ":" in path:
        module_name, attr_path = path.split(":", 1)
    else:
        module_name, _, attr_path = path.rpartition(".")
    if not module_name or not attr_path:
        raise ValueError("factory must be 'module:callable'")
    value: Any = importlib.import_module(module_name)
    for part in attr_path.split("."):
        value = getattr(value, part)
    if not callable(value):
        raise TypeError(f"factory is not callable: {path}")
    return value


def _handler_from(value: Any) -> Callable[[dict[str, Any]], Any]:
    if callable(value):
        return value
    infer = getattr(value, "infer", None)
    if callable(infer):
        return infer
    raise TypeError("factory must return a callable or an object with infer(payload)")


def _frame(payload: Mapping[str, Any], max_frame_bytes: int = MAX_FRAME_BYTES) -> bytes:
    body = pickle.dumps(dict(payload), protocol=5)
    if not body or len(body) > max_frame_bytes:
        raise FrameTooLarge(f"frame size {len(body)} exceeds bound {max_frame_bytes}")
    return len(body).to_bytes(8, "big") + body


def _receive_exact(connection: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise DaemonDisconnected("connection closed before frame completed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_frame(
    connection: socket.socket,
    max_frame_bytes: int = MAX_FRAME_BYTES,
) -> dict[str, Any]:
    header = _receive_exact(connection, 8)
    size = int.from_bytes(header, "big")
    if size <= 0 or size > max_frame_bytes:
        raise FrameTooLarge(f"frame size {size} exceeds bound {max_frame_bytes}")
    try:
        value = pickle.loads(_receive_exact(connection, size))
    except (pickle.PickleError, EOFError, AttributeError, ImportError, ValueError) as error:
        raise ProtocolError("invalid pickle frame") from error
    if not isinstance(value, dict):
        raise ProtocolError("frame must contain a JSON object")
    return value


def _send_frame(
    connection: socket.socket,
    payload: Mapping[str, Any],
    max_frame_bytes: int = MAX_FRAME_BYTES,
) -> None:
    connection.sendall(_frame(payload, max_frame_bytes))


def control_request(
    socket_path: str | Path,
    identity: str,
    command: str = "status",
    timeout: float = 10,
    max_frame_bytes: int = MAX_FRAME_BYTES,
) -> dict[str, Any]:
    """Send one management request without loading or owning a model child."""
    if command not in {"status", "stop"}:
        raise ValueError("control command must be 'status' or 'stop'")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("control timeout must be finite and positive")
    if max_frame_bytes <= 0:
        raise ValueError("max_frame_bytes must be positive")
    endpoint = Path(socket_path).expanduser().resolve()
    request_id = uuid.uuid4().hex
    message = {
        "protocol": PROTOCOL,
        "identity": identity,
        "request_id": request_id,
        "generation": "",
        "type": command,
    }
    frame = _frame(message, max_frame_bytes)
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sent = False
    try:
        _validate_endpoint(endpoint)
        connection.settimeout(timeout)
        connection.connect(str(endpoint))
        if _peer_uid(connection) != os.getuid():
            raise PermissionError("daemon endpoint requires same UID")
        connection.sendall(frame)
        sent = True
        response = _read_frame(connection, max_frame_bytes)
    except (socket.timeout, TimeoutError) as error:
        if sent:
            raise RequestOutcomeUnknown(
                f"control request timed out after send; outcome unknown; command={command}",
                request_id=request_id,
            ) from error
        raise ModelDaemonError("control connection timed out") from error
    except (ConnectionError, FileNotFoundError, DaemonDisconnected) as error:
        raise DaemonDisconnected("daemon disconnected during control request") from error
    finally:
        connection.close()
    if response.get("protocol") != PROTOCOL:
        raise ProtocolError("daemon protocol mismatch")
    if response.get("request_id") != request_id:
        raise ProtocolError("daemon response request id mismatch")
    if response.get("identity") != identity:
        raise IdentityMismatch("daemon identity handshake failed")
    if response.get("type") == "error":
        if response.get("code") == "identity_mismatch":
            raise IdentityMismatch("daemon identity handshake failed")
        raise DaemonRequestFailed(
            str(response.get("message") or response.get("code") or "control request failed"),
            remote_type=str(response.get("error_type") or response.get("code") or "") or None,
        )
    return response


def _peer_uid(connection: socket.socket) -> int:
    if not hasattr(socket, "SO_PEERCRED"):
        return os.getuid()
    credentials = connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
    _, uid, _ = __import__("struct").unpack("3i", credentials)
    return int(uid)


def _validate_endpoint(path: Path) -> None:
    parent = path.parent.stat()
    endpoint = path.stat()
    if parent.st_uid != os.getuid() or stat.S_IMODE(parent.st_mode) != 0o700:
        raise PermissionError(f"daemon directory must be owner-only 0700: {path.parent}")
    if endpoint.st_uid != os.getuid() or stat.S_IMODE(endpoint.st_mode) != 0o600:
        raise PermissionError(f"daemon socket must be owner-only 0600: {path}")
    if not stat.S_ISSOCK(endpoint.st_mode):
        raise ProtocolError(f"daemon endpoint is not a socket: {path}")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _event(event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "protocol": PROTOCOL,
        "pid": os.getpid(),
        "time": time.time(),
        **fields,
    }
    sys.stdout.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
    sys.stdout.flush()


class _Server:
    def __init__(
        self,
        handler: Callable[[dict[str, Any]], Any],
        *,
        socket_path: Path,
        identity: str,
        backend: str,
        max_pending: int,
        max_frame_bytes: int,
        parent_pid: int | None,
    ) -> None:
        self.handler = handler
        self.socket_path = socket_path
        self.identity = identity
        self.backend = backend
        self.max_pending = max_pending
        self.max_frame_bytes = max_frame_bytes
        self.parent_pid = parent_pid
        self.started_at = time.time()
        self.requests = 0
        self.completed = 0
        self.failed = 0
        self.active_request: str | None = None
        self.tasks: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=max_pending)
        self.admission = threading.Lock()
        self.stopping = False
        self.listener: socket.socket | None = None
        self.shutdown_complete = threading.Event()
        self.worker = threading.Thread(target=self._worker, name="model-daemon-inference")

    def _status(self) -> dict[str, Any]:
        return {
            "protocol": PROTOCOL,
            "type": "status",
            "backend": self.backend,
            "identity": self.identity,
            "pid": os.getpid(),
            "parent_pid": self.parent_pid,
            "state": "draining" if self.stopping else "running",
            "queue_depth": self.tasks.qsize(),
            "active_request": self.active_request,
            "max_pending": self.max_pending,
            "request_count": self.requests,
            "completed_count": self.completed,
            "failed_count": self.failed,
            "model_load_count": 1,
            "started_at": self.started_at,
            "socket": str(self.socket_path),
        }

    def _worker(self) -> None:
        while True:
            item = self.tasks.get()
            try:
                if item.get("stop"):
                    return
                request_id = item["request_id"]
                generation = item["generation"]
                self.active_request = request_id
                _event("inference_start", backend=self.backend, request_id=request_id)
                try:
                    result = self.handler(dict(item["payload"]))
                    response = {"type": "result", "result": result}
                    self.completed += 1
                except BaseException as error:
                    self.failed += 1
                    response = {
                        "type": "error",
                        "code": "inference_failed",
                        "error_type": type(error).__name__,
                        "message": str(error),
                    }
                    print(traceback.format_exc(), file=sys.stderr, flush=True)
                try:
                    _frame(response, self.max_frame_bytes)
                except (TypeError, ValueError, FrameTooLarge) as error:
                    response = {
                        "type": "error",
                        "code": "response_not_materializable",
                        "error_type": type(error).__name__,
                        "message": str(error),
                    }
                item["response"] = {
                    "protocol": PROTOCOL,
                    "request_id": request_id,
                    "identity": self.identity,
                    "generation": generation,
                    "backend": self.backend,
                    "pid": os.getpid(),
                    **response,
                }
                _event(
                    "inference_end",
                    backend=self.backend,
                    request_id=request_id,
                    ok=response["type"] == "result",
                )
            finally:
                self.active_request = None
                item["done"].set()
                if item.get("stop"):
                    self.tasks.task_done()

    def _handle(self, connection: socket.socket) -> None:
        request_id: str | None = None
        generation: str | None = None
        admitted_item: dict[str, Any] | None = None
        try:
            _check_child_parent(self.parent_pid)
            if _peer_uid(connection) != os.getuid():
                raise PermissionError("daemon requires same UID")
            request = _read_frame(connection, self.max_frame_bytes)
            request_id = str(request.get("request_id") or uuid.uuid4().hex)
            generation = str(request.get("generation") or "")
            if request.get("protocol") != PROTOCOL:
                raise ProtocolError("daemon protocol mismatch")
            if request.get("identity") != self.identity:
                _send_frame(
                    connection,
                    {
                        "protocol": PROTOCOL,
                        "type": "error",
                        "code": "identity_mismatch",
                        "request_id": request_id,
                        "generation": generation,
                        "identity": self.identity,
                        "backend": self.backend,
                        "pid": os.getpid(),
                    },
                    self.max_frame_bytes,
                )
                return
            request_type = request.get("type")
            if request_type == "status":
                _send_frame(
                    connection,
                    {
                        **self._status(),
                        "request_id": request_id,
                        "generation": generation,
                    },
                    self.max_frame_bytes,
                )
                return
            if request_type in {"stop", "shutdown"}:
                self._drain(connection, request_id, generation)
                return
            if request_type != "infer":
                raise ProtocolError(f"unknown request type: {request_type}")

            payload = request.get("payload")
            if not isinstance(payload, dict):
                raise ProtocolError("infer payload must be a JSON object")
            item = {
                "request_id": request_id,
                "generation": generation,
                "payload": payload,
                "done": threading.Event(),
                "response": None,
            }
            with self.admission:
                if self.stopping:
                    raise ProtocolError("daemon is stopping")
                self.requests += 1
                try:
                    self.tasks.put_nowait(item)
                except queue.Full as error:
                    raise ModelDaemonError("daemon pending queue is full") from error
                admitted_item = item
            item["done"].wait()
            _send_frame(connection, item["response"], self.max_frame_bytes)
        except (BrokenPipeError, ConnectionResetError, DaemonDisconnected):
            return
        except BaseException as error:
            try:
                _send_frame(
                    connection,
                    {
                        "protocol": PROTOCOL,
                        "type": "error",
                        "code": "protocol_error",
                        "request_id": request_id,
                        "generation": generation,
                        "identity": self.identity,
                        "backend": self.backend,
                        "error_type": type(error).__name__,
                        "message": str(error),
                        "pid": os.getpid(),
                    },
                    self.max_frame_bytes,
                )
            except OSError:
                pass
        finally:
            try:
                connection.close()
            finally:
                if admitted_item is not None:
                    self.tasks.task_done()

    def _drain(self, connection: socket.socket, request_id: str, generation: str) -> None:
        with self.admission:
            self.stopping = True
        _event("shutdown_received", backend=self.backend, request_id=request_id)
        self.tasks.join()
        self.tasks.put({"stop": True, "done": threading.Event()})
        self.worker.join()
        try:
            _send_frame(
                connection,
                {
                    "protocol": PROTOCOL,
                    "type": "result",
                    "request_id": request_id,
                    "generation": generation,
                    "identity": self.identity,
                    "backend": self.backend,
                    "pid": os.getpid(),
                    "result": {
                        "drained": True,
                        "completed_count": self.completed,
                        "failed_count": self.failed,
                    },
                },
                self.max_frame_bytes,
            )
        except (OSError, DaemonDisconnected):
            pass  # A peer may disconnect after requesting an otherwise clean stop.
        finally:
            self.shutdown_complete.set()

    def run(self) -> None:
        parent = self.socket_path.parent
        parent.mkdir(mode=0o700, parents=False, exist_ok=True)
        parent.chmod(0o700)
        if parent.stat().st_uid != os.getuid():
            raise PermissionError(f"daemon directory has wrong owner: {parent}")
        if self.socket_path.exists():
            raise FileExistsError(f"daemon socket already exists: {self.socket_path}")
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener = listener
        listener.bind(str(self.socket_path))
        self.socket_path.chmod(0o600)
        listener.listen(max(16, self.max_pending * 4))
        listener.settimeout(0.2)
        self.worker.start()
        _event(
            "service_ready",
            backend=self.backend,
            identity=self.identity,
            socket=str(self.socket_path),
            max_pending=self.max_pending,
            model_load_count=1,
        )
        try:
            while not self.stopping:
                try:
                    connection, _ = listener.accept()
                except socket.timeout:
                    continue
                except OSError:
                    if self.stopping:
                        break
                    raise
                threading.Thread(
                    target=self._handle,
                    args=(connection,),
                    name="model-daemon-request",
                    daemon=True,
                ).start()
        finally:
            try:
                listener.close()
            except OSError:
                pass
            if self.stopping:
                self.shutdown_complete.wait()
            try:
                self.socket_path.unlink()
            except FileNotFoundError:
                pass
        _event("service_stopped", backend=self.backend, identity=self.identity)


def _set_parent_death_signal(parent_pid: int | None) -> None:
    if parent_pid is None or not sys.platform.startswith("linux"):
        return
    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")
    except Exception as error:
        raise RuntimeError("Linux parent-death protection is unavailable") from error
    if os.getppid() != parent_pid:
        raise RuntimeError("daemon parent ended during startup")


def _check_child_parent(parent_pid: int | None) -> None:
    if parent_pid is not None and os.getppid() != parent_pid:
        raise RuntimeError("daemon parent is no longer alive")


def serve(
    factory: Callable[[dict[str, Any]], Any],
    settings: dict[str, Any],
    *,
    socket_path: str | Path,
    identity: str = "local",
    backend: str = "local",
    max_pending: int = 4,
    max_frame_bytes: int = MAX_FRAME_BYTES,
    parent_pid: int | None = None,
) -> None:
    if max_pending < 1:
        raise ValueError("max_pending must be positive")
    if max_frame_bytes <= 0:
        raise ValueError("max_frame_bytes must be positive")
    _event("model_load_start", backend=backend)
    handler = _handler_from(factory(settings))
    _event("model_load_end", backend=backend, model_load_count=1)
    _Server(
        handler,
        socket_path=Path(socket_path),
        identity=identity,
        backend=backend,
        max_pending=max_pending,
        max_frame_bytes=max_frame_bytes,
        parent_pid=parent_pid,
    ).run()


class OwnedModelDaemon:
    def __init__(
        self,
        spec: DaemonSpec,
        *,
        startup_timeout: float = 300,
        request_timeout: float | None = None,
        shutdown_timeout: float = 30,
        max_pending: int = 4,
        max_frame_bytes: int | None = None,
        socket_path: str | Path | None = None,
    ) -> None:
        if max_pending < 1:
            raise ValueError("max_pending must be positive")
        for name, value in (
            ("startup_timeout", startup_timeout),
            ("shutdown_timeout", shutdown_timeout),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if request_timeout is not None and (not math.isfinite(request_timeout) or request_timeout <= 0):
            raise ValueError("request_timeout must be finite and positive when set")
        if max_frame_bytes is not None and max_frame_bytes <= 0:
            raise ValueError("max_frame_bytes must be positive")
        self.spec = _prepare_provider_spec(spec)
        self.startup_timeout = float(startup_timeout)
        self.request_timeout = request_timeout
        self.shutdown_timeout = float(shutdown_timeout)
        self.max_pending = int(max_pending)
        self.max_frame_bytes = int(
            self.spec.max_frame_bytes if max_frame_bytes is None else max_frame_bytes
        )
        self.identity = model_identity(self.spec, max_pending=self.max_pending)
        self._attached = socket_path is not None
        self._socket_path = Path(socket_path).expanduser().resolve() if socket_path else None
        self._proc: subprocess.Popen[bytes] | None = None
        self._log_handle = None
        self._start_lock = threading.Lock()
        self._shutdown_lock = threading.Lock()
        self._publication_lock = threading.Lock()
        self._process_stop_lock = threading.Lock()
        self._cancelled = threading.Event()
        self.cleanup_errors: list[BaseException] = []
        self._startup_deadline: float | None = None
        self._ready = False
        self._remote_pid: int | None = None
        self.generation = uuid.uuid4().hex
        self._closed = False
        self.temp_dir: Path | None = None
        self.log_path: Path | None = None
        if not self._attached:
            atexit.register(self._atexit)
            from astrolabe.model_execution import register_model_owner

            register_model_owner(self)

    @property
    def pid(self) -> int | None:
        if self._proc is not None:
            return self._proc.pid
        return self._remote_pid

    @property
    def socket_path(self) -> Path | None:
        return self._socket_path

    @property
    def attached(self) -> bool:
        return self._attached

    def _child_env(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment.update({str(key): str(value) for key, value in (self.spec.env or {}).items()})
        inherited = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            str(_REPO_ROOT) if not inherited else f"{_REPO_ROOT}{os.pathsep}{inherited}"
        )
        environment["PYTHONUNBUFFERED"] = "1"
        environment["ASTROLABE_MODEL_PARENT_PID"] = str(os.getpid())
        return environment

    def _start(self) -> None:
        if self._attached:
            return
        with self._start_lock:
            if self._closed or self._cancelled.is_set():
                raise ModelDaemonError("daemon owner is closed")
            if self._startup_deadline is None:
                self._startup_deadline = time.monotonic() + self.startup_timeout
            if time.monotonic() >= self._startup_deadline:
                self._startup_failed("startup deadline expired")
            proc = self._proc
            if proc is not None and proc.poll() is None:
                return
            if proc is not None:
                raise DaemonChildExited(
                    f"{self.spec.backend} child exited with code {proc.returncode}; "
                    f"log: {self.log_path}"
                )
            self.temp_dir = Path(tempfile.mkdtemp(prefix="astrolabe-model-daemon-")).resolve()
            self.temp_dir.chmod(0o700)
            self._socket_path = self.temp_dir / "model.sock"
            self.log_path = self.temp_dir / "owner.jsonl"
            spec_path = self.temp_dir / "spec.json"
            _atomic_json(
                spec_path,
                {
                    "protocol": PROTOCOL,
                    "backend": self.spec.backend,
                    "factory": self.spec.factory,
                    "settings": self.spec.settings,
                    "python_bin": self.spec.python_bin,
                    "cwd": self.spec.cwd,
                    "env": _provider_environment_for_spec(self.spec),
                    "provider": self.spec.provider,
                    "identity": self.identity,
                    "generation": self.generation,
                    "max_pending": self.max_pending,
                    "max_frame_bytes": self.spec.max_frame_bytes,
                },
            )
            spec_path.chmod(0o600)
            self._log_handle = self.log_path.open("ab", buffering=0)
            command = [
                self.spec.python_bin,
                "-u",
                "-m",
                "astrolabe.model_daemon",
                "child",
                "--spec",
                str(spec_path),
                "--socket",
                str(self._socket_path),
            ]
            launched = threading.Event()
            launch_error: list[BaseException] = []
            environment = self._child_env()

            def launch_child() -> None:
                proc: subprocess.Popen[bytes] | None = None
                try:
                    from astrolabe.model_execution import execution_directory

                    proc = subprocess.Popen(
                        command,
                        cwd=execution_directory(self.spec.cwd),
                        env=environment,
                        stdin=subprocess.DEVNULL,
                        stdout=self._log_handle,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        close_fds=True,
                    )
                    with self._publication_lock:
                        if self._cancelled.is_set():
                            _signal_owned_group(proc, signal.SIGTERM)
                            try:
                                proc.wait(timeout=2.0)
                            except subprocess.TimeoutExpired:
                                _signal_owned_group(proc, signal.SIGKILL)
                                proc.wait()
                            _signal_owned_group(proc, signal.SIGKILL)
                            _atomic_json(self.temp_dir / "lifecycle.json", {
                                "status": "stopped", "pid": proc.pid,
                                "exit_code": proc.returncode, "stopped_at": time.time(),
                            })
                            self._remove_owned_socket()
                            self._close_log()
                        else:
                            self._proc = proc
                except BaseException as error:
                    launch_error.append(error)
                    if self._cancelled.is_set() and proc is not None:
                        self.cleanup_errors.append(error)
                        logging.exception("Failed to reap late model daemon launch")
                        try:
                            if proc.poll() is None:
                                _signal_owned_group(proc, signal.SIGKILL)
                            proc.wait(timeout=2.0)
                        except BaseException as recovery_error:
                            self.cleanup_errors.append(recovery_error)
                            logging.exception("Late model daemon launch recovery failed")
                finally:
                    launched.set()
                    if proc is None or (self._cancelled.is_set() and self._proc is None):
                        with self._publication_lock:
                            self._close_log()
                if proc is not None:
                    proc.wait()

            self._launch_thread = threading.Thread(
                target=launch_child,
                name=f"model-daemon-owner-{self.spec.backend}",
                daemon=True,
            )
            self._launch_thread.start()
            while not launched.is_set():
                if self._cancelled.is_set():
                    self._startup_failed("child launch cancelled")
                remaining = self._startup_deadline - time.monotonic()
                if remaining <= 0:
                    self._startup_failed("child process launch timed out")
                launched.wait(min(0.05, remaining))
            if launch_error:
                self._startup_failed(f"child process launch failed: {launch_error[0]}")
            proc = self._proc
            if self._cancelled.is_set() or proc is None:
                self._startup_failed("child launch cancelled")
            with self._publication_lock:
                publish_starting = not self._cancelled.is_set() and self._proc is proc
                if publish_starting:
                    _atomic_json(
                        self.temp_dir / "lifecycle.json",
                        {
                            "status": "starting",
                            "pid": proc.pid,
                            "parent_pid": os.getpid(),
                            "socket": str(self._socket_path),
                            "identity": self.identity,
                            "command": command,
                            "started_at": time.time(),
                        },
                    )
            if not publish_starting:
                self._startup_failed("child startup cancelled")
            deadline = self._startup_deadline
            while time.monotonic() < deadline:
                if self._cancelled.is_set():
                    self._startup_failed("child startup cancelled")
                if proc.poll() is not None:
                    self._startup_failed("child exited during startup")
                if self._socket_path.exists():
                    try:
                        status = self._exchange({"type": "status"}, timeout=min(0.25, max(0.001, deadline - time.monotonic())))
                    except (OSError, ModelDaemonError):
                        status = None
                    if status and status.get("state") == "running":
                        with self._publication_lock:
                            publish_ready = not self._cancelled.is_set() and self._proc is proc
                            if publish_ready:
                                self._remote_pid = int(status["pid"])
                                _atomic_json(
                                    self.temp_dir / "lifecycle.json",
                                    {
                                        "status": "ready",
                                        "pid": proc.pid,
                                        "parent_pid": os.getpid(),
                                        "socket": str(self._socket_path),
                                        "identity": self.identity,
                                        "generation": self.generation,
                                        "command": command,
                                        "started_at": time.time(),
                                    },
                                )
                                self._ready = True
                        if not publish_ready:
                            self._startup_failed("child startup cancelled")
                        return
                time.sleep(0.05)
            self._startup_failed("child did not become ready before timeout")

    def _startup_failed(self, message: str) -> None:
        detail = self._startup_error_detail()
        if detail:
            message = f"{message}; child error: {detail}"
        try:
            self._stop_process(graceful=False)
        except BaseException as cleanup_error:
            failure = DaemonStartupError(f"{message}; log: {self.log_path}")
            failure.add_note(f"startup cleanup failed: {cleanup_error!r}")
            raise failure from cleanup_error
        raise DaemonStartupError(f"{message}; log: {self.log_path}")

    def _startup_error_detail(self) -> str | None:
        if self.temp_dir is None:
            return None
        for filename in ("startup_error.json", "lifecycle.json"):
            try:
                record = json.loads((self.temp_dir / filename).read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                continue
            if not isinstance(record, dict) or record.get("status") not in {"startup_failed", "failed"}:
                continue
            error_type = " ".join(str(record.get("error_type") or "").split())[:80]
            error_message = " ".join(str(record.get("message") or "").split())[:400]
            if error_type or error_message:
                return f"{error_type}: {error_message}".strip(": ")
        return None

    def _exchange(
        self,
        request: Mapping[str, Any],
        *,
        timeout: float | None,
    ) -> dict[str, Any]:
        if self._socket_path is None:
            raise DaemonStartupError("daemon socket is not available")
        request_id = str(request.get("request_id") or uuid.uuid4().hex)
        message = {
            "protocol": PROTOCOL,
            "identity": self.identity,
            "request_id": request_id,
            "generation": self.generation,
            **request,
        }
        frame = _frame(message, self.max_frame_bytes)
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sent = False
        try:
            _validate_endpoint(self._socket_path)
            connection.settimeout(timeout if timeout is not None else min(self.startup_timeout, 10.0))
            connection.connect(str(self._socket_path))
            if _peer_uid(connection) != os.getuid():
                raise PermissionError("daemon endpoint requires same UID")
            connection.sendall(frame)
            sent = True
            connection.settimeout(timeout)
            response = _read_frame(connection, self.max_frame_bytes)
        except (socket.timeout, TimeoutError) as error:
            if sent:
                raise RequestOutcomeUnknown(
                    f"{self.spec.backend} request timed out after send; outcome unknown; not replayed",
                    request_id=request_id,
                    generation=self.generation,
                ) from error
            raise ModelDaemonError(f"{self.spec.backend} connection timed out") from error
        except (ConnectionError, FileNotFoundError, DaemonDisconnected) as error:
            proc = self._proc
            if proc is not None:
                try:
                    proc.wait(timeout=min(1.0, timeout) if timeout is not None else 1.0)
                except subprocess.TimeoutExpired:
                    pass
            if proc is not None and proc.poll() is not None:
                raise DaemonChildExited(
                    f"{self.spec.backend} child exited with code {proc.returncode}; log: {self.log_path}"
                ) from error
            raise DaemonDisconnected(f"{self.spec.backend} daemon disconnected") from error
        finally:
            connection.close()
        if response.get("protocol") != PROTOCOL:
            raise ProtocolError("daemon protocol mismatch")
        if response.get("request_id") != request_id:
            raise ProtocolError("daemon response request id mismatch")
        if response.get("generation") != self.generation:
            raise ProtocolError("daemon response generation mismatch")
        if response.get("identity") != self.identity:
            raise IdentityMismatch("daemon identity handshake failed")
        if response.get("type") == "error":
            code = response.get("code")
            if code == "identity_mismatch":
                raise IdentityMismatch("daemon identity handshake failed")
            raise DaemonRequestFailed(
                str(response.get("message") or code or "daemon request failed"),
                remote_type=str(response.get("error_type") or code or "") or None,
            )
        if response.get("pid") is not None:
            self._remote_pid = int(response["pid"])
        return response

    def request(self, payload: Mapping[str, Any]) -> Any:
        if not isinstance(payload, Mapping):
            raise TypeError("request payload must be a mapping")
        self._start()
        timeout = self.request_timeout
        if payload.get("operation") == "describe" and not self._attached:
            remaining = self._startup_deadline - time.monotonic()
            if remaining <= 0:
                self._startup_failed("describe exceeded startup deadline")
            timeout = remaining if timeout is None else min(timeout, remaining)
        response = self._exchange(
            {"type": "infer", "payload": dict(payload)},
            timeout=timeout,
        )
        return response.get("result")

    def status(self) -> dict[str, Any]:
        self._start()
        return self._exchange(
            {"type": "status"},
            timeout=self.request_timeout if self.request_timeout is not None else min(self.startup_timeout, 10.0),
        )

    def _stop_process(self, *, graceful: bool) -> None:
        with self._process_stop_lock:
            self._stop_process_locked(graceful=graceful)

    def _stop_process_locked(self, *, graceful: bool) -> None:
        with self._publication_lock:
            if not graceful:
                self._cancelled.set()
            proc = self._proc
        if proc is None:
            return
        if graceful and self._ready and proc.poll() is None:
            try:
                self._exchange({"type": "stop"}, timeout=self.shutdown_timeout)
            except ModelDaemonError:
                pass
        if not graceful:
            _signal_owned_group(proc, signal.SIGTERM)
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            _signal_owned_group(proc, signal.SIGTERM)
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                _signal_owned_group(proc, signal.SIGKILL)
                proc.wait()
        _signal_owned_group(proc, signal.SIGKILL)
        if self.temp_dir is not None:
            _atomic_json(
                self.temp_dir / "lifecycle.json",
                {
                    "status": "stopped",
                    "pid": proc.pid,
                    "parent_pid": os.getpid(),
                    "socket": str(self._socket_path) if self._socket_path else None,
                    "identity": self.identity,
                    "exit_code": proc.returncode,
                    "stopped_at": time.time(),
                },
            )
        self._proc = None
        self._remove_owned_socket()
        self._close_log()

    def _remove_owned_socket(self) -> None:
        if self.temp_dir is not None and self._socket_path is not None and self._socket_path.parent == self.temp_dir:
            self._socket_path.unlink(missing_ok=True)

    def _close_log(self) -> None:
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def shutdown(self) -> None:
        if self._attached:
            return
        with self._shutdown_lock:
            if self._closed:
                return
            self._cancelled.set()
            self._stop_process(graceful=True)
            self._closed = True
            atexit.unregister(self._atexit)

    def request_stop(self) -> dict[str, Any]:
        self._start()
        response = self._exchange({"type": "stop"}, timeout=self.shutdown_timeout)
        if not self._attached:
            with self._shutdown_lock:
                self._stop_process(graceful=False)
                self._closed = True
                atexit.unregister(self._atexit)
        return response

    def _atexit(self) -> None:
        try:
            self.shutdown()
        except BaseException:
            pass

    def __enter__(self) -> "OwnedModelDaemon":
        return self

    def __exit__(self, exc_type, error, traceback) -> None:
        self.shutdown()


def _child_main(spec_path: Path, socket_path: Path) -> int:
    parent_pid_value = os.environ.get("ASTROLABE_MODEL_PARENT_PID")
    parent_pid = int(parent_pid_value) if parent_pid_value else None
    lifecycle_path = spec_path.with_name("lifecycle.json")
    try:
        _set_parent_death_signal(parent_pid)
        payload = json.loads(spec_path.read_text(encoding="utf-8"))
        if payload.get("protocol") != PROTOCOL:
            raise ValueError("spec protocol mismatch")
        spec = DaemonSpec(
            backend=str(payload["backend"]),
            factory=str(payload["factory"]),
            settings=dict(payload.get("settings") or {}),
            python_bin=str(payload.get("python_bin") or sys.executable),
            cwd=payload.get("cwd"),
            env=payload.get("env"),
            provider=payload.get("provider"),
            max_frame_bytes=int(payload.get("max_frame_bytes", MAX_FRAME_BYTES)),
        )
        provider_report = _activate_child_provider(spec)
        if provider_report is not None:
            _event(
                "provider_activated",
                backend=spec.backend,
                identity=str(payload["identity"]),
                provider=spec.provider,
                report=provider_report,
            )
        factory = _load_factory(str(payload["factory"]))
        _atomic_json(
            lifecycle_path,
            {
                "status": "loading",
                "pid": os.getpid(),
                "parent_pid": parent_pid,
                "socket": str(socket_path),
                "identity": payload["identity"],
                "provider": spec.provider,
                "provider_report": provider_report,
            },
        )
        serve(
            factory,
            dict(payload.get("settings") or {}),
            socket_path=socket_path,
            identity=str(payload["identity"]),
            backend=str(payload["backend"]),
            max_pending=int(payload.get("max_pending", 4)),
            max_frame_bytes=int(payload.get("max_frame_bytes", MAX_FRAME_BYTES)),
            parent_pid=parent_pid,
        )
        _atomic_json(
            lifecycle_path,
            {
                "status": "exited",
                "pid": os.getpid(),
                "parent_pid": parent_pid,
                "socket": str(socket_path),
                "identity": payload["identity"],
                "exit_code": 0,
            },
        )
        return 0
    except BaseException as error:
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        try:
            failure = {
                "status": "startup_failed" if not socket_path.exists() else "failed",
                "pid": os.getpid(),
                "parent_pid": parent_pid,
                "socket": str(socket_path),
                "error_type": type(error).__name__,
                "message": str(error),
            }
            _atomic_json(lifecycle_path.with_name("startup_error.json"), failure)
            _atomic_json(
                lifecycle_path,
                failure,
            )
        except BaseException:
            pass
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Astrolabe persistent model daemon")
    commands = parser.add_subparsers(dest="command", required=True)
    child = commands.add_parser("child")
    child.add_argument("--spec", required=True, type=Path)
    child.add_argument("--socket", required=True, type=Path)
    arguments = parser.parse_args(argv)
    if arguments.command == "child":
        return _child_main(arguments.spec, arguments.socket)
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
