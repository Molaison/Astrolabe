"""Public diagnostics and explicit foreground control for model owners.

Task execution remains the owner of normal model lifecycles. These commands
inspect the private directories created by ``OwnedModelDaemon``, query live
workers through their verified Unix sockets, and provide an advanced foreground
service mode for development or site-managed supervision.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import stat
import sys
import tempfile
import threading
from typing import Any, Mapping, Sequence

from astrolabe.backend_catalog import BACKENDS, catalog_payload
from astrolabe.cli_command_definitions import set_handler
from astrolabe.model_daemon import (
    MAX_FRAME_BYTES,
    PROTOCOL,
    DaemonChildExited,
    DaemonSpec,
    IdentityMismatch,
    ModelDaemonError,
    OwnedModelDaemon,
    RequestOutcomeUnknown,
    control_request,
    model_identity,
)


DAEMON_DIRECTORY_PREFIX = "astrolabe-model-daemon-"
SPEC_FILENAME = "spec.json"
LIFECYCLE_FILENAME = "lifecycle.json"
SOCKET_FILENAME = "model.sock"
MAX_EVIDENCE_BYTES = 1024 * 1024
CONTROL_TIMEOUT_SECONDS = 10.0
_TERMINAL_LIFECYCLE_STATES = frozenset(
    {"stopped", "exited", "startup_failed", "failed"}
)
_LIVE_STATES = frozenset({"running", "draining"})


class DaemonCliError(ValueError):
    """A daemon request is invalid or cannot be completed safely."""


@dataclass(frozen=True)
class DaemonRecord:
    directory: Path
    backend: str
    identity: str
    generation: str | None
    socket_path: Path
    max_pending: int
    max_frame_bytes: int
    lifecycle_status: str
    lifecycle_pid: int | None
    lifecycle_started_at: float | None
    lifecycle_stopped_at: float | None
    lifecycle_exit_code: int | None
    lifecycle_error_type: str | None
    lifecycle_message: str | None


def _read_owner_json(path: Path) -> dict[str, Any]:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise DaemonCliError(f"Required evidence file is missing: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise DaemonCliError(f"Evidence path must be a regular file, not a symlink: {path}")
    if metadata.st_uid != os.getuid():
        raise DaemonCliError(f"Evidence file has the wrong owner: {path}")
    if metadata.st_size > MAX_EVIDENCE_BYTES:
        raise DaemonCliError(f"Evidence file is too large: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DaemonCliError(f"Cannot read daemon evidence file {path}: {error}") from error
    if not isinstance(value, dict):
        raise DaemonCliError(f"Daemon evidence must be a JSON object: {path}")
    return value


def _validate_daemon_directory(directory: str | Path) -> Path:
    candidate = Path(directory).expanduser()
    try:
        metadata = candidate.lstat()
    except FileNotFoundError as error:
        raise DaemonCliError(f"Daemon directory does not exist: {candidate}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise DaemonCliError(f"Daemon path must be a real directory: {candidate}")
    resolved = candidate.resolve()
    temp_root = Path(tempfile.gettempdir()).resolve()
    if resolved.parent != temp_root:
        raise DaemonCliError(
            f"Daemon directory must be directly under the current temp root {temp_root}: {resolved}"
        )
    if not resolved.name.startswith(DAEMON_DIRECTORY_PREFIX):
        raise DaemonCliError(
            f"Daemon directory must start with {DAEMON_DIRECTORY_PREFIX!r}: {resolved}"
        )
    current = resolved.stat()
    if current.st_uid != os.getuid():
        raise DaemonCliError(f"Daemon directory has the wrong owner: {resolved}")
    if stat.S_IMODE(current.st_mode) != 0o700:
        raise DaemonCliError(f"Daemon directory must be owner-only mode 0700: {resolved}")
    for name in (SPEC_FILENAME, LIFECYCLE_FILENAME):
        if not (resolved / name).is_file():
            raise DaemonCliError(f"Daemon directory is missing {name}: {resolved}")
    return resolved


def _evidence_int(
    value: Any,
    *,
    label: str,
    default: int | None = None,
) -> int | None:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise DaemonCliError(f"{label} must be an integer") from error
    return parsed


def _record_from_directory(directory: str | Path) -> DaemonRecord:
    root = _validate_daemon_directory(directory)
    spec = _read_owner_json(root / SPEC_FILENAME)
    lifecycle = _read_owner_json(root / LIFECYCLE_FILENAME)

    protocol = spec.get("protocol")
    if protocol is not None and protocol != PROTOCOL:
        raise DaemonCliError(f"Daemon protocol mismatch in {root / SPEC_FILENAME}")
    backend = str(spec.get("backend") or "").strip()
    identity = str(spec.get("identity") or "").strip()
    if not backend or not identity:
        raise DaemonCliError(f"Daemon spec lacks backend or identity: {root}")

    lifecycle_identity = str(lifecycle.get("identity") or "").strip()
    if lifecycle_identity and lifecycle_identity != identity:
        raise DaemonCliError(f"Spec and lifecycle identities differ in {root}")
    lifecycle_backend = str(lifecycle.get("backend") or "").strip()
    if lifecycle_backend and lifecycle_backend != backend:
        raise DaemonCliError(f"Spec and lifecycle backends differ in {root}")

    raw_socket = lifecycle.get("socket") or str(root / SOCKET_FILENAME)
    socket_path = Path(str(raw_socket)).expanduser()
    if not socket_path.is_absolute():
        raise DaemonCliError(f"Daemon socket path must be absolute: {socket_path}")
    socket_path = socket_path.resolve(strict=False)
    if socket_path.parent != root or socket_path.name != SOCKET_FILENAME:
        raise DaemonCliError(
            f"Daemon socket must be {root / SOCKET_FILENAME}, got {socket_path}"
        )

    max_pending = _evidence_int(
        spec.get("max_pending"),
        label="max_pending",
        default=4,
    )
    max_frame_bytes = _evidence_int(
        spec.get("max_frame_bytes"),
        label="max_frame_bytes",
        default=MAX_FRAME_BYTES,
    )
    if max_pending is None or max_pending < 1:
        raise DaemonCliError("Daemon max_pending must be positive")
    if max_frame_bytes is None or max_frame_bytes < 1:
        raise DaemonCliError("Daemon max_frame_bytes must be positive")

    generation_value = spec.get("generation") or lifecycle.get("generation")
    lifecycle_error_type = lifecycle.get("error_type")
    lifecycle_message = lifecycle.get("message")
    return DaemonRecord(
        directory=root,
        backend=backend,
        identity=identity,
        generation=str(generation_value) if generation_value else None,
        socket_path=socket_path,
        max_pending=max_pending,
        max_frame_bytes=max_frame_bytes,
        lifecycle_status=str(lifecycle.get("status") or "unknown"),
        lifecycle_pid=_evidence_int(lifecycle.get("pid"), label="lifecycle pid"),
        lifecycle_started_at=(
            float(lifecycle["started_at"])
            if lifecycle.get("started_at") is not None
            else None
        ),
        lifecycle_stopped_at=(
            float(lifecycle["stopped_at"])
            if lifecycle.get("stopped_at") is not None
            else None
        ),
        lifecycle_exit_code=_evidence_int(
            lifecycle.get("exit_code"),
            label="lifecycle exit_code",
        ),
        lifecycle_error_type=(
            str(lifecycle_error_type)[:120] if lifecycle_error_type else None
        ),
        lifecycle_message=(
            str(lifecycle_message)[:500] if lifecycle_message else None
        ),
    )


def _daemon_directories(explicit: str | Path | None) -> tuple[Path, ...]:
    if explicit is not None:
        return (_validate_daemon_directory(explicit),)
    root = Path(tempfile.gettempdir()).resolve()
    directories: list[Path] = []
    try:
        candidates = sorted(root.iterdir(), key=lambda path: path.name)
    except OSError as error:
        raise DaemonCliError(f"Cannot inspect temp directory {root}: {error}") from error
    for candidate in candidates:
        if not candidate.name.startswith(DAEMON_DIRECTORY_PREFIX):
            continue
        try:
            directories.append(_validate_daemon_directory(candidate))
        except (DaemonCliError, OSError):
            continue
    return tuple(directories)


def _lifecycle_evidence(record: DaemonRecord) -> dict[str, Any]:
    return {
        "status": record.lifecycle_status,
        "pid": record.lifecycle_pid,
        "started_at": record.lifecycle_started_at,
        "stopped_at": record.lifecycle_stopped_at,
        "exit_code": record.lifecycle_exit_code,
        "error_type": record.lifecycle_error_type,
        "message": record.lifecycle_message,
    }


def _owner_acknowledged_drain(owner: OwnedModelDaemon) -> bool:
    log_path = owner.log_path
    if log_path is None:
        return False
    try:
        size = log_path.stat().st_size
        with log_path.open("rb") as handle:
            handle.seek(max(0, size - MAX_EVIDENCE_BYTES))
            for raw_line in handle:
                try:
                    event = json.loads(raw_line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if (
                    isinstance(event, Mapping)
                    and event.get("event") == "shutdown_received"
                ):
                    return True
    except OSError:
        return False
    return False


def _child_exited_after_acknowledged_drain(owner: OwnedModelDaemon) -> bool:
    if owner.temp_dir is None or not _owner_acknowledged_drain(owner):
        return False
    try:
        record = _record_from_directory(owner.temp_dir)
    except (DaemonCliError, OSError):
        return False
    return (
        record.lifecycle_status in _TERMINAL_LIFECYCLE_STATES
        and record.lifecycle_exit_code == 0
    )


def _status_record(record: DaemonRecord) -> dict[str, Any]:
    base = {
        "directory": str(record.directory),
        "backend": record.backend,
        "identity": record.identity,
        "generation_recorded": record.generation,
        "socket": str(record.socket_path),
        "max_pending": record.max_pending,
        "max_frame_bytes": record.max_frame_bytes,
        "actual_rpc": False,
        "generation_verified": False,
        "last_evidence": _lifecycle_evidence(record),
    }
    if not record.socket_path.exists():
        state = (
            "stopped"
            if record.lifecycle_status in _TERMINAL_LIFECYCLE_STATES
            else "unknown"
        )
        return {**base, "state": state, "ready": False}

    try:
        response = control_request(
            record.socket_path,
            record.identity,
            command="status",
            timeout=CONTROL_TIMEOUT_SECONDS,
            max_frame_bytes=record.max_frame_bytes,
        )
        actual_backend = str(response.get("backend") or "")
        if actual_backend and actual_backend != record.backend:
            raise DaemonCliError(
                f"Live daemon backend mismatch: expected {record.backend}, got {actual_backend}"
            )
        actual_socket = response.get("socket")
        if actual_socket:
            if Path(str(actual_socket)).resolve(strict=False) != record.socket_path:
                raise DaemonCliError("Live daemon reported a different socket path")
        state = str(response.get("state") or "")
        if state not in _LIVE_STATES:
            raise DaemonCliError(f"Unexpected live daemon state: {state!r}")
        actual_generation = response.get("generation")
        generation_verified = bool(
            record.generation
            and actual_generation
            and str(actual_generation) == record.generation
        )
        return {
            **base,
            "actual_rpc": True,
            "generation_verified": generation_verified,
            "state": state,
            "ready": state == "running",
            "pid": response.get("pid"),
            "parent_pid": response.get("parent_pid"),
            "queue_depth": response.get("queue_depth"),
            "active_request": response.get("active_request"),
            "request_count": response.get("request_count"),
            "completed_count": response.get("completed_count"),
            "failed_count": response.get("failed_count"),
            "model_load_count": response.get("model_load_count"),
            "started_at": response.get("started_at"),
        }
    except (
        DaemonCliError,
        IdentityMismatch,
        ModelDaemonError,
        OSError,
        PermissionError,
    ) as error:
        state = (
            "stopped"
            if record.lifecycle_status in _TERMINAL_LIFECYCLE_STATES
            else "unknown"
        )
        return {
            **base,
            "state": state,
            "ready": False,
            "error": f"{type(error).__name__}: {str(error)[:500]}",
        }


def collect_status(
    *,
    directory: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return actual live status or stale/dead evidence; never mutate a daemon."""

    records: list[dict[str, Any]] = []
    for path in _daemon_directories(directory):
        try:
            records.append(_status_record(_record_from_directory(path)))
        except (DaemonCliError, OSError) as error:
            if directory is not None:
                raise DaemonCliError(f"Cannot inspect daemon {path}: {error}") from error
    return records


def stop_daemon(*, directory: str | Path) -> dict[str, Any]:
    """Drain one exact same-UID daemon through its verified control socket."""

    record = _record_from_directory(directory)
    if not record.socket_path.exists():
        raise DaemonCliError(
            f"Daemon socket is not available: {record.socket_path}; "
            "the stale lifecycle record was not used to signal a PID."
        )
    try:
        response = control_request(
            record.socket_path,
            record.identity,
            command="stop",
            timeout=CONTROL_TIMEOUT_SECONDS,
            max_frame_bytes=record.max_frame_bytes,
        )
    except RequestOutcomeUnknown as error:
        raise DaemonCliError(
            f"Stop request timed out after send; outcome unknown: {error}"
        ) from error
    except (IdentityMismatch, ModelDaemonError, OSError, PermissionError) as error:
        raise DaemonCliError(
            f"Refusing to stop {record.directory}: {type(error).__name__}: {error}"
        ) from error
    result = response.get("result")
    if not isinstance(result, Mapping) or result.get("drained") is not True:
        raise DaemonCliError("Daemon did not acknowledge a completed drain")
    return {
        "directory": str(record.directory),
        "backend": record.backend,
        "identity": record.identity,
        "state": "drained",
        "actual_rpc": True,
        "completed_count": result.get("completed_count"),
        "failed_count": result.get("failed_count"),
    }


def _load_launch_spec(path: str | Path) -> tuple[DaemonSpec, int, str | None]:
    spec_path = Path(path).expanduser().resolve()
    payload = _read_owner_json(spec_path)
    protocol = payload.get("protocol")
    if protocol is not None and protocol != PROTOCOL:
        raise DaemonCliError("Service spec protocol mismatch")
    backend = str(payload.get("backend") or "").strip()
    factory = str(payload.get("factory") or "").strip()
    settings = payload.get("settings")
    if not backend or not factory:
        raise DaemonCliError("Service spec requires backend and factory")
    if ":" not in factory and "." not in factory:
        raise DaemonCliError("Service factory must name a module and callable")
    if not isinstance(settings, dict):
        raise DaemonCliError("Service spec settings must be a JSON object")
    env = payload.get("env")
    if env is not None and (
        not isinstance(env, dict)
        or any(not isinstance(key, str) or not isinstance(value, str) for key, value in env.items())
    ):
        raise DaemonCliError("Service spec env must map strings to strings")
    provider = payload.get("provider")
    if provider is not None and not isinstance(provider, dict):
        raise DaemonCliError("Service spec provider must be a JSON object")
    cwd_value = payload.get("cwd")
    cwd = None
    if cwd_value:
        cwd_path = Path(str(cwd_value)).expanduser().resolve()
        if not cwd_path.is_dir():
            raise DaemonCliError(f"Service working directory does not exist: {cwd_path}")
        cwd = str(cwd_path)
    max_pending = _evidence_int(
        payload.get("max_pending"),
        label="max_pending",
        default=4,
    )
    max_frame_bytes = _evidence_int(
        payload.get("max_frame_bytes"),
        label="max_frame_bytes",
        default=MAX_FRAME_BYTES,
    )
    if max_pending is None or max_pending < 1:
        raise DaemonCliError("Service max_pending must be positive")
    if max_frame_bytes is None or max_frame_bytes < 1:
        raise DaemonCliError("Service max_frame_bytes must be positive")
    python_bin = str(payload.get("python_bin") or sys.executable)
    spec = DaemonSpec(
        backend=backend,
        factory=factory,
        settings=dict(settings),
        python_bin=python_bin,
        cwd=cwd,
        env=dict(env) if env is not None else None,
        provider=dict(provider) if provider is not None else None,
        max_frame_bytes=max_frame_bytes,
    )
    expected_identity = str(payload.get("identity") or "") or None
    if expected_identity is not None:
        actual_identity = model_identity(spec, max_pending=max_pending)
        if actual_identity != expected_identity:
            raise DaemonCliError(
                "Service spec identity does not match its backend, factory, settings, "
                "interpreter, working directory, environment, provider, and bounds"
            )
    recorded_generation = payload.get("generation")
    return spec, max_pending, str(recorded_generation) if recorded_generation else None


def _event(event: str, **fields: Any) -> None:
    print(
        json.dumps(
            {"event": event, "protocol": PROTOCOL, **fields},
            sort_keys=True,
            default=str,
        ),
        flush=True,
    )


def serve_foreground(
    spec_path: str | Path,
    *,
    poll_interval: float = 1.0,
) -> int:
    """Run one explicitly managed model owner in the current foreground process."""

    spec: DaemonSpec | None = None
    stop_event = threading.Event()
    previous_handlers: dict[int, Any] = {}
    owner: OwnedModelDaemon | None = None

    def request_shutdown(signum: int, _frame: Any) -> None:
        _event("shutdown_requested", signal=signal.Signals(signum).name)
        stop_event.set()

    def drain_requested_owner(active_owner: OwnedModelDaemon) -> None:
        socket_path = active_owner.socket_path
        if socket_path is None or not socket_path.exists():
            return
        try:
            control_request(
                socket_path,
                active_owner.identity,
                command="stop",
                timeout=CONTROL_TIMEOUT_SECONDS * 3,
                max_frame_bytes=active_owner.max_frame_bytes,
            )
        except (IdentityMismatch, ModelDaemonError, OSError, PermissionError) as error:
            _event(
                "drain_request_failed",
                error_type=type(error).__name__,
                message=str(error),
            )
            return
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, request_shutdown)
    try:
        spec, max_pending, recorded_generation = _load_launch_spec(spec_path)
        owner = OwnedModelDaemon(
            spec,
            startup_timeout=300,
            request_timeout=CONTROL_TIMEOUT_SECONDS,
            shutdown_timeout=30,
            max_pending=max_pending,
        )
        status = owner.status()
        if status.get("state") != "running":
            raise DaemonCliError(f"Model owner did not become ready: {status.get('state')}")
        _event(
            "service_ready",
            backend=spec.backend,
            directory=str(owner.temp_dir),
            socket=str(owner.socket_path),
            identity=owner.identity,
            recorded_generation=recorded_generation,
            status=status,
        )
        while not stop_event.wait(poll_interval):
            try:
                owner.status()
            except DaemonChildExited as error:
                drained = _child_exited_after_acknowledged_drain(owner)
                _event(
                    "service_stopped",
                    reason="drained" if drained else "child_failed",
                    detail=str(error),
                )
                return 0 if drained else 2
        return 0
    except (
        DaemonCliError,
        ModelDaemonError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        _event("service_failed", error_type=type(error).__name__, message=str(error))
        return 2
    finally:
        if owner is not None:
            if stop_event.is_set():
                drain_requested_owner(owner)
            owner.shutdown()
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)
        _event("service_exited", backend=spec.backend if spec is not None else None)


def _print_backends(*, as_json: bool) -> None:
    payload = catalog_payload()
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print(f"{'backend':26} {'component':12} {'execution':22} software")
    for record in BACKENDS:
        software = ",".join(record.software) or "-"
        print(
            f"{record.name:26} {record.component:12} "
            f"{record.execution:22} {software}"
        )
        print(f"  {record.detail}")


def _print_status(records: Sequence[Mapping[str, Any]], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps({"schema_version": 1, "daemons": list(records)}, indent=2, sort_keys=True))
        return
    if not records:
        print("No owner-only Astrolabe model daemon directories found.")
        return
    for record in records:
        line = (
            f"{record['state']:8} backend={record['backend']} "
            f"directory={record['directory']} identity={str(record['identity'])[:12]}"
        )
        if "pid" in record:
            line += f" pid={record['pid']}"
        print(line)
        evidence = record.get("last_evidence")
        if isinstance(evidence, Mapping):
            print(
                f"  last_evidence status={evidence.get('status')} "
                f"pid={evidence.get('pid')} exit_code={evidence.get('exit_code')}"
            )
        if record.get("error"):
            print(f"  error={record['error']}")


def _cmd_backends(namespace: argparse.Namespace) -> int:
    _print_backends(as_json=namespace.json)
    return 0


def _cmd_status(namespace: argparse.Namespace) -> int:
    try:
        records = collect_status(directory=namespace.directory)
    except DaemonCliError as error:
        if namespace.json:
            print(
                json.dumps(
                    {"schema_version": 1, "daemons": [], "error": str(error)},
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(f"astrolabe daemon status: {error}", file=sys.stderr)
        return 2
    _print_status(records, as_json=namespace.json)
    return 0


def _cmd_stop(namespace: argparse.Namespace) -> int:
    try:
        payload = stop_daemon(directory=namespace.directory)
    except DaemonCliError as error:
        if namespace.json:
            print(
                json.dumps(
                    {
                        "directory": str(Path(namespace.directory).expanduser().resolve()),
                        "state": "error",
                        "actual_rpc": False,
                        "error": str(error),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(f"astrolabe daemon stop: {error}", file=sys.stderr)
        return 2
    if namespace.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            f"drained backend={payload['backend']} directory={payload['directory']} "
            f"completed={payload['completed_count']} failed={payload['failed_count']}"
        )
    return 0


def _cmd_serve(namespace: argparse.Namespace) -> int:
    return serve_foreground(namespace.spec)


def register_commands(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``astrolabe daemon`` management commands."""

    daemon = subparsers.add_parser(
        "daemon",
        help="Inspect or explicitly manage persistent model owners.",
        description=(
            "Normal task execution owns and drains its private model daemons. "
            "These commands are diagnostics and advanced operator controls."
        ),
    )
    daemon.set_defaults(_cuda_preinitialize=False)
    commands = daemon.add_subparsers(dest="daemon_command", required=True)

    backends = commands.add_parser(
        "backends",
        help="List known execution policies from the backend catalog.",
    )
    backends.set_defaults(_cuda_preinitialize=False)
    backends.add_argument("--json", action="store_true")
    set_handler(backends, _cmd_backends)

    status = commands.add_parser(
        "status",
        help="Query live owner daemons and report stale records.",
    )
    status.set_defaults(_cuda_preinitialize=False)
    status.add_argument("--directory", type=Path)
    status.add_argument("--json", action="store_true")
    set_handler(status, _cmd_status)

    stop = commands.add_parser(
        "stop",
        help="Drain one exact same-UID owner and its owned model child.",
    )
    stop.set_defaults(_cuda_preinitialize=False)
    stop.add_argument("--directory", type=Path, required=True)
    stop.add_argument("--json", action="store_true")
    set_handler(stop, _cmd_stop)

    serve = commands.add_parser(
        "serve",
        help="Run one explicitly managed model owner in the foreground.",
        description=(
            "Advanced use only. The owner stays attached to this process and drains "
            "its child on SIGINT or SIGTERM. It does not install or detach services."
        ),
    )
    serve.set_defaults(_cuda_preinitialize=False)
    serve.add_argument("--spec", type=Path, required=True)
    set_handler(serve, _cmd_serve)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="astrolabe daemon")
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_commands(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    namespace = parser.parse_args(list(argv) if argv is not None else None)
    handler = getattr(namespace, "_handler", None)
    if handler is None:
        parser.error("No daemon command handler selected.")
    return int(handler(namespace))


__all__ = [
    "DAEMON_DIRECTORY_PREFIX",
    "DaemonCliError",
    "collect_status",
    "register_commands",
    "serve_foreground",
    "stop_daemon",
]


if __name__ == "__main__":
    raise SystemExit(main())
