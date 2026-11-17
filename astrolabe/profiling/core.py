"""Process-local profiling, shared only through a JSON environment descriptor.

Schema version 1 events have kind, name, ts_ns (Unix wall clock), duration_ns
(monotonic elapsed time), pid, tid (Python thread identifier), hostname, attrs.
Spans additionally have thread_cpu_ns and error. Instant events have duration 0.
Module globs filter spans and events only, never resource snapshots.
The event budget applies independently to each process; process sidecars retain
dropped counts even after the event budget is exhausted. No file stays open.
"""

from __future__ import annotations

import atexit
from collections.abc import Mapping
from dataclasses import dataclass, fields
import fnmatch
from functools import wraps
import json
import math
import os
from pathlib import Path
import socket
import sys
import threading
import time
import uuid
import warnings


SCHEMA_VERSION = 1
_ENV = "ASTROLABE_PROFILE_SESSION"
_BUFFER_EVENTS = 128
_active = None
_process_pid = os.getpid()
_recorders = []
_sessions = []
_init_lock = threading.Lock()
_bad_descriptor = None


def _warn(message):
    # A user's warnings-as-errors policy must not break scientific execution.
    try:
        warnings.warn(f"Astrolabe profiling: {message}", RuntimeWarning, stacklevel=3)
    except Exception:
        pass


@dataclass(frozen=True)
class ProfileConfig:
    enabled: bool = False
    modules: tuple[str, ...] = ("*",)
    sample_interval_s: float = 1.0
    gpu: bool = False
    cprofile: bool = False
    max_events: int = 200000
    output_dir: str | None = None

    def __post_init__(self):
        for name in ("enabled", "gpu", "cprofile"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"profiling.{name} must be a boolean")
        if (not isinstance(self.modules, (list, tuple))
                or any(not isinstance(item, str) or not item for item in self.modules)):
            raise ValueError("profiling.modules must be a list of nonempty glob strings")
        object.__setattr__(self, "modules", tuple(self.modules))
        value = self.sample_interval_s
        try:
            valid_interval = (not isinstance(value, bool)
                              and isinstance(value, (int, float))
                              and math.isfinite(value) and value > 0)
        except (OverflowError, ValueError):
            valid_interval = False
        if not valid_interval:
            raise ValueError("profiling.sample_interval_s must be finite and positive")
        if type(self.max_events) is not int or self.max_events <= 0:
            raise ValueError("profiling.max_events must be a positive integer")
        if self.output_dir is not None:
            if not isinstance(self.output_dir, (str, os.PathLike)):
                raise ValueError("profiling.output_dir must be a path string or null")
            path = os.fspath(self.output_dir)
            if not isinstance(path, str) or not path or "\0" in path:
                raise ValueError("profiling.output_dir must be a nonempty path string")
            object.__setattr__(self, "output_dir", path)

    @classmethod
    def from_mapping(cls, mapping):
        """Validate configuration, also accepting an already parsed instance."""
        if isinstance(mapping, cls):
            return mapping
        if mapping is None:
            return cls()
        if not isinstance(mapping, Mapping):
            raise ValueError("profiling must be a mapping")
        unknown = set(mapping) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"unknown profiling fields: {sorted(map(str, unknown))}")
        return cls(**dict(mapping))

    def as_dict(self):
        return {field.name: (list(self.modules) if field.name == "modules"
                             else getattr(self, field.name)) for field in fields(self)}


def _safe_host(host):
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in host)


def _json(value):
    return json.dumps(value, ensure_ascii=True, allow_nan=False, default=str)


def _atomic_json(path, value):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        temporary.write_text(_json(value) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


class _Recorder:
    def __init__(self, config, directory, run_id):
        self.config = config
        self.directory = directory
        self.run_id = run_id
        self.pid = os.getpid()
        self.hostname = socket.gethostname()
        self.epoch = time.time_ns()
        self.stem = f"{_safe_host(self.hostname)}-{self.pid}-{self.epoch}"
        self.path = directory / f"events-{self.stem}.jsonl"
        self.lock = threading.RLock()
        self.local = threading.local()
        self.buffer = []
        self.accepted = 0
        self.written = 0
        self.dropped_limit = 0
        self.dropped_io = 0
        self.errors = []
        self.io_failed = False
        self.closed = False
        self.profiles = {}
        self.dirty_profiles = set()
        self.monotonic_origin_ns = time.monotonic_ns()
        _recorders.append(self)

    def matches(self, name):
        return isinstance(name, str) and any(
            fnmatch.fnmatchcase(name, pattern) for pattern in self.config.modules)

    def failure(self, operation, exc):
        message = f"{operation}: {type(exc).__name__}: {str(exc)[:500]}"
        with self.lock:
            if message not in self.errors and len(self.errors) < 16:
                self.errors.append(message)
                _warn(message)

    def record(self, kind, name, attrs, *, ts_ns=None, duration_ns=0, **extra):
        if (self.pid != os.getpid() or self.closed
                or (kind != "resource" and not self.matches(name))):
            return
        with self.lock:
            if self.closed:
                return
            if self.accepted >= self.config.max_events:
                self.dropped_limit += 1
                return
            self.accepted += 1
            if self.io_failed:
                self.dropped_io += 1
                return
            try:
                line = _json(dict(
                    kind=kind, name=name, ts_ns=time.time_ns() if ts_ns is None else ts_ns,
                    duration_ns=duration_ns, pid=self.pid, tid=threading.get_ident(),
                    hostname=self.hostname, attrs=attrs, **extra,
                )) + "\n"
            except Exception as exc:
                self.dropped_io += 1
                self.failure("event serialization", exc)
                return
            self.buffer.append(line)
            if len(self.buffer) >= min(_BUFFER_EVENTS, self.config.max_events):
                self._flush_events()

    def _flush_events(self):
        if not self.buffer:
            return
        batch, self.buffer = self.buffer, []
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.writelines(batch)
            self.written += len(batch)
        except Exception as exc:
            # A partial append is uncertain, so conservatively count its batch lost.
            self.dropped_io += len(batch)
            self.io_failed = True
            self.failure("event append (batch may be partial)", exc)

    def counters(self):
        return dict(accepted_events=self.accepted, written_events=self.written,
                    dropped=self.dropped_limit + self.dropped_io,
                    dropped_limit=self.dropped_limit, dropped_io=self.dropped_io,
                    data_loss=bool(self.dropped_limit or self.dropped_io or self.errors),
                    errors=list(self.errors))

    def flush(self):
        if self.pid != os.getpid():
            return
        with self.lock:
            self._flush_events()
            for tid in list(self.dirty_profiles):
                path = self.directory / f"profile-{self.stem}-{tid}.pstats"
                temporary = path.with_name(path.name + ".tmp")
                try:
                    self.profiles[tid].dump_stats(str(temporary))
                    temporary.replace(path)
                    self.dirty_profiles.discard(tid)
                except Exception as exc:
                    self.failure("cProfile write", exc)
                finally:
                    try:
                        temporary.unlink(missing_ok=True)
                    except OSError:
                        pass
            try:
                _atomic_json(self.directory / f"process-{self.stem}.json", dict(
                    schema_version=SCHEMA_VERSION, run_id=self.run_id, pid=self.pid,
                    hostname=self.hostname, start_wall_time_ns=self.epoch,
                    monotonic_origin_ns=self.monotonic_origin_ns,
                    status="closed" if self.closed else "running",
                    event_file=self.path.name, max_events_scope="per_process",
                    counter_scope="this_process",
                    cprofile_files=[f"profile-{self.stem}-{tid}.pstats"
                                    for tid in self.profiles], **self.counters(),
                ))
            except Exception as exc:
                self.failure("process metadata write", exc)

    def close(self):
        if self.pid != os.getpid():
            return
        with self.lock:
            self.closed = True
            self.flush()
            if self in _recorders:
                _recorders.remove(self)


def _after_fork():
    global _active, _process_pid, _recorders, _sessions, _init_lock, _bad_descriptor
    # Never acquire an inherited lock. Active spans and file objects belong only
    # to their creator PID. Child initialization happens on its first API call.
    if _active is not None:
        try:
            profiler = getattr(_active.local, "profiler", None)
            if profiler is not None:
                profiler.disable()
        except Exception:
            pass
    _active = None
    _process_pid = os.getpid()
    _recorders = []
    _sessions = []
    _init_lock = threading.Lock()
    _bad_descriptor = None


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def _current():
    global _active, _bad_descriptor
    if _process_pid != os.getpid():
        _after_fork()
    if _active is not None:
        return None if _active.closed else _active
    descriptor = os.environ.get(_ENV)
    if not descriptor or descriptor == _bad_descriptor:
        return None
    with _init_lock:
        if _active is not None:
            return None if _active.closed else _active
        try:
            data = json.loads(descriptor)
            if data["schema_version"] != SCHEMA_VERSION:
                raise ValueError("unsupported session descriptor schema")
            config = ProfileConfig.from_mapping(data["config"])
            if not config.enabled:
                return None
            directory = Path(data["directory"])
            if not directory.is_absolute() or not directory.is_dir():
                raise ValueError("session directory is unavailable")
            if not isinstance(data["run_id"], str) or not data["run_id"]:
                raise ValueError("invalid run ID")
            _active = _Recorder(config, directory, data["run_id"])
            # multiprocessing's fork worker uses os._exit(), bypassing atexit.
            # Its native finalizer is needed in addition to ordinary atexit.
            from multiprocessing.util import Finalize
            Finalize(None, _finish_process, exitpriority=0)
            return _active
        except Exception as exc:
            _bad_descriptor = descriptor
            _warn(f"child initialization failed: {type(exc).__name__}: {exc}")
            return None


class _Session:
    def __init__(self, config, output_dir, metadata):
        self.config = config
        self.output_dir = output_dir
        self.metadata = metadata
        self.directory = None
        self.recorder = None
        self.sampler = None
        self.owner_pid = os.getpid()
        self.previous = None
        self.previous_environment = None
        self.manifest = None
        self.finished = False
        self.exiting_at_atexit = False
        self.initialization_failed = False

    def _write_manifest(self):
        if self.manifest is not None:
            self.manifest.update(self.recorder.counters())
            try:
                _atomic_json(self.directory / "session.json", self.manifest)
                return True
            except Exception as exc:
                self.recorder.failure("session metadata write", exc)
        return False

    def __enter__(self):
        global _active
        if not self.config.enabled:
            return self
        self.previous = _current()
        self.previous_environment = os.environ.get(_ENV)
        try:
            run_id = uuid.uuid4().hex
            base = (Path(self.config.output_dir) if self.config.output_dir is not None
                    else Path(self.output_dir) / "profile")
            directory = base.expanduser().absolute() / run_id
            directory.mkdir(parents=True, exist_ok=False)
            self.directory = directory
            self.recorder = _Recorder(self.config, self.directory, run_id)
            _sessions.append(self)
            wall_origin = time.time_ns()
            monotonic_origin = time.monotonic_ns()
            try:
                affinity = sorted(os.sched_getaffinity(0))
            except (AttributeError, OSError):
                affinity = None
            self.manifest = dict(
                schema_version=SCHEMA_VERSION, run_id=run_id, config=self.config.as_dict(),
                start_wall_time_ns=wall_origin, end_wall_time_ns=None,
                wall_time_origin_ns=wall_origin, monotonic_origin_ns=monotonic_origin,
                end_monotonic_ns=None, status="running", hostname=self.recorder.hostname,
                pid=self.owner_pid, python=sys.version, python_executable=sys.executable,
                cpu_affinity=affinity, metadata=self.metadata or {},
                max_events_scope="per_process", thread_cpu_scope="calling_thread_only",
                counter_scope="root_process_only",
                cprofile_scope="per_process_per_thread_outermost_selected_span",
                cprofile_perturbs_execution=self.config.cprofile,
                accelerator_synchronization=False,
                initialization_failed=False,
            )
            if not self._write_manifest():
                self.initialization_failed = True
                self.manifest.update(initialization_failed=True, status="error")
            descriptor = dict(schema_version=SCHEMA_VERSION, run_id=run_id,
                              directory=str(self.directory), config=self.config.as_dict(),
                              root_pid=self.owner_pid, root_hostname=self.recorder.hostname,
                              wall_time_origin_ns=wall_origin,
                              monotonic_origin_ns=monotonic_origin)
            os.environ[_ENV] = _json(descriptor)
            _active = self.recorder
            from .resources import ResourceSampler
            self.sampler = ResourceSampler(self.recorder)
            self.sampler.start()
        except Exception as exc:
            self.initialization_failed = True
            if self.manifest is not None:
                self.manifest.update(initialization_failed=True, status="error")
            if self.recorder is not None:
                self.recorder.failure("session initialization", exc)
            else:
                _warn(f"session initialization failed: {type(exc).__name__}: {exc}")
        return self

    def __exit__(self, exc_type, exc, traceback):
        global _active
        if self.finished or self.owner_pid != os.getpid() or not self.config.enabled:
            return False
        self.finished = True
        try:
            if self.sampler is not None:
                self.sampler.stop()
            if self.recorder is not None:
                self.recorder.close()
                if self.manifest is not None:
                    self.manifest.update(
                        end_wall_time_ns=time.time_ns(), end_monotonic_ns=time.monotonic_ns(),
                        status=("error" if exc_type is not None or self.initialization_failed else
                                "interrupted" if self.exiting_at_atexit else "completed"),
                    )
                    if exc_type is not None:
                        self.manifest["exception_type"] = exc_type.__name__
                    self._write_manifest()
        except Exception as cleanup_error:
            _warn(f"session cleanup failed: {type(cleanup_error).__name__}: {cleanup_error}")
        finally:
            if self in _sessions:
                _sessions.remove(self)
            _active = self.previous
            try:
                if self.previous_environment is None:
                    os.environ.pop(_ENV, None)
                else:
                    os.environ[_ENV] = self.previous_environment
            except Exception as cleanup_error:
                _warn(f"environment restoration failed: {cleanup_error}")
        return False


def profile_session(config, *, output_dir, metadata=None):
    """Create a unique session under config.output_dir or output_dir/profile.

    The returned context exposes ``directory`` (None when disabled). Configuration
    errors raise before entry; collection failures only warn and retain counters.
    Children inheriting child_environment() write independent event files lazily.
    """
    return _Session(ProfileConfig.from_mapping(config), output_dir, metadata)


class _Span:
    def __init__(self, name, attrs):
        self.name = name
        self.attrs = attrs
        self.recorder = None
        self.profiler = None

    def __enter__(self):
        try:
            recorder = _current()
            if recorder is None or not recorder.matches(self.name):
                return self
            self.recorder = recorder
            self.wall_start = time.time_ns()
            self.monotonic_start = time.monotonic_ns()
            self.cpu_start = time.thread_time_ns()
            self.depth = getattr(recorder.local, "depth", 0)
            recorder.local.depth = self.depth + 1
            if recorder.config.cprofile and self.depth == 0:
                if sys.getprofile() is not None:
                    recorder.failure("cProfile skipped", RuntimeError("existing profiler retained"))
                else:
                    import cProfile
                    self.profiler = cProfile.Profile()
                    recorder.local.profiler = self.profiler
                    self.profiler.enable()
        except Exception as exc:
            _warn(f"span initialization failed: {type(exc).__name__}: {exc}")
        return self

    def __exit__(self, exc_type, exc, traceback):
        recorder = self.recorder
        if recorder is None or recorder.pid != os.getpid():
            return False
        try:
            elapsed = time.monotonic_ns() - self.monotonic_start
            cpu_elapsed = time.thread_time_ns() - self.cpu_start
            if self.profiler is not None:
                self.profiler.disable()
                recorder.local.profiler = None
                import pstats
                tid = threading.get_ident()
                stats = pstats.Stats(self.profiler)
                with recorder.lock:
                    if tid in recorder.profiles:
                        recorder.profiles[tid].add(stats)
                    else:
                        recorder.profiles[tid] = stats
                    recorder.dirty_profiles.add(tid)
            recorder.record("span", self.name, self.attrs, ts_ns=self.wall_start,
                            duration_ns=elapsed, thread_cpu_ns=cpu_elapsed,
                            error=exc_type is not None)
        except Exception as cleanup_error:
            recorder.failure("span completion", cleanup_error)
        finally:
            recorder.local.depth = getattr(self, "depth", 0)
        return False


def span(name, **attrs):
    """Time a selected module name; no accelerator synchronization is performed."""
    return _Span(name, attrs)


def profiled(name):
    """Decorate a synchronous function, preserving its result and exceptions."""
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with span(name):
                return function(*args, **kwargs)
        return wrapped
    return decorate


def event(name, **attrs):
    """Record an instantaneous selected event (duration_ns is zero)."""
    try:
        recorder = _current()
        if recorder is not None:
            recorder.record("event", name, attrs)
    except Exception as exc:
        _warn(f"event failed: {type(exc).__name__}: {exc}")


def flush():
    """Flush this process's bounded buffer and cumulative completed cProfiles."""
    try:
        recorder = _current()
        if recorder is not None:
            recorder.flush()
    except Exception as exc:
        _warn(f"flush failed: {type(exc).__name__}: {exc}")


def child_environment():
    """Return only the session variable that subprocess launchers must forward."""
    try:
        recorder = _current()
        descriptor = os.environ.get(_ENV)
        return {_ENV: descriptor} if recorder is not None and descriptor else {}
    except Exception as exc:
        _warn(f"child environment unavailable: {exc}")
        return {}


def _finish_process():
    for session in list(reversed(_sessions)):
        try:
            session.exiting_at_atexit = True
            session.__exit__(None, None, None)
        except Exception as exc:
            _warn(f"session atexit cleanup failed: {exc}")
    for recorder in list(_recorders):
        try:
            recorder.close()
        except Exception as exc:
            _warn(f"process cleanup failed: {exc}")


atexit.register(_finish_process)
