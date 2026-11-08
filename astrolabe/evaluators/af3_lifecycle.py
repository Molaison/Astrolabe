"""Lifecycle of the private AF3 subprocess owned by one interface invocation."""
from __future__ import annotations

import json
from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, replace
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
import uuid

from astrolabe.runtime.evaluation import (
    AdaptiveEvaluationSettings, adaptive_settings, require_private_evaluator_owner,
)


@dataclass(frozen=True)
class AF3ServiceSettings:
    startup_timeout_s: float = 600.0
    shutdown_timeout_s: float = 120.0
    terminate_timeout_s: float = 10.0
    max_pending: int = 4
    concurrency_mode: str = "process_pool"
    adaptive: AdaptiveEvaluationSettings = AdaptiveEvaluationSettings()


@dataclass(frozen=True)
class AF3ChildEnvironment:
    python_bin: str
    environment: dict[str, str]
    library_directories: tuple[str, ...]


_AF3_SERVICE_FIELDS = frozenset(AF3ServiceSettings.__dataclass_fields__)


def service_settings(cfg, explicit_settings=None):
    """Resolve and validate AF3 transport settings without changing *cfg*."""
    if explicit_settings is None:
        protein_beam = cfg.get("protein_beam") if hasattr(cfg, "get") else None
        source = (
            protein_beam.get("service")
            if protein_beam is not None and hasattr(protein_beam, "get")
            else None
        )
    else:
        source = explicit_settings
    if isinstance(source, AF3ServiceSettings):
        values = {
            name: getattr(source, name)
            for name in _AF3_SERVICE_FIELDS
        }
    elif source is None:
        values = {}
    elif isinstance(source, Mapping):
        values = dict(source)
    else:
        raise TypeError("AF3 service settings must be a mapping")

    unknown = sorted(str(key) for key in set(values) - _AF3_SERVICE_FIELDS)
    if unknown:
        raise ValueError(f"Unknown AF3 service settings: {', '.join(unknown)}")

    timeout_values = {}
    for name in (
        "startup_timeout_s",
        "shutdown_timeout_s",
        "terminate_timeout_s",
    ):
        default = getattr(AF3ServiceSettings(), name)
        value = values.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite positive number")
        number = float(value)
        if not math.isfinite(number) or number <= 0:
            raise ValueError(f"{name} must be a finite positive number")
        timeout_values[name] = number

    max_pending = values.get("max_pending", AF3ServiceSettings().max_pending)
    if isinstance(max_pending, bool) or not isinstance(max_pending, int):
        raise ValueError("max_pending must be a positive integer")
    if max_pending <= 0:
        raise ValueError("max_pending must be a positive integer")
    concurrency_mode = values.get("concurrency_mode", "process_pool")
    if concurrency_mode not in {"process_pool", "resident"}:
        raise ValueError("concurrency_mode must be process_pool or resident")
    return AF3ServiceSettings(max_pending=max_pending,
                              concurrency_mode=concurrency_mode,
                              adaptive=adaptive_settings(values.get("adaptive")), **timeout_values)


def guard_owner_parent():
    """An owned model must not outlive a killed coordinator (Linux runtime)."""
    expected = os.environ.get('ASTROLABE_AF3_PARENT_PID')
    if expected is None:
        return
    if os.getppid() != int(expected):
        raise RuntimeError('Owned AF3 coordinator has already ended')
    from astrolabe.process_lifetime import protect_worker_lifetime
    protect_worker_lifetime()
    if os.getppid() != int(expected):
        raise RuntimeError('Owned AF3 coordinator ended during startup')


def _owner_command(config_path, socket_path):
    return [sys.executable, '-u', '-m', 'astrolabe.evaluators.af3_service',
            '--config', str(config_path), '--socket', str(socket_path)]


def _resolve_python(python_bin):
    value = os.fspath(python_bin)
    resolved = shutil.which(value) if not os.path.isabs(value) else value
    if resolved is None:
        raise FileNotFoundError(f'AF3 child interpreter does not exist: {value}')
    path = Path(resolved).expanduser().resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise FileNotFoundError(f'AF3 child interpreter is not executable: {path}')
    return path


def _target_sysconfig_paths(python_bin):
    if python_bin == Path(sys.executable).resolve():
        paths = sysconfig.get_paths()
    else:
        command = [
            str(python_bin), '-I', '-S', '-B', '-c',
            'import json,sysconfig; print(json.dumps(sysconfig.get_paths(),sort_keys=True))',
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(
                f'Cannot inspect AF3 child interpreter {python_bin}: {detail}'
            )
        try:
            paths = json.loads(completed.stdout)
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                f'AF3 child interpreter returned invalid sysconfig data: {python_bin}'
            ) from error
    if not isinstance(paths, Mapping):
        raise RuntimeError(f'AF3 child interpreter has invalid sysconfig data: {python_bin}')
    missing = [name for name in ('purelib', 'platlib') if not isinstance(paths.get(name), str)]
    if missing:
        raise RuntimeError(
            f'AF3 child interpreter sysconfig lacks: {", ".join(missing)}'
        )
    return paths


def _nvidia_library_directories(python_bin):
    sys_paths = _target_sysconfig_paths(python_bin)
    roots = []
    for name in ('purelib', 'platlib'):
        root = Path(sys_paths[name]).resolve()
        if root not in roots:
            roots.append(root)
    directories = []
    for root in roots:
        for directory in sorted((root / 'nvidia').glob('*/lib')):
            if not directory.is_dir():
                continue
            if not any(path.is_file() or path.is_symlink() for path in directory.iterdir()):
                continue
            resolved = directory.resolve()
            if resolved not in directories:
                directories.append(resolved)
    return tuple(directories)


def af3_child_environment(python_bin, environment):
    """Return a copied child env with target-environment NVIDIA wheel libs."""
    executable = _resolve_python(python_bin)
    directories = _nvidia_library_directories(executable)
    child_environment = {
        str(key): str(value) for key, value in environment.items()
    }
    if directories:
        existing = child_environment.get('LD_LIBRARY_PATH', '')
        entries, seen = [], set()
        for entry in existing.split(os.pathsep) if existing else ():
            if not entry:
                entries.append(entry)
                continue
            value = str(Path(entry).resolve())
            if value not in seen:
                entries.append(entry)
                seen.add(value)
        for directory in directories:
            value = str(directory)
            if value not in seen:
                entries.append(value)
                seen.add(value)
        child_environment['LD_LIBRARY_PATH'] = os.pathsep.join(entries)
    return AF3ChildEnvironment(
        python_bin=str(executable),
        environment=child_environment,
        library_directories=tuple(str(path) for path in directories),
    )


@contextmanager
def deferred_termination_signals():
    # Python dispatches signal handlers in the main thread even if the kernel
    # delivered SIGINT to an executor thread. Per-thread masking is insufficient.
    signals = (signal.SIGINT, signal.SIGTERM)
    if threading.current_thread() is not threading.main_thread():
        previous = signal.pthread_sigmask(signal.SIG_BLOCK, set(signals))
        try:
            yield
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous)
        return
    handlers = {sig: signal.getsignal(sig) for sig in signals}
    pending = []
    def defer(signum, frame):
        if signum not in pending:
            pending.append(signum)
    for sig in signals:
        signal.signal(sig, defer)
    try:
        yield
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        # Nested scopes forward to their enclosing deferral handler. The outer
        # scope delivers the user's original signal after owner/workers end.
        for sig in pending:
            signal.raise_signal(sig)


def _process_identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return None
    return {'start_ticks': fields[19], 'state': fields[0]}


def _boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def recover_owned_services(output):
    """Remove a dead *verified private* owner's socket before native retry checks.

    No process is killed and no result/quota is changed. Original lifecycle
    records are preserved; recovery is a separate artifact.
    """
    import stat
    from astrolabe.design.io import atomic_json
    for path in sorted((Path(output) / 'services').glob('*/lifecycle.json')):
        state = json.loads(path.read_text())
        if not state.get('pid') or not state.get('socket'):
            continue
        socket_path = Path(state['socket'])
        confirmed_ended = state.get('ended_at') is not None and state.get('exit_code') is not None
        if confirmed_ended and not socket_path.exists():
            continue
        if not all(k in state for k in ('pid_start_ticks', 'boot_id', 'directory_inode', 'pid_namespace', 'uid')):
            raise RuntimeError(f'Cannot verify old owned AF3 identity: {path}')
        if not confirmed_ended and state['pid_namespace'] != Path('/proc/self/ns/pid').stat().st_ino:
            raise RuntimeError(f'Cannot verify old AF3 across PID namespaces: {path}')
        process = _process_identity(state['pid'])
        if (not confirmed_ended and state['boot_id'] == _boot_id() and process is not None and
                process['start_ticks'] == state['pid_start_ticks'] and process['state'] != 'Z'):
            raise RuntimeError(f'Previous owned AF3 is still active; inspect {path}')
        if not socket_path.exists():
            continue
        directory = socket_path.parent
        parent_info, info = directory.stat(), socket_path.lstat()
        if (socket_path.name != 'owner.sock' or not directory.name.startswith('astrolabe-af3-') or
                parent_info.st_uid != os.getuid() or state['uid'] != os.getuid() or
                parent_info.st_ino != state['directory_inode'] or
                (state.get('socket_inode') is not None and info.st_ino != state['socket_inode']) or
                stat.S_IMODE(parent_info.st_mode) != 0o700 or not stat.S_ISSOCK(info.st_mode)):
            raise RuntimeError(f'Old AF3 socket ownership changed; retained: {socket_path}')
        from astrolabe.evaluators.af3_buffers import cleanup_namespace
        retained = cleanup_namespace(socket_path)
        socket_path.unlink()
        buffers = directory / 'buffers'
        if buffers.exists():
            # Recovery proves owner death, not that every client has consumed
            # its output. Leave retained data discoverable, never replay it.
            try:
                buffers.rmdir()
            except OSError:
                pass
        if not buffers.exists():
            directory.rmdir()
        recovery = path.parent / 'recovery.json'
        if not recovery.exists():
            atomic_json(recovery, dict(owner_verified_ended=True, recovered_at=time.time(),
                                      lifecycle=str(path), removed_socket=str(socket_path),
                                      **({"retained_buffer_root": retained} if retained is not None else {}),
                                      original_lifecycle_preserved=True, predictions_replayed=0))


class OwnedAF3:
    """A serial model or process-pool owner; stop only the process created here."""
    def __init__(self, cfg, output, gpu_environment, *, settings=None):
        from omegaconf import OmegaConf
        from astrolabe.design.io import atomic_json
        self.service_settings = service_settings(cfg, explicit_settings=settings)
        require_private_evaluator_owner(self.service_settings.adaptive.enabled
                                        or self.service_settings.concurrency_mode == "resident",
                                        environment=gpu_environment)
        self.startup_timeout = self.service_settings.startup_timeout_s
        self.shutdown_timeout = self.service_settings.shutdown_timeout_s
        self.terminate_timeout = self.service_settings.terminate_timeout_s
        self.max_pending = self.service_settings.max_pending
        self.directory = Path(output) / 'services' / uuid.uuid4().hex
        self.directory.mkdir(parents=True)
        self.config_path = self.directory / 'owner.yaml'
        self.log_path = self.directory / 'owner.jsonl'
        self.diagnostics_path = self.directory / 'owner.stderr.log'
        self.state_path = self.directory / 'lifecycle.json'
        owner = OmegaConf.create(OmegaConf.to_container(cfg.evaluator.alphafold3, resolve=True))
        owner.service_socket = None
        self.transport = str(owner.get('service_transport', 'pickle'))
        if self.transport not in {'pickle', 'buffers'}:
            raise ValueError('AF3 service_transport must be pickle or buffers')
        owner._service_adaptive = asdict(self.service_settings.adaptive)
        owner._service_settings = asdict(self.service_settings)
        self.config_path.write_text(OmegaConf.to_yaml(owner))
        self.environment = dict(gpu_environment)
        # Native request preparation constructs RNG keys on CPU. Keep CUDA
        # first; load_runner still explicitly requires a real GPU device.
        self.environment.update(JAX_PLATFORMS='cuda,cpu', GH_CUDA_PREINITIALIZE='0',
                                PYTHONUNBUFFERED='1', ASTROLABE_AF3_PARENT_PID=str(os.getpid()))
        self.environment.pop('JAX_PLATFORM_NAME', None)
        self.python_bin = str(Path(sys.executable).resolve())
        self.library_directories = ()
        self.process = None
        self.log = None
        self.stderr = None
        self.socket_directory = None
        self.ready = False
        self.stopped = False
        self.state = dict(status='created', parent_pid=os.getpid(), log=str(self.log_path),
                          stdout_path=str(self.log_path), stderr_path=str(self.diagnostics_path),
                          started_at=time.time(), cuda_visible=self.environment.get('CUDA_VISIBLE_DEVICES'),
                          jax_platforms=self.environment['JAX_PLATFORMS'])
        self.state['adaptive'] = asdict(self.service_settings.adaptive)
        self.state['execution_mode'] = ('resident_model' if self.service_settings.concurrency_mode == 'resident'
                                        else 'independent_process_pool' if self.service_settings.adaptive.enabled
                                        else 'serial_model')
        if self.service_settings.adaptive.enabled:
            self.state.update(max_owner_count=self.service_settings.adaptive.max_concurrency,
                              pool_manifest_path=str(self.directory / 'process_pool.json'))
        self._write = lambda: atomic_json(self.state_path, self.state)
        self._write()

    def start(self):
        """Start this owner without blocking independent CPU initialization."""
        if self.process is not None or self.stopped:
            raise RuntimeError("Owned AF3 can only be started once")
        try:
            # Always short, private, unique; old-owner sockets remain distinguishable
            # on interrupted resume, even for very long project output paths.
            self.socket_directory = tempfile.mkdtemp(prefix='astrolabe-af3-')
            self.socket_path = Path(self.socket_directory) / 'owner.sock'
            command = _owner_command(self.config_path, self.socket_path)
            command += ['--max-pending', str(self.max_pending)]
            child_runtime = af3_child_environment(command[0], self.environment)
            self.environment = child_runtime.environment
            self.python_bin = child_runtime.python_bin
            self.library_directories = child_runtime.library_directories
            self.state.update(
                python_bin=self.python_bin,
                python_library_directories=list(self.library_directories),
            )
            self._write()
            self.log = self.log_path.open('ab', buffering=0)
            self.stderr = self.diagnostics_path.open('ab', buffering=0)
            self.process = subprocess.Popen(command, env=self.environment, stdout=self.log,
                                            stderr=self.stderr, start_new_session=True)
            self.state.update(status='starting', pid=self.process.pid, socket=str(self.socket_path), command=command,
                              pid_start_ticks=_process_identity(self.process.pid)['start_ticks'], boot_id=_boot_id(),
                              directory_inode=Path(self.socket_directory).stat().st_ino, uid=os.getuid(),
                              pid_namespace=Path('/proc/self/ns/pid').stat().st_ino)
            self._write()
            self._startup_deadline = time.monotonic() + self.startup_timeout
            return str(self.socket_path)
        except BaseException as error:
            self._startup_failed(error)
            raise

    def wait_ready(self):
        """Barrier before scientific work; never issue inference as a probe."""
        if self.process is None or self.stopped:
            raise RuntimeError("Owned AF3 is not starting")
        if self.ready:
            return str(self.socket_path)
        try:
            deadline = self._startup_deadline
            while time.monotonic() < deadline:
                code = self.process.poll()
                if code is not None:
                    raise RuntimeError(
                        f'AF3 owner exited during startup (exit {code}); '
                        f'events: {self.log_path}; diagnostics: {self.diagnostics_path}'
                    )
                # The native service emits readiness only after loading the model
                # and opening its listener. Never send a prediction as a health probe.
                for line in self.log_path.read_text(errors='replace').splitlines():
                    try:
                        event = json.loads(line)
                    except (ValueError, TypeError):
                        continue
                    if (event.get('event') == 'service_ready' and event.get('pid') == self.process.pid
                            and event.get('socket') == str(self.socket_path) and self.socket_path.is_socket()):
                        self.ready = True
                        self.state.update(status='ready', ready_at=event['ts'],
                                          ready_observed_at=time.time(),
                                          socket_inode=self.socket_path.stat().st_ino)
                        self._write()
                        return str(self.socket_path)
                time.sleep(.1)
            raise TimeoutError(
                f'AF3 owner did not become ready; events: {self.log_path}; '
                f'diagnostics: {self.diagnostics_path}'
            )
        except BaseException as error:
            self._startup_failed(error)
            raise

    def _startup_failed(self, error):
        self.state.update(status='startup_failed', error=repr(error))
        try:
            errors = self._stop(graceful=False)
            if errors:
                error.add_note(f'Owned AF3 startup cleanup unresolved: {errors}; see {self.state_path}')
        except BaseException as cleanup_error:
            error.add_note(f'Owned AF3 startup cleanup raised: {cleanup_error!r}; owner retained')

    def __enter__(self):
        self.start()
        return self.wait_ready()

    @contextmanager
    def starting(self):
        """Own startup/cleanup while the caller prepares independent CPU work."""
        socket_path = self.start()
        try:
            yield socket_path
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise
        else:
            self.__exit__(None, None, None)

    def _force_reap(self):
        if self.process is None or self.process.poll() is not None:
            return
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            self.process.wait(timeout=self.terminate_timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.process.wait(timeout=self.terminate_timeout)

    def _stop(self, graceful):
        if self.stopped:
            return self.state['cleanup_errors']
        errors, interrupted = [], None
        acknowledged = False
        with deferred_termination_signals():
            try:
                if self.process is not None and graceful:
                    if self.process.poll() is not None:
                        errors.append('AF3 owner ended before managed shutdown')
                    else:
                        result = {}
                        def shutdown():
                            try:
                                from astrolabe.evaluators.inference_service import call, json_call
                                invoke = json_call if self.transport == 'buffers' else call
                                invoke(self.socket_path, {'type': 'shutdown'})
                                result['acknowledged'] = True
                            except BaseException as error:
                                result['error'] = repr(error)
                        thread = threading.Thread(target=shutdown, daemon=True)
                        thread.start()
                        deadline = time.monotonic() + self.shutdown_timeout
                        thread.join(max(0, deadline-time.monotonic()))
                        acknowledged = result.get('acknowledged', False)
                        if not acknowledged:
                            errors.append(result.get('error', 'AF3 shutdown drain timed out'))
                        elif self.process.poll() is None:
                            try:
                                self.process.wait(timeout=max(.01, deadline-time.monotonic()))
                            except subprocess.TimeoutExpired:
                                errors.append('AF3 acknowledged drain but did not exit')
            except BaseException as error:
                interrupted = error
                errors.append(f'Cleanup interrupted: {error!r}')
            finally:
                try:
                    # This final fallback runs even when join/wait was interrupted.
                    self._force_reap()
                except BaseException as error:
                    errors.append(f'Owner reap failed: {error!r}')
                ended = self.process is None or self.process.poll() is not None
                if not ended:
                    errors.append('Owned AF3 is still alive; socket and evidence retained')
                if self.process is not None:
                    self.state['exit_code'] = self.process.returncode
                    if graceful and self.process.returncode != 0:
                        errors.append(f'AF3 owner exit {self.process.returncode}')
                for name, stream in (('stdout', self.log), ('stderr', self.stderr)):
                    if stream is not None:
                        try:
                            stream.close()
                        except BaseException as error:
                            errors.append(f'{name} close failed: {error!r}')
                if ended and self.socket_directory is not None:
                    try:
                        if self.transport == 'buffers':
                            from astrolabe.evaluators.af3_buffers import cleanup_namespace
                            retained = cleanup_namespace(self.socket_path)
                            if retained is not None:
                                self.state['retained_buffer_root'] = retained
                        self.socket_path.unlink(missing_ok=True)
                        directory = Path(self.socket_directory)
                        buffers = directory / 'buffers'
                        if buffers.exists():
                            try:
                                buffers.rmdir()
                            except OSError:
                                # Completed replies may not yet be mapped by a
                                # caller; EOF may be ambiguous. Never reap data
                                # merely because the service process exited.
                                self.state['retained_buffer_root'] = str(buffers)
                        if not buffers.exists():
                            directory.rmdir()
                    except OSError as error:
                        errors.append(f'Socket cleanup: {error!r}')
                self.state.update(shutdown_acknowledged=acknowledged, cleanup_errors=errors)
                if ended:
                    self.state['ended_at'] = time.time()
                    self.stopped = True
                if self.state['status'] != 'startup_failed':
                    self.state['status'] = 'drained' if ended and not errors and acknowledged else 'shutdown_failed'
                self._write()
            if interrupted is not None:
                raise interrupted
        return errors

    def abort(self):
        errors = self._stop(graceful=self.ready)
        if errors:
            raise RuntimeError(f'Owned AF3 abort: {errors}; see {self.state_path}')

    def __exit__(self, exc_type, error, traceback):
        errors = self._stop(graceful=self.ready)
        if errors:
            message = f'Owned AF3 cleanup failed: {errors}; see {self.state_path}'
            if error is None:
                raise RuntimeError(message)
            error.add_note(message)
        return False


class AF3ChildGroup(list):
    """Serial child lifetimes, started only when a real request needs a slot."""

    def ensure(self, index):
        # Linux PDEATHSIG follows the thread that spawned the process, not
        # merely its TGID. RPC threads end before child drain, so every spawn
        # must belong to a launcher thread that outlives the child ExitStack.
        return self.launcher.submit(self._start, index).result()

    def _start(self, index):
        with self.start_lock:
            child = self[index]
            if not child.ready:
                self.stack.enter_context(child)
                self.record()
            return dict(index=index, pid=child.process.pid, socket=str(child.socket_path),
                        lifecycle=str(child.state_path), config=str(child.config_path),
                        log=str(child.log_path), stderr=str(child.diagnostics_path))


@contextmanager
def owned_af3_children(config, directory, *, gpu_environment=None, lazy=False):
    """Own serial children without adding predictions or scientific work.

    Lazy mode loads owner0 before serving. Expansion is started by a real
    request in its forwarding thread, while owner0 continues useful work.
    Layout beside parent owner.yaml: process_pool.json and
    children/NN/services/UUID/{owner.yaml,owner.jsonl,owner.stderr.log,lifecycle.json}.
    """
    from omegaconf import OmegaConf
    from astrolabe.design.io import atomic_json
    settings = service_settings({}, explicit_settings=config.get('_service_settings'))
    adaptive = adaptive_settings(config.get('_service_adaptive'))
    environment = dict(os.environ if gpu_environment is None else gpu_environment)
    require_private_evaluator_owner(adaptive.enabled, environment=environment)
    if not adaptive.enabled:
        raise ValueError('Independent AF3 children require an adaptive pool owner')
    directory = Path(directory)
    manifest_path = directory / 'process_pool.json'
    metadata = dict(parent_pid=os.getpid(), parent_config=str(directory / 'owner.yaml'),
                    execution_mode='independent_process_pool', capacity_per_owner=1,
                    max_owner_count=adaptive.max_concurrency,
                    startup_policy='on_demand' if lazy else 'eager',
                    admission_scope='complete_evaluator_request', status='starting', children=[])
    children = AF3ChildGroup()
    children.start_lock = threading.Lock()
    def record():
        metadata['resident_owner_count'] = sum(child.ready and not child.stopped for child in children)
        metadata['children'] = [dict(index=i, pid=child.state.get('pid'),
                                     socket=child.state.get('socket'), config=str(child.config_path),
                                     log=str(child.log_path), stderr=str(child.diagnostics_path),
                                     lifecycle=str(child.state_path), status=child.state['status'])
                                for i, child in enumerate(children)]
        atomic_json(manifest_path, metadata)
    record()
    # Child OwnedAF3 copies the exact same AF3 mapping/cache, overriding only
    # transport metadata. max_pending=1 plus the serial kernel forbids overlap.
    cfg = OmegaConf.create({'evaluator': {'alphafold3': OmegaConf.to_container(config, resolve=True)}})
    serial = replace(settings, max_pending=1,
                     adaptive=replace(adaptive, enabled=False, max_concurrency=1))
    try:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix='af3-owner-launch') as launcher, ExitStack() as stack:
            children.launcher = launcher
            children.stack, children.record = stack, record
            for index in range(adaptive.max_concurrency):
                child = OwnedAF3(cfg, directory / 'children' / f'{index:02d}', environment, settings=serial)
                child.state.update(pool_index=index, pool_parent_pid=os.getpid(),
                                   pool_parent_config=str(directory / 'owner.yaml'))
                child._write()
                children.append(child)
            for index in range(1 if lazy else len(children)):
                children.ensure(index)
            metadata['status'] = 'ready'
            record()
            try:
                yield children
            finally:
                metadata['status'] = 'draining'
                record()
        metadata['status'] = 'drained'
    except BaseException as error:
        metadata.update(status='failed', error=repr(error))
        raise
    finally:
        record()
