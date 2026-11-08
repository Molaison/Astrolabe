"""Execution policy and random-state transfer for native model kernels."""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
import logging
import hashlib
import json
import importlib.metadata
import importlib.util
import os
import random
import sys
import traceback
import threading
import time
import uuid
from pathlib import Path
import subprocess
from typing import Any, Callable


_RUN_OWNERS = ContextVar("astrolabe_model_run_owners", default=None)
DEFAULT_MODEL_FRAME_BYTES = 256 * 1024 * 1024
_REMOTE_CUDA_STATE = None
_INITIALIZATION_RANDOM_STATE = ContextVar("astrolabe_initialization_random_state", default=None)
_STARTUP_SCOPE = ContextVar("astrolabe_startup_scope", default=None)
_CALLER_RANDOM_LOCK = threading.RLock()
_KERNEL_CONTEXT = ContextVar("astrolabe_kernel_context", default=None)
_ACTIVE_DEVICE_LEASE = ContextVar("astrolabe_active_device_lease", default=None)
# Python, NumPy and Torch default generators are process global. A transport
# worker is not an independent RNG owner.
_KERNEL_RANDOM_LOCK = threading.RLock()


class StartupScope:
    """Retain shared cleanup receipts across startup, construction and run scopes."""

    def __init__(self, parent=None):
        self.lock = threading.RLock()
        self.cancelled = False
        self.owners = []
        self._cleanups = {}
        self._parent = parent
        self.first_error = None

    def _find_cleanup(self, owner):
        with self.lock:
            cleanup = self._cleanups.get(id(owner))
        if cleanup is None and self._parent is not None:
            return self._parent._find_cleanup(owner)
        return cleanup

    @property
    def cleanup_report(self):
        from astrolabe.runtime_support import CloseReport
        with self.lock:
            cleanups = list(self._cleanups.values())
        reports = [cleanup.report for cleanup in cleanups]
        return CloseReport(tuple(record for report in reports for record in report.records),
                           any(report.draining for report in reports))

    @property
    def stopped(self):
        return {id(record.resource) for record in self.cleanup_report.records
                if record.state == "closed"}

    @property
    def cleanup_errors(self):
        return [record.error for record in self.cleanup_report.failed]

    def add(self, owner, *, source=None):
        from astrolabe.runtime_support import ResourceOwner
        if owner is None:
            return
        # Look up shared receipts before taking the lifecycle lock. Never invoke
        # shutdown under this lock, including registration after cancellation.
        inherited = (source or self)._find_cleanup(owner)
        with self.lock:
            cleanup = self._cleanups.get(id(owner))
            if cleanup is None:
                cleanup = inherited
                if cleanup is None:
                    cleanup = ResourceOwner()
                    cleanup.add(owner, type(owner).__name__)
                self._cleanups[id(owner)] = cleanup
                self.owners.append(owner)
            cancelled = self.cancelled
        if cancelled:
            report = cleanup.close()
            error = RuntimeError("evaluator startup was cancelled" +
                                 ("; owner cleanup failed" if report.failed else ""))
            self.retain_errors(error)
            if report.failed:
                raise error from report.failed[0].error
            raise error

    def retain_errors(self, error):
        """Keep failed/in-flight ownership reachable through the original error."""
        report = self.cleanup_report
        error.cleanup_report = report
        scopes = getattr(error, "cleanup_scopes", ())
        if not any(scope is self for scope in scopes):
            error.cleanup_scopes = (*scopes, self)
        for record in report.failed:
            error.add_note(f"model owner cleanup failed: {record.error!r}")

    def cancel(self, *, retry_failed=False):
        """Retry only when the caller asserts adapter shutdown is idempotent."""
        from astrolabe.runtime_support import cleanup_pass
        with self.lock:
            self.cancelled = True
            cleanups = [self._cleanups[id(owner)] for owner in reversed(self.owners)]
        with cleanup_pass():
            for cleanup in cleanups:
                cleanup.close(retry_failed=retry_failed)
        if self.first_error is not None and not self.cleanup_report.complete:
            self.retain_errors(self.first_error)
        return self.cleanup_errors


@contextmanager
def startup_scope():
    scope = StartupScope()
    token = _STARTUP_SCOPE.set(scope)
    try:
        yield scope
    finally:
        _STARTUP_SCOPE.reset(token)


def _identity(value):
    return value


def _process_epoch(pid):
    """Linux start ticks distinguish a retained model PID from later reuse."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def _request_evidence(event, owner, metadata, **fields):
    """Optional host-only association of consumers, owners and native stages."""
    target = os.environ.get("ASTROLABE_MODEL_REQUEST_TRACE_PATH")
    if not target:
        return
    lifecycle = metadata.get("runtime_lifecycle", {})
    record = {
        "schema": "astrolabe.model-request-evidence/v1", "event": event,
        "time": time.time(), "client_pid": os.getpid(),
        "client_epoch": _process_epoch(os.getpid()),
        "model_pid": owner.pid, "model_epoch": _process_epoch(owner.pid),
        "identity": owner.identity, "model_id": lifecycle.get("model_id"),
        "attached": bool(getattr(owner, "attached", False)),
        "broker_socket": os.environ.get("ASTROLABE_RESOURCE_SOCKET"),
        **fields,
    }
    data = (json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
    try:
        if os.write(descriptor, data) != len(data):
            raise OSError("Incomplete model request evidence")
    finally:
        os.close(descriptor)


def resource_shape(value):
    """Stable, conservative request scope without transferring tensor values.

    Array shape/dtype define device workloads; file-backed structural inputs
    include their content digest. Scalar controls and sequence strings remain
    in scope. This is an admission/profile key, never a feature-cache key.
    """
    def describe(item):
        if hasattr(item, "shape") and hasattr(item, "dtype"):
            return {"shape": list(item.shape), "dtype": str(item.dtype)}
        if isinstance(item, Mapping):
            return {str(key): describe(value) for key, value in sorted(
                item.items(), key=lambda pair: str(pair[0]))}
        if isinstance(item, (tuple, list)):
            return [describe(value) for value in item]
        if isinstance(item, (str, Path)):
            text = str(item)
            if text.lower().endswith((".pdb", ".cif", ".npz", ".npy")):
                path = Path(text)
                if path.is_file():
                    digest = hashlib.sha256()
                    with path.open("rb") as handle:
                        for block in iter(lambda: handle.read(1024 * 1024), b""):
                            digest.update(block)
                    return {"file_sha256": digest.hexdigest()}
            return text
        if item is None or isinstance(item, (bool, int, float)):
            return item
        # Opaque objects are never eligible for a portable resource profile.
        raise TypeError(f"Cannot describe resource scope for {type(item).__name__}")
    raw = json.dumps(describe(value), sort_keys=True, allow_nan=False,
                     separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _verify_asset_hashes(values):
    for path, expected in values.items():
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"Resource profile asset changed: {path}")


def _profile_device(settings):
    """Resolve only unambiguous device declarations for production evidence."""
    def normalize(value):
        if value is None or str(value).lower() in {"auto", "none"}:
            return None
        if isinstance(value, bool):
            raise ValueError("Resource profile GPU device cannot be boolean")
        value = str(value).lower().replace("gpu:", "cuda:")
        if value in {"cuda", "gpu"}:
            return "cuda:0"
        if value.isdigit():
            return f"cuda:{int(value)}"
        if value.startswith("cuda:") and value[5:].isdigit():
            return f"cuda:{int(value[5:])}"
        raise ValueError("Resource profile cannot resolve selected model device")

    declarations = []
    scopes = [settings] + [settings.get(key) for key in (
        "runner_config", "model_config", "config", "factory_kwargs", "constructor")]
    for scope in scopes:
        if not isinstance(scope, Mapping):
            continue
        # Adapters prefer a non-None gpu_device. Refuse conflicting declarations
        # rather than certifying a different adapter's field precedence.
        gpu_device, device = scope.get("gpu_device"), scope.get("device")
        if gpu_device is not None:
            # A non-None "auto"/"none" still takes precedence in adapters and
            # selects their default card; it does not expose the device field.
            selected = normalize(gpu_device) or "cuda:0"
            other = normalize(device)
            if other is not None and other != selected:
                raise ValueError("Resource profile has conflicting configured GPU devices")
            declarations.append(selected)
        elif device is not None:
            declarations.append(normalize(device) or "cuda:0")
    if len(set(declarations)) > 1:
        raise ValueError("Resource profile has conflicting configured GPU devices")
    return declarations[0] if declarations else "cuda:0"


def _validate_production_profile(profile, backend, settings, factory, cwd=None):
    """Reject incomplete certificates before loading any scientific model."""
    if profile.get("verified") is not True:
        raise ValueError("Automatic resource profiles require verified evidence")
    assets = profile.get("asset_sha256")
    weights = profile.get("weights_sha256")
    shapes = profile.get("shapes")
    hardware = profile.get("hardware")
    environment = profile.get("environment")
    if (not isinstance(assets, dict) or not assets
            or not isinstance(weights, dict) or not weights
            or not isinstance(shapes, list) or not shapes
            or any(not isinstance(shape, str) or len(shape) != 64 for shape in shapes)
            or not isinstance(hardware, dict)
            or not isinstance(environment, dict)):
        raise ValueError("Verified resource profile needs source, weight, shape and hardware evidence")
    configured_weights = set()

    def collect(value, key=""):
        if isinstance(value, Mapping):
            for name, item in value.items():
                collect(item, str(name).lower())
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item, key)
        elif isinstance(value, str) and any(
                token in key for token in ("checkpoint", "ckpt", "weight", "model_path")):
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = Path(cwd or os.getcwd()) / path
            if path.is_file():
                configured_weights.add(str(path.resolve()))
    collect(settings)
    certified_weights = {str(Path(path).expanduser().resolve()) for path in weights}
    if not configured_weights or not configured_weights <= certified_weights:
        raise ValueError("Verified resource profile does not identify the selected weight contents")
    factory_module = factory.split(":", 1)[0]
    module_spec = importlib.util.find_spec(factory_module)
    required_source = {Path(__file__).resolve(),
                       Path(__file__).with_name("model_daemon.py").resolve(),
                       Path(__file__).parent / "runtime/admission.py",
                       Path(__file__).parent / "runtime/session.py"}
    if module_spec is None or not module_spec.origin:
        raise ValueError("Cannot resolve selected model factory implementation")
    required_source.add(Path(module_spec.origin).resolve())
    if not required_source <= {Path(path).resolve() for path in assets}:
        raise ValueError("Resource profile lacks the actual selected source evidence")
    _verify_asset_hashes(assets)
    _verify_asset_hashes(weights)
    if environment.get("python_abi") != sys.implementation.cache_tag:
        raise ValueError("Resource profile Python runtime changed")
    distributions = environment.get("distributions", {})
    framework = "jax" if backend in {"alphafold2", "alphafold3"} else "torch"
    if not isinstance(distributions, dict) or framework not in distributions:
        raise ValueError("Resource profile lacks the selected framework runtime")
    for name, expected in distributions.items():
        if importlib.metadata.version(name) != expected:
            raise ValueError(f"Resource profile runtime changed: {name}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and not visible.strip():
        raise ValueError("Resource profile has no visible GPU allocation")
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,driver_version",
         "--format=csv,noheader,nounits"], text=True, timeout=5,
    )
    devices = [tuple(part.strip() for part in row.split(",")) for row in rows.splitlines()]
    matched = [row for row in devices if row[1] == hardware.get("gpu_uuid")
               and row[2] == hardware.get("driver_version")]
    if len(matched) != 1:
        raise ValueError("Resource profile hardware or driver changed")
    if visible is not None:
        tokens = [token.strip() for token in visible.split(",")]
        if not any(token == matched[0][0] or matched[0][1].startswith(token)
                   for token in tokens if token):
            raise ValueError("Resource profile GPU is outside this allocation")
    # A certificate for another visible GPU must not authorize this model.
    logical = _profile_device(settings)
    ordinal = int(logical[5:])
    selected = ([token.strip() for token in visible.split(",")]
                if visible is not None else [row[0] for row in devices])
    if ordinal >= len(selected):
        raise ValueError("Resource profile model device is outside this allocation")
    token = selected[ordinal]
    if token != matched[0][0] and not matched[0][1].startswith(token):
        raise ValueError("Resource profile does not match the model's selected GPU")


@dataclass(frozen=True)
class KernelStages:
    """A backend's proven CPU/device boundary, without changing its algorithm.

    ``resource`` is evaluated against the original request, so native feature
    operations can remain CPU work. An unclassified callable remains one
    conservative device operation, rather than claiming a fictitious split.
    """

    prepare: Callable[[Any], Any]
    execute: Callable[[Any], Any]
    finalize: Callable[[Any], Any] = _identity
    resource: str | Callable[[Any], str] = "gpu"
    # A backend may overlap host stages only after proving they neither touch
    # default generators nor mutate shared model/request state. Device execution
    # keeps its native RNG critical section. This is not a GPU concurrency
    # certificate and does not alter the broker's admission limits.
    pipeline_slots: int = 1
    rng_free_host_stages: bool = False


@dataclass(frozen=True)
class EvaluatorStages(KernelStages):
    """Strict model boundary, retaining native outputs until finalize.

    Prepare includes feature construction and input transfer. Execute contains
    the complete model algorithm and must return a Pending completion (or an
    already completed value). Finalize owns native reductions and output
    materialization. GPU prepare/finalize borrow capacity, never an inference
    slot. All request memory is retained from before prepare through finalize.

    Default-generator backends remain serial. This declaration alone is not a
    certificate for concurrent inference or RNG-free host work.
    """

    prepare_resource: str = "gpu"
    finalize_resource: str = "gpu"
    # AF3's immutable model receives an explicit JAX key for every request.
    # No other backend may opt out of the process-global RNG critical section.
    request_local_rng: bool = False
    adaptive: Any = None


@contextmanager
def model_kernel_context(backend, identity):
    token = _KERNEL_CONTEXT.set({"backend": backend, "identity": identity})
    try:
        yield
    finally:
        _KERNEL_CONTEXT.reset(token)


def _resource_client():
    if not os.environ.get("ASTROLABE_RESOURCE_SOCKET"):
        return None
    from astrolabe.runtime.admission import ResourceClient

    return ResourceClient.from_env()


@contextmanager
def device_operation(backend, *, phase="compute", identity=None, shape="",
                     exclusive=True):
    """Fence an unconverted GPU/CLI subgraph, including its child processes."""
    client = _resource_client()
    if client is None:
        yield None
        return
    active = _ACTIVE_DEVICE_LEASE.get()
    if active is not None and active[0] == str(client.socket_path):
        # An already admitted native device subgraph can call guarded legacy
        # helpers locally. It still holds the outer conservative device lease.
        yield active[1]
        return
    with client.lease(
        resource="gpu", backend=backend, identity=identity or backend,
        shape=shape, phase=phase, concurrency_certified=not exclusive,
    ) as lease:
        token = _ACTIVE_DEVICE_LEASE.set((str(client.socket_path), lease))
        try:
            yield lease
        finally:
            _ACTIVE_DEVICE_LEASE.reset(token)


def _complete_device(value):
    # Keep a lease until actual completion, including any backend-owned buffers.
    from astrolabe.inference_pipeline import Pending

    if isinstance(value, Pending):
        completion = value.completion
        if hasattr(completion, "result"):
            completion.result()
        else:
            completion()
        value = value.value
    return host_result(value)


def _complete_native(value):
    """Wait without D2H, dtype changes, or walking a backend's raw output."""
    from astrolabe.inference_pipeline import Pending

    if isinstance(value, Pending):
        completion = value.completion
        if hasattr(completion, "result"):
            completion.result()
        else:
            completion()
        return value.value
    return value


def reset_remote_random_state() -> None:
    """Begin a new CUDA stream when the scientific coordinator is reseeded."""
    global _REMOTE_CUDA_STATE
    _REMOTE_CUDA_STATE = None


@contextmanager
def preserve_remote_cuda_state():
    """Preserve the opaque CUDA receipt around a native restoring seed scope.

    Vendor fork/restore contexts restore actual Python/NumPy/Torch generators.
    A temporary manual_seed also passes through our CUDA seed recorder before
    Torch changes its CPU seed, so restoring the CPU generator alone does not
    restore this receipt. This helper changes no native generator.
    """
    global _REMOTE_CUDA_STATE
    previous = _REMOTE_CUDA_STATE
    try:
        yield
    finally:
        _REMOTE_CUDA_STATE = previous


def register_model_owner(owner) -> None:
    from astrolabe.runtime.model_pool import pool_owns_registration
    if pool_owns_registration(owner):
        return
    scope = _STARTUP_SCOPE.get()
    if scope is not None and not getattr(owner, "attached", False):
        scope.add(owner)
    owners = _RUN_OWNERS.get()
    if owners is not None and not getattr(owner, "attached", False):
        owners.add(owner, source=scope)


@contextmanager
def model_run_scope():
    owners = StartupScope(parent=_RUN_OWNERS.get())
    token = _RUN_OWNERS.set(owners)
    active_error = None
    try:
        yield owners
    except BaseException as error:
        active_error = error
        raise
    finally:
        try:
            errors = owners.cancel()
        finally:
            _RUN_OWNERS.reset(token)
        if errors or not owners.cleanup_report.complete:
            if active_error is not None:
                owners.retain_errors(active_error)
            else:
                error = RuntimeError("A model owner failed to stop after the run")
                owners.retain_errors(error)
                raise error from (errors[0] if errors else None)


class ExecutionMode(str, Enum):
    DAEMON = "daemon"
    IN_PROCESS = "in_process"


def execution_mode(value=None) -> ExecutionMode:
    if value is None:
        value = os.environ.get("ASTROLABE_MODEL_EXECUTION", "daemon")
    try:
        return ExecutionMode(value)
    except ValueError as error:
        raise ValueError("Model execution must be 'daemon' or 'in_process'.") from error


def execution_directory(cwd=None) -> str:
    return str(Path(cwd).expanduser().resolve() if cwd else Path(__file__).resolve().parents[1])


def capture_random_state() -> dict:
    state = {"python": random.getstate()}
    numpy = sys.modules.get("numpy")
    if numpy is not None:
        algorithm, keys, position, has_gauss, cached_gauss = numpy.random.get_state()
        state["numpy"] = [algorithm, keys.tolist(), position, has_gauss, cached_gauss]
    torch = sys.modules.get("torch")
    if torch is not None:
        state["torch"] = torch.get_rng_state().tolist()
        state["torch_seed"] = torch.initial_seed()
        if torch.cuda.is_initialized():
            state["cuda"] = [item.tolist() for item in torch.cuda.get_rng_state_all()]
            state["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        elif _REMOTE_CUDA_STATE is not None:
            pending = _REMOTE_CUDA_STATE
            if pending["cpu_seed"] == state["torch_seed"]:
                if pending["states"] is not None:
                    state["cuda"] = [list(item) for item in pending["states"]]
                    state["cuda_visible_devices"] = pending["visible_devices"]
                else:
                    state["cuda_seed"] = pending["seed"]
            else:
                state["cuda_seed"] = state["torch_seed"]
        else:
            state["cuda_seed"] = state["torch_seed"]
    return state


def restore_random_state(state: Mapping, *, defer_cuda=False) -> None:
    global _REMOTE_CUDA_STATE
    if "python" in state:
        version, internal, gaussian = state["python"]
        random.setstate((version, tuple(internal), gaussian))
    if "numpy" in state:
        import numpy as np
        algorithm, keys, position, has_gauss, cached_gauss = state["numpy"]
        np.random.set_state((algorithm, np.asarray(keys, dtype=np.uint32),
                             position, has_gauss, cached_gauss))
    if "torch" in state:
        import torch
        torch.set_rng_state(torch.tensor(state["torch"], dtype=torch.uint8))
        _REMOTE_CUDA_STATE = {
            "cpu_seed": state["torch_seed"],
            "seed": state.get("cuda_seed", state["torch_seed"]),
            "states": [list(item) for item in state["cuda"]] if "cuda" in state else None,
            "visible_devices": state.get("cuda_visible_devices"),
        }
        if defer_cuda or not torch.cuda.is_initialized():
            return
        elif "cuda" in state:
            if ("cuda_visible_devices" in state and
                    state["cuda_visible_devices"] != os.environ.get("CUDA_VISIBLE_DEVICES")):
                raise ValueError("CUDA RNG continuation requires the same CUDA_VISIBLE_DEVICES mapping")
            torch.cuda.set_rng_state_all([
                torch.tensor(item, dtype=torch.uint8) for item in state["cuda"]
            ])
        else:
            torch.cuda.manual_seed_all(state.get("cuda_seed", state["torch_seed"]))


@contextmanager
def _kernel_random_scope(state, torch_device, *, defer_cuda=False):
    global _REMOTE_CUDA_STATE
    if torch_device is not None:
        import torch
    else:
        torch = sys.modules.get("torch")
    restore_random_state(state, defer_cuda=defer_cuda)
    if torch is not None and torch_device is not None:
        device = str(torch_device)
        if device == "auto":
            uses_cuda = torch.cuda.is_available()
        elif device == "cpu":
            uses_cuda = False
        elif device.startswith("cuda"):
            uses_cuda = True
        else:
            raise ValueError(f"Unsupported kernel torch_device {device!r}")
        if uses_cuda:
            if ("cuda_visible_devices" in state and
                    state["cuda_visible_devices"] != os.environ.get("CUDA_VISIBLE_DEVICES")):
                raise ValueError("CUDA RNG continuation requires the same CUDA_VISIBLE_DEVICES mapping")
            torch.cuda.init()
            restore_random_state(state)
    if torch is None:
        yield
        return
    original_seed = torch.cuda.manual_seed_all
    def record_seed(seed):
        global _REMOTE_CUDA_STATE
        _REMOTE_CUDA_STATE = {
            "cpu_seed": torch.initial_seed(), "seed": int(seed),
            "states": None, "visible_devices": None,
        }
        return original_seed(seed)
    torch.cuda.manual_seed_all = record_seed
    try:
        yield
    finally:
        torch.cuda.manual_seed_all = original_seed


def host_result(value):
    if isinstance(value, Mapping):
        return {key: host_result(item) for key, item in value.items()}
    if isinstance(value, tuple):
        items = [host_result(item) for item in value]
        return type(value)(*items) if hasattr(value, "_fields") else tuple(items)
    if isinstance(value, list):
        return [host_result(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return replace(value, **{field.name: host_result(getattr(value, field.name))
                                 for field in fields(value) if field.init})
    module = type(value).__module__
    if module.startswith("torch") and hasattr(value, "detach"):
        return value.detach().cpu()
    if module.startswith(("jax", "jaxlib")) and hasattr(value, "shape"):
        import numpy as np
        return np.asarray(value)
    if module.startswith("pyrosetta"):
        raise TypeError("PyRosetta objects belong to the scientific coordinator.")
    return value


def kernel_handler(settings: dict, loader, *, torch_device=None, lifecycle=None):
    """Load once; Torch CUDA kernels declare 'auto' or their configured device.

    CPU and JAX kernels carry CUDA state opaquely. Default-generator operations
    are serialized; explicitly qualified host stages can overlap those sections.
    """
    context = dict(_KERNEL_CONTEXT.get() or {})
    pipeline_enabled = os.environ.get("ASTROLABE_MODEL_PIPELINE", "0")
    if pipeline_enabled not in {"0", "1"}:
        raise ValueError("ASTROLABE_MODEL_PIPELINE must be 0 or 1")
    declaration = dict(lifecycle or {})
    backend = context.get("backend", declaration.get("backend", loader.__module__))
    identity = context.get("identity", backend)
    profile = dict(settings.get("_runtime_profile") or {})
    if profile:
        if profile.get("identity") != identity:
            raise ValueError("Resource profile belongs to a different exact model identity")
        for name in ("resident_bytes", "scratch_bytes", "ram_bytes",
                     "resident_ram_bytes", "scratch_ram_bytes", "output_bytes"):
            if name not in profile:
                continue
            value = profile.get(name, 0)
            if type(value) is not int or value < 0:
                raise ValueError(f"Resource profile {name} must be a nonnegative integer")
            declaration[name] = value
        certified = profile.get("concurrency_certified", False)
        if type(certified) is not bool:
            raise ValueError("concurrency_certified must be boolean")
        declaration["concurrency_certified"] = certified
    loader_settings = {key: value for key, value in settings.items()
                       if key != "_runtime_profile"}
    from astrolabe.runtime.evaluation import require_private_evaluator_owner
    require_private_evaluator_owner(declaration.get("request_local_rng", False))
    client = _resource_client()
    require_private_evaluator_owner(declaration.get("request_local_rng", False), broker=client)
    model_id = f"{identity}:{os.getpid()}:{uuid.uuid4().hex}"
    initial_state = settings.get("_initial_random_state") or capture_random_state()
    cpu_only = torch_device == "cpu" or declaration.get("resource") == "cpu"
    device_hint = declaration.get("device")
    if device_hint is None and str(torch_device).startswith("cuda:"):
        device_hint = str(torch_device)
    resident_ram = int(declaration.get("resident_ram_bytes",
                                      declaration.get("ram_bytes", 0)))
    scratch_ram = int(declaration.get("scratch_ram_bytes", 0))

    @contextmanager
    def lease(resource, phase, shape="", profile_active=True, concurrency_allowed=True,
              prepared_model_id=None, reservation=None, capacity_only=False,
              evaluator_inference=False):
        if client is None:
            yield None
            return
        borrowed = reservation is not None
        with client.lease(
            resource=resource, backend=backend, identity=identity, shape=shape,
            phase=phase,
            resident_bytes=(int(declaration.get("resident_bytes", 0))
                            if resource == "gpu" and phase == "load" else 0),
            scratch_bytes=(int(declaration.get("scratch_bytes", 0))
                           if resource == "gpu" and profile_active and not borrowed else 0),
            ram_bytes=(0 if prepared_model_id is not None or borrowed else
                       resident_ram + scratch_ram if phase == "load" else scratch_ram),
            output_bytes=0 if borrowed else int(declaration.get("output_bytes", 0)),
            concurrency_certified=(profile_active and concurrency_allowed and
                                   bool(declaration.get("concurrency_certified", False))),
            device=device_hint if resource == "gpu" else None,
            **({"prepared_model_id": prepared_model_id}
               if prepared_model_id is not None else {}),
            **({"reservation_token": reservation.token,
                "evaluator_inference": evaluator_inference and resource == "gpu"}
               if borrowed else {}),
            **({"capacity_only": True} if capacity_only else {}),
        ) as acquired:
            if resource != "gpu" or capacity_only:
                yield acquired
                return
            certified_device = (profile.get("hardware") or {}).get("gpu_uuid")
            if certified_device and acquired.device != certified_device:
                # No backend/device operation has run yet. Complete this empty
                # lease before rejecting the physical admission mismatch.
                finish = getattr(acquired, "finish", None)
                if callable(finish):
                    finish()
                raise ValueError("Resource profile does not match the admitted physical GPU")
            token = _ACTIVE_DEVICE_LEASE.set((str(client.socket_path), acquired))
            try:
                yield acquired
            finally:
                _ACTIVE_DEVICE_LEASE.reset(token)

    from astrolabe.runtime.loading import StagedLoader

    def register_loaded(metadata, prepared_model_id=None):
        if client is not None:
            reported_device = metadata.get("device")
            registration_device = "cpu" if cpu_only else device_hint
            if (registration_device is None and isinstance(reported_device, str)
                    and (reported_device == "cpu" or reported_device.startswith(
                        ("cuda", "GPU-")))):
                registration_device = reported_device
            client.register_model(
                # JAX display names (e.g. TFRT_GPU_0) are diagnostics, not
                # physical GPU identifiers. The current load lease remains
                # authoritative when no configured logical device is supplied.
                model_id, device=registration_device,
                resident_bytes=int(declaration.get("resident_bytes", 0)),
                resident_ram_bytes=resident_ram,
                metadata={"backend": backend, "identity": identity,
                          "residency": declaration.get("residency", "unknown")},
                **({"prepared_model_id": prepared_model_id}
                   if prepared_model_id is not None else {}),
            )

    started = time.perf_counter()
    initialization_transition = None
    load_stages = {}
    staged_loading = isinstance(loader, StagedLoader) and resident_ram + scratch_ram > 0
    if staged_loading:
        # CPU construction does not monopolize the physical GPU. Its prepared
        # weights remain charged to RAM while waiting for device placement.
        cpu_model_id = f"{identity}:prepared:{os.getpid()}:{uuid.uuid4().hex}"
        begin = time.perf_counter()
        with lease("cpu", "load"):
            admitted = time.perf_counter()
            with _kernel_random_scope(initial_state, "cpu", defer_cuda=True):
                prepared = loader.prepare(loader_settings)
                prepared_state = capture_random_state()
            if client is not None:
                client.register_model(
                    cpu_model_id, device="cpu",
                    resident_ram_bytes=resident_ram + scratch_ram,
                    metadata={"backend": backend, "identity": identity,
                              "residency": "prepared_cpu_weights"},
                )
        load_stages["cpu_prepare"] = {
            "wait_seconds": admitted - begin,
            "run_seconds": time.perf_counter() - admitted,
        }
        begin = time.perf_counter()
        with lease("cpu" if cpu_only else "gpu", "load",
                   prepared_model_id=cpu_model_id):
            admitted = time.perf_counter()
            with _kernel_random_scope(prepared_state, torch_device):
                # This observed CUDA state, before placement, proves whether a
                # later initialized caller matches the first lazy seed. Reuse
                # never guesses CUDA state from the seed alone.
                cuda_before_place = capture_random_state()
                infer, metadata = loader.place(prepared)
                if not cpu_only and torch_device is not None:
                    import torch
                    if torch.cuda.is_initialized():
                        torch.cuda.synchronize()
            register_loaded(metadata, prepared_model_id=cpu_model_id)
        load_stages["device_place"] = {
            "wait_seconds": admitted - begin,
            "run_seconds": time.perf_counter() - admitted,
        }
        initialization_transition = {
            "before": initial_state, "cuda_before_place": cuda_before_place,
        }
        del prepared
    else:
        with lease("cpu" if cpu_only else "gpu", "load"):
            with _kernel_random_scope(initial_state, torch_device):
                if isinstance(loader, StagedLoader):
                    # Unknown host-memory needs retain the original conservative
                    # loading lease until a measured prepare envelope exists.
                    infer, metadata = loader.place(loader.prepare(loader_settings))
                else:
                    infer, metadata = loader(loader_settings)
            register_loaded(metadata)
    load_seconds = time.perf_counter() - started
    initialized_state = capture_random_state()
    if isinstance(infer, EvaluatorStages) and type(infer.request_local_rng) is not bool:
        raise ValueError("request_local_rng must be boolean")
    request_local_rng = isinstance(infer, EvaluatorStages) and infer.request_local_rng
    require_private_evaluator_owner(request_local_rng, broker=client)
    controller = None
    if request_local_rng:
        from astrolabe.runtime.evaluation import EvaluatorInferController, adaptive_settings
        adaptive = adaptive_settings(infer.adaptive)
        if backend != "alphafold3" or torch_device is not None:
            raise ValueError("Request-local RNG requires an immutable AF3 owner")
        from astrolabe.evaluators.inference_service import mark
        controller = EvaluatorInferController(adaptive, device=device_hint or "cuda:0", emit=mark)
    pipeline_slots = 1
    if isinstance(infer, KernelStages):
        if type(infer.pipeline_slots) is not int or infer.pipeline_slots < 1:
            raise ValueError("Kernel pipeline_slots must be a positive integer")
        if type(infer.rng_free_host_stages) is not bool:
            raise ValueError("rng_free_host_stages must be boolean")
        if (pipeline_enabled == "1" and infer.rng_free_host_stages
                and not isinstance(infer, EvaluatorStages)):
            pipeline_slots = infer.pipeline_slots
    status_lock = threading.RLock()
    pipeline_limit = threading.BoundedSemaphore(pipeline_slots)
    active_requests = 0
    metadata = dict(metadata)
    if isinstance(loader, StagedLoader):
        metadata["split_loading"] = staged_loading
        metadata["split_loading_reason"] = (
            "declared_prepared_ram" if staged_loading else
            "unknown_prepared_ram_conservative_load")
    if initialization_transition is not None:
        initialization_transition["after"] = initialized_state
        metadata["initialization_transition"] = initialization_transition
        metadata["load_admission_stages"] = load_stages
    status = {
        "backend": backend, "identity": identity, "model_id": model_id,
        "pid": os.getpid(), "epoch": model_id.rsplit(":", 1)[1],
        "process_alive": True, "loader_count": 1,
        "residency": declaration.get("residency", "unknown"),
        "weights_loaded": declaration.get("weights_loaded"),
        "gpu_resident": declaration.get("gpu_resident"),
        "device": metadata.get("device", declaration.get("device")),
        "load_seconds": load_seconds, "requests": 0, "failures": 0,
        "split_stages": isinstance(infer, KernelStages),
        "state": "ready",
        "resource_profile": profile,
        "pipeline_slots": pipeline_slots,
        "device_execution_slots": 1,
        "pipeline_contract": "rng_free_host_stages_with_serial_request_rng",
        "active_requests": 0,
        "peak_active_requests": 0,
    }
    if isinstance(infer, EvaluatorStages):
        for resource in (infer.prepare_resource, infer.finalize_resource):
            if resource not in {"cpu", "gpu"}:
                raise ValueError(f"Unknown evaluator stage resource {resource!r}")
        status["stage_contract"] = "prepare_complete_model_finalize"
        status["native_outputs_until_finalize"] = True
    if controller is not None:
        status["pipeline_contract"] = "af3_request_local_jax_rng"
        status["device_execution_slots"] = controller.limit

    def describe():
        # Return a detached snapshot: a retained describe result is not live state.
        with status_lock:
            refresh_lifecycle()
            result = dict(metadata, runtime_lifecycle=dict(status))
        if controller is not None:
            result["adaptive"] = controller.snapshot()
            result["runtime_lifecycle"]["device_execution_slots"] = result["adaptive"]["limit"]
        return result

    def refresh_lifecycle():
        for key in ("model_state", "component_state"):
            facts = metadata.get(key)
            if isinstance(facts, dict):
                for fact in ("weights_loaded", "gpu_resident", "model_load_count",
                             "cached_components"):
                    if fact in facts:
                        status[fact] = facts[fact]

    def run_stage(name, resource, function, value, timings, shape, profile_active,
                  concurrency_allowed=True, reservation=None, strict=False,
                  device_state=None):
        begin = time.perf_counter()
        evidence = bool(os.environ.get("ASTROLABE_MODEL_REQUEST_TRACE_PATH"))
        wall_begin = time.time() if evidence else None
        invoked = False
        try:
            with lease(resource, name, shape, profile_active, concurrency_allowed,
                       reservation=reservation,
                       evaluator_inference=strict and name == "infer"):
                admitted = time.perf_counter()
                wall_admitted = time.time() if evidence else None
                with status_lock:
                    if request_local_rng and resource == "gpu" and status["state"] == "quarantined":
                        raise RuntimeError("Kernel is quarantined; refusing a new device stage")
                    invoked = True
                if device_state is not None and resource == "gpu":
                    device_state["uncertain"] = True
                result = function(value)
                if strict:
                    result = _complete_native(result)
                elif resource == "gpu":
                    result = _complete_device(result)
                if device_state is not None:
                    device_state["uncertain"] = False
        except BaseException:
            if resource == "gpu" and (client is not None or request_local_rng) and invoked:
                # The broker fences exceptional GPU exits until this worker
                # ends. A new request must not queue behind its own live fence.
                with status_lock:
                    status["state"] = "quarantined"
                if controller is not None:
                    controller.quarantine()
            raise
        timings[name] = {
            "wait_seconds": admitted - begin,
            "run_seconds": time.perf_counter() - admitted,
        }
        if evidence:
            timings[name].update(started_unix=wall_begin, admitted_unix=wall_admitted,
                                 finished_unix=time.time())
        return result

    def run_evaluator(payload, timings, shape, profile_active, concurrency_allowed=False):
        """One memory lifetime around three independently admitted operations."""
        resource = infer.resource(payload) if callable(infer.resource) else infer.resource
        if resource not in {"cpu", "gpu"}:
            raise ValueError(f"Unknown evaluator inference resource {resource!r}")
        needs_gpu = "gpu" in (resource, infer.prepare_resource, infer.finalize_resource)
        memory = lease("gpu" if needs_gpu else "cpu", "evaluation_memory",
                       shape, profile_active, capacity_only=True)
        with memory as reservation:
            device_state = {"uncertain": False}

            def stage(name, kind, function, value):
                value = run_stage(name, kind, function, value, timings, shape,
                                  profile_active, concurrency_allowed=concurrency_allowed,
                                  reservation=reservation, strict=True,
                                  device_state=device_state)
                return value

            try:
                value = stage("prepare", infer.prepare_resource, infer.prepare, payload)
                # The local infer slot covers complete native completion only.
                # Capacity and raw outputs remain live until finalize below.
                gate = (controller.inference_slot() if controller is not None and resource == "gpu"
                        else nullcontext())
                with gate:
                    value = stage("infer", resource, infer.execute, value)
                # Inference has ended; raw tensors and capacity remain live
                # through native reduction, transfer, decoding and I/O.
                return stage(
                    "finalize", infer.finalize_resource,
                    lambda raw: host_result(_complete_native(infer.finalize(raw))), value)
            except BaseException as error:
                if reservation is not None and not device_state["uncertain"]:
                    reservation.finish(error=error)
                raise

    def serial_handle(request):
        operation = request.get("operation")
        if operation == "describe":
            return {"ok": True, "metadata": describe(),
                    "random_state": initialized_state}
        if operation != "infer":
            raise ValueError(f"Unknown kernel operation {operation!r}")
        if status["state"] == "quarantined":
            raise RuntimeError("Kernel is quarantined after device failure; rebuild its owner")
        timings = {}
        status["state"] = "busy"
        status["requests"] += 1
        try:
            shape = resource_shape(request["payload"])
        except (TypeError, ValueError, OSError):
            # Unclassified payloads retain the original scientific path, with
            # unknown memory and no concurrency certification.
            shape = "unclassified"
        profile_active = not profile or shape in profile.get("shapes", ())
        # A reused owner keeps its certified upper bound. An individual
        # trajectory may only narrow concurrency, never widen that evidence.
        request_serial = request.get("concurrency_certified") is False
        try:
            with _kernel_random_scope(request["random_state"], torch_device):
                payload = request["payload"]
                if isinstance(infer, EvaluatorStages):
                    result = run_evaluator(payload, timings, shape, profile_active)
                elif isinstance(infer, KernelStages):
                    resource = (infer.resource(payload) if callable(infer.resource)
                                else infer.resource)
                    if resource not in {"cpu", "gpu"}:
                        raise ValueError(f"Unknown kernel resource {resource!r}")
                    value = run_stage("prepare", "cpu", infer.prepare, payload,
                                      timings, shape, profile_active)
                    value = run_stage("execute", resource, infer.execute, value,
                                      timings, shape, profile_active,
                                      concurrency_allowed=not request_serial)
                    result = run_stage("finalize", "cpu", infer.finalize, value,
                                       timings, shape, profile_active)
                    result = host_result(result)
                else:
                    result = run_stage("execute", "cpu" if cpu_only else "gpu",
                                       infer, payload, timings, shape, profile_active,
                                       concurrency_allowed=not request_serial)
                    result = host_result(result)
            response = {"ok": True, "result": result}
        except Exception as error:
            status["failures"] += 1
            if ("outofmemory" in type(error).__name__.lower()
                    or "out of memory" in str(error).lower()):
                status["state"] = "quarantined"
            traceback.print_exc(file=sys.stderr)
            response = {"ok": False, "error": str(error),
                        "error_type": type(error).__name__}
        finally:
            if status["state"] != "quarantined":
                status["state"] = "ready"
            refresh_lifecycle()
        response["random_state"] = capture_random_state()
        response["runtime"] = {"request_id": request.get("runtime_request_id"),
                               "model_id": model_id, "timings": timings,
                               "shape": shape, "profile_matched": profile_active,
                               "lifecycle": dict(status)}
        return response

    def pipelined_handle(request):
        """Prepare the next request while the current request owns native RNG.

        The executor returns host materialized values before releasing its
        device lease, so finalization/transport never hide unaccounted GPU work.
        The returned RNG receipt is captured inside its request's critical
        section; another request cannot replace it before response delivery.
        """
        nonlocal active_requests
        timings = {}
        state = request["random_state"]
        shape = "unclassified"
        profile_active = False
        with status_lock:
            if status["state"] == "quarantined":
                raise RuntimeError("Kernel is quarantined after device failure; rebuild its owner")
            active_requests += 1
            status["active_requests"] = active_requests
            status["peak_active_requests"] = max(status["peak_active_requests"], active_requests)
            status["requests"] += 1
            status["state"] = "busy"
        try:
            try:
                shape = resource_shape(request["payload"])
            except (TypeError, ValueError, OSError):
                pass
            profile_active = not profile or shape in profile.get("shapes", ())
            payload = request["payload"]
            resource = infer.resource(payload) if callable(infer.resource) else infer.resource
            if resource not in {"cpu", "gpu"}:
                raise ValueError(f"Unknown kernel resource {resource!r}")
            value = run_stage("prepare", "cpu", infer.prepare, payload,
                              timings, shape, profile_active)
            waiting = time.perf_counter()
            with _KERNEL_RANDOM_LOCK:
                acquired = time.perf_counter()
                timings["rng_owner"] = {"wait_seconds": acquired - waiting,
                                        "run_seconds": 0.0}
                with status_lock:
                    if status["state"] == "quarantined":
                        raise RuntimeError("Kernel is quarantined after device failure; rebuild its owner")
                with _kernel_random_scope(state, torch_device):
                    try:
                        value = run_stage(
                            "execute", resource, infer.execute, value,
                            timings, shape, profile_active,
                            concurrency_allowed=False,
                        )
                        # Keep any generic D2H conversion inside the native RNG
                        # section. Backends must materialize device results in
                        # execute so conversion here only walks host values.
                        value = host_result(value)
                    except Exception as error:
                        # Publish isolation before another prepared request can
                        # acquire this RNG owner. The outer exception handler
                        # runs after releasing the lock and is too late.
                        if (resource == "gpu"
                                or "outofmemory" in type(error).__name__.lower()
                                or "out of memory" in str(error).lower()):
                            with status_lock:
                                status["state"] = "quarantined"
                        raise
                    finally:
                        state = capture_random_state()
                timings["rng_owner"]["run_seconds"] = time.perf_counter() - acquired
            result = run_stage("finalize", "cpu", infer.finalize, value,
                               timings, shape, profile_active)
            response = {"ok": True, "result": host_result(result)}
        except Exception as error:
            with status_lock:
                status["failures"] += 1
                if ("outofmemory" in type(error).__name__.lower()
                        or "out of memory" in str(error).lower()):
                    status["state"] = "quarantined"
            traceback.print_exc(file=sys.stderr)
            response = {"ok": False, "error": str(error),
                        "error_type": type(error).__name__}
        finally:
            with status_lock:
                active_requests -= 1
                status["active_requests"] = active_requests
                if status["state"] != "quarantined":
                    status["state"] = "busy" if active_requests else "ready"
                refresh_lifecycle()
        response["random_state"] = state
        with status_lock:
            snapshot = dict(status)
        response["runtime"] = {
            "request_id": request.get("runtime_request_id"), "model_id": model_id,
            "timings": timings, "shape": shape, "profile_matched": profile_active,
            "lifecycle": snapshot,
        }
        return response

    def request_local_handle(request):
        """Only the opt-in immutable AF3 path bypasses global RNG mutation."""
        nonlocal active_requests
        timings = {}
        shape = "unclassified"
        profile_active = False
        with status_lock:
            if status["state"] == "quarantined":
                raise RuntimeError("Kernel is quarantined after device failure; rebuild its owner")
            active_requests += 1
            status.update(state="busy", active_requests=active_requests,
                          peak_active_requests=max(status["peak_active_requests"], active_requests),
                          requests=status["requests"] + 1)
        try:
            try:
                shape = resource_shape(request["payload"])
            except (TypeError, ValueError, OSError):
                pass
            profile_active = not profile or shape in profile.get("shapes", ())
            result = run_evaluator(request["payload"], timings, shape, profile_active,
                                   concurrency_allowed=request.get("concurrency_certified") is not False)
            response = {"ok": True, "result": result}
        except Exception as error:
            with status_lock:
                status["failures"] += 1
                if ("outofmemory" in type(error).__name__.lower()
                        or "out of memory" in str(error).lower()
                        or "resource_exhausted" in str(error).lower()):
                    status["state"] = "quarantined"
                quarantined = status["state"] == "quarantined"
            if quarantined:
                controller.quarantine()
            traceback.print_exc(file=sys.stderr)
            response = {"ok": False, "error": str(error), "error_type": type(error).__name__}
        finally:
            with status_lock:
                active_requests -= 1
                status["active_requests"] = active_requests
                if status["state"] != "quarantined":
                    status["state"] = "busy" if active_requests else "ready"
                refresh_lifecycle()
                snapshot = dict(status)
            snapshot["device_execution_slots"] = controller.snapshot()["limit"]
        # Explicit JAX keys advance no Python/NumPy/Torch default generators.
        response["random_state"] = request["random_state"]
        response["runtime"] = {
            "request_id": request.get("runtime_request_id"), "model_id": model_id,
            "timings": timings, "shape": shape, "profile_matched": profile_active,
            "lifecycle": snapshot,
        }
        return response

    def adaptive_control(*, enabled, reset=False):
        if controller is None:
            raise RuntimeError("adaptive_control requires an explicitly enabled AF3 owner")
        with status_lock:
            if active_requests:
                raise RuntimeError("adaptive_control requires all evaluator stages to drain")
            result = controller.control(enabled=enabled, reset=reset)
            status["device_execution_slots"] = result["limit"]
            return result

    def handle(request):
        if request.get("operation") == "describe":
            return {"ok": True, "metadata": describe(), "random_state": initialized_state}
        if request.get("operation") != "infer":
            raise ValueError(f"Unknown kernel operation {request.get('operation')!r}")
        if request_local_rng:
            return request_local_handle(request)
        if pipeline_slots > 1:
            with pipeline_limit:
                return pipelined_handle(request)
        with _KERNEL_RANDOM_LOCK:
            return serial_handle(request)

    handle.max_concurrency = controller.settings.max_concurrency if controller is not None else pipeline_slots
    handle.request_local_rng = bool(request_local_rng)
    handle.adaptive_control = adaptive_control
    handle.describe_snapshot = describe
    handle.close = controller.close if controller is not None else lambda: None
    if controller is not None:
        controller.start()
    return handle


def _resident_model_endpoint(spec, identity, socket_path):
    """Resolve an explicit registry entry; never authorize a replacement load."""
    registry_path = os.environ.get("ASTROLABE_RESIDENT_MODEL_REGISTRY")
    if registry_path is None:
        return None
    if not registry_path:
        raise ValueError("Resident model registry path must be nonempty")
    with Path(registry_path).expanduser().open(encoding="utf-8") as handle:
        registry = json.load(handle)
    if (not isinstance(registry, dict)
            or registry.get("schema") != "astrolabe.resident-model-registry/v1"):
        raise ValueError("Invalid resident model registry schema")
    broker = os.environ.get("ASTROLABE_RESOURCE_SOCKET")
    broker = str(Path(broker).expanduser().resolve()) if broker else None
    if "broker_socket" not in registry or registry["broker_socket"] != broker:
        raise ValueError("Resident model registry broker_socket does not match current broker")
    models = registry.get("models")
    if not isinstance(models, dict) or identity not in models:
        raise ValueError(f"Resident model registry has no exact identity: {identity}")
    entry = models[identity]
    if not isinstance(entry, dict) or entry.get("backend") != spec.backend:
        raise ValueError("Resident model registry backend does not match selected model")
    endpoint = entry.get("socket")
    if not isinstance(endpoint, str) or not endpoint or not Path(endpoint).is_absolute():
        raise ValueError("Resident model registry socket must be an absolute private socket path")
    endpoint = Path(endpoint).resolve()
    if socket_path is not None and Path(socket_path).expanduser().resolve() != endpoint:
        raise ValueError("Explicit socket_path conflicts with resident model registry socket")
    return endpoint, broker


def _shared_model_metadata(metadata, owner, spec, identity, broker):
    """Bind describe to this broker's existing ledger without registering anew."""
    lifecycle = metadata.get("runtime_lifecycle", {})
    if (not isinstance(lifecycle, Mapping)
            or lifecycle.get("identity") != identity
            or lifecycle.get("backend") != spec.backend
            or lifecycle.get("pid") != owner.pid
            or not lifecycle.get("model_id")):
        raise ValueError("Shared model describe does not match the attached owner identity/backend/PID")
    lifecycle = dict(lifecycle)
    resources = {"broker_socket": broker}
    metadata = dict(metadata)
    if broker is not None:
        from astrolabe.runtime.admission import ResourceClient

        snapshot = ResourceClient(broker).snapshot()
        records = [record for record in snapshot["models"]
                   if record["model_id"] == lifecycle["model_id"]]
        if len(records) != 1 or records[0]["pid"] != owner.pid:
            raise ValueError("Shared model owner is not registered with the current broker")
        record = records[0]
        device = record["device"]
        if device is not None and device not in snapshot["devices"]:
            raise ValueError("Shared model ledger device is not a physical device of this broker")
        resources.update(record)
        # The consumer's CUDA visibility may differ from the GPU owner's.
        # Preserve physical ledger identity instead of relabeling its cuda:0.
        metadata["device"] = lifecycle["device"] = device if device is not None else "cpu"
    lifecycle["resources"] = resources
    metadata["runtime_lifecycle"] = lifecycle
    return metadata


@contextmanager
def independent_initialization(seed: int):
    """Give an explicitly seeded model load a private random stream."""
    import numpy as np

    algorithm, keys, position, has_gauss, cached = np.random.RandomState(seed).get_state()
    state = {
        "python": random.Random(seed).getstate(),
        "numpy": [algorithm, keys.tolist(), position, has_gauss, cached],
        "torch_seed": seed,
        "cuda_seed": seed,
    }
    torch = sys.modules.get("torch")
    if torch is not None:
        state["torch"] = torch.Generator(device="cpu").manual_seed(seed).get_state().tolist()
    token = _INITIALIZATION_RANDOM_STATE.set(state)
    try:
        yield
    finally:
        _INITIALIZATION_RANDOM_STATE.reset(token)


@contextmanager
def model_construction_scope():
    """Roll back even owners created by a constructor that never returned."""
    parent = _RUN_OWNERS.get()
    owners = StartupScope(parent=parent)
    token = _RUN_OWNERS.set(owners)
    try:
        yield owners
    except BaseException as error:
        owners.cancel()
        if not owners.cleanup_report.complete:
            owners.retain_errors(error)
        raise
    else:
        if parent is not None:
            try:
                for owner in owners.owners:
                    parent.add(owner, source=owners)
            except BaseException as error:
                # A parent may drain while construction is finishing. Retain
                # and clean every local owner, not only the first transfer.
                owners.cancel()
                if not owners.cleanup_report.complete:
                    owners.retain_errors(error)
                raise
    finally:
        _RUN_OWNERS.reset(token)


class NativeModelClient:
    """One typed backend kernel behind a private resident model owner."""
    def __init__(self, backend, factory, settings, *, python_bin=None, cwd=None,
                 env=None, request_timeout=None,
                 max_frame_bytes=DEFAULT_MODEL_FRAME_BYTES, socket_path=None,
                 startup_timeout=300, shutdown_timeout=30, max_pending=4,
                 provider=None, resource_profile=None, direct_payload=False):
        independent_state = _INITIALIZATION_RANDOM_STATE.get()
        guard = (_CALLER_RANDOM_LOCK if independent_state is None and not direct_payload
                 else nullcontext())
        with guard:
            self._initialize(
                backend, factory, settings, python_bin=python_bin, cwd=cwd, env=env,
                request_timeout=request_timeout, max_frame_bytes=max_frame_bytes,
                socket_path=socket_path, startup_timeout=startup_timeout,
                shutdown_timeout=shutdown_timeout, max_pending=max_pending,
                provider=provider, resource_profile=resource_profile,
                direct_payload=direct_payload, independent_state=independent_state,
            )

    def _initialize(self, backend, factory, settings, *, python_bin, cwd, env,
                    request_timeout, max_frame_bytes, socket_path, startup_timeout,
                    shutdown_timeout, max_pending, provider, resource_profile,
                    direct_payload, independent_state):
        from astrolabe.model_daemon import (
            DaemonSpec, LocalModelOwner, OwnedModelDaemon,
            model_identity, _prepare_provider_spec,
        )
        self._request_lock = threading.RLock()
        self.last_runtime_receipt = None
        self.direct_payload = direct_payload
        settings = dict(settings)
        if not direct_payload:
            settings["_initial_random_state"] = (
                capture_random_state() if independent_state is None else independent_state
            )
        spec = DaemonSpec(backend=backend, factory=factory, settings=settings,
                          python_bin=python_bin or sys.executable, cwd=cwd, env=env,
                          max_frame_bytes=max_frame_bytes, provider=provider)
        spec = _prepare_provider_spec(spec)
        identity = model_identity(spec, max_pending=max_pending)
        shared_endpoint = _resident_model_endpoint(spec, identity, socket_path)
        if shared_endpoint is not None:
            socket_path, shared_broker = shared_endpoint
        profiles_path = os.environ.get("ASTROLABE_RESOURCE_PROFILES")
        automatic_profile = resource_profile is None and bool(profiles_path)
        if resource_profile is None and profiles_path:
            with open(profiles_path) as handle:
                resource_profile = json.load(handle).get("profiles", {}).get(identity)
            if resource_profile and resource_profile.get("verified") is not True:
                raise ValueError("Automatic resource profiles require verified evidence")
        if resource_profile is not None:
            resource_profile = dict(resource_profile)
            if resource_profile.get("identity", identity) != identity:
                raise ValueError("Resource profile does not match the selected model")
            if automatic_profile or resource_profile.get("qualification") is not True:
                if (Path(spec.python_bin).resolve() != Path(sys.executable).resolve()
                        or spec.provider is not None
                        or any(str(value) != os.environ.get(str(key))
                               for key, value in (spec.env or {}).items())):
                    raise ValueError(
                        "Production resource profiles require attestation in the "
                        "actual child runtime; cross-environment qualification "
                        "must use an explicit bounded trial"
                    )
                _validate_production_profile(
                    resource_profile, backend, spec.settings, spec.factory, spec.cwd)
            else:
                _verify_asset_hashes(resource_profile.get("asset_sha256", {}))
            resource_profile["identity"] = identity
            spec.settings["_runtime_profile"] = resource_profile
        self._resource_profile = resource_profile
        owner_type = (LocalModelOwner if not direct_payload and shared_endpoint is None
                      and os.environ.get("ASTROLABE_MODEL_OWNER") == "trajectory"
                      else OwnedModelDaemon)
        owner_kwargs = dict(request_timeout=request_timeout,
                            socket_path=socket_path,
                            startup_timeout=startup_timeout,
                            shutdown_timeout=shutdown_timeout,
                            max_pending=max_pending)
        from astrolabe.runtime.model_pool import get_local_model_pool
        # Direct drivers own their job/seed loops in the isolated daemon; they
        # are not typed infer kernels eligible for trajectory-local pooling.
        pool = get_local_model_pool() if shared_endpoint is None and not direct_payload else None
        self.owner = None
        try:
            self.owner = (pool.acquire(spec, owner_type, owner_kwargs, identity)
                          if pool is not None and socket_path is None else None)
            if self.owner is None:
                self.owner = owner_type(spec, **owner_kwargs)
            description = self.owner.request({"operation": "describe"})
            self.metadata = description["metadata"]
            if shared_endpoint is not None:
                self.metadata = _shared_model_metadata(
                    self.metadata, self.owner, spec, identity, shared_broker)
            if not direct_payload:
                if independent_state is None:
                    restore_random_state(description["random_state"])
                else:
                    _INITIALIZATION_RANDOM_STATE.set(description["random_state"])
            _request_evidence("client_ready", self.owner, self.metadata)
        except BaseException as error:
            if self.owner is not None:
                quarantine = getattr(self.owner, "quarantine", None)
                if callable(quarantine):
                    quarantine()
                try:
                    self.owner.shutdown()
                except BaseException as cleanup_error:
                    error.add_note(f"native owner cleanup failed: {cleanup_error!r}")
            raise

    def request(self, payload):
        guard = nullcontext() if getattr(self, "direct_payload", False) else _CALLER_RANDOM_LOCK
        with guard:
            return self._request(payload)

    def _request(self, payload):
        with self._request_lock:
            request_id = uuid.uuid4().hex
            payload = host_result(payload)
            try:
                _request_evidence("request_started", self.owner, self.metadata,
                                  request_id=request_id)
                if getattr(self, "direct_payload", False):
                    # The native driver dispatches its own per-sample stages.
                    # Do not wrap the call or replace the owner's RNG stream.
                    result = self.owner.request(payload)
                    _request_evidence("request_finished", self.owner, self.metadata,
                                      request_id=request_id, ok=True, runtime=None)
                    return result
                response = self.owner.request({
                    "operation": "infer", "payload": payload,
                    "random_state": capture_random_state(),
                    "runtime_request_id": request_id,
                    "concurrency_certified": (
                        os.environ.get("ASTROLABE_REQUEST_CONCURRENCY") != "0"),
                })
                runtime = response.get("runtime")
                if runtime is not None and runtime["request_id"] != request_id:
                    raise RuntimeError("Kernel result belongs to a different runtime request")
                if callable(getattr(self.owner, "quarantine", None)):
                    expected_model = self.metadata["runtime_lifecycle"]["model_id"]
                    if runtime is None or runtime.get("model_id") != expected_model:
                        raise RuntimeError("Pooled kernel result has no matching model identity")
                self.last_runtime_receipt = runtime
                restore_random_state(response["random_state"])
                _request_evidence("request_finished", self.owner, self.metadata,
                                  request_id=request_id, ok=response["ok"],
                                  runtime=runtime)
                if not response["ok"]:
                    raise RuntimeError(f"{response['error_type']}: {response['error']}")
                return response["result"]
            except BaseException as error:
                try:
                    _request_evidence(
                        "request_error", self.owner, self.metadata, request_id=request_id,
                        error_type=type(error).__name__,
                        outcome_unknown=bool(getattr(error, "outcome_unknown", False)),
                    )
                except Exception:
                    logging.getLogger(__name__).exception("Failed to persist model request error")
                quarantine = getattr(self.owner, "quarantine", None)
                if callable(quarantine):
                    quarantine()
                raise

    def shutdown(self):
        self.owner.shutdown()
        # An allocator can retain VRAM after del/offload. Only acknowledge
        # resident release after the private worker has actually exited.
        lifecycle = self.metadata.get("runtime_lifecycle", {})
        client = _resource_client()
        if (client is not None and lifecycle.get("model_id")
                and not getattr(self.owner, "attached", False)):
            if not getattr(self.owner, "release_requires_process_exit", False):
                client.unregister_model(lifecycle["model_id"])

    @property
    def pid(self):
        return self.owner.pid
