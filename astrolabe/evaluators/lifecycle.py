"""Observed backend lifecycle state, separate from process liveness.

Load counters are incremented at successful loader calls by the adapter.  This
module only inspects already-loaded tensors; it never loads a model or imports
a GPU framework to answer a status request.
"""
from __future__ import annotations

from collections.abc import Mapping
import os
import sys
import time
import hashlib
import json
from pathlib import Path
from contextlib import contextmanager


def tensor_residency(value):
    """Return observed device bytes without allocating or moving tensors."""
    torch = sys.modules.get("torch")
    seen = set()
    storage_seen = set()
    devices = {}
    tensor_count = 0

    def walk(item):
        nonlocal tensor_count
        if id(item) in seen:
            return
        seen.add(id(item))
        if torch is not None and isinstance(item, torch.Tensor):
            tensor_count += 1
            device = str(item.device)
            try:
                storage = item.untyped_storage()
                key = (device, storage.data_ptr())
                size = storage.nbytes()
            except (AttributeError, RuntimeError):
                key = (device, id(item))
                size = item.numel() * item.element_size()
            if key not in storage_seen:
                storage_seen.add(key)
                devices[device] = devices.get(device, 0) + int(size)
        elif hasattr(item, "devices") and hasattr(item, "nbytes"):
            # jax.Array, including CPU arrays.  Accessors do not initialize a
            # device or materialize the array on the host.
            actual = tuple(item.devices())
            if len(actual) == 1:
                dev = actual[0]
                name = f"{dev.platform}:{dev.id}"
                devices[name] = devices.get(name, 0) + int(item.nbytes)
                tensor_count += 1
        elif isinstance(item, Mapping):
            for child in item.values():
                walk(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                walk(child)
        elif hasattr(item, "parameters") and hasattr(item, "buffers"):
            for child in item.parameters():
                walk(child)
            for child in item.buffers():
                walk(child)

    if value is not None:
        walk(value)
    gpu = sum(size for dev, size in devices.items() if dev.split(":")[0] in {"cuda", "gpu"})
    cpu = sum(size for dev, size in devices.items() if dev.split(":")[0] == "cpu")
    return {"tensor_count": tensor_count, "devices": devices,
            "resident_bytes": gpu, "host_parameter_bytes": cpu,
            "gpu_resident": bool(gpu) and cpu == 0 if tensor_count else None}


def model_lifecycle(model=None, *, load_count, loaded, device=None, components=None,
                    ready=True, residency_kind="resident_model", release_requires_exit=True):
    observed = tensor_residency(model)
    known_empty = model is None or isinstance(model, (Mapping, tuple, list)) and len(model) == 0
    if not observed['tensor_count'] and (loaded or not known_empty):
        observed['resident_bytes'] = None
        observed['host_parameter_bytes'] = None
    return {**observed, "process_alive": True, "pid": os.getpid(), "weights_loaded": bool(loaded),
            "gpu_resident": observed["gpu_resident"] if loaded else False,
            "ready": bool(ready), "load_count": int(load_count), "device": device,
            "residency_kind": residency_kind, "components": components or {},
            "release_requires_exit": bool(release_requires_exit)}


@contextmanager
def measured_phase(metrics, phase):
    """Wall time of a real stage; never equate admitted time with GPU activity."""
    start = time.perf_counter()
    try:
        yield
    finally:
        key = phase + "_seconds"
        metrics[key] = metrics.get(key, 0.0) + time.perf_counter() - start


def cpu_tree(value):
    """Finish device work by moving tensor results to CPU, preserving dtype."""
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    return value


def shape_key(value):
    """Shape-only grouping, without leaking sequence data into telemetry."""
    shapes = []
    def walk(item, path=""):
        if hasattr(item, 'shape') and hasattr(item, 'dtype'):
            shapes.append((path, list(item.shape), str(item.dtype)))
        elif isinstance(item, Mapping):
            for key in sorted(item):
                walk(item[key], path + '/' + str(key))
        elif isinstance(item, (list, tuple)):
            shapes.append((path, [len(item)], type(item).__name__))
            for i, child in enumerate(item):
                walk(child, path + '/' + str(i))
        elif isinstance(item, str):
            shapes.append((path, [len(item)], 'text'))
    walk(value)
    return hashlib.sha256(json.dumps(shapes, sort_keys=True).encode()).hexdigest()


def request_runtime(directory, phase, value=None, *, cold=None, execution_settings=None):
    """Account file-backed input bytes as well as the tiny JSON control frame."""
    path = Path(directory)
    prepared = sum(p.stat().st_size for p in path.rglob('*') if p.is_file())
    result = {'phase': phase, 'shape_key': shape_key(value),
              'resources': {'prepared_bytes': prepared, 'output_bytes': None}}
    if execution_settings is not None:
        # Native backends can change process-global precision between calls.
        # Do not mix those requests in one shape/performance observation.
        settings_key = hashlib.sha256(json.dumps(execution_settings, sort_keys=True, allow_nan=False).encode()).hexdigest()
        result['shape_key'] += ':' + settings_key
    if cold is not None:
        result['cold'] = bool(cold)
    return result


def validate_output_directory(directory):
    """Reject vendor-written output above its allowance before publication.

    This is a completion check, not a hard filesystem quota. The broker also
    samples live artifact usage and terminates an over-budget process tree.
    """
    from astrolabe.runtime.server import current_request
    context = current_request()
    size = sum(path.stat().st_size for path in Path(directory).rglob('*') if path.is_file())
    if context and context.get('output_limit_bytes') is not None and size > context['output_limit_bytes']:
        raise ValueError('backend artifacts exceed admitted output byte allowance')
    return size
