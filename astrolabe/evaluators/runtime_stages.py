"""Small evaluator adapters for the shared execution contract.

Resource declarations describe actual operations, never method naming alone.
Host materialization belongs inside the device lease, before CPU finalization.
"""
from __future__ import annotations

from concurrent.futures import Future
from collections.abc import Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import builtins
import os


_LOCAL_EVALUATOR = ContextVar("astrolabe_local_evaluator_lease", default=None)


def configured_device_hint(config, *, default=None):
    """Logical CUDA ordinal; the broker alone maps CUDA_VISIBLE_DEVICES."""
    get = config.get if hasattr(config, "get") else lambda key: getattr(config, key, None)
    gpu_device = get("gpu_device")
    value = gpu_device if gpu_device is not None else get("device")
    if value is None or str(value).lower() in {"auto", "none"}:
        return default
    if isinstance(value, bool):
        raise ValueError("GPU device must be an ordinal, not a boolean")
    value = str(value).lower()
    if value == "cpu":
        return "cpu"
    if value in {"cuda", "gpu"}:
        return "cuda:0"
    if value.startswith(("cuda:", "gpu:")):
        value = value.split(":", 1)[1]
    if value.isdigit():
        return f"cuda:{int(value)}"
    raise ValueError(f"Unsupported configured device {value!r}")


def validate_legacy_device(config, *, backend):
    hint = configured_device_hint(config)
    if os.environ.get("ASTROLABE_RESOURCE_SOCKET") and hint not in {None, "cpu", "cuda:0"}:
        raise ValueError(f"Explicit nonzero GPU for legacy {backend} requires native device admission")


@contextmanager
def jax_device_scope(device_hint):
    """Bind explicit AF2 placement both while loading and while predicting."""
    if device_hint is None:
        yield None
        return
    import jax

    backend = "cpu" if device_hint == "cpu" else "gpu"
    index = 0 if backend == "cpu" else int(device_hint.split(":", 1)[1])
    devices = jax.local_devices(backend=backend)
    if index >= len(devices):
        raise ValueError(f"Configured {device_hint} is not visible to JAX")
    with jax.default_device(devices[index]):
        yield devices[index]


def identity(value):
    return value


def operation_resource(cpu_operations, *, device=None):
    cpu_operations = frozenset(cpu_operations)

    def resource(payload):
        if str(device) == "cpu" or payload.get("operation") in cpu_operations:
            return "cpu"
        return "gpu"

    return resource


def complete_stage(stage, value):
    """Drain native Pending objects without mistaking dispatch for completion."""
    from astrolabe.inference_pipeline import Pending

    output = stage.run(value)
    if isinstance(output, Pending):
        # Keep the Pending and its keepalive alive through completion, including
        # the exceptional path. Do not change the native result payload.
        if isinstance(output.completion, Future):
            output.completion.result()
        else:
            output.completion()
        return output.value
    return output


def local_evaluator_guard(backend, handle_attr):
    """Fence the legacy in-process path without enclosing an RPC in a lease."""
    def decorate(function):
        @wraps(function)
        def guarded(self, *args, **kwargs):
            with local_evaluator_operation(self, backend, handle_attr):
                return function(self, *args, **kwargs)
        guarded._local_device_operation = (backend, handle_attr)
        return guarded
    return decorate


@contextmanager
def local_evaluator_operation(evaluator, backend=None, handle_attr=None):
    """Legacy evaluate includes any native-tensor scoring under one lease."""
    from astrolabe.model_execution import device_operation

    if getattr(evaluator, "_staged_evaluation", False):
        # Converted backends admit their actual model stages in the owner.
        # Reacquiring a lease here would collapse the complete evaluation and
        # deadlock the owner's capacity reservation in the in-process path.
        yield
        return

    if backend is None:
        declaration = getattr(evaluator.run_function, "_local_device_operation", None)
        if declaration is None:
            yield
            return
        backend, handle_attr = declaration
    handles = getattr(evaluator, handle_attr, None)
    handles = list(handles.values()) if isinstance(handles, dict) else [handles]
    remote = handles and all(
        getattr(handle, "_client", None) is not None
        or getattr(handle, "_socket", None) is not None
        for handle in handles
    )
    device = getattr(evaluator, "_device", None)
    for config_name in ("config", "model_config"):
        config = getattr(evaluator, config_name, None)
        if config is not None and getattr(config, "device", None) is not None:
            device = config.device
    cpu_only = str(device) == "cpu" or all(
        getattr(handle, "accelerator", None) == "cpu" for handle in handles
    )
    if remote or cpu_only or _LOCAL_EVALUATOR.get() is evaluator:
        yield
        return
    hints = [getattr(evaluator, "_device", None)]
    for config_name in ("config", "model_config", "af3_conf"):
        config = getattr(evaluator, config_name, None)
        if config is not None:
            hints.append(configured_device_hint(config))
    for handle in handles:
        devices = getattr(handle, "devices", None)
        if isinstance(devices, Sequence) and not isinstance(devices, str):
            hints.extend(f"cuda:{int(index)}" for index in devices)
        hints.append(getattr(handle, "device", None))
    if os.environ.get("ASTROLABE_RESOURCE_SOCKET") and any(
        str(hint).startswith("cuda:") and str(hint) != "cuda:0" for hint in hints
    ):
        raise ValueError("Explicit nonzero GPU in a legacy evaluator requires native device admission")
    with device_operation(backend, exclusive=True):
        token = _LOCAL_EVALUATOR.set(evaluator)
        try:
            yield
        finally:
            _LOCAL_EVALUATOR.reset(token)


class DirectKernel:
    """Keep an existing daemon's direct payload protocol around kernel_handler.

    The caller stays an RPC client; admission and model ownership stay here, in
    the process that actually loads and executes the vendor model.
    """
    def __init__(self, settings, loader, *, torch_device=None, lifecycle=None):
        from astrolabe.model_execution import kernel_handler

        self.handle = kernel_handler(
            settings, loader, torch_device=torch_device, lifecycle=lifecycle,
        )

    @property
    def metadata(self):
        return self.handle({"operation": "describe"})["metadata"]

    @property
    def request_local_rng(self):
        return self.handle.request_local_rng

    def describe_snapshot(self):
        return self.handle.describe_snapshot()

    def adaptive_control(self, *, enabled, reset=False):
        return self.handle.adaptive_control(enabled=enabled, reset=reset)

    def close(self):
        self.handle.close()

    def __call__(self, payload):
        from astrolabe.model_execution import capture_random_state

        response = self.handle({
            "operation": "infer", "payload": payload,
            "random_state": {} if self.request_local_rng else capture_random_state(),
        })
        if not response["ok"]:
            error_type = getattr(builtins, response["error_type"], RuntimeError)
            if not isinstance(error_type, type) or not issubclass(error_type, Exception):
                error_type = RuntimeError
            raise error_type(response["error"])
        return response["result"]


def resident_engine_kernel(engine_cls, config):
    """RF3 keeps its existing initialize/run API and scientific RNG owner."""
    loaded = []

    def loader(raw_settings):
        engine = engine_cls(**raw_settings)
        initialize = getattr(engine, "initialize", None)
        if callable(initialize):
            initialize()
        loaded.append(engine)

        def infer(kwargs):
            engine.run(**kwargs)
            # The public RF3 contract is its existing output files. Do not
            # materialize an unused vendor return value for the RPC response.
            return None

        return infer, {"backend": "rf3", "residency": "resident_model",
                       "device": str(getattr(engine, "device", "unknown"))}

    device_hint = configured_device_hint(config)
    kernel = DirectKernel(config, loader, torch_device=device_hint or "auto", lifecycle={
        "backend": "rf3", "residency": "resident_model",
        "device": device_hint,
        "staging": "native_engine_run",
    })
    return loaded[0], kernel


def cli_bridge_kernel(function, *, backend, device_hint=None):
    """Admit a complete CLI process tree on its configured logical device."""
    from astrolabe.model_execution import KernelStages

    metadata = {"backend": backend, "device": device_hint,
                "residency": "resident_bridge", "weights_loaded": False,
                "gpu_resident": False, "child_residency": "one_shot"}
    return DirectKernel(
        {}, lambda settings: (KernelStages(identity, function), metadata),
        # The parent only launches the CLI; it must not initialize Torch CUDA.
        torch_device=None,
        # Only construction of this bridge is CPU work. The child CLI remains
        # one exclusive GPU execute stage; no resident model is registered for
        # the parent bridge by kernel_handler.
        lifecycle=dict(metadata, resource="cpu", concurrency_certified=False),
    )
