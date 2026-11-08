"""Admission glue for mutator implementations outside NativeModelClient.

Only the child owns a native kernel lease. Direct/CLI compatibility calls use
an exclusive device guard unless their execution device is demonstrably CPU.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

from astrolabe import model_execution

_ADMITTED = ContextVar("astrolabe_mutator_admitted", default=False)


@contextmanager
def admitted_kernel():
    token = _ADMITTED.set(True)
    try:
        yield
    finally:
        _ADMITTED.reset(token)


@contextmanager
def device_guard(backend, *, device=None, phase="compute", identity=None):
    if _ADMITTED.get() or str(device).split(":", 1)[0] == "cpu":
        yield
    else:
        with model_execution.device_operation(
            backend, phase=phase, identity=identity, exclusive=True
        ):
            yield


def guarded_device(backend, *, phase="compute"):
    """Guard direct runner calls; a factory kernel already owns admission."""
    def decorate(function):
        @wraps(function)
        def guarded(self, *args, **kwargs):
            with device_guard(backend, device=getattr(self, "device", None), phase=phase):
                result = function(self, *args, **kwargs)
                if phase == "compute" and not _ADMITTED.get():
                    return model_execution.host_result(result)
                return result
        return guarded
    return decorate


def mutator_kernel_handler(settings, loader, *, backend, torch_device=None,
                           residency="resident_model", **lifecycle):
    if residency == "resident_model":
        lifecycle.setdefault("weights_loaded", True)
    def load(config):
        with admitted_kernel():
            implementation, metadata = loader(config)
        if isinstance(implementation, model_execution.KernelStages):
            implementation_execute = implementation.execute
            # Bind separately: implementation is replaced below.
            def execute(payload):
                with admitted_kernel():
                    return implementation_execute(payload)
            implementation = model_execution.KernelStages(
                implementation.prepare, execute, implementation.finalize,
                resource=implementation.resource,
            )
        else:
            implementation_infer = implementation
            def infer(payload):
                with admitted_kernel():
                    return implementation_infer(payload)
            implementation = infer
        metadata = dict(metadata)
        metadata.setdefault("backend", backend)
        metadata.setdefault("residency", residency)
        return implementation, metadata
    return model_execution.kernel_handler(
        settings, load, torch_device=torch_device,
        lifecycle={"backend": backend, "residency": residency, **lifecycle},
    )


def device_resource(device):
    return "cpu" if str(device).split(":", 1)[0] == "cpu" else "gpu"


def torch_load_device(requested="auto", *, cpu_fallback=False):
    """Resolve availability only for backends that already support CPU fallback."""
    import torch
    if requested == "cpu" or (cpu_fallback and not torch.cuda.is_available()):
        return "cpu"
    return requested or "auto"


def to_device(value, device):
    """Move only tensors, preserving native nested feature structure."""
    import torch
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(to_device(item, device) for item in value)
    return value


def protocol_infer(handler, payload):
    """Reuse kernel accounting from an existing JSONL/raw daemon protocol."""
    response = handler({"operation": "infer", "payload": payload,
                        "random_state": model_execution.capture_random_state()})
    handler.last_runtime = response.get("runtime")
    if not response["ok"]:
        raise RuntimeError(f"{response['error_type']}: {response['error']}")
    return response["result"]
