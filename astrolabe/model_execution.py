"""Execution policy and random-state transfer for native model kernels."""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import fields, is_dataclass, replace
from enum import Enum
import logging
import os
import random
import sys
import traceback


_RUN_OWNERS = ContextVar("astrolabe_model_run_owners", default=None)
DEFAULT_MODEL_FRAME_BYTES = 256 * 1024 * 1024
_REMOTE_CUDA_STATE = None


def reset_remote_random_state() -> None:
    """Begin a new CUDA stream when the scientific coordinator is reseeded."""
    global _REMOTE_CUDA_STATE
    _REMOTE_CUDA_STATE = None


def register_model_owner(owner) -> None:
    owners = _RUN_OWNERS.get()
    if owners is not None and not getattr(owner, "attached", False):
        owners.append(owner)


@contextmanager
def model_run_scope():
    owners = []
    token = _RUN_OWNERS.set(owners)
    try:
        yield
    finally:
        active_error = sys.exc_info()[0] is not None
        first_error = None
        closed = set()
        try:
            for owner in reversed(owners):
                if id(owner) in closed:
                    continue
                closed.add(id(owner))
                try:
                    owner.shutdown()
                except Exception as error:
                    logging.exception("Failed to stop a model owner at the end of its run")
                    if first_error is None:
                        first_error = error
        finally:
            _RUN_OWNERS.reset(token)
        if first_error is not None and not active_error:
            raise RuntimeError("A model owner failed to stop after the run") from first_error


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


def restore_random_state(state: Mapping) -> None:
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
        if not torch.cuda.is_initialized():
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
def _kernel_random_scope(state, torch_device):
    global _REMOTE_CUDA_STATE
    if torch_device is not None:
        import torch
    else:
        torch = sys.modules.get("torch")
    restore_random_state(state)
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


def kernel_handler(settings: dict, loader, *, torch_device=None):
    """Load once; Torch CUDA kernels declare 'auto' or their configured device.

    CPU and JAX kernels carry CUDA state opaquely. Scientific execution is
    serialized so the temporary Torch seed observer has one owner.
    """
    initial_state = settings.get("_initial_random_state") or capture_random_state()
    with _kernel_random_scope(initial_state, torch_device):
        infer, metadata = loader(settings)
    initialized_state = capture_random_state()

    def handle(request):
        operation = request.get("operation")
        if operation == "describe":
            return {"ok": True, "metadata": metadata,
                    "random_state": initialized_state}
        if operation != "infer":
            raise ValueError(f"Unknown kernel operation {operation!r}")
        try:
            with _kernel_random_scope(request["random_state"], torch_device):
                result = host_result(infer(request["payload"]))
            response = {"ok": True, "result": result}
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            response = {"ok": False, "error": str(error),
                        "error_type": type(error).__name__}
        response["random_state"] = capture_random_state()
        return response

    return handle


class NativeModelClient:
    """One typed backend kernel behind a private resident model owner."""
    def __init__(self, backend, factory, settings, *, python_bin=None, cwd=None,
                 env=None, request_timeout=None,
                 max_frame_bytes=DEFAULT_MODEL_FRAME_BYTES, socket_path=None,
                 startup_timeout=300, shutdown_timeout=30, max_pending=4,
                 provider=None):
        from astrolabe.model_daemon import DaemonSpec, OwnedModelDaemon
        settings = dict(settings)
        settings["_initial_random_state"] = capture_random_state()
        spec = DaemonSpec(backend=backend, factory=factory, settings=settings,
                          python_bin=python_bin or sys.executable, cwd=cwd, env=env,
                          max_frame_bytes=max_frame_bytes, provider=provider)
        self.owner = OwnedModelDaemon(spec, request_timeout=request_timeout,
                                      socket_path=socket_path,
                                      startup_timeout=startup_timeout,
                                      shutdown_timeout=shutdown_timeout,
                                      max_pending=max_pending)
        try:
            description = self.owner.request({"operation": "describe"})
            self.metadata = description["metadata"]
            restore_random_state(description["random_state"])
        except BaseException:
            self.owner.shutdown()
            raise

    def request(self, payload):
        response = self.owner.request({"operation": "infer", "payload": host_result(payload),
                                       "random_state": capture_random_state()})
        restore_random_state(response["random_state"])
        if not response["ok"]:
            raise RuntimeError(f"{response['error_type']}: {response['error']}")
        return response["result"]

    def shutdown(self):
        self.owner.shutdown()

    @property
    def pid(self):
        return self.owner.pid
