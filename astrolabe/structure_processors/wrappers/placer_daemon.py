from __future__ import annotations

import hashlib
import importlib
import inspect
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from astrolabe.model_execution import (
    ExecutionMode,
    KernelStages,
    NativeModelClient,
    capture_random_state,
    execution_mode,
    kernel_handler,
    restore_random_state,
)


PLACER_DAEMON_FACTORY = (
    "astrolabe.structure_processors.wrappers.placer_daemon:create_handler"
)
PLACER_AUDITED_REVISION = "7dc5563300e7143582d85636bdd413d4c7688e9a"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
def _source_revision(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _load_placer(settings: dict[str, Any]):
    try:
        module = importlib.import_module("PLACER")
    except ImportError as exc:
        raise ImportError(
            "PLACER is a manually managed dependency and is not importable. "
            "Initialize its existing installation manually."
        ) from exc
    placer_type = getattr(module, "PLACER", None)
    input_type = getattr(module, "PLACERinput", None)
    if not inspect.isclass(placer_type) or not inspect.isclass(input_type):
        raise TypeError("PLACER module must expose PLACER and PLACERinput classes.")
    run_signature = inspect.signature(placer_type.run)
    if "input_object" not in run_signature.parameters or "nsamples" not in run_signature.parameters:
        raise TypeError(
            "PLACER.PLACER.run must accept input_object and nsamples."
        )

    origin = Path(module.__file__).resolve()
    weights_path = settings.get("weights_path")
    configured_weights = (
        Path(str(weights_path)).expanduser().resolve()
        if weights_path
        else origin.parent / "weights" / "PLACER_model_1.pt"
    )
    if os.environ.get("ASTROLABE_RESOURCE_SOCKET"):
        # The audited upstream constructor hardcodes cuda:0; it has no device
        # argument. Do not reserve cuda:N (or CPU) and then load on cuda:0.
        import torch
        actual_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        requested_device = settings.get("torch_device", "auto")
        if requested_device not in ("auto", "cuda", actual_device):
            raise ValueError(
                f"PLACER selects {actual_device}; requested {requested_device}. "
                "Use a matching CUDA_VISIBLE_DEVICES mapping for this upstream."
            )
    placer = placer_type(str(configured_weights)) if weights_path else placer_type()
    device_getter = getattr(placer, "device", None)
    device = str(device_getter()) if callable(device_getter) else None
    metadata = {
        "backend": "chemnet_placer",
        "implementation": "PLACER",
        "dependency_management": "manual",
        "install_automatically": False,
        "audited_revision": PLACER_AUDITED_REVISION,
        "module_origin": str(origin),
        "module_sha256": _sha256(origin),
        "source_revision": _source_revision(origin.parent),
        "python_bin": sys.executable,
        "pythonpath": os.environ.get("PYTHONPATH", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_device": settings.get("torch_device", "auto"),
        "weights_path": str(configured_weights),
        "weights_exists": configured_weights.is_file(),
        "device": device,
    }

    def prepare(request: Mapping[str, Any]):
        if set(request) != {"input_file", "num_samples"}:
            raise ValueError(
                "PLACER daemon request must contain only input_file and num_samples."
            )
        input_file = Path(str(request["input_file"])).expanduser().resolve()
        if not input_file.is_file():
            raise FileNotFoundError(f"PLACER input file not found: {input_file}")
        pl_input = input_type()
        if input_file.suffix == ".pdb":
            pl_input.pdb(str(input_file))
        elif input_file.suffix == ".cif":
            pl_input.cif(str(input_file))
        else:
            raise ValueError(
                f"PLACER supports .pdb and .cif inputs, got {input_file.suffix!r}."
            )
        pl_input.name(input_file.name)
        return pl_input, int(request["num_samples"])

    def execute(prepared):
        return placer.run(*prepared)

    def finalize(outputs):
        ordered = []
        for key, value in outputs.items():
            ordered.append(
                {
                    "key": key,
                    "prmsd": value["prmsd"],
                    "model": value["model"],
                }
            )
        return {"outputs": ordered}

    resource = "cpu" if settings.get("torch_device") == "cpu" or device == "cpu" else "gpu"
    return KernelStages(prepare, execute, finalize, resource=resource), metadata


def create_handler(settings: dict[str, Any]):
    return kernel_handler(
        settings,
        _load_placer,
        torch_device=settings.get("torch_device") or "auto",
        lifecycle={"backend": "chemnet_placer", "residency": "resident_model",
                   "weights_loaded": True, "gpu_resident": None},
    )


class PlacerClient:
    backend = "chemnet_placer"

    def __init__(
        self,
        settings: Mapping[str, Any],
        *,
        mode: ExecutionMode | str | None = None,
        python_bin: str | None = None,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        request_timeout: float | None = None,
    ) -> None:
        self.settings = dict(settings)
        self.settings["torch_device"] = self.settings.get("torch_device") or "auto"
        self.mode = execution_mode(mode)
        self._kernel = None
        self._native = None
        if self.mode is ExecutionMode.DAEMON:
            self._native = NativeModelClient(
                self.backend,
                PLACER_DAEMON_FACTORY,
                self.settings,
                python_bin=python_bin,
                cwd=cwd,
                env=dict(env or {}),
                request_timeout=request_timeout,
            )
            self.metadata = self._native.metadata
        else:
            self._kernel = create_handler(self.settings)
            description = self._kernel({"operation": "describe"})
            self.metadata = description["metadata"]
            restore_random_state(description["random_state"])

    @property
    def pid(self) -> int | None:
        return self._native.pid if self._native is not None else os.getpid()

    def predict(self, payload: Mapping[str, Any]) -> list[dict[str, Any]]:
        if self._native is not None:
            result = self._native.request(dict(payload))
        else:
            response = self._kernel(
                {
                    "operation": "infer",
                    "payload": dict(payload),
                    "random_state": capture_random_state(),
                }
            )
            restore_random_state(response["random_state"])
            if not response["ok"]:
                raise RuntimeError(
                    f"{response['error_type']}: {response['error']}"
                )
            result = response["result"]
        outputs = result.get("outputs")
        if not isinstance(outputs, list):
            raise TypeError("PLACER backend response must contain an outputs list.")
        return outputs

    def shutdown(self) -> None:
        if self._native is not None:
            self._native.shutdown()
