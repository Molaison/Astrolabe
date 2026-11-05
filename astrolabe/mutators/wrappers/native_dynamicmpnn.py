from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from astrolabe.model_execution import kernel_handler


DYNAMICMPNN_DAEMON_FACTORY = (
    "astrolabe.mutators.wrappers.native_dynamicmpnn:create_handler"
)
def _resolve_callable(path: str) -> Any:
    module_name, separator, attr_path = str(path).strip().partition(":")
    if not separator:
        module_name, _, attr_path = str(path).strip().rpartition(".")
    if not module_name or not attr_path:
        raise ValueError(
            "DynamicMPNN backend_factory must use 'module:callable' or "
            "'module.callable'."
        )
    value: Any = importlib.import_module(module_name)
    for part in attr_path.split("."):
        value = getattr(value, part)
    if not callable(value):
        raise TypeError(f"DynamicMPNN backend_factory is not callable: {path!r}")
    return value


def _load_backend(settings: dict[str, Any]):
    factory_path = str(settings.get("backend_factory") or "").strip()
    if not factory_path:
        raise ValueError(
            "DynamicMPNN daemon execution requires backend_factory."
        )
    factory_settings = {
        key: value for key, value in settings.items() if not key.startswith("_")
    }
    backend = _resolve_callable(factory_path)(factory_settings)
    infer = getattr(backend, "infer", None)
    if infer is not None and not callable(infer):
        raise TypeError("DynamicMPNN resident backend infer attribute is not callable.")
    if infer is None and not callable(backend):
        raise TypeError(
            "DynamicMPNN backend_factory must return a callable or an object "
            "with infer(manifest=..., features=...)."
        )

    def run(request: Mapping[str, Any]):
        if set(request) != {"manifest", "features"}:
            raise ValueError(
                "DynamicMPNN daemon request must contain exactly manifest and features."
            )
        kwargs = {
            "manifest": request["manifest"],
            "features": request["features"],
        }
        return infer(**kwargs) if infer is not None else backend(**kwargs)

    metadata = {
        "backend": "dynamicmpnn",
        "implementation": "Alex-Abrudan/DynamicMPNN",
        "backend_factory": factory_path,
        "resident_backend_type": type(backend).__qualname__,
    }
    return run, metadata


def create_handler(settings: dict[str, Any]):
    return kernel_handler(
        settings,
        _load_backend,
        torch_device=settings.get("torch_device"),
    )
