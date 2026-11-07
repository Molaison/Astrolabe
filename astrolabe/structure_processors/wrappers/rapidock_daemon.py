#!/usr/bin/env python3
"""JSONL daemon that owns one retained RAPiDock Python API object."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Dict

from astrolabe.model_execution import KernelStages, kernel_handler

try:
    from .rapidock_runtime import build_rapidock_runtime
except ImportError:
    from rapidock_runtime import build_rapidock_runtime


_PROTOCOL_STDOUT = sys.stdout


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAPiDock JSONL daemon.")
    parser.add_argument("--repo-path", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--confidence-model-dir")
    parser.add_argument("--confidence-ckpt", default="best_model.pt")
    parser.add_argument("--device")
    return parser.parse_args()


def _write(obj: Dict[str, Any]) -> None:
    _PROTOCOL_STDOUT.write(json.dumps(obj) + "\n")
    _PROTOCOL_STDOUT.flush()


def _error_response(message: Dict[str, Any], exc: Exception) -> Dict[str, Any]:
    return {
        "type": "error",
        "id": message.get("id"),
        "error_type": type(exc).__name__,
        "message": str(exc),
    }


def _create_runtime_handler(runtime, device):
    def load(_settings):
        runtime.load()
        if not runtime.status().get("model_ready"):
            raise RuntimeError("RAPiDock initialization did not produce a ready model")

        def prepare(message):
            tasks = message.get("tasks") or []
            params = message.get("params") or {}
            if not isinstance(tasks, list):
                raise ValueError("dock tasks must be a list.")
            if not isinstance(params, dict):
                raise ValueError("dock params must be an object.")
            return tasks, params

        # prepare_data includes ESM CUDA work and lazy confidence loading. Keep
        # the runtime's complete call together under the device execute lease.
        # ESM may retain CUDA allocator memory even with a CPU score model.
        # Register the conservative whole graph on its admitted GPU.
        return KernelStages(prepare, lambda args: runtime.run(*args)), {
            "device": device if str(device).startswith("cuda") else None,
        }

    return kernel_handler(
        {}, load,
        lifecycle={"backend": "rapidock", "residency": "resident_model",
                   "weights_loaded": True, "gpu_resident": None,
                   "device": device if str(device).startswith("cuda:") else None},
    )


def main() -> int:
    args = _parse_args()
    repo_path = pathlib.Path(args.repo_path).expanduser().resolve()
    model_dir = pathlib.Path(args.model_dir).expanduser().resolve()
    ckpt_path = model_dir / args.ckpt

    if not repo_path.exists():
        raise FileNotFoundError(f"RAPiDock repo path does not exist: {repo_path}")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"RAPiDock checkpoint not found: {ckpt_path}")

    confidence_model_dir = args.confidence_model_dir
    if confidence_model_dir not in (None, "", "null", "None"):
        confidence_path = pathlib.Path(confidence_model_dir).expanduser()
        confidence_ckpt_path = confidence_path / args.confidence_ckpt
        if not confidence_ckpt_path.is_file():
            raise FileNotFoundError(
                f"RAPiDock confidence checkpoint not found: {confidence_ckpt_path}"
            )
        confidence_model_dir = str(confidence_path.resolve())

    runtime = build_rapidock_runtime(
        repo_path=repo_path,
        model_dir=model_dir,
        ckpt=str(args.ckpt),
        confidence_model_dir=confidence_model_dir,
        confidence_ckpt=str(args.confidence_ckpt),
        device=args.device,
    )
    handler = None
    load_error = None
    try:
        handler = _create_runtime_handler(runtime, args.device)
    except Exception as exc:
        # Preserve the raw ping/error protocol, but never report a partially
        # initialized or unadmitted runtime as ready.
        load_error = f"{type(exc).__name__}: {exc}"

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue

        message: Dict[str, Any] = {}
        try:
            decoded = json.loads(line)
            if not isinstance(decoded, dict):
                raise ValueError("JSONL request must be an object.")
            message = decoded
            message_type = message.get("type")
            if message_type == "ping":
                status = runtime.status()
                if load_error is not None:
                    status = dict(status, model_ready=False, message=load_error)
                _write(
                    {
                        "type": "pong",
                        "id": message.get("id"),
                        **status,
                    }
                )
                continue
            if message_type == "shutdown":
                _write({"type": "ok", "id": message.get("id"), "message": "bye"})
                return 0
            if message_type != "dock":
                raise ValueError(f"unknown type: {message_type}")

            if handler is None:
                raise RuntimeError(f"RAPiDock model is not ready: {load_error}")
            response = handler({"operation": "infer", "payload": message,
                                "random_state": {}})
            if not response["ok"]:
                _write({"type": "error", "id": message.get("id"),
                        "error_type": response["error_type"],
                        "message": response["error"]})
            else:
                _write({"type": "result", "id": message.get("id"),
                        "results": response["result"]})
        except Exception as exc:
            _write(_error_response(message, exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
