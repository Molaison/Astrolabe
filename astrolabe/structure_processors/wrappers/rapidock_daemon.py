#!/usr/bin/env python3
"""JSONL daemon that owns one retained RAPiDock Python API object."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Dict

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
    try:
        runtime.load()
    except Exception:
        pass

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
                _write(
                    {
                        "type": "pong",
                        "id": message.get("id"),
                        **runtime.status(),
                    }
                )
                continue
            if message_type == "shutdown":
                _write({"type": "ok", "id": message.get("id"), "message": "bye"})
                return 0
            if message_type != "dock":
                raise ValueError(f"unknown type: {message_type}")

            tasks = message.get("tasks") or []
            params = message.get("params") or {}
            if not isinstance(tasks, list):
                raise ValueError("dock tasks must be a list.")
            if not isinstance(params, dict):
                raise ValueError("dock params must be an object.")
            results = runtime.run(tasks, params)
            _write({"type": "result", "id": message.get("id"), "results": results})
        except Exception as exc:
            _write(_error_response(message, exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
