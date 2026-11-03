"""Bounded JSONL I/O for private model subprocesses.

EOF, partial-line timeout, and process exit are terminal. Callers must not
silently re-submit an inference whose result may already have been produced.
"""
from __future__ import annotations

import json
import os
import select
import subprocess
import time


def read_message(proc, buffer, *, timeout, max_bytes=16 * 1024**2):
    deadline = None if timeout is None else time.monotonic() + float(timeout)
    while True:
        newline = buffer.find(b"\n")
        if newline >= 0:
            raw = bytes(buffer[:newline])
            del buffer[:newline + 1]
            if len(raw) > max_bytes:
                raise ValueError("private model response exceeds byte budget")
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError("private model response must be a JSON object")
            return value
        if len(buffer) > max_bytes:
            raise ValueError("private model response exceeds byte budget")
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            raise TimeoutError("private model response timed out")
        if proc.stdout is None:
            raise RuntimeError("private model stdout is unavailable")
        ready, _, _ = select.select([proc.stdout.fileno()], [], [], remaining)
        if not ready:
            raise TimeoutError("private model response timed out")
        part = os.read(proc.stdout.fileno(), min(65536, max_bytes + 1 - len(buffer)))
        if not part:
            raise EOFError("private model process exited before a complete response")
        buffer.extend(part)


def stop_process(proc, *, timeout=5):
    """Observe process exit before reporting private residency released."""
    if proc is None:
        return
    if proc.poll() is None:
        proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=timeout)
    for stream in (getattr(proc, "stdin", None), getattr(proc, "stdout", None)):
        if stream is not None:
            stream.close()
