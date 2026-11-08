"""One-request stdio transport; transport failure never authorizes replay."""
from __future__ import annotations

import json
import math
import os
import select
import time

from astrolabe.model_daemon import DaemonStartupError, RequestOutcomeUnknown


def positive_timeout(value, *, default=None):
    if value is None:
        return default
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("daemon timeout must be finite and positive")
    return value


def json_line_request(proc, payload, *, timeout=None):
    """Freeze once, send once, and bind the complete response to this request.

    Callers serialize access. Raw nonblocking descriptors bound both partial
    writes and partial lines; TextIO.readline after select could still hang.
    An error after the first write attempt may have executed remotely. Legacy
    daemon errors have no terminal/RNG proof and are conservatively unknown.
    """
    frame = (json.dumps(payload) + "\n").encode("utf-8")
    request_id = payload.get("id")
    scientific = payload.get("type") in {"infer", "dock"}
    if proc.poll() is not None:
        raise DaemonStartupError("stdio daemon exited before request send")
    incoming, outgoing = proc.stdout.fileno(), proc.stdin.fileno()
    original = {fd: os.get_blocking(fd) for fd in (incoming, outgoing)}
    deadline = None if timeout is None else time.monotonic() + timeout
    attempted = False
    active_error = None

    def ready(*, write=False):
        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        readable, writable, _ = select.select(
            [] if write else [incoming], [outgoing] if write else [], [], remaining)
        if not (readable or writable):
            raise TimeoutError("stdio daemon request deadline exceeded")

    try:
        for fd in original:
            os.set_blocking(fd, False)
        view = memoryview(frame)
        while view:
            ready(write=True)
            # Even a failed write can have partially delivered a request.
            attempted = True
            try:
                written = os.write(outgoing, view)
            except BlockingIOError:
                continue
            if written == 0:
                raise BrokenPipeError("zero-byte stdio write")
            view = view[written:]
        response = bytearray()
        while not response.endswith(b"\n"):
            ready()
            try:
                chunk = os.read(incoming, 65536)
            except BlockingIOError:
                continue
            if not chunk:
                raise EOFError("stdio daemon closed before response completion")
            response.extend(chunk)
        result = json.loads(response)
        if not isinstance(result, dict) or result.get("id") != request_id:
            raise ValueError("stdio response request identity mismatch")
        if result.get("type") == "error":
            raise RuntimeError(str(result.get("message") or "daemon returned an unproven failure"))
        return result
    except BaseException as error:
        active_error = error
        if isinstance(error, Exception) and scientific and attempted:
            active_error = RequestOutcomeUnknown(
                f"stdio request outcome unknown; not replayed: {error}",
                request_id=request_id, generation=str(proc.pid),
            )
            raise active_error from error
        raise
    finally:
        restore_error = None
        for fd, blocking in original.items():
            try:
                os.set_blocking(fd, blocking)
            except OSError as error:
                restore_error = error
        if restore_error is not None:
            if active_error is not None:
                active_error.add_note(f"stdio descriptor restoration failed: {restore_error!r}")
            elif scientific and attempted:
                raise RequestOutcomeUnknown(
                    "stdio request completed with unresolved descriptor state",
                    request_id=request_id, generation=str(proc.pid),
                ) from restore_error
            else:
                raise restore_error


def stop_owned_stdio_process(proc):
    """Use the existing owned-process-group teardown and retain errors.

    Only for children created with start_new_session=True. A successful return
    proves no live group members remain, including descendants of an exited
    leader. The caller clears its last handle only after this returns.
    """
    from astrolabe.runtime.session import _terminate_group

    _terminate_group(proc)
    for stream in (proc.stdin, proc.stdout):
        if stream is not None:
            stream.close()
