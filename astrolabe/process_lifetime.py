"""Parent lifetime protection shared by native workers and model services."""
from __future__ import annotations

import os


def protect_worker_lifetime():
    """Terminate a Linux child when its owner exits, including startup races."""
    import ctypes
    import signal

    parent = os.getppid()
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Cannot set worker parent-death signal")
    if os.getppid() != parent:
        os._exit(1)
