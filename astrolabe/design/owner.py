"""The explicit serial owner of legacy scientific globals and global RNG."""
import threading
from functools import wraps

SCIENTIFIC_OWNER = threading.RLock()


def scientific_session(function):
    """Protect native setup, execution and cleanup with the shared owner lock."""
    @wraps(function)
    def execute(*args, **kwargs):
        with SCIENTIFIC_OWNER:
            return function(*args, **kwargs)

    return execute
