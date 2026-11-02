"""Process-local serialization shared by the legacy scientific adapters."""
import threading
from functools import wraps

SCIENTIFIC_OWNER = threading.RLock()


def scientific_session(function):
    """Serialize an entry point with other users of the same scientific owner."""
    @wraps(function)
    def execute(*args, **kwargs):
        with SCIENTIFIC_OWNER:
            return function(*args, **kwargs)

    return execute
