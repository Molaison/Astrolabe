"""Test-only startup boundary observer, injected through PYTHONPATH."""
import atexit
import importlib.abc
import json
import os
import sys
import time

_events = []
_armed = False
_forbidden = {"torch", "jax", "pyrosetta", "tensorflow", "alphafold", "alphafold3", "run"}


class StartupBoundary(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        global _armed
        if fullname == "astrolabe.cli":
            _armed = True
            _events.append({"event": "cli_entered", "time": time.monotonic()})
        if fullname.split(".")[0] in _forbidden:
            _events.append({"event": "forbidden_import", "module": fullname})
            raise RuntimeError("FORMAL_CONFIG_BOUNDARY: heavy import " + fullname)
        return None


def audit(event, args):
    if _armed and event in {"subprocess.Popen", "os.fork", "os.posix_spawn", "socket.connect"}:
        _events.append({"event": "forbidden_effect", "operation": event})
        raise RuntimeError("FORMAL_CONFIG_BOUNDARY: " + event)


def save():
    path = os.environ.get("ASTROLABE_CONFIG_BOUNDARY_LOG")
    if path:
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps({"pid": os.getpid(), "events": _events}) + "\n")


sys.meta_path.insert(0, StartupBoundary())
sys.addaudithook(audit)
atexit.register(save)
