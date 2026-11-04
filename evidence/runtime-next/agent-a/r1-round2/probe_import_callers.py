"""Observe real Biopython import requests; preserve the original import behavior."""
import builtins
import json
import sys
import traceback

sys.path.insert(0, sys.argv[1])
original = builtins.__import__
observations = []


def observe(name, globals=None, locals=None, fromlist=(), level=0):
    if name in ("Bio.SeqIO", "Bio.PDB") or name == "Bio" and "SeqIO" in (fromlist or ()):
        stack = traceback.extract_stack()[:-1]
        observations.append({"import": name, "fromlist": list(fromlist or ()),
                             "callers": [{"file": frame.filename, "line": frame.lineno,
                                          "function": frame.name, "code": frame.line}
                                         for frame in stack if "/astrolabe/" in frame.filename
                                         or "/Bio/" in frame.filename or frame.filename.endswith("run.py")]})
    return original(name, globals, locals, fromlist, level)


builtins.__import__ = observe
try:
    from astrolabe.app import _load_run_module
    _load_run_module()
finally:
    builtins.__import__ = original
print(json.dumps({"scope": "transparent import-call observation; no dependency replaced, no scientific evaluations",
                  "scientific_evaluation_count": 0, "observations": observations}, indent=2))
