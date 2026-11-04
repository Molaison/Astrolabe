"""Independent root admission-cleanup probe; real State and ledger, no worker execution."""
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

source = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(source))
from astrolabe.runtime.broker_state import State
from astrolabe.runtime.contracts import ResourceLimits

class NoDispatch:
    hello = {"worker_epoch": "root-controlled", "capabilities": {}}
    lifecycle = {}
    last_metrics = {}
    def __init__(self):
        self.dispatches = 0
    def call(self, payload, **kwargs):
        self.dispatches += 1
        raise AssertionError("expired request must not reach worker")
    def close(self):
        pass

with tempfile.TemporaryDirectory(prefix="astrolabe-root-admission-") as temp:
    os.environ["ASTROLABE_RUNTIME_TRACE"] = str(Path(temp) / "trace.jsonl")
    state = State("probe", 1, 1, limits=ResourceLimits(
        cpu_slots=1, ram_bytes=4096, prepared_bytes=4096, output_bytes=1024, pending_items=2))
    client = NoDispatch()
    state.workers["worker"] = {
        "client": client, "device": "cpu", "clients": {"client": None},
        "busy": 0, "shapes": set(), "cohort": "root-controlled", "call_lock": threading.Lock(),
        "retained": {}, "delegated": None, "last_used": time.monotonic()}
    acquire = state.ledger.acquire_compute
    def delayed_acquire(*args, **kwargs):
        acquire(*args, **kwargs)
        # Expire the actual monotonic deadline after real successful admission.
        # This is a controlled scheduler delay, not a performance measurement.
        time.sleep(.2)
    state.ledger.acquire_compute = delayed_acquire
    error = None
    start = time.monotonic()
    try:
        state.call({"epoch": "client", "request_id": "request", "handle": "worker",
            "timeout": .1, "payload": {"value": 1},
            "runtime": {"resources": {"ram_bytes": 0, "output_bytes": 0}}})
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    elapsed = time.monotonic()-start
    snapshot = state.ledger.snapshot()
    observed = {"source": str(source), "error": error, "elapsed_seconds": elapsed,
                "waiting": state.metrics["waiting"], "busy": state.workers["worker"]["busy"],
                "dispatches": client.dispatches, "ledger": snapshot,
                "events": [json.loads(line) for line in (Path(temp)/"trace.jsonl").read_text().splitlines()]}
    state.close()
    print(json.dumps(observed, indent=2))
    assert error and error["type"] == "TimeoutError", error
    assert client.dispatches == 0
    assert observed["busy"] == 0
    assert snapshot["pending"] == 0 and snapshot["cpu_slots"] == 0
    assert observed["waiting"] == 0, "waiting metric must retire the expired admitted request"

