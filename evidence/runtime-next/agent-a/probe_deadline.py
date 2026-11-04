#!/usr/bin/env python3
"""Real State/ResourceLedger waits; controlled worker boundary, no IPC/GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    repo, out = args.repo.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(repo))
    os.environ["ASTROLABE_RUNTIME_TRACE"] = str(out / "runtime.jsonl")
    from astrolabe.runtime.broker_state import State
    from astrolabe.runtime.contracts import ResourceLimits, request_metadata

    class DispatchReached(Exception):
        pass

    class ControlledWorker:
        hello = {"worker_epoch": "controlled-worker", "capabilities": {}}
        lifecycle = {}
        dispatches = []

        def call(self, payload, **kwargs):
            self.dispatches.append({"monotonic": time.monotonic(), "worker_timeout": kwargs["timeout"]})
            raise DispatchReached("controlled client boundary reached; no scientific worker ran")

        def close(self):
            pass

    # Resolve metadata imports before starting the timing observation.
    request_metadata({})
    client = ControlledWorker()
    state = State("probe", 1, 1, limits=ResourceLimits(
        cpu_slots=1, ram_bytes=4096, prepared_bytes=4096, output_bytes=1024, pending_items=2))
    state.workers["worker"] = {"client": client, "device": "cpu", "clients": {"client": None},
        "busy": 0, "shapes": set(), "cohort": "controlled-boundary", "call_lock": threading.Lock(),
        "retained": {}, "delegated": None, "last_used": time.monotonic()}
    zero = {"ram_bytes": 0, "output_bytes": 0}
    state.ledger.admit_pending("compute-blocker", zero)
    state.ledger.acquire_compute("compute-blocker", "cpu")
    state.ledger.admit_pending("pending-blocker", zero)
    entered, admitted, cancelled = threading.Event(), threading.Event(), threading.Event()
    timings = {}
    admit_pending = state.ledger.admit_pending

    def observed_admit(token, resources, **kwargs):
        if token == "client:request":
            timings["pending_entered"] = time.monotonic()
            entered.set()
        result = admit_pending(token, resources, **kwargs)
        if token == "client:request":
            timings["pending_admitted"] = time.monotonic()
            admitted.set()
        return result

    state.ledger.admit_pending = observed_admit

    def release_in_order():
        if entered.wait(2) and not cancelled.wait(.6):
            state.ledger.release("pending-blocker")
            if admitted.wait(2) and not cancelled.wait(.7):
                state.ledger.release("compute-blocker")

    thread = threading.Thread(target=release_in_order, daemon=True)
    thread.start()
    request = {"epoch": "client", "request_id": "request", "handle": "worker", "timeout": 1.0,
               "payload": {"value": 1}, "runtime": {"resources": zero}}
    start = time.monotonic()
    try:
        state.call(request)
    except Exception as exc:
        result = {"exception": type(exc).__name__, "message": str(exc)}
    else:
        result = {"exception": None}
    elapsed = time.monotonic() - start
    result.update(kind="real_control_plane_waits_with_controlled_worker_boundary",
                  scientific_worker_executed=False, gpu_executed=False,
                  repository=str(repo), commit=subprocess.check_output(
                      ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
                  broker_source_sha256=hashlib.sha256((repo / "astrolabe/runtime/broker_state.py").read_bytes()).hexdigest(),
                  request_timeout_seconds=1.0, elapsed_seconds=elapsed,
                  pending_wait_seconds=timings.get("pending_admitted", start) - timings.get("pending_entered", start),
                  dispatched=bool(client.dispatches), dispatches=client.dispatches,
                  busy=state.workers["worker"]["busy"], waiting=state.metrics["waiting"],
                  request_reservation_retained="client:request" in state.ledger.pending)
    source_diff = subprocess.check_output(["git", "-C", str(repo), "diff", "HEAD", "--",
                                           "astrolabe/runtime/broker_state.py"], text=True)
    result["broker_source_dirty"] = bool(source_diff)
    if source_diff:
        (out / "source.patch").write_text(source_diff)
    cancelled.set()
    state.ledger.release("pending-blocker")
    state.ledger.release("compute-blocker")
    thread.join(timeout=3)
    try:
        state.call(request)
    except ValueError as exc:
        result["same_request_id_rejected"] = str(exc) == "duplicate request"
    finally:
        state.close()
    events = [json.loads(line) for line in (out / "runtime.jsonl").read_text().splitlines()]
    queued = next(row["time"] for row in events if row["event"] == "request_queued")
    for dispatch in result["dispatches"]:
        dispatch["seconds_after_queued_event"] = dispatch["monotonic"] - queued
    result["pending_after_cleanup"] = state.ledger.snapshot()["pending"]
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
