"""Read-only broker ownership probe with controlled worker/residency boundaries."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    from astrolabe.runtime.broker_state import State
    from astrolabe.runtime.contracts import ResourceLimits, request_metadata

    request_metadata({})

    def run_case(waiter_timeout):
        entered, release = threading.Event(), threading.Event()

        class ControlledWorker:
            hello = {"worker_epoch": "controlled", "capabilities": {}}
            lifecycle = {}
            last_metrics = {}

            def call(self, payload, **kwargs):
                entered.set()
                if not release.wait(3.0):
                    raise RuntimeError("probe did not release controlled worker")
                return {"ok": True}

            def close(self):
                pass

        state = State("probe", 1, 1, limits=ResourceLimits(
            cpu_slots=1, ram_bytes=4096, prepared_bytes=4096,
            output_bytes=1024, pending_items=2,
        ))
        record = {"client": ControlledWorker(), "device": "cpu", "clients": {"client": None},
                  "busy": 0, "shapes": set(), "cohort": "controlled", "call_lock": threading.Lock(),
                  "retained": {}, "delegated": None, "last_used": time.monotonic()}
        state.workers["worker"] = record
        # No real model process is involved. Keep actual request admission and
        # cleanup; bypass only post-response process/lifecycle observations.
        state.ledger.update_resident = lambda *args, **kwargs: None
        state.ledger.observe_resident = lambda *args, **kwargs: None
        outcomes = {}

        def request(request_id, timeout):
            return {"epoch": "client", "request_id": request_id, "handle": "worker", "timeout": timeout,
                    "payload": {"value": 1}, "runtime": {"resources": {"ram_bytes": 0, "output_bytes": 0}}}

        def first_call():
            try:
                outcomes["first_result"] = state.call(request("first", 2.0))
            except BaseException as exc:
                outcomes["first_error"] = {"type": type(exc).__name__, "message": str(exc)}

        thread = threading.Thread(target=first_call)
        thread.start()
        try:
            assert entered.wait(1.0)
            outcomes["delegated_before_waiter"] = record["delegated"] is not None
            if waiter_timeout:
                try:
                    state.call(request("waiter", .03))
                except BaseException as exc:
                    outcomes["waiter_error"] = {"type": type(exc).__name__, "message": str(exc)}
            outcomes["delegated_after_waiter"] = record["delegated"] is not None
            outcomes["active_first_reservation_after_waiter"] = "client:first" in state.ledger.active
            outcomes["busy_after_waiter"] = record["busy"]
        finally:
            release.set()
            thread.join(timeout=3.0)
            assert not thread.is_alive()
            outcomes["pending_after_completion"] = state.ledger.snapshot()["pending"]
            outcomes["busy_after_completion"] = record["busy"]
            state.close()
        return outcomes

    report = {
        "repo": str(repo),
        "head": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
        "broker_source_sha256": hashlib.sha256((repo / "astrolabe/runtime/broker_state.py").read_bytes()).hexdigest(),
        "scope": "Actual broker admission/cleanup, controlled worker and post-response residency observations; no IPC/GPU/model execution",
        "control": run_case(False),
        "waiter_times_out": run_case(True),
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
