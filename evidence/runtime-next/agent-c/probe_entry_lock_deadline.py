"""Read-only scope probe: State.call timeout after _track_request lock waiting."""
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
    parser.add_argument("--timeout-seconds", type=float, default=.03)
    parser.add_argument("--lock-hold-seconds", type=float, default=.12)
    args = parser.parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    from astrolabe.runtime.broker_state import State
    from astrolabe.runtime.contracts import ResourceLimits, request_metadata
    request_metadata({})
    dispatches = []
    result = {}
    entered = threading.Event()

    class DispatchReached(Exception):
        pass

    class Worker:
        hello = {"worker_epoch": "controlled", "capabilities": {}}
        lifecycle = {}

        def call(self, payload, **kwargs):
            dispatches.append({"at": time.monotonic(), "remaining": kwargs["timeout"]})
            raise DispatchReached("controlled boundary, no worker/model execution")

        def close(self):
            pass

    state = State("probe", 1, 1, limits=ResourceLimits(
        cpu_slots=1, ram_bytes=4096, prepared_bytes=4096, output_bytes=1024, pending_items=2,
    ))
    state.workers["worker"] = {"client": Worker(), "device": "cpu", "clients": {"client": None},
        "busy": 0, "shapes": set(), "cohort": "controlled", "call_lock": threading.Lock(),
        "retained": {}, "delegated": None, "last_used": time.monotonic()}
    track = state._track_request

    def observe_track(request):
        entered.set()
        return track(request)

    state._track_request = observe_track
    request = {"epoch": "client", "request_id": "one", "handle": "worker", "timeout": args.timeout_seconds,
               "payload": {}, "runtime": {"resources": {"ram_bytes": 0, "output_bytes": 0}}}

    def call():
        start = time.monotonic()
        try:
            state.call(request)
        except BaseException as exc:
            result["exception"] = type(exc).__name__
            result["message"] = str(exc)
        result["elapsed_seconds"] = time.monotonic() - start
        for dispatch in dispatches:
            dispatch["seconds_after_call_entry"] = dispatch.pop("at") - start

    with state.lock:
        thread = threading.Thread(target=call)
        thread.start()
        assert entered.wait(1.0)
        threading.Event().wait(args.lock_hold_seconds)
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    result.update({
        "request_timeout_seconds": request["timeout"],
        "dispatches": dispatches,
        "pending_after_completion": state.ledger.snapshot()["pending"],
        "head": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
        "broker_source_sha256": hashlib.sha256((repo / "astrolabe/runtime/broker_state.py").read_bytes()).hexdigest(),
        "scope": "Actual State entry lock wait, controlled worker boundary; no IPC/GPU/model execution",
    })
    state.close()
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
