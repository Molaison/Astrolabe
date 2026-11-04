#!/usr/bin/env python3
"""Run the two frozen native correctness processes once, sequentially.

Host controller is compatible with Python 3.6. It must be launched only after
the A-owned profiling slot has ended. Every failed/unknown attempt stops the pair.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import traceback


def save(path, obj):
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    os.replace(str(tmp), str(path))


def cpu():
    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    s = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime + s.ru_utime + s.ru_stime


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task-root", type=Path, required=True)
    ap.add_argument("--stage", type=Path, required=True)
    a = ap.parse_args()
    a.task_root, a.stage = a.task_root.resolve(), a.stage.resolve()
    assert a.task_root in a.stage.parents
    protocol = json.loads((a.stage / "native-control.json").read_text())
    out = a.stage / "execution"
    out.mkdir()  # Exclusive: a second execution must never silently replace a trial.
    state = {"status": "RUNNING", "controller_pid": os.getpid(), "attempts": [],
        "controller_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256((a.stage / "native-control.json").read_bytes()).hexdigest(),
        "scope": "native CPU correctness, no performance comparison"}
    start_cpu = cpu()
    start_wall = time.monotonic()
    save(out / "state.json", state)
    try:
        plans = [
            ("baseline", a.task_root / "startup-baseline-47c6f6d", protocol["baseline_commit"]),
            ("candidate", a.task_root / "startup-candidate-f9ce87b", protocol["candidate_commit"]),
        ]
        for label, source, head in plans:
            assert cpu() - start_cpu < 45.0, "reserve a full native-process budget before starting the next attempt"
            output = out / (label + ".json")
            cmd = [str(a.task_root / "control/test-container-cpu"), str(source),
                str(a.stage / "probe_native_startup.py"), "--repo", str(source),
                "--expect-head", head, "--protocol", str(a.stage / "native-control.json"),
                "--output", str(output)]
            record = {"label": label, "command": cmd, "started_monotonic": time.monotonic(),
                "status": "RUNNING"}
            state["attempts"].append(record)
            save(out / "state.json", state)
            env = os.environ.copy()
            env.update({"PYTHONDONTWRITEBYTECODE": "1", "GIT_OPTIONAL_LOCKS": "0"})
            private_tmp = out / (label + "-tmp")
            private_tmp.mkdir()
            env["TMPDIR"] = str(private_tmp)
            env["APPTAINER_TMPDIR"] = str(private_tmp)
            before_cpu = cpu()
            with (out / (label + ".stdout.log")).open("xb") as stdout, (out / (label + ".stderr.log")).open("xb") as stderr:
                process = subprocess.Popen(cmd, cwd=str(source), env=env, stdout=stdout, stderr=stderr, start_new_session=True)
                record["pid"] = process.pid
                save(out / "state.json", state)
                try:
                    code = process.wait(timeout=int(protocol["budgets"]["wall_seconds_per_process"]))
                except subprocess.TimeoutExpired:
                    record["timeout"] = True
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        code = process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        code = process.wait(timeout=2)
                try:
                    os.killpg(process.pid, 0)
                except ProcessLookupError:
                    record["process_group_released"] = True
                else:
                    record["process_group_released"] = False
                    os.killpg(process.pid, signal.SIGKILL)
            record.update({"exit_code": code, "wall_seconds": time.monotonic() - record["started_monotonic"],
                "observed_host_child_and_controller_cpu_seconds": cpu() - before_cpu})
            if output.exists():
                record["output_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
                record["native_status"] = json.loads(output.read_text()).get("status")
            record["status"] = "PASS" if code == 0 and record.get("native_status") == "PASS" and record["process_group_released"] and not record.get("timeout") else "FAIL"
            save(out / "state.json", state)
            if record["status"] != "PASS":
                raise RuntimeError("native attempt failed; stop without retry: " + label)
        compare_cmd = [sys.executable, str(a.stage / "compare_native_startup.py"), str(out / "baseline.json"),
            str(out / "candidate.json"), "--output", str(out / "comparison.json")]
        with (out / "comparison.stdout.log").open("xb") as stdout, (out / "comparison.stderr.log").open("xb") as stderr:
            completed = subprocess.run(compare_cmd, stdout=stdout, stderr=stderr, timeout=10)
        assert completed.returncode == 0, "native pair comparison failed; keep raw outputs"
        state["status"] = "PASS"
    except BaseException as exc:
        state["status"] = "FAIL"
        state["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    finally:
        state["observed_total_host_cpu_seconds"] = cpu() - start_cpu
        state["observed_total_wall_seconds"] = time.monotonic() - start_wall
        if state["observed_total_host_cpu_seconds"] > 90:
            state["status"] = "FAIL"
            state["budget_exceeded"] = True
        save(out / "state.json", state)
    print(json.dumps({"status": state["status"], "output": str(out), "attempts": len(state["attempts"])}))
    return 0 if state["status"] == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
