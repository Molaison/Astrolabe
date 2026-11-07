import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

from astrolabe.runtime.session import (
    IndependentCommand, resource_session, run_independent,
)


def test_independent_processes_keep_rng_namespaces_and_inventory_order(tmp_path, monkeypatch):
    monkeypatch.delenv("ASTROLABE_RESOURCE_SOCKET", raising=False)
    script = (
        "import json,os,random,time; random.seed(int(os.environ['SEED'])); "
        "time.sleep(float(os.environ['DELAY'])); "
        "print(json.dumps({'pid':os.getpid(),'draw':random.random()}))"
    )
    commands = [
        IndependentCommand(
            identity=f"trajectory-{index}", argv=(sys.executable, "-c", script),
            output_dir=tmp_path / str(index),
            env={"SEED": "71", "DELAY": delay},
        )
        for index, delay in enumerate(("0.15", "0"))
    ]
    results = run_independent(commands, max_trajectories=2)
    assert [item["identity"] for item in results] == [
        "trajectory-0", "trajectory-1"]
    assert all(item["state"] == "completed" for item in results)
    rows = [json.loads((item.output_dir / "stdout.log").read_text()) for item in commands]
    assert rows[0]["pid"] != rows[1]["pid"]
    assert rows[0]["draw"] == rows[1]["draw"]
    assert rows[0]["pid"] != os.getpid()


def test_output_collision_fails_before_any_command_starts(tmp_path):
    commands = [
        IndependentCommand(identity=name, argv=(sys.executable, "-c", "raise Exception()"),
                           output_dir=tmp_path / "shared")
        for name in ("a", "b")
    ]
    with pytest.raises(ValueError, match="unique"):
        run_independent(commands, max_trajectories=2)
    assert not (tmp_path / "shared").exists()


def test_timeout_is_unknown_and_never_replayed(tmp_path, monkeypatch):
    monkeypatch.delenv("ASTROLABE_RESOURCE_SOCKET", raising=False)
    marker = tmp_path / "count"
    command = IndependentCommand(
        identity="one", output_dir=tmp_path / "output",
        argv=(sys.executable, "-c",
              "import pathlib,time; pathlib.Path(" + repr(str(marker))
              + ").write_text('one'); time.sleep(100)"),
    )
    result = run_independent([command], max_trajectories=1, timeout=1)
    assert result[0]["state"] == "outcome_unknown"
    assert marker.read_text() == "one"
    with pytest.raises(FileExistsError):
        run_independent([command], max_trajectories=1)


def test_adaptive_requires_actual_allocation_before_model_load(monkeypatch):
    for name in ("ASTROLABE_RESOURCE_SOCKET", "CUDA_VISIBLE_DEVICES",
                 "SLURM_JOB_GPUS", "SLURM_STEP_GPUS"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="explicit device allocation"):
        with resource_session({"mode": "adaptive"}):
            pytest.fail("entered without allocation")


# Run the subreaper in a dedicated interpreter: pytest must never adopt unrelated
# workers' orphans. Only the sibling launch failure is injected; process groups,
# TERM/KILL, wait(), /proc observations and orphan reaping are real OS operations.
_GROUP_CLEANUP_SUPERVISOR = r'''
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from unittest import mock

from astrolabe.runtime import session

root = Path(sys.argv[1])
scenario = sys.argv[2]
manifest = root / "group.json"
launched = []
leader_waiting = threading.Event()
facts = None
libc = ctypes.CDLL(None, use_errno=True)
previous_subreaper = ctypes.c_int()
assert libc.prctl(37, ctypes.byref(previous_subreaper), 0, 0, 0) == 0
assert libc.prctl(36, 1, 0, 0, 0) == 0

def record(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_bytes().rsplit(b")", 1)[1].split()
    except FileNotFoundError:
        return None
    return {"pid": pid, "state": fields[0].decode("ascii"), "pgid": int(fields[2]),
            "epoch": fields[19].decode("ascii")}

def alive(original):
    current = record(original["pid"])
    return (current is not None and current["epoch"] == original["epoch"]
            and current["state"] not in ("Z", "X", "x"))

leader_code = r"""
import ctypes
import json
import os
from pathlib import Path
import signal
import sys
import time

def record(pid):
    fields = Path(f"/proc/{pid}/stat").read_bytes().rsplit(b")", 1)[1].split()
    return {"pid": pid, "pgid": int(fields[2]), "epoch": fields[19].decode("ascii")}

ready_read, ready_write = os.pipe()
descendant = os.fork()
if descendant == 0:
    os.close(ready_read)
    # Linux comm is arbitrary bytes, including for processes seen while scanning
    # another group. Exercise real non-UTF8 /proc data without changing pytest.
    assert ctypes.CDLL(None).prctl(15, ctypes.c_char_p(b"review-\xff"), 0, 0, 0) == 0
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    os.write(ready_write, b"ready")
    os.close(ready_write)
    time.sleep(60)
    os._exit(0)
os.close(ready_write)
assert os.read(ready_read, 5) == b"ready"
os.close(ready_read)
path = Path(sys.argv[1])
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps({"leader": record(os.getpid()),
                                "descendant": record(descendant)}))
temporary.replace(path)
# The leader keeps default SIGTERM behavior; its child explicitly ignores TERM.
time.sleep(60)
"""

real_popen = subprocess.Popen
def tracked_popen(argv, *args, **kwargs):
    if argv[0] == "review-inject-sibling-launch-error":
        # Throw only after the real same-PGID descendant has installed SIG_IGN.
        raise OSError("review sibling launch failure after descendant ready")
    process = real_popen(argv, *args, **kwargs)
    launched.append(process)
    real_wait = process.wait
    def observed_wait(*wait_args, **wait_kwargs):
        # run() enters wait only after adding this process to its owned set.
        leader_waiting.set()
        return real_wait(*wait_args, **wait_kwargs)
    process.wait = observed_wait
    return process

class ReadyExecutor(session.ThreadPoolExecutor):
    def submit(self, function, item):
        if item.identity == "bad-sibling":
            # Gate submission, not Popen: production may correctly serialize
            # process creation and ownership registration under one lock.
            deadline = time.monotonic() + 10
            while not manifest.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert manifest.exists(), "leader/descendant did not become ready"
            assert leader_waiting.wait(10), "leader was not registered for supervision"
        return super().submit(function, item)

commands = [session.IndependentCommand(
    "leader", (sys.executable, "-c", leader_code, str(manifest)), root / "leader")]
if scenario == "batch_exception":
    commands.append(session.IndependentCommand(
        "bad-sibling", ("review-inject-sibling-launch-error",), root / "bad-sibling"))

try:
    with mock.patch.object(session.subprocess, "Popen", side_effect=tracked_popen), \
            mock.patch.object(session, "ThreadPoolExecutor", ReadyExecutor):
        if scenario == "timeout":
            results = session.run_independent(commands, max_trajectories=1, timeout=2)
            assert results[0]["state"] == "outcome_unknown", results
        else:
            try:
                session.run_independent(commands, max_trajectories=2)
            except OSError as error:
                assert "review sibling launch failure" in str(error), str(error)
            else:
                raise AssertionError("sibling launch exception was not propagated")
    original = json.loads(manifest.read_text())
    assert original["leader"]["pgid"] == original["leader"]["pid"]
    assert original["descendant"]["pgid"] == original["leader"]["pgid"]
    assert len(launched) == 1
    assert launched[0].poll() == -signal.SIGTERM, "leader did not exit on TERM first"
    # Snapshot BEFORE our finally cleanup, so the test cannot pass on its own KILL.
    facts = {"scenario": scenario, "same_pgid": True,
             "leader_alive_after_runtime": alive(original["leader"]),
             "descendant_alive_after_runtime": alive(original["descendant"])}
finally:
    try:
        for process in launched:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
        deadline = time.monotonic() + 5
        while True:
            try:
                reaped, _ = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                break
            if reaped == 0:
                assert time.monotonic() < deadline, "owned orphan not reaped"
                time.sleep(0.01)
    finally:
        assert libc.prctl(36, previous_subreaper.value, 0, 0, 0) == 0

print(json.dumps(facts), flush=True)
assert not facts["leader_alive_after_runtime"], facts
assert not facts["descendant_alive_after_runtime"], facts
'''


def _kill_recorded_review_group(manifest):
    """Emergency cleanup is limited to a PID/epoch/PGID in this test's manifest."""
    if not manifest.exists():
        return
    original = json.loads(manifest.read_text())
    for member in original.values():
        try:
            fields = Path(f"/proc/{member['pid']}/stat").read_bytes().rsplit(b")", 1)[1].split()
        except FileNotFoundError:
            continue
        if fields[19].decode("ascii") == member["epoch"] and int(fields[2]) == member["pgid"]:
            try:
                os.killpg(member["pgid"], signal.SIGKILL)
            except ProcessLookupError:
                pass
            return


@pytest.mark.skipif(sys.platform != "linux", reason="Linux scoped subreaper and /proc")
@pytest.mark.parametrize("scenario", ["timeout", "batch_exception"])
def test_cleanup_waits_for_same_pgid_descendant_after_leader_exits(
        tmp_path, monkeypatch, scenario):
    monkeypatch.delenv("ASTROLABE_RESOURCE_SOCKET", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    # Subreaper state is changed only in this dedicated supervisor process.
    supervisor = subprocess.Popen(
        [sys.executable, "-c", _GROUP_CLEANUP_SUPERVISOR, str(tmp_path), scenario],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        try:
            stdout, stderr = supervisor.communicate(timeout=40)
        except subprocess.TimeoutExpired:
            # Leave the subreaper alive while killing its owned group, so it can
            # finish its finally block and reap the orphan before we stop it.
            _kill_recorded_review_group(tmp_path / "group.json")
            try:
                stdout, stderr = supervisor.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                supervisor.kill()
                stdout, stderr = supervisor.communicate(timeout=5)
            pytest.fail(f"cleanup supervisor timed out\n{stdout}\n{stderr}")
        assert supervisor.returncode == 0, f"{stdout}\n{stderr}"
        facts = json.loads(stdout)
        assert facts["same_pgid"] is True
        assert facts["leader_alive_after_runtime"] is False
        assert facts["descendant_alive_after_runtime"] is False
    finally:
        _kill_recorded_review_group(tmp_path / "group.json")
        if supervisor.poll() is None:
            supervisor.kill()
            supervisor.wait(timeout=5)
