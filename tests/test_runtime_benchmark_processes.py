"""CPU process ownership evidence; these tests do not certify GPU performance."""
from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import pytest

from scripts.validation import runtime_benchmark_processes as processes
from astrolabe.runtime.telemetry import ProcessTree, process_identity, process_namespace_supported


real_processes = pytest.mark.skipif(
    not process_namespace_supported(), reason="executor PID API and /proc namespaces disagree"
)


def _wait_for(predicate, seconds=3):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    pytest.fail("process fixture did not reach its synchronization point")


def _finish_fixture(process, observer, worker=None):
    observer.request_stop("test_fixture_cleanup")
    if worker is not None:
        worker.join(timeout=3)
    # Each remaining PID is checked against its original start time. Never
    # kill an unverified numerical PGID while cleaning a failed assertion.
    for identity in observer.observed_identities:
        observer._signal_identity(identity, signal.SIGKILL)
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pytest.fail("fixture root could not be reaped after verified cleanup")


@real_processes
def test_normal_cli_exit_is_independent_and_release_confirmed():
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(.2)"],
                               start_new_session=True)
    observer = processes.BenchmarkProcessObserver(process)
    try:
        started = time.monotonic()
        result = observer.observe_until_exit(process, started + 3)
        assert result["exit_code"] == 0
        assert result["release_confirmed"] is True
        assert result["stop_reason"] is None
        assert not result["cleanup_attempted"]
        assert result["observed_tree_empty"] is True
        assert started <= result["main_exit_monotonic"] <= result["completion_monotonic"]
        assert result["completion_monotonic"] - started < 2
        assert result["remaining_identities"] == []
    finally:
        _finish_fixture(process, observer)


PARENT_WITH_CHILD = r"""
import pathlib, subprocess, sys, time
child = subprocess.Popen(
    [sys.executable, '-c',
     'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)'],
    start_new_session=sys.argv[3] == 'new_session')
pid_path = pathlib.Path(sys.argv[1])
pending_pid = pid_path.with_suffix('.pending')
pending_pid.write_text(str(child.pid))
pending_pid.replace(pid_path)
while not pathlib.Path(sys.argv[2]).exists():
    time.sleep(.005)
"""


@real_processes
@pytest.mark.parametrize("new_session", [False, True], ids=["same_group", "new_session"])
def test_lingering_child_invalidates_trial_and_is_cleaned_by_identity(tmp_path, new_session):
    pid_file, exit_gate = tmp_path / "child.pid", tmp_path / "parent.exit"
    process = subprocess.Popen(
        [sys.executable, "-u", "-c", PARENT_WITH_CHILD, str(pid_file), str(exit_gate),
         "new_session" if new_session else "same_group"], start_new_session=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    observer = processes.BenchmarkProcessObserver(process, exit_grace=.12,
                                                   terminate_grace=.1, kill_grace=.4)
    result = {}
    worker = threading.Thread(target=lambda: result.update(
        observer.observe_until_exit(process, time.monotonic() + 5)), daemon=True)
    worker.start()
    try:
        _wait_for(pid_file.is_file)
        child_pid = int(pid_file.read_text())
        _wait_for(lambda: any(pid == child_pid for pid, _ in observer.observed_identities))
        child_identity = next(identity for identity in observer.observed_identities
                              if identity[0] == child_pid)
        # Main exits only after the child has really entered the ledger.
        exit_gate.touch()
        worker.join(timeout=6)
        assert not worker.is_alive()
        assert result["exit_code"] == 0
        assert result["stop_reason"] == "descendants_outlived_main"
        assert result["release_confirmed"] is False
        assert result["cleanup_attempted"] is True
        assert result["observed_tree_empty"] is True
        assert result["completion_monotonic"] >= result["main_exit_monotonic"] + .12
        assert result["remaining_identities"] == []
        assert process_identity(child_pid) != child_identity
        assert list(child_identity) in result["observed_identities"]
    finally:
        _finish_fixture(process, observer, worker)


@real_processes
def test_external_stop_cleans_observed_new_session_and_keeps_first_reason(tmp_path):
    pid_file, exit_gate = tmp_path / "child.pid", tmp_path / "parent.exit"
    process = subprocess.Popen(
        [sys.executable, "-u", "-c", PARENT_WITH_CHILD, str(pid_file), str(exit_gate), "new_session"],
        start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    observer = processes.BenchmarkProcessObserver(process, terminate_grace=.1, kill_grace=.4)
    result = {}
    worker = threading.Thread(target=lambda: result.update(
        observer.observe_until_exit(process, time.monotonic() + 5)), daemon=True)
    worker.start()
    try:
        _wait_for(pid_file.is_file)
        child_pid = int(pid_file.read_text())
        _wait_for(lambda: any(pid == child_pid for pid, _ in observer.observed_identities))
        observer.request_stop("output_budget_exhausted")
        observer.request_stop("later_sampling_error")
        worker.join(timeout=6)
        assert not worker.is_alive()
        assert result["stop_reason"] == "output_budget_exhausted"
        assert result["release_confirmed"] is False
        assert result["remaining_identities"] == []
        assert result["exit_code"] is not None
    finally:
        _finish_fixture(process, observer, worker)


class FakeProcess:
    pid = 117
    returncode = None

    def poll(self):
        return self.returncode


def _simulated_observer(monkeypatch, **timing):
    from astrolabe.runtime import telemetry
    process = FakeProcess()
    identities = {process.pid: (process.pid, 401)}
    monkeypatch.setattr(processes, "process_namespace_supported", lambda: True)
    monkeypatch.setattr(telemetry, "process_namespace_supported", lambda: True)
    monkeypatch.setattr(processes.BenchmarkProcessObserver, "_identity",
                        staticmethod(lambda pid: identities.get(pid)))

    def snapshot(roots, *, groups):
        return ProcessTree({pid: {"start": identity[1], "pgrp": pid, "ppid": 1}
                            for pid, identity in identities.items()
                            if roots.get(pid) == identity})

    monkeypatch.setattr(processes, "process_tree", snapshot)
    observer = processes.BenchmarkProcessObserver(process, **timing)
    return process, observer, identities


def test_unverified_group_and_reused_pid_are_never_signalled(monkeypatch):
    process, observer, identities = _simulated_observer(monkeypatch)
    original = identities[process.pid]
    identities[process.pid] = (process.pid, 999)  # unrelated reused PID
    process.returncode = 0
    monkeypatch.setattr(processes, "process_tree",
                        lambda *_, **__: ProcessTree({}, unverified_groups=[process.pid]))
    monkeypatch.setattr(observer, "_signal_identity",
                        lambda *_: pytest.fail("unverified process must not be signalled"))
    result = observer.observe_until_exit(process, time.monotonic() + 1)
    assert result["release_confirmed"] is False
    assert result["stop_reason"] == "process_observation_unknown"
    assert result["unverified_process_groups"] == [process.pid]
    assert result["observed_identities"] == [list(original)]
    assert result["remaining_identities"] == []


def test_failed_proc_enumeration_is_not_empty_tree_evidence(monkeypatch):
    process, observer, identities = _simulated_observer(monkeypatch)
    process.returncode = 0
    identities.clear()
    monkeypatch.setattr(processes, "process_tree", lambda *_, **__: {})
    result = observer.observe_until_exit(process, time.monotonic() + 1)
    assert result["release_confirmed"] is False
    assert result["observed_tree_empty"] is None
    assert result["stop_reason"] == "process_observation_unknown"
    assert any("enumeration" in error for error in result["observation_errors"])


def test_namespace_mismatch_is_unknown_without_unsafe_signalling(monkeypatch):
    monkeypatch.setattr(processes, "process_namespace_supported", lambda: False)
    process = FakeProcess()
    observer = processes.BenchmarkProcessObserver(process)
    monkeypatch.setattr(observer, "_signal_identity",
                        lambda *_: pytest.fail("unverified namespace cannot be signalled"))
    started = time.monotonic()
    result = observer.observe_until_exit(process, started + 5)
    assert result["stop_reason"] == "process_observation_unknown"
    assert result["release_confirmed"] is False
    assert result["exit_code"] is None
    assert result["completion_monotonic"] - started < .5


def test_expired_total_deadline_does_not_wait_through_cleanup_graces(monkeypatch):
    process, observer, identities = _simulated_observer(monkeypatch)
    signalled = []

    def send(identity, signum):
        assert identities[identity[0]] == identity
        signalled.append(signum)
        if signum == signal.SIGKILL:
            identities.clear()
            process.returncode = -signal.SIGKILL

    monkeypatch.setattr(observer, "_signal_identity", send)
    started = time.monotonic()
    result = observer.observe_until_exit(process, started - 1)
    assert signalled == [signal.SIGTERM, signal.SIGKILL]
    assert result["stop_reason"] == "walltime_budget_exhausted"
    assert result["release_confirmed"] is False
    assert result["remaining_identities"] == []
    assert result["completion_monotonic"] - started < .5


def _stat_text(identity, *, ppid=1, pgrp=None):
    pid, start = identity
    fields = ["0"] * 22
    fields[0], fields[1], fields[2] = "S", str(ppid), str(pgrp or pid)
    fields[19] = str(start)
    return f"{pid} (owned fixture) " + " ".join(fields)


def test_partial_global_snapshot_keeps_known_live_child_without_killing_or_releasing(monkeypatch):
    """Reproduce CPU trial 01: global scan missed a still-readable owned PID."""
    process, observer, identities = _simulated_observer(monkeypatch)
    child = (process.pid + 1, 402)
    identities[child[0]] = child
    observer._remember(child)
    observer._observe()
    identities.pop(process.pid)
    process.returncode = 0
    original_read = Path.read_text

    def read_stat(path, *args, **kwargs):
        if path == Path(f"/proc/{child[0]}/stat"):
            return _stat_text(child)
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_stat)
    samples = []

    def partial_snapshot(roots, *, groups):
        samples.append(dict(roots))
        assert roots[child[0]] == child
        if len(samples) == 1:
            return ProcessTree({})  # child is still alive and readable
        assert groups[child[0]] == child[1]
        identities.clear()  # child exits normally only after another observation
        return ProcessTree({})

    monkeypatch.setattr(processes, "process_tree", partial_snapshot)
    monkeypatch.setattr(observer, "_signal_identity",
                        lambda *_: pytest.fail("a partial snapshot must not kill a healthy child"))
    result = observer.observe_until_exit(process, time.monotonic() + 1)
    assert len(samples) >= 2
    assert result["release_confirmed"] is True
    assert not result["cleanup_attempted"]
    assert result["observation_errors"] == []
    assert list(child) in result["observed_identities"]


@real_processes
def test_real_known_root_can_be_recovered_from_one_partial_snapshot(monkeypatch):
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(.25)"],
                               start_new_session=True)
    observer = processes.BenchmarkProcessObserver(process)
    original_snapshot = processes.process_tree
    omissions = []

    def omit_once(roots, *, groups):
        snapshot = original_snapshot(roots, groups=groups)
        if process.pid in snapshot and not omissions:
            omissions.append(snapshot.pop(process.pid))
        return snapshot

    monkeypatch.setattr(processes, "process_tree", omit_once)
    try:
        result = observer.observe_until_exit(process, time.monotonic() + 3)
        assert len(omissions) == 1
        assert result["exit_code"] == 0
        assert result["release_confirmed"] is True
        assert result["observation_errors"] == []
        assert not result["cleanup_attempted"]
    finally:
        _finish_fixture(process, observer)


def test_individual_stat_permission_failure_remains_unknown(monkeypatch):
    process, observer, identities = _simulated_observer(monkeypatch)
    original_read = Path.read_text

    def denied(path, *args, **kwargs):
        if path == Path(f"/proc/{process.pid}/stat"):
            raise PermissionError("known process stat denied")
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", denied)
    monkeypatch.setattr(processes, "process_tree", lambda *_, **__: ProcessTree({}))
    signalled = []

    def stop_known(identity, signum):
        assert identities[identity[0]] == identity
        signalled.append(identity)
        identities.clear()
        process.returncode = -signum

    monkeypatch.setattr(observer, "_signal_identity", stop_known)
    result = observer.observe_until_exit(process, time.monotonic() + 1)
    assert result["release_confirmed"] is False
    assert result["stop_reason"] == "process_observation_unknown"
    assert any("stat denied" in error for error in result["observation_errors"])
    assert signalled == [(process.pid, 401)]


def test_cleanup_repolls_exit_status_after_root_disappears(monkeypatch):
    process, observer, identities = _simulated_observer(
        monkeypatch, terminate_grace=0, kill_grace=0)
    delayed_polls = []

    def poll():
        if identities:
            return None
        if delayed_polls:
            delayed_polls.pop()
            return None
        return -signal.SIGKILL

    def send(identity, signum):
        assert identities[identity[0]] == identity
        if signum == signal.SIGKILL:
            identities.clear()
            # Cleanup's first poll races with exit; its final poll has not
            # yet observed the zombie. The post-snapshot poll must reconcile.
            delayed_polls.extend([None, None])

    monkeypatch.setattr(process, "poll", poll)
    monkeypatch.setattr(observer, "_signal_identity", send)
    observer.request_stop("output_budget_exhausted")
    result = observer.observe_until_exit(process, time.monotonic() + 1)
    assert result["exit_code"] == -signal.SIGKILL
    assert result["main_exit_monotonic"] is not None
    assert result["main_exit_monotonic"] <= result["completion_monotonic"]
    assert result["release_confirmed"] is False
    assert result["stop_reason"] == "output_budget_exhausted"
    assert result["observation_errors"] == []
    assert result["remaining_identities"] == []
