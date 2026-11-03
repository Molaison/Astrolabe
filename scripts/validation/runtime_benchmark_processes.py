"""Observe benchmark CLI completion independently of resource sampling.

This is an observer for processes seen in the original tree or in a group with
a live, verified identity anchor. It is not a containment boundary: an arbitrary
double fork can disappear between samples. A dedicated job cgroup is required
to prove complete containment of arbitrary external programs.
"""
from __future__ import annotations

import errno
import math
import os
import signal
import threading
import time

from astrolabe.runtime.telemetry import (
    ProcessTree, process_identity, process_namespace_supported, process_record, process_tree,
)


SCOPE = (
    "Observed PID/start-time identities, their descendants, and identity-anchored "
    "process groups only. Unobserved double forks or namespace escapes are not "
    "fully proved absent; complete arbitrary-program containment requires a job cgroup."
)


class BenchmarkProcessObserver:
    """Construct immediately after Popen, then run observation in a wait thread.

    ``release_confirmed`` is false whenever forced cleanup or an observation
    failure occurred, even if cleanup eventually stopped all known processes.
    The benchmark must invalidate that trial and stop launching later trials.
    Completion time includes normal shutdown grace and forced cleanup costs.
    Main exit time is the first independent observation, at the stated polling
    resolution, rather than a timestamp delayed by GPU/RSS telemetry.
    """

    def __init__(self, process, *, interval=0.03, exit_grace=5.0,
                 terminate_grace=1.0, kill_grace=1.0):
        if not 0.02 <= interval <= 0.05:
            raise ValueError("process observation interval must be 20–50 ms")
        if not 0 <= exit_grace <= 5:
            raise ValueError("normal shutdown grace must be between zero and five seconds")
        if any(not math.isfinite(value) or not 0 <= value <= 5
               for value in (terminate_grace, kill_grace)):
            raise ValueError("cleanup grace must be finite and at most five seconds")
        self.process = process
        self.interval = float(interval)
        self.exit_grace = float(exit_grace)
        self.terminate_grace = float(terminate_grace)
        self.kill_grace = float(kill_grace)
        self._known = {}
        self._root_identity = None
        self._groups = {}
        self._observed = set()
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._stop_reason = None
        self._errors = []
        self._unverified_groups = set()
        self._main_exit = None
        self._exit_code = None
        self._cleanup_attempted = False
        self._result = None
        self._namespace_ok = process_namespace_supported()
        if not self._namespace_ok:
            self._unknown("process and /proc namespaces do not match")
            return
        try:
            identity = self._identity(process.pid)
            if identity is None:
                self._unknown("fresh CLI root identity was unavailable before observation")
                return
            self._root_identity = identity
            self._remember(identity)
            self._observe()
        except (OSError, RuntimeError, ValueError) as exc:
            self._unknown("root observation failed: " + str(exc))

    @property
    def observed_identities(self):
        """A read-only snapshot, useful for external ownership diagnostics."""
        with self._state_lock:
            return tuple(sorted(self._observed))

    def request_stop(self, reason):
        """Ask the observer thread to stop safely; the first reason is sticky."""
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("a nonempty stop reason is required")
        with self._state_lock:
            if self._stop_reason is None:
                self._stop_reason = reason
        self._stop_event.set()

    def _requested_stop(self):
        with self._state_lock:
            return self._stop_reason

    def _remember(self, identity):
        self._known[identity[0]] = identity
        with self._state_lock:
            self._observed.add(identity)

    def _unknown(self, message):
        if message not in self._errors and len(self._errors) < 16:
            self._errors.append(message)

    @staticmethod
    def _identity(pid):
        # The shared reader distinguishes disappearance from observation
        # failure. Permission/I/O/format errors raise instead of returning None.
        return process_identity(pid)

    def _poll_main(self):
        if self._main_exit is not None:
            return
        try:
            code = self.process.poll()
        except (OSError, RuntimeError) as exc:
            self._unknown("CLI exit observation failed: " + str(exc))
            return
        if code is not None:
            self._exit_code = code
            self._main_exit = time.monotonic()

    def _known_record(self, identity):
        """Repair a non-atomic global snapshot using one already-owned PID.

        A global /proc walk can omit a process that is readable immediately
        afterwards. Only this exact previously observed identity may be added;
        permission/format failures remain unknown, and PID reuse is excluded.
        """
        pid, start = identity
        record = process_record(pid)
        if record is None or record["start"] != start:
            return None
        # A second identity check fences exit/reuse during the per-PID read.
        return record if self._identity(pid) == identity else None

    def _settle_main_exit(self, deadline):
        """Reconcile a just-observed zombie with Popen.poll, without wait()."""
        self._poll_main()
        if self._main_exit is not None or self._root_identity is None:
            return
        try:
            if self._identity(self.process.pid) == self._root_identity:
                return
        except (OSError, RuntimeError, ValueError) as exc:
            self._unknown("final CLI identity verification failed: " + str(exc))
            return
        # The original root no longer runs, but it can become a zombie between
        # the cleanup poll and the last /proc scan. Concurrent Popen users may
        # also briefly hold its waitpid lock. Recheck only within this small,
        # total-deadline-bounded interval; never invent an exit code.
        settle_deadline = min(deadline, time.monotonic() + self.interval)
        while self._main_exit is None:
            self._poll_main()
            if self._main_exit is not None:
                return
            if time.monotonic() >= settle_deadline:
                self._unknown("CLI root stopped but exit status was not available within the deadline")
                return
            time.sleep(min(.005, max(0, settle_deadline - time.monotonic())))

    def _observe(self):
        if not self._namespace_ok:
            return None
        try:
            tree = process_tree(dict(self._known), groups=dict(self._groups))
            # Require the verified snapshot contract; an ambiguous empty dict
            # from a failed/older observer is never proof of release.
            if not isinstance(tree, ProcessTree):
                raise RuntimeError("process-tree enumeration did not produce a verified snapshot")
            self._unverified_groups.update(tree.unverified_groups)
            if tree.unverified_groups:
                self._unknown("process group lost every observed identity anchor")
            live = {}
            for pid, info in tree.items():
                identity = (int(pid), int(info["start"]))
                if self._identity(pid) == identity:
                    live[pid] = info
                    self._remember(identity)
            # /proc enumeration is not atomic. Repair an omitted known live
            # identity from its own stat instead of either claiming release or
            # killing a healthy trial solely because of one partial snapshot.
            for pid, identity in tuple(self._known.items()):
                if self._identity(pid) == identity and pid not in live:
                    record = self._known_record(identity)
                    if record is not None:
                        live[pid] = record
            present_groups = ({info["pgrp"] for info in tree.values()}
                              | {info["pgrp"] for info in live.values()}
                              | set(tree.unverified_groups))
            self._groups = {group: start for group, start in self._groups.items()
                            if group in present_groups}
            for pid, info in live.items():
                # Never enroll the benchmark parent's shared process group.
                # New-session workers are enrolled only while their actual
                # process-group leader identity is observed alive.
                if info["pgrp"] == pid:
                    self._groups[pid] = info["start"]
            return live
        except (OSError, RuntimeError, ValueError, KeyError, IndexError) as exc:
            self._unknown("process observation failed: " + str(exc))
            return None

    def _live_known(self):
        active = []
        for identity in tuple(self._known.values()):
            try:
                if self._identity(identity[0]) == identity:
                    active.append(identity)
            except (OSError, RuntimeError, ValueError) as exc:
                self._unknown("release identity verification failed: " + str(exc))
        return active

    def _signal_identity(self, identity, signum):
        """Signal one verified identity; never signal a numerical process group."""
        pid = identity[0]
        try:
            if self._identity(pid) != identity:
                return
            if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
                try:
                    fd = os.pidfd_open(pid)
                except OSError as exc:
                    if exc.errno != errno.ENOSYS:
                        raise
                else:
                    try:
                        # Opening the fd and rechecking prevents PID reuse
                        # between verification and signal delivery.
                        if self._identity(pid) == identity:
                            signal.pidfd_send_signal(fd, signum)
                    finally:
                        os.close(fd)
                    return
            if self._identity(pid) == identity:
                os.kill(pid, signum)
        except ProcessLookupError:
            pass
        except (OSError, RuntimeError, ValueError) as exc:
            self._unknown("verified process cleanup failed: " + str(exc))

    def _cleanup(self, deadline):
        self._cleanup_attempted = True
        for signum, grace in ((signal.SIGTERM, self.terminate_grace),
                              (signal.SIGKILL, self.kill_grace)):
            for identity in reversed(self._live_known()):
                self._signal_identity(identity, signum)
            phase_deadline = min(deadline, time.monotonic() + grace)
            while True:
                self._poll_main()
                tree = self._observe()
                if self._main_exit is not None and tree is not None and not tree:
                    return tree
                if not self._live_known() or time.monotonic() >= phase_deadline:
                    break
                time.sleep(min(self.interval, max(0, phase_deadline - time.monotonic())))
        self._poll_main()
        return self._observe()

    def observe_until_exit(self, process, deadline):
        """Observe/reap through the absolute monotonic deadline without unbounded wait."""
        if process is not self.process:
            raise ValueError("observer must be used with its original Popen process")
        if not math.isfinite(deadline):
            raise ValueError("a finite monotonic deadline is required")
        if self._result is not None:
            return dict(self._result)
        reason = None
        while True:
            reason = self._requested_stop()
            if reason is not None:
                break
            self._poll_main()
            tree = self._observe()
            self._poll_main()
            now = time.monotonic()
            if self._errors:
                reason = "process_observation_unknown"
                break
            if now >= deadline:
                reason = "walltime_budget_exhausted"
                break
            if self._main_exit is not None and tree is not None and not tree:
                break
            if self._main_exit is not None and now >= self._main_exit + self.exit_grace:
                reason = "descendants_outlived_main"
                break
            next_check = min(deadline, now + self.interval)
            if self._main_exit is not None:
                next_check = min(next_check, self._main_exit + self.exit_grace)
            self._stop_event.wait(max(0, next_check - time.monotonic()))
        reason = reason or self._requested_stop()
        if reason is not None:
            tree = self._cleanup(deadline)
        remaining = self._live_known()
        if reason is None and (self._errors or remaining):
            reason = "process_observation_unknown"
            tree = self._cleanup(deadline)
            remaining = self._live_known()
        self._settle_main_exit(deadline)
        if reason is None and self._errors:
            reason = "process_observation_unknown"
        completion = time.monotonic()
        tree_empty = (None if tree is None else not bool(tree))
        self._result = {
            "exit_code": self._exit_code,
            "main_exit_monotonic": self._main_exit,
            "completion_monotonic": completion,
            "release_confirmed": reason is None and not self._errors and tree_empty is True,
            "stop_reason": reason,
            "cleanup_attempted": self._cleanup_attempted,
            "observed_tree_empty": tree_empty,
            "observed_processes": len(self.observed_identities),
            "observed_identities": [list(identity) for identity in self.observed_identities],
            "remaining_identities": [list(identity) for identity in remaining],
            "unverified_process_groups": sorted(self._unverified_groups),
            "observation_errors": list(self._errors),
            "observation_interval_seconds": self.interval,
            "scope": SCOPE,
        }
        return dict(self._result)
