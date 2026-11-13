"""Root-session resource sampling, deliberately independent of model runtimes.

GPU sampling reuses runtime.telemetry.NvidiaSMISampler and its physical UUID /
process identity contract. Its device-wide metrics are not run attribution.
CPU times are cumulative process CPU seconds (all threads); summed RSS can
double-count shared pages. These are snapshots, not utilization percentages.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import threading
import time


def _linux_cpu(root_pid):
    ticks = os.sysconf("SC_CLK_TCK")
    page_size = os.sysconf("SC_PAGE_SIZE")
    rows = {}
    skipped = 0
    for path in Path("/proc").iterdir():
        if not path.name.isdecimal():
            continue
        try:
            text = (path / "stat").read_text()
            start, end = text.index("("), text.rindex(")")
            pid = int(text[:start].strip())
            fields = text[end + 1:].split()
            if fields[0] in ("Z", "X", "x"):
                continue
            rows[pid] = dict(
                pid=pid, ppid=int(fields[1]), name=text[start + 1:end],
                start_ticks=int(fields[19]), cpu_user_s=int(fields[11]) / ticks,
                cpu_system_s=int(fields[12]) / ticks,
                cpu_time_s=(int(fields[11]) + int(fields[12])) / ticks,
                rss_bytes=max(0, int(fields[21])) * page_size,
            )
        except (OSError, ValueError, IndexError):
            skipped += 1
    if root_pid not in rows:
        raise RuntimeError("root process unavailable in /proc snapshot")
    children = {}
    for row in rows.values():
        children.setdefault(row["ppid"], []).append(row["pid"])
    pending = [root_pid]
    selected = set()
    while pending:
        pid = pending.pop()
        if pid not in selected:
            selected.add(pid)
            pending.extend(children.get(pid, ()))
    return dict(source="linux_proc", root_pid=root_pid, status="ok",
                processes=[rows[pid] for pid in sorted(selected)],
                unreadable_or_exited_system_processes=skipped)


def _psutil_cpu(root_pid):
    # Optional fallback only on platforms without the Linux /proc interface.
    import psutil
    root = psutil.Process(root_pid)
    rows, errors = [], []
    for process in [root, *root.children(recursive=True)]:
        try:
            with process.oneshot():
                cpu = process.cpu_times()
                rows.append(dict(
                    pid=process.pid, ppid=process.ppid(), name=process.name(),
                    create_time_s=process.create_time(), cpu_user_s=cpu.user,
                    cpu_system_s=cpu.system, cpu_time_s=cpu.user + cpu.system,
                    rss_bytes=process.memory_info().rss,
                ))
        except (psutil.Error, OSError) as exc:
            if len(errors) < 16:
                errors.append(f"pid {process.pid}: {type(exc).__name__}")
    return dict(source="psutil", root_pid=root_pid,
                status="partial" if errors else "ok", processes=rows, errors=errors)


def sample_cpu(root_pid):
    """Return process-tree snapshots without modifying affinity or priority."""
    result = _linux_cpu(root_pid) if sys.platform.startswith("linux") else _psutil_cpu(root_pid)
    result.update(cpu_time_scope="process_all_threads_cumulative_seconds",
                  rss_scope="per_process_resident_bytes_shared_pages_may_repeat")
    return result


class ResourceSampler:
    """Exactly one daemon thread owned and joined by an explicit root session."""

    def __init__(self, recorder):
        self.recorder = recorder
        self.pid = os.getpid()
        self.stop_event = threading.Event()
        self.thread = None
        self.gpu_sampler = None

    def start(self):
        self.thread = threading.Thread(target=self._run, name="astrolabe-profile-resources",
                                       daemon=True)
        self.thread.start()

    def stop(self):
        if os.getpid() != self.pid:
            return
        self.stop_event.set()
        if self.thread is not None and self.thread.ident is not None:
            self.thread.join()

    def _sample(self, name, collect):
        wall, monotonic = time.time_ns(), time.monotonic_ns()
        try:
            attrs = collect()
        except Exception as exc:
            attrs = dict(status="error", error=f"{type(exc).__name__}: {str(exc)[:1000]}")
            self.recorder.failure(name, exc)
        self.recorder.record("resource", name, attrs, ts_ns=wall,
                             duration_ns=time.monotonic_ns() - monotonic)

    def _gpu(self):
        from astrolabe.runtime.telemetry import NvidiaSMISampler, validate_sample
        if self.gpu_sampler is None:
            self.gpu_sampler = NvidiaSMISampler(
                timeout=min(2.0, self.recorder.config.sample_interval_s / 2))
        devices = self.gpu_sampler()
        validate_sample(devices)
        return dict(status="ok", source="runtime.telemetry.NvidiaSMISampler",
                    attribution="device_wide_not_attributed_to_run", devices=devices)

    def _run(self):
        try:
            while not self.stop_event.is_set():
                self._sample("resource.cpu", lambda: sample_cpu(self.pid))
                if self.recorder.config.gpu and not self.stop_event.is_set():
                    self._sample("resource.gpu", self._gpu)
                if self.stop_event.wait(min(self.recorder.config.sample_interval_s,
                                            threading.TIMEOUT_MAX)):
                    break
        except Exception as exc:
            self.recorder.failure("resource sampler stopped unexpectedly", exc)
