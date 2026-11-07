"""Standard-library bounded NVIDIA sampling and optional fail-closed JSONL events.

An injected sampler is a callable returning a list (or {'devices': list}) of
{index, uuid, total_bytes, free_bytes, utilization, processes: [{pid, used_bytes}]}.
All fields are required: absent/failed process accounting is unknown, never idle.
Sampler calls must complete within their own bounded timeout. No GPU is sampled
when the broker has an empty devices list.

NVIDIA may report a Linux task ID instead of a process leader. The built-in
sampler retains that raw ID and a live /proc membership/epoch proof, normalizing
same-UID tasks to their TGID. Missing or conflicting proof, including multiple
rows per TGID on one GPU, is unknown accounting and raises; rows are never dropped.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time


def nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _validate_identity(proof, pid, reported_id):
    """Check retained proof shape and relationships, including namespace IDs."""
    if proof["reported_id"] != reported_id or proof["reread_verified"] is not True:
        raise ValueError("invalid task identity proof")
    task, leader, member = (proof[key] for key in ("task", "leader", "member"))
    for identity in (task, leader, member):
        for key in ("pid", "tgid"):
            if nonnegative(identity[key], key) == 0:
                raise ValueError("invalid task ID")
        epoch = identity["start_ticks"]
        if not isinstance(epoch, str) or not epoch.isascii() or not epoch.isdigit() or str(int(epoch)) != epoch:
            raise ValueError("invalid task epoch")
        if len(identity["uid"]) != 4:
            raise ValueError("invalid task UID")
        for uid in identity["uid"]:
            nonnegative(uid, "uid")
        for key, first in (("nspid", identity["pid"]), ("nstgid", identity["tgid"])):
            values = identity[key]
            if not values or values[0] != first:
                raise ValueError("ambiguous PID namespace")
            for value in values:
                if nonnegative(value, key) == 0:
                    raise ValueError("invalid namespace ID")
    if (task["pid"] != reported_id or task["tgid"] != pid
            or leader["pid"] != pid or leader["tgid"] != pid
            or task != member or task["uid"] != leader["uid"]
            or task["nstgid"] != leader["nstgid"]
            or leader["nspid"] != leader["nstgid"]
            or len(task["nspid"]) != len(task["nstgid"])
            or int(task["start_ticks"]) < int(leader["start_ticks"])):
        raise ValueError("inconsistent task/leader identity")
    if reported_id != pid and any(uid != os.getuid() for uid in task["uid"]):
        raise ValueError("task normalization requires same UID")


def _read_proc_identity(path, expected_id):
    def stat_epoch():
        text = (path / "stat").read_text()
        fields = text.rsplit(")", 1)[1].split()
        if int(text.split("(", 1)[0]) != expected_id or fields[0] in ("Z", "X", "x"):
            raise ValueError("task unavailable")
        return fields[19]

    before = stat_epoch()
    status = {}
    for line in (path / "status").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in ("Pid", "Tgid", "Uid", "NSpid", "NStgid"):
            if key in status:
                raise ValueError("ambiguous task status")
            status[key] = [int(item) for item in value.split()]
    if len(status["Pid"]) != 1 or len(status["Tgid"]) != 1:
        raise ValueError("ambiguous task ID")
    if before != stat_epoch() or status["Pid"][0] != expected_id:
        raise ValueError("task epoch changed")
    return dict(pid=status["Pid"][0], tgid=status["Tgid"][0], uid=status["Uid"],
                nspid=status["NSpid"], nstgid=status["NStgid"], start_ticks=before)


def _prove_task_identity(reported_id, proc_root):
    """Only live kernel membership can turn a reported Linux TID into a TGID.

    Each read brackets status with stat epochs, then all three identities are
    reread in reverse order. Unreadable, exited, reused or ambiguous is unknown;
    callers must close admission rather than drop the accounting row.
    """
    task_path = proc_root / str(reported_id)
    task = _read_proc_identity(task_path, reported_id)
    pid = task["tgid"]
    leader_path = proc_root / str(pid)
    member_path = leader_path / "task" / str(reported_id)
    leader = _read_proc_identity(leader_path, pid)
    member = _read_proc_identity(member_path, reported_id)
    if (member != _read_proc_identity(member_path, reported_id)
            or leader != _read_proc_identity(leader_path, pid)
            or task != _read_proc_identity(task_path, reported_id)):
        raise ValueError("task identity proof changed")
    proof = dict(reported_id=reported_id, task=task, leader=leader, member=member,
                 reread_verified=True)
    _validate_identity(proof, pid, reported_id)
    return pid, proof


def validate_sample(raw):
    rows = raw.get("devices") if isinstance(raw, dict) else raw
    if not isinstance(rows, list):
        raise ValueError("sampler must return a device list")
    result = {}
    indices = set()
    for row in rows:
        uuid = row["uuid"]
        index = nonnegative(row["index"], "index")
        if not isinstance(uuid, str) or not uuid.startswith("GPU-"):
            raise ValueError("physical GPU UUID required (MIG is unsupported)")
        if uuid in result or index in indices:
            raise ValueError("duplicate device in sample")
        total = nonnegative(row["total_bytes"], "total_bytes")
        free = nonnegative(row["free_bytes"], "free_bytes")
        util = row["utilization"]
        if total == 0 or free > total or isinstance(util, bool) or not isinstance(util, (int, float)) or not math.isfinite(util) or not 0 <= util <= 100:
            raise ValueError("invalid device telemetry")
        processes = {}
        identities = {}
        if not isinstance(row["processes"], list):
            raise ValueError("process accounting unavailable")
        for proc in row["processes"]:
            pid = nonnegative(proc["pid"], "pid")
            if pid == 0 or pid in processes:
                raise ValueError("invalid/duplicate process")
            processes[pid] = nonnegative(proc["used_bytes"], "used_bytes")
            if "identity" in proc or "reported_id" in proc:
                reported_id = nonnegative(proc["reported_id"], "reported_id")
                if reported_id == 0:
                    raise ValueError("invalid reported task ID")
                _validate_identity(proc["identity"], pid, reported_id)
                identities[pid] = proc["identity"]
        result[uuid] = dict(index=index, uuid=uuid, total_bytes=total,
                            free_bytes=free, utilization=float(util), processes=processes)
        if identities:
            result[uuid]["process_identities"] = identities
        indices.add(index)
    return result


class NvidiaSMISampler:
    """Two subprocess queries, each bounded by timeout (seconds)."""

    def __init__(self, timeout=2.0, *, proc_root="/proc"):
        self.timeout = timeout
        self._proc_root = Path(proc_root)

    def _query(self, query):
        result = subprocess.run(
            ["nvidia-smi", query, "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=self.timeout, check=True,
        )
        return list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))

    def __call__(self):
        devices = self._query("--query-gpu=index,uuid,memory.total,memory.free,utilization.gpu")
        processes = self._query("--query-compute-apps=gpu_uuid,pid,used_gpu_memory")
        rows = []
        for index, uuid, total, free, util in devices:
            rows.append(dict(index=int(index), uuid=uuid.strip(),
                             total_bytes=int(total) * 1048576,
                             free_bytes=int(free) * 1048576,
                             utilization=float(util), processes=[]))
        by_uuid = {row["uuid"]: row for row in rows}
        # N/A, unsupported accounting, and malformed output deliberately raise.
        seen = set()
        for uuid, pid, used in processes:
            reported_id = int(pid)
            if reported_id <= 0:
                raise ValueError("invalid reported task ID")
            normalized, proof = _prove_task_identity(reported_id, self._proc_root)
            key = (uuid.strip(), normalized)
            if key in seen:
                # Neither max nor sum is qualified for multiple task records.
                raise ValueError("ambiguous GPU memory for duplicate TGID")
            seen.add(key)
            by_uuid[uuid.strip()]["processes"].append(dict(
                pid=normalized, reported_id=reported_id, identity=proof,
                used_bytes=int(used) * 1048576))
        return rows


class EventLog:
    """A failed append latches failure; cleanup must still proceed."""

    def __init__(self, path=None):
        self.path = path
        self.error = None
        self._lock = threading.Lock()

    def emit(self, event, **fields):
        with self._lock:
            if self.error:
                raise RuntimeError(self.error)
            if self.path is None:
                return
            try:
                with open(self.path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(dict(time=time.time(), event=event, **fields),
                                            allow_nan=False) + "\n")
            except Exception as exc:
                self.error = f"telemetry write failed: {type(exc).__name__}"
                raise RuntimeError(self.error) from exc
