"""One physical-UUID-locked Unix broker, with process-fenced resource lifetimes.

ResourceBroker(config: dict, sampler=None).start() -> self exports socket_path as
ASTROLABE_RESOURCE_SOCKET for child daemons. close() rejects pending/new work and
drains GPU compute/residency. A shutdown timeout raises BrokerBusy, retaining UUID
locks, the control socket and owner monitoring; retry close() after work ends.
Each GPU admission is also atomically journaled beside its UUID lock (mode 0600,
fsync + replace + directory fsync). A replacement broker refuses live/unknown
PID/start-epoch orphan records even if the previous broker process has exited.
Only completion/unregister or positive worker exit evidence removes a record.
ResourceClient(socket_path).lease(resource='gpu', backend='', identity='',
shape='', phase='compute', resident_bytes=0, scratch_bytes=0, ram_bytes=0,
output_bytes=0, concurrency_certified=False, timeout=None) is a context manager
yielding Lease (token, device, finish(error=None, oom=False), cancel()). Errors,
disconnect and cancellation fence running GPU resources until the original
PID/start epoch exits or explicitly reports true completion. CPU/queued requests
can cancel immediately. timeout is only the queue deadline. Successful context
exit/finish() asserts actual device completion (the caller must synchronize GPU
work); exceptional GPU context exit fences. finish() can reconnect after cancel()
using its token and the same PID/start epoch, once the device really has stopped.

register_model(model_id, device=None, resident_bytes=0, metadata=None,
resident_ram_bytes=0, prepared_model_id=None) reserves
VRAM until unregister_model(model_id) or process exit. Registration inside a load
lease atomically transfers its resident reservation (not its scratch reservation).
Registration parses identity:worker_pid:worker_epoch when supplied, and fences
that actual worker's /proc start epoch; plain IDs use the peer PID. Leases use
SO_PEERCRED PID plus /proc start epoch. A parent may unregister after worker exit;
forked clients automatically use their new PID. Model IDs are global to a broker.
unregister during the owner's device computation is rejected. Register inside
the load lease after allocation, unregister after releasing the allocation. Zero GPU byte
estimates conservatively reserve the device exclusively, including registered
residency belonging to that process; external compute processes block unknown
sizes. Driver/graphics memory consumes measured free capacity and is not treated
as missing process telemetry. Known sizes may use free memory at nonzero SM.
Unknown model residency is reconciled to measured worker VRAM when available;
until then it is recorded as unknown. cuda:N registration uses the CUDA visibility
order sent by the client, while configured device indices are always physical.
lease(..., device=None, visible_devices=None) optionally pins a device; by default
the client's first visible device is used, never an opportunistic second GPU.
Clients send CUDA_VISIBLE_DEVICES on every resource RPC. CPU leases ignore
GPU memory declarations; existing model reservations remain held.
An optional prepared_model_id binds a zero-RAM load lease to this live owner's
positive declared CPU RAM registration with residency='prepared_cpu_weights'. Only one active
or pending placement can claim it. Registering with the same ID inside that
target load lease atomically replaces the prepared RAM with final residency;
failed validation leaves the prepared registration charged and unchanged.
resident_ram_bytes is independent of GPU residency (device='cpu' is supported)
and atomically transfers from the owner's load RAM lease where available. It
consumes no CPU slot and remains charged until unregister/actual worker exit.
observe_resident_ram=False preserves explicit accounting. When True, a zero RAM
declaration reads the actual worker's /proc/PID/status VmRSS (kB * 1024), charges
the per-PID/start-epoch RSS high water mark, and refreshes it on sampler ticks.
Multiple models share one RSS ledger; explicit declarations for that owner are
charged as max(sum(declarations), observed RSS), never one RSS per model. RSS
accounting survives unregister until positive process exit, including allocator
caches. Missing observations fail closed. Unknown-RAM loads serialize globally
with other initializers; pre-load and transient unknown peaks cannot be predicted.

Configuration: devices=[{'index': physical_index, 'uuid': 'GPU-...',
'vram_bytes': safe_budget}] (index or UUID may be omitted, never both). All must
fit CUDA_VISIBLE_DEVICES, SLURM_JOB_GPUS and SLURM_STEP_GPUS when present; ambiguous
or MIG visibility fails closed. CPU-only devices=[] is explicit. cpu_slots,
ram_bytes and output_bytes default to 1, 0, 0; zero means no capacity, not unlimited.
max_pending=128, max_pending_bytes=64MiB bound queued host RAM+output bytes only
(GPU resident/scratch bytes remain a separate device reservation). sample_interval=1s,
sample_timeout=2s per NVIDIA query, sample_max_age=5s, rpc_timeout=5s,
max_frame_bytes=64KiB, shutdown_timeout=10s are tunable. socket_dir is optional,
must be owned by this UID with mode 0700. UUID lock files are host-wide in /tmp.
telemetry_path optionally appends JSONL; failure closes admission, never cleanup.
telemetry_recovery=False preserves fail-fast admission. Opting in lets ungranted
CPU/GPU leases wait for a new valid sample, charged to the same pending count/bytes.
A single sample_max_age readiness window starts on the first unknown sample;
subsequent faults never reset it or extend the original lease queue deadline.
No fit/device/reservation action occurs while telemetry is unknown. Startup,
model registration and already-granted work keep their old contracts.

Controller key is (backend, identity, shape, phase/operation). Unknown concurrency
starts at one. Uncertified GPU work and all GPU load phases exclusively hold the
physical device. All overlapping work must be certified and share a device-wide
max_concurrency cap. Only explicitly certified ready work may probe up to configured
max_concurrency (default 1), after min_observations=4 completions AND a 1s
observation_window of busy time. A throughput_drop_fraction=0.15 relative to the
preceding window halves concurrency; OOM does so immediately. These are tunable
conservative policy defaults, not measured performance claims. Sampling and
waiting happen on broker threads, with one thread per bounded client connection.
Scientific budgets, model semantics, batch sizes, seeds and retries are untouched.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import select
import socket
import stat
import struct
import threading
import time
import uuid as uuid_module

from .telemetry import EventLog, NvidiaSMISampler, nonnegative, validate_sample


class AdmissionError(RuntimeError):
    pass


class AdmissionTimeout(AdmissionError, TimeoutError):
    pass


class BrokerClosed(AdmissionError):
    pass


class BrokerBusy(AdmissionError):
    """Shutdown did not prove that GPU work/residency ended; locks remain held."""


def _epoch(pid):
    try:
        # comm may contain spaces and closing parentheses.
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (OSError, IndexError):
        return None


def _owner_exited(owner):
    """Require positive exit/reuse evidence; unreadable process state is unknown."""
    try:
        fields = Path(f"/proc/{owner[0]}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[0] in ("Z", "X", "x") or fields[19] != owner[1]
    except FileNotFoundError:
        return Path("/proc/self/stat").exists()
    except (OSError, IndexError):
        return False


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _send(sock, obj, limit):
    payload = json.dumps(obj, allow_nan=False, separators=(",", ":")).encode()
    if len(payload) > limit:
        raise AdmissionError("frame too large")
    sock.sendall(struct.pack("!I", len(payload)) + payload)


def _receive(sock, limit):
    def read(size):
        result = bytearray()
        while len(result) < size:
            part = sock.recv(size - len(result))
            if not part:
                raise EOFError("broker connection closed")
            result.extend(part)
        return result
    size = struct.unpack("!I", read(4))[0]
    if size > limit or size == 0:
        raise AdmissionError("invalid frame length")
    obj = json.loads(read(size))
    if not isinstance(obj, dict):
        raise AdmissionError("object frame required")
    return obj


class _Controller:
    def __init__(self):
        self.limit = 1
        self.active = 0
        self.completed = 0
        self.window_completed = 0
        self.busy = 0.0
        self.last = time.monotonic()
        self.throughput = None
        self.peak_vram_bytes = 0
        self.utilization_sum = 0.0
        self.utilization_samples = 0
        self.peak_memory_fraction = 0.0
        self.last_decision = "conservative_start"

    def tick(self):
        now = time.monotonic()
        if self.active:
            self.busy += now - self.last
        self.last = now


class ResourceBroker:
    def __init__(self, config: dict, sampler=None):
        self.config = dict(config)
        if "devices" not in config or not isinstance(config["devices"], list):
            raise ValueError("explicit devices list required")
        self.cpu_slots = nonnegative(config.get("cpu_slots", 1), "cpu_slots")
        self.ram_bytes = nonnegative(config.get("ram_bytes", 0), "ram_bytes")
        self.observe_resident_ram = config.get("observe_resident_ram", False)
        if not isinstance(self.observe_resident_ram, bool):
            raise ValueError("observe_resident_ram must be boolean")
        self.output_bytes = nonnegative(config.get("output_bytes", 0), "output_bytes")
        self.max_pending = nonnegative(config.get("max_pending", 128), "max_pending")
        self.max_pending_bytes = nonnegative(config.get("max_pending_bytes", 64 * 1024**2), "max_pending_bytes")
        self.cap = nonnegative(config.get("max_concurrency", 1), "max_concurrency")
        self.min_observations = nonnegative(config.get("min_observations", 4), "min_observations")
        if self.cap < 1 or self.min_observations < 1:
            raise ValueError("concurrency and observation counts must be positive")
        self.interval = _positive(config.get("sample_interval", 1.0), "sample_interval")
        self.max_age = _positive(config.get("sample_max_age", 5.0), "sample_max_age")
        self.telemetry_recovery = config.get("telemetry_recovery", False)
        if not isinstance(self.telemetry_recovery, bool):
            raise ValueError("telemetry_recovery must be boolean")
        self.rpc_timeout = _positive(config.get("rpc_timeout", 5.0), "rpc_timeout")
        self.shutdown_timeout = _positive(config.get("shutdown_timeout", 10.0), "shutdown_timeout")
        self.window = _positive(config.get("observation_window", 1.0), "observation_window")
        self.drop = config.get("throughput_drop_fraction", 0.15)
        if isinstance(self.drop, bool) or not isinstance(self.drop, (int, float)) or not 0 < self.drop < 1:
            raise ValueError("throughput_drop_fraction must be between zero and one")
        self.target_utilization = config.get("target_utilization", 90.0)
        self.memory_pressure_fraction = config.get("memory_pressure_fraction", 0.9)
        if (isinstance(self.target_utilization, bool)
                or not isinstance(self.target_utilization, (int, float))
                or not 0 < self.target_utilization <= 100):
            raise ValueError("target_utilization must be in (0, 100]")
        if (isinstance(self.memory_pressure_fraction, bool)
                or not isinstance(self.memory_pressure_fraction, (int, float))
                or not 0 < self.memory_pressure_fraction < 1):
            raise ValueError("memory_pressure_fraction must be in (0, 1)")
        self.frame = nonnegative(config.get("max_frame_bytes", 65536), "max_frame_bytes")
        if not 1024 <= self.frame <= 1024**2:
            raise ValueError("max_frame_bytes must be between 1024 and 1048576")
        self.max_connections = nonnegative(config.get("max_connections", self.max_pending + self.cpu_slots + self.cap * max(1, len(config["devices"])) + 16), "max_connections")
        if self.max_connections < 1:
            raise ValueError("max_connections must be positive")
        self.sampler = sampler or NvidiaSMISampler(_positive(config.get("sample_timeout", 2.0), "sample_timeout"))
        self.log = EventLog(config.get("telemetry_path"))
        self.socket_path = None
        self._cv = threading.Condition(threading.RLock())
        self._close_lock = threading.Lock()
        self._stop = threading.Event()
        self._closed = False
        self._finalized = False
        self._started = False
        self._sockets = set()
        self._threads = set()
        self._locks = []
        self._fence_paths = {}
        self._durable_records = {}
        self._fence_error = None
        self._boot_id = None
        self._leases = {}
        self._models = {}
        self._rss_residents = {}
        self._ram_observation_error = None
        self._pending = []
        self._controllers = {}
        self._devices = {}
        self._device_limits = {}
        self._sample = {}
        self._sample_at = 0.0
        self._sample_error = None
        self._sample_error_detail = None
        self._listener = None
        self._old_env = None

    @staticmethod
    def _private_dir(path):
        try:
            os.mkdir(path, 0o700)
        except FileExistsError:
            pass
        info = os.lstat(path)
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
            raise AdmissionError("socket directory must be owned by this UID, mode 0700")

    def _resolve_devices(self, sample):
        for spec in self.config["devices"]:
            if not isinstance(spec, dict) or ("index" not in spec and "uuid" not in spec):
                raise AdmissionError("device requires physical index or UUID")
            if "index" in spec:
                nonnegative(spec["index"], "physical index")
            candidates = [row for row in sample.values()
                          if ("index" not in spec or row["index"] == spec["index"])
                          and ("uuid" not in spec or row["uuid"] == spec["uuid"])]
            if len(candidates) != 1:
                raise AdmissionError("configured device is absent or ambiguous")
            row = candidates[0]
            for env in ("CUDA_VISIBLE_DEVICES", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS"):
                if env not in os.environ:
                    continue
                tokens = [s.strip() for s in os.environ[env].split(",") if s.strip()]
                allowed = set()
                for token in tokens:
                    matches = [r["uuid"] for r in sample.values()
                               if (token.isdigit() and r["index"] == int(token))
                               or (token.startswith("GPU-") and r["uuid"].startswith(token))]
                    if len(matches) != 1:
                        raise AdmissionError(f"ambiguous or unsupported {env}")
                    allowed.update(matches)
                if row["uuid"] not in allowed:
                    raise AdmissionError(f"device exceeds {env}")
            budget = nonnegative(spec["vram_bytes"], "vram_bytes")
            if not 0 < budget <= row["total_bytes"] or row["uuid"] in self._devices:
                raise AdmissionError("invalid/duplicate safe VRAM budget")
            self._devices[row["uuid"]] = dict(index=row["index"], vram_bytes=budget)
            self._device_limits[row["uuid"]] = 1

    def _take_locks(self):
        if self._devices:
            self._boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            if not self._boot_id:
                raise AdmissionError("Linux boot identity unavailable")
        for uuid in sorted(self._devices):
            name = "/tmp/astrolabe-gpu-" + hashlib.sha256(uuid.encode()).hexdigest() + ".lock"
            fd = os.open(name, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1:
                    raise AdmissionError("unsafe physical UUID lock file")
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except Exception:
                os.close(fd)
                raise AdmissionError(f"GPU {uuid} already owned or lock unavailable") from None
            self._locks.append(fd)
            self._fence_paths[uuid] = name.removesuffix(".lock") + ".state.json"
            records = self._read_fences(uuid)
            live = [record for record in records.values() if not _owner_exited((record["pid"], record["start_epoch"]))]
            if live:
                raise BrokerBusy(f"GPU {uuid} has live or unverified orphan GPU owners: "
                                 f"{sorted({record['pid'] for record in live})}; startup refused")
            self._durable_records[uuid] = {}
            if records:
                self._store_fences(uuid, {})  # Every old worker is positively gone.

    @staticmethod
    def _check_fence_file(fd):
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1
                or info.st_size > 1024**2):
            raise AdmissionError("unsafe or oversized persistent GPU fence")

    def _read_fences(self, device):
        try:
            fd = os.open(self._fence_paths[device], os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        except FileNotFoundError:
            return {}
        try:
            self._check_fence_file(fd)
            with os.fdopen(fd, "r", encoding="utf-8", closefd=False) as handle:
                raw = handle.read(1024**2 + 1)
            payload = json.loads(raw)
            if (not isinstance(payload, dict) or payload.get("version") != 1
                    or payload.get("uuid") != device or not isinstance(payload.get("boot_id"), str)
                    or not payload["boot_id"] or not isinstance(payload.get("records"), list)):
                raise ValueError("invalid persistent fence schema")
            if str(uuid_module.UUID(payload["boot_id"])) != payload["boot_id"]:
                raise ValueError("invalid persistent boot identity")
            records = {}
            for record in payload["records"]:
                if (not isinstance(record, dict) or record.get("kind") not in ("lease", "model")
                        or not isinstance(record.get("id"), str) or not 0 < len(record["id"]) <= 512
                        or isinstance(record.get("pid"), bool) or not isinstance(record.get("pid"), int)
                        or record["pid"] <= 0 or not isinstance(record.get("start_epoch"), str)
                        or not record["start_epoch"].isascii() or not record["start_epoch"].isdigit()
                        or len(record["start_epoch"]) > 32
                        or str(int(record["start_epoch"])) != record["start_epoch"]):
                    raise ValueError("invalid persistent fence owner")
                key = record["kind"] + ":" + record["id"]
                if key in records:
                    raise ValueError("duplicate persistent fence")
                records[key] = record
            return records if payload["boot_id"] == self._boot_id else {}
        except (ValueError, UnicodeError) as exc:
            raise AdmissionError("persistent GPU fence unreadable; startup refused") from exc
        finally:
            os.close(fd)

    def _store_fences(self, device, records):
        """Caller holds the UUID flock; never truncate the previous valid journal."""
        path = self._fence_paths[device]
        payload = json.dumps(dict(version=1, uuid=device, boot_id=self._boot_id,
                                  records=list(records.values())), separators=(",", ":")).encode()
        if len(payload) > 1024**2:
            raise AdmissionError("persistent GPU fence exceeds 1MiB bound")
        temporary = path + ".tmp-" + uuid_module.uuid4().hex
        fd = None
        try:
            # Validate any existing destination before atomic replacement.
            try:
                existing = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
            except FileNotFoundError:
                pass
            else:
                try:
                    self._check_fence_file(existing)
                finally:
                    os.close(existing)
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
            with os.fdopen(fd, "wb", closefd=False) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(fd)
            os.close(fd)
            fd = None
            os.replace(temporary, path)
            directory = os.open(os.path.dirname(path), os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if fd is not None:
                os.close(fd)
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    def _persist_fence(self, device, kind, identity, owner=None):
        if device is None:
            return
        records = dict(self._durable_records[device])
        key = kind + ":" + identity
        if owner is None:
            records.pop(key, None)
        else:
            records[key] = dict(kind=kind, id=identity, pid=owner[0], start_epoch=owner[1])
        try:
            self._store_fences(device, records)
        except Exception as exc:
            self._fence_error = f"persistent GPU fencing failed: {type(exc).__name__}"
            self._cv.notify_all()
            raise AdmissionError(self._fence_error) from exc
        self._durable_records[device] = records

    def start(self):
        with self._cv:
            if self._closed:
                raise BrokerClosed("broker closed")
            if self._started:
                return self
            try:
                self._sample = validate_sample(self.sampler()) if self.config["devices"] else {}
                self._sample_at = time.monotonic()
                self._resolve_devices(self._sample)
                self._take_locks()
                directory = str(self.config.get("socket_dir", f"/tmp/astrolabe-resource-{os.getuid()}"))
                self._private_dir(directory)
                self.socket_path = os.path.join(directory, "broker-" + uuid_module.uuid4().hex[:16] + ".sock")
                self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self._listener.bind(self.socket_path)
                os.chmod(self.socket_path, 0o600)
                self._listener.listen(min(self.max_connections, 128))
                self._listener.settimeout(0.2)
                self.log.emit("start", devices=self._devices, socket=self.socket_path)
                self._started = True
                self._old_env = os.environ.get("ASTROLABE_RESOURCE_SOCKET")
                os.environ["ASTROLABE_RESOURCE_SOCKET"] = self.socket_path
                self._spawn(self._accept, "accept")
                self._spawn(self._monitor, "monitor")
                return self
            except Exception:
                self.close()
                raise

    def _spawn(self, target, name, *args):
        def run():
            try:
                target(*args)
            finally:
                with self._cv:
                    self._threads.discard(threading.current_thread())
                    self._cv.notify_all()
        thread = threading.Thread(target=run, name=f"resource-broker-{name}", daemon=True)
        self._threads.add(thread)
        thread.start()

    def _accept(self):
        while not self._stop.is_set():
            try:
                conn, _ = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            with self._cv:
                if self._stop.is_set() or len(self._sockets) >= self.max_connections:
                    conn.close()
                    continue
                self._sockets.add(conn)
                self._spawn(self._serve, "rpc", conn)

    def _healthy(self):
        if self._closed:
            raise BrokerClosed("broker closed")
        if self.log.error or self._sample_error or self._fence_error or self._ram_observation_error or time.monotonic() - self._sample_at > self.max_age:
            raise AdmissionError("telemetry unavailable; admission closed")

    def _admission_sample_ready(self):
        # Only sampler failure/staleness is waitable. Preserve fatal errors and
        # their existing messages even when a sample failure is also present.
        if self._closed or self.log.error or self._fence_error or self._ram_observation_error:
            self._healthy()
        return not self._sample_error and time.monotonic() - self._sample_at <= self.max_age

    def _event(self, event, **fields):
        try:
            self.log.emit(event, **fields)
        except Exception as exc:
            self._cv.notify_all()
            raise AdmissionError(str(exc)) from exc

    def _monitor(self):
        while not self._stop.wait(self.interval):
            try:
                sample = validate_sample(self.sampler()) if self._devices else {}
                if any(uuid not in sample or sample[uuid]["index"] != spec["index"] or sample[uuid]["total_bytes"] < spec["vram_bytes"] for uuid, spec in self._devices.items()):
                    raise ValueError("device telemetry changed or disappeared")
                error, detail = None, None
            except Exception as exc:
                sample, error, detail = {}, type(exc).__name__, str(exc)
            with self._cv:
                if self._stop.is_set():
                    return
                self._sample = sample
                self._sample_at = time.monotonic()
                self._sample_error = error
                self._sample_error_detail = detail
                self._reap_exited()
                self._refresh_resident_rss()
                self._reconcile_residents()
                active_keys = {lease["key"]: lease["device"]
                               for lease in self._leases.values()
                               if lease["device"] in self._sample}
                for key, device in active_keys.items():
                    row = self._sample[device]
                    controller = self._controllers[key]
                    controller.peak_vram_bytes = max(
                        controller.peak_vram_bytes, row["total_bytes"] - row["free_bytes"])
                    controller.utilization_sum += row["utilization"]
                    controller.utilization_samples += 1
                    fraction = 1 - row["free_bytes"] / row["total_bytes"]
                    controller.peak_memory_fraction = max(
                        controller.peak_memory_fraction, fraction)
                    if fraction >= self.memory_pressure_fraction:
                        self._device_limits[device] = max(
                            1, self._device_limits[device] // 2)
                        controller.limit = max(1, controller.limit // 2)
                        controller.last_decision = "memory_pressure"
                try:
                    snapshot = self._snapshot()
                    self._event("sample", healthy=snapshot["healthy"],
                                telemetry_error=snapshot["telemetry_error"],
                                sample_error_detail=self._sample_error_detail,
                                devices=snapshot["devices"], controllers=snapshot["controllers"],
                                pending=snapshot["pending"], pending_bytes=snapshot["pending_bytes"])
                except AdmissionError:
                    pass
                self._cv.notify_all()

    def _reap_exited(self):
        dead = {record["owner"] for record in list(self._models.values()) + list(self._leases.values())
                if _owner_exited(record["owner"])}
        dead.update(owner for owner in self._rss_residents if _owner_exited(owner))
        for token, record in list(self._leases.items()):
            if record["owner"] in dead:
                conn = record.get("connection")
                if conn is not None:
                    try:
                        conn.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                self._release(token, error="process exited")
        for model_id, record in list(self._models.items()):
            if record["owner"] in dead:
                try:
                    self._persist_fence(record["device"], "model", model_id)
                except AdmissionError:
                    continue
                del self._models[model_id]
        for owner in dead:
            self._rss_residents.pop(owner, None)
        if not self._rss_residents:
            self._ram_observation_error = None
        self._cv.notify_all()

    def _serve(self, conn):
        token = None
        try:
            conn.settimeout(self.rpc_timeout)
            pid, uid, _ = struct.unpack("3i", conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
            owner = (pid, _epoch(pid))
            if uid != os.getuid() or owner[1] is None:
                raise AdmissionError("peer UID/epoch rejected")
            request = _receive(conn, self.frame)
            op = request.get("op")
            with self._cv:
                if self._stop.is_set():
                    raise BrokerClosed("broker closed")
                if op == "snapshot":
                    response = self._snapshot()
                elif op == "register_model":
                    response = self._register(request, owner)
                elif op == "unregister_model":
                    response = self._unregister(request, owner)
                elif op == "finish":
                    response = self._complete(request, owner)
                elif op == "lease":
                    token = self._acquire(request, owner, conn)
                    record = self._leases[token]
                    response = dict(token=token, device=record["device"])
                else:
                    raise AdmissionError("unknown operation")
            _send(conn, dict(ok=True, **response), self.frame)
            if token:
                # A disconnect does not prove that the worker/device stopped.
                conn.settimeout(None)
                done = _receive(conn, self.frame)
                if done.get("op") not in ("finish", "cancel") or done.get("token") != token:
                    raise AdmissionError("invalid completion token")
                error = done.get("error")
                if error is not None and not isinstance(error, str):
                    raise AdmissionError("error must be text")
                oom = done.get("oom", False)
                if not isinstance(oom, bool):
                    raise AdmissionError("oom must be boolean")
                with self._cv:
                    if done["op"] == "finish":
                        if not self._release(token, error=error, oom=oom):
                            raise AdmissionError("durable GPU fence cleanup failed; lease retained")
                    else:
                        self._abandon(token, error=error or "cancelled")
                    token = None
                _send(conn, dict(ok=True), self.frame)
        except Exception as exc:
            try:
                _send(conn, dict(ok=False, error=str(exc), kind=type(exc).__name__), self.frame)
            except (OSError, AdmissionError):
                pass
        finally:
            with self._cv:
                if token:
                    self._abandon(token, error="connection lost")
                self._sockets.discard(conn)
                self._cv.notify_all()
            conn.close()

    def _request(self, request, owner, defer_affinity=False):
        resource = request.get("resource", "gpu")
        if resource not in ("gpu", "cpu"):
            raise AdmissionError("resource must be gpu or cpu")
        if resource == "gpu" and not self._devices:
            raise AdmissionError("no configured GPUs")
        result = dict(owner=owner, resource=resource)
        for name in ("backend", "identity", "shape", "phase"):
            value = request.get(name, "compute" if name == "phase" else "")
            if not isinstance(value, str) or len(value) > 512:
                raise AdmissionError(f"invalid {name}")
            result[name] = value
        for name in ("resident_bytes", "scratch_bytes", "ram_bytes", "output_bytes"):
            result[name] = nonnegative(request.get(name, 0), name)
        if resource == "cpu":
            result["resident_bytes"] = result["scratch_bytes"] = 0
        certified = request.get("concurrency_certified", False)
        if not isinstance(certified, bool):
            raise AdmissionError("concurrency_certified must be boolean")
        result["certified"] = certified
        capacity_only = request.get("capacity_only", False)
        evaluator_inference = request.get("evaluator_inference", False)
        if type(capacity_only) is not bool or type(evaluator_inference) is not bool:
            raise AdmissionError("capacity_only and evaluator_inference must be boolean")
        reservation_token = request.get("reservation_token")
        if capacity_only and (reservation_token is not None or evaluator_inference):
            raise AdmissionError("invalid evaluation capacity reservation")
        if evaluator_inference and (reservation_token is None or resource != "gpu"):
            raise AdmissionError("evaluator inference requires a capacity reservation")
        if reservation_token is not None:
            if not isinstance(reservation_token, str):
                raise AdmissionError("invalid capacity reservation token")
            if any(result[name] for name in
                   ("resident_bytes", "scratch_bytes", "ram_bytes", "output_bytes")):
                raise AdmissionError("reserved operations cannot charge capacity twice")
        result.update(capacity_only=capacity_only,
                      evaluator_inference=evaluator_inference,
                      reservation_token=reservation_token)
        result["key"] = tuple(result[name] for name in ("backend", "identity", "shape", "phase"))
        result["unknown"] = (resource == "gpu" and reservation_token is None
                             and not (result["resident_bytes"] + result["scratch_bytes"]))
        result["bytes"] = result["ram_bytes"] + result["output_bytes"]
        prepared_model_id = request.get("prepared_model_id")
        if prepared_model_id is not None:
            result["prepared_model_id"] = prepared_model_id
            self._check_prepared_placement(result)
        result["unknown_ram_initialization"] = (self.observe_resident_ram
            and result["phase"] == "load" and result["ram_bytes"] == 0
            and prepared_model_id is None)
        if result["ram_bytes"] > self.ram_bytes or result["output_bytes"] > self.output_bytes or (resource == "cpu" and self.cpu_slots == 0):
            raise AdmissionError("request exceeds configured capacity")
        if resource == "gpu":
            # Placeholder is fully charged to _pending, never used by _fit.
            result["affinity"] = None
            result["key"] += (None,)
            if not defer_affinity:
                self._bind_request(result, request)
        return result

    def _evaluation_reservation(self, request):
        """Resolve an operation's live, same-owner memory reservation."""
        token = request.get("reservation_token")
        if token is None:
            return None
        record = self._leases.get(token)
        if (record is None or not record.get("capacity_only") or record.get("fenced")
                or record["owner"] != request["owner"]
                or record["identity"] != request["identity"]
                or record["backend"] != request["backend"]
                or record["shape"] != request["shape"]
                or (request["resource"] == "gpu"
                    and (record["resource"] != "gpu"
                         or record["device"] != request["affinity"]))):
            raise AdmissionError("capacity reservation is absent or belongs to another request")
        return record

    def _bind_request(self, result, raw):
        affinity = self._affinity(raw.get("device"), raw.get("visible_devices"))
        if affinity not in self._devices:
            raise AdmissionError("actual client device is outside configured broker devices")
        if result["resident_bytes"] + result["scratch_bytes"] > self._devices[affinity]["vram_bytes"]:
            raise AdmissionError("request exceeds safe VRAM budget")
        result["affinity"] = affinity
        result["key"] = result["key"][:4] + (affinity,)

    @staticmethod
    def _sample_owner_matches(sample, owner):
        proof = sample.get("process_identities", {}).get(owner[0])
        # Legacy injected samples keep their existing contract. Kernel-attested
        # rows must match the registered PID epoch AND its current live epoch.
        return proof is None or (
            proof["leader"]["start_ticks"] == owner[1]
            and _epoch(owner[0]) == owner[1])

    def _device_fit(self, request, device, extra=0):
        if request.get("reservation_token") is not None:
            # The entire input/workspace/output allowance was acquired before
            # prepare. This operation borrows it; it cannot allocate a second
            # allowance or move it to another device/owner.
            self._evaluation_reservation(request)
            return 0
        budget = self._devices[device]["vram_bytes"]
        models = [r for r in self._models.values() if r["device"] == device]
        leases = [r for r in self._leases.values() if r["device"] == device]
        sample = self._sample[device]
        if any(not self._sample_owner_matches(sample, owner)
               for owner in [r["owner"] for r in models + leases] + [request["owner"]]):
            return None
        external = set(sample["processes"]) - {r["owner"][0] for r in models + leases} - {request["owner"][0]}
        own_unknown_models = [r for r in models if r.get("unknown") and not r.get("observed") and r["owner"] == request["owner"]]
        reserved = sum(r["reserved"] for r in models + leases)
        if reserved > budget:
            return None
        managed_pids = {r["owner"][0] for r in models + leases}
        measured_managed = sum(used for pid, used in sample["processes"].items() if pid in managed_pids)
        unmaterialized = max(0, reserved - measured_managed)
        exclusive = request["unknown"] or bool(own_unknown_models)
        if exclusive:
            if external or leases or any(r.get("unknown") and not r.get("observed") and r["owner"] != request["owner"] for r in models):
                return None
            if request.get("phase") == "load" and own_unknown_models:
                return None
            # A still-unobserved own resident can run exclusively within its
            # reservation; its actual process sample refines that reservation.
            if own_unknown_models and (request["resident_bytes"] or request["scratch_bytes"] > sample["free_bytes"]):
                return None
            needed = min(budget - reserved, sample["free_bytes"] - unmaterialized)
            if needed < 0 or (needed == 0 and (not models or request.get("phase") == "load")):
                return None
        else:
            if any(r.get("unknown", False) and not r.get("observed", False) for r in models + leases):
                return None
            resident = request["resident_bytes"]
            if request.get("phase") != "load":
                resident = max(0, resident - sum(r["reserved"] for r in models if self._model_matches(r, request)))
            needed = resident + request["scratch_bytes"] + extra
        if needed < 0 or reserved > budget:
            return None
        if reserved + needed > budget or needed + unmaterialized > sample["free_bytes"]:
            return None
        return needed

    def _fit(self, request):
        if request.get("reservation_token") is not None:
            self._evaluation_reservation(request)
            # One request's peak allowance cannot fund simultaneous operations.
            # Prepare, infer and finalize pass this token in sequence.
            if any(r.get("reservation_token") == request["reservation_token"]
                   for r in self._leases.values()):
                return None
        if request.get("prepared_model_id") is not None:
            # Revalidate queued claims under the same admission lock. The
            # prepared registration must survive until final registration.
            self._check_prepared_placement(request)
        leases = list(self._leases.values())
        if self.observe_resident_ram and request["phase"] == "load":
            loads = [lease for lease in leases if lease["phase"] == "load"]
            if loads and (request["unknown_ram_initialization"] or any(lease["unknown_ram_initialization"] for lease in loads)):
                return None
        if self._ram_used() + request["ram_bytes"] > self.ram_bytes:
            return None
        if sum(r["output_bytes"] for r in leases) + request["output_bytes"] > self.output_bytes:
            return None
        if request["resource"] == "cpu":
            if request.get("capacity_only"):
                return None, 0
            return (None, 0) if sum(r["resource"] == "cpu" and not r.get("capacity_only")
                                    for r in leases) < self.cpu_slots else None
        controller = self._controllers.setdefault(request["key"], _Controller())
        bound_devices = {r["device"] for r in self._models.values() if self._model_matches(r, request) and r["device"]}
        for device in self._devices:
            if device != request["affinity"]:
                continue
            if request["phase"] != "load" and bound_devices and device not in bound_devices:
                continue
            device_leases = [r for r in leases if r["device"] == device]
            if any(r.get("fenced") for r in device_leases):
                continue
            if request.get("evaluator_inference"):
                self._evaluation_reservation(request)
                # Evaluator slots count complete model computations only.
                active = [r for r in device_leases if r.get("evaluator_inference")
                          and r["backend"] == request["backend"]
                          and r["identity"] == request["identity"]]
                if len(active) >= (controller.limit if request["certified"] else 1):
                    continue
            elif not request.get("capacity_only") and not request.get("reservation_token"):
                # Preserve legacy scheduling for unconverted modules. A staged
                # evaluator and its transfers do not consume this compute gate.
                active = [r for r in device_leases
                          if not r.get("capacity_only") and not r.get("reservation_token")]
                if len(active) >= self._device_limits[device]:
                    continue
                if active and (not request["certified"] or request["phase"] == "load"
                               or any(not r["certified"] or r["phase"] == "load" for r in active)):
                    continue
                same_key = [r for r in active if r["key"] == request["key"]]
                if len(same_key) >= (controller.limit if request["certified"] else 1):
                    continue
            reserved = self._device_fit(request, device)
            if reserved is not None:
                return device, reserved
        return None

    def _acquire(self, raw, owner, conn):
        recover = self.telemetry_recovery
        if recover:
            ready = self._admission_sample_ready()
        else:
            self._healthy()
            ready = True
        request = self._request(raw, owner, defer_affinity=not ready)
        timeout = raw.get("timeout")
        if timeout is not None:
            nonnegative_timeout = isinstance(timeout, (int, float)) and not isinstance(timeout, bool) and math.isfinite(timeout) and timeout >= 0
            if not nonnegative_timeout:
                raise AdmissionError("invalid timeout")
        deadline = None if timeout is None else time.monotonic() + timeout
        fit = self._fit(request) if ready else None
        queued = False
        recovery_deadline = None
        recovery_sample_at = None
        waiting = False
        try:
            if fit is None:
                if len(self._pending) >= self.max_pending or sum(r["bytes"] for r in self._pending) + request["bytes"] > self.max_pending_bytes:
                    raise AdmissionError("pending queue budget exceeded")
                self._pending.append(request)
                queued = True
            while fit is None:
                if recover:
                    ready = self._admission_sample_ready()
                else:
                    self._healthy()
                    ready = True
                if _epoch(owner[0]) != owner[1]:
                    raise AdmissionError("worker epoch expired")
                readable, _, _ = select.select([conn], [], [], 0)
                if readable:
                    raise AdmissionError("queued request cancelled/disconnected")
                now = time.monotonic()
                remaining = None if deadline is None else deadline - now
                if remaining is not None and remaining <= 0:
                    raise AdmissionTimeout("admission deadline expired")
                # A cached sample cannot complete a readiness wait, even if its
                # age is still within the ordinary freshness limit.
                if waiting and self._sample_at <= recovery_sample_at:
                    ready = False
                if not ready:
                    if recovery_deadline is None:
                        recovery_deadline = now + self.max_age
                    if not waiting:
                        recovery_sample_at = self._sample_at
                        waiting = True
                        self._event("telemetry_recovery_wait", pid=owner[0],
                                    resource=request["resource"], key=request["key"],
                                    error="telemetry unavailable; admission closed",
                                    sample_error=self._sample_error,
                                    sample_error_detail=self._sample_error_detail, sample_at=self._sample_at,
                                    recovery_deadline=recovery_deadline, queue_deadline=deadline,
                                    pending=len(self._pending),
                                    pending_bytes=sum(r["bytes"] for r in self._pending))
                    if now >= recovery_deadline:
                        self._event("telemetry_recovery_timeout", pid=owner[0],
                                    error="telemetry unavailable; admission closed",
                                    sample_error=self._sample_error)
                        raise AdmissionTimeout("telemetry unavailable; admission closed; readiness deadline expired")
                    remaining = min(recovery_deadline - now, remaining) if remaining is not None else recovery_deadline - now
                else:
                    if waiting:
                        if now >= recovery_deadline:
                            raise AdmissionTimeout("telemetry unavailable; admission closed; readiness deadline expired")
                        if request["resource"] == "gpu":
                            self._bind_request(request, raw)
                        self._event("telemetry_recovery_ready", pid=owner[0],
                                    resource=request["resource"], key=request["key"],
                                    failed_sample_at=recovery_sample_at, sample_at=self._sample_at)
                        waiting = False
                    fit = self._fit(request)
                    if fit is not None:
                        break
                self._cv.wait(min(0.1, remaining) if remaining is not None else 0.1)
            self._healthy()
            if queued and deadline is not None and time.monotonic() >= deadline:
                raise AdmissionTimeout("admission deadline expired")
            token = uuid_module.uuid4().hex
            device, reserved = fit
            request.update(device=device, reserved=reserved, token=token, connection=conn, fenced=False)
            self._event("acquire", token=token, pid=owner[0], device=device, key=request["key"], reserved_bytes=reserved)
            self._persist_fence(device, "lease", token, owner)
            self._leases[token] = request
            if (device and not request.get("capacity_only")
                    and (not request.get("reservation_token") or request.get("evaluator_inference"))):
                controller = self._controllers[request["key"]]
                controller.tick()
                controller.active += 1
                controller.peak_vram_bytes = max(controller.peak_vram_bytes, self._sample[device]["total_bytes"] - self._sample[device]["free_bytes"])
            return token
        finally:
            if queued:
                # Equal requests from one owner can wait concurrently. Remove
                # this exact ledger entry, never another equal placeholder.
                for index, pending in enumerate(self._pending):
                    if pending is request:
                        del self._pending[index]
                        break
            self._cv.notify_all()

    def _abandon(self, token, error):
        record = self._leases.get(token)
        if record is None:
            return
        if record["device"] and not _owner_exited(record["owner"]):
            record["fenced"] = True
            record["connection"] = None
            self._controllers[record["key"]].limit = 1
            try:
                self._event("fence", token=token, pid=record["owner"][0], error=error)
            except AdmissionError:
                pass
            self._cv.notify_all()
        else:
            self._release(token, error=error)

    def _complete(self, request, owner):
        token = request.get("token")
        if not isinstance(token, str):
            raise AdmissionError("completion token required")
        record = self._leases.get(token)
        if record is None:
            return dict(released=False)
        if record["owner"] != owner:
            raise AdmissionError("completion belongs to another PID/start epoch")
        error, oom = request.get("error"), request.get("oom", False)
        if (error is not None and not isinstance(error, str)) or not isinstance(oom, bool):
            raise AdmissionError("invalid completion fields")
        conn = record.get("connection")
        if not self._release(token, error=error, oom=oom):
            raise AdmissionError("durable GPU fence cleanup failed; lease retained")
        if conn is not None:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        return dict(released=True)

    def _release(self, token, error=None, oom=False):
        request = self._leases.get(token)
        if request is None:
            return True
        if (request.get("capacity_only") and not _owner_exited(request["owner"])
                and any(r.get("reservation_token") == token for r in self._leases.values())):
            # A disconnect or out-of-order finish is not device completion.
            request["fenced"] = True
            return False
        try:
            self._persist_fence(request["device"], "lease", token)
        except AdmissionError:
            request["fenced"] = True
            return False
        del self._leases[token]
        if (request["device"] and not request.get("capacity_only")
                and (not request.get("reservation_token") or request.get("evaluator_inference"))):
            controller = self._controllers[request["key"]]
            controller.tick()
            controller.active -= 1
            if oom or (error and ("out of memory" in error.lower() or "oom" in error.lower())):
                controller.limit = max(1, controller.limit // 2)
                self._device_limits[request["device"]] = max(
                    1, self._device_limits[request["device"]] // 2)
                controller.last_decision = "oom"
                controller.window_completed = 0
                controller.busy = 0.0
                controller.throughput = None
            elif error is None:
                controller.completed += 1
                controller.window_completed += 1
                if controller.window_completed >= self.min_observations and controller.busy >= self.window:
                    rate = controller.window_completed / controller.busy
                    degraded = controller.throughput is not None and rate < controller.throughput * (1 - self.drop)
                    ready = any(
                        r["resource"] == "gpu" and r["certified"]
                        and r["phase"] != "load"
                        and r["affinity"] == request["device"]
                        for r in self._pending)
                    utilization = (controller.utilization_sum / controller.utilization_samples
                                   if controller.utilization_samples else None)
                    pressure = controller.peak_memory_fraction >= self.memory_pressure_fraction
                    if degraded:
                        controller.limit = max(1, controller.limit // 2)
                        self._device_limits[request["device"]] = max(
                            1, self._device_limits[request["device"]] // 2)
                        controller.last_decision = "throughput_regression"
                    elif pressure:
                        controller.last_decision = "memory_pressure"
                    elif (ready and request["certified"] and utilization is not None
                          and utilization < self.target_utilization):
                        controller.limit = min(self.cap, controller.limit + 1)
                        self._device_limits[request["device"]] = min(
                            self.cap, self._device_limits[request["device"]] + 1)
                        controller.last_decision = "probe_with_activity_headroom"
                    else:
                        controller.last_decision = (
                            "no_ready_work" if not ready else
                            "activity_saturated" if utilization is not None else
                            "insufficient_activity_samples")
                    try:
                        self._event("controller_decision", device=request["device"],
                                    backend=request["backend"], identity=request["identity"],
                                    shape=request["shape"], phase=request["phase"],
                                    reason=controller.last_decision, throughput=rate,
                                    mean_utilization=utilization,
                                    memory_fraction=controller.peak_memory_fraction,
                                    concurrency=controller.limit,
                                    device_concurrency=self._device_limits[request["device"]])
                    except AdmissionError:
                        pass
                    controller.throughput = rate
                    controller.busy = 0.0
                    controller.window_completed = 0
                    controller.utilization_sum = 0.0
                    controller.utilization_samples = 0
                    controller.peak_memory_fraction = 0.0
        try:
            self._event("release", token=token, error=error, oom=oom)
        except AdmissionError:
            pass  # A telemetry fault must never prevent release.
        self._cv.notify_all()
        return True

    @staticmethod
    def _model_matches(model, request):
        metadata = model.get("metadata") or {}
        return model["owner"] == request["owner"] and all(
            name not in metadata or metadata[name] == request.get(name)
            for name in ("backend", "identity"))

    def _reconcile_residents(self):
        groups = {}
        for model in self._models.values():
            if model["device"] is not None:
                groups.setdefault((model["device"], model["owner"]), []).append(model)
        for (device, owner), models in groups.items():
            if not self._sample_owner_matches(self._sample.get(device, {}), owner):
                continue  # A reused/stale leader cannot refine residency.
            active = [r for r in self._leases.values() if r["owner"] == owner and r["device"] == device]
            if any(r["phase"] == "load" for r in active):
                continue  # Do not shrink an in-progress load's reservation.
            used = self._sample.get(device, {}).get("processes", {}).get(owner[0])
            if used is None:
                continue  # Absence of a PID is not evidence of zero resident memory.
            unknown = [model for model in models if model["unknown"]]
            known = [model for model in models if not model["unknown"]]
            for model in known:
                model["reserved"] = max(model["reserved"], model["declared_bytes"]) if active else model["declared_bytes"]
                model["observed"] = True
            remainder = max(0, used - sum(model["reserved"] for model in known))
            for index, model in enumerate(unknown):
                # During execute, the observed process peak is a conservative
                # resident upper bound. This also avoids requiring an idle
                # sampler tick between every pair of short compute requests.
                observed = remainder if index == 0 else 0
                model["reserved"] = max(model["reserved"], observed) if active and model["observed"] else observed
                model["observed"] = True
            if not unknown and known:
                known[0]["reserved"] += remainder

    def _model_owner(self, model_id, peer):
        parts = model_id.rsplit(":", 2)
        if len(parts) != 3 or not parts[1].isdigit() or not parts[2]:
            return peer, None
        pid = int(parts[1])
        epoch = _epoch(pid)
        try:
            same_uid = os.stat(f"/proc/{pid}").st_uid == os.getuid()
        except OSError:
            same_uid = False
        if not same_uid or epoch is None:
            raise AdmissionError("model worker PID is unavailable or belongs to another UID")
        return (pid, epoch), parts[2]

    def _affinity(self, chosen, visible):
        inventory = sorted(self._sample.values(), key=lambda row: row["index"])
        if visible is None:
            order = [row["uuid"] for row in inventory]
        else:
            tokens = visible.split(",") if isinstance(visible, str) else visible
            if not isinstance(tokens, (list, tuple)):
                raise AdmissionError("visible_devices must be a CUDA mask or list")
            order = []
            for item in tokens:
                token = str(item).strip()
                if not token:
                    continue
                matches = [row["uuid"] for row in inventory if
                           (token.isdigit() and row["index"] == int(token)) or
                           (token.startswith("GPU-") and row["uuid"].startswith(token))]
                if len(matches) != 1 or matches[0] in order:
                    raise AdmissionError("ambiguous client CUDA visibility")
                order.extend(matches)
        if not order:
            raise AdmissionError("client has no visible GPU")
        if chosen is None or chosen == "cuda":
            return order[0]
        if isinstance(chosen, str) and chosen.startswith("cuda:"):
            try:
                logical = int(chosen.split(":", 1)[1])
            except ValueError:
                raise AdmissionError("invalid CUDA device") from None
            if not 0 <= logical < len(order):
                raise AdmissionError("CUDA logical device is not visible")
            return order[logical]
        matches = [row["uuid"] for row in inventory if
                   chosen == row["uuid"] or chosen == row["index"] or chosen == str(row["index"])]
        if len(matches) != 1 or matches[0] not in order:
            raise AdmissionError("explicit device is not visible")
        return matches[0]

    def _registration_devices(self, chosen, owner, visible):
        if chosen == "cpu":
            return [None]
        if chosen is not None:
            device = self._affinity(chosen, visible)
            return [device] if device in self._devices else []
        loads = {r["device"] for r in self._leases.values() if r["owner"] == owner and r["phase"] == "load" and r["device"]}
        if len(loads) == 1:
            return list(loads)
        observed = [device for device in self._devices
                    if owner[0] in self._sample.get(device, {}).get("processes", {})
                    and self._sample_owner_matches(self._sample[device], owner)]
        if len(observed) == 1:
            return observed
        device = self._affinity(None, visible)
        return [device] if device in self._devices else []

    def _resident_ram_by_owner(self):
        totals = {}
        for model in self._models.values():
            if model["resident_ram_source"] == "declared":
                totals[model["owner"]] = totals.get(model["owner"], 0) + model["resident_ram_bytes"]
        for owner, record in self._rss_residents.items():
            observed = self.ram_bytes if record["bytes"] is None else record["bytes"]
            totals[owner] = max(totals.get(owner, 0), observed)
        return totals

    def _ram_used(self):
        return (sum(self._resident_ram_by_owner().values())
                + sum(lease["ram_bytes"] for lease in self._leases.values()))

    @staticmethod
    def _read_resident_rss(owner):
        try:
            if _epoch(owner[0]) != owner[1]:
                raise ValueError("worker epoch unavailable")
            rows = [line.split() for line in Path(f"/proc/{owner[0]}/status").read_text().splitlines()
                    if line.startswith("VmRSS:")]
            if (len(rows) != 1 or len(rows[0]) != 3 or rows[0][2] != "kB"
                    or not rows[0][1].isascii() or not rows[0][1].isdigit()
                    or int(rows[0][1]) <= 0 or _epoch(owner[0]) != owner[1]):
                raise ValueError("VmRSS unavailable or worker epoch changed")
            return int(rows[0][1]) * 1024
        except (OSError, ValueError) as exc:
            raise AdmissionError("worker resident RAM observation unavailable (proc status VmRSS)") from exc

    def _record_resident_rss(self, owner, value):
        record = self._rss_residents.setdefault(owner, dict(bytes=None, source="proc_status:VmRSS", observed_at=None))
        if value is not None:
            record["bytes"] = max(record["bytes"] or 0, value)
            record["observed_at"] = time.monotonic()

    def _refresh_resident_rss(self):
        error = None
        for owner in list(self._rss_residents):
            try:
                self._record_resident_rss(owner, self._read_resident_rss(owner))
            except AdmissionError as exc:
                error = str(exc)
        if sum(self._resident_ram_by_owner().values()) > self.ram_bytes:
            error = "observed resident RAM exceeds configured capacity"
        self._ram_observation_error = error

    def _resident_ram_plan(self, owner, device, resident_ram, source):
        load = next((lease for lease in self._leases.values()
                     if lease["owner"] == owner and lease["phase"] == "load"
                     and lease["device"] == device), None)
        transfer = min(resident_ram, load["ram_bytes"]) if load is not None else 0
        previous = self._resident_ram_by_owner().get(owner, 0)
        declared = sum(model["resident_ram_bytes"] for model in self._models.values()
                       if model["owner"] == owner and model["resident_ram_source"] == "declared")
        rss = self._rss_residents.get(owner, {}).get("bytes") or 0
        proposed = max(declared, rss, resident_ram) if source != "declared" else max(declared + resident_ram, rss)
        if self._ram_used() + proposed - previous - transfer > self.ram_bytes:
            if source != "declared":
                self._record_resident_rss(owner, resident_ram)
                self._ram_observation_error = "observed resident RAM exceeds configured available capacity"
            raise AdmissionError("model resident RAM exceeds configured available capacity")
        return load, transfer

    def _prepared_record(self, model_id, owner):
        if not isinstance(model_id, str) or not model_id or len(model_id) > 512:
            raise AdmissionError("valid prepared_model_id required")
        record = self._models.get(model_id)
        if (record is None or record["owner"] != owner
                or _epoch(owner[0]) != owner[1]):
            raise AdmissionError("prepared model requires the same live owner/PID epoch")
        if (record["device"] is not None or record["reserved"] != 0
                or record["resident_ram_source"] != "declared"
                or record["resident_ram_bytes"] <= 0
                or (record.get("metadata") or {}).get("residency") != "prepared_cpu_weights"):
            raise AdmissionError("prepared model must be positive declared CPU prepared weights without VRAM")
        return record

    def _check_prepared_placement(self, request):
        prepared_id = request["prepared_model_id"]
        record = self._prepared_record(prepared_id, request["owner"])
        if request["phase"] != "load" or request["ram_bytes"] != 0:
            raise AdmissionError("prepared placement requires phase=load and ram_bytes=0")
        for key in ("backend", "identity"):
            if key in record["metadata"] and record["metadata"][key] != request[key]:
                raise AdmissionError("prepared placement model identity/backend mismatch")
        if any(other is not request and other.get("prepared_model_id") == prepared_id
               for other in [*self._leases.values(), *self._pending]):
            raise AdmissionError("prepared model already claimed by an active or pending placement")
        return record

    def _register_prepared(self, request, owner, peer, worker_epoch, size, resident_ram):
        prepared_id = request["prepared_model_id"]
        if owner != peer:
            raise AdmissionError("prepared transfer must be registered by its live worker")
        prepared = self._prepared_record(prepared_id, owner)
        prepared_ram = prepared["resident_ram_bytes"]
        if resident_ram > prepared_ram:
            raise AdmissionError("prepared RAM does not cover final resident RAM")
        metadata = request.get("metadata")
        if (metadata or {}).get("residency") == "prepared_cpu_weights":
            raise AdmissionError("final residency cannot be prepared CPU weights")
        for key in ("backend", "identity"):
            if key in prepared["metadata"] and (metadata or {}).get(key) != prepared["metadata"][key]:
                raise AdmissionError("prepared transfer model identity/backend mismatch")
        devices = self._registration_devices(request.get("device"), owner, request.get("visible_devices"))
        loads = [lease for lease in self._leases.values()
                 if lease["owner"] == owner and lease["phase"] == "load"
                 and lease.get("prepared_model_id") == prepared_id
                 and lease["device"] in devices and not lease.get("fenced")]
        if len(loads) != 1:
            raise AdmissionError("prepared transfer requires its matching target load lease")
        load = loads[0]
        device = load["device"]
        if load["ram_bytes"] != 0 or (device is None and size):
            raise AdmissionError("invalid prepared placement RAM/device declaration")
        size_for_device = 0
        if device is not None:
            if not self._sample_owner_matches(self._sample[device], owner):
                raise AdmissionError("prepared transfer GPU owner epoch mismatch")
            size_for_device = size or (load["reserved"] if load["unknown"]
                                       else self._devices[device]["vram_bytes"])
            if (load["reserved"] < size_for_device
                    or (not load["unknown"] and load["resident_bytes"] < size_for_device)):
                raise AdmissionError("prepared target load does not cover resident VRAM")
        previous = self._resident_ram_by_owner().get(owner, 0)
        declared = sum(model["resident_ram_bytes"] for model in self._models.values()
                       if model["owner"] == owner and model["resident_ram_source"] == "declared")
        rss = self._rss_residents.get(owner, {}).get("bytes") or 0
        proposed = max(declared - prepared_ram + resident_ram, rss)
        if self._ram_used() + proposed - previous > self.ram_bytes:
            raise AdmissionError("final resident RAM exceeds configured available capacity")
        record = dict(owner=owner, registrar=peer, worker_epoch=worker_epoch,
                      device=device, reserved=size_for_device, declared_bytes=size if device else 0,
                      resident_ram_bytes=resident_ram, resident_ram_source="declared",
                      unknown=device is not None and size == 0, observed=device is None,
                      metadata=metadata)
        # Both operations may fail. Keep the entire prepared ledger and load
        # reservation unchanged until validation, event and fencing succeed.
        self._event("register_model", model_id=request["model_id"], device=device,
                    reserved_bytes=size_for_device, resident_ram_bytes=resident_ram,
                    resident_ram_source="declared", pid=owner[0], prepared_model_id=prepared_id,
                    prepared_ram_bytes=prepared_ram, transferred_ram_bytes=resident_ram,
                    released_ram_bytes=prepared_ram - resident_ram)
        self._persist_fence(device, "model", request["model_id"], owner)
        self._models[request["model_id"]] = record
        del self._models[prepared_id]
        load["reserved"] -= size_for_device
        load["resident_bytes"] = max(0, load["resident_bytes"] - size_for_device)
        self._cv.notify_all()
        return dict(model_id=request["model_id"], device=device, resident_bytes=size_for_device,
                    resident_ram_bytes=resident_ram, resident_ram_source="declared",
                    prepared_model_id=prepared_id, transferred_ram_bytes=resident_ram)

    def _register(self, request, owner):
        self._healthy()
        model_id = request.get("model_id")
        if not isinstance(model_id, str) or not model_id or len(model_id) > 512:
            raise AdmissionError("nonempty model_id required")
        if model_id in self._models:
            raise AdmissionError("model_id already registered; unregister explicitly")
        size = nonnegative(request.get("resident_bytes", 0), "resident_bytes")
        resident_ram = nonnegative(request.get("resident_ram_bytes", 0), "resident_ram_bytes")
        metadata = request.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            raise AdmissionError("metadata must be an object")
        peer = owner
        owner, worker_epoch = self._model_owner(model_id, peer)
        if request.get("prepared_model_id") is not None:
            return self._register_prepared(request, owner, peer, worker_epoch, size, resident_ram)
        source = "declared"
        if self.observe_resident_ram and resident_ram == 0:
            source = "proc_status:VmRSS"
            try:
                resident_ram = self._read_resident_rss(owner)
            except AdmissionError as exc:
                self._record_resident_rss(owner, None)
                self._ram_observation_error = str(exc)
                raise
        devices = self._registration_devices(request.get("device"), owner, request.get("visible_devices"))
        for device in devices:
            if any(lease["owner"] == owner and lease["device"] == device
                   and lease.get("prepared_model_id") is not None for lease in self._leases.values()):
                raise AdmissionError("prepared placement registration requires matching prepared_model_id")
            if device is not None and not self._sample_owner_matches(self._sample[device], owner):
                continue  # Also guard registration backed by an existing load.
            ram_load, ram_transfer = self._resident_ram_plan(owner, device, resident_ram, source)
            if device is None:
                self._event("register_model", model_id=model_id, device=None, reserved_bytes=0, resident_ram_bytes=resident_ram, resident_ram_source=source, pid=owner[0])
                if ram_load is not None:
                    ram_load["ram_bytes"] -= ram_transfer
                if source != "declared":
                    self._record_resident_rss(owner, resident_ram)
                self._models[model_id] = dict(owner=owner, registrar=peer, worker_epoch=worker_epoch, device=None, reserved=0, declared_bytes=0, resident_ram_bytes=resident_ram, resident_ram_source=source, unknown=False, observed=True, metadata=metadata)
                return dict(model_id=model_id, device=None, resident_bytes=0, resident_ram_bytes=resident_ram, resident_ram_source=source)
            size_for_device = size or self._devices[device]["vram_bytes"]
            load = next((r for r in self._leases.values() if r["owner"] == owner and r["device"] == device and r["phase"] == "load"
                         and ((r["resident_bytes"] >= size_for_device and r["reserved"] >= size_for_device)
                              or (r["unknown"] and (size == 0 or r["reserved"] >= size)))), None)
            if load is not None and size == 0:
                size_for_device = load["reserved"]
            if load is None:
                probe = dict(owner=owner, unknown=(size == 0), resident_bytes=size, scratch_bytes=0, phase="load")
                fit = self._device_fit(probe, device)
                if fit is None:
                    continue
                size_for_device = fit
                if size == 0:
                    used = self._sample[device]["processes"].get(owner[0], 0)
                    other_reserved = sum(r["reserved"] for r in list(self._models.values()) + list(self._leases.values()) if r["device"] == device)
                    available = self._devices[device]["vram_bytes"] - other_reserved
                    if used > available:
                        continue
                    size_for_device = min(available, fit + used)
            self._event("register_model", model_id=model_id, device=device, reserved_bytes=size_for_device, resident_ram_bytes=resident_ram, resident_ram_source=source, pid=owner[0])
            self._persist_fence(device, "model", model_id, owner)
            if ram_load is not None:
                ram_load["ram_bytes"] -= ram_transfer
            if source != "declared":
                self._record_resident_rss(owner, resident_ram)
            if load is not None:
                load["reserved"] -= size_for_device
                load["resident_bytes"] = max(0, load["resident_bytes"] - size_for_device)
            self._models[model_id] = dict(owner=owner, registrar=peer, worker_epoch=worker_epoch,
                                         device=device, reserved=size_for_device, declared_bytes=size,
                                         resident_ram_bytes=resident_ram, resident_ram_source=source, unknown=(size == 0), observed=False, metadata=metadata)
            return dict(model_id=model_id, device=device, resident_bytes=size_for_device, resident_ram_bytes=resident_ram, resident_ram_source=source)
        raise AdmissionError("model resident reservation unavailable")

    def _unregister(self, request, owner):
        model_id = request.get("model_id")
        record = self._models.get(model_id)
        if record is None:
            return dict(removed=False)
        if record["owner"] != owner and not _owner_exited(record["owner"]):
            raise AdmissionError("model belongs to another PID/epoch")
        if any(lease.get("prepared_model_id") == model_id
               for lease in [*self._leases.values(), *self._pending]):
            raise AdmissionError("cannot unregister prepared weights claimed by placement")
        if any(r["owner"] == owner and r["device"] == record["device"] for r in self._leases.values()):
            raise AdmissionError("cannot unregister a model while its owner holds a device lease")
        self._persist_fence(record["device"], "model", model_id)
        del self._models[model_id]
        try:
            self._event("unregister_model", model_id=model_id)
        except AdmissionError:
            pass
        self._cv.notify_all()
        return dict(removed=True)

    def _snapshot(self):
        age = time.monotonic() - self._sample_at
        def device_state(device, spec):
            row = self._sample.get(device)
            active = [r for r in self._leases.values() if r["device"] == device]
            ready = [r for r in self._pending
                     if r["resource"] == "gpu" and r["affinity"] == device]
            reason = ("telemetry_unavailable" if row is None else
                      "running_device_work" if active else
                      "awaiting_resource_admission" if ready else
                      "cpu_stage_in_progress" if any(r["resource"] == "cpu"
                                                    for r in self._leases.values()) else
                      "no_ready_device_request")
            return dict(**spec, sample=row, concurrency=self._device_limits[device],
                        activity_reason=reason)
        return dict(closed=self._closed, healthy=not (self._closed or self.log.error or self._sample_error or self._fence_error or self._ram_observation_error or age > self.max_age),
                    telemetry_error=self.log.error or self._sample_error, fence_error=self._fence_error,
                    ram_observation_error=self._ram_observation_error, sample_age=age,
                    devices={uuid: device_state(uuid, spec) for uuid, spec in self._devices.items()},
                    active=len(self._leases), resident_models=len(self._models), pending=len(self._pending),
                    fenced=sum(bool(r.get("fenced")) for r in self._leases.values()),
                    pending_bytes=sum(r["bytes"] for r in self._pending),
                    reserved_vram_bytes=sum(r["reserved"] for r in list(self._leases.values()) + list(self._models.values())),
                    models=[dict(model_id=model_id, pid=record["owner"][0], process_epoch=record["owner"][1],
                                 worker_epoch=record["worker_epoch"], device=record["device"],
                                 resident_bytes=record["reserved"], resident_ram_bytes=record["resident_ram_bytes"], resident_ram_source=record["resident_ram_source"], unknown=record["unknown"], observed=record["observed"])
                            for model_id, record in self._models.items()],
                    ram_bytes=self._ram_used(),
                    resident_ram_bytes=sum(self._resident_ram_by_owner().values()),
                    resident_ram_observations=[dict(pid=owner[0], process_epoch=owner[1], **record)
                                               for owner, record in self._rss_residents.items()],
                    transient_ram_bytes=sum(r["ram_bytes"] for r in self._leases.values()),
                    output_bytes=sum(r["output_bytes"] for r in self._leases.values()),
                    controllers=[dict(backend=k[0], identity=k[1], shape=k[2], phase=k[3], limit=v.limit,
                                      active=v.active, completions=v.completed, throughput=v.throughput,
                                      peak_vram_bytes=v.peak_vram_bytes,
                                      utilization_samples=v.utilization_samples,
                                      last_decision=v.last_decision,
                                      queue=sum(r["key"] == k for r in self._pending)) for k, v in self._controllers.items()])

    def close(self):
        with self._close_lock:
            self._close()

    def _close(self):
        deadline = time.monotonic() + self.shutdown_timeout
        with self._cv:
            if self._finalized:
                return
            self._closed = True
            self._cv.notify_all()
            # The monitor and completion/control RPCs stay alive during drain,
            # including after a busy timeout. Neither EOF nor low GPU utilization
            # proves that a live process released its CUDA context or work.
            while True:
                self._reap_exited()
                gpu_leases = sum(r["device"] is not None for r in self._leases.values())
                gpu_models = sum(r["device"] is not None or r["resident_ram_bytes"] > 0 for r in self._models.values())
                if not gpu_leases and not gpu_models and not self._rss_residents:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise BrokerBusy(f"resources busy: {gpu_leases} GPU leases, {gpu_models} resident models, {len(self._rss_residents)} RSS owners; UUID locks and owner monitoring retained")
                self._cv.wait(remaining)
            self._stop.set()
            if self._listener:
                try:
                    self._listener.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                self._listener.close()
            for conn in list(self._sockets):
                try:
                    conn.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                conn.close()
            for token in list(self._leases):
                self._release(token, error="broker closed")
            self._models.clear()
            threads = list(self._threads)
        for thread in threads:
            if thread is not threading.current_thread():
                thread.join(max(0, deadline - time.monotonic()))
        # A stuck sampler must keep the UUID locks, rather than allow a new owner.
        if any(thread.is_alive() for thread in threads):
            raise BrokerBusy("broker threads failed to stop; UUID locks retained")
        if self.socket_path:
            try:
                os.unlink(self.socket_path)
            except FileNotFoundError:
                pass
        for fd in self._locks:
            os.close(fd)
        self._locks.clear()
        if self.socket_path and os.environ.get("ASTROLABE_RESOURCE_SOCKET") == self.socket_path:
            if self._old_env is None:
                os.environ.pop("ASTROLABE_RESOURCE_SOCKET", None)
            else:
                os.environ["ASTROLABE_RESOURCE_SOCKET"] = self._old_env
        self._finalized = True


class ResourceClient:
    def __init__(self, socket_path):
        self.socket_path = os.fspath(socket_path)
        self.frame = 65536

    @classmethod
    def from_env(cls):
        path = os.environ.get("ASTROLABE_RESOURCE_SOCKET")
        return cls(path) if path else None

    def _connect(self):
        info = os.lstat(os.path.dirname(self.socket_path))
        endpoint = os.lstat(self.socket_path)
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700 or not stat.S_ISSOCK(endpoint.st_mode) or endpoint.st_uid != os.getuid():
            raise AdmissionError("unsafe broker socket")
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            conn.settimeout(5.0)
            conn.connect(self.socket_path)
            _, uid, _ = struct.unpack("3i", conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
            if uid != os.getuid():
                raise AdmissionError("broker UID rejected")
            return conn
        except Exception:
            conn.close()
            raise

    @staticmethod
    def _check(response):
        if not response.get("ok"):
            kind = response.get("kind")
            error_type = AdmissionTimeout if kind == "AdmissionTimeout" else BrokerClosed if kind == "BrokerClosed" else AdmissionError
            raise error_type(response.get("error", "broker error"))
        return {key: value for key, value in response.items() if key != "ok"}

    def _rpc(self, request):
        with self._connect() as conn:
            _send(conn, request, self.frame)
            return self._check(_receive(conn, self.frame))

    def snapshot(self):
        return self._rpc(dict(op="snapshot"))

    def register_model(self, model_id, device=None, resident_bytes=0, metadata=None, resident_ram_bytes=0,
                       prepared_model_id=None):
        request = dict(op="register_model", model_id=model_id, device=device,
                              resident_bytes=resident_bytes, metadata=metadata,
                              resident_ram_bytes=resident_ram_bytes,
                              visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
        if prepared_model_id is not None:
            request["prepared_model_id"] = prepared_model_id
        return self._rpc(request)

    def unregister_model(self, model_id):
        return self._rpc(dict(op="unregister_model", model_id=model_id))

    def lease(self, resource="gpu", backend="", identity="", shape="", phase="compute",
              resident_bytes=0, scratch_bytes=0, ram_bytes=0, output_bytes=0,
              concurrency_certified=False, timeout=None, device=None, visible_devices=None,
              prepared_model_id=None, capacity_only=False, reservation_token=None,
              evaluator_inference=False):
        request = dict(op="lease", resource=resource, backend=backend,
                               identity=identity, shape=shape, phase=phase,
                               resident_bytes=resident_bytes, scratch_bytes=scratch_bytes,
                               ram_bytes=ram_bytes, output_bytes=output_bytes,
                               concurrency_certified=concurrency_certified, timeout=timeout,
                               device=device, visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES") if visible_devices is None else visible_devices)
        if prepared_model_id is not None:
            request["prepared_model_id"] = prepared_model_id
        if capacity_only:
            request["capacity_only"] = True
        if reservation_token is not None:
            request["reservation_token"] = reservation_token
        if evaluator_inference:
            request["evaluator_inference"] = True
        return Lease(self, request)


class Lease:
    def __init__(self, client, request):
        self.client = client
        self.request = request
        self.token = None
        self.device = None
        self._conn = None
        self._lock = threading.Lock()
        self._cancelled = False
        self._used = False
        self._finished = False

    def __enter__(self):
        with self._lock:
            if self._used or self._cancelled:
                raise AdmissionError("lease is single-use or cancelled")
            self._used = True
            self._conn = self.client._connect()
            conn = self._conn
        try:
            _send(conn, self.request, self.client.frame)
            # Server enforces the queue deadline. Cancellation closes this socket.
            conn.settimeout(None)
            reply = self.client._check(_receive(conn, self.client.frame))
            with self._lock:
                self.token, self.device = reply["token"], reply["device"]
                if self._cancelled or self._conn is not conn:
                    raise AdmissionError("lease cancelled during admission")
            return self
        except Exception:
            self.cancel()
            raise

    def finish(self, error=None, oom=False):
        """Report synchronized device completion; also works after cancel/EOF."""
        with self._lock:
            if self._finished:
                return
            conn, self._conn = self._conn, None
            token = self.token
        request = dict(op="finish", token=token, error=None if error is None else str(error), oom=oom)
        if conn is None:
            if token is not None:
                self.client._rpc(request)
                with self._lock:
                    self._finished = True
            return
        try:
            conn.settimeout(5.0)
            _send(conn, request, self.client.frame)
            self.client._check(_receive(conn, self.client.frame))
            with self._lock:
                self._finished = True
        except (EOFError, OSError) as exc:
            raise BrokerClosed("broker disconnected while finishing lease") from exc
        finally:
            conn.close()

    def cancel(self):
        with self._lock:
            self._cancelled = True
            conn, self._conn = self._conn, None
        if conn is not None:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            conn.close()

    def __exit__(self, exc_type, exc, tb):
        if exc is not None:
            if self.request["resource"] == "gpu" and not self._finished:
                self.cancel()  # An exception is not a CUDA completion barrier.
            else:
                try:
                    self.finish(error=exc)
                except AdmissionError:
                    pass
        else:
            self.finish()
        return False
