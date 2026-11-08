"""Session ownership around the existing scientific engine.

Independent trajectories remain independent processes. This layer never
creates candidates, picks a winner, changes seeds, or commits scientific state.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Mapping, Sequence

from .admission import ResourceBroker, ResourceClient


def _visibility_tokens(visible):
    if not visible.strip():
        return []
    tokens = [token.strip() for token in visible.split(",")]
    if any(not token for token in tokens):
        raise ValueError("Malformed CUDA allocation mask")
    return tokens


def _resolved_devices(devices, inventory):
    """Resolve only declared devices using already available physical metadata."""
    from .telemetry import nonnegative

    if not isinstance(devices, (list, tuple)):
        raise ValueError("execution.devices must be allocated or a device list")
    resolved = []
    for item in devices:
        spec = (dict(item) if isinstance(item, Mapping) else
                {"index": int(item)} if str(item).isdigit() else {"uuid": item})
        if "index" in spec:
            nonnegative(spec["index"], "physical index")
        if (not {"index", "uuid"} & spec.keys() or
                ("uuid" in spec and (not isinstance(spec["uuid"], str)
                                     or not spec["uuid"].startswith("GPU-")))):
            raise ValueError("Unsupported allocated GPU identity")
        matches = [row for row in inventory.values()
                   if ("index" not in spec or row["index"] == spec["index"])
                   and ("uuid" not in spec or row["uuid"].startswith(spec["uuid"]))]
        if len(matches) != 1:
            raise ValueError("Allocated GPU is absent or ambiguous")
        row = matches[0]
        if any(other["uuid"] == row["uuid"] for other in resolved):
            raise ValueError("Duplicate allocated GPU")
        spec.update(index=row["index"], uuid=row["uuid"])
        resolved.append(spec)
    return resolved


def _visibility_order(visible, inventory):
    return [spec["uuid"] for spec in
            _resolved_devices(_visibility_tokens(visible), inventory)]


def _check_visibility_change(visible):
    """Inspect loaded framework state; never import or initialize a backend."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") == visible:
        return
    torch = sys.modules.get("torch")
    cuda = sys.modules.get("torch.cuda")
    if torch is not None:
        cuda = vars(torch).get("cuda", cuda)
        if cuda is None:
            raise RuntimeError("Cannot establish loaded Torch CUDA initialization state")
    if cuda is not None:
        initialized = vars(cuda).get("is_initialized")
        if not callable(initialized):
            raise RuntimeError("Cannot establish loaded Torch CUDA initialization state")
        if initialized():
            raise RuntimeError("Cannot change CUDA visibility after Torch CUDA initialization")
    bridges = [module for name in ("jax._src.xla_bridge", "jax.lib.xla_bridge")
               if (module := sys.modules.get(name)) is not None]
    if "jax" in sys.modules and not bridges:
        raise RuntimeError("Cannot establish loaded JAX backend initialization state")
    for bridge in bridges:
        backends = vars(bridge).get("_backends")
        if not isinstance(backends, dict):
            raise RuntimeError("Cannot establish loaded JAX backend initialization state")
        # Conservatively refuse a rewrite once JAX has cached any backend.
        # Reading this cache does not call get_backend()/devices().
        if backends or vars(bridge).get("_default_backend") is not None:
            raise RuntimeError("Cannot change CUDA visibility after JAX backend initialization")


@contextmanager
def _bound_visibility(devices):
    visible = ",".join(spec["uuid"] for spec in devices)
    _check_visibility_change(visible)
    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous


def _inherited_devices(snapshot, config):
    """The owner's full ordered allocation defines every client's cuda:N."""
    if not snapshot.get("healthy") or snapshot.get("closed"):
        raise ValueError("Inherited broker allocation is unavailable")
    inventory = {uuid: dict(index=spec["index"], uuid=uuid)
                 for uuid, spec in snapshot["devices"].items()}
    devices = _resolved_devices(list(inventory), inventory)
    expected = list(inventory)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and _visibility_order(visible, inventory) != expected:
        raise ValueError("CUDA visibility must match the full ordered broker allocation")
    declared = config.get("devices", "allocated")
    if declared != "allocated":
        if [spec["uuid"] for spec in _resolved_devices(declared, inventory)] != expected:
            raise ValueError("execution.devices must match the full ordered broker allocation")
    return devices


def _host_ram_capacity():
    """Available host/cgroup/job capacity, without assuming exclusive host RAM."""
    available = next(int(line.split()[1]) * 1024
                     for line in Path("/proc/meminfo").read_text().splitlines()
                     if line.startswith("MemAvailable:"))
    bounds = [available]
    # cgroup v2 may constrain this job below the host's available memory.
    for record in Path("/proc/self/cgroup").read_text().splitlines():
        hierarchy, controllers, relative = record.split(":", 2)
        if hierarchy == "0" and not controllers:
            directory = Path("/sys/fs/cgroup") / relative.lstrip("/")
            while directory.is_relative_to("/sys/fs/cgroup"):
                try:
                    maximum = (directory / "memory.max").read_text().strip()
                    used = int((directory / "memory.current").read_text())
                    if maximum != "max":
                        bounds.append(max(0, int(maximum) - used))
                except FileNotFoundError:
                    pass
                if directory == Path("/sys/fs/cgroup"):
                    break
                directory = directory.parent
    if os.environ.get("SLURM_MEM_PER_NODE"):
        bounds.append(int(os.environ["SLURM_MEM_PER_NODE"]) * 1024**2)
    elif os.environ.get("SLURM_MEM_PER_CPU"):
        bounds.append(int(os.environ["SLURM_MEM_PER_CPU"]) * 1024**2
                      * len(os.sched_getaffinity(0)))
    return min(bounds)


def _allocation_config(config):
    """Resolve physical capacity once; ongoing admission still uses live samples."""
    from . import admission
    from .telemetry import validate_sample

    config = dict(config)
    devices = config.get("devices", "allocated")
    if devices == "allocated":
        visible = next(
            (os.environ[key] for key in (
                "CUDA_VISIBLE_DEVICES", "SLURM_STEP_GPUS", "SLURM_JOB_GPUS")
             if key in os.environ), None,
        )
        if visible is None:
            raise ValueError(
                "Adaptive execution needs an explicit device allocation; set "
                "CUDA_VISIBLE_DEVICES or execution.devices"
            )
        devices = _visibility_tokens(visible)
    default_vram = config.pop("vram_bytes", None)
    if not isinstance(devices, (list, tuple)):
        raise ValueError("execution.devices must be allocated or a device list")
    inventory = {}
    if devices:
        inventory = validate_sample(admission.NvidiaSMISampler(
            config.get("sample_timeout", 2.0))())
    devices = _resolved_devices(devices, inventory)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        order = _visibility_order(visible, inventory)
        by_uuid = {spec["uuid"]: spec for spec in devices}
        if set(order) != set(by_uuid):
            raise ValueError("execution.devices must match the full CUDA allocation")
        devices = [by_uuid[uuid] for uuid in order]
    if devices:
        for spec in devices:
            row = inventory[spec["uuid"]]
            spec.setdefault("vram_bytes", default_vram if default_vram is not None
                            else row["free_bytes"])
    config["devices"] = devices
    capacity = _host_ram_capacity()
    if config.get("ram_bytes", capacity) > capacity:
        raise ValueError("execution.ram_bytes exceeds available host/job memory")
    config.setdefault("ram_bytes", capacity)
    config.setdefault("observe_resident_ram", True)
    if config.get("cpu_slots") == "auto":
        config["cpu_slots"] = min(2, len(os.sched_getaffinity(0)))
    if config.get("cpu_slots", 1) > len(os.sched_getaffinity(0)):
        raise ValueError("execution.cpu_slots exceeds process CPU allocation")
    return config


@dataclass(frozen=True)
class IndependentCommand:
    identity: str
    argv: tuple[str, ...]
    output_dir: Path
    cwd: Path | None = None
    env: Mapping[str, str] | None = None


@contextmanager
def resource_session(config=None):
    config = dict(config or {})
    owner = config.pop("model_owner", os.environ.get("ASTROLABE_MODEL_OWNER", "daemon"))
    if owner not in {"daemon", "trajectory"}:
        raise ValueError("execution.model_owner must be daemon or trajectory")
    previous = os.environ.get("ASTROLABE_MODEL_OWNER")
    os.environ["ASTROLABE_MODEL_OWNER"] = owner
    try:
        with _resource_session_owned(config) as client:
            yield client
    finally:
        if previous is None:
            os.environ.pop("ASTROLABE_MODEL_OWNER", None)
        else:
            os.environ["ASTROLABE_MODEL_OWNER"] = previous


@contextmanager
def _resource_session_owned(config=None):
    """Use one inherited broker, or own and drain an explicitly enabled one."""
    config = dict(config or {})
    mode = config.pop("mode", "legacy")
    if mode not in {"legacy", "resident_serial", "adaptive"}:
        raise ValueError("execution.mode must be legacy, resident_serial or adaptive")
    inherited = ResourceClient.from_env()
    if inherited is not None:
        # Resolve the owner before any expensive scientific constructor runs.
        devices = _inherited_devices(inherited.snapshot(), config)
        with _bound_visibility(devices):
            yield inherited
        return
    if mode == "legacy":
        yield None
        return
    if mode == "resident_serial":
        config["max_concurrency"] = 1
    config = _allocation_config(config)
    with _bound_visibility(config["devices"]):
        previous = os.environ.get("ASTROLABE_RESOURCE_SOCKET")
        broker = ResourceBroker(config)
        broker.start()
        os.environ["ASTROLABE_RESOURCE_SOCKET"] = str(broker.socket_path)
        try:
            yield ResourceClient(str(broker.socket_path))
        finally:
            try:
                broker.close()
            finally:
                if previous is None:
                    os.environ.pop("ASTROLABE_RESOURCE_SOCKET", None)
                else:
                    os.environ["ASTROLABE_RESOURCE_SOCKET"] = previous


def _live_group_members(pgid):
    """Linux process-group members still capable of executing, excluding zombies."""
    members = []
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_bytes().rsplit(b")", 1)[1].split()
            if int(fields[2]) == pgid and fields[0] not in {b"Z", b"X"}:
                members.append(int(path.parent.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return members


def _terminate_group(child):
    """Terminate the group we created, even when its leader exits first."""
    def send(sig):
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            pass

    send(signal.SIGTERM)
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    # wait() only proves leader exit. A same-group tool may ignore SIGTERM.
    send(signal.SIGKILL)
    child.wait(timeout=10)
    deadline = time.monotonic() + 10
    while members := _live_group_members(child.pid):
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Owned process group {child.pid} has live members: {members}")
        time.sleep(0.02)
    return child.returncode


def run_independent(commands: Sequence[IndependentCommand], *, max_trajectories: int,
                    timeout: float | None = None):
    """Run the given fixed inventory, preserving its input/result ordering.

    Threads supervise child processes only; the legacy scientific loop and its
    global RNG are never run in these threads. Each command owns a unique
    namespace. No command is replayed after timeout or nonzero exit.
    """
    commands = tuple(commands)
    if max_trajectories < 1:
        raise ValueError("max_trajectories must be positive")
    identities = [item.identity for item in commands]
    roots = [item.output_dir.resolve() for item in commands]
    if len(set(identities)) != len(identities) or len(set(roots)) != len(roots):
        raise ValueError("Independent commands require unique identities and output directories")
    for root in roots:
        if any(root != other and root in other.parents for other in roots):
            raise ValueError("Independent output namespaces cannot overlap")
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"Refusing to overwrite command output: {root}")
    # Freeze and validate every child's allocation before starting any process.
    parent_environment = os.environ.copy()
    inherited = ResourceClient.from_env()
    devices = (_inherited_devices(inherited.snapshot(), {})
               if inherited is not None else None)
    environments = {}
    for item in commands:
        environment = dict(parent_environment)
        environment.update(item.env or {})
        if parent_environment.get("ASTROLABE_RESOURCE_SOCKET") is not None:
            environment["ASTROLABE_RESOURCE_SOCKET"] = parent_environment["ASTROLABE_RESOURCE_SOCKET"]
        if devices is not None:
            inventory = {spec["uuid"]: spec for spec in devices}
            if item.env and "CUDA_VISIBLE_DEVICES" in item.env:
                if _visibility_order(item.env["CUDA_VISIBLE_DEVICES"], inventory) != list(inventory):
                    raise ValueError("Child CUDA visibility must match the full ordered broker allocation")
            for key in ("ASTROLABE_RESOURCE_SOCKET", "SLURM_STEP_GPUS", "SLURM_JOB_GPUS",
                        "CUDA_DEVICE_ORDER", "ASTROLABE_ALLOCATED_GPU_UUID"):
                if key in parent_environment:
                    environment[key] = parent_environment[key]
                else:
                    environment.pop(key, None)
            environment["CUDA_VISIBLE_DEVICES"] = ",".join(inventory)
        environments[item.identity] = environment
    children = set()
    lock = threading.Lock()
    stop = threading.Event()

    def run(item):
        if stop.is_set():
            return {"identity": item.identity, "state": "cancelled_before_start"}
        item.output_dir.mkdir(parents=True, exist_ok=True)
        environment = environments[item.identity]
        started = time.perf_counter()
        with (item.output_dir / "stdout.log").open("wb") as stdout, \
                (item.output_dir / "stderr.log").open("wb") as stderr:
            with lock:
                if stop.is_set():
                    return {"identity": item.identity, "state": "cancelled_before_start"}
                child = subprocess.Popen(item.argv, cwd=item.cwd, env=environment,
                                         stdout=stdout, stderr=stderr, start_new_session=True)
                children.add(child)
            try:
                try:
                    returncode = child.wait(timeout=timeout)
                    state = ("outcome_unknown" if stop.is_set() else
                             "completed" if returncode == 0 else "failed")
                except subprocess.TimeoutExpired:
                    state = "outcome_unknown"
                    returncode = _terminate_group(child)
            finally:
                with lock:
                    children.discard(child)
        result = {"identity": item.identity, "state": state,
                  "returncode": returncode, "elapsed_seconds": time.perf_counter() - started,
                  "output_dir": str(item.output_dir)}
        (item.output_dir / "command_result.json").write_text(json.dumps(result, indent=2) + "\n")
        return result

    with ThreadPoolExecutor(max_workers=max_trajectories,
                            thread_name_prefix="trajectory-supervision") as pool:
        futures = {pool.submit(run, item): index for index, item in enumerate(commands)}
        results = [None] * len(commands)
        try:
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        except BaseException as error:
            for future in futures:
                future.cancel()
            with lock:
                stop.set()
                owned = tuple(children)
            for child in owned:
                try:
                    _terminate_group(child)
                except BaseException as cleanup_error:
                    error.add_note(f"Process-group cleanup failed for {child.pid}: {cleanup_error}")
            raise
    return results
