"""Real CPU trajectory/broker process boundary; no scientific model or CUDA."""
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time

import pytest

from astrolabe.model_daemon import DaemonSpec, LocalModelOwner, model_identity
from astrolabe.runtime.admission import ResourceBroker, ResourceClient
from astrolabe.runtime.session import resource_session


ADAPTER = r'''
import os
import random
import threading
import numpy as np
from astrolabe.model_execution import kernel_handler

LOADS = 0

def make(settings):
    def load(config):
        global LOADS
        LOADS += 1
        if "caller_settings" in config:
            config["caller_settings"]["labels"].append("loader")
        random.random()
        np.random.random()
        # Touch real resident CPU storage; the broker observes actual VmRSS.
        resident = np.ones(128 * 1024, dtype=np.float64)
        buffer = np.zeros(3, dtype=np.float64)
        if config.get("torch_reuse"):
            import torch
            torch.rand(1, device="cpu")
            torch_buffer = torch.zeros(3, dtype=torch.float64, device="cpu")
        def infer(payload):
            assert resident.sum() == resident.size
            value = np.asarray(payload["sample"], dtype=np.float64) + payload["offset"]
            if config.get("reuse_output"):
                buffer[:] = value
                value = buffer
            extra = {}
            if config.get("mutate_input"):
                payload["sample"][:] = -100
                payload["nested"]["labels"].append("kernel")
                payload["nested"]["mapping"]["count"] = 9
                extra["input_was_mutated"] = bool(np.all(payload["sample"] == -100))
                extra["nested_was_mutated"] = payload["nested"] == {
                    "labels": ["caller", "kernel"], "mapping": {"count": 9}}
            if config.get("torch_reuse"):
                torch_buffer.copy_(torch.as_tensor(value, device="cpu"))
                value = torch_buffer
                extra["torch_draw"] = torch.rand(1, device="cpu").item()
            return dict(value=value, python=random.random(), numpy=np.random.random(),
                        pid=os.getpid(), loads=LOADS,
                        main=threading.current_thread() is threading.main_thread(), **extra)
        return infer, {"device": "cpu"}
    return kernel_handler(settings, load, lifecycle={"resource": "cpu"})
'''


CHILD = r'''
import json
import os
import random
import sys
import threading
from pathlib import Path
import numpy as np
from astrolabe.model_execution import NativeModelClient, capture_random_state
from astrolabe.model_daemon import LocalModelOwner, ModelDaemonError
from astrolabe.runtime.session import resource_session
from astrolabe.runtime.admission import ResourceClient

scenario = sys.argv[1]
before = dict(os.environ)
if scenario == "torch_reuse":
    import torch
    def no_cuda(*args, **kwargs):
        raise AssertionError("CUDA initialization forbidden in CPU review")
    torch.cuda.init = no_cuda
    torch.cuda._lazy_init = no_cuda
    assert not torch.cuda.is_initialized()
    torch.manual_seed(71)
    reference_torch = torch.Generator(device="cpu").manual_seed(71)
random.seed(71)
np.random.seed(71)
reference_python = random.Random(71)
reference_numpy = np.random.RandomState(71)
def draw():
    return reference_python.random(), reference_numpy.random_sample()

with resource_session({"model_owner": "trajectory"}):
    kwargs = {}
    if scenario == "python":
        kwargs["python_bin"] = "/nonexistent/alternate-python"
    elif scenario == "cwd":
        kwargs["cwd"] = str(Path.cwd().parent)
    elif scenario == "env":
        kwargs["env"] = {"TRAJECTORY_TEST_OVERRIDE": "changed"}
    elif scenario == "provider":
        kwargs["provider"] = {"provider": "native", "mode": "off"}
    elif scenario == "socket":
        kwargs["socket_path"] = "/tmp/unattached-trajectory-review.sock"
    if kwargs:
        try:
            NativeModelClient("esmfold", "trajectory_cpu_adapter:make", {}, **kwargs)
        except ValueError as error:
            assert "Trajectory model ownership" in str(error), str(error)
        else:
            raise AssertionError("runtime mismatch silently accepted: " + scenario)
        assert ResourceClient.from_env().snapshot()["resident_models"] == 0
    else:
        # No additional model subprocess may be launched in the trajectory.
        import subprocess
        def forbidden(*args, **kwargs):
            raise AssertionError("trajectory launched an extra model process")
        subprocess.Popen = forbidden
        env = {"CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"]}
        settings = {"reuse_output": scenario == "reuse",
                    "mutate_input": scenario == "mutate_input",
                    "torch_reuse": scenario == "torch_reuse",
                    "caller_settings": {"labels": ["caller"]}}
        client = NativeModelClient("cpu-review", "trajectory_cpu_adapter:make",
                                  settings,
                                  python_bin=sys.executable, cwd=str(Path.cwd()), env=env)
        assert settings["caller_settings"] == {"labels": ["caller"]}, "loader mutated caller settings"
        assert isinstance(client.owner, LocalModelOwner)
        assert client.pid == os.getpid()
        assert client.metadata["runtime_lifecycle"]["loader_count"] == 1
        draw()  # Constructor consumes exactly one draw per RNG.
        if scenario == "torch_reuse":
            torch.rand(1, generator=reference_torch, device="cpu")
        assert (random.random(), np.random.random()) == draw()
        sample = (np.array([1.0, 2.0, 3.0]) if scenario == "mutate_input"
                  else [1.0, 2.0, 3.0])
        nested = {"labels": ["caller"], "mapping": {"count": 0}}
        def check_caller_input(result):
            assert np.array_equal(sample, [1.0, 2.0, 3.0]), "kernel mutated caller array"
            assert nested == {"labels": ["caller"], "mapping": {"count": 0}}, "kernel mutated caller nested input"
            if scenario == "mutate_input":
                assert result["input_was_mutated"] and result["nested_was_mutated"]
        first = client.request({"sample": sample, "offset": 1, "nested": nested})
        check_caller_input(first)
        first_receipt = client.last_runtime_receipt
        assert (first["python"], first["numpy"]) == draw()
        if scenario == "torch_reuse":
            assert first["value"].device.type == "cpu"
            assert first["torch_draw"] == torch.rand(1, generator=reference_torch, device="cpu").item()
        saved = (first["value"].clone() if scenario == "torch_reuse"
                 else first["value"].copy())
        state_before_thread = capture_random_state()
        errors = []
        def wrong_thread(operation):
            try:
                operation()
            except ValueError as error:
                errors.append(str(error))
        for operation in (
            lambda: client.request({"sample": sample, "offset": 99}),
            client.shutdown,
            lambda: NativeModelClient("cpu-review", "trajectory_cpu_adapter:make", {}),
        ):
            thread = threading.Thread(target=wrong_thread, args=(operation,))
            thread.start()
            thread.join(timeout=5)
            assert not thread.is_alive()
        assert len(errors) == 3 and all("main thread" in error for error in errors), errors
        assert capture_random_state() == state_before_thread
        second = client.request({"sample": sample, "offset": 2, "nested": nested})
        check_caller_input(second)
        assert (second["python"], second["numpy"]) == draw()
        if scenario == "torch_reuse":
            assert second["value"].device.type == "cpu"
            assert first["value"].data_ptr() != second["value"].data_ptr()
            assert second["torch_draw"] == torch.rand(1, generator=reference_torch, device="cpu").item()
            assert not torch.cuda.is_initialized()
        assert first["pid"] == second["pid"] == os.getpid()
        assert first["loads"] == second["loads"] == 1
        assert first["main"] and second["main"]
        assert np.array_equal(second["value"], [3, 4, 5])
        assert np.array_equal(first["value"], saved), "later request mutated retained output"
        assert not np.shares_memory(first["value"], second["value"])
        second["value"][0] = -1
        assert np.array_equal(first["value"], saved)
        assert np.array_equal(sample, [1.0, 2.0, 3.0])
        receipt = client.last_runtime_receipt
        assert receipt["lifecycle"]["requests"] == 2
        assert receipt["request_id"] != first_receipt["request_id"]
        assert receipt["model_id"] == first_receipt["model_id"]
        model_id = receipt["model_id"]
        live = ResourceClient.from_env().snapshot()
        client.shutdown()
        try:
            client.request({"sample": sample, "offset": 3})
        except ModelDaemonError as error:
            assert "closed" in str(error)
        else:
            raise AssertionError("closed owner accepted a request")
        closed = ResourceClient.from_env().snapshot()
        assert live["resident_models"] == closed["resident_models"] == 1
        assert live["models"][0]["model_id"] == closed["models"][0]["model_id"] == model_id
        assert closed["models"][0]["pid"] == os.getpid()
        assert closed["models"][0]["resident_ram_source"] == "proc_status:VmRSS"
        assert closed["resident_ram_bytes"] > 0
        assert closed["active"] == 0
        print("TRAJECTORY_REVIEW:" + json.dumps({"pid": os.getpid(), "live": live,
                                                  "closed": closed}), flush=True)
        assert sys.stdin.readline().strip() == "exit"
assert dict(os.environ) == before, "resource_session did not restore environment"
'''


@pytest.fixture
def cpu_broker(tmp_path, monkeypatch):
    for name in ("ASTROLABE_RESOURCE_SOCKET", "ASTROLABE_MODEL_OWNER",
                 "ASTROLABE_MODEL_PROVIDERS", "ASTROLABE_RESOURCE_PROFILES"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    broker = ResourceBroker({
        "devices": [], "cpu_slots": 1, "ram_bytes": 1024**3,
        "observe_resident_ram": True, "sample_interval": 0.02,
        "sample_max_age": 2, "shutdown_timeout": 1,
        "socket_dir": str(tmp_path / "broker"),
    }).start()
    try:
        yield broker, ResourceClient(broker.socket_path)
    finally:
        broker.close()


def _child_env(tmp_path):
    (tmp_path / "trajectory_cpu_adapter.py").write_text(ADAPTER)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(tmp_path), str(Path(__file__).resolve().parents[1]),
         environment.get("PYTHONPATH", "")))
    return environment


def _read_evidence(child, timeout=30):
    deadline = time.monotonic() + timeout
    lines = []
    while time.monotonic() < deadline:
        readable, _, _ = select.select([child.stdout], [], [], max(0, deadline - time.monotonic()))
        assert readable, "trajectory evidence handshake timed out"
        line = child.stdout.readline()
        if not line:
            raise AssertionError("trajectory exited before evidence: " + child.stderr.read() + "".join(lines))
        if line.startswith("TRAJECTORY_REVIEW:"):
            return json.loads(line.partition(":")[2])
        lines.append(line)
    raise AssertionError("trajectory evidence handshake timed out")


@pytest.mark.parametrize("scenario", ["normal", "reuse", "mutate_input", "torch_reuse"])
def test_cross_process_cpu_owner_rng_outputs_and_exit(cpu_broker, tmp_path, scenario):
    _, resource = cpu_broker
    child = subprocess.Popen([sys.executable, "-u", "-c", CHILD, scenario],
                             env=_child_env(tmp_path), stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        evidence = _read_evidence(child)
        assert evidence["pid"] == child.pid != os.getpid()
        assert child.poll() is None
        charged = resource.snapshot()
        assert charged["healthy"] and charged["devices"] == {}
        assert charged["resident_models"] == 1
        assert charged["models"][0]["pid"] == child.pid
        assert charged["resident_ram_bytes"] > 0
        assert charged["resident_ram_observations"][0]["pid"] == child.pid
        stdout, stderr = child.communicate("exit\n", timeout=10)
        assert child.returncode == 0, stdout + stderr
        deadline = time.monotonic() + 5
        while True:
            released = resource.snapshot()
            if released["resident_models"] == 0 and released["resident_ram_bytes"] == 0:
                break
            assert time.monotonic() < deadline, released
            time.sleep(0.02)
        assert released["healthy"] and released["active"] == 0
        assert released["ram_bytes"] == 0
        assert released["resident_ram_observations"] == []
        print(json.dumps({"scenario": scenario, "broker_pid": os.getpid(),
                          "trajectory_pid": child.pid,
                          "resident_after_load": evidence["live"]["resident_ram_bytes"],
                          "resident_after_logical_shutdown": evidence["closed"]["resident_ram_bytes"],
                          "resident_after_exit": released["resident_ram_bytes"],
                          "models_after_exit": released["resident_models"]}))
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=10)
        deadline = time.monotonic() + 5
        while resource.snapshot()["resident_models"] and time.monotonic() < deadline:
            time.sleep(0.02)
        for stream in (child.stdin, child.stdout, child.stderr):
            stream.close()


@pytest.mark.parametrize("scenario", ["python", "cwd", "env", "provider", "socket"])
def test_trajectory_rejects_unhonored_runtime_options(cpu_broker, tmp_path, scenario):
    result = subprocess.run([sys.executable, "-c", CHILD, scenario],
                            env=_child_env(tmp_path), capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_same_pid_broker_and_missing_broker_rejected(cpu_broker, monkeypatch):
    spec = DaemonSpec("cpu-review", "unused:make", {})
    with pytest.raises(ValueError, match="different process"):
        LocalModelOwner(spec)
    # Restore the live endpoint before broker.close(); a later monkeypatch
    # teardown must not resurrect the already-closed socket for other modules.
    with monkeypatch.context() as environment:
        environment.delenv("ASTROLABE_RESOURCE_SOCKET")
        with pytest.raises(ValueError, match="supervising resource broker"):
            LocalModelOwner(spec)


def test_exact_identity_includes_owner_mode(monkeypatch):
    spec = DaemonSpec("cpu-review", "unused:make", {"sample": [1, 2, 3]})
    monkeypatch.delenv("ASTROLABE_MODEL_OWNER", raising=False)
    default = model_identity(spec)
    monkeypatch.setenv("ASTROLABE_MODEL_OWNER", "daemon")
    assert model_identity(spec) == default
    monkeypatch.setenv("ASTROLABE_MODEL_OWNER", "trajectory")
    assert model_identity(spec) != default


@pytest.mark.parametrize("previous", [None, "daemon", "trajectory"])
@pytest.mark.parametrize("error", [False, True])
def test_resource_session_restores_owner_environment(monkeypatch, previous, error):
    monkeypatch.delenv("ASTROLABE_RESOURCE_SOCKET", raising=False)
    if previous is None:
        monkeypatch.delenv("ASTROLABE_MODEL_OWNER", raising=False)
    else:
        monkeypatch.setenv("ASTROLABE_MODEL_OWNER", previous)
    before = dict(os.environ)
    class ExpectedError(Exception):
        pass
    try:
        with resource_session({"model_owner": "trajectory"}):
            assert os.environ["ASTROLABE_MODEL_OWNER"] == "trajectory"
            if error:
                raise ExpectedError
    except ExpectedError:
        assert error
    assert dict(os.environ) == before


def test_default_owner_remains_actual_daemon(tmp_path, monkeypatch):
    for name in ("ASTROLABE_RESOURCE_SOCKET", "ASTROLABE_MODEL_OWNER",
                 "ASTROLABE_MODEL_PROVIDERS", "ASTROLABE_RESOURCE_PROFILES"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    # The same actual CPU factory is used through the unchanged default route.
    environment = _child_env(tmp_path)
    code = r'''
import os
from astrolabe.model_execution import NativeModelClient
from astrolabe.model_daemon import OwnedModelDaemon
from astrolabe.runtime.session import resource_session
with resource_session():
    client = NativeModelClient("cpu-review", "trajectory_cpu_adapter:make", {"reuse_output": True})
    try:
        assert isinstance(client.owner, OwnedModelDaemon)
        result = client.request({"sample": [1, 2, 3], "offset": 1})
        assert result["pid"] == client.pid != os.getpid()
        saved = result["value"].copy()
        second = client.request({"sample": [1, 2, 3], "offset": 2})
        import numpy as np
        assert np.array_equal(result["value"], saved)
        assert np.array_equal(second["value"], [3, 4, 5])
        assert not np.shares_memory(result["value"], second["value"])
    finally:
        client.shutdown()
assert "ASTROLABE_MODEL_OWNER" not in os.environ
'''
    result = subprocess.run([sys.executable, "-c", code], env=environment,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
