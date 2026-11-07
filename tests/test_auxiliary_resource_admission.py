"""CPU acceptance of real wrapper boundaries; all scientific vendors are fake.

The IPC tests use the real broker with a synthetic GPU inventory, never CUDA.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import json
import io
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid

import pytest

import astrolabe.model_execution as execution
from astrolabe.initializers.wrappers import rfd3_worker
from astrolabe.structure_processors.wrappers import placer_daemon, rapidock_daemon


class RecordingClient:
    socket_path = "cpu-test-only"

    def __init__(self):
        self.active = []
        self.events = []
        self.models = []

    @contextmanager
    def lease(self, **kwargs):
        assert not self.active, "a local helper reentered the broker"
        self.active.append(kwargs)
        self.events.append(("enter", kwargs["resource"], kwargs["phase"]))
        try:
            yield SimpleNamespace(device="GPU-synthetic")
        finally:
            self.events.append(("exit", kwargs["resource"], kwargs["phase"]))
            self.active.pop()

    def register_model(self, model_id, **kwargs):
        assert self.active[-1]["phase"] == "load"
        self.models.append((model_id, kwargs))

    def check(self, resource, phase):
        assert len(self.active) == 1
        assert self.active[0]["resource"] == resource
        assert self.active[0]["phase"] == phase


@pytest.fixture
def recorder(monkeypatch):
    client = RecordingClient()
    monkeypatch.setattr(execution, "_resource_client", lambda: client)
    monkeypatch.delenv("ASTROLABE_RESOURCE_SOCKET", raising=False)
    return client


@pytest.mark.parametrize("failure", [None, "load", "execute", "finalize"])
@pytest.mark.parametrize("mode", ["daemon", "one_shot"])
def test_rfd3_worker_admits_lazy_load_run_and_cpu_output(
    tmp_path, monkeypatch, recorder, failure, mode,
):
    seen = []

    class Engine:
        def run(self, **kwargs):
            recorder.check("gpu", "execute")
            assert kwargs == {"inputs": "input.json", "n_batches": 3, "out_dir": None}
            seen.append("run")
            if failure == "execute":
                raise ValueError("synthetic inference error")
            return {"example": [SimpleNamespace(example_id="example", atom_array=[], metadata={"seed": 42})]}

    def load(repo, config):
        recorder.check("gpu", "load")
        assert config == {"seed": 42}
        seen.append("load")
        if failure == "load":
            raise ValueError("synthetic load error")
        return Engine()

    class PDBFile:
        def set_structure(self, array):
            recorder.check("cpu", "finalize")

        def write(self, path):
            recorder.check("cpu", "finalize")
            if failure == "finalize":
                raise OSError("synthetic file error")
            Path(path).write_text("ATOM\n")

    monkeypatch.setitem(sys.modules, "biotite.structure.io.pdb", SimpleNamespace(PDBFile=PDBFile))
    monkeypatch.setattr(rfd3_worker, "_load_engine", load)
    monkeypatch.setattr(rfd3_worker, "_capture_rng_state", lambda: {"snapshot": 42})

    def restore(snapshot):
        recorder.check("gpu", "execute")
        assert snapshot == {"snapshot": 42}
        seen.append("restore")

    monkeypatch.setattr(rfd3_worker, "_restore_rng_state", restore)
    request = {"repo_path": str(tmp_path), "config": {"seed": 42},
               "input_path": "input.json", "n_batches": 3,
               "output_dir": str(tmp_path), "response_path": str(tmp_path / "response.json")}

    def run():
        if mode == "one_shot":
            return rfd3_worker.run_once(request)
        return rfd3_worker.daemon_factory({"repo_path": str(tmp_path), "engine_config": {"seed": 42}})(request)

    if failure:
        with pytest.raises((RuntimeError, ValueError), match="synthetic"):
            run()
    else:
        response = run()
        assert json.loads(Path(request["response_path"]).read_text()) == response
        assert Path(response["outputs"][0]["structure_path"]).read_text() == "ATOM\n"
        assert seen == ["load", "restore", "run"]
    assert not recorder.active
    if failure != "load":
        assert len(recorder.models) == 1
        model_id, registration = recorder.models[0]
        assert f":{os.getpid()}:" in model_id
        assert registration["metadata"]["residency"] == ("one_shot" if mode == "one_shot" else "resident_bridge")
    else:
        assert not recorder.models


def test_placer_splits_input_and_output_and_releases_on_error(tmp_path, monkeypatch, recorder):
    inputs = []
    failure = [False]
    origin = tmp_path / "PLACER.py"
    origin.write_text("# fake vendor\n")

    class Input:
        def __init__(self):
            recorder.check("cpu", "prepare")
            inputs.append(self)

        def pdb(self, value):
            recorder.check("cpu", "prepare")

        def name(self, value):
            recorder.check("cpu", "prepare")

    class Model:
        def __init__(self):
            recorder.check("gpu", "load")

        def run(self, input_object, nsamples):
            recorder.check("gpu", "execute")
            assert nsamples == 3
            if failure[0]:
                raise ValueError("synthetic PLACER failure")
            return {2: {"prmsd": .3, "model": "model-two"}, 0: {"prmsd": .1, "model": "model-zero"}}

    monkeypatch.setitem(sys.modules, "PLACER", SimpleNamespace(PLACER=Model, PLACERinput=Input, __file__=str(origin)))
    client = placer_daemon.PlacerClient({}, mode="in_process")
    lifecycle = client.metadata["runtime_lifecycle"]
    assert lifecycle["residency"] == "resident_model"
    assert lifecycle["weights_loaded"] is True
    assert lifecycle["gpu_resident"] is None
    assert lifecycle["split_stages"] is True
    path = tmp_path / "input.pdb"
    path.write_text("ATOM\n")
    payload = {"input_file": str(path), "num_samples": 3}
    assert [item["key"] for item in client.predict(payload)] == [2, 0]
    failure[0] = True
    with pytest.raises(RuntimeError, match="synthetic PLACER failure"):
        client.predict(payload)
    assert len(inputs) == 2 and inputs[0] is not inputs[1]
    assert not recorder.active
    client.shutdown()
    # In-process logical close does not establish allocator/PID death.
    assert len(recorder.models) == 1


def test_rfd3_adapter_preserves_original_post_constructor_cpu_rng(tmp_path, monkeypatch, recorder):
    import random
    import numpy as np
    import torch

    seed = 42
    reference_python = random.Random(seed)
    reference_numpy = np.random.RandomState(seed)
    reference_torch = torch.Generator().manual_seed(seed)

    def reference_draw():
        return [reference_python.random(), float(reference_numpy.random()),
                float(torch.rand((), generator=reference_torch))]

    reference_draw()  # the vendor constructor consumes one draw
    expected = reference_draw()
    loads = []

    class Engine:
        def __init__(self):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.random(), np.random.random(), torch.rand(())
            loads.append(True)

        def run(self, **kwargs):
            recorder.check("gpu", "execute")
            draws = [random.random(), float(np.random.random()), float(torch.rand(()))]
            return {"x": [SimpleNamespace(example_id="x", atom_array=[], metadata={"rng": draws})]}

    class PDBFile:
        def set_structure(self, value):
            pass

        def write(self, path):
            Path(path).write_text("ATOM\n")

    monkeypatch.setattr(rfd3_worker, "_load_engine", lambda *args: Engine())
    monkeypatch.setitem(sys.modules, "biotite.structure.io.pdb", SimpleNamespace(PDBFile=PDBFile))
    infer = rfd3_worker.daemon_factory({"repo_path": str(tmp_path), "engine_config": {"seed": seed}})
    for index in range(2):
        # Interleaved activity must not replace the existing worker snapshot.
        random.random(), np.random.random(), torch.rand(())
        request = {"repo_path": str(tmp_path), "config": {"seed": seed},
                   "input_path": "input.json", "n_batches": 3,
                   "output_dir": str(tmp_path / str(index)),
                   "response_path": str(tmp_path / f"response{index}.json")}
        output = infer(request)["outputs"][0]
        assert json.loads(Path(output["metadata_path"]).read_text())["rng"] == expected
    assert len(loads) == 1


def test_rapidock_failed_initialization_never_registers_a_resident(recorder):
    class Runtime:
        def load(self):
            recorder.check("gpu", "load")
            raise ValueError("synthetic constructor error")

    with pytest.raises(ValueError, match="constructor error"):
        rapidock_daemon._create_runtime_handler(Runtime(), "cuda:0")
    assert not recorder.active and not recorder.models


def test_rapidock_jsonl_does_not_report_ready_after_partial_load(tmp_path, monkeypatch, recorder):
    checkpoint = tmp_path / "fake.pt"
    checkpoint.write_text("fake")

    class Runtime:
        def load(self):
            recorder.check("gpu", "load")
            raise RuntimeError("partial initialization failed")

        def status(self):
            return {"model_ready": True}

    monkeypatch.setattr(rapidock_daemon, "_parse_args", lambda: SimpleNamespace(
        repo_path=str(tmp_path), model_dir=str(tmp_path), ckpt="fake.pt",
        confidence_model_dir=None, confidence_ckpt="fake", device=None))
    monkeypatch.setattr(rapidock_daemon, "build_rapidock_runtime", lambda **_: Runtime())
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"type":"ping","id":"p"}\n{"type":"shutdown"}\n'))
    output = io.StringIO()
    monkeypatch.setattr(rapidock_daemon, "_PROTOCOL_STDOUT", output)
    assert rapidock_daemon.main() == 0
    ping = json.loads(output.getvalue().splitlines()[0])
    assert ping["model_ready"] is False
    assert "partial initialization failed" in ping["message"]
    assert not recorder.models and not recorder.active


def test_placer_rejects_cpu_admission_for_upstream_cuda_load(tmp_path, monkeypatch, recorder):
    import torch
    origin = tmp_path / "PLACER.py"
    origin.write_text("# fake vendor\n")

    class Model:
        def __init__(self):
            pytest.fail("an incompatible device request must fail before construction")

        def run(self, input_object, nsamples):
            pass

    class Input:
        pass

    monkeypatch.setitem(sys.modules, "PLACER", SimpleNamespace(PLACER=Model, PLACERinput=Input, __file__=str(origin)))
    monkeypatch.setenv("ASTROLABE_RESOURCE_SOCKET", "synthetic")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(ValueError, match="PLACER selects cuda:0; requested cpu"):
        placer_daemon.create_handler({"torch_device": "cpu"})
    assert not recorder.active and not recorder.models


@pytest.mark.parametrize("configured", ["cuda:0", "cuda:1", "cpu", "auto"])
def test_placer_configured_device_controls_load_execute_and_resident(
    monkeypatch, recorder, configured,
):
    expected_resource = "cpu" if configured == "cpu" else "gpu"
    expected_hint = configured if configured.startswith("cuda:") else None
    reported_device = "cuda:0" if configured == "auto" else configured
    leases = []

    def check(phase):
        recorder.check(expected_resource, phase)
        leases.append(dict(recorder.active[0]))

    def load(settings):
        assert settings["torch_device"] == configured
        check("load")
        return execution.KernelStages(
            lambda value: value,
            lambda value: (check("execute"), value)[1],
            resource=expected_resource,
        ), {"device": reported_device}

    # Simulate logical devices without importing or initializing CUDA. The real
    # kernel and ResourceClient boundary still choose the lease/registration.
    monkeypatch.setattr(execution, "_kernel_random_scope", lambda *args: nullcontext())
    monkeypatch.setattr(placer_daemon, "_load_placer", load)
    handle = placer_daemon.create_handler({"torch_device": configured})
    response = handle({"operation": "infer", "payload": {"samples": 3}, "random_state": {}})
    assert response["ok"] and response["result"] == {"samples": 3}
    assert [lease["phase"] for lease in leases] == ["load", "execute"]
    assert [lease["device"] for lease in leases] == [expected_hint, expected_hint]
    assert recorder.models[0][1]["device"] == reported_device
    assert not recorder.active


def test_placer_cpu_vendor_never_requests_gpu_execute(tmp_path, monkeypatch, recorder):
    import torch
    origin = tmp_path / "PLACER.py"
    origin.write_text("# CPU synthetic vendor\n")

    class Input:
        def __init__(self):
            recorder.check("cpu", "prepare")

        def pdb(self, value):
            pass

        def name(self, value):
            pass

    class Model:
        def __init__(self):
            recorder.check("cpu", "load")

        def device(self):
            return "cpu"

        def run(self, input_object, nsamples):
            recorder.check("cpu", "execute")
            assert nsamples == 3
            return {0: {"prmsd": .1, "model": "cpu-model"}}

    monkeypatch.setenv("ASTROLABE_RESOURCE_SOCKET", "synthetic")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setitem(sys.modules, "PLACER", SimpleNamespace(PLACER=Model, PLACERinput=Input, __file__=str(origin)))
    path = tmp_path / "input.pdb"
    path.write_text("ATOM\n")
    client = placer_daemon.PlacerClient({"torch_device": "cpu"}, mode="in_process")
    assert client.predict({"input_file": str(path), "num_samples": 3}) == [
        {"key": 0, "prmsd": .1, "model": "cpu-model"}]
    assert recorder.models[0][1]["device"] == "cpu"
    assert all(resource == "cpu" for kind, resource, phase in recorder.events)
    assert not recorder.active


def test_placer_nonzero_device_is_not_silently_loaded_on_zero(tmp_path, monkeypatch, recorder):
    import torch
    origin = tmp_path / "PLACER.py"
    origin.write_text("# fixed cuda:0 upstream\n")

    class Model:
        def __init__(self):
            pytest.fail("fixed-device upstream must not load on the wrong GPU")

        def run(self, input_object, nsamples):
            pass

    class Input:
        pass

    monkeypatch.setenv("ASTROLABE_RESOURCE_SOCKET", "synthetic")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(execution, "_kernel_random_scope", lambda *args: nullcontext())
    monkeypatch.setitem(sys.modules, "PLACER", SimpleNamespace(PLACER=Model, PLACERinput=Input, __file__=str(origin)))
    lease_devices = []

    def check_available():
        recorder.check("gpu", "load")
        lease_devices.append(recorder.active[0]["device"])
        return True

    monkeypatch.setattr(torch.cuda, "is_available", check_available)
    with pytest.raises(ValueError, match="PLACER selects cuda:0; requested cuda:1"):
        placer_daemon.create_handler({"torch_device": "cuda:1"})
    assert lease_devices == ["cuda:1"]
    assert not recorder.models and not recorder.active


def test_rfd3_distributed_workers_fail_before_unadmitted_ranks(monkeypatch, recorder):
    monkeypatch.setenv("ASTROLABE_RESOURCE_SOCKET", "synthetic")
    monkeypatch.setattr(rfd3_worker, "_load_engine", lambda *args: pytest.fail("distributed engine was constructed"))
    with pytest.raises(ValueError, match="distributed ranks need their own admission"):
        rfd3_worker.daemon_factory({"repo_path": ".", "engine_config": {"devices_per_node": 2}})
    assert not recorder.models and not recorder.active


def test_rapidock_rejects_mixed_default_and_nonzero_device_before_load(tmp_path, monkeypatch, recorder):
    from astrolabe.structure_processors.wrappers import rapidock_runtime
    monkeypatch.setenv("ASTROLABE_RESOURCE_SOCKET", "synthetic")
    monkeypatch.setattr(rapidock_runtime, "_load_inference_class", lambda _: pytest.fail("unsafe device load"))
    runtime = rapidock_runtime.RAPiDockRuntime(
        repo_path=tmp_path, model_dir=tmp_path, ckpt="fake", confidence_model_dir=None,
        confidence_ckpt="fake", device="cuda:1")
    with pytest.raises(ValueError, match="logical cuda:0"):
        runtime.load()
    assert not recorder.models and not recorder.active


def test_rapidock_direct_runtime_guard_reuses_kernel_lease(tmp_path, monkeypatch, recorder):
    from astrolabe.structure_processors.wrappers import rapidock_runtime

    class API:
        def __init__(self, **kwargs):
            recorder.check("gpu", "load")

        def run_inference_strict(self, **kwargs):
            recorder.check("gpu", "execute")
            raise ValueError("synthetic API error")

    monkeypatch.setattr(rapidock_runtime, "_load_inference_class", lambda _: API)
    runtime = rapidock_runtime.RAPiDockRuntime(
        repo_path=tmp_path, model_dir=tmp_path, ckpt="fake", confidence_model_dir=None,
        confidence_ckpt="fake", device="cpu",
    )
    handle = rapidock_daemon._create_runtime_handler(runtime, "cpu")
    description = handle({"operation": "describe"})["metadata"]["runtime_lifecycle"]
    assert description["weights_loaded"] is True
    assert description["gpu_resident"] is None
    response = handle({"operation": "infer", "random_state": {}, "payload": {
        "tasks": [{"name": "x", "protein_pdb": "p", "peptide_pdb": "q", "output_dir": str(tmp_path)}], "params": {},
    }})
    assert response["ok"] is False and response["error_type"] == "ValueError"
    assert not recorder.active
    assert len(recorder.models) == 1
    assert [(r, p) for kind, r, p in recorder.events if kind == "enter"] == [
        ("gpu", "load"), ("cpu", "prepare"), ("gpu", "execute")]


@pytest.mark.parametrize("fail", [False, True])
def test_autodock_cli_admits_only_the_gpu_binary(tmp_path, monkeypatch, recorder, fail):
    from astrolabe.structure_processors import autodock

    processor = object.__new__(autodock.AutoDockGPUProcessor)
    processor._get_output_path = lambda name: str(tmp_path / name)
    processor._separate_protein_ligand_with_prody = lambda _: ("protein.pdb", "ligand.pdb")
    processor._infer_oligo_name = lambda _: "case"
    processor._determine_binding_site = lambda *a, **k: {
        "center_x": 1, "center_y": 2, "center_z": 3, "size_x": 4, "size_y": 5, "size_z": 6}
    processor._write_binding_site_metadata = lambda **_: None
    processor._coordinate_transformed_docking_inputs = lambda *args: args
    processor._prepare_ligand_with_obabel = lambda *args: None
    processor._mk_prepare_env = lambda: {}
    processor.mk_prepare_receptor = "prepare"
    processor.mk_prepare_python_bin = None
    processor.autogrid4_executable = "grid"
    processor.autodock_gpu_executable = "gpu"
    processor.num_docking_modes = 7
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[0] == "gpu":
            recorder.check("gpu", "compute")
            assert command == ["gpu", "--lfile", "ligand.pdbqt", "--ffile", "input.maps.fld", "-n", "7"]
            if fail:
                raise subprocess.CalledProcessError(1, command)
        else:
            assert not recorder.active

    monkeypatch.setattr(autodock.subprocess, "run", run)
    if fail:
        with pytest.raises(subprocess.CalledProcessError):
            processor.run_autodock_gpu_docking("input.pdb")
    else:
        processor.run_autodock_gpu_docking("input.pdb")
    assert [cmd[0] for cmd in calls] == ["prepare", "grid", "gpu"]
    assert not recorder.active


@pytest.fixture
def broker(tmp_path, monkeypatch):
    from astrolabe.runtime.admission import ResourceBroker, ResourceClient

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    gpu_id = "GPU-test-" + uuid.uuid4().hex
    sampler = lambda: [dict(index=0, uuid=gpu_id, total_bytes=1000, free_bytes=1000,
                            utilization=0, processes=[])]
    owner = ResourceBroker(dict(
        devices=[dict(uuid=gpu_id, index=0, vram_bytes=800)], socket_dir=str(tmp_path / "broker"),
        cpu_slots=2, ram_bytes=1000, output_bytes=1000, sample_interval=.02,
        sample_max_age=2, rpc_timeout=2, shutdown_timeout=3,
        telemetry_path=str(tmp_path / "broker.jsonl"),
    ), sampler=sampler).start()
    client = ResourceClient(owner.socket_path)
    # start()/close() own this env var. A monkeypatch here would restore the
    # dead socket after close() and contaminate subsequent test modules.
    monkeypatch.setenv("AUXILIARY_TELEMETRY", str(tmp_path / "broker.jsonl"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    yield client
    deadline = time.monotonic() + 3
    while (client.snapshot()["resident_models"] or client.snapshot()["active"]) and time.monotonic() < deadline:
        time.sleep(.02)
    owner.close()


def _package(path, source):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    directory = path.parent
    while directory.name not in ("src", "vendor"):
        (directory / "__init__.py").touch()
        directory = directory.parent


_AUDIT_SOURCE = '''
import json, os
from pathlib import Path
from astrolabe.runtime.admission import ResourceClient
def check(phase):
    snapshot = ResourceClient.from_env().snapshot()
    assert snapshot["active"] == 1, snapshot
    events = [json.loads(line) for line in Path(os.environ["AUXILIARY_TELEMETRY"]).read_text().splitlines()]
    acquire = [event for event in events if event["event"] == "acquire"][-1]
    assert acquire["pid"] == os.getpid(), acquire
    assert acquire["key"][3] == phase, acquire
    assert (acquire["device"] is None) == (phase in ("prepare", "finalize")), acquire
    with Path(os.environ["AUXILIARY_AUDIT"]).open("a") as handle:
        handle.write(json.dumps({"pid": os.getpid(), "phase": phase}) + "\\n")
'''


@pytest.mark.parametrize("mode", ["daemon", "subprocess"])
def test_rfd3_real_runner_child_owns_admission(tmp_path, monkeypatch, broker, mode):
    from astrolabe.initializers.wrappers.rfd3 import RFD3Runner

    repo = tmp_path / "foundry"
    src = repo / "src"
    src.mkdir(parents=True)
    (src / "aux_audit.py").write_text(_AUDIT_SOURCE)
    _package(src / "rfd3/engine.py", '''
from types import SimpleNamespace
from aux_audit import check
class RFD3InferenceConfig(dict): pass
class RFD3InferenceEngine:
    def __init__(self, **config):
        check("load")
        assert config == {"ckpt_path": "rfd3", "seed": 42}
    def run(self, *, inputs, n_batches, out_dir):
        check("execute")
        assert n_batches == 3 and out_dir is None
        return {"case": [SimpleNamespace(example_id="sample", atom_array=[], metadata={"seed": 42})]}
''')
    _package(src / "biotite/structure/io/pdb.py", '''
from pathlib import Path
from aux_audit import check
class PDBFile:
    def set_structure(self, value): check("finalize")
    def write(self, path): Path(path).write_text("ATOM\\n")
''')
    audit = tmp_path / "audit.jsonl"
    monkeypatch.setenv("AUXILIARY_AUDIT", str(audit))
    runner = RFD3Runner({"repo_path": str(repo), "python_bin": sys.executable,
                         "ckpt_path": "rfd3", "seed": 42}, execution_mode=mode)
    try:
        result = runner.run(specification={"length": "4"}, output_dir=tmp_path / "out", n_batches=3)
        assert result[0].structure_path.read_text() == "ATOM\n"
        rows = [json.loads(line) for line in audit.read_text().splitlines()]
        child = rows[0]["pid"]
        assert child != os.getpid()
        assert {row["pid"] for row in rows} == {child}
        assert [row["phase"] for row in rows] == ["load", "execute", "finalize"]
        if mode == "daemon":
            assert broker.snapshot()["models"][0]["pid"] == child == runner.pid
            result2 = runner.run(specification={"length": "5"}, output_dir=tmp_path / "out2", n_batches=3)
            assert result2[0].structure_path.is_file()
            assert len([line for line in audit.read_text().splitlines() if json.loads(line)["phase"] == "load"]) == 1
    finally:
        runner.shutdown()


def test_placer_native_rpc_has_only_child_leases(tmp_path, monkeypatch, broker):
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    (vendor / "aux_audit.py").write_text(_AUDIT_SOURCE)
    (vendor / "PLACER.py").write_text('''
from aux_audit import check
class PLACERinput:
    def __init__(self): check("prepare")
    def pdb(self, path): pass
    def name(self, name): pass
class PLACER:
    def __init__(self): check("load")
    def device(self): return "cuda:0"  # synthetic; no tensors or CUDA calls
    def run(self, input_object, nsamples):
        check("execute")
        assert nsamples == 3
        return {0: {"prmsd": 0.2, "model": "fake"}}
''')
    audit = tmp_path / "audit.jsonl"
    monkeypatch.setenv("AUXILIARY_AUDIT", str(audit))
    input_path = tmp_path / "input.pdb"
    input_path.write_text("ATOM\n")
    client = placer_daemon.PlacerClient({}, mode="daemon", request_timeout=20,
        env={"PYTHONPATH": str(vendor) + os.pathsep + os.environ.get("PYTHONPATH", "")})
    try:
        assert client.predict({"input_file": str(input_path), "num_samples": 3}) == [
            {"key": 0, "prmsd": .2, "model": "fake"}]
        assert client.metadata["runtime_lifecycle"]["weights_loaded"] is True
        rows = [json.loads(line) for line in audit.read_text().splitlines()]
        assert {row["pid"] for row in rows} == {client.pid}
        assert client.pid != os.getpid()
        assert [row["phase"] for row in rows] == ["load", "prepare", "execute"]
        assert broker.snapshot()["models"][0]["pid"] == client.pid
    finally:
        pid = client.pid
        client.shutdown()
    assert not Path(f"/proc/{pid}").exists()


def test_rapidock_raw_jsonl_runner_has_only_child_leases(tmp_path, monkeypatch, broker):
    from astrolabe.structure_processors.wrappers.rapidock_runner import RAPiDockRunner

    repo = tmp_path / "vendor"
    repo.mkdir()
    (repo / "aux_audit.py").write_text(_AUDIT_SOURCE)
    _package(repo / "RAPiDock/API.py", '''
from aux_audit import check
class RAPiDockInference:
    def __init__(self, **kwargs):
        check("load")
        self.confidence_model = None
    def run_inference_strict(self, **kwargs):
        check("execute")
        raise ValueError("synthetic request failure")
''')
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "fake.pt").write_text("fake")
    audit = tmp_path / "audit.jsonl"
    monkeypatch.setenv("AUXILIARY_AUDIT", str(audit))
    runner = RAPiDockRunner(dict(python_bin=sys.executable, repo_path=str(repo),
        model_dir=str(model_dir), ckpt="fake.pt", device="cpu", response_cache=False,
        response_cache_dir=str(tmp_path / "cache")))
    try:
        assert runner.ping() == "pong"
        with pytest.raises(RuntimeError, match="ValueError: synthetic request failure"):
            runner._daemon_request({"type": "dock", "id": "fail", "tasks": [{
                "name": "case", "protein_pdb": "p", "peptide_pdb": "q", "output_dir": str(tmp_path)}]})
        # A device exception is not a completion barrier. Keep the fence and
        # resident until the actual child exits, rather than replaying work.
        deadline = time.monotonic() + 2
        while not broker.snapshot()["fenced"] and time.monotonic() < deadline:
            time.sleep(.01)
        assert broker.snapshot()["fenced"] == 1
        rows = [json.loads(line) for line in audit.read_text().splitlines()]
        assert {row["pid"] for row in rows} == {runner._proc.pid}
        assert rows[0]["pid"] != os.getpid()
        assert [row["phase"] for row in rows] == ["load", "execute"]
        assert broker.snapshot()["models"][0]["pid"] == runner._proc.pid
    finally:
        proc = runner._proc
        runner.shutdown()
        if proc is not None:
            proc.wait(timeout=10)
    deadline = time.monotonic() + 3
    while (broker.snapshot()["resident_models"] or broker.snapshot()["active"]) and time.monotonic() < deadline:
        time.sleep(.02)
    assert broker.snapshot()["resident_models"] == broker.snapshot()["active"] == 0
