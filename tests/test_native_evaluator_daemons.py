from __future__ import annotations

from dataclasses import replace
import importlib.util
import os
from pathlib import Path
import random
import sys

import numpy as np
from omegaconf import OmegaConf
import pytest

from astrolabe.evaluators.native_daemons import (
    NativeAF2Runner,
    NativeAF3Runner,
    NativeBoltz1Runner,
    NativeChai1Runner,
    NativeESMFoldModel,
    NativeProtenixRunner,
    NativeRhoFoldRunner,
    _KernelClient,
    _factory_settings,
    _install_chai_component_cache,
    _rhofold_forward,
)
from astrolabe.evaluators.utils import _execution_controls, shutdown_model_handles
from astrolabe.model_daemon import DaemonSpec, model_identity
from astrolabe.model_execution import ExecutionMode


ADAPTER = r'''
import random
import numpy as np
from astrolabe.model_execution import kernel_handler

LOAD_COUNT = 0


def factory(settings):
    def loader(raw):
        global LOAD_COUNT
        LOAD_COUNT += 1

        def infer(payload):
            operation = payload["operation"]
            if operation == "arrays":
                value = np.asarray(payload["array"], dtype=np.float32)
                return {"array": value + LOAD_COUNT, "load_count": LOAD_COUNT}
            if operation == "random":
                return {"value": random.random(), "load_count": LOAD_COUNT}
            if operation == "options":
                return {
                    "options": payload["options"],
                    "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
                    "load_count": LOAD_COUNT,
                }
            raise ValueError(operation)

        return infer, {"backend": "fake", "load_count": LOAD_COUNT}

    return kernel_handler(settings, loader)
'''


FAKE_STOCK_CHAI = r'''
from contextlib import contextmanager
import torch

LOAD_COUNTS = {}

class JitModule:
    def __init__(self):
        self.devices = []
    def to(self, device):
        self.devices.append(str(device))
        return self

class Wrapper:
    def __init__(self):
        self.jit_module = JitModule()

def load_exported(comp_key, device):
    LOAD_COUNTS[comp_key] = LOAD_COUNTS.get(comp_key, 0) + 1
    return Wrapper()

@contextmanager
def _component_moved_to(comp_key, device):
    component = load_exported(comp_key, device)
    component.jit_module.to(device)
    yield component
    component.jit_module.to("cpu")

def run_inference(comp_key):
    with _component_moved_to(comp_key, torch.device("cpu")) as component:
        return component

def unsupported_run_inference(comp_key):
    return load_exported(comp_key, torch.device("cpu"))
'''


class RecordingClient:
    def __init__(self, metadata=None):
        self.metadata = metadata or {}
        self.requests = []
        self.pid = 123
        self.shutdown_count = 0

    def request(self, payload):
        self.requests.append(payload)
        operation = payload.get("operation")
        if operation == "featurize_data":
            return [{"data": payload["data"], "out_dir": payload["out_dir"]}]
        if operation == "run_inference":
            return {"completed": True}
        if operation == "predict_and_dump":
            return {"confidence": np.asarray([0.75], dtype=np.float32)}
        if operation == "chai1":
            return {"sample_type": "fake"}
        if operation == "forward":
            return {"cord_tns_pred": np.zeros((1, 2, 3), dtype=np.float32)}
        if operation == "export_pdb":
            return {"path": str(payload["path"])}
        if operation == "output_to_pdb":
            return ["ATOM\n"]
        if operation == "infer":
            return {"plddt": np.asarray([0.8], dtype=np.float32)}
        return {"operation": operation}

    def shutdown(self):
        self.shutdown_count += 1


def _spec(tmp_path: Path) -> DaemonSpec:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter.py").write_text(ADAPTER, encoding="utf-8")
    return DaemonSpec(
        backend="native-facade-test",
        factory="adapter:factory",
        settings={"variant": "a"},
        python_bin="python3.11",
        env={"PYTHONPATH": str(adapter_dir), "CUDA_VISIBLE_DEVICES": ""},
        max_frame_bytes=32 * 1024 * 1024,
    )


def test_kernel_client_loads_once_keeps_rng_and_preserves_numpy(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    random.seed(123)
    expected_rng = [random.random(), random.random()]
    random.seed(123)
    client = _KernelClient(
        spec.backend,
        spec.factory,
        spec.settings,
        python_bin=spec.python_bin,
        env=spec.env,
        max_frame_bytes=spec.max_frame_bytes,
    )
    try:
        array = np.arange(12, dtype=np.float32).reshape(3, 4)
        first = client.request({"operation": "arrays", "array": array})
        second = client.request({"operation": "arrays", "array": array})
        assert first["load_count"] == second["load_count"] == 1
        assert first["array"].dtype == np.float32
        assert first["array"].shape == (3, 4)
        assert np.array_equal(second["array"], array + 1)

        random_values = [
            client.request({"operation": "random"})["value"],
            client.request({"operation": "random"})["value"],
        ]
        assert random_values == pytest.approx(expected_rng)
        options = client.request(
            {"operation": "options", "options": {"request_directory": str(tmp_path / "new")}}
        )
        assert options["options"]["request_directory"] == str(tmp_path / "new")
        assert options["cuda_visible_devices"] == ""
    finally:
        client.shutdown()


def test_facades_emit_only_fixed_typed_operations(tmp_path: Path) -> None:
    af2 = NativeAF2Runner.__new__(NativeAF2Runner)
    af2._client = RecordingClient({"multimer_mode": True, "initial_guess": {"enabled": False}})
    af2.multimer_mode = True
    af2._gh_initial_guess_cfg = {"enabled": False}
    feature = np.zeros((2, 3), dtype=np.float32)
    af2.process_features({"aatype": feature}, random_seed=7)
    af2.predict({"aatype": feature}, random_seed=7)

    af3 = NativeAF3Runner.__new__(NativeAF3Runner)
    af3._client = RecordingClient()
    af3._settings = {"model_dir": "/model"}
    af3._cyclic_asym_ids = ()
    af3._cyclic_relative_patches = ()
    af3.set_cyclic_asym_ids([1])
    af3.set_cyclic_relative_patches([[0, 2]])
    af3.run_inference({"aatype": feature}, np.asarray([0], dtype=np.uint32))

    boltz = NativeBoltz1Runner.__new__(NativeBoltz1Runner)
    boltz._client = RecordingClient()
    boltz.featurize_data(tmp_path / "input.yaml", tmp_path / "out-a")
    boltz.run_inference([{"a": 1}], tmp_path / "out-b")

    protenix = NativeProtenixRunner.__new__(NativeProtenixRunner)
    protenix._client = RecordingClient()
    protenix.configs = OmegaConf.create({"seed": 11, "use_template": False})
    protenix._pending_configs = None
    protenix.update_model_configs(OmegaConf.create({"seed": 12}))
    protenix.predict_and_dump(
        {"N_token": np.asarray([4])},
        atom_array=["atom"],
        dataset_name="step_001",
        pdb_id="case",
        seed=12,
        entity_poly_type={"A": "protein"},
    )

    esmfold = NativeESMFoldModel.__new__(NativeESMFoldModel)
    esmfold._client = RecordingClient()
    output = esmfold.infer(["ACD"])
    assert esmfold.output_to_pdb(output) == ["ATOM\n"]

    rhofold = NativeRhoFoldRunner.__new__(NativeRhoFoldRunner)
    rhofold._client = RecordingClient()
    rhofold.to("cpu")
    rhofold.forward(tokens="tokens", rna_fm_tokens="rna", seq="ACGU")
    rhofold.export_pdb_file(
        "ACGU",
        np.zeros((3, 3), dtype=np.float32),
        path=tmp_path / "rna.pdb",
        chain_id=None,
        confidence=np.ones(3, dtype=np.float32),
    )

    chai = NativeChai1Runner.__new__(NativeChai1Runner)
    chai._client = RecordingClient()
    chai.supports_constraints = True
    chai.run_inference(fasta_file=tmp_path / "input.fasta", output_dir=tmp_path / "chai")

    assert [request["operation"] for request in af2._client.requests] == [
        "process_features",
        "predict",
    ]
    assert [request["random_seed"] for request in af2._client.requests] == [7, 7]
    assert [request["operation"] for request in boltz._client.requests] == [
        "featurize_data",
        "run_inference",
    ]
    assert boltz._client.requests[0]["out_dir"] != boltz._client.requests[1]["out_dir"]
    assert protenix._client.requests[0]["operation"] == "predict_and_dump"
    assert protenix._client.requests[0]["configs"]["seed"] == 12
    assert af3._client.requests[0]["cyclic_asym_ids"] == (1,)
    assert af3._client.requests[0]["cyclic_relative_patches"] == ((0, 2),)


def test_factory_metadata_and_execution_controls_are_not_model_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = {
        "model": {"depth": 2},
        "_initial_random_state": {"python": "allocation"},
        "_provider_request": {"backend": "cuda"},
    }
    assert _factory_settings(settings) == {"model": {"depth": 2}}

    monkeypatch.delenv("ASTROLABE_MODEL_EXECUTION", raising=False)
    conf = OmegaConf.create(
        {
            "model_name": "fake.pt",
            "execution": "in_process",
            "execution_request_timeout": 123,
            "execution_max_frame_bytes": 456,
        }
    )
    mode, options = _execution_controls(conf)
    assert mode is ExecutionMode.IN_PROCESS
    assert options == {"request_timeout": 123, "max_frame_bytes": 456}
    assert "execution" not in conf
    assert "execution_request_timeout" not in conf

    canonical_conf = OmegaConf.create(
        {"execution_mode": "daemon", "execution": "daemon"}
    )
    mode, _ = _execution_controls(canonical_conf)
    assert mode is ExecutionMode.DAEMON
    assert "execution_mode" not in canonical_conf
    assert "execution" not in canonical_conf

    conflicting_conf = OmegaConf.create(
        {"execution_mode": "daemon", "execution": "in_process"}
    )
    with pytest.raises(ValueError, match="disagree"):
        _execution_controls(conflicting_conf)

    spec = DaemonSpec("backend", "module:factory", {"model": 1, "_initial_random_state": 1})
    assert model_identity(spec) == model_identity(
        replace(spec, settings={"model": 1, "_initial_random_state": 2})
    )


def test_evaluator_shutdown_closes_unique_owned_handles_only() -> None:
    owned = RecordingClient()
    external = object()
    shutdown_model_handles({"first": owned, "second": owned}, external)
    assert owned.shutdown_count == 1

    from astrolabe.evaluators.base import Evaluator

    base_evaluator = Evaluator.__new__(Evaluator)
    base_evaluator.model_runner = RecordingClient()
    base_evaluator.shutdown()
    assert base_evaluator.model_runner.shutdown_count == 1

    from astrolabe.evaluators.alphafold2 import AlphaFold2Evaluator
    from astrolabe.evaluators.alphafold3 import AlphaFold3Evaluator
    from astrolabe.evaluators.boltz import BoltzEvaluator
    from astrolabe.evaluators.chai1 import Chai1Evaluator
    from astrolabe.evaluators.esmfold import ESMFoldEvaluator
    from astrolabe.evaluators.protenix import ProtenixEvaluator
    from astrolabe.evaluators.rf3 import RoseTTAFold3Evaluator

    cases = [
        (AlphaFold2Evaluator, {"model_runners": {"a": RecordingClient()}}),
        (AlphaFold3Evaluator, {"model_runner": RecordingClient()}),
        (BoltzEvaluator, {"model_runner": RecordingClient()}),
        (Chai1Evaluator, {"model_runner": RecordingClient()}),
        (ESMFoldEvaluator, {"_model": RecordingClient()}),
        (ProtenixEvaluator, {"model_runner": RecordingClient()}),
        (RoseTTAFold3Evaluator, {"model_runner": RecordingClient()}),
    ]
    for evaluator_type, attributes in cases:
        evaluator = evaluator_type.__new__(evaluator_type)
        for name, value in attributes.items():
            setattr(evaluator, name, value)
        evaluator.shutdown()
        handle = next(iter(attributes.values()))
        owned_handle = next(iter(handle.values())) if isinstance(handle, dict) else handle
        assert owned_handle.shutdown_count == 1


def test_chai_stock_component_cache_loads_each_export_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_path = tmp_path / "fake_stock_chai.py"
    module_path.write_text(FAKE_STOCK_CHAI, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("fake_stock_chai", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "fake_stock_chai", module)
    spec.loader.exec_module(module)

    cache = _install_chai_component_cache(
        module,
        provider_identity="provider:chai:exact",
    )
    first = module.run_inference("trunk.pt")
    second = module.run_inference("trunk.pt")
    assert first is second
    assert module.LOAD_COUNTS == {"trunk.pt": 1}
    assert next(iter(cache)) == ("trunk.pt", "cpu", "provider:chai:exact")
    assert first.jit_module.devices[-1] == "cpu"

    second_cache = _install_chai_component_cache(
        module,
        provider_identity="provider:chai:second-owner",
    )
    module.run_inference("trunk.pt")
    assert module.LOAD_COUNTS == {"trunk.pt": 2}
    assert next(iter(second_cache))[2] == "provider:chai:second-owner"

    module.run_inference = module.unsupported_run_inference
    with pytest.raises(RuntimeError, match="does not use _component_moved_to"):
        _install_chai_component_cache(module, provider_identity=None)


def test_rhofold_daemon_keeps_parent_cpu_and_child_disables_grad(tmp_path: Path) -> None:
    pytest.importorskip("rhofold")
    import torch

    from astrolabe.evaluators.rhofold import RhoFoldEvaluator

    class FakeRunner:
        native_daemon = True

        def __init__(self):
            self.device = None

        def to(self, device):
            self.device = device
            return self

    runner = FakeRunner()
    evaluator = RhoFoldEvaluator(
        runner,
        tmp_path,
        OmegaConf.create({"device": "cuda", "relax_steps": None}),
    )
    assert evaluator.model_runner.device == "cpu"

    observed = {}

    def model(**kwargs):
        observed["grad_enabled"] = torch.is_grad_enabled()
        observed["tokens_cuda"] = kwargs["tokens"].is_cuda
        observed["rna_cuda"] = kwargs["rna_fm_tokens"].is_cuda
        return (None, {"ok": True})

    result = _rhofold_forward(
        model,
        {
            "tokens": torch.ones(1, 2),
            "rna_fm_tokens": torch.ones(1, 2),
            "seq": "ACGU",
        },
        "cpu",
    )
    assert result == {"ok": True}
    assert observed == {
        "grad_enabled": False,
        "tokens_cuda": False,
        "rna_cuda": False,
    }

    class ParentDeviceTensor:
        is_cuda = True

    with pytest.raises(ValueError, match="CPU input tensors"):
        _rhofold_forward(
            model,
            {"tokens": ParentDeviceTensor(), "rna_fm_tokens": torch.ones(1), "seq": "A"},
            "cpu",
        )


def test_rhofold_shutdown_closes_unique_owned_handle():
    pytest.importorskip("rhofold")
    from astrolabe.evaluators.rhofold import RhoFoldEvaluator
    evaluator = RhoFoldEvaluator.__new__(RhoFoldEvaluator)
    evaluator.model_runner = RecordingClient()
    evaluator.shutdown()
    assert evaluator.model_runner.shutdown_count == 1
