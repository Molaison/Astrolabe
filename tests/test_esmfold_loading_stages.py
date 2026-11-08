"""CPU-only loading contracts: no Torch/vendor dependency or real checkpoint.

The evaluator source is imported with stand-ins for its scientific imports.
All loading methods and the native factory itself are the production code;
kernel admission/completion is deliberately left to its separate coordinator.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import random
import sys
from types import ModuleType, SimpleNamespace

import pytest

from astrolabe import model_execution
from astrolabe.evaluators import native_daemons, esmfold_stages
from astrolabe.runtime.loading import StagedLoader


@pytest.fixture
def fixture(monkeypatch, tmp_path):
    calls = []
    checkpoint = tmp_path / "weights.pt"
    checkpoint.touch()  # Existence sentinel; torch.load below never opens it.
    weight = object()
    old_key = "trunk.structure_module.ipa.linear_q_points.weight"
    new_key = "trunk.structure_module.ipa.linear_q_points.linear.weight"
    state = SimpleNamespace(
        phase="cpu_prepare", failure=None, error=ValueError("load failure"),
        expected={"fold.weight": weight, new_key: weight, "esm.optional": weight},
        data={"cfg": {"model": {"precision": "unchanged"}},
              "model": {"fold.weight": weight, old_key: weight}},
    )

    def event(name):
        calls.append((name, state.phase))
        if state.failure == name:
            raise state.error

    class Model:
        def __init__(self, *, esmfold_config):
            event("model_constructor")
            assert state.phase == "cpu_prepare"
            assert esmfold_config is state.data["cfg"]["model"]
            self.device = "cpu"

        def state_dict(self):
            event("state_dict")
            assert self.device == "cpu"
            return state.expected

        def load_state_dict(self, value, *, strict):
            event("load_state_dict")
            assert self.device == "cpu" and strict is False
            assert value[old_key] is weight and value[new_key] is weight
            assert value["fold.weight"] is weight
            assert new_key not in state.data["model"]

        def eval(self):
            event("eval")
            assert self.device == "cpu"
            return self

        def to(self, device):
            event("to")
            assert calls[-2][0] == "eval"
            self.device = device
            return self

        def infer(self, sequences):
            event("infer")
            return sequences

        def output_to_pdb(self, output):
            event("output_to_pdb")
            return output

    torch = ModuleType("torch")

    def forbidden(*args, **kwargs):
        pytest.fail("Loading instrumentation must not probe or initialize CUDA")

    def load(*args, **kwargs):
        event("torch_load")
        assert state.phase == "cpu_prepare"
        assert args == (str(state.load_path),)
        assert kwargs == {"map_location": "cpu", "weights_only": False}
        return state.data

    torch.load = load
    torch.hub = SimpleNamespace(get_dir=lambda: str(tmp_path))
    torch.cuda = SimpleNamespace(is_available=forbidden, init=forbidden,
                                 Event=forbidden, synchronize=forbidden)
    parser = ModuleType("astrolabe.core.protomer_config_parser")
    parser.ProtomerTypes = SimpleNamespace(PROTEIN="protein")
    utils = ModuleType("astrolabe.evaluators.utils")
    utils.StandardScores = object
    utils.shutdown_model_handles = lambda model: None
    module_path = Path(native_daemons.__file__).with_name("esmfold.py")
    spec = importlib.util.spec_from_file_location("_esmfold_loading_contract", module_path)
    module = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as imports:
        for name, value in (("torch", torch), (parser.__name__, parser),
                            (utils.__name__, utils)):
            imports.setitem(sys.modules, name, value)
        spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_load_esmfold_model_class", lambda: Model)
    monkeypatch.setitem(sys.modules, "astrolabe.evaluators.esmfold", module)
    state.load_path = checkpoint
    return SimpleNamespace(module=module, calls=calls, state=state, torch=torch,
                           Model=Model, checkpoint=checkpoint)


def names(fixture):
    return [name for name, _ in fixture.calls]


def test_prepare_is_cpu_only_and_place_keeps_order_and_device(fixture):
    evaluator = fixture.module.ESMFoldEvaluator(
        model_name=str(fixture.checkpoint), device="cuda:3")
    prepared = evaluator._prepare_model_cpu()
    assert prepared.device == "cpu" and evaluator._model is None
    assert names(fixture) == ["torch_load", "model_constructor", "state_dict", "load_state_dict"]
    fixture.state.phase = "device_placement"
    assert evaluator._place_model(prepared) is prepared
    assert prepared.device == "cuda:3"
    assert fixture.calls[-2:] == [("eval", "device_placement"), ("to", "device_placement")]
    assert evaluator._ensure_model() is prepared
    assert len(fixture.calls) == 6  # Cached access performs no extra loading or placement.


def test_optional_trace_counts_actual_load_and_reuse_without_rng_or_cuda(fixture, monkeypatch, tmp_path):
    trace = tmp_path / "actual-load.jsonl"
    monkeypatch.setenv("ASTROLABE_MODEL_LIFECYCLE_PATH", str(trace))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-test-allocation")
    before = random.getstate()
    evaluator = fixture.module.ESMFoldEvaluator(
        model_name=str(fixture.checkpoint), device="cpu",
    )
    assert not trace.exists()  # Construction alone supplies no loading evidence.
    model = evaluator._ensure_model()
    assert evaluator._ensure_model() is model
    assert evaluator._ensure_model() is model
    evaluator.shutdown()
    records = [json.loads(line) for line in trace.read_text().splitlines()]
    assert [row["event"] for row in records] == [
        "cpu_weights_loaded", "device_placement_completed",
        "model_reused", "model_reused", "shutdown_requested",
    ]
    assert {row["loader_count"] for row in records} == {1}
    assert {row["loader_attempt_count"] for row in records} == {1}
    assert [row["placement_count"] for row in records] == [0, 1, 1, 1, 1]
    assert len({(row["pid"], row["model_id"]) for row in records}) == 1
    assert all(row["cuda_visible_devices"] == "GPU-test-allocation" for row in records)
    assert records[0]["load_metadata"]["stages"]["cpu_prepare"]["status"] == "completed"
    assert records[1]["load_metadata"]["stages"]["device_placement"]["status"] == "completed"
    assert records[1]["load_metadata"]["checkpoint_path"] == str(fixture.checkpoint.resolve())
    assert evaluator._model is model  # Shutdown request does not prove unloading.
    assert random.getstate() == before
    assert names(fixture).count("torch_load") == 1
    assert names(fixture).count("load_state_dict") == 1


def test_trace_does_not_claim_injected_or_failed_model_loaded(fixture, monkeypatch, tmp_path):
    trace = tmp_path / "failed-load.jsonl"
    monkeypatch.setenv("ASTROLABE_MODEL_LIFECYCLE_PATH", str(trace))
    injected = fixture.module.ESMFoldEvaluator(model=object())
    injected._ensure_model()
    injected.shutdown()
    assert not trace.exists()
    evaluator = fixture.module.ESMFoldEvaluator(
        model_name=str(fixture.checkpoint), device="cpu",
    )
    fixture.state.failure = "torch_load"
    with pytest.raises(ValueError):
        evaluator._ensure_model()
    assert evaluator._loader_attempts == 1
    assert evaluator._successful_loads == 0
    assert not trace.exists()
    evaluator.shutdown()
    record = json.loads(trace.read_text())
    assert record["event"] == "shutdown_requested"
    assert record["loader_count"] == 0
    assert record["loader_attempt_count"] == 1
    assert record["load_metadata"]["stages"]["torch_load"]["status"] == "failed"


@pytest.mark.parametrize("available, expected", [(False, "cpu"), (True, "cuda")])
def test_legacy_default_device_is_selected_only_after_cpu_loading(fixture, available, expected):
    evaluator = fixture.module.ESMFoldEvaluator(model_name=str(fixture.checkpoint))

    def available_cuda():
        assert names(fixture) == ["torch_load", "model_constructor", "state_dict", "load_state_dict"]
        assert "device_placement" in evaluator._load_metadata["stages"]
        return available

    fixture.torch.cuda.is_available = available_cuda
    assert evaluator._ensure_model().device == expected
    assert evaluator._device == expected
    assert names(fixture)[-2:] == ["eval", "to"]


def test_explicit_cpu_and_injected_model_preserve_legacy_behavior(fixture):
    evaluator = fixture.module.ESMFoldEvaluator(model_name=str(fixture.checkpoint), device="cpu")
    assert evaluator._ensure_model().device == "cpu"
    fixture.calls.clear()
    injected = object()
    evaluator = fixture.module.ESMFoldEvaluator(model=injected)
    assert evaluator._ensure_model() is injected
    assert evaluator._prepare_model_cpu() is injected
    assert evaluator._place_model(injected) is injected
    assert evaluator._load_metadata == {} and not fixture.calls


def test_default_checkpoint_cache_path_and_load_options_are_preserved(fixture):
    cached = fixture.checkpoint.parent / "checkpoints" / "esmfold_3B_v1.pt"
    cached.parent.mkdir()
    cached.touch()
    fixture.state.load_path = cached
    evaluator = fixture.module.ESMFoldEvaluator(device="cpu")
    evaluator._ensure_model()
    assert names(fixture)[0] == "torch_load"


@pytest.mark.parametrize("cached", [False, True])
def test_missing_checkpoint_never_loads_or_downloads(fixture, cached):
    model_name = "missing" if cached else str(fixture.checkpoint.parent / "missing.pt")
    evaluator = fixture.module.ESMFoldEvaluator(model_name=model_name)
    with pytest.raises(FileNotFoundError, match="auto-download is disabled" if cached else "weights not found"):
        evaluator._ensure_model()
    assert not fixture.calls
    assert set(evaluator._load_metadata["stages"]) == {"cpu_prepare"}
    assert evaluator._load_metadata["stages"]["cpu_prepare"]["status"] == "failed"


@pytest.mark.parametrize("failure, stages", [
    ("torch_load", ["torch_load"]),
    ("model_constructor", ["torch_load", "model_constructor"]),
    ("load_state_dict", ["torch_load", "model_constructor", "load_state_dict"]),
    ("eval", ["torch_load", "model_constructor", "load_state_dict", "device_placement"]),
    ("to", ["torch_load", "model_constructor", "load_state_dict", "device_placement"]),
])
def test_loading_errors_propagate_without_running_later_stages(fixture, failure, stages):
    fixture.state.failure = failure
    evaluator = fixture.module.ESMFoldEvaluator(model_name=str(fixture.checkpoint), device="cpu")
    with pytest.raises(ValueError) as error:
        evaluator._ensure_model()
    assert error.value is fixture.state.error
    assert names(fixture)[-1] == failure and evaluator._model is None
    metadata = evaluator._load_metadata["stages"]
    assert set(metadata) == {"cpu_prepare", *stages}
    assert metadata[stages[-1]]["status"] == "failed"
    assert metadata["cpu_prepare"]["status"] == (
        "completed" if failure in {"eval", "to"} else "failed")


def test_essential_key_validation_stays_before_state_loading(fixture):
    fixture.state.expected["essential.missing"] = object()
    evaluator = fixture.module.ESMFoldEvaluator(model_name=str(fixture.checkpoint), device="cpu")
    with pytest.raises(RuntimeError, match="essential.missing"):
        evaluator._ensure_model()
    assert names(fixture) == ["torch_load", "model_constructor", "state_dict"]
    assert "load_state_dict" not in evaluator._load_metadata["stages"]


def test_metadata_records_real_nested_host_intervals_and_anchor(fixture, monkeypatch):
    wall = iter(range(100, 200))
    process = iter(range(10, 200, 2))
    monkeypatch.setattr(fixture.module, "time", SimpleNamespace(
        time=lambda: 1234.0, perf_counter=lambda: next(wall),
        process_time=lambda: next(process)))
    evaluator = fixture.module.ESMFoldEvaluator(model_name=str(fixture.checkpoint), device="cpu")
    prepared = evaluator._prepare_model_cpu()
    metadata = evaluator._load_metadata
    assert metadata["clock_anchor"] == {"time_s": 1234.0, "perf_counter_s": 100, "process_time_s": 10}
    assert metadata["timing_scope"] == "host_calls"
    assert "device_placement" not in metadata["stages"]
    evaluator._place_model(prepared)
    stages = metadata["stages"]
    cpu = stages["cpu_prepare"]
    for name in ("torch_load", "model_constructor", "load_state_dict"):
        stage = stages[name]
        assert cpu["start_perf_counter_s"] < stage["start_perf_counter_s"] < stage["end_perf_counter_s"] < cpu["end_perf_counter_s"]
    assert cpu["end_perf_counter_s"] < stages["device_placement"]["start_perf_counter_s"]
    for stage in stages.values():
        assert stage["wall_s"] == stage["end_perf_counter_s"] - stage["start_perf_counter_s"]
        assert stage["process_time_s"] == 2 * stage["wall_s"]
        assert stage["status"] == "completed"
    assert set(stages) == {"cpu_prepare", "torch_load", "model_constructor", "load_state_dict", "device_placement"}
    json.dumps(metadata, allow_nan=False)


def native_loader(fixture, monkeypatch, device="cuda:2"):
    def handler(settings, loader, **kwargs):
        assert isinstance(loader, StagedLoader)
        assert kwargs["torch_device"] == (device or "auto")
        assert kwargs["lifecycle"]["backend"] == "esmfold"
        assert kwargs["lifecycle"]["weights_loaded"] is None
        assert kwargs["lifecycle"]["gpu_resident"] is None
        return loader

    monkeypatch.setattr(model_execution, "kernel_handler", handler)
    settings = {"model_name": str(fixture.checkpoint), "device": device,
                "_initial_random_state": "opaque", "_provider_request": "opaque"}
    return native_daemons._esmfold_factory(settings), settings


@pytest.mark.parametrize("boundary", [False, True])
def test_native_factory_passes_stages_and_preserves_inference_routes(fixture, monkeypatch, boundary):
    def encoding(model):
        assert model.device == "cuda:2"
        if not boundary:
            return None
        return (lambda request: (request["operation"], request.get("sequences", request.get("output"))),
                lambda value: value, lambda value: value[1])

    monkeypatch.setattr(esmfold_stages, "native_encoding_boundary", encoding)
    monkeypatch.setattr(model_execution, "host_result", lambda value: value)
    loader, settings = native_loader(fixture, monkeypatch)
    prepared = loader.prepare(settings)
    assert prepared[1].device == "cpu" and names(fixture)[-1] == "load_state_dict"
    fixture.state.phase = "device_placement"
    if not boundary:
        with pytest.raises(ValueError, match="Unsupported ESMFold infer body"):
            loader.place(prepared)
        return
    stages, metadata = loader.place(prepared)
    assert isinstance(stages, model_execution.EvaluatorStages)
    assert metadata["device"] == "cuda:2"
    assert metadata["load_metadata"] is prepared[0]._load_metadata
    assert metadata["model_state"] is metadata["load_metadata"]["model_state"]
    assert metadata["model_state"]["weights_loaded"] is None
    assert metadata["model_state"]["gpu_resident"] is None
    assert metadata["cpu_encoding"] == "verified_native_infer"
    for operation, key in (("infer", "sequences"), ("output_to_pdb", "output")):
        value = ["preserved"]
        request = {"operation": operation, key: value}
        assert stages.finalize(stages.execute(stages.prepare(request))) is value
        assert stages.resource(request) == ("cpu" if operation == "output_to_pdb" else "gpu")


class FakeTensor:
    """Only host descriptors are supported; tensor operations must never run."""

    def __init__(self, device, dtype, numel, element_size):
        self.device = device
        self.dtype = dtype
        self._numel = numel
        self._element_size = element_size

    def numel(self):
        return self._numel

    def element_size(self):
        return self._element_size

    def __getattr__(self, name):
        raise AssertionError(f"Inventory attempted tensor operation {name}")


def test_inventory_aggregates_actual_devices_dtypes_and_logical_bytes(fixture):
    cpu = FakeTensor("cpu", "torch.float32", 12, 4)
    cuda = FakeTensor("cuda:3", "torch.bfloat16", 3, 2)
    buffer = FakeTensor("cuda:3", "torch.int64", 2, 8)
    model = SimpleNamespace(
        named_parameters=lambda: iter((("cpu", cpu), ("tied_cpu", cpu), ("cuda", cuda))),
        named_buffers=lambda: iter((("count", buffer),)),
    )
    rng = random.getstate()
    state = fixture.module._model_tensor_inventory(model)
    assert random.getstate() == rng
    assert state["weights_loaded"] is True and state["gpu_resident"] is True
    assert state["byte_accounting"] == "logical_numel_times_element_size"
    assert state["parameters"] == {
        "available": True, "tensor_count": 2, "numel": 15, "bytes": 54,
        "by_device_dtype": [
            {"device": "cpu", "dtype": "torch.float32", "tensor_count": 1, "numel": 12, "bytes": 48},
            {"device": "cuda:3", "dtype": "torch.bfloat16", "tensor_count": 1, "numel": 3, "bytes": 6},
        ],
    }
    assert state["buffers"] == {
        "available": True, "tensor_count": 1, "numel": 2, "bytes": 16,
        "by_device_dtype": [
            {"device": "cuda:3", "dtype": "torch.int64", "tensor_count": 1, "numel": 2, "bytes": 16},
        ],
    }
    json.dumps(state, allow_nan=False)  # No tensor references or copies in metadata.


@pytest.mark.parametrize("parameter_devices, buffer_devices, expected", [
    (None, None, (None, None)),
    (None, ["cuda:0"], (None, None)),
    ([], ["cuda:0"], (None, None)),
    (["cpu"], [], (True, False)),
    (["cuda:2"], ["cpu"], (True, True)),
    (["cpu"], ["cuda:2"], (True, True)),
    (["meta"], [], (False, False)),
    (["meta", "cuda:2"], [], (False, True)),
])
def test_inventory_flags_require_nonempty_parameter_evidence(
    fixture, parameter_devices, buffer_devices, expected,
):
    model = SimpleNamespace()
    for method, devices in (("named_parameters", parameter_devices),
                             ("named_buffers", buffer_devices)):
        if devices is not None:
            tensors = [(str(i), FakeTensor(device, "torch.float32", 3, 4))
                       for i, device in enumerate(devices)]
            setattr(model, method, lambda tensors=tensors: iter(tensors))
    state = fixture.module._model_tensor_inventory(model)
    assert (state["weights_loaded"], state["gpu_resident"]) == expected
    assert state["parameters"]["available"] is (parameter_devices is not None)
    assert state["buffers"]["available"] is (buffer_devices is not None)


def test_zero_element_parameters_do_not_prove_loaded_weights(fixture):
    model = SimpleNamespace(named_parameters=lambda: iter([
        ("empty", FakeTensor("cuda:0", "torch.float32", 0, 4))]))
    state = fixture.module._model_tensor_inventory(model)
    assert state["parameters"]["tensor_count"] == 1
    assert state["parameters"]["numel"] == state["parameters"]["bytes"] == 0
    assert state["weights_loaded"] is None and state["gpu_resident"] is None


@pytest.mark.parametrize("moves_tensors", [True, False])
def test_inventory_is_rechecked_after_to_and_before_native_ready(fixture, monkeypatch, moves_tensors):
    parameter = FakeTensor("cpu", "torch.bfloat16", 7, 2)
    buffer = FakeTensor("cpu", "torch.int64", 3, 8)
    observations = []

    def parameters(model):
        observations.append(("parameters", parameter.device))
        return iter([("weight", parameter)])

    def buffers(model):
        observations.append(("buffers", buffer.device))
        return iter([("count", buffer)])

    original_to = fixture.Model.to

    def to(model, device):
        result = original_to(model, device)
        observations.append(("to", device))
        if moves_tensors:
            parameter.device = buffer.device = device
        return result

    monkeypatch.setattr(fixture.Model, "named_parameters", parameters, raising=False)
    monkeypatch.setattr(fixture.Model, "named_buffers", buffers, raising=False)
    monkeypatch.setattr(fixture.Model, "to", to)
    actual_device = "cuda:2" if moves_tensors else "cpu"

    def boundary(model):
        assert observations[-2:] == [("parameters", actual_device), ("buffers", actual_device)]
        observations.append(("boundary", model.device))
        return lambda value: value, lambda value: value, lambda value: value

    monkeypatch.setattr(esmfold_stages, "native_encoding_boundary", boundary)
    loader, settings = native_loader(fixture, monkeypatch)
    prepared = loader.prepare(settings)
    cpu_state = prepared[0]._load_metadata["model_state"]
    assert cpu_state["weights_loaded"] is True and cpu_state["gpu_resident"] is False
    assert names(fixture) == ["torch_load", "model_constructor", "state_dict", "load_state_dict"]
    fixture.state.phase = "device_placement"
    rng = random.getstate()
    _, metadata = loader.place(prepared)
    assert random.getstate() == rng
    final_state = metadata["model_state"]
    assert final_state is metadata["load_metadata"]["model_state"]
    assert final_state["weights_loaded"] is True
    assert final_state["gpu_resident"] is moves_tensors
    assert final_state["parameters"]["by_device_dtype"] == [{
        "device": actual_device, "dtype": "torch.bfloat16", "tensor_count": 1, "numel": 7, "bytes": 14,
    }]
    assert final_state["buffers"]["bytes"] == 24
    assert cpu_state["parameters"]["by_device_dtype"][0]["device"] == "cpu"
    assert names(fixture)[-2:] == ["eval", "to"]
    assert observations == [("parameters", "cpu"), ("buffers", "cpu"),
                            ("to", "cuda:2"), ("parameters", actual_device),
                            ("buffers", actual_device), ("boundary", "cuda:2")]


def test_injected_model_with_inventory_is_inspected_without_loading_or_placement(fixture):
    injected = SimpleNamespace(named_parameters=lambda: iter([
        ("weight", FakeTensor("cpu", "torch.float32", 2, 4))]))
    evaluator = fixture.module.ESMFoldEvaluator(model=injected, device="cuda:0")
    assert evaluator._ensure_model() is injected
    assert evaluator._load_metadata["model_state"]["gpu_resident"] is False
    assert evaluator._load_metadata["model_state"]["weights_loaded"] is True
    assert evaluator._prepare_model_cpu() is injected
    assert evaluator._place_model(injected) is injected
    assert not fixture.calls


@pytest.mark.parametrize("failure_phase", ["cpu_prepare", "device_placement"])
def test_inventory_errors_prevent_ready_and_propagate(fixture, monkeypatch, failure_phase):
    def parameters(model):
        if fixture.state.phase == failure_phase:
            raise fixture.state.error
        return iter([("weight", FakeTensor("cpu", "torch.float32", 2, 4))])

    monkeypatch.setattr(fixture.Model, "named_parameters", parameters, raising=False)
    monkeypatch.setattr(esmfold_stages, "native_encoding_boundary",
                        lambda model: pytest.fail("Inventory failure must prevent ready"))
    loader, settings = native_loader(fixture, monkeypatch)
    if failure_phase == "cpu_prepare":
        with pytest.raises(ValueError) as error:
            loader.prepare(settings)
    else:
        prepared = loader.prepare(settings)
        fixture.state.phase = failure_phase
        with pytest.raises(ValueError) as error:
            loader.place(prepared)
        assert prepared[0]._model is None
    assert error.value is fixture.state.error
