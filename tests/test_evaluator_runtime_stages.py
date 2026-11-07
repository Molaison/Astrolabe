"""Real evaluator factories with CPU vendor doubles; no model/GPU acceptance."""
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
import json
import logging
import os
import random
import subprocess
import sys
import threading
from types import MethodType

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from astrolabe import model_execution as execution
from astrolabe.evaluators import native_daemons as native
from astrolabe.evaluators.runtime_stages import (
    DirectKernel, local_evaluator_guard, resident_engine_kernel,
)
from astrolabe.inference_pipeline import Pending, Stage


class Broker:
    def __init__(self):
        self.socket_path = "/fake/broker"
        self.active = []
        self.events = []
        self.models = []
        self.leases = []

    @contextmanager
    def lease(self, *, resource, phase, **kwargs):
        self.leases.append(dict(resource=resource, phase=phase, **kwargs))
        entry = (resource, phase)
        self.active.append(entry)
        self.events.append(("enter", entry))
        try:
            yield
        finally:
            self.active.remove(entry)
            self.events.append(("exit", entry))

    def register_model(self, model_id, **kwargs):
        self.models.append((model_id, kwargs))


@pytest.fixture
def broker(monkeypatch):
    broker = Broker()
    monkeypatch.delenv("ASTROLABE_RESOURCE_SOCKET", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(execution, "_resource_client", lambda: broker)
    return broker


def call(handler, payload):
    return handler(dict(operation="infer", payload=payload,
                        random_state=execution.capture_random_state()))


def lifecycle(handler):
    return handler({"operation": "describe"})["metadata"]["runtime_lifecycle"]


def vendor(monkeypatch, name, **attributes):
    monkeypatch.setitem(sys.modules, name, SimpleNamespace(**attributes))


def staged_esmfold_double(evaluator):
    """Keep vendor doubles aligned with the existing two-part loader contract."""
    evaluator._prepare_model_cpu = evaluator._ensure_model
    evaluator._place_model = lambda prepared: prepared
    evaluator._load_metadata = {"model_state": {
        "weights_loaded": None, "gpu_resident": None,
    }}
    return evaluator


def staged_af2_double(runner, monkeypatch):
    """Pin the vendor predict body while replacing only its numerical model."""
    class ReadyArray(np.ndarray):
        def block_until_ready(self):
            return self

    def map_tree(function, value):
        if isinstance(value, dict):
            return {key: map_tree(function, item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(map_tree(function, item) for item in value)
        return function(value)

    original_predict = runner.predict
    runner.params = np.asarray(1).view(ReadyArray)
    runner.init_params = lambda features: None
    runner.multimer_mode = getattr(runner, "multimer_mode", False)
    runner.apply = lambda params, key, features: map_tree(
        lambda value: np.asarray(value).view(ReadyArray),
        original_predict(features, int(key)),
    )
    monkeypatch.setitem(globals(), "tree", SimpleNamespace(map=map_tree))
    monkeypatch.setitem(globals(), "jax", SimpleNamespace(
        random=SimpleNamespace(PRNGKey=lambda seed: np.asarray(seed).view(ReadyArray)),
        device_put=lambda value: map_tree(lambda x: np.asarray(x).view(ReadyArray), value),
    ))
    monkeypatch.setitem(globals(), "get_confidence_metrics", lambda result, multimer_mode: {})
    runner.predict = MethodType(NativeAF2Predict.predict, runner)
    return runner


class NativeAF2Predict:
    def predict(self, feat, random_seed):
        self.init_params(feat)
        logging.info(
            'Running predict with shape(feat) = %s',
            tree.map(lambda x: x.shape, feat),
        )
        result = self.apply(self.params, jax.random.PRNGKey(random_seed), feat)
        tree.map(lambda x: x.block_until_ready(), result)
        result.update(
            get_confidence_metrics(result, multimer_mode=self.multimer_mode)
        )
        logging.info('Output shape was %s', tree.map(lambda x: x.shape, result))
        return result


def test_af2_factory_loads_once_classifies_operations_preserves_features_and_seed(monkeypatch, broker):
    vendor(monkeypatch, "astrolabe.evaluators.utils", _ensure_alphafold2_import_path=lambda _: None)
    loads = []
    seeds = []

    class Runner:
        multimer_mode = True
        _gh_initial_guess_cfg = {"enabled": True}

        def process_features(self, features, random_seed):
            assert broker.active[-1] == ("gpu", "prepare")
            features["nested"]["values"].append(4)
            features["nested"]["values"] = np.asarray(features["nested"]["values"], dtype=np.float32)
            seeds.append(random_seed)
            return features

        def predict(self, features, random_seed):
            assert broker.active[-1] == ("gpu", "infer")
            seeds.append(random_seed)
            return {"plddt": np.array(features["nested"]["values"], dtype=np.float32)}

    def setup(**kwargs):
        assert broker.active == [("gpu", "load")]
        loads.append(kwargs)
        return {"A": staged_af2_double(Runner(), monkeypatch)}

    vendor(monkeypatch, "astrolabe.evaluators.wrappers.alphafold", alphafold_setup=setup)
    handler = native._af2_factory({"factory_kwargs": {"num_recycles": 7}, "oligo_names": ["A"]})
    original = {"nested": {"values": [1, 2]}}
    prepared = call(handler, dict(operation="process_features", features=original, random_seed=19))
    assert prepared["ok"] and original == {"nested": {"values": [1, 2]}}
    result = call(handler, dict(operation="predict", features=prepared["result"], random_seed=19))
    np.testing.assert_array_equal(result["result"]["plddt"], [1, 2, 4])
    assert result["result"]["plddt"].dtype == np.float32
    assert len(loads) == 1 and seeds == [19, 19]
    assert lifecycle(handler)["loader_count"] == 1
    assert lifecycle(handler)["residency"] == "resident_model"
    assert not broker.active


def test_af2_explicit_jax_device_matches_load_and_execute_admission(monkeypatch, broker):
    vendor(monkeypatch, "astrolabe.evaluators.utils", _ensure_alphafold2_import_path=lambda _: None)
    selected = []
    devices = ["TFRT_GPU_0", "TFRT_GPU_1", "TFRT_GPU_2"]

    @contextmanager
    def default_device(device):
        selected.append(device)
        try:
            yield
        finally:
            selected.pop()

    vendor(monkeypatch, "jax", local_devices=lambda **kwargs: devices, default_device=default_device)

    class Runner:
        def process_features(self, features, random_seed):
            assert selected == [] and broker.active[-1] == ("gpu", "prepare")
            return features

        def predict(self, features, random_seed):
            assert selected == ["TFRT_GPU_2"]
            return {"seed": random_seed}

    def setup(**kwargs):
        assert selected == ["TFRT_GPU_2"] and "gpu_device" not in kwargs
        return {"A": staged_af2_double(Runner(), monkeypatch)}

    vendor(monkeypatch, "astrolabe.evaluators.wrappers.alphafold", alphafold_setup=setup)
    handler = native._af2_factory({"oligo_names": ["A"], "factory_kwargs": {"gpu_device": 2}})
    assert call(handler, dict(operation="process_features", features={}, random_seed=7))["ok"]
    assert call(handler, dict(operation="predict", features={}, random_seed=7))["result"] == {"seed": 7}
    assert all(lease["device"] == "cuda:2" for lease in broker.leases if lease["resource"] == "gpu")
    assert all(not lease["concurrency_certified"] for lease in broker.leases)
    assert not broker.active and not selected


def test_af2_native_completion_precedes_confidence_and_preserves_alias(monkeypatch):
    from astrolabe.evaluators.af2_stages import evaluator_stages

    events = []
    runner = staged_af2_double(SimpleNamespace(
        multimer_mode=False,
        predict=lambda features, random_seed: {"value": features["value"] + random_seed},
    ), monkeypatch)
    array_type = type(runner.params)
    monkeypatch.setattr(array_type, "block_until_ready", lambda value: events.append("ready") or value)
    monkeypatch.setitem(globals(), "get_confidence_metrics", lambda result, multimer_mode: (
        events.append("confidence") or {"alias": result["value"]}
    ))
    stages = evaluator_stages(runner)
    params = runner.params
    prepared = stages.prepare({"operation": "predict", "features": {
        "value": np.asarray([1, 2], dtype=np.float32),
    }, "random_seed": 19})
    assert isinstance(prepared, Pending)
    prepared.completion()
    events.clear()
    raw = stages.execute(prepared.value)
    assert isinstance(raw, Pending) and events == []
    raw.completion()
    assert events == ["ready"]
    result = stages.finalize(raw.value)
    assert events == ["ready", "confidence"]
    assert runner.params is params and result["value"] is result["alias"]
    np.testing.assert_array_equal(result["value"], [20, 21])
    assert result["value"].dtype == np.float32
    runner.predict = lambda *args, **kwargs: None
    with pytest.raises(ValueError, match="Unsupported AF2 predict body"):
        evaluator_stages(runner)


def test_esmfold_unsupported_native_infer_is_rejected(monkeypatch, broker):
    loads = []

    class Model:
        def infer(self, sequences):
            assert broker.active == [("gpu", "execute")]
            assert sequences == ["ACD:EF"]
            return {"plddt": torch.tensor([0.75], dtype=torch.float16)}

        def output_to_pdb(self, output):
            assert broker.active == [("cpu", "execute")]
            assert output["plddt"].dtype == torch.float16
            return ["ATOM\n"]

    class Evaluator:
        def __init__(self, **kwargs):
            self._device = "cuda:0"

        def _ensure_model(self):
            loads.append(1)
            return Model()

    vendor(monkeypatch, "astrolabe.evaluators.esmfold",
           ESMFoldEvaluator=lambda **kwargs: staged_esmfold_double(Evaluator(**kwargs)))
    with pytest.raises(ValueError, match="Unsupported ESMFold infer body"):
        native._esmfold_factory({"model_name": "original", "device": "cpu"})
    assert loads == [1]
    assert not broker.active


def test_esmfold_confidence_reduction_precedes_host_materialization(monkeypatch, broker, esmfold_native):
    """The public adapter's reduction must execute under its device lease."""
    output = {
        "plddt": torch.arange(18, dtype=torch.float32).reshape(1, 3, 6),
        "chain_index": torch.tensor([[0, 0, -1]]),
        "atom37_atom_exists": torch.ones(1, 3, 6),
    }
    means = []
    original_mean = torch.Tensor.mean

    def observed_mean(tensor, *args, **kwargs):
        means.append(list(broker.active))
        return original_mean(tensor, *args, **kwargs)

    model = esmfold_native.Model()
    model.forward = lambda *args, **kwargs: dict(output)

    class Evaluator:
        _device = "cuda:0"

        def _ensure_model(self):
            return model

    vendor(monkeypatch, "astrolabe.evaluators.esmfold",
           ESMFoldEvaluator=lambda **kwargs: staged_esmfold_double(Evaluator()))
    handler = native._esmfold_factory({"model_name": "original", "device": "cpu"})
    with monkeypatch.context() as active:
        active.setattr(torch.Tensor, "mean", observed_mean)
        result = call(handler, {
            "operation": "infer", "sequences": ["A:B"],
            "return_residue_plddt": True,
        })["result"]
    assert len(means) == 1 and means[0][-1] == ("gpu", "finalize")
    torch.testing.assert_close(
        result["residue_plddt"][0], output["plddt"][0, :2].mean(-1),
        rtol=0, atol=0,
    )
    assert set(result["output"]) == set(output) | {"mean_plddt"}
    assert not broker.active


@pytest.fixture
def esmfold_native(monkeypatch, tmp_path):
    import importlib.util

    # Native infer body, with CPU-only vendor encoder/forward doubles. The
    # adapter must refuse modified bodies rather than silently use this split.
    source = '''
import torch
def batch_encode_sequences(sequences, offset, linker):
    audit("encode")
    assert sequences == ["A:B"] and offset == 512 and linker == "G" * 25
    return (torch.tensor([[1, 2, 3]]), torch.ones(1, 3), torch.tensor([[0, 512, 513]]),
            torch.tensor([[1., 1., 0.]]), torch.tensor([[0, 0, -1]]))
def collate_dense_tensors(values):
    return torch.stack(values)
def audit(name):
    pass
class Model:
    device = torch.device("cpu")
    def forward(self, aatype, **kwargs):
        audit("forward")
        assert kwargs["num_recycles"] is None and kwargs["masking_pattern"] is None
        return {"plddt": torch.rand(1, 3, 2), "atom37_atom_exists": torch.ones(1, 3, 2)}
    def output_to_pdb(self, output):
        return ["ATOM\\n"]
    @torch.no_grad()
    def infer(self, sequences, residx=None, masking_pattern=None, num_recycles=None,
              residue_index_offset=512, chain_linker="G" * 25):
        if isinstance(sequences, str):
            sequences = [sequences]
        aatype, mask, _residx, linker_mask, chain_index = batch_encode_sequences(
            sequences, residue_index_offset, chain_linker
        )
        if residx is None:
            residx = _residx
        elif not isinstance(residx, torch.Tensor):
            residx = collate_dense_tensors(residx)
        aatype, mask, residx, linker_mask = map(
            lambda x: x.to(self.device), (aatype, mask, residx, linker_mask)
        )
        output = self.forward(
            aatype, mask=mask, residx=residx, masking_pattern=masking_pattern,
            num_recycles=num_recycles,
        )
        output["atom37_atom_exists"] = output["atom37_atom_exists"] * linker_mask.unsqueeze(2)
        output["mean_plddt"] = (output["plddt"] * output["atom37_atom_exists"]).sum(
            dim=(1, 2)
        ) / output["atom37_atom_exists"].sum(dim=(1, 2))
        output["chain_index"] = chain_index
        return output
'''
    path = tmp_path / "fake_esmfold_native.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("fake_esmfold_native", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_esmfold_verified_native_body_splits_encoder_preserving_rng_and_reductions(monkeypatch, broker, esmfold_native):
    module = esmfold_native
    model = module.Model()
    torch.manual_seed(19)
    expected = model.infer(["A:B"])
    expected_rng = torch.get_rng_state().clone()

    def audit(name):
        assert broker.active[-1] == ("gpu", "prepare" if name == "encode" else "infer")

    module.audit = audit
    evaluator = staged_esmfold_double(SimpleNamespace(
        _device="cuda:0", _ensure_model=lambda: model,
    ))
    vendor(monkeypatch, "astrolabe.evaluators.esmfold", ESMFoldEvaluator=lambda **kwargs: evaluator)
    torch.manual_seed(19)
    handler = native._esmfold_factory({"model_name": "original", "device": "cpu"})
    actual = call(handler, dict(operation="infer", sequences=["A:B"]))
    assert actual["ok"]
    assert set(actual["result"]) == set(expected)
    for key, value in expected.items():
        assert torch.equal(actual["result"][key], value)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert handler({"operation": "describe"})["metadata"]["cpu_encoding"] == "verified_native_infer"
    assert lifecycle(handler)["stage_contract"] == "prepare_complete_model_finalize"
    assert lifecycle(handler)["pipeline_slots"] == 1
    assert call(handler, {"operation": "output_to_pdb", "output": actual["result"]})["result"] == ["ATOM\n"]
    assert not broker.active


def test_esmfold_event_wait_and_native_pdb_precede_host_copy(monkeypatch, esmfold_native):
    from astrolabe.evaluators.esmfold_stages import evaluator_stages

    events = []
    model = esmfold_native.Model()
    model.device = "cuda:2"
    stream = object()

    class Event:
        def record(self, actual):
            assert actual is stream
            events.append("record")

        def synchronize(self):
            events.append("wait")

    def transfer(value, device, *args, **kwargs):
        assert device == "cuda:2"
        events.append("transfer")
        return value

    def pdb(output):
        assert "mean_plddt" in output and "chain_index" in output
        events.append("pdb")
        return ["ATOM\n"]

    monkeypatch.setattr(torch.Tensor, "to", transfer)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: stream)
    monkeypatch.setattr(model, "output_to_pdb", pdb)
    monkeypatch.setattr(execution, "host_result", lambda value: events.append("host") or value)
    stages = evaluator_stages(model, device="cuda:2")
    prepared = stages.prepare({"operation": "infer", "sequences": ["A:B"],
                               "return_residue_plddt": True, "return_pdb": True})
    assert events == ["transfer"] * 4
    raw = stages.execute(prepared)
    assert isinstance(raw, Pending) and events[-1] == "record"
    assert "mean_plddt" not in raw.value[1] and "chain_index" not in raw.value[1]
    raw.completion()
    result = stages.finalize(raw.value)
    assert events == ["transfer"] * 4 + ["record", "wait", "pdb", "host"]
    assert result["pdbs"] == ["ATOM\n"] and len(result["residue_plddt"][0]) == 2
    with pytest.raises(ValueError, match="requires an infer request"):
        stages.prepare({"operation": "output_to_pdb", "output": {}, "return_pdb": True})


def test_esmfold_local_facade_uses_staged_model_without_whole_evaluation_lease(
        monkeypatch, broker, esmfold_native, tmp_path):
    from astrolabe.core import global_config
    from astrolabe.evaluators.esmfold import ESMFoldEvaluator

    model = esmfold_native.Model()
    forward = model.forward
    def with_scores(*args, **kwargs):
        output = forward(*args, **kwargs)
        output.update(ptm=torch.tensor([0.7]), predicted_aligned_error=torch.ones(1, 3, 3))
        return output
    model.forward = with_scores
    model.output_to_pdb = lambda output: ["MODEL\nENDMDL\n"]
    evaluator = ESMFoldEvaluator(model=model, device="cpu")
    evaluator.prepare_input = lambda inputs, config: inputs
    evaluator.prepare_output = lambda output, inputs, config: output
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("protomers", {name: SimpleNamespace(length=1) for name in "AB"})
    global_config.set_variable("step", 1)
    (tmp_path / "structures").mkdir()
    try:
        scores = evaluator.evaluate({"AB": "A:B"}, {})
        assert scores.complex_paths["AB"].read_text() == "MODEL\nENDMDL\n"
        assert scores.ptm["AB"][0] == pytest.approx(0.7)
        assert scores.pae["AB"][0].shape == (2, 2)
        assert all(lease["resource"] == "cpu" for lease in broker.leases)
        assert not any(lease["phase"] == "compute" for lease in broker.leases)
        assert {lease["phase"] for lease in broker.leases} >= {"prepare", "infer", "finalize"}
    finally:
        evaluator.shutdown()
        global_config.reset()


def test_boltz_native_featurize_cpu_predict_device_and_actual_load_once(monkeypatch, broker, tmp_path):
    loads = []

    class Runner:
        def __init__(self, config):
            assert config.diffusion_samples == 3

        def load_model_once(self):
            assert broker.active == [("gpu", "load") if not loads else ("gpu", "execute")]
            if not hasattr(self, "model_module"):
                loads.append(1)
                self.model_module = object()

        def featurize_data(self, data, out_dir):
            assert broker.active == [("cpu", "execute")]
            return [data, out_dir]

        def run_inference(self, data, out_dir):
            assert broker.active == [("gpu", "load") if not loads else ("gpu", "execute")]
            assert data == ["input", tmp_path]
            self.load_model_once()

    vendor(monkeypatch, "astrolabe.evaluators.wrappers.boltz", Boltz1Runner=Runner)
    handler = native._boltz_factory({"config": {"accelerator": "gpu", "diffusion_samples": 3}})
    assert not loads
    assert handler({"operation": "describe"})["metadata"]["model_state"]["model_load_count"] == 0
    data = call(handler, dict(operation="featurize_data", data="input", out_dir=tmp_path))["result"]
    for ordinal in range(2):
        response = call(handler, dict(operation="run_inference", data=data, out_dir=tmp_path))
        assert response["result"] == {"completed": True}
        assert response["runtime"]["scientific_requests"] == ordinal + 1
        assert bool(response["runtime"].get("first_request_executed_during_load")) is (ordinal == 0)
    assert loads == [1] and not broker.active
    assert handler({"operation": "describe"})["metadata"]["model_state"]["model_load_count"] == 1


def test_boltz_checkpoint_uses_configured_omegaconf_device_without_gpu_loading(monkeypatch, broker, tmp_path):
    pytest.importorskip("boltz")
    from astrolabe.evaluators.wrappers import boltz
    captured = []

    class Model:
        def eval(self):
            return self

    def load(checkpoint, **kwargs):
        captured.append(kwargs)
        return Model()

    monkeypatch.setattr(boltz.Boltz1, "load_from_checkpoint", load)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    runner = boltz.Boltz1Runner.__new__(boltz.Boltz1Runner)
    runner.accelerator = "gpu"
    runner.devices = OmegaConf.create([2])
    runner.checkpoint = tmp_path / "original.ckpt"
    runner.recycling_steps = 3
    runner.sampling_steps = 11
    runner.diffusion_samples = 2
    runner.write_full_pae = True
    runner.write_full_pde = False
    runner.load_model_once()
    runner.load_model_once()
    assert len(captured) == 1 and str(captured[0]["map_location"]) == "cuda:2"
    assert captured[0]["predict_args"]["diffusion_samples"] == 2


def test_protenix_separates_transfer_forward_and_native_tensor_dump(monkeypatch, broker, tmp_path):
    seen = []
    destinations = []

    class Runner:
        def __init__(self, config):
            self.configs = config
            self.device = "cpu"
            self.dump_dir = "unchanged"
            self.dumper = SimpleNamespace(dump=self.dump)

        def update_model_configs(self, config):
            seen.append(config.num_recycles)

        def prepare_data(self, data):
            assert broker.active[-1] == ("cpu", "prepare")
            return data

        def predict_prepared(self, data):
            assert broker.active[-1] == ("cpu", "infer")
            return {"coordinate": torch.ones(2, 3, dtype=torch.bfloat16)}

        def dump(self, **kwargs):
            assert broker.active[-1] == ("cpu", "finalize")
            assert kwargs["pred_dict"]["coordinate"].dtype == torch.bfloat16
            assert self.dumper.base_dir == self.dump_dir
            destinations.append(self.dump_dir)
            seen.append(kwargs["seed"])

    vendor(monkeypatch, "astrolabe.evaluators.wrappers.protenix", ProtenixRunner=Runner)
    handler = native._protenix_factory({"config": {"dtype": "bf16"}})
    for index in range(2):
        result = call(handler, dict(
            operation="predict_and_dump",
            configs={"num_recycles": 9, "dump_dir": str(tmp_path / str(index))},
            data={}, dataset_name="step", pdb_id="A", seed=17 + index,
            atom_array=None, entity_poly_type={},
        ))
        assert result["result"]["coordinate"].dtype == torch.bfloat16
    assert destinations == [str(tmp_path / str(index)) for index in range(2)]
    assert seen == [9, 17, 9, 18] and lifecycle(handler)["split_stages"]
    assert lifecycle(handler)["loader_count"] == 1
    assert not broker.active


class ChaiModule:
    __package__ = "fake_chai"
    __name__ = "fake_chai"
    __file__ = "fake_chai.py"

    def __init__(self):
        self.loads = []
        self.moves = []

    def load_exported(self, key, device):
        self.loads.append(key)
        return SimpleNamespace(jit_module=SimpleNamespace(to=lambda device: self.moves.append(str(device))))

    def run_inference(self, **kwargs):
        with self._component_moved_to("trunk", kwargs["device"]):
            if kwargs.get("fail"):
                raise ValueError("native component failure")
            return {"samples": kwargs["num_diffn_samples"]}


def test_chai_component_offload_on_failure_and_cache_reuse(monkeypatch, broker):
    from astrolabe.evaluators import x_masking
    from contextlib import nullcontext
    module = ChaiModule()
    vendor(monkeypatch, "astrolabe.evaluators.chai1", _load_chai1_module=lambda: module,
           _CHAI1_SUPPORTS_CONSTRAINTS=True, _CHAI1_MODULE_PATH="fake_chai.py")
    monkeypatch.setattr(x_masking, "patch_chai1_get_esm_embedding_context", lambda _: nullcontext())
    handler = native._chai1_factory({"model_config": {"device": "cpu"}})
    good = {"operation": "chai1", "kwargs": {"num_diffn_samples": 4}}
    assert call(handler, good)["result"] == {"samples": 4}
    failed = call(handler, {"operation": "chai1", "kwargs": {"num_diffn_samples": 4, "fail": True}})
    assert not failed["ok"] and module.moves[-1] == "cpu"
    assert call(handler, good)["result"] == {"samples": 4}
    assert module.loads == ["trunk"] and not broker.active
    assert lifecycle(handler)["failures"] == 1
    assert handler({"operation": "describe"})["metadata"]["component_state"] == {
        "cached_components": 1, "weights_loaded": True,
    }


def test_chai_failed_device_transfer_attempts_cpu_cleanup():
    module = ChaiModule()
    moves = []

    def move(device):
        moves.append(str(device))
        if str(device).startswith("cuda"):
            raise RuntimeError("transfer failed")

    module.load_exported = lambda key, device: SimpleNamespace(jit_module=SimpleNamespace(to=move))
    native._install_chai_component_cache(module, provider_identity="original")
    with pytest.raises(RuntimeError, match="transfer failed"):
        with module._component_moved_to("trunk", "cuda:1"):
            pytest.fail("failed transfer must not enter inference")
    assert moves == ["cuda:1", "cpu"]


def test_rhofold_cpu_inputs_native_forward_and_cpu_export(monkeypatch, broker, tmp_path):
    loads = []

    class Model:
        def __init__(self, config):
            self.structure_module = SimpleNamespace(converter=SimpleNamespace(export_pdb_file=self.export))

        def load_state_dict(self, state):
            loads.append(state)

        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, **kwargs):
            assert not torch.is_grad_enabled()
            assert broker.active == [("cpu", "execute")]
            return [{"coordinate": torch.zeros(2, 3, dtype=torch.float16)}]

        def export(self, sequence, coordinates, **kwargs):
            assert broker.active == [("cpu", "execute")]
            Path(kwargs["path"]).write_text(sequence)

    vendor(monkeypatch, "rhofold.config", rhofold_config=object())
    vendor(monkeypatch, "rhofold.rhofold", RhoFold=Model)
    monkeypatch.setattr(torch, "load", lambda *a, **k: {"model": "original-state"})
    handler = native._rhofold_factory({"config": {"checkpoint": "original", "device": "cpu"}})
    predicted = call(handler, dict(operation="forward", tokens=torch.ones(1, 2),
                                  rna_fm_tokens=torch.ones(1, 2), seq="AC"))["result"]
    assert predicted["coordinate"].dtype == torch.float16
    path = tmp_path / "rna.pdb"
    exported = call(handler, dict(operation="export_pdb", sequence="AC", coordinates=predicted["coordinate"],
                                 path=path, chain_id="A", confidence=np.ones(2)))
    assert exported["result"] == {"path": str(path)} and path.read_text() == "AC"
    assert loads == ["original-state"] and not broker.active


def af3_capacity_broker(monkeypatch, broker):
    """The AF3 stage contract borrows a real token from its memory lease."""
    original = broker.lease

    @contextmanager
    def lease(**fields):
        with original(**fields):
            yield SimpleNamespace(token="af3-request-capacity", device=fields.get("device"))

    monkeypatch.setattr(broker, "lease", lease)


def af3_factory(monkeypatch, broker, completion, *, config=None):
    from astrolabe.evaluators import af3_service
    af3_capacity_broker(monkeypatch, broker)

    dispatched = threading.Event()
    materialized = []

    def stages(key):
        def prepare(features):
            assert broker.active == [("gpu", "evaluation_memory"), ("gpu", "prepare")]
            return {"features": features}

        def upload(value):
            assert broker.active == [("gpu", "evaluation_memory"), ("gpu", "prepare")]
            return Pending(value, lambda: None)

        def dispatch(value):
            assert broker.active == [("gpu", "evaluation_memory"), ("gpu", "infer")]
            dispatched.set()
            return Pending(value, completion, keepalive=value["features"])

        def materialize(value):
            assert broker.active == [("gpu", "evaluation_memory"), ("gpu", "finalize")]
            materialized.append(key)
            return {"coordinates": np.asarray(value["features"], dtype=np.float32), "__identifier__": b"native"}

        return [Stage("af3_features", "cpu", prepare), Stage("af3_upload", "transfer", upload),
                Stage("af3_model", "gpu", dispatch), Stage("af3_materialize", "cpu", materialize)]

    loads = []
    runner = SimpleNamespace(_device="fake-jax-gpu", inference_stages=stages,
                             _model_config=SimpleNamespace(global_config=SimpleNamespace()))
    monkeypatch.setattr(af3_service, "load_runner", lambda config: loads.append(config) or runner)
    vendor(monkeypatch, "astrolabe.evaluators.wrappers.alphafold3", service_model_settings=lambda config: {"fixed": True})
    handler = native._af3_factory({"config": config or {}, "service_settings": {"fixed": True}})
    request = dict(model="alphafold3", settings={"fixed": True}, features=[1, 2], rng_key=7,
                   cyclic_asym_ids=[], cyclic_relative_patches=[])
    return handler, request, dispatched, materialized, loads


def test_af3_native_pending_keeps_inference_until_completion_and_capacity_until_finalize(monkeypatch, broker):
    completion = Future()
    handler, request, dispatched, materialized, loads = af3_factory(monkeypatch, broker, completion)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(call, handler, request)
        try:
            assert dispatched.wait(3)
            assert not pending.done() and not materialized
            assert broker.active == [("gpu", "evaluation_memory"), ("gpu", "infer")]
        finally:
            completion.set_result(None)
        result = pending.result(timeout=3)
    assert result["result"]["__identifier__"] == b"native"
    assert result["result"]["coordinates"].dtype == np.float32
    assert materialized == [7] and len(loads) == 1 and not broker.active
    invalid = call(handler, dict(request, settings={"fixed": False}))
    assert not invalid["ok"] and "settings" in invalid["error"]


def test_af3_native_pending_failure_releases_slot_without_materializing(monkeypatch, broker):
    completion = Future()
    completion.set_exception(RuntimeError("device failed"))
    handler, request, _, materialized, loads = af3_factory(monkeypatch, broker, completion)
    result = call(handler, request)
    assert not result["ok"] and "device failed" in result["error"]
    assert not materialized and not broker.active and len(loads) == 1
    assert lifecycle(handler)["failures"] == 1


def test_af3_configured_gpu_ordinal_is_admitted_for_load_and_execution(monkeypatch, broker):
    complete = Future()
    complete.set_result(None)
    handler, request, _, _, _ = af3_factory(monkeypatch, broker, complete, config={"gpu_device": 2})
    assert call(handler, request)["ok"]
    leases = [lease for lease in broker.leases if lease["resource"] == "gpu"]
    assert [lease["phase"] for lease in leases] == ["load", "evaluation_memory", "prepare", "infer", "finalize"]
    assert all(lease["device"] == "cuda:2" and not lease["concurrency_certified"] for lease in leases)


def test_af3_setup_uses_configured_visible_device_without_loading_model(monkeypatch, broker):
    pytest.importorskip("alphafold3")
    from astrolabe.evaluators.wrappers import alphafold3

    devices = [SimpleNamespace(platform="gpu", id=index) for index in range(3)]
    monkeypatch.setattr(alphafold3.jax, "local_devices", lambda **kwargs: devices)
    monkeypatch.setattr(alphafold3, "_configure_refstructure_env", lambda _: None)
    monkeypatch.setattr(alphafold3, "_configure_msa_env", lambda _: None)
    monkeypatch.setattr(alphafold3, "make_model_config", lambda **kwargs: kwargs)
    monkeypatch.setattr(alphafold3, "ModelRunner", lambda **kwargs: SimpleNamespace(**kwargs))
    config = OmegaConf.create(dict(gpu_device=2, model_dir="unchanged", flash_attention_implementation="cudnn",
                                   num_samples=3, num_recycles=7, save_embeddings=False))
    runner = alphafold3.alphafold3_setup(config)
    assert runner.device is devices[2]
    assert runner.config["num_diffusion_samples"] == 3 and runner.config["num_recycles"] == 7


@pytest.mark.parametrize("pipeline", [False, True])
def test_af3_service_entrypoint_admits_configured_device_and_reuses_nested_guard(monkeypatch, broker, pipeline):
    from astrolabe.evaluators import af3_service, af3_pipeline, af3_lifecycle
    from astrolabe.evaluators.runtime_stages import complete_stage
    af3_capacity_broker(monkeypatch, broker)
    results = []
    monkeypatch.setenv("ASTROLABE_RESOURCE_SOCKET", "/fake/broker")
    monkeypatch.setattr(af3_lifecycle, "guard_owner_parent", lambda: None)
    monkeypatch.setattr(af3_service, "load_service_config", lambda path: OmegaConf.create({"gpu_device": 2}))
    vendor(monkeypatch, "astrolabe.evaluators.wrappers.alphafold3", service_model_settings=lambda config: {})
    vendor(monkeypatch, "jax", tree=SimpleNamespace(map=lambda fn, result: {key: fn(value) for key, value in result.items()}))

    def run(features, key):
        assert broker.active == [("gpu", "evaluation_memory"), ("gpu", "infer")]
        return {"value": np.asarray(features)}

    runner = SimpleNamespace(
        _device="TFRT_GPU_2", _model_config=SimpleNamespace(global_config=SimpleNamespace()),
        set_cyclic_asym_ids=lambda ids: None, set_cyclic_relative_patches=lambda patches: None,
        run_inference=run, inference_stages=lambda key: [
            Stage("af3_model", "gpu", lambda features: run(features, key)),
            Stage("af3_materialize", "cpu", lambda result: result),
        ],
    )
    monkeypatch.setattr(af3_service, "load_runner", lambda config: runner)
    request = dict(model="alphafold3", settings={}, features=[1, 2], rng_key=7,
                   cyclic_asym_ids=[], cyclic_relative_patches=[])

    def serve_pipeline(actual_runner, config, socket, pending, *, device_kernel):
        item = dict(payload=request, request_id=1)
        for stage in af3_pipeline.stage_factory(actual_runner, config, device_kernel=device_kernel)(request):
            item = complete_stage(stage, item)
        results.append(item["value"])

    monkeypatch.setattr(af3_pipeline, "serve_pipeline", serve_pipeline)
    monkeypatch.setattr(af3_service, "serve", lambda handler, socket, pending: results.append(handler(request)))
    arguments = ["--socket", "unused", "--config", "unused"] + (["--pipeline"] if pipeline else [])
    af3_service.main(arguments)
    np.testing.assert_array_equal(results[0]["value"], [1, 2])
    leases = [lease for lease in broker.leases if lease["resource"] == "gpu"]
    assert [lease["phase"] for lease in leases] == ["load", "evaluation_memory", "prepare", "infer", "finalize"]
    assert all(lease["device"] == "cuda:2" for lease in leases) and not broker.active


def test_unrepresented_multi_device_backends_fail_before_loading(monkeypatch, broker):
    from astrolabe.evaluators.wrappers.opendde_daemon import create_handler
    monkeypatch.setenv("ASTROLABE_RESOURCE_SOCKET", "/fake/broker")
    with pytest.raises(ValueError, match="multi-device admission"):
        native._boltz_factory({"config": {"accelerator": "gpu", "devices": [0, 1]}})
    with pytest.raises(ValueError, match="multi-device admission"):
        create_handler({"foldcp_size_cp": 2})
    assert broker.leases == []


def test_esmfold2_existing_worker_is_admitted_without_changing_protocol(monkeypatch, broker, tmp_path):
    from astrolabe.evaluators.wrappers import esmfold2
    loads = []
    identity = {"model_name": "fake", "device": "cpu"}

    class Builder:
        def prepare_input(self, prediction_input, *, seed, device):
            assert broker.active[-1] == ("cpu", "prepare")
            assert seed == 19 and device == "cpu"
            assert prediction_input == {"sequences": [{"id": "A", "sequence": "AC"}]}
            return {"feature": "native feature"}, ["A"]

        def decode(self, output, features, chain_infos, **kwargs):
            assert broker.active[-1] == ("cpu", "finalize")
            assert output == "native output" and features == {"feature": "native feature"}
            assert chain_infos == ["A"]
            assert kwargs == {"complex_id": "A", "num_diffusion_samples": 3}
            return SimpleNamespace(complex=SimpleNamespace(to_mmcif=lambda: "native cif"),
                                   plddt=torch.tensor([0.8]), ptm=torch.tensor(0.7), iptm=None)

    class Model:
        device = "cpu"

        def __call__(self, *, feature, num_diffusion_samples):
            assert broker.active[-1] == ("cpu", "infer")
            assert feature == "native feature" and num_diffusion_samples == 3
            assert not torch.is_grad_enabled()
            return "native output"

    vendor(monkeypatch, "esm.models.esmfold2", ProteinInput=lambda **kw: kw,
           StructurePredictionInput=lambda **kw: kw)
    # Admission/protocol double only; the wrapper suite checks Biohub's real RNG scope.
    vendor(monkeypatch, "esm.models.esmfold2.processor", _seed_context=lambda seed: nullcontext())

    def load(settings):
        loads.append(1)
        return esmfold2._ESMFold2Worker(Model(), Builder(), identity, "cpu")

    monkeypatch.setattr(esmfold2, "_load_worker", load)
    worker = esmfold2.daemon_factory({"identity": identity})
    payload = dict(model_identity=identity, output_dir=str(tmp_path), fold_kwargs={"seed": 19, "num_diffusion_samples": 3},
                   jobs=[{"name": "A", "chains": [{"id": "A", "sequence": "AC"}]}])
    for _ in range(2):
        result = worker(payload)
        assert set(result) == {"result", "stdout", "stderr"}
        assert result["result"]["jobs"][0]["name"] == "A"
    assert (tmp_path / "A_model.cif").read_text() == "native cif"
    assert loads == [1] and worker.metadata["runtime_lifecycle"]["residency"] == "resident_model"
    phases = [lease["phase"] for lease in broker.leases if lease["phase"] != "load"]
    assert phases == ["evaluation_memory", "prepare", "infer", "finalize"] * 2
    with pytest.raises(ValueError, match="owner"):
        worker(dict(payload, model_identity={"different": True}))
    assert not broker.active


def test_rf3_resident_engine_preserves_run_kwargs_rng_and_failure_cleanup(broker):
    loads = []
    runs = []

    class Engine:
        def __init__(self, **config):
            loads.append(config)

        def initialize(self):
            assert broker.active == [("gpu", "load")]

        def run(self, **kwargs):
            assert broker.active == [("gpu", "execute")]
            runs.append((kwargs, random.random()))
            if kwargs.get("fail"):
                raise ValueError("native run failure")

    engine, kernel = resident_engine_kernel(Engine, {"num_samples": 3})
    random.seed(19)
    expected = random.random()
    random.seed(19)
    kernel({"template_selection": ["A"], "seed": 17})
    with pytest.raises(ValueError, match="native run failure"):
        kernel({"fail": True})
    assert isinstance(engine, Engine) and len(loads) == 1
    assert runs[0] == ({"template_selection": ["A"], "seed": 17}, expected)
    assert kernel.metadata["runtime_lifecycle"]["failures"] == 1 and not broker.active


def test_rf3_legacy_daemon_reuses_engine_and_recovers_without_broker_or_pythonpath(tmp_path):
    package = tmp_path / "vendor" / "src" / "rf3" / "inference_engines"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").touch()
    (package / "__init__.py").touch()
    (package / "rf3.py").write_text('''
import json, random
from pathlib import Path
class RF3InferenceEngine:
    def __init__(self, audit_path, num_samples):
        assert num_samples == 3
        self.audit_path = Path(audit_path)
    def record(self, value):
        with self.audit_path.open("a") as stream:
            stream.write(json.dumps(value) + "\\n")
    def initialize(self):
        self.record({"loaded": True})
    def run(self, **kwargs):
        self.record({"random": random.random(), "template_selection": kwargs["template_selection"]})
        if kwargs["template_selection"] == ["fail"]:
            raise ValueError("native failure")
        Path(kwargs["out_dir"], "output.cif").write_text("native cif")
''')
    script = Path(native.__file__).parent / "wrappers" / "rf3_daemon.py"
    audit = tmp_path / "audit.jsonl"
    config = dict(repo_path=str(tmp_path / "vendor"), audit_path=str(audit), num_samples=3)
    request = dict(type="infer", config=config, prepared_input_path=str(tmp_path / "input.json"),
                   output_dir=str(tmp_path))
    messages = [dict(request, id="one", template_selection=["A"]),
                dict(request, id="two", template_selection=["B"]),
                dict(request, id="three", template_selection=["fail"]),
                dict(request, id="four", template_selection=["C"]), {"type": "shutdown"}]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    env.pop("PYTHONPATH", None)
    env.pop("ASTROLABE_RESOURCE_SOCKET", None)
    result = subprocess.run([sys.executable, str(script)], cwd=tmp_path, env=env,
                            input="".join(json.dumps(message) + "\n" for message in messages),
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    replies = [json.loads(line) for line in result.stdout.splitlines()]
    assert [reply["status"] for reply in replies[:-1]] == ["success", "success", "failed", "success"]
    records = [json.loads(line) for line in audit.read_text().splitlines()]
    assert records[0] == {"loaded": True} and len(records) == 5
    assert records[1]["random"] == records[2]["random"] == records[3]["random"] == records[4]["random"]
    assert (tmp_path / "output.cif").read_text() == "native cif"


def test_legacy_local_evaluate_holds_one_lease_but_rpc_client_holds_none(broker):
    from astrolabe.evaluators.base import Evaluator

    class Example(Evaluator):
        def __init__(self, handle):
            self.model_runner = handle

        def prepare_input(self, inputs, runtime_config):
            return inputs

        @local_evaluator_guard("test", "model_runner")
        def run_function(self, value):
            return value

        def prepare_output(self, value, inputs, runtime_config):
            assert len(broker.active) == (0 if hasattr(self.model_runner, "_client") else 1)
            return value

    assert Example(object()).evaluate("original") == "original"
    assert broker.events == [("enter", ("gpu", "compute")), ("exit", ("gpu", "compute"))]
    broker.events.clear()
    assert Example(SimpleNamespace(_client=object())).evaluate("rpc") == "rpc"
    assert broker.events == []


def test_load_failure_releases_admission(broker):
    def loader(settings):
        raise RuntimeError("bad checkpoint")

    with pytest.raises(RuntimeError, match="bad checkpoint"):
        DirectKernel({}, loader, lifecycle={"backend": "test", "residency": "one_shot"})
    assert not broker.active and not broker.models


def test_opendde_existing_factory_reuses_runner_and_quarantines_failed_request(monkeypatch, broker, tmp_path):
    from contextlib import nullcontext
    from astrolabe.evaluators.wrappers import opendde_daemon
    import logging
    loaded = []
    seen = []
    events = []
    completion = Future()
    dispatched = threading.Event()
    runner = SimpleNamespace(
        configs=OmegaConf.create({"seeds": [], "dtype": "fp32", "deterministic": False}), device="cuda:0",
        foldcp_config=SimpleNamespace(enabled=False),
        foldcp_recorder=SimpleNamespace(jsonl_path=""),
    )

    class Event:
        def record(self, stream):
            self.infer = ("gpu", "infer") in broker.active

        def synchronize(self):
            if self.infer:
                completion.result(timeout=10)
                events.append("completed")

    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: object())
    monkeypatch.setattr(torch.cuda, "Event", Event)

    def dump(**kwargs):
        assert completion.done() and events[-1] == "completed"
        assert broker.active == [("gpu", "evaluation_memory"), ("gpu", "finalize")]
        path = Path(runner.dump_dir) / "job" / "seed_11" / "predictions"
        path.mkdir(parents=True)
        (path / "job_summary_confidence_sample_0.json").write_text("{}")
        (path / "job_sample_0.cif").write_text("native")

    def init_dumper(**kwargs):
        runner.dumper = SimpleNamespace(dump=dump)

    runner.init_dumper = init_dumper

    def get_default_runner(**kwargs):
        loaded.append(kwargs)
        assert broker.active == [("gpu", "load")]
        return runner

    def preprocess(input_json, **kwargs):
        assert not broker.active
        seen.append((input_json, kwargs["out_dir"]))
        return input_json

    def predict(**kwargs):
        assert broker.active == [("gpu", "evaluation_memory"), ("gpu", "infer")]
        if len(seen) == 3:
            raise RuntimeError("vendor failed")
        dispatched.set()
        return {"raw": torch.tensor([1.0])}, None, None

    def to_device(data, device):
        assert broker.active == [("gpu", "evaluation_memory"), ("gpu", "prepare")]
        return data

    class Loader:
        dataset = [1]

        def __len__(self):
            return 1

        def __iter__(self):
            assert broker.active == [("gpu", "evaluation_memory"), ("gpu", "prepare")]
            data = dict(sample_name="job", sample_index=0, input_feature_dict={}, entity_poly_type={})
            data.update({key: torch.tensor([1]) for key in ("N_asym", "N_token", "N_atom", "N_msa")})
            yield [(data, None, "")]

    runner.model = predict
    runner.update_model_configs = lambda config: None
    monkeypatch.setattr(opendde_daemon, "verify_vendor", lambda _: tmp_path)
    vendor(monkeypatch, "runner.batch_inference", __file__=str(tmp_path / "runner" / "batch_inference.py"),
           get_default_runner=get_default_runner, preprocess_input=preprocess)
    vendor(monkeypatch, "runner.inference", __file__=str(tmp_path / "runner" / "inference.py"),
           torch=torch, logger=logging.getLogger("opendde"), DIST_WRAPPER=SimpleNamespace(rank=0),
           to_device=to_device, infer_n_token=lambda data: 1,
           update_inference_configs=lambda configs, n_token: configs,
           get_inference_dataloader=lambda **kwargs: Loader(), seed_everything=lambda **kwargs: None,
           cleanup_cuda_memory=lambda **kwargs: None, disable_cudnn_benchmark=nullcontext,
           measure_foldcp_stage=lambda **kwargs: nullcontext())
    handler = opendde_daemon.create_handler({"repo_path": str(tmp_path), "num_samples": 1})
    (tmp_path / "input.json").write_text(json.dumps([{"name": "job", "modelSeeds": [11]}]))
    payload = {"input_json": str(tmp_path / "input.json"), "out_dir": str(tmp_path / "one")}
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(handler, payload)
        try:
            assert dispatched.wait(10)
            assert not result.done() and ("gpu", "infer") in broker.active
            assert not (tmp_path / "one" / "job").exists()
        finally:
            completion.set_result(None)
        assert result.result(10) == dict(input_json=payload["input_json"], out_dir=payload["out_dir"])
    handler(dict(payload, out_dir=str(tmp_path / "two")))
    with pytest.raises(RuntimeError, match="vendor failed"):
        handler(payload)
    with pytest.raises(RuntimeError, match="quarantined"):
        handler(payload)
    assert len(loaded) == 1 and len(seen) == 3 and not broker.active
    assert (tmp_path / "two" / "job" / "seed_11" / "predictions" / "job_sample_0.cif").read_text() == "native"
    assert handler.metadata["runtime_lifecycle"]["failures"] == 1
    phases = [entry["phase"] for entry in broker.leases if entry["resource"] == "gpu"]
    assert phases[:5] == ["load", "evaluation_memory", "prepare", "infer", "finalize"]


def test_esmfold2_cli_forwards_broker_and_child_owns_stage_admission(monkeypatch, broker, tmp_path):
    from astrolabe.evaluators.wrappers.esmfold2 import ESMFold2Runner
    from astrolabe.evaluators.wrappers import esmfold2

    def run(command, **kwargs):
        assert broker.active == [] and broker.leases == []
        assert kwargs["env"]["ASTROLABE_RESOURCE_SOCKET"] == "/fake/broker"
        payload = json.loads(Path(command[command.index("--input-json") + 1]).read_text())
        assert payload["device"] == "cuda:2"
        output = Path(command[command.index("--output-json") + 1])
        output.write_text(json.dumps({"jobs": []}))
        return SimpleNamespace(returncode=0, stdout="native", stderr="")

    monkeypatch.setenv("ASTROLABE_RESOURCE_SOCKET", "/fake/broker")
    monkeypatch.setattr(esmfold2.subprocess, "run", run)
    runner = ESMFold2Runner(execution_mode="subprocess", python_bin=sys.executable, device="cuda:2")
    assert runner.run([], tmp_path) == {"jobs": []}
    assert broker.leases == [] and not broker.active


def test_af3_existing_pipeline_with_broker_separates_infer_and_d2h(monkeypatch, broker):
    from astrolabe.evaluators.af3_pipeline import stage_factory
    af3_capacity_broker(monkeypatch, broker)
    completion = Future()
    dispatched = threading.Event()
    materialized = []
    monkeypatch.setenv("ASTROLABE_RESOURCE_SOCKET", "/fake/broker")
    vendor(monkeypatch, "astrolabe.evaluators.wrappers.alphafold3", service_model_settings=lambda config: {})

    def dispatch(value):
        assert broker.active == [("gpu", "evaluation_memory"), ("gpu", "infer")]
        dispatched.set()
        return Pending(value, completion)

    def materialize(value):
        assert broker.active == [("gpu", "evaluation_memory"), ("gpu", "finalize")]
        materialized.append(value)
        return value + 1

    runner = SimpleNamespace(_device="fake-jax-gpu", _model_config=SimpleNamespace(global_config=SimpleNamespace()),
                             inference_stages=lambda key: [Stage("af3_model", "gpu", dispatch),
                                                           Stage("af3_materialize", "cpu", materialize)])
    request = dict(model="alphafold3", settings={}, cyclic_asym_ids=[], cyclic_relative_patches=[],
                   rng_key=17, features=3)
    stages = stage_factory(runner, object())(request)
    item = dict(request_id=1, payload=request)
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(stages[0].run, item)
        try:
            assert dispatched.wait(3) and not result.done()
            assert broker.active == [("gpu", "evaluation_memory"), ("gpu", "infer")] and not materialized
        finally:
            completion.set_result(None)
        assert result.result(timeout=3)["value"] == 4
    assert materialized == [3] and not broker.active


def af3_private_protocol(monkeypatch, broker):
    """Public facade -> real factory/kernel protocol -> original sync adapter."""
    from astrolabe.evaluators import af3_service
    af3_capacity_broker(monkeypatch, broker)
    loaded = []

    def tree_map(function, value):
        if isinstance(value, dict):
            return {key: tree_map(function, item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return type(value)(tree_map(function, item) for item in value)
        return function(value)

    vendor(monkeypatch, "jax", tree=SimpleNamespace(map=tree_map))
    vendor(monkeypatch, "astrolabe.evaluators.wrappers.alphafold3",
           service_model_settings=lambda config: {"fixed": True})

    def draw():
        return np.array([random.random(), np.random.random(), torch.rand(()).item()])

    class Runner:
        def __init__(self, config):
            assert broker.active == [("gpu", "load")]
            self._device = "TFRT_GPU_0"
            self._model_config = SimpleNamespace(global_config=SimpleNamespace(
                cyclic_asym_ids=tuple(config.get("bound_ids", [])),
                cyclic_relative_patches=tuple(tuple(patch) for patch in config.get("bound_patches", [])),
            ))
            self.weights = object()
            self.initial_draw = draw()
            self.stage_builds = 0
            self.calls = []
            self.staged = False
            loaded.append(self)

        def set_cyclic_asym_ids(self, ids):
            assert broker.active == [("gpu", "evaluation_memory"), ("gpu", "infer")]
            self._model_config.global_config.cyclic_asym_ids = tuple(int(value) for value in ids)

        def set_cyclic_relative_patches(self, patches):
            assert broker.active == [("gpu", "evaluation_memory"), ("gpu", "infer")]
            self._model_config.global_config.cyclic_relative_patches = tuple(tuple(int(value) for value in patch) for patch in patches)

        def run_inference(self, features, rng_key):
            assert broker.active == [("gpu", "evaluation_memory"), ("gpu", "infer")]
            config = self._model_config.global_config
            self.calls.append((config.cyclic_asym_ids, config.cyclic_relative_patches,
                               "staged" if self.staged else "monolithic", self.weights))
            values = draw()
            if features.get("fail", False):
                raise ValueError("fake forward failure")
            return dict(random=values, cyclic=np.asarray(config.cyclic_asym_ids, dtype=np.int32),
                        patches=np.asarray(config.cyclic_relative_patches, dtype=np.int32),
                        features=np.asarray(features["value"], dtype=np.float32),
                        rng_key=np.asarray(rng_key), __identifier__=b"native")

        def extract_structures(self, batch, result, target_name):
            from astrolabe.evaluators.wrappers.alphafold3 import ModelRunner
            return ModelRunner.extract_structures(self, batch, result, target_name)

        def inference_stages(self, key):
            self.stage_builds += 1

            def prepare(features):
                assert broker.active == [("gpu", "evaluation_memory"), ("gpu", "prepare")]
                return features

            def execute(features):
                self.staged = True
                try:
                    return self.run_inference(features, key)
                finally:
                    self.staged = False

            return (Stage("af3_features", "cpu", prepare), Stage("af3_model", "gpu", execute),
                    Stage("af3_materialize", "cpu", lambda result: result))

    monkeypatch.setattr(af3_service, "load_runner", Runner)

    class ProtocolClient:
        def __init__(self, backend, factory, settings, **kwargs):
            assert factory == "astrolabe.evaluators.native_daemons:_af3_factory"
            settings = dict(settings, _initial_random_state=execution.capture_random_state())
            # A supplied profile cannot certify unqualified AF3 inference.
            settings["_runtime_profile"] = {"identity": "alphafold3", "concurrency_certified": True}
            self.handler = native._af3_factory(settings)
            description = self.handler({"operation": "describe"})
            self.metadata = description["metadata"]
            execution.restore_random_state(description["random_state"])

        def request(self, payload):
            response = call(self.handler, execution.host_result(payload))
            execution.restore_random_state(response["random_state"])
            if not response["ok"]:
                raise RuntimeError(f"{response['error_type']}: {response['error']}")
            return response["result"]

    monkeypatch.setattr(native, "_KernelClient", ProtocolClient)
    return af3_service, loaded


def test_af3_remote_seed_and_extraction_do_not_initialize_caller_jax(monkeypatch):
    requests = []
    # Backend enumeration and PRNG generation are deliberately unavailable in
    # this coordinator. Tree traversal alone does not initialize a device.
    vendor(monkeypatch, "jax", tree=SimpleNamespace(map=lambda f, d: {
        key: f(value) for key, value in d.items()}))
    runner = native.NativeAF3Runner.__new__(native.NativeAF3Runner)
    runner._client = SimpleNamespace(request=lambda request: requests.append(request))
    runner._settings = {"fixed": True}
    runner._cyclic_asym_ids = ()
    runner._cyclic_relative_patches = ()
    runner.run_inference_for_seed({"value": np.ones(3)}, 17)
    runner.extract_structures({"batch": 1}, {"result": 2}, "target")
    assert requests[0]["rng_seed"] == 17 and "rng_key" not in requests[0]
    assert requests[1] == dict(operation="extract_structures", batch={"batch": 1},
                               result={"result": 2}, target_name="target")


def test_af3_seed_key_and_native_extraction_belong_to_loaded_owner(monkeypatch, broker):
    _, loaded = af3_private_protocol(monkeypatch, broker)
    jax = sys.modules["jax"]

    @contextmanager
    def cpu_scope(device):
        assert device == "cpu" and broker.active == [("gpu", "evaluation_memory"), ("gpu", "prepare")]
        assert len(loaded) == 1
        yield

    jax.local_devices = lambda backend: ["cpu"] if backend == "cpu" else []
    jax.default_device = cpu_scope
    jax.random = SimpleNamespace(PRNGKey=lambda seed: np.asarray([0, seed], dtype=np.uint32))
    def extract(owner, batch, result, target_name):
        assert owner is loaded[0] and broker.active == [("gpu", "evaluation_memory"), ("gpu", "finalize")]
        return [{"batch": batch, "result": result, "target": target_name}]
    sys.modules["astrolabe.evaluators.wrappers.alphafold3"].ModelRunner = SimpleNamespace(
        extract_structures=extract)
    runner = native.NativeAF3Runner(OmegaConf.create({}))
    initial = execution.capture_random_state()
    seeded = runner.run_inference_for_seed({"value": [1, 2]}, 17)
    execution.restore_random_state(initial)
    explicit = runner.run_inference({"value": [1, 2]}, np.asarray([0, 17], dtype=np.uint32))
    for key in explicit:
        np.testing.assert_equal(seeded[key], explicit[key])
    result = runner.extract_structures({"x": 1}, {"y": 2}, "A")
    assert result == [{"batch": {"x": 1}, "result": {"y": 2}, "target": "A"}]
    assert len(loaded) == 1 and not broker.active


def test_native_af3_cyclic_mismatch_rejects_without_mutation_or_fallback(monkeypatch, broker):
    _, loaded = af3_private_protocol(monkeypatch, broker)
    runner = native.NativeAF3Runner(OmegaConf.create({}))
    runner.set_cyclic_asym_ids([1])
    runner.set_cyclic_relative_patches([[1, 2]])
    with pytest.raises(RuntimeError, match="immutable owner cyclic"):
        runner.run_inference({"value": [1, 2]}, np.asarray([0, 7], dtype=np.uint32))
    model = loaded[0]
    assert model.stage_builds == 0 and model.calls == []
    assert model._model_config.global_config.cyclic_asym_ids == ()
    metadata = runner._client.handler({"operation": "describe"})["metadata"]
    assert metadata["owner_configuration"] == "immutable_cyclic"
    assert "cyclic_fallback" not in metadata and not broker.active


def test_native_af3_private_owners_isolate_cyclic_state_weights_and_rng(monkeypatch, broker):
    _, loaded = af3_private_protocol(monkeypatch, broker)
    owners, states = [], []
    for seed in (101, 202):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        owners.append(native.NativeAF3Runner(OmegaConf.create({"bound_ids": [len(owners) + 1]})))
        states.append(execution.capture_random_state())
    models = loaded[:]
    assert models[0]._model_config is not models[1]._model_config
    assert models[0].weights is not models[1].weights
    reference = [random.Random(seed) for seed in (101, 202)]
    for generator in reference:
        generator.random()  # constructor's original initialization draw
    for owner_index, ids in [(0, [1]), (1, [2]), (0, [1]), (1, [2])]:
        execution.restore_random_state(states[owner_index])
        owner = owners[owner_index]
        owner.set_cyclic_asym_ids(ids)
        owner.set_cyclic_relative_patches([])
        features = {"value": [owner_index, 7]}
        result = owner.run_inference(features, np.asarray([0, 11], dtype=np.uint32))
        states[owner_index] = execution.capture_random_state()
        np.testing.assert_array_equal(result["cyclic"], ids)
        np.testing.assert_array_equal(result["features"], features["value"])
        assert result["random"][0] == reference[owner_index].random()
        if owner_index == 0:
            assert models[1]._model_config.global_config.cyclic_asym_ids == (2,)
    assert [model.stage_builds for model in models] == [2, 2]
    assert [len(model.calls) for model in models] == [2, 2]
    assert all(lifecycle(owner._client.handler)["loader_count"] == 1 for owner in owners)
    assert len(loaded) == 2 and not broker.active


def test_native_af3_compatible_prebound_cyclic_owner_retains_stages(monkeypatch, broker):
    _, loaded = af3_private_protocol(monkeypatch, broker)
    runner = native.NativeAF3Runner(OmegaConf.create({"bound_ids": [1], "bound_patches": [[1, 2]]}))
    for ids, patches in [([1], [[1, 2]]), ([1], [[1, 2]])]:
        runner.set_cyclic_asym_ids(ids)
        runner.set_cyclic_relative_patches(patches)
        result = runner.run_inference({"value": [1, 2]}, np.asarray([0, 7], dtype=np.uint32))
        np.testing.assert_array_equal(result["cyclic"], ids)
        np.testing.assert_array_equal(result["patches"], patches)
    assert [call[2] for call in loaded[0].calls] == ["staged", "staged"]
    assert lifecycle(runner._client.handler)["loader_count"] == 1 and len(loaded) == 1
    runner.set_cyclic_asym_ids([2])
    with pytest.raises(RuntimeError, match="immutable owner cyclic"):
        runner.run_inference({"value": [1, 2]}, np.asarray([0, 7], dtype=np.uint32))
    assert loaded[0]._model_config.global_config.cyclic_asym_ids == (1,)
    assert not broker.active


@pytest.mark.parametrize("device,resource,selected_device", [
    ("cuda:2", "gpu", "TFRT_GPU_2"), ("cpu", "cpu", "TFRT_CPU_0"),
])
def test_af2_merger_uses_existing_owner_admission_and_materializes_in_scope(
        monkeypatch, broker, device, resource, selected_device):
    """Synthetic GPU lease/device only: no real device or model is used."""
    vendor(monkeypatch, "astrolabe.evaluators.utils", _ensure_alphafold2_import_path=lambda _: None)
    selected, loads, operations = [], [], []

    @contextmanager
    def default_device(value):
        selected.append(value)
        try:
            yield
        finally:
            selected.pop()

    vendor(monkeypatch, "jax", local_devices=lambda backend: (
        ["TFRT_CPU_0"] if backend == "cpu" else ["TFRT_GPU_0", "TFRT_GPU_1", "TFRT_GPU_2"]),
        default_device=default_device)

    class DeviceArray:
        __module__ = "jaxlib.cpu_test_double"
        shape = (3,)
        def __array__(self, dtype=None, **kwargs):
            assert broker.active[-1] == (resource, "finalize")
            assert selected == [selected_device]
            operations.append("host_copy")
            return np.arange(3, dtype=np.float32)

    def merge(*, all_chain_features):
        assert broker.active[-1] == (resource, "prepare")
        assert selected == [selected_device]
        assert all_chain_features == {"A": {"marker": 1}}
        operations.append("merge")
        return {"aatype": DeviceArray()}

    vendor(monkeypatch, "alphafold.data", feature_processing=SimpleNamespace(pair_and_merge=merge))

    class Runner:
        def process_features(self, features, random_seed):
            assert selected == [] and broker.active[-1] == (resource, "prepare")
            assert random_seed == 17
            operations.append("process")
            return features
        def predict(self, features, random_seed):
            assert selected == [selected_device] and broker.active[-1] == (resource, "infer")
            assert random_seed == 17
            operations.append("predict")
            return features

    def setup(**kwargs):
        assert broker.active == [(resource, "load")] and selected == [selected_device]
        loads.append(kwargs)
        return {"A": staged_af2_double(Runner(), monkeypatch)}

    vendor(monkeypatch, "astrolabe.evaluators.wrappers.alphafold", alphafold_setup=setup)
    handler = native._af2_factory({"oligo_names": ["A"], "factory_kwargs": {"device": device}})
    for _ in (1, 2):
        merged = call(handler, {"operation": "pair_and_merge", "all_chain_features": {"A": {"marker": 1}}})
        assert merged["ok"] and isinstance(merged["result"]["aatype"], np.ndarray)
        assert merged["result"]["aatype"].dtype == np.float32
        processed = call(handler, {"operation": "process_features", "features": merged["result"], "random_seed": 17})
        predicted = call(handler, {"operation": "predict", "features": processed["result"], "random_seed": 17})
        np.testing.assert_array_equal(predicted["result"]["aatype"], np.arange(3, dtype=np.float32))
    assert len(loads) == 1 and lifecycle(handler)["loader_count"] == 1
    assert operations == ["merge", "host_copy", "process", "predict"] * 2
    from astrolabe.evaluators.af2_stages import staged_pair_and_merge

    local_runner = staged_af2_double(Runner(), monkeypatch)
    local_runner._astrolabe_device_hint = device
    merged = staged_pair_and_merge(
        local_runner, all_chain_features={"A": {"marker": 1}},
    )
    np.testing.assert_array_equal(merged["aatype"], np.arange(3, dtype=np.float32))
    assert operations[-2:] == ["merge", "host_copy"]
    local_runner._astrolabe_staged_kernel.close()
    assert not broker.active and not selected
