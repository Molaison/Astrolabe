"""CPU contract checks; vendor stand-ins do not qualify real GPU parity."""
from contextlib import contextmanager
import inspect
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from astrolabe import model_execution as execution
from astrolabe.mutators import native_daemons
from astrolabe.mutators.runtime_stages import device_guard, protocol_infer


class Ledger:
    def __init__(self):
        self.socket_path = "/mock-mutator-broker"
        self.active = []
        self.events = []

    @contextmanager
    def lease(self, **spec):
        assert not self.active, "nested mutator admission would deadlock"
        self.active.append(spec)
        self.events.append(("enter", spec["resource"], spec["phase"]))
        try:
            yield spec
        finally:
            self.events.append(("leave", spec["resource"], spec["phase"]))
            self.active.pop()

    def register_model(self, *args, **kwargs):
        pass

    def phase(self, name, resource=None):
        assert self.active[-1]["phase"] == name
        if resource is not None:
            assert self.active[-1]["resource"] == resource


@pytest.fixture
def ledger(monkeypatch):
    ledger = Ledger()
    monkeypatch.setattr(execution, "_resource_client", lambda: ledger)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    # No test may initialize CUDA, including factories declaring auto.
    monkeypatch.setattr(torch.cuda, "init", lambda: pytest.fail("CUDA initialized"))
    yield ledger
    assert ledger.active == []


def infer(handler, payload):
    return handler({"operation": "infer", "payload": payload,
                    "random_state": execution.capture_random_state()})


@pytest.mark.parametrize("backend", ["ligandmpnn", "carbondesign"])
@pytest.mark.parametrize("failure", [None, "load", "prepare", "execute", "finalize"])
def test_actual_sequence_factories_release_and_load_once(monkeypatch, ledger, backend, failure):
    loads = []

    class Runner:
        def __init__(self, _settings):
            ledger.phase("load", "cpu")
            loads.append(1)
            if failure == "load":
                raise ValueError("load failed")
            self.device = "cuda:0"  # protocol stand-in; all tensors stay on CPU
            self.base_folder = self.output_dir = "/unused"

        def prepare_design(self, **features):
            ledger.phase("prepare", "cpu")
            if failure == "prepare":
                raise ValueError("prepare failed")
            assert features["fixed_residues"] == "A1"
            return features

        def execute_design(self, prepared):
            ledger.phase("execute", "gpu")
            # Direct guards inside kernels must not acquire another lease.
            with device_guard(backend):
                if failure == "execute":
                    raise ValueError("execute failed")
                return {"sampling_probs": torch.ones(1, 2, 20) / 20}

        def finalize_design(self, raw):
            ledger.phase("finalize", "cpu")
            if failure == "finalize":
                raise ValueError("finalize failed")
            return raw

        def design_sequences(self, **features):
            return self.finalize_design(self.execute_design(self.prepare_design(**features)))

    module_name = ("astrolabe.mutators.wrappers.ligandmpnn_run" if backend == "ligandmpnn"
                   else "astrolabe.mutators.wrappers.carbondesign_runner")
    module = ModuleType(module_name)
    setattr(module, "LigandMPNN" if backend == "ligandmpnn" else "CarbonDesignRunner", Runner)
    monkeypatch.setitem(sys.modules, module_name, module)
    factory = getattr(native_daemons, backend + "_daemon_factory")
    if failure == "load":
        with pytest.raises(ValueError, match="load failed"):
            factory({"runner_config": {}})
        return
    handler = factory({"runner_config": {}})
    for _ in range(2):
        response = infer(handler, {"features": {"fixed_residues": "A1"}})
        assert response["ok"] is (failure is None)
        assert not ledger.active
        if failure is None:
            assert response["result"]["sampling_probs"].device.type == "cpu"
            assert set(response["runtime"]["timings"]) == {"prepare", "execute", "finalize"}
        if failure == "execute":
            assert response["runtime"]["lifecycle"]["state"] == "quarantined"
            with pytest.raises(RuntimeError, match="quarantined"):
                infer(handler, {"features": {"fixed_residues": "A1"}})
            break
    description = handler({"operation": "describe"})["metadata"]
    assert description["runtime_lifecycle"]["loader_count"] == len(loads) == 1
    assert description["runtime_lifecycle"]["residency"] == "resident_model"


def test_ligand_loader_keeps_real_harness_callable_closure(monkeypatch, ledger):
    module = ModuleType("astrolabe.mutators.wrappers.ligandmpnn_run")
    class Runner:
        base_folder = "/unused"
        device = "cpu"
        def __init__(self, config):
            self.config = config
        def design_sequences(self, **features):
            return features
    module.LigandMPNN = Runner
    monkeypatch.setitem(sys.modules, module.__name__, module)
    def load(settings):
        implementation, metadata = native_daemons._load_ligandmpnn(settings)
        runner = inspect.getclosurevars(implementation).nonlocals["runner"]
        assert isinstance(runner, Runner)
        assert callable(implementation)
        return implementation, metadata
    # The real performance harness supplies its own kernel_handler around loader.
    handler = execution.kernel_handler({"runner_config": {}}, load)
    assert infer(handler, {"features": {"fixed_residues": "A1"}})["result"] == {"fixed_residues": "A1"}
    assert [event for event in ledger.events if event[0] == "enter"] == [
        ("enter", "gpu", "load"), ("enter", "gpu", "execute")]


def test_ligand_preparation_is_cpu_and_does_not_consume_rng(monkeypatch, tmp_path):
    from astrolabe.mutators.wrappers import ligandmpnn_run as module
    runner = object.__new__(module.LigandMPNN)
    runner.ligand_mpnn_use_side_chain_context = True
    runner.parse_atoms_with_zero_occupancy = False
    calls = []
    def parse(path, **kwargs):
        assert str(kwargs["device"]) == "cpu"
        calls.append((path, kwargs["chains"]))
        return ({"mask_c": [torch.tensor([True])]}, None, None, [""], None)
    monkeypatch.setattr(module, "parse_PDB", parse)
    paths = tmp_path / "paths.json"
    paths.write_text(json.dumps(["first.pdb", "second.pdb"]))
    state = torch.get_rng_state().clone()
    features = {"pdb_path_multi": str(paths), "fixed_residues_multi": "fixed.json",
                "symmetry_residues": "A1,B1", "symmetry_weights": "0.5,0.5",
                "parse_these_chains_only": "AB", "batch_size": 3}
    prepared = runner.prepare_design(**features)
    assert calls == [("first.pdb", "AB"), ("second.pdb", "AB")]
    assert all(prepared[key] == value for key, value in features.items())
    assert torch.equal(state, torch.get_rng_state())
    assert prepared["_parsed"]["first.pdb"][0]["mask_c"][0].device.type == "cpu"


def _fake_dplm(monkeypatch, ledger, failure):
    for name in ["byprot", "byprot.models", "byprot.models.dplm", "byprot.models.dplm.dplm"]:
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    class Tokenizer:
        def batch_encode_plus(self, sequences, **kwargs):
            if ledger.active:
                ledger.phase("prepare", "cpu")
            if failure == "prepare":
                raise ValueError("tokenization failed")
            return {"input_ids": torch.tensor([[1, 0, 2]] * len(sequences))}
        def batch_decode(self, outputs, **kwargs):
            if ledger.active:
                ledger.phase("finalize", "cpu")
            assert outputs.device.type == "cpu"
            if failure == "finalize":
                raise ValueError("decode failed")
            return [" A " + str(row[1].item()) for row in outputs]
    class Model:
        mask_id = 0
        tokenizer = Tokenizer()
        @classmethod
        def from_pretrained(cls, name):
            ledger.phase("load", "cpu")
            return cls()
        def to(self, device):
            assert device == "cpu"
            return self
        def eval(self):
            return self
        def generate(self, **kwargs):
            if ledger.active:
                ledger.phase("execute", "cpu")
            if failure == "execute":
                raise ValueError("generation failed")
            assert kwargs["max_iter"] == 7
            assert kwargs["sampling_strategy"] == "gumbel_argmax"
            assert kwargs["temperature"] == 0.75
            return kwargs["input_tokens"] + torch.randint(1, 4, kwargs["input_tokens"].shape)
    sys.modules["byprot.models.dplm.dplm"].DiffusionProteinLanguageModel = Model
    return Model


@pytest.mark.parametrize("failure", [None, "prepare", "execute", "finalize"])
def test_dplm_real_factory_cpu_stages_rng_and_failure(monkeypatch, ledger, failure):
    from astrolabe.mutators.wrappers import dplm_daemon
    Model = _fake_dplm(monkeypatch, ledger, failure)
    handler = dplm_daemon.create_handler({"device": "cpu", "max_iter": 7,
                                         "temperature": 0.75, "sampling_strategy": "gumbel_argmax"})
    payload = {"tasks": [{"name": "A", "masked_sequence": "A<mask>C", "num_samples": 3},
                         {"name": "B", "masked_sequence": "D<mask>E", "num_samples": 2}]}
    torch.manual_seed(91)
    if failure is None:
        model = Model()
        expected = [{"name": task["name"], "samples": dplm_daemon._run_generation(
            model=model, tokenizer=model.tokenizer, masked_sequence=task["masked_sequence"],
            num_samples=task["num_samples"], max_iter=7, sampling_strategy="gumbel_argmax",
            temperature=0.75, device="cpu") } for task in payload["tasks"]]
        final_rng = torch.get_rng_state()
        torch.manual_seed(91)
        assert protocol_infer(handler, payload) == expected
        assert torch.equal(final_rng, torch.get_rng_state())
    else:
        response = infer(handler, payload)
        assert not response["ok"]
        assert not ledger.active


@pytest.mark.parametrize("phase", ["load", "compute"])
def test_direct_guard_releases_on_failure(ledger, phase):
    with pytest.raises(ValueError, match="direct failure"):
        with device_guard("dynamicmpnn", phase=phase):
            ledger.phase(phase, "gpu")
            raise ValueError("direct failure")
    assert not ledger.active


def test_na_handler_factory_uses_native_model_and_cpu_argument_stages(monkeypatch, ledger):
    from astrolabe.mutators.wrappers import na_mpnn_daemon
    loads = []
    resident = {"model": torch.nn.Linear(1, 1)}
    def load(args):
        ledger.phase("load", "cpu")
        loads.append(args)
        return resident
    def run(args, model):
        ledger.phase("execute", "cpu")
        assert model is resident
        assert args.fixed_residues == "A1 A3"
        if args.seed == 3:
            raise ValueError("native failure")
    monkeypatch.setattr(na_mpnn_daemon.NAMPNNHandler, "_load_upstream", lambda self: SimpleNamespace(load_model=load, run_design=run))
    handler = na_mpnn_daemon.create_handler({"repo_path": "/unused", "checkpoint": "/unused.pt"})
    payload = {"job_dir": "/output", "structure_path": "/input.pdb", "fixed_residues": ["A1", "A3"], "seed": 2}
    assert handler(payload) == {"job_dir": "/output", "structure_path": "/input.pdb"}
    with pytest.raises(RuntimeError, match="native failure"):
        handler({**payload, "seed": 3})
    assert not ledger.active
    assert len(loads) == 1


@pytest.mark.parametrize("kind", ["esm2", "esm_if1"])
def test_esm_factories_keep_device_math_and_if1_rng(monkeypatch, ledger, kind):
    import importlib
    # Replace only the vendor API; load the actual Astrolabe mutator class.
    esm = ModuleType("esm2")
    util = ModuleType("esm2.inverse_folding.util")
    alphabet = SimpleNamespace(standard_toks=list("ACDEFGHIKLMNPQRSTVWY"),
        prepend_toks=["p"] * 4, append_toks=[], prepend_bos=True,
        append_eos=True, use_msa=False, all_toks=["p"] * 4 + list("ACDEFGHIKLMNPQRSTVWY"))
    class Dictionary:
        def get_idx(self, token):
            return {"<mask>": 0, "<cath>": 1}.get(token, 2)
    class Decoder:
        dictionary = Dictionary()
        def __call__(self, tokens, encoder_out, incremental_state):
            if ledger.active:
                ledger.phase("execute", "cpu")
            return torch.ones((1, 25, 1)), None
    class Model:
        decoder = Decoder()
        def eval(self):
            return self
        def cuda(self):
            return self  # explicit CPU vendor stand-in
        def parameters(self):
            yield SimpleNamespace(device=torch.device("cpu"))
        def encoder(self, *args):
            if ledger.active:
                ledger.phase("execute", "cpu")
            return None
        def __call__(self, batch_tokens):
            if ledger.active:
                ledger.phase("execute", "cpu")
            return {"logits": torch.randn(*batch_tokens.shape, 25)}
    def load():
        if ledger.active:
            ledger.phase("load", "gpu")
        return Model(), alphabet
    class Converter:
        def __init__(self, dictionary):
            pass
        def __call__(self, rows, device):
            assert device == "cpu"
            if ledger.active:
                ledger.phase("prepare", "cpu")
            length = len(rows[0][0])
            return torch.zeros(1, length, 3, 3), torch.ones(1, length), None, None, torch.zeros(1, length, dtype=torch.bool)
    util.CoordBatchConverter = Converter
    util.get_atom_coords_residuewise = lambda *args: None
    esm.pretrained = SimpleNamespace(esm2_t33_650M_UR50D=load, esm_if1_gvp4_t16_142M_UR50=load)
    monkeypatch.setitem(sys.modules, "esm2", esm)
    monkeypatch.setitem(sys.modules, "esm2.inverse_folding", ModuleType("esm2.inverse_folding"))
    monkeypatch.setitem(sys.modules, util.__name__, util)
    monkeypatch.delitem(sys.modules, "astrolabe.mutators." + kind, raising=False)
    module = importlib.import_module("astrolabe.mutators." + kind)
    if kind == "esm2":
        features = {"batch_tokens": torch.tensor([[1, 2, 3, 4, 5]])}
        baseline = lambda: module.extract_esm2_probs(Model(), **features)
    else:
        features = {"coords": torch.zeros(3, 3, 3), "partial_seq": ["<mask>"] * 3}
        baseline = lambda: module.esm_extract_probs(Model(), **features)
    handler = getattr(native_daemons, kind + "_daemon_factory")({})
    torch.manual_seed(47)
    expected = baseline()
    expected_rng = torch.get_rng_state()
    torch.manual_seed(47)
    response = infer(handler, {"features": {"AB": features}})
    assert response["ok"]
    assert torch.equal(response["result"]["AB"]["sampling_probs"], expected["sampling_probs"])
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert response["result"]["AB"]["name"] == "AB"
    assert response["result"]["AB"]["feature_dict"] is not None


def test_surfpro_factory_prepares_surfaces_and_formats_on_cpu(monkeypatch, ledger, tmp_path):
    from astrolabe.mutators.wrappers import surfpro_daemon as module
    runtime = SimpleNamespace(torch=torch, device="cpu", dictionary=SimpleNamespace(get_tok=lambda value: "AC"[value]))
    monkeypatch.setattr(module, "_load_runtime", lambda args: runtime)
    class Builder:
        def __init__(self, **kwargs):
            pass
        def build(self, path, seed):
            ledger.phase("prepare", "cpu")
            assert seed == 11
            return module.ProteinInput(path, "AC", ["A"], {"A": 2}), torch.zeros(2, 3), torch.zeros(2)
    monkeypatch.setattr(module, "SurfaceBuilder", Builder)
    def encode(*args):
        ledger.phase("execute", "gpu")
        return None, None
    def decode(runtime, encoded, lengths, native, fixed, **kwargs):
        ledger.phase("execute", "gpu")
        assert native == "AC" and fixed == {0}
        assert kwargs["_raw"]
        return [torch.ones(20)/20] * 2, [0, 1]
    monkeypatch.setattr(module, "_encode_surface", encode)
    monkeypatch.setattr(module, "_decode_sequence", decode)
    original = module._finalize_decode
    def finalize(runtime, raw):
        ledger.phase("finalize", "cpu")
        return original(runtime, raw)
    monkeypatch.setattr(module, "_finalize_decode", finalize)
    settings = {"device": "cpu", "repo_path": str(tmp_path), "work_dir": str(tmp_path),
                "msms_bin": "/mock/msms", "pdb_to_xyzrn_bin": "/mock/xyzrn", "max_surface_vertices": 5000}
    handler = module.create_handler(settings)
    result = protocol_infer(handler, {"pdb_path": str(tmp_path/"a.pdb"), "fixed_residues": "A1", "seed": 11})
    assert result[1] == "AC" and len(result[0]) == 2


def test_surfdesign_factory_has_cpu_prepare_finalize_and_cleanup(monkeypatch, ledger, tmp_path):
    from astrolabe.mutators.wrappers import surfdesign_daemon as module
    closed = []
    class Builder:
        def __init__(self, *args):
            pass
        def close(self):
            closed.append(True)
    runtime = SimpleNamespace(device="cpu")
    monkeypatch.setattr(module, "_load_runtime", lambda args: runtime)
    monkeypatch.setattr(module, "SurfaceBuilder", Builder)
    def prepare(runtime, builder, path, fixed):
        ledger.phase("prepare", "cpu")
        assert fixed == "A1"
        return torch.tensor([1, 2])
    def execute(runtime, prepared):
        ledger.phase("execute", "cpu")
        return prepared + 1
    def finalize(runtime, result):
        ledger.phase("finalize", "cpu")
        return result.tolist()
    monkeypatch.setattr(module, "_prepare", prepare)
    monkeypatch.setattr(module, "_execute", execute)
    monkeypatch.setattr(module, "_finalize", finalize)
    handler = module.create_handler({"device": "cpu", "work_dir": str(tmp_path), "surface_mode": "pymol"})
    assert protocol_infer(handler, {"pdb_path": str(tmp_path/"a.pdb"), "fixed_residues": "A1"}) == [2, 3]
    handler.close()
    assert closed == [True]


def test_dynamic_cli_failure_releases_actual_runner_guard(monkeypatch, ledger, tmp_path):
    import subprocess
    from astrolabe.mutators.wrappers import dynamicmpnn_run
    paths = tmp_path / "paths.json"
    paths.write_text(json.dumps(["a.pdb", "b.pdb"]))
    def fail(*args, **kwargs):
        ledger.phase("compute", "gpu")
        raise subprocess.CalledProcessError(1, args[0])
    monkeypatch.setattr(dynamicmpnn_run.subprocess, "run", fail)
    runner = dynamicmpnn_run.DynamicMPNNRunner({"execution_mode": "in_process", "output_dir": str(tmp_path/"out"), "command_template": "mock {manifest}"})
    assert runner.model_metadata["residency"] == "one_shot"
    with pytest.raises(subprocess.CalledProcessError):
        runner.design_sequences(pdb_path_multi=str(paths))
    assert not ledger.active


def test_dynamic_native_client_has_no_parent_gpu_lease(monkeypatch, ledger, tmp_path):
    import numpy as np
    from astrolabe.mutators.wrappers import dynamicmpnn_run
    class Client:
        metadata = {"device": "cpu", "residency": "unknown"}
        def __init__(self, *args, **kwargs):
            assert not ledger.active
        def request(self, payload):
            assert not ledger.active
            return np.ones((1, 2, 20)) / 20
        def shutdown(self):
            pass
    monkeypatch.setattr(dynamicmpnn_run, "NativeModelClient", Client)
    paths = tmp_path / "paths.json"
    paths.write_text(json.dumps(["a.pdb", "b.pdb"]))
    runner = dynamicmpnn_run.DynamicMPNNRunner({"execution_mode": "daemon", "backend_factory": "mock:factory", "output_dir": str(tmp_path/"out")})
    assert runner.design_sequences(pdb_path_multi=str(paths))["sampling_probs"].shape == (1, 2, 20)
    assert ledger.events == []


def test_cpu_model_factory_with_real_broker_releases_failed_request(monkeypatch, tmp_path):
    from astrolabe.runtime.admission import ResourceBroker, ResourceClient
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "init", lambda: pytest.fail("CUDA initialized"))
    module = ModuleType("astrolabe.mutators.wrappers.ligandmpnn_run")
    class Runner:
        device = "cpu"
        base_folder = "/unused"
        def __init__(self, config):
            pass
        def prepare_design(self, **features):
            return features
        def execute_design(self, prepared):
            if prepared.get("fail"):
                raise ValueError("device stage failed")
            return {"sampling_probs": torch.ones(1, 2, 20)/20}
        def finalize_design(self, result):
            return result
        def design_sequences(self, **features):
            return self.finalize_design(self.execute_design(self.prepare_design(**features)))
    module.LigandMPNN = Runner
    monkeypatch.setitem(sys.modules, module.__name__, module)
    broker = ResourceBroker({"devices": [], "cpu_slots": 1, "socket_dir": str(tmp_path/"broker"), "shutdown_timeout": 2}).start()
    try:
        client = ResourceClient(broker.socket_path)
        handler = native_daemons.ligandmpnn_daemon_factory({"runner_config": {}})
        response = infer(handler, {"features": {"fail": True}})
        assert not response["ok"]
        assert client.snapshot()["active"] == 0
        success = infer(handler, {"features": {}})
        assert success["ok"]
        assert set(success["runtime"]["timings"]) == {"prepare", "execute", "finalize"}
        snapshot = client.snapshot()
        assert snapshot["active"] == snapshot["pending"] == 0
        assert snapshot["resident_models"] == 1  # Live CPU weights remain registered.
        assert handler({"operation": "describe"})["metadata"]["runtime_lifecycle"]["loader_count"] == 1
        model_id = handler({"operation": "describe"})["metadata"]["runtime_lifecycle"]["model_id"]
        del handler
        client.unregister_model(model_id)
        assert client.snapshot()["resident_models"] == 0
    finally:
        broker.close()


def test_carbondesign_actual_stages_keep_probabilities_and_mask_slicing(monkeypatch, ledger):
    from astrolabe.mutators.wrappers import carbondesign_runner as module
    runner = object.__new__(module.CarbonDesignRunner)
    runner.device = "cpu"
    runner.output_dir = "/unused"
    runner._feats = []
    runner.batch_size, runner.max_seq_len, runner.temperature, runner.num_aa = 2, 10, 0.5, 20
    logits = torch.arange(80, dtype=torch.float32).reshape(2, 2, 20)
    def batches(**kwargs):
        ledger.phase("execute", "cpu")
        assert kwargs["name_idx"] == ["first", "second"]
        assert kwargs["shuffle"] is False
        return [{"seq": torch.zeros(2, 2), "mask": torch.tensor([[1, 1], [1, 0]]), "name": ["first", "second"]}]
    runner._dataset_test = SimpleNamespace(load=batches)
    runner.model = lambda **kwargs: {"heads": {"seqhead": {"logits": logits}}}
    monkeypatch.setattr(module, "CarbonDesignRunner", lambda settings: runner)
    handler = native_daemons.carbondesign_daemon_factory({"runner_config": {"device": "cpu"}})
    result = protocol_infer(handler, {"features": {"data_dir": "/input", "name_idx": ["first", "second"]}})
    expected = torch.softmax(logits/0.5, dim=-1)
    assert torch.equal(result["first"]["sampling_probs"], expected[0:1])
    assert torch.equal(result["second"]["sampling_probs"], expected[1:2, :1])


@pytest.mark.parametrize("backend", ["na_mpnn", "dplm"])
def test_external_cli_failure_releases_actual_adapter(monkeypatch, ledger, tmp_path, backend):
    import subprocess
    def fail(*args, **kwargs):
        ledger.phase("compute", "gpu")
        raise subprocess.CalledProcessError(1, "mock external command")
    if backend == "na_mpnn":
        from astrolabe.mutators.wrappers import na_mpnn_runner as module
        runner = object.__new__(module.NAMPNNRunner)
        runner.repo_path = tmp_path
        runner.python = sys.executable
        runner.checkpoint = tmp_path/"mock.pt"
        for key in ["temperature", "batch_size", "number_of_batches", "design_na_only", "parse_na_only", "na_shared_tokens", "output_pdbs"]:
            setattr(runner, key, 1)
        monkeypatch.setattr(module.subprocess, "run", fail)
        with pytest.raises(subprocess.CalledProcessError):
            runner._run_subprocess({"structure_path": "/input.pdb", "job_dir": "/output", "seed": 1,
                                    "chains_to_design": "A", "fixed_residues": ["A1"], "parse_these_chains_only": "A"})
    else:
        from astrolabe.mutators.wrappers import dplm_runner as module
        runner = module.DPLMRunner({"python_bin": sys.executable, "repo_path": str(tmp_path),
            "daemon": False, "work_dir": str(tmp_path/"work"), "response_cache_dir": str(tmp_path/"cache")})
        assert runner.model_metadata["residency"] == "one_shot"
        monkeypatch.setattr(module.subprocess, "run", fail)
        with pytest.raises(module.RequestOutcomeUnknown, match="subprocess exited 1"):
            runner._infer_tasks_no_cache([{"name": "A", "masked_sequence": "A<mask>"}])
        runner.close()
    assert not ledger.active
