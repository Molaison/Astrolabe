"""CPU-only contract tests; synthetic evidence is never real-model acceptance."""
from __future__ import annotations

import copy
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import pickle
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validation/scientific_workflow_acceptance.py"
SPEC = importlib.util.spec_from_file_location("_scientific_workflow_test", SCRIPT)
H = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(H)
SEQ = "TTCCP" + "A" * 41


def pdb_text(offset=0):
    names = {"T": "THR", "C": "CYS", "P": "PRO", "A": "ALA"}
    return "HEADER    SYNTHETIC CPU FIXTURE\n" + "".join(
        f"ATOM  {i:5d}  CA  {names[aa]:3s} A{i:4d}    "
        f"{i + offset:8.3f}{0.:8.3f}{0.:8.3f}{1.:6.2f}{80.:6.2f}           C  \n"
        for i, aa in enumerate(SEQ, 1)) + "TER\nEND\n"


def state():
    return {"python": (3, (1, 2, 3), None), "numpy": ("MT19937", np.arange(4, dtype=np.uint32), 0, 0, 0.),
            "torch_cpu": np.arange(5, dtype=np.uint8), "torch_seed": 17,
            "cuda_initialized": False}


def fixture(root):
    run = root / "trajectory/scientific_workflow/001_run"
    (run / "structures").mkdir(parents=True)
    (run / "scores.csv").write_text("step,weighted_score,accepted,stage,note,pA_sequence,timestamp,pdb_path\n" +
        "".join(f"{i},0.8,True,default,,{SEQ},time-{root.name},{run}/step_{i:03d}.pdb\n" for i in range(3)))
    for i in range(3):
        (run / f"structures/step_{i:03d}_A_model.pdb").write_text(pdb_text())
    evidence = {"original_config": {"seed": 17, "steps": 3, "repeats": 1, "seqs": {"A": "+TTCCP " + "A" * 41},
                                    "initializer": {"mode": "random"}, "out_dir": str(root)},
                "evaluations": [{"step": i, "input": {"A": SEQ}, "scores": {
                    "plddt": {"A": [np.array(0.8)]}, "complex_paths": {"A": str(run / f"step_{i:03d}.pdb")},
                    "extra_raw_field": np.arange(3)}, "rng_before": state(), "rng_after": state()} for i in range(3)],
                "final_rng": state(), "model_loads": [{"rng_before": state(), "rng_after": state()}],
                "torch_loads": [{"resolved": "/models/test.pt"}],
                "native_clients": [], "native_requests": [], "artifacts": H.artifacts(root)}
    report = {"schema": H.SCHEMA, "status": "completed", "lane": "direct", "source": "/synthetic/source/" + root.name,
              "output": str(root), "source_unchanged": True, "module_origins": {"astrolabe": "synthetic"}}
    persist(root, evidence, report)
    return evidence, report


def persist(root, evidence, report):
    (root / "evidence.pkl").write_bytes(pickle.dumps(evidence, protocol=5))
    H.save_npz(root / "arrays.npz", evidence)
    report["evidence_sha256"] = H.sha256(root / "evidence.pkl")
    report["arrays_sha256"] = H.sha256(root / "arrays.npz")
    H.write_json(root / "worker.json", report)


def test_complete_outputs_ignore_paths_and_timestamps(tmp_path):
    fixture(tmp_path / "left")
    fixture(tmp_path / "right")
    result = H.compare_workers(tmp_path / "left", tmp_path / "right")
    assert result["equal"], result
    assert all(result["checks"].values())


@pytest.mark.parametrize("change", ["accepted", "score", "sequence", "raw_score", "after_rng", "warm_before_rng", "final_rng", "pdb", "note"])
def test_scientific_differences_fail(tmp_path, change):
    fixture(tmp_path / "left")
    ev, report = fixture(tmp_path / "right")
    if change in {"accepted", "score", "sequence", "note"}:
        row = ev["artifacts"]["tables"][0]["rows"][1]
        row[{"accepted": "accepted", "score": "weighted_score", "sequence": "pA_sequence", "note": "note"}[change]] = {
            "accepted": "False", "score": "0.81", "sequence": SEQ[:-1] + "V", "note": "changed scientific row"}[change]
    elif change == "raw_score":
        ev["evaluations"][2]["scores"]["extra_raw_field"][0] = 99
    elif change == "after_rng":
        ev["evaluations"][0]["rng_after"]["torch_cpu"][0] = 99
    elif change == "warm_before_rng":
        ev["evaluations"][1]["rng_before"]["torch_cpu"][0] = 99
    elif change == "final_rng":
        ev["final_rng"]["torch_cpu"][0] = 99
    elif change == "pdb":
        ev["artifacts"]["pdbs"][1]["structure"]["coordinates"][0, 0] += .001
    persist(tmp_path / "right", ev, report)
    result = H.compare_workers(tmp_path / "left", tmp_path / "right")
    assert not result["equal"]
    assert not result["checks"].get("scientific_equal", False)


@pytest.mark.parametrize("field", ["first_before", "load_before", "load_after"])
def test_load_boundary_differences_obey_aligned_scientific_boundaries(tmp_path, field):
    fixture(tmp_path / "left")
    ev, report = fixture(tmp_path / "right")
    if field == "first_before":
        ev["evaluations"][0]["rng_before"]["torch_cpu"][0] = 99
    else:
        ev["model_loads"][0]["rng_" + field.split("_")[1]]["torch_cpu"][0] = 99
    persist(tmp_path / "right", ev, report)
    result = H.compare_workers(tmp_path / "left", tmp_path / "right")
    assert result["checks"]["scientific_equal"]
    if field == "first_before":
        assert result["equal"]
        assert result["requires_review"]
    else:
        assert not result["equal"]
        assert not result["checks"]["loader_before_rng_equal" if field == "load_before"
                                     else "loader_after_full_rng_equal"]


@pytest.mark.parametrize("lifecycle_difference", ["initialized_flag", "cuda_state_availability"])
def test_preloader_cuda_lifecycle_difference_is_diagnostic(tmp_path, lifecycle_difference):
    fixture(tmp_path / "left")
    ev, report = fixture(tmp_path / "right")
    before = ev["model_loads"][0]["rng_before"]
    before["cuda_initialized"] = True
    if lifecycle_difference == "cuda_state_availability":
        before["cuda"] = {"0": np.array([17, 18], dtype=np.uint8)}
        before["cuda_scope"] = "current initialized device only; no other visible device touched"
    persist(tmp_path / "right", ev, report)
    result = H.compare_workers(tmp_path / "left", tmp_path / "right")
    assert result["equal"], result
    assert result["requires_review"]
    assert result["checks"]["loader_before_rng_equal"]
    assert result["checks"]["loader_after_full_rng_equal"]
    assert not result["diagnostics"]["loader_before_full_rng"]["equal"]


def test_preloader_cuda_values_are_exact_when_both_observed(tmp_path):
    for name, value in (("left", 17), ("right", 18)):
        ev, report = fixture(tmp_path / name)
        before = ev["model_loads"][0]["rng_before"]
        before["cuda_initialized"] = True
        before["cuda"] = {"0": np.array([value], dtype=np.uint8)}
        persist(tmp_path / name, ev, report)
    result = H.compare_workers(tmp_path / "left", tmp_path / "right")
    assert not result["equal"]
    assert not result["checks"]["loader_before_rng_equal"]


@pytest.mark.parametrize("cpu_field", ["python", "numpy", "torch_cpu", "torch_seed"])
def test_preloader_missing_cpu_field_cannot_pass_against_itself(tmp_path, cpu_field):
    ev, report = fixture(tmp_path / "left")
    del ev["model_loads"][0]["rng_before"][cpu_field]
    persist(tmp_path / "left", ev, report)
    result = H.compare_workers(tmp_path / "left", tmp_path / "left")
    assert not result["equal"]
    assert not result["validation"]["expected"]["rng_fields_present"]


@pytest.mark.parametrize("missing", ["capture", "row", "pdb", "loader_rng", "final_rng", "fixed_prefix", "score_column"])
def test_incomplete_or_invalid_evidence_fails_even_against_itself(tmp_path, missing):
    ev, report = fixture(tmp_path / "left")
    if missing == "capture":
        ev["evaluations"].pop()
    elif missing == "row":
        ev["artifacts"]["tables"][0]["rows"].pop()
    elif missing == "pdb":
        ev["artifacts"]["pdbs"].pop()
    elif missing == "loader_rng":
        ev["model_loads"][0].pop("rng_before")
    elif missing == "final_rng":
        ev.pop("final_rng")
    elif missing == "fixed_prefix":
        ev["evaluations"][0]["input"]["A"] = "A" + SEQ[1:]
    else:
        ev["artifacts"]["tables"][0]["rows"][0].pop("accepted")
    persist(tmp_path / "left", ev, report)
    result = H.compare_workers(tmp_path / "left", tmp_path / "left")
    assert not result["equal"]


def test_native_missing_receipts_cannot_pass(tmp_path):
    ev, report = fixture(tmp_path / "left")
    report["lane"] = "adaptive"
    persist(tmp_path / "left", ev, report)
    result = H.compare_workers(tmp_path / "left", tmp_path / "left")
    assert not result["equal"]
    assert not result["validation"]["expected"]["native_receipts"]


@pytest.mark.parametrize("field", ["per_proto_scores", "selector_mean"])
def test_single_ulp_residue_or_selector_difference_fails_exactly(tmp_path, field):
    roots = [tmp_path / name for name in ("left", "right")]
    fixtures = [fixture(root) for root in roots]
    selector = "mutator_selector_MCMC_pA_plddt_after_mean"
    for ev, _ in fixtures:
        ev["evaluations"][0]["scores"]["per_proto_scores"] = {"A": [np.full(46, .8, dtype=np.float32)]}
        table = ev["artifacts"]["tables"][0]
        table["columns"].append(selector)
        for row in table["rows"]:
            row[selector] = "0.8"
    ev, _ = fixtures[1]
    if field == "per_proto_scores":
        values = ev["evaluations"][0]["scores"][field]["A"][0]
        values[0] = np.nextafter(values[0], np.float32(np.inf))
    else:
        ev["artifacts"]["tables"][0]["rows"][1][selector] = repr(float(np.nextafter(.8, np.inf)))
    for root, (ev, report) in zip(roots, fixtures):
        persist(root, ev, report)
    result = H.compare_workers(*roots)
    assert result["checks"]["expected_complete"] and result["checks"]["actual_complete"]
    assert not result["equal"] and not result["checks"]["scientific_equal"]
    assert not result["checks"]["evaluations_equal" if field == "per_proto_scores" else "tables_equal"]
    assert result["tolerance"] is None


def native_requests():
    return [{"backend": "esmfold", "operation": "infer",
             "payload": {"operation": "infer", "sequences": [SEQ], "return_residue_plddt": True},
             "metadata": {"synthetic": True}, "receipt": {"profile_matched": True}} for _ in range(3)]


@pytest.mark.parametrize("change", [None, "missing_flag", "false_flag", "missing_payload", "wrong_sequence"])
def test_native_reduction_protocol_evidence_is_required(tmp_path, change):
    ev, report = fixture(tmp_path / "native")
    report["lane"] = "resident_serial"
    ev["native_requests"] = native_requests()
    ev["native_clients"] = [{"profile": {"qualification": True}}]
    first = ev["native_requests"][0]
    if change == "missing_flag":
        del first["payload"]["return_residue_plddt"]
    elif change == "false_flag":
        first["payload"]["return_residue_plddt"] = False
    elif change == "missing_payload":
        del first["payload"]
    elif change == "wrong_sequence":
        first["payload"]["sequences"] = [SEQ[:-1] + "V"]
    persist(tmp_path / "native", ev, report)
    result = H.compare_workers(tmp_path / "native", tmp_path / "native")
    assert result["equal"] is (change is None)
    if change:
        assert not all(result["validation"]["actual"].values())


@pytest.mark.parametrize("file", ["evidence.pkl", "arrays.npz", "trajectory/scientific_workflow/001_run/scores.csv",
                                  "trajectory/scientific_workflow/001_run/structures/step_002_A_model.pdb"])
def test_corrupted_or_missing_files_fail_closed(tmp_path, file):
    fixture(tmp_path / "left")
    (tmp_path / "left" / file).unlink()
    result = H.compare_workers(tmp_path / "left", tmp_path / "left")
    assert not result["equal"] and "error" in result


def test_capture_keeps_original_behavior_and_rng(monkeypatch):
    fake_torch = SimpleNamespace(get_rng_state=lambda: SimpleNamespace(numpy=lambda: np.array([1, 2], dtype=np.uint8)),
                                 initial_seed=lambda: 17, cuda=SimpleNamespace(is_initialized=lambda: False))
    monkeypatch.setitem(sys.modules, "astrolabe.core", SimpleNamespace(global_config=SimpleNamespace(step=0)))
    cap = H.Capture(fake_torch, "direct", {})
    random.seed(17)
    np.random.seed(17)
    before = H.rng_state(fake_torch)
    class Evaluator:
        pass
    score = SimpleNamespace(new_raw_field=np.arange(5), complex_paths={"A": Path("/test/model.pdb")})
    observed = []
    def original(evaluator, inputs):
        observed.append((random.random(), np.random.random()))
        return score
    result = cap.evaluate(original)(Evaluator(), {"A": SEQ})
    assert result is score
    assert len(observed) == 1
    assert H.reference().compare_outputs(cap.evaluations[0]["rng_before"], before)["equal"]
    assert H.reference().compare_outputs(cap.evaluations[0]["rng_after"], H.rng_state(fake_torch))["equal"]
    score.new_raw_field[0] = 100
    assert cap.evaluations[0]["scores"]["new_raw_field"][0] == 0
    assert cap.evaluations[0]["scores"]["complex_paths"]["A"] == "/test/model.pdb"


def test_cuda_rng_samples_current_device_only():
    touched = []
    class Generator:
        def __init__(self, i):
            self.i = i
        def get_state(self):
            touched.append(self.i)
            return SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: np.arange(4, dtype=np.uint8)))
    cuda = SimpleNamespace(is_initialized=lambda: True, current_device=lambda: 1,
                           default_generators=(Generator(0), Generator(1), Generator(2)))
    torch = SimpleNamespace(get_rng_state=lambda: SimpleNamespace(numpy=lambda: np.arange(4, dtype=np.uint8)),
                            initial_seed=lambda: 17, cuda=cuda)
    value = H.rng_state(torch)
    assert touched == [1]
    assert set(value["cuda"]) == {"1"}


def test_torch_load_delegates_keyword_f_without_hashing(tmp_path):
    checkpoint = tmp_path / "test.pt"
    checkpoint.write_bytes(b"not a real checkpoint")
    cap = H.Capture(None, "direct", {})
    calls = []
    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return object()
    sentinel = cap.loader(original)(f=checkpoint, weights_only=False)
    assert sentinel is not None
    assert calls == [((), {"f": checkpoint, "weights_only": False})]
    assert cap.loads[0]["resolved"] == str(checkpoint)


def test_explicit_profile_keeps_settings_and_parent_envelope(monkeypatch):
    cap = H.Capture(None, "adaptive", {"profiles_by_backend": {"esmfold": {
        "qualification": True, "concurrency_certified": True, "shapes": ["infer-shape"],
        "asset_sha256": {"/large/model.pt": "parent-owned-digest"}, "resident_bytes": 42}}})
    settings = {"model_name": "unchanged", "seed": 17}
    observed = []
    class Client:
        pid = 1
    def original(client, backend, factory, settings, *, resource_profile=None):
        observed.append((backend, factory, settings, copy.deepcopy(resource_profile)))
        client.metadata = {"runtime_lifecycle": {"identity": "one"}}
        client._resource_profile = resource_profile
    client = Client()
    cap.client_init(original)(client, "esmfold", "target:factory", settings)
    assert observed[0][2] is settings
    assert settings == {"model_name": "unchanged", "seed": 17}
    assert "asset_sha256" not in observed[0][3]
    assert cap.envelope["profiles_by_backend"]["esmfold"]["asset_sha256"]
    assert observed[0][3]["concurrency_certified"] is True
    runtime_settings = {"_runtime_profile": dict(observed[0][3], identity="one")}
    runtime_profile = cap.kernel(lambda cfg: dict(cfg["_runtime_profile"])) (runtime_settings)
    monkeypatch.setitem(sys.modules, "astrolabe.model_execution", SimpleNamespace(resource_shape=lambda p: "infer-shape"))
    def request(client, payload):
        assert runtime_profile["shapes"] == ["infer-shape"]
        client.last_runtime_receipt = {"profile_matched": True, "timings": {
            "execute": {"wait_seconds": .1, "run_seconds": .2}}}
        return settings
    payload = {"operation": "infer", "sequences": [SEQ], "return_residue_plddt": True}
    saved = copy.deepcopy(payload)
    assert cap.client_request(request)(client, payload) is settings
    assert payload == saved
    assert cap.requests[0]["payload"] == saved
    payload["sequences"][0] = "changed after capture"
    assert cap.requests[0]["payload"] == saved
    assert cap.requests[0]["receipt"] == client.last_runtime_receipt
    assert H.esmfold_infer_payloads({"native_requests": cap.requests}, "adaptive") == [saved]


@pytest.mark.parametrize("flag", [None, False, 1])
def test_capture_refuses_old_infer_protocol_before_dispatch(monkeypatch, flag):
    cap = H.Capture(None, "adaptive", {})
    client = SimpleNamespace()
    cap.backend_by_client[id(client)] = "esmfold"
    monkeypatch.setitem(sys.modules, "astrolabe.model_execution", SimpleNamespace(resource_shape=lambda p: "shape"))
    payload = {"operation": "infer", "sequences": [SEQ]}
    if flag is not None:
        payload["return_residue_plddt"] = flag
    def forbidden(*args, **kwargs):
        pytest.fail("Old infer protocol must not reach the runtime")
    with pytest.raises(ValueError, match="owner-side residue confidence reduction"):
        cap.client_request(forbidden)(client, payload)
    assert not cap.requests


def test_help_and_offline_compare_never_import_astrolabe_or_torch(tmp_path):
    fixture(tmp_path / "left")
    fixture(tmp_path / "right")
    code = """
import importlib.abc, importlib.util, sys
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'astrolabe', 'torch'}:
            raise AssertionError('forbidden import: ' + fullname)
sys.meta_path.insert(0, Deny())
spec = importlib.util.spec_from_file_location('offline_test', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module.compare_workers(sys.argv[2], sys.argv[3])['equal']
"""
    p = subprocess.run([sys.executable, "-c", code, str(SCRIPT), str(tmp_path / "left"), str(tmp_path / "right")],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    p = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True)
    assert p.returncode == 0 and "--envelope" in p.stdout


def test_compare_cli_writes_explicit_result_and_rejects_overwrite(tmp_path):
    fixture(tmp_path / "left")
    fixture(tmp_path / "right")
    output = tmp_path / "comparison.json"
    args = ["--compare", "--roots", str(tmp_path / "left"), str(tmp_path / "right"), "--output", str(output)]
    assert H.main(args) == 0
    assert json.loads(output.read_text())["comparisons"][0]["checks"]["scientific_equal"]
    with pytest.raises(SystemExit):
        H.main(args)


@pytest.mark.parametrize("lane", ["direct", "resident_serial", "adaptive"])
def test_compose_preserves_science_and_uses_container_environment(tmp_path, monkeypatch, lane):
    from omegaconf import OmegaConf
    config = tmp_path / "input.yaml"
    original = {"seed": 17, "steps": 3, "repeats": 1, "seqs": {"A": "+TTCCP " + "A" * 41},
                "initializer": {"mode": "random"}, "custom_scientific_field": {"untouched": True}}
    OmegaConf.save(OmegaConf.create(original), config)
    observed = []
    def compose(**kwargs):
        observed.append(kwargs)
        base = OmegaConf.create({"execution": {"mode": "legacy"}})
        OmegaConf.set_struct(base, True)
        return base
    monkeypatch.setitem(sys.modules, "astrolabe.cli", SimpleNamespace(_compose_hydra_config=compose))
    cfg, raw, _ = H.compose_config(tmp_path, config, tmp_path, lane)
    assert observed[0]["overrides"] == ["env=env_container"]
    assert raw == original
    assert cfg.custom_scientific_field.untouched is True
    assert cfg.execution.mode == ("legacy" if lane == "direct" else lane)
    assert "process_candidates" not in cfg


@pytest.mark.parametrize("lane", ["direct", "resident_serial", "adaptive"])
def test_structure_inline_mapping_is_relocated_exactly_and_recorded(tmp_path, monkeypatch, lane):
    from omegaconf import OmegaConf
    original_path = tmp_path / "structure.yaml"
    original = {"seed": 17, "steps": 3, "repeats": 1, "seqs": {"A": "+TTCCP " + "A" * 41},
                "initializer": {"mode": "structure", "sequence_policy": "auto",
                    "structure": {"source": "backbone", "chain_map": {"A": "A"}},
                    "sequence_designer": {"_type": "LigandMPNN", "model_type": "protein_mpnn",
                        "checkpoint_protein_mpnn": "/models/ligandmpnn/proteinmpnn_v_48_020.pt"}},
                "mutator": {"mutator_terms": {"MCMC": {"_type": "MCMC", "avoid_current_residue": True}}},
                "stages": {"default": {"mutators": ["MCMC"], "weighted_score_expression": "plddt",
                                       "acceptance_threshold": None, "early_stopping": None}},
                "task_name": "real_structure", "out_dir": "/original/output"}
    OmegaConf.save(OmegaConf.create(original), original_path)
    designer = copy.deepcopy(original["initializer"]["sequence_designer"])
    def compose(**kwargs):
        base = OmegaConf.create({"execution": {"mode": "legacy"}})
        OmegaConf.set_struct(base, True)
        return base
    monkeypatch.setitem(sys.modules, "astrolabe.cli", SimpleNamespace(_compose_hydra_config=compose))
    changes = []
    cfg, raw, composed = H.compose_config(tmp_path, original_path, tmp_path, lane, canonicalizations=changes)
    assert raw == original
    assert raw["initializer"]["sequence_designer"] == designer
    assert cfg.initializer.sequence_designer == H.INITIALIZER_ALIAS
    assert OmegaConf.to_container(cfg.mutator.mutator_terms[H.INITIALIZER_ALIAS], resolve=False) == designer
    # Reverse just the authorized relocation and output/execution namespaces;
    # every original scientific field must then be exactly identical.
    restored = copy.deepcopy(composed)
    del restored["mutator"]["mutator_terms"][H.INITIALIZER_ALIAS]
    restored["initializer"]["sequence_designer"] = designer
    for field in ("out_dir", "task_name", "execution"):
        restored.pop(field, None)
    expected = {k: v for k, v in original.items() if k not in {"out_dir", "task_name", "execution"}}
    assert restored == expected
    assert cfg.stages.default.mutators == ["MCMC"]
    assert len(changes) == 1
    change = changes[0]
    assert change["mapping_preserved_exactly"] is True
    assert change["before"]["initializer.sequence_designer"] == designer
    assert change["after"]["mutator.mutator_terms." + H.INITIALIZER_ALIAS] == designer
    assert H.canonicalize_initializer_designer(cfg) == []  # Already canonical.


def test_sequence_canonicalization_is_noop():
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({"initializer": {"mode": "random"}, "mutator": {"mutator_terms": {"MCMC": {"_type": "MCMC"}}}})
    before = OmegaConf.to_container(cfg, resolve=False)
    assert H.canonicalize_initializer_designer(cfg) == []
    assert OmegaConf.to_container(cfg, resolve=False) == before


@pytest.mark.parametrize("collision", ["existing_term", "stage_reference"])
def test_initializer_reserved_alias_never_overwrites_or_enters_mutation_stage(collision):
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({"initializer": {"mode": "structure", "sequence_designer": {"_type": "LigandMPNN"}},
                            "mutator": {"mutator_terms": {}}, "stages": {"default": {"mutators": ["MCMC"]}}})
    if collision == "existing_term":
        cfg.mutator.mutator_terms[H.INITIALIZER_ALIAS] = {"_type": "MCMC"}
    else:
        cfg.stages.default.mutators.append(H.INITIALIZER_ALIAS)
    before = OmegaConf.to_container(cfg, resolve=False)
    with pytest.raises(ValueError, match="Reserved initializer alias"):
        H.canonicalize_initializer_designer(cfg)
    assert OmegaConf.to_container(cfg, resolve=False) == before


@pytest.mark.parametrize("lane", ["direct", "resident_serial", "adaptive"])
def test_public_worker_lifecycle_cpu_fixture(tmp_path, monkeypatch, lane):
    """Exercise actual worker persistence/exit path with a synthetic public app."""
    class Tensor:
        def numpy(self):
            return np.arange(5, dtype=np.uint8)
    torch = SimpleNamespace(Tensor=Tensor, get_rng_state=Tensor, initial_seed=lambda: 17,
                             cuda=SimpleNamespace(is_initialized=lambda: False), __version__="synthetic-cpu",
                             version=SimpleNamespace(cuda=None),
                             backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: None)))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "astrolabe.core", SimpleNamespace(global_config=SimpleNamespace(step=0)))
    original = {"seed": 17, "steps": 3, "repeats": 1, "seqs": {"A": "+TTCCP " + "A" * 41},
                "initializer": {"mode": "random"}}
    output = tmp_path / "worker"
    public_calls = []
    caps = []
    @contextmanager
    def install(cap):
        caps.append(cap)
        yield
    class Model:
        pass
    class Evaluator:
        _model = None
        _device = "cpu"
    class PublicApp:
        def __init__(self, cfg):
            assert cfg is original
        def run(self):
            public_calls.append("Astrolabe.run")
            cap = caps[0]
            evaluator = Evaluator()
            def load(evaluator):
                cap.loader(lambda f: {})("/models/synthetic.pt")
                evaluator._model = Model()
                return evaluator._model
            cap.ensure_model(load)(evaluator)
            if lane != "direct":
                cap.requests.extend(native_requests())
                cap.clients.append({"profile": {"qualification": True}, "start": 0., "end": 0.})
            run = output / "trajectory/scientific_workflow/001_run"
            (run / "structures").mkdir(parents=True)
            for i in range(3):
                sys.modules["astrolabe.core"].global_config.step = i
                def evaluate(evaluator, inputs):
                    return SimpleNamespace(plddt={"A": [np.array(.8)]}, complex_paths={"A": run / "x.pdb"})
                cap.evaluate(evaluate)(evaluator, {"A": SEQ})
                (run / f"structures/step_{i:03d}_A_model.pdb").write_text(pdb_text())
            (run / "scores.csv").write_text("step,weighted_score,accepted,pA_sequence\n" +
                "".join(f"{i},0.8,True,{SEQ}\n" for i in range(3)))
    monkeypatch.setattr(H.Capture, "install", install)
    monkeypatch.setattr(H, "compose_config", lambda *args, **kwargs: (original, original, original))
    monkeypatch.setattr(H, "select_source", lambda source: SimpleNamespace(Astrolabe=PublicApp, __version__="synthetic"))
    monkeypatch.setattr(H, "source_identity", lambda *args: {"synthetic": True})
    monkeypatch.setattr(H, "module_origins", lambda source: {"astrolabe": "synthetic"})
    monkeypatch.setattr(H, "gpu_facts", lambda: {"synthetic": "CPU only"})
    monkeypatch.setenv("ASTROLABE_MODEL_EXECUTION", "in_process" if lane == "direct" else "daemon")
    monkeypatch.setenv("ASTROLABE_MODEL_OWNER", "trajectory")
    args = SimpleNamespace(source=Path("/tmp/as5m7-baseline-src"), output=output,
                           config=tmp_path / "input.yaml", lane=lane, envelope=None)
    assert H.worker(args) == 0
    report = json.loads((output / "worker.json").read_text())
    assert public_calls == ["Astrolabe.run"]
    assert report["cpu_affinity"] == sorted(__import__("os").sched_getaffinity(0))
    assert len(report["esmfold_infer_payloads"]) == 3
    if lane == "direct":
        assert all("return_residue_plddt" not in p for p in report["esmfold_infer_payloads"])
        assert report["esmfold_infer_payloads_origin"] == "reconstructed_baseline_infer_calls"
    else:
        assert report["esmfold_infer_payloads"] == [r["payload"] for r in native_requests()]
        assert report["esmfold_infer_payloads_origin"] == "captured_native_requests"
    assert report["torch_load_count"] == 1
    assert report["score_sequence_columns"] == ["pA_sequence"]
    assert report["canonicalizations"] == []
    with (output / "evidence.pkl").open("rb") as handle:
        assert pickle.load(handle)["canonicalizations"] == []
    assert H.compare_workers(output, output)["equal"]
