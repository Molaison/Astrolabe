from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
from omegaconf import OmegaConf
import pytest

from astrolabe.core.folding_input import Input, RnaChain
from astrolabe.evaluators.rna import RNA_BACKENDS, RnaEvaluator, _backend_kernel
from astrolabe.evaluators.utils import construct_evaluators


def rna_input(*sequences):
    return Input(
        name="rna",
        chains=[
            RnaChain(id=chr(65 + i), sequence=seq, modifications=[])
            for i, seq in enumerate(sequences)
        ],
        rng_seeds=[17],
    )


@pytest.mark.parametrize("backend", RNA_BACKENDS)
def test_registry_is_cold_and_rejects_oligomers(backend, tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Invalid RNA inputs must not start a model owner")

    monkeypatch.setattr("astrolabe.model_execution.NativeModelClient", unexpected)
    imported_before = f"astrolabe.evaluators.wrappers.{backend}" in sys.modules
    repo, weights = tmp_path / "repo", tmp_path / "weights"
    repo.mkdir(); weights.mkdir()
    # Empty asset markers satisfy startup paths; the cold registry test never loads them.
    files = [repo / name for name in ("cfg_for_selection.json", "cfg_for_folding.json", "Arena/Arena", "PotentialFold/Selection.py", "PotentialFold/Optimization.py")]
    files.append(weights / "RCLM/epoch_67000")
    for experiment in ("cfg_95", "cfg_96", "cfg_97", "cfg_99"):
        files.extend(repo / experiment / name for name in ("EvoMSA2XYZ.py", "data.py", "util.py", "base.npy"))
        files.append(weights / experiment / "model_1.pt")
    for asset in files:
        asset.parent.mkdir(parents=True, exist_ok=True)
        asset.touch()
    config = OmegaConf.create({
        "evaluator_terms": {"rna": {"_type": backend, "device": "cpu",
            "repo_path": str(repo), "weights_path": str(weights), "lm_weights_path": str(weights)}},
    })
    evaluator = construct_evaluators(config, tmp_path)["rna"]
    assert evaluator.model_runner is None
    assert (f"astrolabe.evaluators.wrappers.{backend}" in sys.modules) == imported_before
    with pytest.raises(ValueError, match="single RNA chain"):
        evaluator.evaluate({"AA": rna_input("ACGU", "ACGU")})


def test_rna_requests_have_private_directories_and_preserve_seed(tmp_path):
    evaluator = RnaEvaluator("numonomer", tmp_path, {"device": "cpu"})
    first = evaluator.prepare_input({"A": rna_input("ACGU")}, {"step": 2})["A"]
    second = evaluator.prepare_input({"A": rna_input("ACGU")}, {"step": 2})["A"]
    assert first["seed"] == second["seed"] == 17
    assert first["output_dir"] != second["output_dir"]
    assert Path(first["output_dir"]).parent == tmp_path


@pytest.mark.parametrize("backend", ["numonomer", "rnabpflow"])
def test_experimental_multichain_preserves_physical_occurrences(backend, tmp_path):
    evaluator = RnaEvaluator(backend, tmp_path, {"device": "cpu", "multichain_mode": "experimental"})
    request = evaluator.prepare_input({"AA": rna_input("ACGU", "ACGU")})["AA"]
    assert request["chains"] == [
        {"chain_id": "A", "sequence": "ACGU"},
        {"chain_id": "B", "sequence": "ACGU"},
    ]
    assert "sequence" not in request
    assert "chain_id" not in request
    assert evaluator.model_runner is None


@pytest.mark.parametrize("backend,reason", [("drfold2", "O3'-P bonds"), ("zfold", "Rosetta")])
def test_unimplemented_multichain_explains_topology_obstacle(backend, reason, tmp_path):
    with pytest.raises(ValueError, match=reason):
        RnaEvaluator(backend, tmp_path, {"multichain_mode": "experimental"})


def test_multichain_validates_all_chains_before_creating_outputs(tmp_path):
    evaluator = RnaEvaluator("numonomer", tmp_path / "outputs", {"multichain_mode": "experimental"})
    valid = rna_input("AC", "GU")
    bad = SimpleNamespace(chains=[valid.chains[0], object()], rng_seeds=[17])
    with pytest.raises(ValueError, match="RNA chains only"):
        evaluator.prepare_input({"AB": bad})
    assert not evaluator.output_dir.exists()


def test_multichain_confidence_keeps_full_array_and_protomer_copies(tmp_path):
    evaluator = RnaEvaluator("numonomer", tmp_path, {"multichain_mode": "experimental"})
    evaluator.model_runner = SimpleNamespace(metadata={})
    physical = rna_input("AC", "AC", "GUA")
    inputs = {"AAB": SimpleNamespace(name="AAB", chains=physical.chains, rng_seeds=[17])}
    request = evaluator.prepare_input(inputs)["AAB"]
    path = Path(request["output_dir"]) / "model.cif"
    path.write_text("fixture\n")
    values = np.arange(7, dtype=float) + 70
    result = evaluator.prepare_output({
        "AAB": {"request": request, "prediction": {"structure_path": str(path), "plddt": values}},
    }, inputs)
    # StandardScores converts native 0..100 pLDDT to 0..1.
    np.testing.assert_array_equal(result.plddt["AAB"][0], values / 100)
    np.testing.assert_array_equal(result.per_proto_scores["A"][0], values[:2] / 100)
    np.testing.assert_array_equal(result.per_proto_scores["A"][1], values[2:4] / 100)
    np.testing.assert_array_equal(result.per_proto_scores["B"][0], values[4:] / 100)


def test_missing_confidence_is_not_manufactured(tmp_path):
    evaluator = RnaEvaluator("drfold2", tmp_path, {"device": "cpu"})
    evaluator.model_runner = SimpleNamespace(metadata={"stage_contract": "fixture"})
    inputs = {"A": rna_input("ACGU")}
    request = evaluator.prepare_input(inputs)["A"]
    path = Path(request["output_dir"]) / "model.pdb"
    path.write_text("END\n")
    result = evaluator.prepare_output({
        "A": {"request": request, "prediction": {"structure_path": str(path)}},
    }, inputs)
    assert result.plddt == {}
    assert result.iptm is None
    assert result.complex_paths == {"A": path}


def test_confidence_matches_residue_count(tmp_path):
    evaluator = RnaEvaluator("numonomer", tmp_path, {"device": "cpu"})
    evaluator.model_runner = SimpleNamespace(metadata={})
    inputs = {"A": rna_input("ACGU")}
    request = evaluator.prepare_input(inputs)["A"]
    path = Path(request["output_dir"]) / "model.pdb"
    path.write_text("END\n")
    with pytest.raises(ValueError, match="per-residue confidence"):
        evaluator.prepare_output({
            "A": {
                "request": request,
                "prediction": {"structure_path": str(path), "plddt": np.ones(3)},
            },
        }, inputs)


def test_legacy_runtime_cannot_claim_broker_stage_admission(monkeypatch):
    from astrolabe import model_execution

    monkeypatch.delattr(model_execution, "EvaluatorStages", raising=False)
    monkeypatch.setenv("ASTROLABE_RESOURCE_SOCKET", "/unopened/resource.sock")
    with pytest.raises(RuntimeError, match="staged model runtime"):
        _backend_kernel({"backend": "numonomer", "device": "cpu"})
