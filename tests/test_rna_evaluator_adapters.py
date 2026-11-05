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
    config = OmegaConf.create({
        "evaluator_terms": {"rna": {"_type": backend, "device": "cpu"}},
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
