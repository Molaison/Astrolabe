"""CPU-only input, startup and score contracts; never imports BioEmu2 weights."""
from dataclasses import replace
from pathlib import Path
import sys

import pytest
from omegaconf import OmegaConf

from astrolabe.backend_catalog import get_backend
from astrolabe.backend_startup import validate_backend_settings
from astrolabe.core.folding_input import Input, Ligand, ProteinChain
from astrolabe.evaluators.bioemu2 import BioEmu2Evaluator
from astrolabe.evaluators.bioemu2_contract import validate_structure_only_scores
from astrolabe.evaluators.utils import construct_evaluators
from astrolabe.runtime_requirements import task_requirements


@pytest.fixture
def options(tmp_path):
    return dict(python_bin=sys.executable, cache_dir=str(tmp_path))


def test_factory_and_input_seed_do_not_start_worker(options, tmp_path):
    conf = OmegaConf.create({"evaluator_terms": {"emu": {"_type": "bioemu2", **options}}})
    evaluator = construct_evaluators(conf, tmp_path)["emu"]
    data = Input(name="A", chains=[ProteinChain(id="A", sequence="ACDE", ptms=[])], rng_seeds=[17])
    request = evaluator.prepare_input({"A": data}, {})["A"]
    assert request == {"chains": [{"id": "A", "sequence": "ACDE"}], "input_seed": 17}
    assert not evaluator.output_dir.exists()
    scores = evaluator.prepare_output({"A": Path("representative.cif")}, {"A": data}, {})
    assert scores.plddt == {} and scores.ptm is None and scores.iptm is None and scores.pae is None
    assert get_backend("bioemu2").execution == "subprocess"


@pytest.mark.parametrize("change", [
    {"chains": [Ligand(id="A", ccd_ids=["CFF"])]},
    {"chains": [ProteinChain(id="A", sequence="ACDE", ptms=[("SEP", 2)])]},
    {"chains": [ProteinChain(id="A", sequence="ACDX", ptms=[])]},
    {"bonded_atom_pairs": [("A", 1, "SG", "A", 2, "SG")]},
    {"cyclic_chain_ids": ["A"]},
    {"cyclic_relative_patches": [object()]},
])
def test_unsupported_inputs_fail_before_subprocess(options, tmp_path, change):
    evaluator = BioEmu2Evaluator(output_dir=tmp_path / "raw", **options)
    data = Input(name="A", chains=[ProteinChain(id="A", sequence="ACDE", ptms=[])], rng_seeds=[17])
    with pytest.raises(ValueError):
        evaluator.prepare_input({"A": replace(data, **change)}, {})
    assert not evaluator.output_dir.exists()


@pytest.mark.parametrize("change,field", [
    ({"num_samples": True}, "num_samples"), ({"batch_size": 1.5}, "batch_size"),
    ({"propagation_steps": 0}, "propagation_steps"), ({"lag_time_ns": float("nan")}, "lag_time_ns"),
    ({"seed": -1}, "seed"), ({"mode": "propagation"}, "initial_cif"),
    ({"mode": "other"}, "mode"), ({"sampler": "other"}, "sampler"),
    ({"cache_dir": "/missing-bioemu-cache"}, "cache_dir"),
    ({"execution_mode": "daemon"}, "execution_mode"),
])
def test_selected_preflight(options, change, field):
    errors = validate_backend_settings("evaluator", "bioemu2", {**options, **change},
                                      path="evaluator.emu", check_paths=True)
    assert any(f".{field}:" in error for error in errors)


def test_propagation_file_and_mode_relationship(options, tmp_path):
    initial = tmp_path / "initial.cif"
    initial.write_text("data_test\n")
    assert not validate_backend_settings("evaluator", "bioemu2", {
        **options, "mode": "propagation", "initial_cif": str(initial),
    }, path="emu", check_paths=True)
    with pytest.raises(ValueError, match="initial_cif"):
        BioEmu2Evaluator(output_dir=tmp_path / "raw", initial_cif=initial, **options)


def test_no_confidence_losses_selectors_or_terms():
    active = [("emu", "bioemu2", None, {})]
    errors = validate_structure_only_scores(
        active_evaluators=active,
        active_mutators=[("mutate", "mcmc", {"position_selector": "bfactor(0, 70)"})],
        active_losses={"confidence": {"_type": "cross_iptm"}},
        per_residue_scores={"mutator_score_terms": {"plddt": 1}},
    )
    assert len(errors) == 3
    assert not validate_structure_only_scores(
        active_evaluators=active, active_mutators=[],
        active_losses={"rg": {"_type": "radius_of_gyration"}},
    )


def test_requirements_are_local_workers_not_coordinator_bioemu_imports():
    requirements = task_requirements({"enable_pyrosetta": False, "evaluator": {
        "evaluator_terms": {"emu": {"_type": "bioemu2"}},
    }})
    assert "astrolabe/evaluators/wrappers/bioemu2_worker.py" in requirements["required_repo_files"]
    assert "bioemu2" not in requirements["required_imports"]
    from astrolabe.evaluator_startup import validate_independent_plan
    with pytest.raises(ValueError, match="independent initialization is unsupported"):
        validate_independent_plan([("emu", "bioemu2", None, {})], 17)


def test_builtin_confidence_expression_is_rejected_without_defined_loss():
    errors = validate_structure_only_scores(
        active_evaluators=[("emu", "bioemu2", None, {})], active_mutators=[],
        active_losses={}, stages={"default": {"evaluator": "emu", "weighted_score_expression": "quality"}},
        expression_terms={"quality": "plddt"},
    )
    assert any("no confidence metrics" in error and "plddt" in error for error in errors)


def test_confidence_in_separate_prediction_stage_does_not_reject_structure_stage():
    errors = validate_structure_only_scores(
        active_evaluators=[("emu", "bioemu2", None, {}), ("fold", "alphafold3", None, {})],
        active_mutators=[("confidence_mutator", "mcmc", {"position_selector": "bfactor(0, 70)"})],
        active_losses={"confidence": {"_type": "plddt"}, "rg": {"_type": "radius_of_gyration"}},
        stages={
            "default": {"evaluator": "emu", "weighted_score_expression": "rg",
                        "active_loss_terms": ["rg"], "mutators": ["plain"]},
            "refold": {"evaluator": "fold", "weighted_score_expression": "confidence",
                       "active_loss_terms": ["confidence"], "mutators": ["confidence_mutator"]},
        },
    )
    assert errors == []
