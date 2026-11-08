from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import ProtomerTypes
from astrolabe.mutators import proton_pottsmpnn as adapter
from astrolabe.mutators.base import build_mutators
from astrolabe.mutators.wrappers import proton_pottsmpnn_runner as runner_module
from astrolabe.runtime_requirements import (
    _pyrosetta_loss_types_from_source,
    task_requirements,
)


class _CloseableRunner:
    def __init__(self, config=None, *, name="runner"):
        self.config = config
        self.name = name
        self.closed = False

    def close(self):
        self.closed = True


def _factory_config(*, runtime_key=None):
    proton = {
        "python_bin": "/not/started/python",
        "repo_path": "/not/started/proton",
        "checkpoint_path": "/not/started/checkpoint.ckpt",
        "hbplus_bin": "/not/started/hbplus",
        "device": "cpu",
        "daemon": True,
        "criteria": {
            "method": "greedy_energy_block",
            "infill_scope": "chain",
            "center_count": 0,
        },
    }
    if runtime_key is not None:
        proton["runtime_key"] = runtime_key
    return OmegaConf.create(
        {
            "ProtonPottsMPNN": proton,
            "mutator_terms": {
                "ph_core_design": {"_type": "proton_pottsmpnn"},
            },
        }
    )


def test_factory_defaults_to_private_runner(monkeypatch, tmp_path):
    private_calls = []
    shared_calls = []

    class PrivateRunner(_CloseableRunner):
        def __init__(self, config):
            super().__init__(config, name="private")
            private_calls.append(config)

    def forbidden_shared(**kwargs):
        shared_calls.append(kwargs)
        raise AssertionError("default Proton construction must not opt into sharing")

    monkeypatch.setattr(runner_module, "ProtonPottsMPNNRunner", PrivateRunner)
    monkeypatch.setattr(runner_module, "get_shared_proton_runner", forbidden_shared)

    mutators = build_mutators(_factory_config(), tmp_path / "raw")

    assert mutators["ph_core_design"].runner.name == "private"
    assert len(private_calls) == 1
    assert shared_calls == []


def test_factory_routes_network_geometry_to_mutator_not_runner(monkeypatch, tmp_path):
    monkeypatch.setattr(runner_module, "ProtonPottsMPNNRunner", _CloseableRunner)
    config = _factory_config()
    config.ProtonPottsMPNN.criteria.network_pair_bias = 3.0
    config.mutator_terms.ph_core_design.network_proposal = {"enabled": True, "max_preferences": 8}
    mutator = build_mutators(config, tmp_path / "raw")["ph_core_design"]
    assert mutator.network_proposal.enabled
    assert mutator.network_proposal.max_preferences == 8
    assert "network_proposal" not in mutator.runner.config


def test_factory_runtime_key_opts_into_shared_runner(monkeypatch, tmp_path):
    private_calls = []
    shared_calls = []
    shared_runner = _CloseableRunner(name="shared")

    class PrivateRunner(_CloseableRunner):
        def __init__(self, config):
            super().__init__(config, name="private")
            private_calls.append(config)

    def get_shared(*, config, key):
        shared_calls.append((config, key))
        return shared_runner

    monkeypatch.setattr(runner_module, "ProtonPottsMPNNRunner", PrivateRunner)
    monkeypatch.setattr(runner_module, "get_shared_proton_runner", get_shared)

    mutators = build_mutators(
        _factory_config(runtime_key="automatic-test"),
        tmp_path / "raw",
    )

    assert mutators["ph_core_design"].runner is shared_runner
    assert private_calls == []
    assert len(shared_calls) == 1
    assert shared_calls[0][1] == "automatic-test"
    assert "runtime_key" not in shared_calls[0][0]


class _MutatorProtomer:
    def __init__(self, name: str, sequence: str) -> None:
        self.name = name
        self.sequence = sequence
        self.type = ProtomerTypes.PROTEIN
        self.mutable_mask = np.ones(len(sequence), dtype=bool)
        self.tied_positions = []
        self.ptm_dict = {}
        self.components = [name]
        self.components_slices = [slice(0, len(sequence))]


def test_centre_free_mutator_passes_explicit_parent_and_total_budget(monkeypatch, tmp_path):
    protomer = _MutatorProtomer("P1", "AAAA")
    scores = SimpleNamespace(
        names=[("P1",)],
        complex_paths={("P1",): tmp_path / "complex.pdb"},
    )
    mutator = adapter.ProtonPottsMPNNMutator(
        sequence_model_runner=_CloseableRunner(),
        criteria={
            "method": "greedy_energy_block",
            "infill_scope": "chain",
            "center_count": 0,
        },
        reference_sequence="AAAA",
        max_total_mutations=2,
    )
    monkeypatch.setattr(adapter, "prepare_exclude_sites", lambda *_a, **_k: {})
    global_config.reset()
    global_config.init_const("exclude_AA", [])
    global_config.set_variable("mutation_num", 8)
    try:
        prepared = mutator.prepare_mutation_input(
            {"P1": protomer},
            scores,
            runtime_config=None,
        )
    finally:
        global_config.reset()

    features = prepared[("P1",)]
    assert features["binder_sequence"] == "AAAA"
    assert features["fixed_positions"] == []
    assert features["max_mutations"] == 8
    assert features["pin_budget"] == 0
    assert features["criteria"]["mutation_reference_sequence"] == "AAAA"
    assert features["criteria"]["max_total_mutations"] == 2
    assert features["criteria"]["center_count"] == 0


def test_mutator_rejects_output_beyond_original_parent_budget(monkeypatch):
    mutator = adapter.ProtonPottsMPNNMutator(
        sequence_model_runner=_CloseableRunner(),
        criteria={
            "method": "greedy_energy_block",
            "infill_scope": "chain",
            "center_count": 0,
        },
        reference_sequence="AAAA",
        max_total_mutations=2,
    )
    features = {
        "binder_protomer": "P1",
        "binder_sequence": "AAAA",
        "fixed_positions": [],
        "criteria": {"forbidden_tokens": []},
    }
    output = {
        ("P1",): {
            "feature_dict": features,
            "selected": {
                "canonical_sequence": "CCCC",
                "extended_tokens": ["CYS", "CYS", "CYS", "CYS"],
                "final_potts_energy": -1.0,
            },
        }
    }
    global_config.reset()
    global_config.init_const("protomers", {"P1": SimpleNamespace(name="P1")})
    global_config.set_variable("mutation_num", 8)
    try:
        with pytest.raises(ValueError, match="original-parent mutation budget"):
            mutator.prepare_mutation_output(
                output,
                {"P1": SimpleNamespace(name="P1")},
                evaluation_scores=None,
                runtime_config=None,
            )
    finally:
        global_config.reset()


def test_runtime_requirements_cover_ph_core_pyrosetta_and_proton():
    config = {
        "evaluator": {"evaluator_terms": {}},
        "mutator": {"mutator_terms": {}},
        "loss": {
            "loss_terms": {
                "core_release": {
                    "_type": "ph_core_network",
                    "binder_chain": "A",
                    "metric": "release",
                    "runner_key": "automatic-test",
                }
            }
        },
        "stages": {
            "default": {
                "mutators": [],
                "weighted_score_expression": "core_release",
            }
        },
    }
    _pyrosetta_loss_types_from_source.cache_clear()
    requirements = task_requirements(config)

    assert "ph_core_network" in requirements["loss_types"]
    assert "pyrosetta" in requirements["required_imports"]
    assert {
        "weight_path.proton_pottsmpnn",
        "subprocess_envs.proton_pottsmpnn.python_bin",
        "subprocess_envs.proton_pottsmpnn.repo_path",
        "subprocess_envs.proton_pottsmpnn.hbplus_bin",
    }.issubset(requirements["required_env_keys"])
    assert any(
        item["label"] == "proton_pottsmpnn_source"
        for item in requirements["required_env_repo_files"]
    )


@pytest.mark.parametrize(
    ("term_flags", "default_flags"),
    [
        ({"protect_gly_pro": True}, {"preserve_disulfides": False}),
        ({"network_proposal": {"enabled": True}}, {}),
        ({}, {"network_proposal": {"enabled": True}}),
        (
            {},
            {"preserve_disulfides": True, "protect_gly_pro": False},
        ),
    ],
)
def test_runtime_requirements_infer_pyrosetta_from_mutator_residue_flags(
    term_flags,
    default_flags,
):
    term = {"_type": "proton_pottsmpnn", "binder_chain": "A", **term_flags}
    config = {
        "evaluator": {"evaluator_terms": {}},
        "mutator": {
            "ProtonPottsMPNN": default_flags,
            "mutator_terms": {"ph_core_design": term},
        },
        "loss": {"loss_terms": {}},
        "stages": {
            "default": {
                "mutators": ["ph_core_design"],
                "weighted_score_expression": "iptm",
            }
        },
    }

    requirements = task_requirements(config)

    assert requirements["loss_types"] == []
    assert "pyrosetta" in requirements["required_imports"]
    assert requirements["mutator_types"] == ["proton_pottsmpnn"]


def test_automatic_overlay_is_centre_free_and_rewards_release():
    root = Path(__file__).resolve().parents[1]
    overlay = OmegaConf.load(root / "examples/02_ph_core_automatic.overlay.yaml")
    criteria = overlay.mutator.mutator_terms.ph_core_design.criteria
    stage = overlay.stages.ligand

    assert criteria.method == "greedy_energy_block"
    assert criteria.infill_scope == "chain"
    assert criteria.center_count == 0
    assert list(criteria.center_types) == []
    assert list(criteria.explicit_centers) == []
    assert criteria.aa_bias_region == ["core"]
    assert criteria.aa_bias.HIS == pytest.approx(2.0)
    assert criteria.aa_bias_caps.HIS == 2

    assert overlay.loss.loss_terms.core_release.metric == "release"
    assert overlay.loss.loss_terms.core_release.runner_key == "auto_ph_core"
    expression = str(stage.weighted_score_expression)
    release_coefficient = re.search(r"([+-]?)\s*\d+(?:\.\d+)?\s*\*\s*core_release", expression)
    assert release_coefficient is not None
    assert release_coefficient.group(1) != "-"
    assert stage.acceptance_gate is None


def test_overlay_native_centre_free_passes_normalizer_and_mutator():
    root = Path(__file__).resolve().parents[1]
    overlay = OmegaConf.load(root / "examples/02_ph_core_automatic.overlay.yaml")
    raw_criteria = overlay.mutator.mutator_terms.ph_core_design.criteria

    normalized = adapter._normalize_criteria(raw_criteria)
    mutator = adapter.ProtonPottsMPNNMutator(
        sequence_model_runner=_CloseableRunner(),
        criteria=normalized,
        preserve_disulfides=True,
        protect_gly_pro=True,
    )

    assert normalized["seed_source"] == "native"
    assert normalized["center_count"] == 0
    assert normalized["infill_scope"] == "chain"
    assert normalized["center_types"] == []
    assert normalized["explicit_centers"] == []
    assert mutator.criteria["seed_source"] == "native"
    assert adapter._is_centre_free(mutator.criteria) is True
