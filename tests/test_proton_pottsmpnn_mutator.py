from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import ProtomerTypes
from astrolabe.mutators import proton_pottsmpnn as adapter
from astrolabe.mutators.base import MutatorRegistry
from astrolabe.mutators.registry import MutatorRegistry as Registry
from astrolabe.mutators.wrappers import proton_pottsmpnn_daemon as daemon
from astrolabe.mutators.wrappers.proton_pottsmpnn_runner import ProtonPottsMPNNRunner
from astrolabe.runtime_requirements import task_requirements


def _executable(path: Path) -> Path:
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _runner_config(tmp_path: Path) -> dict[str, object]:
    repo = tmp_path / "proton"
    engine = (
        repo
        / "foundry"
        / "models"
        / "mpnn"
        / "src"
        / "mpnn"
        / "inference_engines"
        / "potts_mpnn_ph.py"
    )
    engine.parent.mkdir(parents=True, exist_ok=True)
    engine.write_text("# pinned engine stub\n", encoding="utf-8")
    checkpoint = repo / "checkpoint.ckpt"
    checkpoint.write_text("stub\n", encoding="utf-8")
    return {
        "python_bin": sys.executable,
        "repo_path": str(repo),
        "checkpoint_path": str(checkpoint),
        "hbplus_bin": str(_executable(tmp_path / "hbplus")),
        "extended_vocab": "v6",
        "device": "cpu",
        "daemon": True,
        "output_dir": str(tmp_path / "out"),
        "startup_timeout": 5,
    }


def test_factory_registers_both_proton_names() -> None:
    assert MutatorRegistry.get("proton_pottsmpnn") is Registry.get("Proton-PottsMPNN")


def test_default_criteria_use_v6_neutral_histidine_mapping() -> None:
    criteria = adapter._normalize_criteria({"method": "block_descent"})

    assert criteria["seed_source"] == "inverse"
    assert criteria["dep_map"] == {
        "HIS-P": ["HIS-S"],
        "ASP-P": ["ASP-D"],
        "GLU-P": ["GLU-D"],
    }
    assert criteria["dep_map"]["HIS-P"] != ["HID", "HIE"]


def test_criteria_can_override_but_missing_v6_keys_are_filled() -> None:
    criteria = adapter._normalize_criteria(
        {"dep_map": {"ASP-P": ["ASP-D"]}, "seed_source": "inverse"}
    )

    assert criteria["dep_map"]["ASP-P"] == ["ASP-D"]
    assert criteria["dep_map"]["HIS-P"] == ["HIS-S"]
    assert criteria["dep_map"]["GLU-P"] == ["GLU-D"]


def test_pin_reserve_and_center_free_budget() -> None:
    block = {"method": "block_descent", "center_count": 1}
    assert adapter.estimate_pin_count(block) == 1
    assert adapter.resolve_designable_cap(block, 8, binder_length=10) == (7, 1)
    assert adapter.resolve_designable_cap(
        {**block, "max_mutations": 4}, 8, binder_length=10
    ) == (4, 1)

    centre_free = {
        "method": "greedy_energy_block",
        "infill_scope": "chain",
        "center_count": 0,
    }
    assert adapter.estimate_pin_count(centre_free) == 0
    assert adapter.resolve_designable_cap(centre_free, 8, binder_length=10) == (8, 0)
    assert adapter.resolve_designable_cap(
        {**centre_free, "max_mutations": 3}, 8, binder_length=10
    ) == (3, 0)


def test_pin_budget_must_be_strictly_below_gh_budget() -> None:
    with pytest.raises(ValueError, match="must exceed"):
        adapter.resolve_designable_cap(
            {"method": "block_descent", "center_count": 1}, 1, binder_length=10
        )
    with pytest.raises(ValueError, match="must exceed"):
        adapter.resolve_designable_cap(
            {"method": "block_descent", "center_count": 2}, 2, binder_length=10
        )


def test_center_free_rejects_implicit_or_invalid_modes() -> None:
    with pytest.raises(ValueError, match="centre-free form"):
        adapter.resolve_designable_cap(
            {"method": "greedy_energy_block", "center_count": 0},
            8,
            binder_length=10,
        )
    with pytest.raises(ValueError, match="exceeds the GH mutation budget"):
        adapter.resolve_designable_cap(
            {
                "method": "greedy_energy_block",
                "infill_scope": "chain",
                "center_count": 0,
                "max_mutations": 9,
            },
            8,
            binder_length=10,
        )


def test_whole_chain_requires_budget_for_entire_binder() -> None:
    assert adapter.resolve_designable_cap(
        {"method": "gibbs"}, 5, binder_length=5
    ) == (5, 0)
    with pytest.raises(ValueError, match="whole binder"):
        adapter.resolve_designable_cap({"method": "gibbs"}, 4, binder_length=5)


def test_canonical_edits_are_joint_and_respect_fixed_positions() -> None:
    assert adapter.canonical_edits("ACDE", "ACDE") == []
    assert adapter.canonical_edits("ACDE", "ACDF") == [(3, "E", "F")]

    with pytest.raises(ValueError, match="fixed position"):
        adapter.canonical_edits("ACDE", "ACDF", fixed_positions=[3])
    with pytest.raises(ValueError, match="non-canonical"):
        adapter.canonical_edits("ACDE", "ACDX")
    with pytest.raises(ValueError, match="length"):
        adapter.canonical_edits("ACDE", "ACD")


def test_runner_requires_persistent_v6_configuration(tmp_path: Path) -> None:
    config = _runner_config(tmp_path)
    runner = ProtonPottsMPNNRunner(config)
    assert runner.extended_vocab == "v6"
    runner.close()

    config["daemon"] = False
    with pytest.raises(ValueError, match="persistent daemon"):
        ProtonPottsMPNNRunner(config)

    config = _runner_config(tmp_path)
    config["extended_vocab"] = "v4"
    with pytest.raises(ValueError, match="extended_vocab='v6'"):
        ProtonPottsMPNNRunner(config)


def test_daemon_criteria_default_and_seed_energies_are_preserved(
    tmp_path: Path, monkeypatch
) -> None:
    class Criteria:
        __dataclass_fields__ = {
            "seed_source": None,
            "dep_map": None,
            "max_mutations": None,
        }

        @staticmethod
        def from_params(params):
            return dict(params)

    class Design:
        canonical_sequence = "ACDE"
        extended_tokens = ["ALA", "CYS", "ASP-P", "GLU-S"]
        energy_trajectory = None

        def to_metadata(self):
            return {"final_potts_energy": -1.25}

    class DesignSet(list):
        seed_energies = [{"potts_energy": -0.5}]

    seen: dict[str, object] = {}

    class Engine:
        def run_ph_redesign(self, **kwargs):
            seen["criteria"] = kwargs["criteria_list"][0]
            return DesignSet([Design()])

    monkeypatch.setattr(daemon, "_load_atom_array", lambda _path: object())
    monkeypatch.setattr(
        daemon,
        "_token_chain_sequences",
        lambda _atom_array: (None, {"A": "ACDE"}),
    )
    monkeypatch.setattr(daemon, "_apply_design_mask", lambda *_args: None)

    pdb_path = tmp_path / "complex.pdb"
    pdb_path.write_text("stub\n", encoding="utf-8")
    result = daemon._infer(
        (Engine(), Criteria),
        {
            "id": "request-1",
            "pdb_path": str(pdb_path),
            "binder_chain": "A",
            "binder_sequence": "ACDE",
            "chain_sequences": {"A": "ACDE"},
            "fixed_positions": [1],
            "criteria": {},
            "max_mutations": 2,
            "n_jobs": 1,
            "seed": 7,
        },
        tmp_path / "out",
    )

    assert seen["criteria"]["dep_map"]["HIS-P"] == ["HIS-S"]
    assert result["seed_energies"] == [{"potts_energy": -0.5}]
    raw = json.loads(Path(result["raw_output_path"]).read_text(encoding="utf-8"))
    assert raw["seed_energies"] == [{"potts_energy": -0.5}]


def test_runtime_requirements_cover_proton_daemon_and_hbplus() -> None:
    requirements = task_requirements(
        {
            "mutator": {
                "mutator_terms": {
                    "ph_design": {"_type": "proton_pottsmpnn"},
                }
            },
            "stages": {"default": {"mutators": ["ph_design"]}},
        }
    )

    assert requirements["mutator_types"] == ["proton_pottsmpnn"]
    assert {
        "weight_path.proton_pottsmpnn",
        "subprocess_envs.proton_pottsmpnn.python_bin",
        "subprocess_envs.proton_pottsmpnn.repo_path",
        "subprocess_envs.proton_pottsmpnn.hbplus_bin",
    }.issubset(set(requirements["required_env_keys"]))
    assert any(
        item["label"] == "proton_pottsmpnn_source"
        for item in requirements["required_env_repo_files"]
    )


def test_factory_deep_merges_partial_term_criteria(tmp_path: Path) -> None:
    from astrolabe.mutators.base import build_mutators
    from omegaconf import OmegaConf

    runner_config = _runner_config(tmp_path)
    runner_config["criteria"] = {
        "method": "block_descent",
        "backend": "potts",
        "center_count": 1,
        "topk_sites": 1,
        "samples_per_site": 1,
        "neighbour_k": 16,
        "combined_lambda": 0.3,
    }
    mutator_config = OmegaConf.create(
        {
            "ProtonPottsMPNN": runner_config,
            "mutator_terms": {
                "ph_design": {
                    "_type": "proton_pottsmpnn",
                    "criteria": {"combined_lambda": 0.4},
                }
            },
        }
    )
    global_config.reset()
    try:
        mutators = build_mutators(mutator_config, tmp_path / "raw")
        criteria = mutators["ph_design"].criteria
        assert criteria["method"] == "block_descent"
        assert criteria["topk_sites"] == 1
        assert criteria["samples_per_site"] == 1
        assert criteria["neighbour_k"] == 16
        assert criteria["combined_lambda"] == 0.4
        assert criteria["dep_map"]["HIS-P"] == ["HIS-S"]
        mutators["ph_design"].shutdown()
    finally:
        global_config.reset()


def test_prepare_rejects_multiple_active_complexes_before_model(monkeypatch) -> None:
    class Proto:
        def __init__(self, name: str, sequence: str, mutable: bool) -> None:
            self.name = name
            self.sequence = sequence
            self.type = ProtomerTypes.PROTEIN
            self.mutable_mask = np.full(len(sequence), mutable, dtype=bool)
            self.tied_positions = []
            self.ptm_dict = {}
            self.components = [name]
            self.components_slices = [slice(0, len(sequence))]

    protomers = {
        "P1": Proto("P1", "ACD", True),
        "T1": Proto("T1", "GG", False),
        "P2": Proto("P2", "EFG", True),
        "T2": Proto("T2", "HH", False),
    }
    scores = SimpleNamespace(
        names=[("P1", "T1"), ("P2", "T2")],
        complex_paths={
            ("P1", "T1"): "/tmp/one.pdb",
            ("P2", "T2"): "/tmp/two.pdb",
        },
    )
    monkeypatch.setattr(adapter, "prepare_exclude_sites", lambda *_args, **_kwargs: {})

    mutator = adapter.ProtonPottsMPNNMutator(
        sequence_model_runner=SimpleNamespace(close=lambda: None)
    )
    global_config.reset()
    global_config.set_variable("mutation_num", 8)
    try:
        with pytest.raises(ValueError, match="exactly one active complex"):
            mutator.prepare_mutation_input(protomers, scores, runtime_config=None)
    finally:
        global_config.reset()


def test_v3_v4_neutral_histidine_dep_map_is_rejected() -> None:
    with pytest.raises(ValueError, match="HIS-S"):
        adapter._normalize_criteria({"dep_map": {"HIS-P": ["HID", "HIE"]}})

    class Criteria:
        __dataclass_fields__ = {
            "seed_source": None,
            "dep_map": None,
            "max_mutations": None,
        }

    with pytest.raises(ValueError, match="HIS-S"):
        daemon._validate_criteria(
            Criteria, {"dep_map": {"HIS-P": ["HID"]}}, 2
        )


def test_daemon_rejects_explicit_centre_at_fixed_position() -> None:
    token_aa = SimpleNamespace(
        chain_id=np.array(["A", "A", "B"]),
        res_id=np.array([10, 11, 1]),
    )
    with pytest.raises(ValueError, match="fixed position"):
        daemon._validate_explicit_centers(
            {"explicit_centers": [{"res_id": 10, "protonation_type": "HIS-P"}]},
            token_aa,
            "A",
            [0],
        )
    with pytest.raises(ValueError, match="not in binder"):
        daemon._validate_explicit_centers(
            {"explicit_centers": [{"res_id": 99, "protonation_type": "HIS-P"}]},
            token_aa,
            "A",
            [0],
        )


def test_excluded_amino_acids_translate_to_v6_tokens() -> None:
    assert adapter.tokens_for_excluded_aas(["C"]) == ["CYS"]
    assert adapter.tokens_for_excluded_aas("CYS") == ["CYS"]
    histidine = set(adapter.tokens_for_excluded_aas(["H"]))
    assert {"HIS-P", "HIS-S", "HIS-A", "HIS-D"}.issubset(histidine)
    assert set(adapter.tokens_for_excluded_aas(["D"])) == {
        "ASP-P",
        "ASP-D",
        "ASP-A",
    }
    assert set(adapter.tokens_for_excluded_aas(["E"])) == {
        "GLU-P",
        "GLU-D",
        "GLU-A",
    }


def test_forbidden_tokens_merge_and_centre_pool_filtering() -> None:
    criteria = adapter._normalize_criteria(
        {"method": "block_descent", "forbidden_tokens": ["ALA"]}
    )
    assert {"HIS-A", "ASP-A", "GLU-A", "HIS-D", "ALA"}.issubset(
        set(criteria["forbidden_tokens"])
    )

    filtered = adapter._apply_excluded_amino_acids(criteria, ["H", "D"])
    assert "CYS" not in filtered["forbidden_tokens"]
    assert filtered["center_protonation_types"] == ["GLU-P"]

    with pytest.raises(ValueError, match="removes every configured"):
        adapter._apply_excluded_amino_acids(criteria, ["H", "D", "E"])

    centre_free = adapter._normalize_criteria(
        {
            "method": "greedy_energy_block",
            "infill_scope": "chain",
            "center_count": 0,
        }
    )
    assert adapter._apply_excluded_amino_acids(
        centre_free, ["H", "D", "E"]
    )["center_protonation_types"] == []


def test_exclusion_conflicts_with_explicit_centres_are_preflight_errors() -> None:
    criteria = adapter._normalize_criteria(
        {
            "method": "block_descent",
            "explicit_centers": [
                {"res_id": 10, "protonation_type": "HIS-P"},
            ],
        }
    )
    with pytest.raises(ValueError, match="explicit centre"):
        adapter._apply_excluded_amino_acids(criteria, ["H"])

    exact = adapter._normalize_criteria(
        {"method": "block_descent", "center_types": ["HIS-P", "ASP-P"]}
    )
    with pytest.raises(ValueError, match="center_types"):
        adapter._apply_excluded_amino_acids(exact, ["H"])

    with pytest.raises(ValueError, match="mpnn_sample"):
        adapter._apply_excluded_amino_acids(
            adapter._normalize_criteria({"method": "mpnn_sample"}), ["C"]
        )


def test_prepare_passes_excluded_c_and_keeps_fixed_c_undesignable(
    monkeypatch, tmp_path: Path
) -> None:
    class Proto:
        def __init__(self, name: str, sequence: str) -> None:
            self.name = name
            self.sequence = sequence
            self.type = ProtomerTypes.PROTEIN
            self.mutable_mask = np.array([False, True], dtype=bool)
            self.tied_positions = []
            self.ptm_dict = {}
            self.components = [name]
            self.components_slices = [slice(0, len(sequence))]

    protomers = {"P1": Proto("P1", "CA")}
    scores = SimpleNamespace(
        names=[("P1",)],
        complex_paths={("P1",): str(tmp_path / "complex.pdb")},
    )
    monkeypatch.setattr(adapter, "prepare_exclude_sites", lambda *_args, **_kwargs: {})

    mutator = adapter.ProtonPottsMPNNMutator(
        sequence_model_runner=SimpleNamespace(close=lambda: None)
    )
    global_config.reset()
    global_config.init_const("exclude_AA", ["C"])
    global_config.set_variable("mutation_num", 8)
    try:
        prepared = mutator.prepare_mutation_input(protomers, scores, runtime_config=None)
    finally:
        global_config.reset()

    features = prepared[("P1",)]
    assert features["binder_sequence"] == "CA"
    assert features["fixed_positions"] == [0]
    assert "CYS" in features["criteria"]["forbidden_tokens"]
    assert features["criteria"]["center_protonation_types"] == [
        "HIS-P",
        "ASP-P",
        "GLU-P",
    ]
    assert adapter.canonical_edits(
        "CA", "CA", fixed_positions=features["fixed_positions"]
    ) == []


def test_selected_joint_design_rejects_forbidden_token_at_mutable_site() -> None:
    features = {
        "binder_sequence": "CA",
        "fixed_positions": [0],
        "criteria": {"forbidden_tokens": ["CYS"]},
    }
    adapter._assert_no_forbidden_tokens(
        {"extended_tokens": ["CYS", "ALA"]}, features
    )
    with pytest.raises(ValueError, match="forbidden token"):
        adapter._assert_no_forbidden_tokens(
            {"extended_tokens": ["ALA", "CYS"]}, features
        )


def test_greedy_energy_block_rejects_existing_mutable_excluded_residue(
    monkeypatch, tmp_path: Path
) -> None:
    class Proto:
        name = "P1"
        sequence = "CA"
        type = ProtomerTypes.PROTEIN
        mutable_mask = np.array([True, True], dtype=bool)
        tied_positions = []
        ptm_dict = {}
        components = ["P1"]
        components_slices = [slice(0, 2)]

    scores = SimpleNamespace(
        names=[("P1",)],
        complex_paths={("P1",): str(tmp_path / "complex.pdb")},
    )
    monkeypatch.setattr(adapter, "prepare_exclude_sites", lambda *_args, **_kwargs: {})
    mutator = adapter.ProtonPottsMPNNMutator(
        sequence_model_runner=SimpleNamespace(close=lambda: None),
        criteria={
            "method": "greedy_energy_block",
            "infill_scope": "chain",
            "center_count": 0,
        },
    )
    global_config.reset()
    global_config.init_const("exclude_AA", ["C"])
    global_config.set_variable("mutation_num", 8)
    try:
        with pytest.raises(ValueError, match="greedy_energy_block"):
            mutator.prepare_mutation_input(
                {"P1": Proto()}, scores, runtime_config=None
            )
    finally:
        global_config.reset()


def test_custom_neutral_histidine_target_is_accepted() -> None:
    criteria = adapter._normalize_criteria(
        {
            "method": "block_descent",
            "center_types": ["HIS-S"],
            "dep_map": {"HIS-S": ["HIS-P"]},
        }
    )
    assert criteria["dep_map"]["HIS-S"] == ["HIS-P"]
    # The global default charged-target mapping remains available for P centres.
    assert criteria["dep_map"]["HIS-P"] == ["HIS-S"]

    prepared = adapter._apply_excluded_amino_acids(criteria, ["C"])
    assert prepared["center_types"] == ["HIS-S"]
    assert "HIS-S" not in prepared["forbidden_tokens"]
    assert adapter.estimate_pin_count(prepared) == 1
    assert adapter.resolve_designable_cap(prepared, 8, binder_length=10) == (7, 1)

    class Criteria:
        __dataclass_fields__ = {
            "seed_source": None,
            "dep_map": None,
            "max_mutations": None,
            "center_types": None,
            "forbidden_tokens": None,
        }

    validated = daemon._validate_criteria(
        Criteria,
        {
            "seed_source": "inverse",
            "dep_map": {"HIS-S": ["HIS-P"]},
            "center_types": ["HIS-S"],
            "forbidden_tokens": ["CYS"],
            "max_mutations": 7,
        },
        8,
    )
    assert validated["dep_map"]["HIS-S"] == ["HIS-P"]
    assert validated["center_types"] == ["HIS-S"]


@pytest.mark.parametrize("timing_enabled", [False, True])
def test_daemon_subprocess_persistent_jsonl_roundtrip(tmp_path: Path, monkeypatch, caplog, timing_enabled) -> None:
    """Exercise the real daemon main loop and runner wire protocol with a stub model."""
    monkeypatch.setenv("GH_PROTON_TIMING", "1" if timing_enabled else "0")
    caplog.set_level("INFO")
    config = _runner_config(tmp_path)
    wrappers_dir = str(Path(daemon.__file__).resolve().parent)
    wrapper = tmp_path / "stub_proton_daemon.py"
    wrapper.write_text(
        '''
import sys
from types import SimpleNamespace

sys.path.insert(0, __WRAPPERS_DIR__)
import proton_pottsmpnn_daemon as daemon


class StubCriteria:
    __dataclass_fields__ = {
        "seed_source": None,
        "dep_map": None,
        "max_mutations": None,
        "forbidden_tokens": None,
        "center_types": None,
        "explicit_centers": None,
        "center_protonation_types": None,
    }

    def __init__(self, **params):
        self.params = params

    @classmethod
    def from_params(cls, params):
        return cls(**params)


class StubDesign:
    canonical_sequence = "ACDE"
    extended_tokens = ["ALA", "CYS", "ASP-P", "GLU-S"]
    energy_trajectory = None

    def to_metadata(self):
        return {"final_potts_energy": -1.25, "selective_energy": -0.5}


class StubDesignSet(list):
    seed_energies = [{"potts_energy": -0.75}]


class StubEngine:
    def __init__(self):
        self.calls = []

    def run_ph_redesign(self, **kwargs):
        self.calls.append(kwargs)
        return StubDesignSet([StubDesign()])


def _load_runtime(_args):
    return StubEngine(), StubCriteria


daemon._load_runtime = _load_runtime
daemon._load_atom_array = lambda _path: object()
daemon._token_chain_sequences = lambda _atom_array: (
    SimpleNamespace(chain_id=["A", "A", "A", "A"], res_id=[1, 2, 3, 4]),
    {"A": "ACDE"},
)
daemon._apply_design_mask = lambda *_args: None

raise SystemExit(daemon.main())
'''.replace("__WRAPPERS_DIR__", repr(wrappers_dir)),
        encoding="utf-8",
    )
    config["daemon_script_path"] = str(wrapper)
    config["request_timeout"] = 5
    pdb_path = tmp_path / "complex.pdb"
    pdb_path.write_text("stub\n", encoding="utf-8")

    runner = ProtonPottsMPNNRunner(config)
    try:
        first = runner.design_sequences(
            pdb_path=str(pdb_path),
            binder_chain="A",
            binder_sequence="ACDE",
            chain_sequences={"A": "ACDE"},
            fixed_positions=[1],
            criteria={},
            max_mutations=2,
            n_jobs=1,
        )
        second = runner.design_sequences(
            pdb_path=str(pdb_path),
            binder_chain="A",
            binder_sequence="ACDE",
            chain_sequences={"A": "ACDE"},
            fixed_positions=[1],
            criteria={},
            max_mutations=2,
            n_jobs=1,
        )

        assert first["daemon_pid"] == second["daemon_pid"]
        assert first["model_load_count"] == second["model_load_count"] == 1
        assert first["request_count"] == 1
        assert second["request_count"] == 2
        assert first["request_id"] != second["request_id"]
        assert first["selected"]["canonical_sequence"] == "ACDE"
        assert first["seed_energies"] == [{"potts_energy": -0.75}]
        assert ("phase=daemon_ready" in caplog.text) == timing_enabled
        assert caplog.text.count("phase=request_roundtrip") == (2 if timing_enabled else 0)

        first_raw = json.loads(
            Path(first["raw_output_path"]).read_text(encoding="utf-8")
        )
        second_raw = json.loads(
            Path(second["raw_output_path"]).read_text(encoding="utf-8")
        )
        assert first_raw["type"] == "proton_pottsmpnn_raw"
        assert first_raw["request_id"] == first["request_id"]
        assert second_raw["request_id"] == second["request_id"]
        assert first_raw["request_id"] != second_raw["request_id"]
        assert first_raw["seed_energies"] == [{"potts_energy": -0.75}]
        assert first_raw["selected"]["canonical_sequence"] == "ACDE"
        assert len(first_raw["designs"]) == 1
        assert len(second_raw["designs"]) == 1

        proc = runner._proc
        assert proc is not None and proc.poll() is None
        runner.close()
        assert proc.poll() == 0
    finally:
        if not runner._closed:
            try:
                runner.close()
            except Exception:
                if runner._proc is not None and runner._proc.poll() is None:
                    runner._proc.kill()
                    runner._proc.wait()


def test_hbplus_export_resolves_without_breaking_other_backends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HBPLUS is a plain env_var export, not an OmegaConf interpolation.

    A ``${GH_...}`` value raised InterpolationKeyError for every backend when
    the variable was unset.  ``$GH_...`` follows the existing external-env
    convention (``python_bin: $GH_...``): env_setup expands it at selection
    time, and an unset variable remains visible as a missing required path.
    """
    import importlib.util

    from omegaconf import OmegaConf

    repo_root = Path(__file__).resolve().parents[1]
    spec_path = repo_root / "config" / "third_party_envs.yaml"
    monkeypatch.delenv("GH_PROTON_POTTSMPNN_HBPLUS", raising=False)

    spec = OmegaConf.to_container(OmegaConf.load(str(spec_path)), resolve=True)
    proton = next(env for env in spec["envs"] if env["name"] == "proton_pottsmpnn")
    entry = proton["export"]["subprocess_envs.proton_pottsmpnn.hbplus_bin"]
    assert entry["env_var"] == "GH_PROTON_POTTSMPNN_HBPLUS"
    assert entry["value"] == "$GH_PROTON_POTTSMPNN_HBPLUS"

    module_spec = importlib.util.spec_from_file_location(
        "_test_hbplus_env_setup",
        repo_root / "scripts" / "installation" / "env_setup.py",
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    subs = {"REPO_ROOT": str(repo_root), "PREFIX": str(tmp_path / "env")}

    value, meta = module._resolve_export_entry("hbplus", entry, subs, {})
    assert value == "$GH_PROTON_POTTSMPNN_HBPLUS"
    missing, unresolved = module._collect_export_path_issues(
        {"hbplus": value}, {"hbplus": meta}
    )
    assert missing == [("hbplus", "$GH_PROTON_POTTSMPNN_HBPLUS")]
    assert unresolved == []

    hbplus = _executable(tmp_path / "hbplus")
    value, meta = module._resolve_export_entry(
        "hbplus", entry, subs, {"GH_PROTON_POTTSMPNN_HBPLUS": str(hbplus)}
    )
    missing, unresolved = module._collect_export_path_issues(
        {"hbplus": value}, {"hbplus": meta}
    )
    assert value == str(hbplus)
    assert missing == []
    assert unresolved == []


@pytest.mark.parametrize("enabled", [False, True])
def test_daemon_timing_never_writes_protocol_stdout(monkeypatch, capsys, enabled):
    import time
    monkeypatch.setenv("GH_PROTON_TIMING", "1" if enabled else "0")
    daemon._timing("CPU_contract", time.perf_counter(), "request-1")
    captured = capsys.readouterr()
    assert captured.out == ""
    if enabled:
        event = json.loads(captured.err)
        assert event["event"] == "proton_timing"
        assert event["request_id"] == "request-1"
        assert event["duration_ms"] >= 0
    else:
        assert captured.err == ""
