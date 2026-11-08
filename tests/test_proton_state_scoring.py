from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from astrolabe.mutators.wrappers import proton_pottsmpnn_daemon as daemon
from astrolabe.mutators.wrappers.proton_pottsmpnn_runner import (
    ProtonPottsMPNNRunner,
    _validated_histidine_positions,
    get_shared_proton_runner,
)


def _executable(path: Path) -> Path:
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _runner_config(tmp_path: Path, *, seed: int = 0) -> dict[str, object]:
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
    engine.write_text("# stub\n", encoding="utf-8")
    checkpoint = repo / "checkpoint.ckpt"
    checkpoint.write_text("stub\n", encoding="utf-8")
    return {
        "python_bin": sys.executable,
        "repo_path": str(repo),
        "checkpoint_path": str(checkpoint),
        "hbplus_bin": str(_executable(tmp_path / f"hbplus-{seed}")),
        "extended_vocab": "v6",
        "device": "cpu",
        "daemon": True,
        "output_dir": str(tmp_path / f"out-{seed}"),
        "startup_timeout": 5,
        "request_timeout": 5,
        "seed": seed,
    }


class _IndexList(list):
    def tolist(self) -> list[int]:
        return list(self)


class _Seq(list):
    def clone(self) -> "_Seq":
        return _Seq(self)


class _Scorer:
    def __init__(self, *, nonfinite: bool = False) -> None:
        self.calls: list[tuple[int, ...]] = []
        self.nonfinite = nonfinite

    def H_of(self, sequence: _Seq) -> float:
        values = tuple(int(value) for value in sequence)
        self.calls.append(values)
        if self.nonfinite:
            return float("nan")
        # Two weak single-site preferences and one explicit HH coupling.
        energy = -1.0 * ((values[1] == 2) + (values[2] == 2))
        if values[1] == 2 and values[2] == 2:
            energy -= 5.0
        return energy


def _toy_runtime(*, nonfinite: bool = False):
    token_aa = SimpleNamespace(
        chain_id=np.array(["A", "A", "A", "A"]),
        res_id=np.array([1, 2, 3, 4]),
        res_name=np.array(["ALA", "HIS", "HIS", "ALA"]),
    )
    scorer = _Scorer(nonfinite=nonfinite)
    ctx = SimpleNamespace(
        S_native=_Seq([0, 1, 1, 0]),
        chainA_all_idx=_IndexList([0, 1, 2, 3]),
        token_aa=token_aa,
        encoding=SimpleNamespace(token_to_idx={"ALA": 0, "HIS-S": 1, "HIS-P": 2}),
        scorer=scorer,
        L=4,
    )

    class Engine:
        calls = 0

        def _build_context(self, _atom_array, binder_chain):
            assert binder_chain == "A"
            self.calls += 1
            return ctx

    return Engine(), scorer, token_aa


def _score_message(tmp_path: Path, positions: list[int]) -> dict[str, object]:
    pdb_path = tmp_path / "single.pdb"
    pdb_path.write_text("stub\n", encoding="utf-8")
    return {
        "id": "score-1",
        "pdb_path": str(pdb_path),
        "binder_chain": "A",
        "binder_sequence": "AHHA",
        "chain_sequences": {"A": "AHHA"},
        "histidine_positions": positions,
    }


def test_score_states_joint_hh_gap_is_not_locked_sum(
    tmp_path: Path, monkeypatch
) -> None:
    engine, scorer, token_aa = _toy_runtime()
    monkeypatch.setattr(daemon, "_load_atom_array", lambda _path: object())
    monkeypatch.setattr(
        daemon, "_token_chain_sequences", lambda _array: (token_aa, {"A": "AHHA"})
    )

    result = daemon._score_states(
        (engine, None), _score_message(tmp_path, [1, 2]), tmp_path / "out"
    )

    assert engine.calls == 1
    assert scorer.calls == [(0, 1, 1, 0), (0, 2, 2, 0)]
    assert result["neutral_energy"] == 0.0
    assert result["protonated_energy"] == -7.0
    assert result["gap"] == -7.0
    assert result["gap"] != -2.0  # locked per-centre sum
    assert result["n_centers"] == 2
    assert result["outside_centers_unchanged"] is True
    assert Path(result["raw_output_path"]).name.startswith("score_states_")
    raw = json.loads(Path(result["raw_output_path"]).read_text(encoding="utf-8"))
    assert raw["type"] == "proton_pottsmpnn_state_score_raw"
    assert raw["gap"] == -7.0


@pytest.mark.parametrize(
    "positions",
    [
        [],
        [0],
        [1, 1],
        [4],
    ],
)
def test_score_states_reject_invalid_histidine_positions(positions: list[int]) -> None:
    with pytest.raises(ValueError):
        _validated_histidine_positions("AHHA", positions)


def test_score_states_rejects_nonfinite_energy(tmp_path: Path, monkeypatch) -> None:
    engine, _scorer, token_aa = _toy_runtime(nonfinite=True)
    monkeypatch.setattr(daemon, "_load_atom_array", lambda _path: object())
    monkeypatch.setattr(
        daemon, "_token_chain_sequences", lambda _array: (token_aa, {"A": "AHHA"})
    )

    with pytest.raises(RuntimeError, match="non-finite"):
        daemon._score_states(
            (engine, None), _score_message(tmp_path, [1, 2]), tmp_path / "out"
        )


def test_shared_runner_config_conflict_and_dead_runner_registry(tmp_path: Path) -> None:
    config = _runner_config(tmp_path, seed=0)
    key = "shared-conflict-test"
    runner = get_shared_proton_runner(config, key=key)
    try:
        assert get_shared_proton_runner(config, key=key) is runner
        with pytest.raises(ValueError, match="Conflicting"):
            get_shared_proton_runner(
                _runner_config(tmp_path, seed=1), key=key
            )
    finally:
        runner.close()

    with pytest.raises(RuntimeError, match="No live shared"):
        get_shared_proton_runner(key=key)

    replacement = get_shared_proton_runner(config, key=key)
    try:
        assert replacement is not runner
    finally:
        replacement.close()


def test_shared_subprocess_design_and_score_reuse_one_model_load(
    tmp_path: Path, monkeypatch
) -> None:
    config = _runner_config(tmp_path, seed=7)
    wrappers_dir = str(Path(daemon.__file__).resolve().parent)
    wrapper = tmp_path / "stub_state_daemon.py"
    wrapper.write_text(
        '''
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, __WRAPPERS_DIR__)
import proton_pottsmpnn_daemon as daemon


class Seq(list):
    def clone(self):
        return Seq(self)


class IndexList(list):
    def tolist(self):
        return list(self)


class Scorer:
    def H_of(self, seq):
        e = -1.0 * ((seq[1] == 2) + (seq[2] == 2))
        if seq[1] == 2 and seq[2] == 2:
            e -= 5.0
        return float(e)


CTX = SimpleNamespace(
    S_native=Seq([0, 1, 1, 0]),
    chainA_all_idx=IndexList([0, 1, 2, 3]),
    token_aa=SimpleNamespace(
        chain_id=np.array(["A"] * 4),
        res_id=np.array([1, 2, 3, 4]),
        res_name=np.array(["ALA", "HIS", "HIS", "ALA"]),
    ),
    encoding=SimpleNamespace(
        token_to_idx={"ALA": 0, "HIS-S": 1, "HIS-P": 2}
    ),
    scorer=Scorer(),
    L=4,
)


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

    @classmethod
    def from_params(cls, params):
        return dict(params)


class StubDesign:
    canonical_sequence = "AHHA"
    extended_tokens = ["ALA", "HIS-S", "HIS-S", "ALA"]
    energy_trajectory = None

    def to_metadata(self):
        return {"final_potts_energy": -1.25}


class StubDesignSet(list):
    seed_energies = [{"potts_energy": -0.75, "global_protonation_dH": 0.5}]


class StubEngine:
    def run_ph_redesign(self, **_kwargs):
        return StubDesignSet([StubDesign()])

    def _build_context(self, _atom_array, binder_chain):
        assert binder_chain == "A"
        return CTX


def _load_runtime(_args):
    return StubEngine(), StubCriteria


daemon._load_runtime = _load_runtime
daemon._load_atom_array = lambda _path: object()
daemon._token_chain_sequences = lambda _atom_array: (
    CTX.token_aa, {"A": "AHHA"}
)
daemon._apply_design_mask = lambda *_args: None

raise SystemExit(daemon.main())
'''.replace("__WRAPPERS_DIR__", repr(wrappers_dir)),
        encoding="utf-8",
    )
    config["daemon_script_path"] = str(wrapper)
    key = "shared-subprocess-test"
    runner = get_shared_proton_runner(config, key=key)
    pdb_path = tmp_path / "single.pdb"
    pdb_path.write_text("stub\n", encoding="utf-8")
    try:
        design = runner.design_sequences(
            pdb_path=str(pdb_path),
            binder_chain="A",
            binder_sequence="AHHA",
            chain_sequences={"A": "AHHA"},
            fixed_positions=[],
            criteria={},
            max_mutations=2,
            n_jobs=1,
        )
        score = runner.score_states(
            pdb_path=str(pdb_path),
            binder_chain="A",
            binder_sequence="AHHA",
            chain_sequences={"A": "AHHA"},
            histidine_positions=[1, 2],
        )

        assert get_shared_proton_runner(key=key) is runner
        assert design["daemon_pid"] == score["daemon_pid"]
        assert design["model_load_count"] == score["model_load_count"] == 1
        assert design["request_count"] == 1
        assert score["request_count"] == 2
        assert design["selected"]["canonical_sequence"] == "AHHA"
        assert score["neutral_energy"] == 0.0
        assert score["protonated_energy"] == -7.0
        assert score["gap"] == -7.0
        assert Path(design["raw_output_path"]).name == f"{design['request_id']}.json"
        assert Path(score["raw_output_path"]).name == f"score_states_{score['request_id']}.json"

        design_raw = json.loads(
            Path(design["raw_output_path"]).read_text(encoding="utf-8")
        )
        score_raw = json.loads(
            Path(score["raw_output_path"]).read_text(encoding="utf-8")
        )
        assert design_raw["type"] == "proton_pottsmpnn_raw"
        assert score_raw["type"] == "proton_pottsmpnn_state_score_raw"
    finally:
        runner.close()

    with pytest.raises(RuntimeError, match="No live shared"):
        get_shared_proton_runner(key=key)


class _CriteriaSchema:
    __dataclass_fields__ = {
        "seed_source": None,
        "method": None,
        "infill_scope": None,
        "center_count": None,
        "center_types": None,
        "explicit_centers": None,
        "dep_map": None,
        "forbidden_tokens": None,
        "max_mutations": None,
    }


def _criteria_for_seed_source(**overrides):
    criteria = {
        "seed_source": "native",
        "method": "greedy_energy_block",
        "infill_scope": "chain",
        "center_count": 0,
        "center_types": [],
        "explicit_centers": [],
        "max_mutations": 8,
    }
    criteria.update(overrides)
    return criteria


def test_daemon_allows_native_seed_for_centre_free_design() -> None:
    validated = daemon._validate_criteria(
        _CriteriaSchema,
        _criteria_for_seed_source(),
        8,
    )

    assert validated["seed_source"] == "native"


@pytest.mark.parametrize(
    "overrides",
    [
        {"method": "block_descent", "center_count": 1},
        {"infill_scope": "neighbourhood"},
        {"center_count": 1},
        {"center_types": ["HIS-S"]},
        {
            "center_count": 1,
            "explicit_centers": [{"res_id": 1, "protonation_type": "HIS-P"}],
        },
    ],
)
def test_daemon_rejects_native_seed_outside_centre_free_design(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="centre-free"):
        daemon._validate_criteria(
            _CriteriaSchema,
            _criteria_for_seed_source(**overrides),
            8,
        )


def test_daemon_retains_inverse_for_placement_design() -> None:
    validated = daemon._validate_criteria(
        _CriteriaSchema,
        _criteria_for_seed_source(
            seed_source="inverse",
            method="block_descent",
            infill_scope="neighbourhood",
            center_count=1,
        ),
        8,
    )

    assert validated["seed_source"] == "inverse"
