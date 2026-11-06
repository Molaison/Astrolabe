from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from omegaconf import OmegaConf
import pytest


ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "scripts" / "validation" / "candidate_recovery_gpu_acceptance.py"


def _load_harness():
    name = "astrolabe_candidate_recovery_gpu_harness_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HARNESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_candidate_recovery_harness_prepares_fixed_bounded_plan(
    tmp_path: Path,
) -> None:
    harness = _load_harness()
    out_root = tmp_path / "acceptance"

    plan = harness._candidate_plan(
        out_root=out_root,
        profile=ROOT / "config" / "container_af3_smoke.yaml",
        max_wait_s=900.0,
    )

    assert plan["base_seed"] == 16
    assert plan["expected_child_seeds"] == {
        "cand_0001": 17,
        "cand_0002": 18,
    }
    assert plan["num_candidates"] == 2
    assert plan["min_completed"] == 2
    assert plan["decision_rounds"] == 1
    assert plan["candidate_steps"] == 2
    assert plan["max_in_flight"] == 2
    assert plan["max_wait_s"] == 900.0
    assert plan["requests_per_child_cap"] == 2
    assert plan["requests_per_scenario_cap"] == 4
    assert plan["total_request_budget"] == 8
    assert set(plan["configs"]) == {"uninterrupted", "committed_cut"}
    assert (out_root / "instrumentation" / "sitecustomize.py").is_file()
    assert json.loads((out_root / "plan.json").read_text(encoding="utf-8")) == plan
    assert harness._load_prepared_plan(
        out_root=out_root,
        profile=(ROOT / "config" / "container_af3_smoke.yaml").resolve(),
        max_wait_s=900.0,
    ) == plan

    for config_path in plan["configs"].values():
        cfg = OmegaConf.load(config_path)
        assert int(cfg.seed) == 16
        assert int(cfg.steps) == 2
        assert len(str(cfg.seqs.A).lstrip("+")) == 20
        assert int(cfg.evaluator.alphafold3.num_seeds) == 1
        assert int(cfg.evaluator.alphafold3.num_recycles) == 1
        assert int(cfg.evaluator.alphafold3.num_samples) == 1
        assert cfg.stages.default.weighted_score_expression == "plddt"
        assert int(cfg.mutation_num) == 1
        assert list(cfg.stages.default.mutators) == ["MCMC"]
        assert cfg.mutator.mutator_terms.MCMC._type == "MCMC"


def test_candidate_recovery_harness_command_and_capture_isolation(
    tmp_path: Path,
) -> None:
    harness = _load_harness()
    plan = harness._candidate_plan(
        out_root=tmp_path / "acceptance",
        profile=ROOT / "config" / "container_af3_smoke.yaml",
        max_wait_s=900.0,
    )
    command = harness._command(
        config=Path(plan["configs"]["committed_cut"]),
        run_root=tmp_path / "acceptance" / "committed_cut" / "run",
        max_wait_s=900.0,
        resume=True,
    )

    assert command[command.index("--num-candidates") + 1] == "2"
    assert command[command.index("--min-completed") + 1] == "2"
    assert command[command.index("--max-in-flight") + 1] == "2"
    assert command[command.index("--decision-rounds") + 1] == "1"
    assert command[command.index("--candidate-steps") + 1] == "2"
    assert command[-1] == "--resume"
    env = harness._scenario_environment(plan, "committed_cut")
    assert env["ASTROLABE_ACCEPTANCE_MAX_CALLS"] == "2"
    assert env["ASTROLABE_ACCEPTANCE_CAPTURE_ROOT"].endswith(
        "/committed_cut/captures"
    )


def test_candidate_recovery_harness_main_only_prepares_without_execute(
    tmp_path: Path,
) -> None:
    harness = _load_harness()
    out_root = tmp_path / "main-plan"

    assert (
        harness.main(
            [
                "--out-root",
                str(out_root),
            ]
        )
        == 0
    )
    plan = json.loads((out_root / "plan.json").read_text(encoding="utf-8"))
    assert plan["status"] == "prepared"
    assert plan["launch_policy"] == "prepare by default; --execute is required"


def _write_child_evidence(
    root: Path,
    *,
    timestamp: str,
    coordinate: float,
) -> None:
    root.mkdir(parents=True)
    np.savez_compressed(
        root / "standard_scores.npz",
        plddt=np.asarray([0.7], dtype=np.float64),
    )
    np.savez_compressed(
        root / "rng_trace.npz",
        draw=np.asarray([17], dtype=np.uint64),
    )
    np.savez_compressed(
        root / "structure_coordinates.npz",
        atom=np.asarray([[coordinate, 2.0, 3.0]], dtype=np.float32),
    )
    (root / "standard_scores.json").write_text("[]\n", encoding="utf-8")
    (root / "rng_trace.json").write_text("[]\n", encoding="utf-8")
    structure = root / "structure.cif"
    structure.write_text(
        "\n".join(
            [
                "data_test",
                f"_atom_site.Cartn_x {coordinate:.3f}",
                "_atom_site.Cartn_y 2.000",
                (
                    '_ma_model_list.model_group_name "AlphaFold-beta-20231127 '
                    f'(3.0.1 @ {timestamp})"'
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )
    (root / "structure_coordinates.json").write_text(
        json.dumps(
            [
                {
                    "evaluation_index": 0,
                    "assembly": "A",
                    "source_path": str(structure),
                    "snapshot_path": str(structure),
                    "source_sha256": "0" * 64,
                    "coordinate_dtype": "<f4",
                    "coordinate_shape": [1, 3],
                    "atoms": {"atom": ["CA"]},
                }
            ]
        ),
        encoding="utf-8",
    )


def test_cross_scenario_evidence_allows_only_af3_cif_timestamp(
    tmp_path: Path,
) -> None:
    harness = _load_harness()
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_child_evidence(
        left,
        timestamp="2026-11-01 13:27:48",
        coordinate=1.0,
    )
    _write_child_evidence(
        right,
        timestamp="2026-11-01 13:28:50",
        coordinate=1.0,
    )

    result = harness._compare_child_evidence(
        left_root=left,
        right_root=right,
    )

    assert result["status"] == "PASS"
    assert result["score_arrays"]["equal"] is True
    assert result["rng_arrays"]["equal"] is True
    assert result["structure_coordinates"]["equal"] is True
    assert result["cif_pairs"][0]["timestamp_differences"]


def test_cross_scenario_evidence_rejects_non_timestamp_difference(
    tmp_path: Path,
) -> None:
    harness = _load_harness()
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_child_evidence(
        left,
        timestamp="2026-11-01 13:27:48",
        coordinate=1.0,
    )
    _write_child_evidence(
        right,
        timestamp="2026-11-01 13:28:50",
        coordinate=1.001,
    )

    with pytest.raises(harness.SHARED.AcceptanceError, match="non-timestamp"):
        harness._compare_child_evidence(
            left_root=left,
            right_root=right,
        )
