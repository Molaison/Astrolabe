from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = REPO_ROOT / "scripts" / "validation" / "unified_gpu_acceptance.py"


def _load_harness():
    module_name = "unified_gpu_acceptance_test_module"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(
        module_name,
        HARNESS_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _write_scores(path: Path, *, value: float, nan: bool = False) -> None:
    path.write_text(
        "\n".join(
            [
                "data_test",
                "loop_",
                "_atom_site.group_PDB",
                "_atom_site.label_atom_id",
                "_atom_site.label_comp_id",
                "_atom_site.auth_asym_id",
                "_atom_site.label_asym_id",
                "_atom_site.label_seq_id",
                "_atom_site.Cartn_x",
                "_atom_site.Cartn_y",
                "_atom_site.Cartn_z",
                f"ATOM CA ALA A A 1 {value:.6f} 2.000000 3.000000",
                "#",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_case_plan_uses_real_frontdoors_and_pair_fixture() -> None:
    harness = _load_harness()
    assert harness.case_plan("default-api").frontdoor == "astrolabe.app.Astrolabe.run"
    assert "astrolabe.cli run" in harness.case_plan("default-cli").frontdoor
    assert (
        harness.case_plan("generic-api").frontdoor
        == "astrolabe.app.Astrolabe.run search.native_refinement"
    )
    assert harness.case_plan("generic-api").reference == "native-generic"
    assert harness.case_plan("paper-cli").fixture == "pair-shared-fixed"
    assert "candidate-batch" in harness.case_plan("candidate-batch").frontdoor


def test_common_config_is_bounded_and_does_not_change_af3_science(tmp_path: Path) -> None:
    harness = _load_harness()
    cfg = harness._build_config(
        profile=REPO_ROOT / "config" / "container_af3_smoke.yaml",
        fixture="single-20aa",
        out_dir=tmp_path / "run",
        task_name="bounded",
        execution="daemon",
        generic_search=True,
    )
    assert cfg.seed == 17
    assert cfg.steps == 2
    assert cfg.repeats == 1
    assert cfg.mutation_num == 1
    assert cfg.seqs.A == harness.SINGLE_SEQUENCE
    assert cfg.stages.default.oligo == ["A"]
    assert cfg.stages.default.mutation_num == 1
    assert cfg.stages.default.weighted_score_expression == "plddt"
    assert cfg.evaluator.alphafold3.num_seeds == 1
    assert cfg.evaluator.alphafold3.num_recycles == 1
    assert cfg.evaluator.alphafold3.num_samples == 1
    assert cfg.evaluator.alphafold3.run_data_pipeline is False
    assert cfg.evaluator.alphafold3.flash_attention_implementation == "cudnn"
    assert cfg.evaluator.alphafold3.model_dir == harness.AF3_MODEL_DIR
    assert cfg.search.max_evaluations == 2
    assert cfg.search.owner == "native_refinement"
    assert harness._digest_config(cfg)["acceptance"][
        "max_evaluations_per_process"
    ] == 2


def test_pair_fixture_has_shared_segment_and_fixed_residues(tmp_path: Path) -> None:
    harness = _load_harness()
    cfg = harness._build_config(
        profile=REPO_ROOT / "config" / "container_af3_smoke.yaml",
        fixture="pair-shared-fixed",
        out_dir=tmp_path / "run",
        task_name="pair",
        execution="in_process",
        generic_search=False,
    )
    assert cfg.seqs.A == "+ACDEFGHIKL@shared"
    assert cfg.seqs.B == "shared +MNPQRSTVWY"
    assert cfg.stages.default.oligo == ["AB"]
    assert cfg.stages.default.mutation_num == 1
    assert cfg.stages.default.weighted_score_expression == "plddt"


def test_capture_sitecustomize_enforces_two_calls_without_retry(tmp_path: Path) -> None:
    harness = _load_harness()
    capture, claims, env = harness._install_capture_environment(tmp_path)
    assert env["ASTROLABE_RUNTIME_ACTIVE"] == "1"
    assert env[harness.MAX_EVALUATIONS_ENV] == "2"
    pythonpath = env["PYTHONPATH"].split(os.pathsep)
    assert pythonpath[0] == str(tmp_path / "instrumentation")
    assert pythonpath[1] == str(REPO_ROOT)
    code = r"""
import os
from types import SimpleNamespace
import numpy as np
from astrolabe.evaluators.base import Evaluator
from astrolabe.evaluators.utils import StandardScores

class Tiny(Evaluator):
    calls = 0
    def prepare_input(self, inputs=None, runtime_config=None):
        return inputs
    def run_function(self, input_feat):
        self.calls += 1
        return {"calls": self.calls}
    def prepare_output(self, output, inputs, runtime_config):
        return StandardScores(
            names=["A"],
            plddt={"A": [np.array([1.0], dtype=np.float32)]},
            complex_paths={},
        )

class StartedOwner:
    @property
    def pid(self):
        return 321
    def status(self):
        return {
            "pid": 321,
            "identity": "model-token",
            "generation": "generation",
            "state": "running",
            "request_count": 2,
            "completed_count": 2,
            "failed_count": 0,
            "model_load_count": 1,
        }

class UnstartedOwner:
    @property
    def pid(self):
        return None
    def status(self):
        raise AssertionError("status() must not lazily start the owner")

value = Tiny()
value.model_runner = SimpleNamespace(
    _client=SimpleNamespace(owner=StartedOwner())
)
value.evaluate({}, {"step": 0})

unstarted = Tiny()
unstarted.model_runner = SimpleNamespace(
    _client=SimpleNamespace(owner=UnstartedOwner())
)
unstarted.evaluate({}, {"step": 1})
try:
    value.evaluate({}, {"step": 2})
except RuntimeError as error:
    assert "budget exceeded" in str(error)
else:
    raise AssertionError("third evaluator call was admitted")
"""
    completed = subprocess.run(
        [os.environ.get("PYTHON", "python"), "-c", code],
        cwd=REPO_ROOT,
        env=dict(env),
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    records = harness._read_capture_records(capture)
    assert [record["status"] for record in records] == ["ok", "ok"]
    assert [record["step"] for record in records] == [0, 1]
    assert all(record["wall_elapsed_s"] >= 0 for record in records)
    assert records[0]["resident_status_before"]["available"] is True
    assert records[0]["resident_status_after"]["model_load_count"] == 1
    assert records[1]["resident_status_before"] == {
        "available": False,
        "reason": "resident_owner_not_started",
    }
    assert sorted(path.name for path in claims.glob("*.claim")) == [
        "0.claim",
        "1.claim",
        "2.claim",
    ]


def test_evaluator_timing_evidence_reports_lightweight_first_and_resident_deltas() -> None:
    harness = _load_harness()
    started = {
        "available": True,
        "pid": 321,
        "model_token": "model-token",
        "generation": "generation",
        "state": "running",
        "request_count": 2,
        "completed_count": 2,
        "failed_count": 0,
        "model_load_count": 1,
    }
    records = [
        {
            "index": 0,
            "wall_elapsed_s": 1.25,
            "resident_status_before": started,
            "resident_status_after": {
                **started,
                "request_count": 3,
                "completed_count": 3,
            },
        },
        {
            "index": 1,
            "wall_elapsed_s": 0.75,
            "resident_status_before": {
                **started,
                "request_count": 3,
                "completed_count": 3,
            },
            "resident_status_after": {
                **started,
                "request_count": 4,
                "completed_count": 4,
            },
        },
    ]
    evidence = harness._evaluator_timing_evidence(
        records,
        process_duration_s=4.5,
        compilation_cache={
            "jax": {
                "environment_variable": "/shared/jax-cache",
                "configured_path": "/shared/jax-cache",
                "cleared": False,
            }
        },
    )

    assert evidence["first_request_s"] == 1.25
    assert evidence["subsequent_request_s"] == [0.75]
    assert [item["wall_elapsed_s"] for item in evidence["requests"]] == [
        1.25,
        0.75,
    ]
    assert evidence["compilation_cache"]["jax"]["cleared"] is False
    assert evidence["total_evaluator_wall_elapsed_s"] == 2.0
    assert evidence["whole_process_duration_s"] == 4.5
    assert evidence["resident"]["completed_deltas"] == [1, 1]
    assert evidence["resident"]["model_load_counts"] == [1]
    assert evidence["resident"]["no_reload_observed"] is True
    assert "compilation caches may be reused" in (
        evidence["performance_scope"]
    )
    assert "insufficient for a statistical performance claim" in (
        evidence["performance_scope"]
    )


def test_evaluator_timing_is_separate_from_exact_numeric_comparison(
    tmp_path: Path,
) -> None:
    harness = _load_harness()

    def write_case(
        root: Path,
        *,
        execution: str,
        wall_elapsed_s: float,
        model_load_count: int,
    ) -> None:
        evidence = root / "evidence"
        evidence.mkdir(parents=True)
        arrays = {"score": np.asarray([1.0], dtype=np.float64)}
        harness._write_npz(evidence / "standard_scores.npz", arrays)
        harness._write_npz(evidence / "rng_trace.npz", arrays)
        harness._write_npz(evidence / "structure_coordinates.npz", arrays)
        for name in (
            "standard_scores.json",
            "rng_trace.json",
            "structure_coordinates.json",
        ):
            harness._atomic_json(evidence / name, [])
        decisions = [
            {
                "evaluation_id": 0,
                "score": 1.0,
                "selected": True,
            }
        ]
        harness._atomic_json(evidence / "decisions.json", decisions)
        harness._atomic_json(
            evidence / "evaluator_timing.json",
            {
                "first_request_s": wall_elapsed_s,
                "subsequent_request_s": [],
                "compilation_cache": {
                    "jax": {
                        "configured_path": f"/cache/{execution}",
                        "cleared": False,
                    }
                },
                "requests": [{
                    "wall_elapsed_s": wall_elapsed_s,
                    "resident": {"model_load_count": model_load_count},
                }],
            },
        )
        harness._atomic_json(
            root / "result.json",
            {
                "status": "evidence-complete",
                "case": "default-api",
                "decision_kind": "trajectory-scores-csv",
                "sequences": [{"evaluation": "0", "A": "AC"}],
                "weighted_scores": [1.0],
                "environment_manifest": {
                    "gpu": {"uuid": "GPU-same"},
                    "weights": {"alphafold3": "same"},
                    "compilation_cache": {
                        "jax": {
                            "configured_path": f"/cache/{execution}",
                            "cleared": False,
                        }
                    },
                },
                "config": {
                    "seed": 17,
                    "steps": 2,
                    "model": {
                        "execution": execution,
                        "num_seeds": 1,
                    },
                },
            },
        )

    reference = tmp_path / "reference"
    case = tmp_path / "case"
    write_case(
        reference,
        execution="in_process",
        wall_elapsed_s=10.0,
        model_load_count=1,
    )
    write_case(
        case,
        execution="daemon",
        wall_elapsed_s=2.0,
        model_load_count=1,
    )

    comparison = harness.compare_case(
        reference_root=reference,
        case_root=case,
    )

    assert comparison["status"] == "PASS"
    assert comparison["exact_numeric_match"] is True
    assert comparison["same_scientific_config_except_execution"] is True


def test_environment_manifest_records_jax_cache_without_clearing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    harness = _load_harness()
    cfg = harness._build_config(
        profile=REPO_ROOT / "config" / "container_af3_smoke.yaml",
        fixture="single-20aa",
        out_dir=tmp_path / "run",
        task_name="cache-observation",
        execution="daemon",
        generic_search=False,
    )
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", "/shared/jax-cache")
    monkeypatch.setattr(harness, "_nvidia_manifest", lambda: {"uuid": "GPU"})
    monkeypatch.setattr(
        harness,
        "_model_manifest",
        lambda _cfg: {"root": "/models/alphafold3", "files": []},
    )

    manifest = harness._environment_manifest(
        cfg,
        harness.SourceManifest((), "source"),
    )

    assert manifest["environment"]["JAX_COMPILATION_CACHE_DIR"] == (
        "/shared/jax-cache"
    )
    assert manifest["environment"][harness.MAX_EVALUATIONS_ENV] is None
    assert manifest["acceptance"]["max_evaluations_per_process"] == 2
    assert manifest["compilation_cache"]["jax"] == {
        "environment_variable": "/shared/jax-cache",
        "configured_path": str(cfg.evaluator.alphafold3.jax_compilation_cache_dir),
        "observed_only": True,
        "cleared": False,
        "note": (
            "Shared or persistent JAX compilation cache may be reused; "
            "the harness does not clear it or infer a cold compilation."
        ),
    }
    assert os.environ["JAX_COMPILATION_CACHE_DIR"] == "/shared/jax-cache"


def test_npz_and_nan_roundtrip_preserves_dtype_shape_and_topology(tmp_path: Path) -> None:
    harness = _load_harness()
    arrays = {
        "scores__A__000000": np.asarray([1.0, np.nan], dtype=np.float32),
        "scores__B": np.asarray([[1, 2]], dtype=np.int16),
    }
    path = tmp_path / "scores.npz"
    harness._write_npz(path, arrays)
    restored = harness._load_tree_arrays(path)
    assert set(restored) == set(arrays)
    assert restored["scores__A__000000"].dtype == np.float32
    assert restored["scores__A__000000"].shape == (2,)
    assert np.array_equal(
        restored["scores__A__000000"],
        arrays["scores__A__000000"],
        equal_nan=True,
    )
    assert restored["scores__B"].dtype == np.int16


def test_float_scalars_are_equality_checked_as_raw_values() -> None:
    harness = _load_harness()
    arrays: dict[str, np.ndarray] = {}
    encoded = harness._encode_tree(
        {"ptm": np.float32(np.nan), "score": 0.125},
        prefix="scores",
        arrays=arrays,
    )
    assert encoded["ptm"]["$array"]["dtype"] == np.dtype(np.float32).str
    assert encoded["score"]["$array"]["dtype"] == np.dtype(np.float64).str
    assert arrays["scores__ptm"].shape == ()
    assert math.isnan(float(arrays["scores__ptm"]))
    assert arrays["scores__score"].shape == ()
    assert float(arrays["scores__score"]) == 0.125


def test_exact_comparison_reports_maxabs_without_tolerance(tmp_path: Path) -> None:
    harness = _load_harness()
    left = tmp_path / "left.npz"
    right = tmp_path / "right.npz"
    harness._write_npz(left, {"x": np.asarray([1.0, np.nan], dtype=np.float64)})
    harness._write_npz(right, {"x": np.asarray([1.25, np.nan], dtype=np.float64)})
    result = harness._compare_arrays(left, right)
    assert result["equal"] is False
    assert result["max_abs_difference"] == 0.25
    assert result["failures"] == [{"key": "x", "reason": "value"}]


def test_structure_evidence_keeps_exact_cif_atom_coordinates(tmp_path: Path) -> None:
    harness = _load_harness()
    source = tmp_path / "model.cif"
    _write_scores(source, value=1.25)
    coordinates, metadata = harness._structure_atoms(source)
    assert coordinates.dtype == np.float64
    assert coordinates.tolist() == [[1.25, 2.0, 3.0]]
    assert metadata["atom"] == ["CA"]
    assert metadata["chain"] == ["A"]
    assert metadata["sequence_id"] == ["1"]


def test_dry_run_writes_only_out_root_plan(tmp_path: Path, capsys) -> None:
    harness = _load_harness()
    out_root = tmp_path / "acceptance"
    code = [
        "--out-root",
        str(out_root),
        "--all",
        "--dry-run",
    ]
    assert harness.main(code) == 0
    payload = json.loads((out_root / "plan.json").read_text(encoding="utf-8"))
    assert payload["status"] == "planned"
    assert payload["coverage_scope"] == "implemented-five-family-only"
    assert [item["case"] for item in payload["cases"]] == list(harness.CASE_NAMES)
    assert payload["acceptance_budget"]["max_evaluations_per_process"] == 2
    assert payload["native_call_budget"] == {
        "reference_count": 3,
        "reference_calls": 6,
        "case_count": 5,
        "case_calls": 10,
        "total_calls": 16,
    }
    by_case = {item["case"]: item for item in payload["cases"]}
    assert by_case["default-api"]["config"]["seed"] == 17
    assert by_case["generic-api"]["config"]["seed"] == 17
    assert by_case["candidate-batch"]["config"]["seed"] == 16
    assert by_case["candidate-batch"]["invocation_seed"] == 16
    assert by_case["candidate-batch"]["effective_seed"] == 17
    assert all(item["config"]["mutation_num"] == 1 for item in payload["cases"])
    assert all(
        item["config"]["weighted_score_expression"] == "plddt"
        for item in payload["cases"]
    )
    assert not (out_root / "environment_manifest.json").exists()
    printed = json.loads(capsys.readouterr().out)
    assert printed == {"status": "planned", "plan": str(out_root / "plan.json")}


def test_source_manifest_change_fails_before_execution(
    tmp_path: Path,
    monkeypatch,
) -> None:
    harness = _load_harness()
    expected = harness.SourceManifest(
        files=(("astrolabe/changed.py", "before"),),
        fingerprint="expected",
    )
    current = harness.SourceManifest(
        files=(("astrolabe/changed.py", "after"),),
        fingerprint="current",
    )
    monkeypatch.setattr(harness, "_read_source_manifest", lambda: current)
    with pytest.raises(harness.SourceChanged, match="source changed"):
        harness._check_source_manifest(
            expected,
            out_root=tmp_path,
            phase="case:default-api:before",
            check_index=1,
        )
    payload = json.loads((tmp_path / "source_change.json").read_text())
    assert payload["matches_expected"] is False
    assert payload["changes"] == [
        {
            "path": "astrolabe/changed.py",
            "before": "before",
            "after": "after",
        }
    ]


def test_numerical_source_manifest_excludes_validation_wrapper() -> None:
    harness = _load_harness()
    full = {
        path for path, _digest in harness._read_source_manifest().files
    }
    numerical = {
        path
        for path, _digest in harness._read_source_manifest(
            numerical_only=True
        ).files
    }
    assert "run.py" in numerical
    assert "config/env.yaml" in full
    assert "config/env_container.yaml" in full
    assert "config/env.yaml" not in numerical
    assert "scripts/validation/unified_gpu_acceptance.py" in full
    assert "scripts/validation/unified_gpu_acceptance.py" not in numerical


def test_capsule_manifest_checks_run_py_and_rejects_mismatch() -> None:
    harness = _load_harness()
    current = harness._read_source_manifest(numerical_only=True)
    payload = current.as_dict()
    payload["files"]["run.py"] = "0" * 64
    capsule = harness._source_manifest_from_payload(payload)
    assert dict(capsule.files)["run.py"] == "0" * 64
    assert capsule != current


def test_full_capsule_manifest_filters_exact_numerical_set_and_detects_missing() -> None:
    harness = _load_harness()
    current = harness._read_source_manifest(numerical_only=True)
    payload = current.as_dict()
    payload["files"]["config/base.yaml"] = "1" * 64
    payload["files"]["vendor/alphafold3/run_alphafold.py"] = "2" * 64
    filtered = harness._source_manifest_from_payload(payload)
    assert filtered == current

    vendor_path = next(
        path for path, _digest in filtered.files
        if path.startswith("vendor/alphafold3/src/")
    )
    incomplete = dict(filtered.files)
    del incomplete[vendor_path]
    missing_vendor = harness._source_manifest_from_payload(
        {"files": incomplete}
    )
    assert missing_vendor != current
    assert vendor_path not in dict(missing_vendor.files)


def test_sif_manifest_projection_excludes_unused_repo_vendor() -> None:
    harness = _load_harness()
    full_payload = harness._read_source_manifest(
        numerical_only=True,
        include_af3_vendor=True,
    ).as_dict()
    sif_projection = harness._source_manifest_from_payload(
        full_payload,
        include_af3_vendor=False,
    )
    paths = set(dict(sif_projection.files))
    assert "run.py" in paths
    assert any(path.startswith("astrolabe/") for path in paths)
    assert not any(
        path.startswith("vendor/alphafold3/src/") for path in paths
    )


def test_resume_attempt_never_overwrites_failed_attempt(tmp_path: Path) -> None:
    harness = _load_harness()
    first = tmp_path / "resumes" / "default_api_001"
    first.mkdir(parents=True)
    sentinel = first / "failed.log"
    sentinel.write_text("preserve me\n", encoding="utf-8")
    next_root = harness._next_resume_root(tmp_path)
    assert next_root == tmp_path / "resumes" / "default_api_002"
    assert sentinel.read_text(encoding="utf-8") == "preserve me\n"


def test_resume_requires_original_zero_model_call_failure(tmp_path: Path) -> None:
    harness = _load_harness()
    with pytest.raises(harness.AcceptanceError, match="original failed case"):
        harness._validate_resume_baseline(
            out_root=tmp_path,
            profile=REPO_ROOT / "config" / "container_af3_smoke.yaml",
            environment={},
            current_source=harness.SourceManifest((), "source"),
            runtime_imports={},
            expected_af3_module_path=Path("/unused"),
            frozen_source_root=None,
            frozen_source_manifest=None,
        )


def test_cli_commands_are_actual_frontdoors_and_candidate_budget_is_two(
    tmp_path: Path,
) -> None:
    harness = _load_harness()
    candidate = harness.case_plan("candidate-batch")
    config = tmp_path / "config.yaml"
    config.write_text("task_name: x\nout_dir: y\n", encoding="utf-8")
    command, _ = harness._cli_command(
        plan=candidate,
        python_bin=Path("/python3.11"),
        config_path=config,
        paper_input=tmp_path / "paper.yaml",
        case_root=tmp_path,
    )
    assert command[:4] == [
        "/python3.11",
        "-m",
        "astrolabe.cli",
        "experimental",
    ]
    assert command[-2:] == ["--format", "json"]
    assert command[command.index("--candidate-steps") + 1] == "2"
    assert command[command.index("--num-candidates") + 1] == "1"


def test_candidate_batch_compensates_for_child_seed_offset(tmp_path: Path) -> None:
    harness = _load_harness()
    invocation = harness._build_config(
        profile=REPO_ROOT / "config" / "container_af3_smoke.yaml",
        fixture="single-20aa",
        out_dir=tmp_path / "run",
        task_name="candidate-batch",
        execution="daemon",
        generic_search=False,
        seed=harness._invocation_seed("candidate-batch"),
    )
    assert invocation.seed == 16

    child_path = (
        tmp_path
        / "run"
        / "candidate_pool"
        / "cand_0001"
        / "config.yaml"
    )
    child = OmegaConf.create(OmegaConf.to_container(invocation, resolve=False))
    from omegaconf import open_dict

    with open_dict(child):
        child.seed = 17
        child.steps = 2
    child_path.parent.mkdir(parents=True)
    child_path.write_text(OmegaConf.to_yaml(child, resolve=False), encoding="utf-8")

    effective, context = harness._candidate_executed_config(tmp_path, invocation)
    assert effective.seed == 17
    assert context == {
        "candidate_id": "cand_0001",
        "config_path": str(child_path),
        "base_seed": 16,
        "seed_offset": 1,
        "execution_seed": 17,
    }


def test_candidate_child_seed_mismatch_fails_instead_of_claiming_equivalence(
    tmp_path: Path,
) -> None:
    harness = _load_harness()
    invocation = harness._build_config(
        profile=REPO_ROOT / "config" / "container_af3_smoke.yaml",
        fixture="single-20aa",
        out_dir=tmp_path / "run",
        task_name="candidate-batch",
        execution="daemon",
        generic_search=False,
        seed=harness.SEED,
    )
    child = OmegaConf.create(OmegaConf.to_container(invocation, resolve=False))
    from omegaconf import open_dict

    with open_dict(child):
        child.seed = 18
    child_path = (
        tmp_path / "run" / "candidate_pool" / "cand_0001" / "config.yaml"
    )
    child_path.parent.mkdir(parents=True)
    child_path.write_text(OmegaConf.to_yaml(child, resolve=False), encoding="utf-8")

    with pytest.raises(
        harness.AcceptanceError,
        match="requested scientific seed 17",
    ):
        harness._candidate_executed_config(tmp_path, invocation)


def test_scientific_comparison_includes_seed_but_ignores_execution(
    tmp_path: Path,
) -> None:
    harness = _load_harness()
    reference = {
        "sha256": "reference",
        "seed": 17,
        "steps": 2,
        "model": {"execution": "in_process", "num_seeds": 1},
    }
    same = {
        "sha256": "case",
        "seed": 17,
        "steps": 2,
        "model": {"execution": "daemon", "num_seeds": 1},
    }
    different = {
        "sha256": "case",
        "seed": 18,
        "steps": 2,
        "model": {"execution": "daemon", "num_seeds": 1},
    }
    left = harness._comparable_scientific_config(reference)
    assert left == harness._comparable_scientific_config(same)
    differences = harness._config_differences(
        left,
        harness._comparable_scientific_config(different),
    )
    assert differences == [{"path": "$.seed", "reference": 17, "case": 18}]


def test_generic_af3_fixture_resolves_disabled_external_interpolations(
    tmp_path: Path,
) -> None:
    harness = _load_harness()
    cfg = harness._build_config(
        profile=REPO_ROOT / "config" / "container_af3_smoke.yaml",
        fixture="single-20aa",
        out_dir=tmp_path / "run",
        task_name="generic-resolution",
        execution="daemon",
        generic_search=True,
    )

    from astrolabe.app import _load_run_module
    from astrolabe.design.configured_refinement import _scientific_config_payload

    args = _load_run_module().parse_arguments(cfg)
    payload = _scientific_config_payload(args)
    assert payload["seed"] == 17
    assert "rf3" not in payload
    assert "protenix" not in (payload.get("evaluator") or {})
    assert "rfd3" not in (payload.get("initializer") or {})


def test_beam_selection_projection_uses_real_score_order() -> None:
    harness = _load_harness()
    selected, valid = harness._beam_selection_projection(
        [
            {"evaluation_id": 0, "score": 0.4, "selected": False},
            {"evaluation_id": 1, "score": 0.8, "selected": True},
        ]
    )
    assert valid is True
    assert selected == [{"evaluation_id": 1, "score": 0.8}]


def test_native_generic_worker_mode_is_neutral_reference(
    tmp_path: Path,
    monkeypatch,
) -> None:
    harness = _load_harness()
    cfg_path = tmp_path / "native-generic.yaml"
    cfg_path.write_text("seed: 17\nsteps: 2\n", encoding="utf-8")
    seen = []

    def run_generic(cfg):
        seen.append(cfg)

    monkeypatch.setattr(harness, "_run_native_generic_reference", run_generic)
    assert (
        harness._api_worker_main(
            [
                "--mode",
                "native-generic",
                "--config",
                str(cfg_path),
                "--fixture",
                "single-20aa",
            ]
        )
        == 0
    )
    assert len(seen) == 1


def test_internal_api_worker_does_not_require_public_out_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    harness = _load_harness()
    config = tmp_path / "config.yaml"
    config.write_text("task_name: x\n", encoding="utf-8")
    observed = []

    def worker(argv):
        observed.extend(argv)
        return 0

    monkeypatch.setattr(harness, "_api_worker_main", worker)
    assert (
        harness.main(
            [
                "--api-worker",
                "--mode",
                "default-api",
                "--config",
                str(config),
                "--fixture",
                "single-20aa",
            ]
        )
        == 0
    )
    assert observed[-1] == "single-20aa"


def test_failed_case_compare_is_explicit_without_opening_missing_npz(
    tmp_path: Path,
) -> None:
    harness = _load_harness()
    reference = tmp_path / "reference"
    case = tmp_path / "case"
    reference.mkdir()
    case.mkdir()
    (reference / "result.json").write_text(
        json.dumps({"status": "failed", "case": "reference", "error": "boom"}),
        encoding="utf-8",
    )
    (case / "result.json").write_text(
        json.dumps({"status": "failed", "case": "default-api", "error": "also boom"}),
        encoding="utf-8",
    )
    comparison = harness.compare_case(
        reference_root=reference,
        case_root=case,
    )
    assert comparison["status"] == "ISSUES"
    assert comparison["reference_error"] == "boom"
    assert comparison["case_error"] == "also boom"
    assert comparison["exact_numeric_match"] is False


def test_unsupported_complex_pose_does_not_fake_evidence(tmp_path: Path) -> None:
    harness = _load_harness()
    from types import SimpleNamespace
    from astrolabe.evaluators.utils import StandardScores

    scores = StandardScores(
        names=["A"],
        plddt={"A": [np.asarray([1.0])]},
        complex_paths={},
        complex_poses=SimpleNamespace(poses={"A": object()}),
    )
    try:
        harness._score_snapshot(scores)
    except harness.UnsupportedEvidence as error:
        assert "complex_poses" in str(error)
    else:
        raise AssertionError("opaque pose evidence was accepted")


@pytest.mark.parametrize("value", ["0", "1", "3", "5", "two", "2.0"])
def test_acceptance_max_evaluations_rejects_non_2_or_4(value: str) -> None:
    harness = _load_harness()
    with pytest.raises(
        harness.AcceptanceError,
        match="must be exactly 2 or 4",
    ):
        harness.parse_max_evaluations(value)


@pytest.mark.parametrize("case_name", ["default-api", "default-cli"])
def test_max4_dry_run_is_one_eight_request_public_case(
    tmp_path: Path,
    monkeypatch,
    capsys,
    case_name: str,
) -> None:
    harness = _load_harness()
    monkeypatch.setenv(harness.MAX_EVALUATIONS_ENV, "4")
    out_root = tmp_path / case_name
    try:
        assert (
            harness.main(
                [
                    "--out-root",
                    str(out_root),
                    "--case",
                    case_name,
                    "--dry-run",
                ]
            )
            == 0
        )
    finally:
        harness._configure_max_evaluations(2)
    payload = json.loads((out_root / "plan.json").read_text(encoding="utf-8"))
    assert payload["acceptance_budget"] == {
        "schema": "astrolabe.gpu-acceptance.request-budget/v1",
        "max_evaluations_per_process": 4,
        "environment_variable": harness.MAX_EVALUATIONS_ENV,
        "environment_value": "4",
        "profile": "throughput",
        "throughput_cases": ["default-api", "default-cli"],
    }
    assert payload["native_call_budget"] == {
        "reference_count": 1,
        "reference_calls": 4,
        "case_count": 1,
        "case_calls": 4,
        "total_calls": 8,
    }
    assert len(payload["cases"]) == 1
    assert payload["cases"][0]["case"] == case_name
    assert payload["cases"][0]["config"]["steps"] == 4
    assert payload["cases"][0]["config"]["acceptance"][
        "max_evaluations_per_process"
    ] == 4
    printed = json.loads(capsys.readouterr().out)
    assert printed == {
        "status": "planned",
        "plan": str(out_root / "plan.json"),
    }


@pytest.mark.parametrize(
    "selection",
    [
        ["--all", "--dry-run"],
        ["--case", "generic-api", "--dry-run"],
        ["--case", "paper-cli", "--dry-run"],
        ["--case", "candidate-batch", "--dry-run"],
        [
            "--case",
            "default-api",
            "--case",
            "default-cli",
            "--dry-run",
        ],
        ["--dry-run"],
    ],
)
def test_max4_rejects_all_or_non_public_single_case_selection(
    tmp_path: Path,
    monkeypatch,
    selection: list[str],
) -> None:
    harness = _load_harness()
    monkeypatch.setenv(harness.MAX_EVALUATIONS_ENV, "4")
    out_root = tmp_path / "must-not-exist"
    try:
        with pytest.raises(SystemExit) as raised:
            harness.main(["--out-root", str(out_root), *selection])
    finally:
        harness._configure_max_evaluations(2)
    assert raised.value.code == 2
    assert not out_root.exists()


def test_invalid_acceptance_budget_fails_before_output_creation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    harness = _load_harness()
    monkeypatch.setenv(harness.MAX_EVALUATIONS_ENV, "3")
    out_root = tmp_path / "must-not-exist"
    with pytest.raises(SystemExit) as raised:
        harness.main(
            [
                "--out-root",
                str(out_root),
                "--case",
                "default-api",
                "--dry-run",
            ]
        )
    assert raised.value.code == 2
    assert not out_root.exists()


@pytest.mark.parametrize("mode", ["generic-api", "native-generic"])
def test_max4_internal_generic_worker_is_rejected_before_dispatch(
    tmp_path: Path,
    monkeypatch,
    mode: str,
) -> None:
    harness = _load_harness()
    monkeypatch.setenv(harness.MAX_EVALUATIONS_ENV, "4")
    config = tmp_path / "config.yaml"
    config.write_text("seed: 17\n", encoding="utf-8")
    monkeypatch.setattr(
        harness,
        "_api_worker_main",
        lambda _argv: pytest.fail("generic worker was dispatched"),
    )
    try:
        with pytest.raises(SystemExit) as raised:
            harness.main(
                [
                    "--api-worker",
                    "--mode",
                    mode,
                    "--config",
                    str(config),
                    "--fixture",
                    "single-20aa",
                ]
            )
    finally:
        harness._configure_max_evaluations(2)
    assert raised.value.code == 2


def test_capture_child_environment_receives_exact_frozen_budget(
    tmp_path: Path,
) -> None:
    harness = _load_harness()
    harness._configure_max_evaluations(4)
    try:
        _capture, _claims, env = harness._install_capture_environment(tmp_path)
    finally:
        harness._configure_max_evaluations(2)
    assert env[harness.MAX_EVALUATIONS_ENV] == "4"
    assert env["ASTROLABE_ACCEPTANCE_MAX_CALLS"] == "4"


def test_native_reference_max4_performs_root_plus_three_advances(
    tmp_path: Path,
    monkeypatch,
) -> None:
    harness = _load_harness()
    advances: list[int] = []
    shutdowns: list[bool] = []

    class Decisions:
        def dump_results(self, _out_dir, *, incremental=False) -> None:
            assert incremental is False

    def setup_logging(*_args) -> None:
        from astrolabe.core import global_config

        global_config.init_const("out_dir", tmp_path)

    fake_run = SimpleNamespace(
        parse_arguments=lambda _cfg: SimpleNamespace(
            out_dir=tmp_path,
            logging=None,
            steps=4,
            task_name="native-four",
            mutator=SimpleNamespace(
                stack_pre_update=False,
                state_conflict_diagnostics=None,
            ),
        ),
        setup_logging=setup_logging,
        initialize_pipeline_components=lambda _args: (
            object(),
            SimpleNamespace(current_evaluator=object()),
            object(),
        ),
        load_or_evaluate_initial=lambda *_args, **_kwargs: "root",
        _initialize_optimization_state=lambda *_args, **_kwargs: (
            Decisions(),
            "root",
        ),
        perform_evaluation=lambda *_args, **_kwargs: None,
    )

    def advance(*args):
        step = args[-1]
        advances.append(step)
        return f"result-{step}", False, False

    fake_run._advance_optimization_state = advance
    fake_run._shutdown_pipeline_resources = lambda: shutdowns.append(True)

    import astrolabe.app

    monkeypatch.setattr(astrolabe.app, "_load_run_module", lambda: fake_run)
    harness._configure_max_evaluations(4)
    try:
        harness._run_native_reference(SimpleNamespace())
    finally:
        harness._configure_max_evaluations(2)
    assert advances == [1, 2, 3]
    assert shutdowns == [True]


def test_resume_acceptance_budget_mismatch_fails_before_model_work(
    tmp_path: Path,
) -> None:
    harness = _load_harness()
    failed = tmp_path / "cases" / "default-api" / "result.json"
    failed.parent.mkdir(parents=True)
    failed.write_text(
        json.dumps({"status": "failed", "model_calls": 0}),
        encoding="utf-8",
    )
    baseline = tmp_path / "references" / "native-single" / "result.json"
    baseline.parent.mkdir(parents=True)
    baseline.write_text(
        json.dumps(
            {
                "status": "evidence-complete",
                "model_calls": 2,
                "environment_manifest": {
                    "acceptance": {"max_evaluations_per_process": 2}
                },
            }
        ),
        encoding="utf-8",
    )
    harness._configure_max_evaluations(4)
    try:
        with pytest.raises(
            harness.AcceptanceError,
            match="resume acceptance budget mismatch: baseline=2, current=4",
        ):
            harness._validate_resume_baseline(
                out_root=tmp_path,
                profile=REPO_ROOT / "config" / "container_af3_smoke.yaml",
                environment={
                    "acceptance": {"max_evaluations_per_process": 4}
                },
                current_source=harness.SourceManifest((), "source"),
                runtime_imports={},
                expected_af3_module_path=Path("/unused"),
                frozen_source_root=None,
                frozen_source_manifest=None,
            )
    finally:
        harness._configure_max_evaluations(2)


def test_timing_four_request_profile_has_three_subsequent_samples() -> None:
    harness = _load_harness()
    records = [
        {
            "index": index,
            "wall_elapsed_s": float(index + 1),
            "resident_status_before": {
                "available": False,
                "reason": "resident_owner_not_started",
            },
            "resident_status_after": {
                "available": False,
                "reason": "resident_owner_not_started",
            },
        }
        for index in range(4)
    ]
    evidence = harness._evaluator_timing_evidence(
        records,
        process_duration_s=20.0,
        compilation_cache={"jax": {"cleared": False, "observed_only": True}},
    )
    assert evidence["evaluator_call_count"] == 4
    assert evidence["first_request_s"] == 1.0
    assert evidence["subsequent_request_s"] == [2.0, 3.0, 4.0]
    assert evidence["total_evaluator_wall_elapsed_s"] == 10.0
    assert evidence["compilation_cache"]["jax"]["cleared"] is False
    assert "4 request(s)" in evidence["performance_scope"]
    assert "insufficient for a statistical performance claim" in (
        evidence["performance_scope"]
    )
