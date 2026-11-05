from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = (
    REPO_ROOT
    / "scripts"
    / "validation"
    / "na_mpnn_daemon_timing_warm_final.py"
)
REFERENCE_ROOT = (
    REPO_ROOT
    / "outputs/unified-gpu-acceptance"
    / "source/na_mpnn_daemon_timing_v1_warm_002/reference/native"
)


def _load_harness():
    name = "na_mpnn_daemon_timing_warm_final_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HARNESS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _reference_root_with_seed_map(tmp_path: Path) -> Path:
    root = tmp_path / "native"
    shutil.copytree(REFERENCE_ROOT, root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["seed_references"] = {
        "267850363": {
            "fasta": (
                "subprocess_00_seed_267850363/"
                "seqs/step_000_ABC_model.fa"
            ),
            "specificity_npz": (
                "subprocess_00_seed_267850363/"
                "specificity/step_000_ABC_model.npz"
            ),
        },
        "267850364": {
            "fasta": (
                "subprocess_01_seed_267850364/"
                "seqs/step_000_ABC_model.fa"
            ),
            "specificity_npz": (
                "subprocess_01_seed_267850364/"
                "specificity/step_000_ABC_model.npz"
            ),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return root


def test_final_plan_is_seed364_cold_then_seed363_warm() -> None:
    harness = _load_harness()
    plans = harness.request_plans()
    assert [item["seed"] for item in plans] == [267850364, 267850363]
    assert [item["timing_label"] for item in plans] == [
        "fresh_owner_cold",
        "same_owner_warm",
    ]
    assert [item["arm"] for item in plans] == ["daemon", "daemon"]
    assert all(item["temperature"] == 0.3 for item in plans)
    assert all(item["batch_size"] == 1 for item in plans)
    assert all(item["number_of_batches"] == 1 for item in plans)
    assert all(item["chains_to_design"] == "A" for item in plans)
    assert all(len(item["fixed_residues"]) == 192 for item in plans)


def test_budget_is_exactly_two_resident_requests() -> None:
    harness = _load_harness()
    budget = harness.timing.RequestBudget()
    with pytest.raises(
        harness.timing.TimingAcceptanceError,
        match="resident daemon",
    ):
        budget.claim("subprocess")
    assert budget.claim("daemon") == 1
    assert budget.claim("daemon") == 2
    with pytest.raises(
        harness.timing.TimingAcceptanceError,
        match="budget exhausted",
    ):
        budget.claim("daemon")


def test_full_plan_uses_explicit_seed_map_not_new_request_order(
    tmp_path: Path,
) -> None:
    harness = _load_harness()
    root = _reference_root_with_seed_map(tmp_path)
    input_data = harness.timing.base.input_facts(
        harness.timing.DEFAULT_INPUT
    )
    plan = harness.full_run_plan(
        reference_root=root,
        input_data=input_data,
    )
    first, second = plan["requests"]
    assert first["seed"] == 267850364
    assert first["timing_label"] == "fresh_owner_cold"
    assert first["reference"]["fasta_relative"].startswith(
        "subprocess_01_seed_267850364/"
    )
    assert second["seed"] == 267850363
    assert second["timing_label"] == "same_owner_warm"
    assert second["reference"]["fasta_relative"].startswith(
        "subprocess_00_seed_267850363/"
    )
    assert first["reference"]["fasta_header_seeds"] == [267850364, 267850364]
    assert second["reference"]["fasta_header_seeds"] == [267850363, 267850363]


def test_paid_requests_and_all_comparisons_are_separate_phases() -> None:
    source = HARNESS_PATH.read_text(encoding="utf-8")
    run_body = source.split("def _run(", 1)[1]
    paid_loop = run_body.index(
        "for index, plan in enumerate(request_plans()):"
    )
    post_paid_loop = run_body.index(
        "for index, (plan, result, raw_after) in enumerate("
    )
    capture = run_body.index("captured = capture_runner_result(")
    compare = run_body.index("probability_comparison = compare_outputs(")
    assert paid_loop < post_paid_loop < capture < compare
    assert run_body[:post_paid_loop].count("compare_outputs(") == 0


def test_missing_owner_duration_is_unobserved_not_zero() -> None:
    harness = _load_harness()
    assert harness.observed_duration({}) == (None, False)
    assert harness.observed_duration({"duration_s": None}) == (None, False)
    assert harness.observed_duration({"duration_s": 0.0}) == (0.0, True)
    assert harness.observed_duration({"duration_s": 1.25}) == (1.25, True)


def test_duration_formatter_distinguishes_missing_from_real_zero() -> None:
    harness = _load_harness()
    assert harness.format_observed_duration(None, False) == "not observed"
    assert harness.format_observed_duration(0.0, True) == "0.000000 s"
    assert harness.format_observed_duration(1.25, True) == "1.250000 s"
