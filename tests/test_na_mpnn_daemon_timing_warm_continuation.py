from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = (
    REPO_ROOT
    / "scripts"
    / "validation"
    / "na_mpnn_daemon_timing_warm_continuation.py"
)


def _load_harness():
    name = "na_mpnn_daemon_timing_warm_continuation_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HARNESS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_warm_continuation_plan_is_seed364_cold_then_seed363_warm() -> None:
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
    assert all(item["parse_these_chains_only"] == "ABC" for item in plans)
    assert all(len(item["fixed_residues"]) == 192 for item in plans)


def test_warm_continuation_budget_is_exactly_two_resident_requests() -> None:
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


def test_reference_index_uses_native_order_not_new_owner_order() -> None:
    harness = _load_harness()
    assert harness.reference_index(267850364) == 1
    assert harness.reference_index(267850363) == 0


def test_prior_completed_seed363_is_strict_read_only_evidence() -> None:
    harness = _load_harness()
    prior = (
        REPO_ROOT
        / "outputs/unified-gpu-acceptance"
        / "na_mpnn_daemon_timing_v1/gpu/attempt_001"
    )
    reference = (
        REPO_ROOT
        / "outputs/unified-gpu-acceptance"
        / "source/na_mpnn_daemon_timing_v1/reference/native"
    )
    result = harness._prior_request_facts(prior, reference)
    assert result["status"] == "external_validation_PASS"
    assert result["request_count"] == 1
    assert result["request_delta"] == 1
    assert result["model_load_count"] == 1
    assert result["fasta_equal"] is True
    assert result["all_ten_fields_exact"] is True
    assert result["gpu_uuid"] == "0c27fd3e-0fe5-0a04-cb27-01f07aee09b7"


def test_reference_decoder_uses_runner_canonical_payload() -> None:
    harness = _load_harness()
    from astrolabe.mutators.wrappers.na_mpnn_runner import NAMPNNRunner

    prior = (
        REPO_ROOT
        / "outputs/unified-gpu-acceptance"
        / "na_mpnn_daemon_timing_v1/gpu/attempt_001"
    )
    reference = (
        REPO_ROOT
        / "outputs/unified-gpu-acceptance"
        / "source/na_mpnn_daemon_timing_v1/reference/native"
    )
    ref_bundle = harness.timing._load_reference_bundle(
        reference
        / "subprocess_00_seed_267850363"
        / "specificity/step_000_ABC_model.npz"
    )
    loader = NAMPNNRunner.__new__(NAMPNNRunner)
    live_bundle = loader._load_specificity_matrix(
        prior / "daemon/daemon_00_seed_267850363",
        Path("step_000_ABC_model.cif"),
    )
    comparison = harness.compare_outputs(ref_bundle, live_bundle)
    assert comparison["equal"] is True
    assert comparison["differences"] == []
