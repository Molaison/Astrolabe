from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = (
    REPO_ROOT
    / "scripts"
    / "validation"
    / "na_mpnn_daemon_timing_acceptance.py"
)


def _load_harness():
    name = "na_mpnn_daemon_timing_acceptance_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HARNESS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _reference_manifest(reference_root: Path) -> dict:
    harness = _load_harness()
    records = {}
    for index, seed in enumerate(harness.SEEDS):
        paths = harness._reference_relative_paths(seed, index)
        for relative in paths.values():
            path = reference_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{seed}:{relative}".encode())
            records[relative] = harness._sha256(path)
    manifest = {
        "schema": f"{harness.SCHEMA}.native-reference",
        "seeds": list(harness.SEEDS),
        "files": records,
    }
    (reference_root / harness.REFERENCE_MANIFEST_NAME).write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def _bundle() -> dict:
    return {
        "predicted_ppm": np.asarray([[0.1, 0.2]], dtype=np.float64),
        "true_sequence": np.asarray([1], dtype=np.int64),
        "chain_labels": np.asarray([0], dtype=np.int32),
        "mask": np.asarray([1], dtype=np.int32),
        "protein_mask": np.asarray([0], dtype=np.int32),
        "dna_mask": np.asarray([0], dtype=np.int32),
        "rna_mask": np.asarray([1], dtype=np.int32),
        "encoded_residues": np.asarray(["A1"], dtype="<U3"),
        "encoded_residues_dict": {"A1": 0},
        "restype_to_int": {"A": 0},
    }


def _reference_npz_payload() -> dict:
    payload = _bundle()
    encoded_dict = np.empty((), dtype=object)
    encoded_dict[()] = payload["encoded_residues_dict"]
    restype_dict = np.empty((), dtype=object)
    restype_dict[()] = payload["restype_to_int"]
    payload["encoded_residues"] = np.asarray(
        payload["encoded_residues"],
        dtype="<U3",
    )
    payload["encoded_residues_dict"] = encoded_dict
    payload["restype_to_int"] = restype_dict
    return payload


def test_request_plan_is_exactly_two_resident_seeded_requests() -> None:
    harness = _load_harness()
    plans = harness.request_plans()
    assert len(plans) == 2
    assert [item["arm"] for item in plans] == ["daemon", "daemon"]
    assert [item["seed"] for item in plans] == list(harness.SEEDS)
    assert [item["timing_label"] for item in plans] == [
        "first_with_daemon_startup",
        "warm_second",
    ]
    assert all(item["temperature"] == 0.3 for item in plans)
    assert all(item["batch_size"] == 1 for item in plans)
    assert all(item["number_of_batches"] == 1 for item in plans)
    assert all(item["chains_to_design"] == "A" for item in plans)
    assert all(item["parse_these_chains_only"] == "ABC" for item in plans)
    assert all(len(item["fixed_residues"]) == 192 for item in plans)
    assert all(sum(item["design_mask"]) == 40 for item in plans)


def test_budget_rejects_native_and_third_resident_request() -> None:
    harness = _load_harness()
    budget = harness.RequestBudget()
    with pytest.raises(harness.TimingAcceptanceError, match="resident daemon"):
        budget.claim("subprocess")
    assert budget.claim("daemon") == 1
    assert budget.claim("daemon") == 2
    with pytest.raises(harness.TimingAcceptanceError, match="budget exhausted"):
        budget.claim("daemon")


def test_reference_manifest_content_binds_four_old_native_files(
    tmp_path: Path,
) -> None:
    harness = _load_harness()
    manifest = _reference_manifest(tmp_path)
    traced = harness.reference_facts(tmp_path)
    assert set(traced["files"]) == set(manifest["files"])
    assert len(traced["files"]) == 4
    first = tmp_path / sorted(manifest["files"])[0]
    first.write_bytes(b"changed")
    with pytest.raises(harness.TimingAcceptanceError, match="content changed"):
        harness.reference_facts(tmp_path)


def test_reference_bundle_matches_runner_payload_representation(
    tmp_path: Path,
) -> None:
    harness = _load_harness()
    path = tmp_path / "specificity/step_000_ABC_model.npz"
    path.parent.mkdir()
    with path.open("wb") as handle:
        np.savez_compressed(handle, **_reference_npz_payload())

    loaded = harness._load_reference_bundle(path)
    runner_payload = _bundle()
    runner_payload["encoded_residues"] = runner_payload[
        "encoded_residues"
    ].tolist()
    runner_payload["encoded_residues_dict"] = _reference_npz_payload()[
        "encoded_residues_dict"
    ]

    assert isinstance(loaded["encoded_residues"], list)
    assert isinstance(loaded["restype_to_int"], dict)
    assert isinstance(loaded["encoded_residues_dict"], np.ndarray)
    comparison = harness.compare_outputs(loaded, runner_payload)
    assert comparison["equal"] is True, comparison["differences"]
    assert comparison["differences"] == []


def test_resident_handler_writes_uuid_rng_and_owner_before_then_after(
    tmp_path: Path,
) -> None:
    harness = _load_harness()

    class Model:
        @staticmethod
        def parameters():
            return iter([SimpleNamespace(device="cuda:0")])

    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"checkpoint")
    capture = tmp_path / "resident-requests.jsonl"
    request_root = tmp_path / "requests"

    def handler(payload):
        before = request_root / "daemon/request_00_before.json"
        assert before.is_file()
        before_record = json.loads(before.read_text(encoding="utf-8"))
        assert before_record["status"] == "starting"
        assert before_record["owner"]["gpu"]["uuid"] == "GPU-test-uuid"
        assert before_record["daemon_rng_before_sha256"]
        return {
            "probabilities": _bundle(),
            "job_dir": "/job",
            "fasta_path": "/job/output.fa",
            "sequence": "gtgc",
        }

    audited = harness.TimingResidentHandler(
        handler,
        resident_model=Model(),
        capture_path=capture,
        request_root=request_root,
        checkpoint_path=checkpoint,
        checkpoint_reads=[str(checkpoint)],
        gpu_facts=lambda: {
            "uuid": "GPU-test-uuid",
            "current_device": 0,
            "visible_devices": "0",
        },
    )
    result = audited.infer(
        {"job_name": "daemon_00", "job_dir": "/job", "seed": harness.SEEDS[0]}
    )
    assert result["sequence"] == "gtgc"

    after = json.loads(
        (request_root / "daemon/request_00_after.json").read_text(encoding="utf-8")
    )
    assert after["status"] == "completed"
    assert after["request_count"] == 1
    assert after["owner"]["model_load_count"] == 1
    assert after["owner"]["gpu"]["uuid"] == "GPU-test-uuid"
    assert after["raw_result"]["probabilities"]["sha256"]

    records = [
        json.loads(line)
        for line in capture.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 1
    assert records[0]["gpu_uuid"] == "GPU-test-uuid"
    assert records[0]["request_count"] == 1
    assert records[0]["status"] == "completed"
