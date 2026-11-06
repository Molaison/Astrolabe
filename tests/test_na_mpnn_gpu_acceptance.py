from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = (
    REPO_ROOT / "scripts" / "validation" / "na_mpnn_gpu_acceptance.py"
)


def _load_harness():
    name = "na_mpnn_gpu_acceptance_harness_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HARNESS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _valid_bundle(seed: int = 267850363) -> dict:
    harness = _load_harness()
    return {
        "predicted_ppm": np.linspace(
            0.0,
            1.0,
            num=232 * 33,
            dtype=np.float64,
        ).reshape(232, 33),
        "true_sequence": np.zeros(232, dtype=np.int64),
        "chain_labels": np.asarray(
            [0] * 40 + [1] * 96 + [2] * 96,
            dtype=np.int32,
        ),
        "mask": np.ones(232, dtype=np.int32),
        "protein_mask": np.asarray(
            [0] * 40 + [1] * 192,
            dtype=np.int32,
        ),
        "dna_mask": np.zeros(232, dtype=np.int32),
        "rna_mask": np.asarray(list(harness.EXPECTED_DESIGN_MASK), dtype=np.int32),
        "encoded_residues": np.asarray(
            list(harness.EXPECTED_ENCODED_RESIDUES),
            dtype="<U3",
        ),
        "encoded_residues_dict": {
            residue: index
            for index, residue in enumerate(harness.EXPECTED_ENCODED_RESIDUES)
        },
        "restype_to_int": {"A": 0, "C": 1, "G": 2, "U": 3},
        "_seed": seed,
    }


def test_request_plans_fix_exactly_four_bounded_requests() -> None:
    harness = _load_harness()
    plans = harness.request_plans()
    assert len(plans) == 4
    assert [item["arm"] for item in plans] == [
        "subprocess",
        "subprocess",
        "daemon",
        "daemon",
    ]
    assert [item["seed"] for item in plans] == [
        267850363,
        267850364,
        267850363,
        267850364,
    ]
    assert all(item["temperature"] == 0.3 for item in plans)
    assert all(item["batch_size"] == 1 for item in plans)
    assert all(item["number_of_batches"] == 1 for item in plans)
    assert all(item["chains_to_design"] == "A" for item in plans)
    assert all(item["design_na_only"] == 1 for item in plans)
    assert len(harness.FIXED_RESIDUES) == 192
    assert harness.FIXED_RESIDUES[:2] == ("B1", "B2")
    assert harness.FIXED_RESIDUES[-2:] == ("C95", "C96")
    assert sum(harness.EXPECTED_DESIGN_MASK) == 40


def test_runner_flag_contract_covers_every_subprocess_argument() -> None:
    harness = _load_harness()
    required = set(harness.REQUIRED_RUNNER_FLAGS)
    assert {
        "--design_na_only",
        "--parse_na_only",
        "--na_shared_tokens",
        "--output_specificity",
        "--chains_to_design",
        "--fixed_residues",
        "--parse_these_chains_only",
    } <= required


def test_request_budget_rejects_third_arm_request_and_fifth_total() -> None:
    harness = _load_harness()
    budget = harness.RequestBudget({"subprocess": 2, "daemon": 2})
    with pytest.raises(harness.AcceptanceError, match="hard budget exceeded"):
        budget.claim("subprocess", {"subprocess": 2, "daemon": 0})
    with pytest.raises(harness.AcceptanceError, match="total hard budget"):
        budget.claim("daemon", {"subprocess": 2, "daemon": 2})


def test_real_28490_baseline_structure_is_content_bound() -> None:
    harness = _load_harness()
    facts = harness.input_facts(harness.DEFAULT_INPUT)
    assert facts["sha256"] == (
        "3cce305d43bb4db3156ae756690c2ea427e8f6fcaa9838e434f281b7a9b2dc2e"
    )
    assert facts["atom_count"] == 2399
    assert facts["residue_counts"] == {"A": 40, "B": 96, "C": 96}
    assert facts["chains"]["A"]["sequence"] == "UCGGCAGCUGCCAUCUGACU" * 2
    assert facts["chains"]["B"]["sequence"].startswith("AVPETR")
    assert facts["chains"]["C"]["sequence"] == facts["chains"]["B"]["sequence"]


def test_selected_rna_sequence_is_canonicalized_to_acgu() -> None:
    harness = _load_harness()
    selected = harness.canonicalize_sequence(
        "gtgc" * 10 + "/" + "A" * 96 + "/" + "A" * 96
    )
    assert selected["rna"] == "GUGC" * 10
    assert selected["canonical"].startswith("GUGC")
    assert selected["protein_b"] == "A" * 96
    assert selected["protein_c"] == "A" * 96


def test_probability_bundle_shape_dtype_and_fixed_mask_are_exact() -> None:
    harness = _load_harness()
    bundle = _valid_bundle()
    bundle.pop("_seed")
    mask = harness._validate_probability_bundle(
        bundle,
        seed=267850363,
        input_data={"chains": {"B": {}, "C": {}}},
    )
    assert mask["encoded_residues"] == list(harness.EXPECTED_ENCODED_RESIDUES)
    assert mask["design_mask"] == list(harness.EXPECTED_DESIGN_MASK)
    assert mask["chain_labels"] == [0] * 40 + [1] * 96 + [2] * 96

    bundle["predicted_ppm"] = bundle["predicted_ppm"].astype(np.float32)
    with pytest.raises(harness.AcceptanceError, match="predicted_ppm"):
        harness._validate_probability_bundle(
            bundle,
            seed=267850363,
            input_data={"chains": {"B": {}, "C": {}}},
        )


def test_selected_fasta_capture_preserves_seed_and_fixed_proteins() -> None:
    harness = _load_harness()
    seed = 267850364
    sequence = "gtgc" * 10 + "/" + "AV" * 48 + "/" + "AV" * 48
    bundle = _valid_bundle(seed)
    bundle.pop("_seed")

    class FakeRunner:
        @staticmethod
        def _parse_fasta(_path):
            return [
                {
                    "header": f"job, seed={seed}, overall_confidence=0.9",
                    "sequence": sequence,
                    "metadata": {
                        "seed": float(seed),
                        "overall_confidence": 0.9,
                    },
                }
            ]

    captured = harness.capture_runner_result(
        runner=FakeRunner(),
        result={
            "probabilities": bundle,
            "fasta_path": "/tmp/not-read.fa",
            "sequence": sequence,
            "job_dir": "/tmp/job",
        },
        plan={"seed": seed},
        input_data={
            "chains": {
                "B": {"sequence": "AV" * 48},
                "C": {"sequence": "AV" * 48},
            }
        },
    )
    selected = captured["deterministic"]["selected_fasta_entries"][0]["sequence"]
    assert selected["rna"] == "GUGC" * 10
    assert selected["protein_b"] == "AV" * 48
    assert selected["protein_c"] == "AV" * 48


def test_resident_evidence_requires_one_pid_token_and_two_unit_deltas() -> None:
    harness = _load_harness()
    records = [
        {
            "pid": 7001,
            "model_token": 9001,
            "device": "cuda:0",
            "model_load_count": 1,
            "checkpoint_reads": ["/weights/s_19137.pt"],
            "request_count": 1,
            "request_delta": 1,
        },
        {
            "pid": 7001,
            "model_token": 9001,
            "device": "cuda:0",
            "model_load_count": 1,
            "checkpoint_reads": ["/weights/s_19137.pt"],
            "request_count": 2,
            "request_delta": 1,
        },
    ]
    statuses = [
        {"pid": 7001, "model_load_count": 1, "request_count": 1},
        {"pid": 7001, "model_load_count": 1, "request_count": 2},
    ]
    summary = harness.validate_resident_evidence(records, statuses)
    assert summary == {
        "pid": 7001,
        "model_token": 9001,
        "device": "cuda:0",
        "model_load_count": 1,
        "checkpoint_reads": ["/weights/s_19137.pt"],
        "request_counts": [1, 2],
        "request_deltas": [1, 1],
    }
    records[1]["model_token"] = 9002
    with pytest.raises(harness.AcceptanceError, match="residency evidence"):
        harness.validate_resident_evidence(records, statuses)


def test_audited_resident_handler_retains_owner_and_denies_third_request(
    tmp_path: Path,
) -> None:
    harness = _load_harness()

    class Model:
        @staticmethod
        def parameters():
            return iter([SimpleNamespace(device="cuda:0")])

    def handler(payload):
        return {"job_dir": payload["job_dir"]}

    capture = tmp_path / "resident.jsonl"
    audited = harness._AuditedResidentHandler(
        handler,
        resident_model=Model(),
        capture_path=capture,
        checkpoint_path=tmp_path / "weights.pt",
        checkpoint_reads=[str(tmp_path / "weights.pt")],
        maximum_requests=2,
    )
    audited.infer({"job_name": "one", "job_dir": "/one"})
    audited.infer({"job_name": "two", "job_dir": "/two"})
    with pytest.raises(harness.AcceptanceError, match="budget exhausted"):
        audited.infer({"job_name": "three", "job_dir": "/three"})

    records = [
        json.loads(line)
        for line in capture.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 2
    assert {item["pid"] for item in records} == {os.getpid()}
    assert len({item["model_token"] for item in records}) == 1
    assert [item["request_count"] for item in records] == [1, 2]
    assert [item["request_delta"] for item in records] == [1, 1]
    assert {item["model_load_count"] for item in records} == {1}
    assert {
        tuple(item["checkpoint_reads"]) for item in records
    } == {(str((tmp_path / "weights.pt").resolve()),)}


def test_source_manifest_changes_when_repo_source_mutates(tmp_path: Path) -> None:
    harness = _load_harness()
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "model.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    before = harness._repo_source_manifest(repo)
    source.write_text("VALUE = 2\n", encoding="utf-8")
    after = harness._repo_source_manifest(repo)
    assert before != after
    assert before["model.py"] != after["model.py"]


def test_preflight_binds_real_runner_api_and_public_entrypoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    harness = _load_harness()
    import astrolabe.mutators.wrappers.na_mpnn_daemon as daemon_module

    repo = tmp_path / "na_mpnn"
    (repo / "inference").mkdir(parents=True)
    run_py = repo / "inference" / "run.py"
    run_py.write_bytes(b"patched runner")
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"checkpoint")
    source = tmp_path / "input.cif"
    source.write_text("data_input\n#\n", encoding="utf-8")
    monkeypatch.setattr(daemon_module, "verify_vendor", lambda path: Path(path))
    monkeypatch.setattr(harness, "CHECKPOINT_SHA256", harness._sha256(checkpoint))
    monkeypatch.setattr(
        harness,
        "CHECKPOINT_SIZE_BYTES",
        checkpoint.stat().st_size,
    )
    monkeypatch.setattr(harness, "RUN_PY_SHA256", harness._sha256(run_py))
    monkeypatch.setattr(
        harness,
        "input_facts",
        lambda path: {
            "path": str(path),
            "sha256": harness._sha256(path),
            "atom_count": 2399,
            "chains": {"A": {}, "B": {}, "C": {}},
            "residue_counts": {"A": 40, "B": 96, "C": 96},
        },
    )
    monkeypatch.setattr(
        harness,
        "probe_public_entrypoint",
        lambda repo, python_bin: {
            "model_requests": 0,
            "commands": [{"command": ["run.py", "--help"], "returncode": 0}],
        },
    )

    report = harness.preflight(
        repo=repo,
        checkpoint=checkpoint,
        input_path=source,
        python_bin=sys.executable,
        probe_entrypoint=True,
    )

    assert report["verified"] is True
    assert report["checkpoint"]["sha256"] == harness._sha256(checkpoint)
    assert "structure_path" in report["runner_design_sequences_signature"]
    assert report["public_entrypoint"]["model_requests"] == 0
    assert report["request_budget"] == {
        "hard_total": 4,
        "per_arm": {"subprocess": 2, "daemon": 2},
        "af3_requests": 0,
    }


def test_parse_args_and_preflight_main_never_open_cuda(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    harness = _load_harness()
    source = tmp_path / "input.cif"
    source.write_text("data_input\n#\n", encoding="utf-8")
    out = tmp_path / "out"
    called = []

    def preflight(args):
        called.append(args)
        return {"verified": True}

    monkeypatch.setattr(harness, "run_preflight", preflight)
    assert harness.main(
        [
            "--out-dir",
            str(out),
            "--input",
            str(source),
            "--preflight-only",
        ]
    ) == 0
    assert called[0].preflight_only is True
    assert capsys.readouterr().out.strip().endswith("preflight.json")


def test_nonempty_output_is_rejected(tmp_path: Path) -> None:
    harness = _load_harness()
    source = tmp_path / "input.cif"
    source.write_text("data_input\n#\n", encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    (out / "existing").write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit):
        harness.parse_args(
            [
                "--out-dir",
                str(out),
                "--input",
                str(source),
            ]
        )


def test_preflight_only_can_refresh_existing_prep_directory(tmp_path: Path) -> None:
    harness = _load_harness()
    source = tmp_path / "input.cif"
    source.write_text("data_input\n#\n", encoding="utf-8")
    out = tmp_path / "prep"
    out.mkdir()
    (out / "plan.json").write_text("{}", encoding="utf-8")
    args = harness.parse_args(
        [
            "--out-dir",
            str(out),
            "--input",
            str(source),
            "--preflight-only",
        ]
    )
    assert args.preflight_only is True
