from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import astrolabe.model_daemon as model_daemon
from astrolabe.mutators.wrappers.na_mpnn_daemon import (
    NAMPNNHandler,
    build_upstream_args,
)
from astrolabe.mutators.wrappers.na_mpnn_runner import NAMPNNRunner


def _settings(tmp_path: Path) -> dict[str, object]:
    return {
        "repo_path": str(tmp_path / "na_mpnn"),
        "checkpoint": str(tmp_path / "na_mpnn" / "model.pt"),
        "output_root": str(tmp_path / "raw" / "NAMPNN"),
        "temperature": 0.2,
        "batch_size": 3,
        "number_of_batches": 2,
        "design_na_only": 1,
        "parse_na_only": 1,
        "na_shared_tokens": 0,
        "output_pdbs": 0,
        "seed": 11,
        "extra_args": {
            "k_neighbors": 48,
            "pair_bias_AA": "KK:-1.0",
        },
    }


def test_nampnn_daemon_translates_flags_fixed_residues_and_seed(
    tmp_path: Path,
) -> None:
    args = build_upstream_args(
        _settings(tmp_path),
        {
            "job_dir": str(tmp_path / "job"),
            "structure_path": str(tmp_path / "input.cif"),
            "chains_to_design": ["A", "B"],
            "fixed_residues": ["C1", "C2"],
            "parse_these_chains_only": ["A", "B", "C"],
            "seed": 321,
        },
    )

    assert args.mode == "design"
    assert args.model_type == "na_mpnn"
    assert args.temperature == pytest.approx(0.2)
    assert args.batch_size == 3
    assert args.number_of_batches == 2
    assert args.design_na_only == 1
    assert args.parse_na_only == 1
    assert args.na_shared_tokens == 0
    assert args.output_pdbs == 0
    assert args.chains_to_design == "A,B"
    assert args.fixed_residues == "C1 C2"
    assert args.parse_these_chains_only == "ABC"
    assert args.k_neighbors == 48
    assert args.pair_bias_AA == "KK:-1.0"
    assert args.seed == 321


def test_nampnn_daemon_applies_per_request_context_after_static_settings(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    first = build_upstream_args(
        settings,
        {
            "job_dir": str(tmp_path / "job-1"),
            "structure_path": str(tmp_path / "input.cif"),
            "fixed_residues": ["B1"],
            "seed": 41,
            "extra_args": {"seed": 43, "k_neighbors": 32},
        },
    )
    second = build_upstream_args(
        settings,
        {
            "job_dir": str(tmp_path / "job-2"),
            "structure_path": str(tmp_path / "input.cif"),
            "fixed_residues": ["C1"],
            "seed": 47,
            "extra_args": {"seed": 47, "k_neighbors": 64},
        },
    )

    assert (first.seed, first.k_neighbors, first.fixed_residues) == (41, 32, "B1")
    assert (second.seed, second.k_neighbors, second.fixed_residues) == (47, 64, "C1")
    assert first.out_folder != second.out_folder


def test_nampnn_handler_loads_once_and_recovers_after_failure(
    tmp_path: Path,
) -> None:
    class _Resident:
        pass

    class _Upstream:
        def __init__(self) -> None:
            self.load_calls = 0
            self.run_calls = 0
            self.residents = []
            self.fail_next = False
            self.resident = _Resident()

        def load_model(self, args):
            self.load_calls += 1
            return self.resident

        def run_design(self, args, resident):
            self.run_calls += 1
            self.residents.append(resident)
            if self.fail_next:
                self.fail_next = False
                raise RuntimeError("fake design failure")

    upstream = _Upstream()
    handler = NAMPNNHandler(_settings(tmp_path), upstream=upstream)
    payload = {
        "job_dir": str(tmp_path / "job"),
        "structure_path": str(tmp_path / "input.cif"),
        "chains_to_design": ["A"],
        "fixed_residues": ["B1"],
        "parse_these_chains_only": ["A", "B"],
        "seed": 7,
    }

    handler(payload)
    upstream.fail_next = True
    with pytest.raises(RuntimeError, match="fake design failure"):
        handler(payload)
    handler(payload)

    assert upstream.load_calls == 1
    assert upstream.run_calls == 3
    assert upstream.residents == [upstream.resident] * 3


def test_nampnn_handler_loads_once_and_applies_each_request_context(
    tmp_path: Path,
) -> None:
    class _Resident:
        pass

    class _Upstream:
        def __init__(self) -> None:
            self.load_calls = 0
            self.run_args = []
            self.resident = _Resident()

        def load_model(self, _args):
            self.load_calls += 1
            return self.resident

        def run_design(self, args, resident):
            assert resident is self.resident
            self.run_args.append(args)

    upstream = _Upstream()
    handler = NAMPNNHandler(_settings(tmp_path), upstream=upstream)
    for seed, fixed, job in (
        (41, ["B1"], "job-1"),
        (47, ["C1", "C2"], "job-2"),
    ):
        handler(
            {
                "job_dir": str(tmp_path / job),
                "structure_path": str(tmp_path / f"{job}.cif"),
                "chains_to_design": ["A"],
                "fixed_residues": fixed,
                "parse_these_chains_only": ["A", "B", "C"],
                "seed": seed,
                "extra_args": {"k_neighbors": 64},
            }
        )

    assert upstream.load_calls == 1
    assert [args.seed for args in upstream.run_args] == [41, 47]
    assert [args.fixed_residues for args in upstream.run_args] == ["B1", "C1 C2"]
    assert [args.k_neighbors for args in upstream.run_args] == [64, 64]
    assert upstream.run_args[0].out_folder != upstream.run_args[1].out_folder


def test_nampnn_runner_owns_one_daemon_and_preserves_seed(
    tmp_path: Path, monkeypatch
) -> None:
    repo = tmp_path / "na_mpnn"
    (repo / "inference").mkdir(parents=True)
    (repo / "models" / "design_model").mkdir(parents=True)
    (repo / "models" / "design_model" / "s_19137.pt").write_text(
        "checkpoint", encoding="utf-8"
    )
    structure = tmp_path / "step_001_A_model.cif"
    structure.write_text("data_test\n", encoding="utf-8")
    output_root = tmp_path / "raw" / "NAMPNN"
    job_dir = output_root / "job"
    (job_dir / "seqs").mkdir(parents=True)
    (job_dir / "specificity").mkdir()
    (job_dir / "seqs" / "step_001_A_model.fa").write_text(
        ">best, overall_confidence=0.8\nACDE\n",
        encoding="utf-8",
    )
    np.savez(
        job_dir / "specificity" / "step_001_A_model.npz",
        predicted_ppm=np.zeros((1, 1), dtype=np.float32),
        restype_to_int={"ALA": 0},
        encoded_residues=np.array(["A1"], dtype=object),
    )
    created = []

    class _OwnedDaemon:
        def __init__(self, spec, **kwargs):
            self.spec = spec
            self.kwargs = kwargs
            self.requests = []
            created.append(self)

        def request(self, payload):
            self.requests.append(payload)

        def shutdown(self):
            self.closed = True

    monkeypatch.setattr(model_daemon, "OwnedModelDaemon", _OwnedDaemon)
    runner = NAMPNNRunner(
        {
            "repo_path": str(repo),
            "execution_mode": "daemon",
            "seed": 19,
            "temperature": 0.3,
            "na_shared_tokens": 0,
        },
        output_root=output_root,
    )

    first = runner.design_sequences(
        structure_path=structure,
        job_name="job",
        chains_to_design="A",
        fixed_residues=["B1", "B2"],
    )
    second = runner.design_sequences(
        structure_path=structure,
        job_name="job",
        chains_to_design="A",
        fixed_residues=["B1", "B2"],
    )
    runner.extra_args["seed"] = 23
    third = runner.design_sequences(
        structure_path=structure,
        job_name="job",
        chains_to_design="A",
        fixed_residues=["C1", "C2"],
    )

    assert first["sequence"] == second["sequence"] == third["sequence"] == "ACDE"
    assert len(created) == 1
    assert created[0].spec.backend == "na_mpnn"
    assert created[0].spec.settings["seed"] == 19
    assert created[0].spec.settings["temperature"] == pytest.approx(0.3)
    assert "output_root" not in created[0].spec.settings
    assert [request["seed"] for request in created[0].requests] == [19, 19, 23]
    assert "seed" not in created[0].requests[0]["extra_args"]
    assert "seed" not in created[0].requests[1]["extra_args"]
    assert created[0].requests[2]["extra_args"]["seed"] == 23
    assert created[0].requests[2]["fixed_residues"] == ["C1", "C2"]
    runner.shutdown()


def test_nampnn_runner_includes_upstream_inference_modules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "na_mpnn"
    (repo / "inference").mkdir(parents=True)
    (repo / "models" / "design_model").mkdir(parents=True)
    (repo / "models" / "design_model" / "s_19137.pt").write_text(
        "checkpoint", encoding="utf-8"
    )
    monkeypatch.setenv("PYTHONPATH", "/ambient")
    runner = NAMPNNRunner(
        {"repo_path": str(repo), "execution_mode": "subprocess"},
        output_root=tmp_path / "raw",
    )

    assert runner._patched_env()["PYTHONPATH"].split(os.pathsep) == [
        str(repo),
        str(repo / "inference"),
        "/ambient",
    ]


def test_nampnn_subprocess_uses_current_payload_seed_and_extra_args(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "na_mpnn"
    (repo / "inference").mkdir(parents=True)
    (repo / "models" / "design_model").mkdir(parents=True)
    (repo / "models" / "design_model" / "s_19137.pt").write_text(
        "checkpoint", encoding="utf-8"
    )
    structure = tmp_path / "step.cif"
    structure.write_text("data_test\n", encoding="utf-8")
    output_root = tmp_path / "raw"
    job_dir = output_root / "job"
    (job_dir / "seqs").mkdir(parents=True)
    (job_dir / "specificity").mkdir()
    (job_dir / "seqs" / "step.fa").write_text(
        ">best, overall_confidence=0.8\nACDE\n",
        encoding="utf-8",
    )
    np.savez(
        job_dir / "specificity" / "step.npz",
        predicted_ppm=np.zeros((1, 1), dtype=np.float32),
        restype_to_int={"ALA": 0},
        encoded_residues=np.array(["A1"], dtype=object),
    )
    calls = []
    monkeypatch.setattr(
        "subprocess.run",
        lambda command, **_kwargs: calls.append(command),
    )
    runner = NAMPNNRunner(
        {
            "repo_path": str(repo),
            "execution_mode": "subprocess",
            "seed": 19,
            "extra_args": {"k_neighbors": 32},
        },
        output_root=output_root,
    )
    runner.extra_args.update({"seed": 23, "k_neighbors": 64})

    runner.design_sequences(structure_path=structure, job_name="job")

    command = calls[0]
    assert command.count("--seed") == 1
    assert command[command.index("--seed") + 1] == "23"
    assert command[command.index("--k-neighbors") + 1] == "64"
