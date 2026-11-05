from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

import astrolabe.model_daemon as model_daemon
from astrolabe.core import global_config
from astrolabe.core.folding_input import Input, Ligand, ProteinChain
from astrolabe.evaluators.opendde import OpenDDEEvaluator
from astrolabe.evaluators.utils import construct_evaluators
from astrolabe.evaluators.wrappers.opendde_daemon import OpenDDEHandler
from astrolabe.evaluators.wrappers.opendde import (
    OpenDDERunner,
    collect_opendde_outputs,
    opendde_json_from_folding_input,
    write_opendde_input,
)


def _init_global(tmp_path: Path, *, step: int = 7) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.set_variable("step", step)


def test_opendde_json_from_folding_input_preserves_chains_and_bonds(tmp_path: Path) -> None:
    _init_global(tmp_path)
    folding_input = Input(
        name="AB",
        chains=(
            ProteinChain(
                id="A",
                sequence="ACDX",
                ptms=(("MSE", 2),),
                unpaired_msa=">A\nACDX\n",
            ),
            Ligand(id="B", ccd_ids=("ATP",)),
        ),
        rng_seeds=(123,),
        bonded_atom_pairs=((("A", 2, "SG"), ("B", 1, "P")),),
    )

    job, chain_order, protein_lengths = opendde_json_from_folding_input(
        folding_input,
        "AB",
        job_name="step_007_AB",
        include_msa_paths=True,
    )

    assert job["name"] == "step_007_AB"
    assert job["modelSeeds"] == [123]
    assert job["sequences"][0]["proteinChain"]["sequence"] == "ACDX"
    assert job["sequences"][0]["proteinChain"]["modifications"] == [
        {"ptmType": "CCD_MSE", "ptmPosition": 2}
    ]
    msa_path = Path(job["sequences"][0]["proteinChain"]["unpairedMsaPath"])
    assert msa_path.read_text(encoding="utf-8") == ">A\nACDX\n"
    assert job["sequences"][1] == {
        "ligand": {"ligand": "CCD_ATP", "count": 1, "id": ["B"]}
    }
    assert job["covalent_bonds"] == [
        {
            "entity1": 1,
            "copy1": 1,
            "position1": 2,
            "atom1": "SG",
            "entity2": 2,
            "copy2": 1,
            "position2": 1,
            "atom2": "P",
        }
    ]
    assert chain_order == [
        {"id": "A", "kind": "protein", "entity_id": 1},
        {"id": "B", "kind": "ligand", "entity_id": 2},
    ]
    assert protein_lengths == {"A": 4}


def test_opendde_runner_from_config_defaults_to_current_interpreter(tmp_path: Path) -> None:
    runner = OpenDDERunner.from_config({"repo_path": str(tmp_path)})

    assert runner.python_bin == sys.executable
    assert runner.repo_path == tmp_path


def test_opendde_runner_builds_subprocess_command_and_logs(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "opendde"
    repo.mkdir()
    root = tmp_path / "assets"
    root.mkdir()
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\n", encoding="utf-8")

    calls = []

    def fake_run(cmd, *, cwd, env, capture_output, text, timeout):
        calls.append(
            {
                "cmd": cmd,
                "cwd": cwd,
                "env": env,
                "capture_output": capture_output,
                "text": text,
                "timeout": timeout,
            }
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="ok\n", stderr="warn\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    runner = OpenDDERunner(
        python_bin=str(python),
        repo_path=repo,
        opendde_root_dir=root,
        num_samples=1,
        use_msa=False,
        timeout_seconds=12,
        execution_mode="subprocess",
    )
    out_dir = tmp_path / "raw"

    runner.run_inference(input_json=tmp_path / "input.json", out_dir=out_dir)

    assert calls[0]["cwd"] == str(repo)
    assert calls[0]["env"]["OPENDDE_ROOT_DIR"] == str(root)
    assert str(repo) in calls[0]["env"]["PYTHONPATH"].split(os.pathsep)
    assert calls[0]["timeout"] == 12
    cmd = calls[0]["cmd"]
    assert cmd[:4] == [str(python), "-m", "runner.batch_inference", "pred"]
    assert "--use_msa" in cmd
    assert cmd[cmd.index("--use_msa") + 1] == "false"
    assert (out_dir / "opendde.stdout.log").read_text(encoding="utf-8") == "ok\n"
    assert (out_dir / "opendde.stderr.log").read_text(encoding="utf-8") == "warn\n"


def test_collect_opendde_outputs_returns_standard_score_payload(tmp_path: Path) -> None:
    _init_global(tmp_path)
    prepared = write_opendde_input(
        Input(
            name="AB",
            chains=(
                ProteinChain(id="A", sequence="AAAA", ptms=()),
                ProteinChain(id="B", sequence="CC", ptms=()),
            ),
            rng_seeds=(101,),
        ),
        oligo_name="AB",
        job_name="step_007_AB",
        input_dir=tmp_path / "sequences" / "opendde",
        output_dir=tmp_path / "raw" / "opendde",
        include_msa_paths=False,
    )
    pred_dir = prepared.output_dir / "step_007_AB" / "seed_101" / "predictions"
    pred_dir.mkdir(parents=True)
    (pred_dir / "step_007_AB_sample_0.cif").write_text("data_stub\n", encoding="utf-8")
    (pred_dir / "step_007_AB_summary_confidence_sample_0.json").write_text(
        json.dumps(
            {
                "plddt": 91.0,
                "ptm": 0.81,
                "iptm": 0.72,
                "gpde": 3.4,
                "chain_plddt": [90.0, 80.0],
                "chain_pair_iptm": [[0.0, 0.7], [0.7, 0.0]],
            }
        ),
        encoding="utf-8",
    )

    payload = collect_opendde_outputs(prepared, structures_dir=tmp_path / "structures")

    assert payload["plddt"] == [91.0]
    assert payload["ptm"] == [0.81]
    assert payload["iptm"] == [0.72]
    assert payload["pde"] == [3.4]
    assert np.allclose(payload["per_proto_scores"]["A"][0], np.full(4, 90.0))
    assert np.allclose(payload["per_proto_scores"]["B"][0], np.full(2, 80.0))
    assert np.allclose(payload["chain_pair_scores"]["chain_pair_iptm"][0], [[0.0, 0.7], [0.7, 0.0]])
    assert payload["complex_path"].read_text(encoding="utf-8") == "data_stub\n"


def test_opendde_evaluator_wires_outputs_to_standard_scores(tmp_path: Path) -> None:
    _init_global(tmp_path)

    class _Runner:
        use_msa = False
        use_template = False
        use_rna_msa = False

        def run_inference(self, *, input_json: Path, out_dir: Path) -> None:
            data = json.loads(input_json.read_text(encoding="utf-8"))
            job_name = data[0]["name"]
            pred_dir = out_dir / job_name / "seed_101" / "predictions"
            pred_dir.mkdir(parents=True)
            (pred_dir / f"{job_name}_sample_0.cif").write_text("data_stub\n", encoding="utf-8")
            (pred_dir / f"{job_name}_summary_confidence_sample_0.json").write_text(
                json.dumps({"plddt": 88.0, "ptm": 0.7, "iptm": 0.6, "chain_plddt": [88.0]}),
                encoding="utf-8",
            )

    evaluator = OpenDDEEvaluator(_Runner(), tmp_path / "raw" / "opendde")
    scores = evaluator.evaluate(
        {
            "A": Input(
                name="A",
                chains=(ProteinChain(id="A", sequence="AAAA", ptms=()),),
                rng_seeds=(101,),
            )
        },
        {},
    )

    assert scores.names == ["A"]
    assert scores.plddt["A"] == [0.88]
    assert scores.ptm["A"] == [0.7]
    assert np.allclose(scores.per_proto_scores["A"][0], np.full(4, 0.88))
    assert scores.complex_paths["A"].name == "step_007_A_model.cif"


def test_construct_evaluators_builds_opendde_runner(tmp_path: Path) -> None:
    class _EvalConf:
        opendde = OmegaConf.create(
            {
                "python_bin": "/usr/bin/python3",
                "repo_path": str(tmp_path),
                "opendde_root_dir": str(tmp_path / "assets"),
                "num_samples": 1,
            }
        )
        evaluator_terms = {"OpenDDE": {"_type": "OpenDDE"}}

    evaluators = construct_evaluators(_EvalConf(), raw_model_results_dir=tmp_path / "raw")

    assert set(evaluators) == {"OpenDDE"}
    evaluator = evaluators["OpenDDE"]
    assert isinstance(evaluator, OpenDDEEvaluator)
    assert evaluator.model_runner.python_bin == "/usr/bin/python3"
    assert evaluator.model_runner.repo_path == tmp_path
    assert evaluator.output_dir == tmp_path / "raw" / "opendde"


def test_opendde_daemon_retains_runner_and_recovers_after_failure(tmp_path: Path) -> None:
    input_json = tmp_path / "input.json"
    input_json.write_text(
        json.dumps([{"name": "AB", "modelSeeds": [123], "sequences": []}]),
        encoding="utf-8",
    )
    out_dirs = [
        tmp_path / "raw" / "opendde" / "step_1",
        tmp_path / "raw" / "opendde" / "step_2",
    ]
    for out_dir in out_dirs:
        out_dir.mkdir(parents=True)

    class _Config(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__

    class _BatchInference:
        def __init__(self) -> None:
            self.load_calls = 0
            self.seeds: list[list[int]] = []
            self.runner = SimpleNamespace(
                configs=_Config(
                    {
                        "need_atom_confidence": False,
                        "sorted_by_ranking_score": True,
                    }
                ),
                init_dumper=self._init_dumper,
                foldcp_recorder=SimpleNamespace(
                    jsonl_path="",
                    rank=0,
                    write_rank_sidecar=False,
                ),
            )

        def _init_dumper(self, **kwargs):
            self.runner.dumper = SimpleNamespace(base_dir=self.runner.dump_dir)

        def get_default_runner(self, **kwargs):
            self.load_calls += 1
            self.options = kwargs
            self.runner.dumper = SimpleNamespace(base_dir=kwargs["dump_dir"])
            return self.runner

        def preprocess_input(self, input_json, **kwargs):
            payload = json.loads(Path(input_json).read_text(encoding="utf-8"))
            self.seeds.append(payload[0]["modelSeeds"])
            return input_json

    class _Inference:
        def __init__(self) -> None:
            self.calls = 0
            self.fail_next = False
            self.runners = []

        def infer_predict(self, runner, configs):
            self.calls += 1
            self.runners.append(runner)
            if self.fail_next:
                self.fail_next = False
                raise RuntimeError("fake inference failure")
            assert configs["input_json_path"] == str(input_json)
            Path(runner.dumper.base_dir, f"result_{self.calls}.txt").write_text(
                "ok", encoding="utf-8"
            )

    batch = _BatchInference()
    inference = _Inference()
    handler = OpenDDEHandler(
        {
            "repo_path": str(tmp_path),
            "num_samples": 2,
            "use_msa": True,
        },
        batch_inference=batch,
        inference=inference,
    )
    first_payload = {"input_json": str(input_json), "out_dir": str(out_dirs[0])}
    second_payload = {"input_json": str(input_json), "out_dir": str(out_dirs[1])}

    handler(first_payload)
    inference.fail_next = True
    with pytest.raises(RuntimeError, match="fake inference failure"):
        handler(first_payload)
    handler(second_payload)

    assert batch.load_calls == 1
    assert batch.options["n_sample"] == 2
    assert batch.options["use_msa"] is True
    assert batch.seeds == [[123], [123], [123]]
    assert inference.calls == 3
    assert len({id(runner) for runner in inference.runners}) == 1
    assert (out_dirs[0] / "result_1.txt").is_file()
    assert (out_dirs[1] / "result_3.txt").is_file()


def test_opendde_runner_owns_one_daemon_for_multiple_requests(
    tmp_path: Path, monkeypatch
) -> None:
    repo = tmp_path / "opendde"
    repo.mkdir()
    input_json = tmp_path / "input.json"
    input_json.write_text("[]", encoding="utf-8")
    out_dir = tmp_path / "raw"

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
    runner = OpenDDERunner(
        python_bin=sys.executable,
        repo_path=repo,
        num_samples=2,
        timeout_seconds=12,
    )

    runner.run_inference(input_json=input_json, out_dir=out_dir)
    runner.run_inference(input_json=input_json, out_dir=out_dir)

    assert len(created) == 1
    assert created[0].spec.backend == "opendde"
    assert created[0].spec.settings["num_samples"] == 2
    assert "out_dir" not in created[0].spec.settings
    assert "foldcp_metrics_jsonl" not in created[0].spec.settings
    assert created[0].kwargs["request_timeout"] == 12
    assert created[0].requests == [
        {
            "input_json": str(input_json),
            "out_dir": str(out_dir),
            "foldcp_metrics_jsonl": "",
        },
        {
            "input_json": str(input_json),
            "out_dir": str(out_dir),
            "foldcp_metrics_jsonl": "",
        },
    ]
    assert (out_dir / "opendde.stdout.log").is_file()
    runner.shutdown()
