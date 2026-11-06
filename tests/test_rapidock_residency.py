from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from textwrap import dedent
from typing import Any
from types import SimpleNamespace

import pytest


def _write_fake_api(repo_root: Path, *, fail_once: bool = False) -> None:
    package_dir = repo_root / "RAPiDock"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    api_source = """
import json
import os
import sys
from pathlib import Path


class RAPiDockInference:
    def __init__(self, model_dir, ckpt, confidence_model_dir=None, confidence_ckpt="best_model.pt", device=None):
        print("CONSTRUCTOR-STDOUT")
        os.write(1, b"NATIVE-CONSTRUCTOR-STDOUT\\n")
        self.state_path = Path(__file__).resolve().parent / "api_state.json"
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        state["constructor_calls"] += 1
        state["constructor_ids"].append(id(self))
        self.state_path.write_text(json.dumps(state), encoding="utf-8")
        self.confidence_model = None

    def _mutate_state(self, **changes):
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        state.update(changes)
        self.state_path.write_text(json.dumps(state), encoding="utf-8")
        return state

    def run_inference_strict(
        self,
        output_dir,
        complex_name,
        protein_description,
        peptide_description,
        N,
        batch_size,
        no_final_step_noise,
        inference_steps,
        actual_steps,
        save_visualisation,
        scoring_function,
        fastrelax,
        cpu,
        conformation_type,
        conformation_partial,
    ):
        print("INFERENCE-STDOUT")
        os.write(1, b"NATIVE-INFERENCE-STDOUT\\n")
        print("INFERENCE-STDERR", file=sys.stderr)
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        state["run_attempts"] += 1
        state["run_object_ids"].append(id(self))
        state["last_arguments"] = {
            "output_dir": output_dir,
            "complex_name": complex_name,
            "protein_description": protein_description,
            "peptide_description": peptide_description,
            "N": N,
            "batch_size": batch_size,
            "no_final_step_noise": no_final_step_noise,
            "inference_steps": inference_steps,
            "actual_steps": actual_steps,
            "save_visualisation": save_visualisation,
            "scoring_function": scoring_function,
            "fastrelax": fastrelax,
            "cpu": cpu,
            "conformation_type": conformation_type,
            "conformation_partial": conformation_partial,
        }
        self.state_path.write_text(json.dumps(state), encoding="utf-8")
        if __FAIL_ONCE__ and not state["failed_once"]:
            self._mutate_state(failed_once=True)
            raise RuntimeError("synthetic API failure")

        if scoring_function == "confidence" and self.confidence_model is None:
            self.confidence_model = object()
            self._mutate_state(confidence_loads=state["confidence_loads"] + 1)

        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        state["run_calls"] += 1
        self.state_path.write_text(json.dumps(state), encoding="utf-8")
        write_dir = Path(output_dir) / complex_name
        write_dir.mkdir(parents=True, exist_ok=True)
        candidates = [write_dir / "rank2_confidence.pdb", write_dir / "rank10_confidence.pdb"]
        for path in candidates:
            path.write_text("MODEL 1\\nEND\\n", encoding="utf-8")
        return {complex_name: {"candidate_paths": [str(path) for path in candidates]}}
""".replace("__FAIL_ONCE__", repr(fail_once)).strip()
    (package_dir / "API.py").write_text(api_source + "\n", encoding="utf-8")
    (package_dir / "api_state.json").write_text(
        json.dumps(
            {
                "constructor_calls": 0,
                "constructor_ids": [],
                "run_attempts": 0,
                "run_calls": 0,
                "run_object_ids": [],
                "confidence_loads": 0,
                "failed_once": False,
            }
        ),
        encoding="utf-8",
    )


def _write_pdb(path: Path, atom_name: str) -> Path:
    path.write_text(
        "\n".join(
            [
                f"ATOM      1  {atom_name:<4}ALA A   1       0.000   0.000   0.000  1.00 20.00           C",
                "TER",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _start_daemon(
    *,
    repo_root: Path,
    model_dir: Path,
    confidence_model_dir: Path | None = None,
) -> subprocess.Popen[str]:
    from astrolabe.structure_processors.wrappers.rapidock_runner import (
        default_rapidock_daemon_script_path,
    )

    command = [
        sys.executable,
        "-u",
        str(default_rapidock_daemon_script_path()),
        "--repo-path",
        str(repo_root),
        "--model-dir",
        str(model_dir),
        "--ckpt",
        "rapidock_global.pt",
    ]
    if confidence_model_dir is not None:
        command.extend(
            [
                "--confidence-model-dir",
                str(confidence_model_dir),
                "--confidence-ckpt",
                "best_model.pt",
            ]
        )
    return subprocess.Popen(
        command,
        cwd=str(repo_root),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )


def _request(proc: subprocess.Popen[str], payload: dict[str, Any]) -> dict[str, Any]:
    assert proc.stdin is not None
    assert proc.stdout is not None
    proc.stdin.write(json.dumps(payload) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    assert line
    return json.loads(line)


def _dock_payload(request_id: str, *, protein: Path, peptide: Path, output: Path) -> dict[str, Any]:
    return {
        "type": "dock",
        "id": request_id,
        "tasks": [
            {
                "name": "case",
                "protein_pdb": str(protein),
                "peptide_pdb": str(peptide),
                "output_dir": str(output),
            }
        ],
        "params": {
            "num_samples_bank": 2,
            "batch_size": 2,
            "inference_steps": 7,
            "actual_steps": 3,
            "no_final_step_noise": True,
            "scoring_function": "confidence",
            "fastrelax": False,
            "save_visualisation": True,
            "cpu": 2,
            "conformation_type": "E",
            "conformation_partial": None,
        },
    }


def _finish_daemon(proc: subprocess.Popen[str]) -> tuple[str, str]:
    response = _request(proc, {"type": "shutdown", "id": "shutdown"})
    assert response["type"] == "ok"
    proc.wait(timeout=10)
    assert proc.stdout is not None
    assert proc.stderr is not None
    return proc.stdout.read(), proc.stderr.read()


def test_daemon_retains_one_api_object_for_two_requests(tmp_path: Path) -> None:
    repo_root = tmp_path / "rapidock_repo"
    _write_fake_api(repo_root)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "rapidock_global.pt").write_text("stub", encoding="utf-8")
    confidence_dir = tmp_path / "confidence"
    confidence_dir.mkdir()
    (confidence_dir / "best_model.pt").write_text("stub", encoding="utf-8")
    protein = _write_pdb(tmp_path / "protein.pdb", "CA")
    peptide = _write_pdb(tmp_path / "peptide.pdb", "CB")

    proc = _start_daemon(
        repo_root=repo_root,
        model_dir=model_dir,
        confidence_model_dir=confidence_dir,
    )
    ping = _request(proc, {"type": "ping", "id": "ready"})
    assert ping == {
        "type": "pong",
        "id": "ready",
        "runtime": "maintained-rapidock-api",
        "model_ready": True,
        "load_count": 1,
        "load_attempts": 1,
        "confidence_model_ready": False,
        "confidence_load_count": 0,
        "confidence_load_attempts": 0,
    }

    responses = [
        _request(
            proc,
            _dock_payload(
                f"dock-{index}",
                protein=protein,
                peptide=peptide,
                output=tmp_path / f"output-{index}",
            ),
        )
        for index in (1, 2)
    ]
    for index, response in enumerate(responses, start=1):
        assert response["type"] == "result"
        assert response["id"] == f"dock-{index}"
        candidates = response["results"][0]["candidates"]
        assert [item["rank"] for item in candidates] == [1, 2]
        assert [Path(item["peptide_pdb"]).name for item in candidates] == [
            "rank2_confidence.pdb",
            "rank10_confidence.pdb",
        ]
        assert all(Path(item["peptide_pdb"]).is_file() for item in candidates)

    loaded_confidence = _request(proc, {"type": "ping", "id": "confidence-ready"})
    assert loaded_confidence["confidence_model_ready"] is True
    assert loaded_confidence["confidence_load_count"] == 1
    assert loaded_confidence["confidence_load_attempts"] == 1

    stdout_tail, backend_stderr = _finish_daemon(proc)
    assert stdout_tail == ""
    assert "CONSTRUCTOR-STDOUT" in backend_stderr
    assert "NATIVE-CONSTRUCTOR-STDOUT" in backend_stderr
    assert "INFERENCE-STDOUT" in backend_stderr
    assert "NATIVE-INFERENCE-STDOUT" in backend_stderr
    assert "INFERENCE-STDERR" in backend_stderr
    assert not (repo_root / "inference.py").exists()

    state = json.loads((repo_root / "RAPiDock" / "api_state.json").read_text(encoding="utf-8"))
    assert state["constructor_calls"] == 1
    assert state["run_calls"] == 2
    assert state["confidence_loads"] == 1
    assert len(set(state["run_object_ids"])) == 1
    assert state["last_arguments"] == {
        "output_dir": str(tmp_path / "output-2"),
        "complex_name": "case",
        "protein_description": str(protein),
        "peptide_description": str(peptide),
        "N": 2,
        "batch_size": 2,
        "no_final_step_noise": True,
        "inference_steps": 7,
        "actual_steps": 3,
        "save_visualisation": True,
        "scoring_function": "confidence",
        "fastrelax": False,
        "cpu": 2,
        "conformation_type": "E",
        "conformation_partial": None,
    }


def test_daemon_recovers_from_one_request_error_without_reloading(tmp_path: Path) -> None:
    repo_root = tmp_path / "rapidock_repo"
    _write_fake_api(repo_root, fail_once=True)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "rapidock_global.pt").write_text("stub", encoding="utf-8")
    protein = _write_pdb(tmp_path / "protein.pdb", "CA")
    peptide = _write_pdb(tmp_path / "peptide.pdb", "CB")

    proc = _start_daemon(repo_root=repo_root, model_dir=model_dir)
    assert _request(proc, {"type": "ping", "id": "ready"})["load_count"] == 1
    failed = _request(
        proc,
        _dock_payload(
            "fail",
            protein=protein,
            peptide=peptide,
            output=tmp_path / "failed-output",
        ),
    )
    assert failed["type"] == "error"
    assert failed["error_type"] == "RuntimeError"
    assert "synthetic API failure" in failed["message"]

    recovered = _request(
        proc,
        _dock_payload(
            "recover",
            protein=protein,
            peptide=peptide,
            output=tmp_path / "recovered-output",
        ),
    )
    assert recovered["type"] == "result"
    assert _request(proc, {"type": "ping", "id": "still-ready"})["load_count"] == 1
    _finish_daemon(proc)

    state = json.loads((repo_root / "RAPiDock" / "api_state.json").read_text(encoding="utf-8"))
    assert state["constructor_calls"] == 1
    assert state["run_attempts"] == 2
    assert state["run_calls"] == 1
    assert len(set(state["run_object_ids"])) == 1


def test_daemon_reports_missing_python_api_without_loading_model(tmp_path: Path) -> None:
    repo_root = tmp_path / "rapidock_repo"
    (repo_root / "RAPiDock").mkdir(parents=True)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "rapidock_global.pt").write_text("stub", encoding="utf-8")

    proc = _start_daemon(repo_root=repo_root, model_dir=model_dir)
    response = _request(proc, {"type": "ping", "id": "missing-api"})
    assert response["model_ready"] is False
    assert response["load_count"] == 0
    assert response["load_attempts"] == 1
    assert "RAPiDock/API.py" in response["message"]
    assert "RAPiDockInference" in response["message"]
    _finish_daemon(proc)


def test_runner_surfaces_missing_python_api_at_startup(tmp_path: Path) -> None:
    from astrolabe.structure_processors.wrappers.rapidock_runner import RAPiDockRunner

    repo_root = tmp_path / "rapidock_repo"
    (repo_root / "RAPiDock").mkdir(parents=True)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "rapidock_global.pt").write_text("stub", encoding="utf-8")
    runner = RAPiDockRunner(
        {
            "python_bin": sys.executable,
            "repo_path": str(repo_root),
            "model_dir": str(model_dir),
            "ckpt": "rapidock_global.pt",
            "response_cache_dir": str(tmp_path / "cache"),
            "work_dir": str(tmp_path / "work"),
        }
    )

    with pytest.raises(RuntimeError, match="RAPiDock/API.py"):
        runner.ping()
    assert runner._proc is None


def test_official_runtime_retains_functions_and_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from astrolabe.structure_processors.wrappers import rapidock_runtime
    from astrolabe.structure_processors.wrappers.rapidock_runtime import (
        OfficialRAPiDockRuntime,
    )

    def _unexpected_subprocess(*args, **kwargs):
        raise AssertionError("official runtime must not invoke per-request subprocesses")

    monkeypatch.setattr(rapidock_runtime.subprocess, "run", _unexpected_subprocess)
    repo_root = tmp_path / "official_repo"
    repo_root.mkdir()
    (repo_root / "inference.py").write_text("", encoding="utf-8")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "rapidock_global.pt").write_text("stub", encoding="utf-8")
    (model_dir / "model_parameters.yml").write_text(
        "tr_sigma_max: 1.0\n", encoding="utf-8"
    )
    confidence_dir = tmp_path / "confidence"
    confidence_dir.mkdir()
    (confidence_dir / "best_model.pt").write_text("stub", encoding="utf-8")
    (confidence_dir / "model_parameters.yml").write_text(
        "confidence: true\n", encoding="utf-8"
    )

    class _FakeConfidenceModel:
        def load_state_dict(self, state_dict, strict):
            self.state_dict = state_dict
            self.strict = strict

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

    class _FakeTorch:
        load_calls = []

        class cuda:
            @staticmethod
            def is_available():
                return False

        @staticmethod
        def device(value):
            return value

        @staticmethod
        def load(path, map_location):
            _FakeTorch.load_calls.append(path)
            return {"model": "confidence-state"}

    class _FakeDataset:
        def __init__(self, name):
            self.complex_names = [name]

    class _FakeESM2:
        def __init__(
            self,
            num_layers,
            embed_dim,
            attention_heads,
            alphabet,
            token_dropout,
        ):
            self.num_layers = num_layers

    class _FakeOfficial:
        def __init__(self):
            self.torch = _FakeTorch()
            self.score_checkpoint_calls = []
            self.dataset_args = []
            self.dataset_score_args = []
            self.loader_calls = 0
            self.process_args = []
            self.process_score_args = []
            self.confidence_get_model_calls = 0
            self.DataListLoader = self._data_list_loader

        def load_model(self, score_model_args, ckpt_path, device):
            self.score_checkpoint_calls.append(
                (score_model_args, ckpt_path, device)
            )
            return SimpleNamespace(name="score-model")

        def prepare_data(self, args, score_model_args):
            self.dataset_args.append(vars(args).copy())
            self.dataset_score_args.append(score_model_args)
            return _FakeDataset(args.complex_name)

        def _data_list_loader(self, *, dataset, batch_size, shuffle):
            self.loader_calls += 1
            assert batch_size == 1
            assert shuffle is False
            return [[SimpleNamespace(success=True)]]

        def get_model(self, args, *, no_parallel, confidence_mode):
            self.confidence_get_model_calls += 1
            assert no_parallel is True
            assert confidence_mode is True
            return _FakeConfidenceModel()

        def process_complex(
            self,
            model,
            confidence_model,
            score_model_args,
            args,
            original_complex_graph,
            write_dir,
        ):
            self.process_args.append(vars(args).copy())
            self.process_score_args.append(score_model_args)
            assert model is not None
            assert confidence_model is not None
            path = Path(write_dir)
            path.mkdir(parents=True, exist_ok=True)
            for name in ("rank10_confidence.pdb", "rank2_confidence.pdb"):
                (path / name).write_text("MODEL 1\nEND\n", encoding="utf-8")

    official = _FakeOfficial()
    fake_dataset_module = SimpleNamespace(pretrained=SimpleNamespace())
    runtime = OfficialRAPiDockRuntime(
        repo_path=repo_root,
        model_dir=model_dir,
        ckpt="rapidock_global.pt",
        confidence_model_dir=str(confidence_dir),
        confidence_ckpt="best_model.pt",
        device="cpu",
        official_module=official,
        verify_source=False,
        esm2_class=_FakeESM2,
        dataset_module=fake_dataset_module,
    )
    runtime.load()

    startup = runtime.status()
    assert startup["runtime"] == "official-rapidock-functions"
    assert startup["model_ready"] is True
    assert startup["load_count"] == 1
    assert startup["confidence_model_ready"] is False
    assert startup["confidence_load_count"] == 0
    assert startup["esm_model_ready"] is False

    protein = _write_pdb(tmp_path / "protein.pdb", "CA")
    peptide = _write_pdb(tmp_path / "peptide.pdb", "CB")
    params = {
        "num_samples_bank": 3,
        "batch_size": 2,
        "inference_steps": 9,
        "actual_steps": 4,
        "no_final_step_noise": True,
        "scoring_function": "confidence",
        "fastrelax": False,
        "save_visualisation": False,
        "cpu": 2,
        "conformation_type": "E",
        "conformation_partial": None,
    }
    responses = [
        runtime.run(
            [
                {
                    "name": "case",
                    "protein_pdb": str(protein),
                    "peptide_pdb": str(peptide),
                    "output_dir": str(tmp_path / f"official-output-{index}"),
                }
            ],
            params,
        )
        for index in (1, 2)
    ]

    assert len(official.score_checkpoint_calls) == 1
    assert official.loader_calls == 2
    assert len(official.dataset_args) == 2
    assert len(official.process_args) == 2
    assert official.confidence_get_model_calls == 1
    assert [Path(path).name for path in _FakeTorch.load_calls] == [
        "best_model.pt"
    ]
    assert len({id(item) for item in official.dataset_score_args}) == 1
    assert len({id(item) for item in official.process_score_args}) == 1
    for index, response in enumerate(responses, start=1):
        candidates = response[0]["candidates"]
        assert [item["rank"] for item in candidates] == [1, 2]
        assert [Path(item["peptide_pdb"]).name for item in candidates] == [
            "rank2_confidence.pdb",
            "rank10_confidence.pdb",
        ]
        expected = {
            "complex_name": "case",
            "protein_description": str(protein),
            "peptide_description": str(peptide),
            "output_dir": str(tmp_path / f"official-output-{index}"),
            "N": 3,
            "batch_size": 2,
            "no_final_step_noise": True,
            "inference_steps": 9,
            "actual_steps": 4,
            "save_visualisation": False,
            "scoring_function": "confidence",
            "fastrelax": False,
            "cpu": 2,
            "conformation_type": "E",
            "conformation_partial": None,
        }
        for key, value in expected.items():
            assert official.dataset_args[index - 1][key] == value
            assert official.process_args[index - 1][key] == value

    loaded = runtime.status()
    assert loaded["model_ready"] is True
    assert loaded["confidence_model_ready"] is True
    assert loaded["confidence_load_count"] == 1


def test_scoped_esm_cache_preserves_rng_in_child_process() -> None:
    script = r"""
import json
import types

import torch

from astrolabe.structure_processors.wrappers.rapidock_runtime import (
    ESM2_EXPECTED_SHAPE,
    _ScopedESMCache,
    _scoped_pretrained,
)

MODEL_NAME = "esm2_t33_650M_UR50D"
SEED = 731
alphabet = object()
state = {
    "constructor_calls": 0,
    "loader_calls": 0,
    "checkpoint_reads": 0,
    "cuda_calls": 0,
}


class FakeESM2:
    def __init__(
        self,
        num_layers,
        embed_dim,
        attention_heads,
        alphabet,
        token_dropout,
    ):
        state["constructor_calls"] += 1
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.alphabet = alphabet
        self.token_dropout = token_dropout
        self.device = "cpu"
        self.rng_probe = float(torch.rand(4096)[0])

    def eval(self):
        return self

    def cuda(self):
        state["cuda_calls"] += 1
        self.device = "cuda"
        return self

    def cpu(self):
        self.device = "cpu"
        return self

    def embed(self):
        return ("fixed", 0.25)


class FakePretrained:
    def load_model_and_alphabet(self, model_name):
        assert model_name == MODEL_NAME
        state["loader_calls"] += 1
        state["checkpoint_reads"] += 1
        return FakeESM2(
            ESM2_EXPECTED_SHAPE[0],
            ESM2_EXPECTED_SHAPE[1],
            ESM2_EXPECTED_SHAPE[2],
            alphabet,
            True,
        ), alphabet


dataset_module = types.SimpleNamespace(pretrained=FakePretrained())
cache = _ScopedESMCache(
    pretrained=dataset_module.pretrained,
    torch_module=torch,
    esm2_class=FakeESM2,
    require_pinned_class=False,
)

torch.manual_seed(SEED)
with _scoped_pretrained(dataset_module, cache):
    first_model, first_alphabet = dataset_module.pretrained.load_model_and_alphabet(
        MODEL_NAME
    )
    first_model.eval().cuda()
    first_embedding = first_model.embed()
assert dataset_module.pretrained is cache.pretrained
assert first_model.device == "cpu"

interleaved_rng = torch.rand(17).tolist()

with _scoped_pretrained(dataset_module, cache):
    second_model, second_alphabet = dataset_module.pretrained.load_model_and_alphabet(
        MODEL_NAME
    )
    second_model.eval().cuda()
    second_embedding = second_model.embed()
assert dataset_module.pretrained is cache.pretrained
assert second_model.device == "cpu"
final_rng = torch.rand(9).tolist()
cached_constructor_calls = state["constructor_calls"]

torch.manual_seed(SEED)
FakeESM2(
    ESM2_EXPECTED_SHAPE[0],
    ESM2_EXPECTED_SHAPE[1],
    ESM2_EXPECTED_SHAPE[2],
    alphabet,
    True,
)
baseline_interleaved = torch.rand(17).tolist()
FakeESM2(
    ESM2_EXPECTED_SHAPE[0],
    ESM2_EXPECTED_SHAPE[1],
    ESM2_EXPECTED_SHAPE[2],
    alphabet,
    True,
)
baseline_final = torch.rand(9).tolist()


class WrongShapeESM2(FakeESM2):
    def __init__(
        self,
        num_layers,
        embed_dim,
        attention_heads,
        alphabet,
        token_dropout,
    ):
        super().__init__(
            12,
            embed_dim,
            attention_heads,
            alphabet,
            token_dropout,
        )


class WrongShapePretrained:
    def load_model_and_alphabet(self, model_name):
        return WrongShapeESM2(
            ESM2_EXPECTED_SHAPE[0],
            ESM2_EXPECTED_SHAPE[1],
            ESM2_EXPECTED_SHAPE[2],
            alphabet,
            True,
        ), alphabet


wrong_module = types.SimpleNamespace(pretrained=WrongShapePretrained())
wrong_cache = _ScopedESMCache(
    pretrained=wrong_module.pretrained,
    torch_module=torch,
    esm2_class=WrongShapeESM2,
    require_pinned_class=False,
)
unsupported_message = None
try:
    with _scoped_pretrained(wrong_module, wrong_cache):
        wrong_module.pretrained.load_model_and_alphabet(MODEL_NAME)
except RuntimeError as exc:
    unsupported_message = str(exc)
assert wrong_module.pretrained is wrong_cache.pretrained
assert "Unsupported pinned ESM2 shape" in unsupported_message


class WrongConstructorESM2:
    def __init__(self, num_layers, embed_dim, attention_heads, alphabet):
        pass


try:
    _ScopedESMCache(
        pretrained=FakePretrained(),
        torch_module=torch,
        esm2_class=WrongConstructorESM2,
        require_pinned_class=False,
    )
except RuntimeError as exc:
    constructor_message = str(exc)
else:
    raise AssertionError("unsupported ESM2 constructor was accepted")

print(
    json.dumps(
        {
            "loader_calls": state["loader_calls"],
            "checkpoint_reads": state["checkpoint_reads"],
            "constructor_calls": cached_constructor_calls,
            "cuda_calls": state["cuda_calls"],
            "returned_same_model": first_model is second_model,
            "returned_same_alphabet": first_alphabet is second_alphabet,
            "embeddings_equal": first_embedding == second_embedding,
            "offload_count": cache.offload_count,
            "rng_equal": (
                interleaved_rng == baseline_interleaved
                and final_rng == baseline_final
            ),
            "proxy_restored": dataset_module.pretrained is cache.pretrained,
            "unsupported_shape": "Unsupported pinned ESM2 shape"
            in unsupported_message,
            "unsupported_constructor": "constructor parameters"
            in constructor_message,
        },
        sort_keys=True,
    )
)
"""
    completed = subprocess.run(
        [sys.executable, "-c", dedent(script)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result == {
        "checkpoint_reads": 1,
        "constructor_calls": 2,
        "cuda_calls": 2,
        "embeddings_equal": True,
        "loader_calls": 1,
        "offload_count": 2,
        "proxy_restored": True,
        "returned_same_alphabet": True,
        "returned_same_model": True,
        "rng_equal": True,
        "unsupported_constructor": True,
        "unsupported_shape": True,
    }


@pytest.mark.parametrize(
    ("revision", "message"),
    [
        (None, "Cannot verify RAPiDock source revision"),
        (
            "8d4295a380b8d89a5ca62163d866e7d1cf6c7053",
            "restricted to official revision",
        ),
    ],
)
def test_official_runtime_rejects_unknown_or_sibling_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    revision: str | None,
    message: str,
) -> None:
    from astrolabe.structure_processors.wrappers import rapidock_runtime

    repo_root = tmp_path / "unknown_repo"
    repo_root.mkdir()
    (repo_root / "inference.py").write_text("", encoding="utf-8")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "rapidock_global.pt").write_text("stub", encoding="utf-8")
    monkeypatch.setattr(rapidock_runtime, "_read_git_revision", lambda _path: revision)
    runtime = rapidock_runtime.build_rapidock_runtime(
        repo_path=repo_root,
        model_dir=model_dir,
        ckpt="rapidock_global.pt",
        confidence_model_dir=None,
        confidence_ckpt="best_model.pt",
        device="cpu",
    )
    assert isinstance(runtime, rapidock_runtime.OfficialRAPiDockRuntime)

    with pytest.raises(RuntimeError, match=message):
        runtime.load()


def test_runtime_selector_requires_real_maintained_api_capability(
    tmp_path: Path,
) -> None:
    from astrolabe.structure_processors.wrappers import rapidock_runtime

    repo_root = tmp_path / "source"
    (repo_root / "RAPiDock").mkdir(parents=True)
    (repo_root / "RAPiDock" / "API.py").write_text(
        "# no RAPiDockInference here\n", encoding="utf-8"
    )
    (repo_root / "inference.py").write_text("", encoding="utf-8")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "rapidock_global.pt").write_text("stub", encoding="utf-8")

    runtime = rapidock_runtime.build_rapidock_runtime(
        repo_path=repo_root,
        model_dir=model_dir,
        ckpt="rapidock_global.pt",
        confidence_model_dir=None,
        confidence_ckpt="best_model.pt",
        device="cpu",
    )

    assert isinstance(runtime, rapidock_runtime.OfficialRAPiDockRuntime)
