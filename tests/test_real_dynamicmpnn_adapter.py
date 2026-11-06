from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import ProtomerTypes
from astrolabe.mutators.base import build_mutators
from astrolabe.mutators.dynamicmpnn import DynamicMPNNMutator
from astrolabe.mutators.ligandmpnn import GroupedLigandMPNNMutator, LigandMPNNMutator
from astrolabe.mutators.registry import MutatorRegistry
from astrolabe.mutators.wrappers import native_dynamicmpnn as dynamic_backend_module
from astrolabe.mutators.wrappers.dynamicmpnn_run import DynamicMPNNRunner


def _raw_probabilities_backend(*, manifest, features):
    assert manifest["pdb_paths"] == ["open.pdb", "closed.pdb"]
    assert features["pdb_path_multi"]
    probs = np.zeros((1, 2, 20), dtype=float)
    probs[0, 0, 0] = 0.25
    probs[0, 0, 1] = 0.75
    probs[0, 1, 2] = 1.0
    return probs


def test_dynamicmpnn_runner_requires_in_process_backend(tmp_path: Path) -> None:
    try:
        DynamicMPNNRunner({"output_dir": str(tmp_path)})
    except RuntimeError as exc:
        assert "execution_mode='daemon'" in str(exc)
        assert "backend_factory" in str(exc)
    else:
        raise AssertionError("Expected missing DynamicMPNN backend command to fail visibly")


def test_dynamicmpnn_runner_reads_raw_probabilities_in_process(tmp_path: Path) -> None:
    runner = DynamicMPNNRunner(
        {
            "output_dir": str(tmp_path),
            "backend_callable": _raw_probabilities_backend,
            "execution_mode": "in_process",
        }
    )
    pdbs = tmp_path / "pdbs.json"
    pdbs.write_text(json.dumps(["open.pdb", "closed.pdb"]), encoding="utf-8")

    output = runner.design_sequences(pdb_path_multi=str(pdbs))

    assert output["dynamicmpnn_backend"] == "Alex-Abrudan/DynamicMPNN"
    assert output["sampling_probs"].shape == (1, 2, 20)
    np.testing.assert_allclose(output["sampling_probs"][0, 0, :2], [0.25, 0.75])
    assert output["dynamicmpnn_backend_mode"] == "in_process"
    assert "dynamicmpnn_probabilities" not in output
    assert "dynamicmpnn_manifest" not in output


def test_dynamicmpnn_runner_subprocess_reads_raw_probabilities(tmp_path: Path) -> None:
    command = (
        "python -c \"import numpy as np; "
        "p=np.zeros((1, 2, 20)); "
        "p[0, 0, 0]=0.25; p[0, 0, 1]=0.75; "
        "p[0, 1, 2]=1.0; "
        "np.save(__import__('sys').argv[1], p)\" {probabilities}"
    )
    runner = DynamicMPNNRunner(
        {
            "output_dir": str(tmp_path),
            "command_template": command,
            "execution_mode": "in_process",
        }
    )
    pdbs = tmp_path / "pdbs.json"
    pdbs.write_text(json.dumps(["open.pdb", "closed.pdb"]), encoding="utf-8")

    output = runner.design_sequences(pdb_path_multi=str(pdbs))

    assert output["dynamicmpnn_backend_mode"] == "subprocess"
    np.testing.assert_allclose(output["sampling_probs"][0, 0, :2], [0.25, 0.75])
    assert Path(output["dynamicmpnn_probabilities"]).exists()
    assert Path(output["dynamicmpnn_manifest"]).exists()


def test_dynamicmpnn_runner_rejects_sample_only_output(tmp_path: Path) -> None:
    command = "mkdir -p {out_dir}/samples && printf 'sequence\\nAC\\n' > {out_dir}/samples/samples.csv"
    runner = DynamicMPNNRunner(
        {
            "output_dir": str(tmp_path),
            "command_template": command,
            "execution_mode": "in_process",
        }
    )
    pdbs = tmp_path / "pdbs.json"
    pdbs.write_text(json.dumps(["open.pdb", "closed.pdb"]), encoding="utf-8")

    try:
        runner.design_sequences(pdb_path_multi=str(pdbs))
    except FileNotFoundError as exc:
        assert "raw probability file" in str(exc)
    else:
        raise AssertionError("Expected sample-only DynamicMPNN output to be rejected")


def test_mutator_registry_separates_dynamicmpnn_from_grouped_ligandmpnn(tmp_path: Path) -> None:
    global_config.reset()
    global_config.init_const("protomers", {})
    mutator_conf = OmegaConf.create(
        {
            "DynamicMPNN": {
                "backend_callable": f"{__name__}:_raw_probabilities_backend",
                "execution_mode": "in_process",
            },
            "mutator_terms": {"D": {"_type": "dynamicmpnn"}},
        }
    )

    mutators = build_mutators(mutator_conf, tmp_path)

    assert isinstance(mutators["D"], DynamicMPNNMutator)
    assert not isinstance(mutators["D"], LigandMPNNMutator)
    assert not isinstance(mutators["D"], GroupedLigandMPNNMutator)
    assert isinstance(mutators["D"].runner, DynamicMPNNRunner)
    assert MutatorRegistry.get("grouped_ligandmpnn") is not MutatorRegistry.get(
        "dynamicmpnn"
    )


def test_dynamicmpnn_mutator_returns_one_joint_output_to_alias_group(tmp_path: Path) -> None:
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("raw_model_results_dir", tmp_path / "raw")
    global_config.init_const("segments", {})
    shared_segment = SimpleNamespace(is_fixed=False)
    global_config.segments["A_1"] = shared_segment
    global_config.segments["B_1"] = shared_segment
    global_config.init_const(
        "protomers",
        {
            "A": SimpleNamespace(
                type=ProtomerTypes.PROTEIN,
                mutable_length=1,
                length=1,
                components=["A_1"],
            ),
            "B": SimpleNamespace(
                type=ProtomerTypes.PROTEIN,
                mutable_length=1,
                length=1,
                components=["B_1"],
            ),
        },
    )
    global_config.set_variable("step", 1)

    class _JointRunner:
        def __init__(self) -> None:
            self.calls = []

        def design_sequences(self, **kwargs):
            self.calls.append(kwargs)
            assert kwargs["pdb_path_multi"]
            paths = json.loads(Path(kwargs["pdb_path_multi"]).read_text(encoding="utf-8"))
            assert paths == ["open.pdb", "closed.pdb"]
            return {"sampling_probs": np.asarray([[[0.25, 0.75]]], dtype=float)}

    runner = _JointRunner()
    mutator = DynamicMPNNMutator(sequence_model_runner=runner)
    output = mutator.compute_mutation(
        {
            "A": {"pdb_path": "open.pdb", "fixed_residues": ""},
            "B": {"pdb_path": "closed.pdb", "fixed_residues": ""},
        }
    )

    assert len(runner.calls) == 1
    np.testing.assert_allclose(output["A"]["sampling_probs"], output["B"]["sampling_probs"])


def test_dynamicmpnn_daemon_factory_loads_once_and_recovers(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "calls.log"
    module = tmp_path / "fake_dynamicmpnn_backend.py"
    module.write_text(
        f"""
from pathlib import Path
import numpy as np

MARKER = Path({str(marker)!r})

def _append(value):
    with MARKER.open("a", encoding="utf-8") as handle:
        handle.write(value + "\\n")

class Backend:
    def __init__(self):
        _append("load")

    def infer(self, *, manifest, features):
        _append("infer:" + ",".join(manifest["pdb_paths"]))
        if features.get("fail"):
            raise RuntimeError("fake DynamicMPNN failure")
        value = float(features["probability"])
        probs = np.zeros((1, 2, 20), dtype=np.float64)
        probs[0, 0, 0] = value
        probs[0, 0, 1] = 1.0 - value
        probs[0, 1, 2] = 1.0
        return probs

def make(settings):
    _append("torch_device:" + repr(settings.get("torch_device")))
    return Backend()
""",
        encoding="utf-8",
    )
    old_pythonpath = os.environ.get("PYTHONPATH", "")
    env = {"PYTHONPATH": os.pathsep.join([str(tmp_path), old_pythonpath])}
    runner = DynamicMPNNRunner(
        {
            "output_dir": str(tmp_path / "outputs"),
            "backend_factory": "fake_dynamicmpnn_backend:make",
            "execution_mode": "daemon",
            "torch_device": None,
            "env": env,
        }
    )
    first_manifest = tmp_path / "first.json"
    second_manifest = tmp_path / "second.json"
    first_manifest.write_text(
        json.dumps(["open_a.pdb", "closed_a.pdb"]), encoding="utf-8"
    )
    second_manifest.write_text(
        json.dumps(["open_b.pdb", "closed_b.pdb"]), encoding="utf-8"
    )

    first = runner.design_sequences(
        pdb_path_multi=str(first_manifest), probability=0.25
    )
    with np.testing.assert_raises_regex(RuntimeError, "fake DynamicMPNN failure"):
        runner.design_sequences(
            pdb_path_multi=str(first_manifest),
            probability=0.3,
            fail=True,
        )
    second = runner.design_sequences(
        pdb_path_multi=str(second_manifest), probability=0.4
    )
    pid = runner.pid
    runner.shutdown()

    np.testing.assert_allclose(first["sampling_probs"][0, 0, :2], [0.25, 0.75])
    np.testing.assert_allclose(second["sampling_probs"][0, 0, :2], [0.4, 0.6])
    assert first["dynamicmpnn_backend_mode"] == "daemon"
    assert second["dynamicmpnn_backend_mode"] == "daemon"
    assert runner.model_metadata["backend"] == "dynamicmpnn"
    assert runner.model_metadata["implementation"] == "Alex-Abrudan/DynamicMPNN"
    assert runner.model_metadata["resident_backend_type"] == "Backend"
    calls = marker.read_text(encoding="utf-8").splitlines()
    assert calls.count("load") == 1
    assert "infer:open_a.pdb,closed_a.pdb" in calls
    assert "infer:open_b.pdb,closed_b.pdb" in calls
    assert "torch_device:None" in calls
    assert pid is not None
    assert not Path(f"/proc/{pid}").exists()


def test_dynamicmpnn_daemon_refuses_command_only_backend(tmp_path: Path) -> None:
    with np.testing.assert_raises_regex(RuntimeError, "requires backend_factory"):
        DynamicMPNNRunner(
            {
                "output_dir": str(tmp_path),
                "command_template": "python dynamicmpnn.py",
                "execution_mode": "daemon",
            }
        )


def test_dynamicmpnn_daemon_refuses_callable_only_backend(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="requires backend_factory"):
        DynamicMPNNRunner(
            {
                "output_dir": str(tmp_path),
                "backend_callable": _raw_probabilities_backend,
                "execution_mode": "daemon",
            }
        )


def test_dynamicmpnn_environment_selects_in_process_compatibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ASTROLABE_MODEL_EXECUTION", "in_process")
    runner = DynamicMPNNRunner(
        {
            "output_dir": str(tmp_path),
            "backend_callable": _raw_probabilities_backend,
        }
    )
    pdbs = tmp_path / "pdbs.json"
    pdbs.write_text(json.dumps(["open.pdb", "closed.pdb"]), encoding="utf-8")

    output = runner.design_sequences(pdb_path_multi=str(pdbs))

    assert runner.mode.value == "in_process"
    assert output["dynamicmpnn_backend_mode"] == "in_process"


def test_dynamicmpnn_helper_threads_exact_torch_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def _kernel(settings, loader, *, torch_device=None):
        captured["settings"] = settings
        captured["loader"] = loader
        captured["torch_device"] = torch_device
        return object()

    monkeypatch.setattr(dynamic_backend_module, "kernel_handler", _kernel)
    dynamic_backend_module.create_handler(
        {"backend_factory": "example.backend:factory", "torch_device": "cuda:1"}
    )

    assert captured["settings"]["torch_device"] == "cuda:1"
    assert captured["torch_device"] == "cuda:1"
