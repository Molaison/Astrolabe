from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from astrolabe.mutators.base import build_mutators
from astrolabe.mutators.native_daemons import NativeSequenceRunner
from astrolabe.mutators.registry import MutatorRegistry


def _prepend_child_pythonpath(monkeypatch, path: Path) -> None:
    existing = os.environ.get("PYTHONPATH")
    value = str(path) if not existing else f"{path}{os.pathsep}{existing}"
    monkeypatch.setenv("PYTHONPATH", value)


def _write_sequence_factory(path: Path) -> None:
    path.write_text(
        dedent(
            """
            import json
            import os
            import random
            from pathlib import Path

            import numpy as np
            import torch

            from astrolabe.model_execution import kernel_handler


            def _append(payload):
                path = Path(os.environ["FAKE_SEQUENCE_AUDIT"])
                records = (
                    json.loads(path.read_text(encoding="utf-8"))
                    if path.exists()
                    else []
                )
                records.append(payload)
                path.write_text(json.dumps(records), encoding="utf-8")


            def loader(settings):
                config = dict(settings["runner_config"])
                constructor_random = [
                    random.random(),
                    float(np.random.random()),
                    float(torch.rand(1).item()),
                ]
                _append(
                    {
                        "event": "load",
                        "pid": os.getpid(),
                        "config": config,
                        "random": constructor_random,
                    }
                )
                calls = 0

                def infer(payload):
                    nonlocal calls
                    calls += 1
                    draws = [
                        random.random(),
                        float(np.random.random()),
                        float(torch.rand(1).item()),
                    ]
                    features = dict(payload["features"])
                    batch_size = int(features.get("batch_size", 1))
                    length = int(features.get("length", 3))
                    probs = torch.arange(
                        batch_size * length * 20,
                        dtype=torch.float32,
                    ).reshape(batch_size, length, 20)
                    _append(
                        {
                            "event": "infer",
                            "pid": os.getpid(),
                            "call": calls,
                            "random": draws,
                            "features": features,
                        }
                    )
                    return {
                        "sampling_probs": probs,
                        "feature_dict": features,
                        "name": features.get("name"),
                    }

                return infer, {
                    "output_dir": config["out_folder"],
                    "device": "cpu",
                }


            def sequence_factory(settings):
                return kernel_handler(settings, loader)
            """
        ).lstrip(),
        encoding="utf-8",
    )


def test_native_sequence_runner_loads_once_and_preserves_rng_and_cpu_outputs(
    tmp_path: Path, monkeypatch
) -> None:
    factory_path = tmp_path / "fake_sequence_factory.py"
    _write_sequence_factory(factory_path)
    _prepend_child_pythonpath(monkeypatch, tmp_path)
    audit_path = tmp_path / "sequence-audit.json"
    monkeypatch.setenv("FAKE_SEQUENCE_AUDIT", str(audit_path))

    random.seed(41)
    np.random.seed(41)
    torch.manual_seed(41)
    python_reference = random.Random(41)
    numpy_reference = np.random.RandomState(41)
    torch_reference = torch.Generator()
    torch_reference.manual_seed(41)

    constructor_expected = [
        python_reference.random(),
        float(numpy_reference.random()),
        float(torch.rand(1, generator=torch_reference).item()),
    ]
    parent_first_expected = [
        python_reference.random(),
        float(numpy_reference.random()),
        float(torch.rand(1, generator=torch_reference).item()),
    ]
    first_expected = [
        python_reference.random(),
        float(numpy_reference.random()),
        float(torch.rand(1, generator=torch_reference).item()),
    ]
    parent_second_expected = [
        python_reference.random(),
        float(numpy_reference.random()),
        float(torch.rand(1, generator=torch_reference).item()),
    ]
    second_expected = [
        python_reference.random(),
        float(numpy_reference.random()),
        float(torch.rand(1, generator=torch_reference).item()),
    ]
    output_dir = tmp_path / "sequence-output"
    runner = NativeSequenceRunner(
        backend="sequence-test",
        factory="fake_sequence_factory:sequence_factory",
        runner_config={
            "out_folder": str(output_dir),
            "temperature": 0.17,
            "seed": 77,
        },
    )
    try:
        parent_first = [
            random.random(),
            float(np.random.random()),
            float(torch.rand(1).item()),
        ]
        first = runner.design_sequences(
            name="first",
            batch_size=1,
            length=4,
        )
        parent_second = [
            random.random(),
            float(np.random.random()),
            float(torch.rand(1).item()),
        ]
        second = runner.design_sequences(
            name="second",
            batch_size=2,
            length=3,
        )

        records = json.loads(audit_path.read_text(encoding="utf-8"))
        loads = [record for record in records if record["event"] == "load"]
        inferences = [record for record in records if record["event"] == "infer"]
        assert len(loads) == 1
        assert loads[0]["pid"] == runner.pid
        assert loads[0]["pid"] != os.getpid()
        assert loads[0]["config"] == {
            "out_folder": str(output_dir),
            "temperature": 0.17,
            "seed": 77,
        }
        assert loads[0]["random"] == pytest.approx(constructor_expected)
        assert runner.output_dir == output_dir
        assert runner.device == "cpu"
        assert first["sampling_probs"].shape == (1, 4, 20)
        assert second["sampling_probs"].shape == (2, 3, 20)
        assert first["sampling_probs"].device.type == "cpu"
        assert second["sampling_probs"].device.type == "cpu"
        assert first["feature_dict"] == {
            "name": "first",
            "batch_size": 1,
            "length": 4,
        }
        assert second["feature_dict"] == {
            "name": "second",
            "batch_size": 2,
            "length": 3,
        }
        assert inferences[0]["random"] == pytest.approx(first_expected)
        assert inferences[1]["random"] == pytest.approx(second_expected)
        assert parent_first == pytest.approx(parent_first_expected)
        assert parent_second == pytest.approx(parent_second_expected)
    finally:
        runner.shutdown()


def test_native_mutator_factories_consume_mode_before_upstream_construction(
    tmp_path: Path, monkeypatch
) -> None:
    import astrolabe.mutators.native_daemons as native_daemons
    import astrolabe.mutators.wrappers.ligandmpnn_run as ligandmpnn_run

    captured = []

    class _NativeRunner:
        def __init__(self, *, backend, factory, runner_config):
            config = OmegaConf.to_container(runner_config, resolve=True)
            self.output_dir = Path(
                config.get("output_dir") or config.get("out_folder")
            )
            captured.append(
                {
                    "backend": backend,
                    "factory": factory,
                    "config": config,
                }
            )

        def shutdown(self):
            return None

    def _unexpected_upstream(*_args, **_kwargs):
        raise AssertionError("upstream runner was constructed in the parent")

    monkeypatch.setattr(native_daemons, "NativeSequenceRunner", _NativeRunner)
    monkeypatch.setattr(ligandmpnn_run, "LigandMPNN", _unexpected_upstream)
    mutator_conf = OmegaConf.create(
        {
            "mutator_terms": {
                "mpnn": {
                    "_type": "LigandMPNN",
                    "execution_mode": "daemon",
                    "temperature": 0.12,
                },
                "carbon": {
                    "_type": "carbondesign",
                    "execution_mode": "daemon",
                },
            },
            "LigandMPNN": {
                "out_folder": None,
                "temperature": 0.3,
                "checkpoint_ligand_mpnn": "/checkpoint.pt",
            },
            "CarbonDesign": {
                "repo_path": "/carbon-repo",
                "model_ckpt": "model.pt",
                "model_config": "model.json",
                "model_features": "features.json",
                "output_dir": None,
            },
        }
    )

    mutators = build_mutators(mutator_conf, tmp_path / "raw")
    try:
        assert {item["backend"] for item in captured} == {
            "ligandmpnn",
            "carbondesign",
        }
        assert {item["factory"] for item in captured} == {
            native_daemons.LIGANDMPNN_DAEMON_FACTORY,
            native_daemons.CARBONDESIGN_DAEMON_FACTORY,
        }
        mpnn = next(item for item in captured if item["backend"] == "ligandmpnn")
        carbon = next(item for item in captured if item["backend"] == "carbondesign")
        assert mpnn["config"]["temperature"] == 0.12
        assert mpnn["config"]["out_folder"] == str(tmp_path / "raw" / "LigandMPNN")
        assert "execution_mode" not in mpnn["config"]
        assert carbon["config"]["output_dir"] == str(
            tmp_path / "raw" / "CarbonDesign"
        )
        assert "execution_mode" not in carbon["config"]
        assert set(mutators) == {"mpnn", "carbon"}
    finally:
        for mutator in mutators.values():
            mutator.shutdown()


def _write_fake_carbondesign(root: Path) -> Path:
    package = root / "carbondesign"
    model_dir = package / "model"
    model_dir.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (model_dir / "__init__.py").write_text("", encoding="utf-8")
    (model_dir / "carbondesign.py").write_text(
        dedent(
            """
            import json
            import os
            from pathlib import Path

            import torch


            class CarbonDesign(torch.nn.Module):
                def __init__(self, config):
                    super().__init__()
                    audit = Path(os.environ["FAKE_CARBON_AUDIT"])
                    records = (
                        json.loads(audit.read_text(encoding="utf-8"))
                        if audit.exists()
                        else []
                    )
                    records.append(
                        {
                            "event": "load",
                            "pid": os.getpid(),
                            "config_has_embeddings": (
                                "embeddings_and_seqformer" in config
                            ),
                        }
                    )
                    audit.write_text(json.dumps(records), encoding="utf-8")
                    self.anchor = torch.nn.Parameter(torch.zeros(1))

                def forward(self, *, batch, compute_loss):
                    batch_size, length = batch["seq"].shape
                    logits = torch.arange(
                        batch_size * length * 20,
                        dtype=torch.float32,
                    ).reshape(batch_size, length, 20)
                    return {"heads": {"seqhead": {"logits": logits}}}
            """
        ).lstrip(),
        encoding="utf-8",
    )
    (package / "testloader.py").write_text(
        dedent(
            """
            import torch


            class dataset_test:
                @staticmethod
                def load(*, data_dir, name_idx, **_kwargs):
                    size = len(name_idx)
                    return [
                        {
                            "seq": torch.ones((size, 5), dtype=torch.long),
                            "mask": torch.tensor(
                                [[1, 1, 1, 1, 0]] * size,
                                dtype=torch.float32,
                            ),
                            "name": list(name_idx),
                        }
                    ]
            """
        ).lstrip(),
        encoding="utf-8",
    )
    return root


def test_carbondesign_daemon_factory_keeps_one_model_and_output_dir(
    tmp_path: Path, monkeypatch
) -> None:
    fake_root = _write_fake_carbondesign(tmp_path / "fake-carbon")
    _prepend_child_pythonpath(monkeypatch, fake_root)
    audit_path = tmp_path / "carbon-audit.json"
    monkeypatch.setenv("FAKE_CARBON_AUDIT", str(audit_path))
    repo = tmp_path / "carbon-repo"
    repo.mkdir()
    (repo / "model.json").write_text(
        json.dumps(
            {
                "model": {
                    "embeddings_and_seqformer": {
                        "lm": {"model_path": "lm"}
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (repo / "features.json").write_text("[]\n", encoding="utf-8")
    torch.save({"model": {}}, repo / "model.pt")
    output_dir = tmp_path / "carbon-output"
    config = {
        "repo_path": str(repo),
        "model_ckpt": "model.pt",
        "model_config": "model.json",
        "model_features": "features.json",
        "output_dir": str(output_dir),
        "device": "cpu",
        "batch_size": 2,
        "temperature": 0.8,
        "max_seq_len": 10,
    }
    runner = NativeSequenceRunner(
        backend="carbondesign",
        factory="astrolabe.mutators.native_daemons:carbondesign_daemon_factory",
        runner_config=config,
    )
    try:
        first = runner.design_sequences(
            data_dir=str(tmp_path),
            name_idx=["first"],
        )
        second = runner.design_sequences(
            data_dir=str(tmp_path),
            name_idx=["second-a", "second-b"],
        )
        records = json.loads(audit_path.read_text(encoding="utf-8"))
        loads = [record for record in records if record["event"] == "load"]

        assert len(loads) == 1
        assert loads[0]["pid"] == runner.pid
        assert loads[0]["pid"] != os.getpid()
        assert loads[0]["config_has_embeddings"] is True
        assert runner.output_dir == output_dir
        assert runner.device == "cpu"
        assert first["first"]["sampling_probs"].shape == (1, 4, 20)
        assert second["second-a"]["sampling_probs"].shape == (1, 4, 20)
        assert second["second-b"]["sampling_probs"].shape == (1, 4, 20)
        assert first["first"]["sampling_probs"].device.type == "cpu"
    finally:
        runner.shutdown()


def _write_fake_ligandmpnn_site(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "sitecustomize.py").write_text(
        dedent(
            """
            import json
            import os
            import random
            import sys
            import types
            from pathlib import Path

            import torch
            from omegaconf import OmegaConf


            module = types.ModuleType(
                "astrolabe.mutators.wrappers.ligandmpnn_run"
            )


            def _append(payload):
                path = Path(os.environ["FAKE_LIGAND_AUDIT"])
                records = (
                    json.loads(path.read_text(encoding="utf-8"))
                    if path.exists()
                    else []
                )
                records.append(payload)
                path.write_text(json.dumps(records), encoding="utf-8")


            class LigandMPNN:
                def __init__(self, args):
                    self.config = OmegaConf.to_container(args, resolve=True)
                    self.base_folder = self.config["out_folder"]
                    self.device = torch.device("cpu")
                    _append(
                        {
                            "event": "load",
                            "pid": os.getpid(),
                            "config": self.config,
                        }
                    )

                def design_sequences(self, **features):
                    length = int(features.get("length", 3))
                    batch_size = int(features.get("batch_size", 1))
                    _append(
                        {
                            "event": "infer",
                            "pid": os.getpid(),
                            "features": features,
                            "python_random": random.random(),
                        }
                    )
                    probs = torch.softmax(
                        torch.arange(
                            batch_size * length * 20,
                            dtype=torch.float32,
                        ).reshape(batch_size, length, 20),
                        dim=-1,
                    )
                    if features.get("pdb_path_multi"):
                        return {
                            "multi_outputs": {
                                "first": {"sampling_probs": probs},
                                "second": {"sampling_probs": probs},
                            }
                        }
                    return {"sampling_probs": probs}


            module.LigandMPNN = LigandMPNN
            sys.modules[module.__name__] = module
            """
        ).lstrip(),
        encoding="utf-8",
    )
    return root


def test_ligandmpnn_daemon_factory_loads_once_and_preserves_joint_shapes(
    tmp_path: Path, monkeypatch
) -> None:
    fake_root = _write_fake_ligandmpnn_site(tmp_path / "fake-ligand")
    _prepend_child_pythonpath(monkeypatch, fake_root)
    audit_path = tmp_path / "ligand-audit.json"
    monkeypatch.setenv("FAKE_LIGAND_AUDIT", str(audit_path))
    output_dir = tmp_path / "ligand-output"
    config = {
        "out_folder": str(output_dir),
        "model_type": "ligand_mpnn",
        "checkpoint_ligand_mpnn": "/checkpoint.pt",
        "temperature": 0.21,
        "seed": 91,
        "number_of_batches": 2,
        "batch_size": 1,
    }
    runner = NativeSequenceRunner(
        backend="ligandmpnn",
        factory="astrolabe.mutators.native_daemons:ligandmpnn_daemon_factory",
        runner_config=config,
    )
    try:
        first = runner.design_sequences(
            pdb_path="/input.pdb",
            fixed_residues="A1",
            length=4,
        )
        second = runner.design_sequences(
            pdb_path_multi="/multi.json",
            fixed_residues_multi="/fixed.json",
            batch_size=2,
            length=3,
        )
        records = json.loads(audit_path.read_text(encoding="utf-8"))
        loads = [record for record in records if record["event"] == "load"]
        inferences = [record for record in records if record["event"] == "infer"]

        assert len(loads) == 1
        assert loads[0]["pid"] == runner.pid
        assert loads[0]["pid"] != os.getpid()
        assert loads[0]["config"] == config
        assert runner.output_dir == output_dir
        assert runner.device == "cpu"
        assert first["sampling_probs"].shape == (1, 4, 20)
        assert first["sampling_probs"].device.type == "cpu"
        assert set(second["multi_outputs"]) == {"first", "second"}
        assert second["multi_outputs"]["first"]["sampling_probs"].shape == (
            2,
            3,
            20,
        )
        assert inferences[0]["features"]["fixed_residues"] == "A1"
        assert inferences[1]["features"]["pdb_path_multi"] == "/multi.json"
        assert inferences[1]["features"]["fixed_residues_multi"] == "/fixed.json"
    finally:
        runner.shutdown()


def _write_fake_esm2_package(root: Path) -> Path:
    package = root / "esm2"
    inverse = package / "inverse_folding"
    inverse.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "from . import data, pretrained\n",
        encoding="utf-8",
    )
    (inverse / "__init__.py").write_text("", encoding="utf-8")
    (package / "data.py").write_text(
        dedent(
            """
            STANDARD = (
                "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
                "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",
            )
            PREPEND = ("<null_0>", "<pad>", "<eos>", "<unk>")
            APPEND = ("<cls>", "<mask>", "<sep>")


            class Alphabet:
                def __init__(
                    self,
                    standard_toks,
                    prepend_toks=PREPEND,
                    append_toks=APPEND,
                    prepend_bos=True,
                    append_eos=False,
                    use_msa=False,
                ):
                    self.standard_toks = tuple(standard_toks)
                    self.prepend_toks = tuple(prepend_toks)
                    self.append_toks = tuple(append_toks)
                    self.prepend_bos = prepend_bos
                    self.append_eos = append_eos
                    self.use_msa = use_msa
                    self.all_toks = (
                        self.prepend_toks
                        + self.standard_toks
                        + self.append_toks
                    )
            """
        ).lstrip(),
        encoding="utf-8",
    )
    (package / "pretrained.py").write_text(
        dedent(
            """
            import json
            import os
            from pathlib import Path
            from types import SimpleNamespace

            import torch

            from .data import Alphabet, STANDARD


            def _append(payload):
                path = Path(os.environ["FAKE_ESM_AUDIT"])
                records = (
                    json.loads(path.read_text(encoding="utf-8"))
                    if path.exists()
                    else []
                )
                records.append(payload)
                path.write_text(json.dumps(records), encoding="utf-8")


            class FakeModel:
                def __init__(self, kind):
                    self.kind = kind

                def eval(self):
                    return self

                def cuda(self):
                    return self

                def to(self, device):
                    return self

                def parameters(self):
                    yield SimpleNamespace(device=torch.device("cpu"))

                def __call__(self, batch_tokens):
                    batch_size, length = batch_tokens.shape
                    _append(
                        {
                            "event": "infer",
                            "kind": self.kind,
                            "pid": os.getpid(),
                            "shape": [batch_size, length],
                            "device": str(batch_tokens.device),
                        }
                    )
                    return {
                        "logits": torch.rand(
                            batch_size,
                            length,
                            25,
                        )
                    }


            def _load(kind):
                _append({"event": "load", "kind": kind, "pid": os.getpid()})
                return FakeModel(kind), Alphabet(STANDARD)


            def esm2_t33_650M_UR50D():
                return _load("esm2")


            def esm_if1_gvp4_t16_142M_UR50():
                return _load("esm_if1")
            """
        ).lstrip(),
        encoding="utf-8",
    )
    (inverse / "util.py").write_text(
        dedent(
            """
            def get_atom_coords_residuewise(*_args, **_kwargs):
                raise AssertionError("not used by daemon metadata test")


            class CoordBatchConverter:
                def __init__(self, *_args, **_kwargs):
                    pass
            """
        ).lstrip(),
        encoding="utf-8",
    )
    return root


def test_esm2_daemon_reuses_child_model_and_reconstructs_parent_alphabet(
    tmp_path: Path, monkeypatch
) -> None:
    fake_root = _write_fake_esm2_package(tmp_path / "fake-esm2")
    _prepend_child_pythonpath(monkeypatch, fake_root)
    audit_path = tmp_path / "esm2-audit.json"
    monkeypatch.setenv("FAKE_ESM_AUDIT", str(audit_path))

    from astrolabe.mutators.esm2 import ESM2Mutator

    mutator = ESM2Mutator(execution_mode="daemon")
    try:
        first = mutator.compute_mutation(
            {
                "AB": {
                    "batch_tokens": torch.tensor(
                        [[1, 2, 3, 4, 5]],
                        dtype=torch.long,
                    )
                }
            }
        )
        second = mutator.compute_mutation(
            {
                "C": {
                    "batch_tokens": torch.tensor(
                        [[1, 2, 3, 4]],
                        dtype=torch.long,
                    )
                }
            }
        )
        records = json.loads(audit_path.read_text(encoding="utf-8"))
        loads = [
            record
            for record in records
            if record["event"] == "load" and record["kind"] == "esm2"
        ]
        inferences = [
            record
            for record in records
            if record["event"] == "infer" and record["kind"] == "esm2"
        ]

        assert len(loads) == 1
        assert loads[0]["pid"] == mutator.runner.pid
        assert loads[0]["pid"] != os.getpid()
        assert mutator.alphabet.standard_toks == [
            "A",
            "R",
            "N",
            "D",
            "C",
            "Q",
            "E",
            "G",
            "H",
            "I",
            "L",
            "K",
            "M",
            "F",
            "P",
            "S",
            "T",
            "W",
            "Y",
            "V",
        ]
        assert mutator.alphabet.prepend_toks == [
            "<null_0>",
            "<pad>",
            "<eos>",
            "<unk>",
        ]
        assert mutator.alphabet.append_toks == ["<cls>", "<mask>", "<sep>"]
        assert first["AB"]["sampling_probs"].shape == (1, 3, 20)
        assert second["C"]["sampling_probs"].shape == (1, 2, 20)
        assert first["AB"]["sampling_probs"].device.type == "cpu"
        assert first["AB"]["feature_dict"]["batch_tokens"].device.type == "cpu"
        assert first["AB"]["name"] == "AB"
        assert second["C"]["name"] == "C"
        assert [record["shape"] for record in inferences] == [[1, 5], [1, 4]]
        assert [record["device"] for record in inferences] == ["cpu", "cpu"]
    finally:
        mutator.shutdown()


def test_esm_if1_daemon_reuses_child_model_and_exact_alphabet(
    tmp_path: Path, monkeypatch
) -> None:
    fake_root = _write_fake_esm2_package(tmp_path / "fake-esm-if1")
    _prepend_child_pythonpath(monkeypatch, fake_root)
    audit_path = tmp_path / "esm-if1-audit.json"
    monkeypatch.setenv("FAKE_ESM_AUDIT", str(audit_path))

    from astrolabe.mutators.esm_if1 import ESMIF1Mutator

    mutator = ESMIF1Mutator(execution_mode="daemon")
    try:
        assert mutator.compute_mutation({}) == {}
        assert mutator.compute_mutation({}) == {}
        records = json.loads(audit_path.read_text(encoding="utf-8"))
        loads = [
            record
            for record in records
            if record["event"] == "load" and record["kind"] == "esm_if1"
        ]
        assert len(loads) == 1
        assert loads[0]["pid"] == mutator.runner.pid
        assert loads[0]["pid"] != os.getpid()
        assert mutator.alphabet.all_toks[4:24] == [
            "A",
            "R",
            "N",
            "D",
            "C",
            "Q",
            "E",
            "G",
            "H",
            "I",
            "L",
            "K",
            "M",
            "F",
            "P",
            "S",
            "T",
            "W",
            "Y",
            "V",
        ]
    finally:
        mutator.shutdown()


def _write_fake_esm3_package(root: Path) -> Path:
    package = root / "esm3"
    models = package / "models"
    sdk = package / "sdk"
    utils = package / "utils" / "constants"
    models.mkdir(parents=True)
    sdk.mkdir()
    utils.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (models / "__init__.py").write_text("", encoding="utf-8")
    (sdk / "__init__.py").write_text("", encoding="utf-8")
    (package / "utils" / "__init__.py").write_text("", encoding="utf-8")
    (package / "utils" / "constants" / "__init__.py").write_text(
        "", encoding="utf-8"
    )
    (models / "esm3.py").write_text(
        dedent(
            """
            import json
            import os
            from pathlib import Path
            from types import SimpleNamespace

            import torch


            def _append(payload):
                path = Path(os.environ["FAKE_ESM3_AUDIT"])
                records = (
                    json.loads(path.read_text(encoding="utf-8"))
                    if path.exists()
                    else []
                )
                records.append(payload)
                path.write_text(json.dumps(records), encoding="utf-8")


            class _Runner:
                def eval(self):
                    return self

                def to(self, device):
                    self.device = device
                    return self

                def encode(self, protein):
                    coordinates = getattr(protein, "coordinates", None)
                    _append(
                        {
                            "event": "encode",
                            "pid": os.getpid(),
                            "sequence": protein.sequence,
                            "coordinates_device": (
                                str(coordinates.device)
                                if hasattr(coordinates, "device")
                                else None
                            ),
                        }
                    )
                    return {"length": len(protein.sequence)}

                def logits(self, encoded, config):
                    length = encoded["length"]
                    logits = torch.rand(1, length + 2, 200)
                    return SimpleNamespace(
                        logits=SimpleNamespace(sequence=logits)
                    )


            class ESM3:
                @classmethod
                def from_pretrained(cls, model_name, device=None):
                    _append(
                        {
                            "event": "load",
                            "pid": os.getpid(),
                            "model_name": model_name,
                            "device": str(device),
                        }
                    )
                    return _Runner()
            """
        ).lstrip(),
        encoding="utf-8",
    )
    (sdk / "api.py").write_text(
        dedent(
            """
            from dataclasses import dataclass
            from typing import Any


            @dataclass
            class ESMProtein:
                sequence: str
                coordinates: Any = None


            @dataclass
            class LogitsConfig:
                sequence: bool
            """
        ).lstrip(),
        encoding="utf-8",
    )
    (package / "tokenization.py").write_text(
        dedent(
            """
            import json
            import os
            from pathlib import Path
            from types import SimpleNamespace


            class _Tokenizer:
                def convert_tokens_to_ids(self, token):
                    return ord(token[0])


            def get_esm3_model_tokenizers(model_name):
                path = Path(os.environ["FAKE_ESM3_AUDIT"])
                records = (
                    json.loads(path.read_text(encoding="utf-8"))
                    if path.exists()
                    else []
                )
                records.append(
                    {
                        "event": "tokenizer",
                        "pid": os.getpid(),
                        "model_name": model_name,
                    }
                )
                path.write_text(json.dumps(records), encoding="utf-8")
                return SimpleNamespace(sequence=_Tokenizer())
            """
        ).lstrip(),
        encoding="utf-8",
    )
    (package / "pretrained.py").write_text(
        "data_root = lambda model: None\n",
        encoding="utf-8",
    )
    (package / "utils" / "constants" / "esm3.py").write_text(
        "data_root = lambda model: None\n",
        encoding="utf-8",
    )
    return root


def test_esm3_daemon_keeps_tokenizer_in_parent_and_serializes_prompt(
    tmp_path: Path, monkeypatch
) -> None:
    fake_root = _write_fake_esm3_package(tmp_path / "fake-esm3")
    monkeypatch.syspath_prepend(str(fake_root))
    _prepend_child_pythonpath(monkeypatch, fake_root)
    for module_name in (
        "esm3",
        "esm3.models",
        "esm3.models.esm3",
        "esm3.sdk",
        "esm3.sdk.api",
        "esm3.tokenization",
        "esm3.pretrained",
        "esm3.utils",
        "esm3.utils.constants",
        "esm3.utils.constants.esm3",
        "astrolabe.mutators.esm3",
    ):
        monkeypatch.delitem(sys.modules, module_name, raising=False)
    audit_path = tmp_path / "esm3-audit.json"
    monkeypatch.setenv("FAKE_ESM3_AUDIT", str(audit_path))

    from astrolabe.mutators.esm3 import ESM3Mutator

    mutator = ESM3Mutator(
        model_name="esm3-test",
        device="cpu",
        temperature=0.25,
        execution_mode="daemon",
    )
    try:
        first_prompt = mutator._build_protein_prompt("ACD")
        first_prompt.coordinates = torch.tensor(
            [[0.0, 1.0, 2.0]] * 3
        )
        second_prompt = mutator._build_protein_prompt("EFGH")
        first = mutator.compute_mutation(
            {"first": {"protein_prompt": first_prompt}}
        )
        second = mutator.compute_mutation(
            {"second": {"protein_prompt": second_prompt}}
        )
        records = json.loads(audit_path.read_text(encoding="utf-8"))
        tokenizers = [
            record for record in records if record["event"] == "tokenizer"
        ]
        loads = [record for record in records if record["event"] == "load"]
        encodes = [record for record in records if record["event"] == "encode"]

        assert mutator._aa_token_ids is None
        parent_tokenizers = [
            record for record in tokenizers if record["pid"] == os.getpid()
        ]
        child_tokenizers = [
            record for record in tokenizers if record["pid"] == mutator.runner.pid
        ]
        assert len(parent_tokenizers) == 1
        assert len(child_tokenizers) == 1
        assert parent_tokenizers[0]["model_name"] == "esm3-test"
        assert len(loads) == 1
        assert loads[0]["pid"] == mutator.runner.pid
        assert loads[0]["pid"] != os.getpid()
        assert loads[0]["model_name"] == "esm3-test"
        assert loads[0]["device"] == "cpu"
        assert [record["sequence"] for record in encodes] == ["ACD", "EFGH"]
        assert encodes[0]["coordinates_device"] == "cpu"
        assert first["first"]["sampling_probs"].shape == (1, 3, 20)
        assert second["second"]["sampling_probs"].shape == (1, 4, 20)
        assert first["first"]["sampling_probs"].device.type == "cpu"
        assert first["first"]["feature_dict"]["protein_prompt"].sequence == "ACD"
        assert second["second"]["feature_dict"]["protein_prompt"].sequence == "EFGH"
        assert first["first"]["name"] == "first"
        assert second["second"]["name"] == "second"
    finally:
        mutator.shutdown()
