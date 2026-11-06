from __future__ import annotations

import os
from pathlib import Path

import pytest

import astrolabe.structure_processors.ligand_interface_placer as placer_module
import astrolabe.structure_processors.wrappers.placer_daemon as placer_daemon_module
from astrolabe.structure_processors.ligand_interface_placer import (
    ChemNetPlacementProcessor,
)
from astrolabe.structure_processors.wrappers.placer_daemon import PlacerClient


def test_placer_daemon_loads_once_and_uses_fresh_inputs(tmp_path: Path) -> None:
    load_marker = tmp_path / "loads.log"
    input_marker = tmp_path / "inputs.log"
    placer_module_path = tmp_path / "PLACER.py"
    placer_module_path.write_text(
        f"""
from pathlib import Path
import os

LOAD_MARKER = Path({str(load_marker)!r})
INPUT_MARKER = Path({str(input_marker)!r})

def _append(path, value):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(value + "\\n")

class PLACERinput:
    def __init__(self):
        _append(INPUT_MARKER, "input")
        self._pdb = None
        self._cif = None
        self._name = None

    def pdb(self, value=None):
        if value is not None:
            self._pdb = value

    def cif(self, value=None):
        if value is not None:
            self._cif = value

    def name(self, value=None):
        if value is not None:
            self._name = value

class PLACER:
    def __init__(self, weights_path=None):
        _append(LOAD_MARKER, "load")
        self.weights_path = weights_path

    def run(self, input_object, nsamples):
        assert nsamples == 3
        assert input_object._pdb
        return {{
            2: {{"prmsd": 0.3, "model": "model-two"}},
            0: {{"prmsd": 0.1, "model": "model-zero"}},
            1: {{"prmsd": 0.2, "model": "model-one"}},
        }}
""",
        encoding="utf-8",
    )
    first_input = tmp_path / "first.pdb"
    second_input = tmp_path / "second.pdb"
    first_input.write_text("first\n", encoding="utf-8")
    second_input.write_text("second\n", encoding="utf-8")
    env = {
        "PYTHONPATH": os.pathsep.join(
            [str(tmp_path), os.environ.get("PYTHONPATH", "")]
        )
    }
    client = PlacerClient(
        {"weights_path": None},
        mode="daemon",
        env=env,
    )

    first = client.predict(
        {"input_file": str(first_input), "num_samples": 3}
    )
    second = client.predict(
        {"input_file": str(second_input), "num_samples": 3}
    )
    pid = client.pid
    client.shutdown()

    assert first == second
    assert [item["key"] for item in first] == [2, 0, 1]
    assert client.backend == client.metadata["backend"] == "chemnet_placer"
    assert client._native.owner.spec.backend == "chemnet_placer"
    assert client.metadata["implementation"] == "PLACER"
    assert client.metadata["torch_device"] == "auto"
    assert client.metadata["dependency_management"] == "manual"
    assert client.metadata["install_automatically"] is False
    assert client.metadata["audited_revision"] == (
        "7dc5563300e7143582d85636bdd413d4c7688e9a"
    )
    assert client.metadata["module_origin"].endswith("PLACER.py")
    assert client.metadata["python_bin"]
    assert load_marker.read_text(encoding="utf-8").splitlines() == ["load"]
    assert input_marker.read_text(encoding="utf-8").splitlines() == [
        "input",
        "input",
    ]
    assert pid
    assert not Path(f"/proc/{pid}").exists()


def test_placer_processor_keeps_prmsd_selection_local(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_file = tmp_path / "input.pdb"
    input_file.write_text("input\n", encoding="utf-8")
    output_file = tmp_path / "output.pdb"

    class _Client:
        metadata = {}
        shutdown_calls = 0

        def predict(self, payload):
            assert payload == {"input_file": str(input_file), "num_samples": 3}
            return [
                {"key": 2, "prmsd": 0.8, "model": "higher-model"},
                {"key": 0, "prmsd": 0.1, "model": "selected-model"},
                {"key": 1, "prmsd": 0.4, "model": "middle-model"},
            ]

        def shutdown(self):
            self.shutdown_calls += 1

    class _Chain:
        def __init__(self, chain_id):
            self.id = chain_id

        def copy(self):
            return _Chain(self.id)

    class _Model:
        def __init__(self):
            self.chains = {"C": _Chain("C")}

        def __iter__(self):
            return iter(self.chains.values())

        def __contains__(self, chain_id):
            return chain_id in self.chains

        def detach_child(self, chain_id):
            del self.chains[chain_id]

        def add(self, chain):
            self.chains[chain.id] = chain

    class _Structure:
        def __iter__(self):
            return iter([_Model()])

    class _Parser:
        def __init__(self, **_kwargs):
            pass

        def get_structure(self, name, _path):
            return _Structure()

    class _IO:
        def set_structure(self, structure):
            self.structure = structure

        def save(self, path):
            Path(path).write_text("merged\n", encoding="utf-8")

    removed = {}

    def _remove(path):
        removed["contents"] = Path(path).read_text(encoding="utf-8")

    monkeypatch.setattr("Bio.PDB.PDBParser", _Parser)
    monkeypatch.setattr("Bio.PDB.PDBIO", _IO)
    monkeypatch.setattr(placer_module.os, "remove", _remove)
    processor = object.__new__(ChemNetPlacementProcessor)
    processor.num_samples = 3
    processor.placer_client = _Client()
    result = processor.resample_chain(str(input_file), "C", str(output_file))
    processor.shutdown()

    assert result == str(output_file)
    assert removed["contents"] == "selected-model"
    assert output_file.read_text(encoding="utf-8") == "merged\n"
    assert processor.placer_client.shutdown_calls == 1


def test_placer_daemon_threads_auto_and_exact_torch_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = []

    def _kernel(settings, loader, *, torch_device=None):
        captured.append(torch_device)
        return object()

    monkeypatch.setattr(placer_daemon_module, "kernel_handler", _kernel)

    placer_daemon_module.create_handler({})
    placer_daemon_module.create_handler({"torch_device": "cuda:1"})

    assert captured == ["auto", "cuda:1"]
