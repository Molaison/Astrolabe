"""Critical topology/writer checks using upstream code, not trained inference.

Set NUMONOMER_REPO_PATH to the inspected source checkout. No weights are used.
"""
import importlib
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from astrolabe.evaluators.wrappers.numonomer import NUMonomerBackend, _imports


@pytest.fixture
def backend(tmp_path):
    source = os.environ.get("NUMONOMER_REPO_PATH")
    if not source:
        pytest.skip("Set NUMONOMER_REPO_PATH to the pinned upstream checkout")
    with _imports(Path(source)):
        modules = importlib.import_module("modules")
        dataset = importlib.import_module("dataset")
        writer = importlib.import_module("utils.save_struc")
        model_module = importlib.import_module("model")
        compute_plddt = importlib.import_module("utils.seq_utils").compute_plddt
    backend = object.__new__(NUMonomerBackend)
    backend.settings = {"multichain_mode": "experimental"}
    backend.torch, backend.np = torch, np
    backend.device = torch.device("cpu")
    backend.ftype, backend.topology = "cif", "linear"
    backend.num_iter, backend.clamp_plddt = 1, 512
    backend.save_last = False
    backend.initial_seed = 42
    backend.initial_cpu_rng = torch.get_rng_state().clone()
    backend.dataset_class, backend.save_struc = dataset.MonomerDataset, writer
    backend.model_module, backend.compute_plddt = model_module, compute_plddt
    backend.model = torch.nn.Module()
    backend.model.input_embedder = modules.InputEmbedder(8, nums_aa=6, atom_channel=16)
    backend.weights = tmp_path / "unused.pt"
    return backend


def request(tmp_path):
    return {"chains": [{"chain_id": "z", "sequence": "AC"}, {"chain_id": "7", "sequence": "ACG"}],
            "seed": 42, "output_dir": str(tmp_path)}


def test_prepare_requires_opt_in_and_keeps_authoritative_layout(backend, tmp_path):
    backend.settings.clear()
    with pytest.raises(ValueError, match="experimental"):
        backend.prepare(request(tmp_path))
    backend.settings["multichain_mode"] = "experimental"
    prepared = backend.prepare(request(tmp_path))
    assert [c["chain_id"] for c in prepared["chains"]] == ["z", "7"]
    assert [(c["start"], c["stop"]) for c in prepared["chains"]] == [(0, 2), (2, 5)]
    assert prepared["features"]["label_seq"].shape == (1, 5)
    assert (tmp_path / "input.fasta").read_text().count(">") == 2
    with pytest.raises(ValueError, match="linear chains only"):
        backend.prepare({**request(tmp_path), "topology": "circular"})
    with pytest.raises(ValueError, match="unique"):
        backend.prepare({**request(tmp_path), "chains": [request(tmp_path)["chains"][0]] * 2})


@pytest.mark.parametrize("fail", [False, True])
def test_native_positional_hook_neutralizes_only_offchain_and_cleans_up(backend, tmp_path, fail):
    prepared = backend.prepare(request(tmp_path))
    embedder = backend.model.input_embedder
    native_indices, native_table, native_sequence = embedder(prepared["features"])
    parameters = {key: value.clone() for key, value in embedder.state_dict().items()}
    try:
        with backend._multichain_features(prepared["chains"]):
            indices, table, sequence = embedder(prepared["features"])
            torch.testing.assert_close(indices[:2, :2], native_indices[:2, :2])
            torch.testing.assert_close(indices[2:, 2:], native_indices[2:, 2:])
            assert torch.all(indices[:2, 2:] == 65) and torch.all(indices[2:, :2] == 65)
            torch.testing.assert_close(sequence, native_sequence)
            torch.testing.assert_close(table[:65], native_table)
            assert table.shape == (66, 8) and torch.count_nonzero(table[65]) == 0
            if fail:
                raise RuntimeError("deliberate inference failure")
    except RuntimeError as error:
        assert fail and str(error) == "deliberate inference failure"
    assert not embedder._forward_hooks
    for key, value in embedder.state_dict().items():
        torch.testing.assert_close(value, parameters[key])
    for original, restored in zip((native_indices, native_table, native_sequence), embedder(prepared["features"])):
        torch.testing.assert_close(original, restored)


def test_positional_features_are_chain_permutation_equivariant(backend, tmp_path):
    prepared = backend.prepare(request(tmp_path))
    embedder = backend.model.input_embedder
    with backend._multichain_features(prepared["chains"]):
        indices, table, _ = embedder(prepared["features"])
        pair = table[indices].clone()
    swapped = backend.prepare({**request(tmp_path), "chains": list(reversed(request(tmp_path)["chains"]))})
    order = torch.tensor([2, 3, 4, 0, 1])
    with backend._multichain_features(swapped["chains"]):
        indices, table, _ = embedder(swapped["features"])
        torch.testing.assert_close(table[indices], pair[order][:, order])


def test_real_untrained_joint_decoder_and_native_single_chain_path(backend, tmp_path):
    # Reduced channels/blocks exercise actual upstream IPA + confidence code.
    # Random weights are execution evidence, never pretrained accuracy evidence.
    backend.model = backend.model_module.NUMonomer(
        blocks_structure_decoder=1, blocks_confidence=1,
        atom_channel=16, atom_nhead=4, pair_channel=8, pair_nhead=2,
    ).eval()
    backend.native_log_memory = backend.model_module.log_memory
    backend.place("cpu")
    prepared = backend.prepare(request(tmp_path))
    output = backend.execute(prepared)
    assert output["coords"].shape == (5, 28, 3)
    assert output["confidence"].shape == (5,)
    assert torch.isfinite(output["coords"]).all()
    assert torch.isfinite(output["confidence"]).all()
    assert output["coords"].device.type == "cpu"
    assert backend.model.input_embedder.embd_pos.weight.shape == (65, 8)
    assert not backend.model.input_embedder._forward_hooks

    single = backend.prepare({
        "sequence": "ACGUA", "chain_id": "z", "seed": 42,
        "output_dir": str(tmp_path / "single"),
    })
    wrapped = backend.execute(single)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        native = backend.model(single["features"], last_only=False)
    torch.testing.assert_close(wrapped["coords"], native["pred_coords"][0, 0].float(), rtol=0, atol=0)
    assert not backend.model.input_embedder._forward_hooks


@pytest.mark.parametrize("ftype", ["pdb", "cif"])
@pytest.mark.parametrize("confidence_length", [3, 5])
def test_native_writer_preserves_joint_coordinates_and_confidence_prefix(backend, tmp_path, ftype, confidence_length):
    from Bio.PDB import MMCIFParser, PDBParser

    prepared = backend.prepare(request(tmp_path))
    backend.ftype = ftype
    coordinates = torch.arange(5 * 28 * 3, dtype=torch.float32).reshape(5, 28, 3) / 8
    confidence = torch.arange(confidence_length, dtype=torch.float32) + 70
    payload = {
        **prepared, "coords": coordinates, "confidence": confidence,
        "label_seq": prepared["features"]["label_seq"], "mol_type": "rna",
        "selected_output_index": 0, "iteration_mean_plddt": [float(confidence.mean())],
    }
    result = backend.finalize(payload)
    parser = PDBParser(QUIET=True) if ftype == "pdb" else MMCIFParser(QUIET=True)
    model = parser.get_structure("joint", result["structure_path"])[0]
    assert [chain.id for chain in model] == ["z", "7"]
    assert result["metadata"]["accuracy_validation"] == "unvalidated"
    assert result["plddt"] == (confidence.tolist() if confidence_length == 5 else None)
    for chain, start, length in ((model["z"], 0, 2), (model["7"], 2, 3)):
        assert [residue.id[1] for residue in chain] == list(range(1, length + 1))
        for local, residue in enumerate(chain):
            index = start + local
            for atom in residue:
                atom_index = backend.save_struc.resc.atom_order[atom.name]
                np.testing.assert_array_equal(atom.coord, coordinates[index, atom_index])
                assert atom.bfactor == (float(confidence[index]) if index < confidence_length else 0)
