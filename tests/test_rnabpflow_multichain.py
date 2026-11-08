"""CPU contract checks against pinned upstream embedders/writer, not inference.

Set RNABPFLOW_REPO_PATH to a read-only upstream checkout outside this workspace.
No checkpoint, interpolant integration, accelerator, or downloaded asset is used.
"""

import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from astrolabe.evaluators.wrappers.rnabpflow import RNAbpFlowBackend


@pytest.fixture
def upstream(monkeypatch):
    configured_path = os.environ.get("RNABPFLOW_REPO_PATH")
    if not configured_path:
        pytest.skip("RNABPFLOW_REPO_PATH is not set; supply the pinned upstream checkout")
    upstream_path = Path(configured_path).expanduser()
    if not (upstream_path / "src/models/node_embedder.py").is_file():
        pytest.skip("Set RNABPFLOW_REPO_PATH to the pinned upstream checkout")
    pytest.importorskip("tree")
    pytest.importorskip("Bio")
    # Isolate upstream's generic 'src' namespace. The embedding utility imports
    # data.utils but never uses it for index/time embeddings; avoid importing its
    # unrelated torch_scatter/all-atom dependencies in these CPU unit checks.
    for name in ("src", "src.data", "src.models"):
        package = ModuleType(name)
        package.__path__ = [str(upstream_path / name.replace(".", "/"))]
        monkeypatch.setitem(sys.modules, name, package)
    monkeypatch.setitem(sys.modules, "src.data.utils", ModuleType("src.data.utils"))

    def load(name):
        spec = importlib.util.spec_from_file_location(name, upstream_path / (name.replace(".", "/") + ".py"))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        parent, _, child = name.rpartition(".")
        setattr(sys.modules[parent], child, module)
        return module

    constants = load("src.data.base_constants")
    transforms = load("src.data.data_transform")
    writer = load("src.data.nucleicacid")
    utils = load("src.models.utils")
    node = load("src.models.node_embedder")
    edge = load("src.models.edge_embedder")
    return SimpleNamespace(
        constants=constants, transforms=transforms, writer=writer, utils=utils,
        NodeEmbedder=node.NodeEmbedder, EdgeEmbedder=edge.EdgeEmbedder,
    )


@pytest.fixture
def backend(upstream, tmp_path):
    backend = object.__new__(RNAbpFlowBackend)
    backend.settings = {"multichain_mode": "experimental"}
    backend.sample_count = 2
    backend.timesteps = 3
    backend.device = torch.device("cpu")
    backend.weights = tmp_path / "unused-checkpoint.ckpt"
    backend.transforms = upstream.transforms
    backend.constants = upstream.constants
    backend.nucleicacid = upstream.writer
    backend.model_utils = upstream.utils
    return backend


def request(tmp_path, **overrides):
    return {
        "chains": [{"chain_id": "z", "sequence": "AC"}, {"chain_id": "7", "sequence": "GUA"}],
        "conditioning_dot_bracket": "((+)).",
        "output_dir": str(tmp_path), "seed": 21, **overrides,
    }


def test_opt_in_preserves_duplicate_chains_and_cpu_layout(backend, tmp_path):
    item = request(tmp_path, chains=[{"chain_id": "z", "sequence": "AC"},
                                     {"chain_id": "7", "sequence": "AC"}],
                   conditioning_dot_bracket="((+))")
    backend.settings = {}
    with pytest.raises(ValueError, match="defaults to 'reject'"):
        backend.prepare(item)
    prepared = backend.prepare({**item, "multichain_mode": "experimental"})
    assert prepared.sequence == "ACAC"
    assert [(chain.chain_id, chain.sequence, chain.start, chain.stop) for chain in prepared.chains] == [
        ("z", "AC", 0, 2), ("7", "AC", 2, 4),
    ]
    assert prepared.conditioning["row_order"] == ["z", "7"]
    assert prepared.conditioning["interchain_edges"] == [["z", "7"]]
    assert isinstance(prepared.ss, np.ndarray)
    assert all(not isinstance(value, torch.Tensor) or value.device.type == "cpu"
               for value in prepared.mapfeat.values())


@pytest.mark.parametrize("overrides,match", [
    ({"chains": [{"chain_id": "A", "sequence": "AC"}, {"chain_id": "A", "sequence": "GU"}]}, "unique"),
    ({"chains": [{"chain_id": "AA", "sequence": "AC"}]}, "single-character"),
    ({"chains": [{"chain_id": "+", "sequence": "AC"}]}, "PDB-compatible"),
    ({"chains": [{"chain_id": "A", "sequence": ""}]}, "nonempty RNA"),
    ({"chains": []}, "nonempty ordered"),
    ({"sequence": "AC"}, "not both"),
    ({"conditioning_dot_bracket": "((.)."}, "strand lengths"),
    ({"conditioning_dot_bracket": "(+.).."}, "strand lengths"),
    ({"conditioning_dot_bracket": "..+..."}, "connect the chain graph"),
    ({"conditioning_dot_bracket": None, "target_dot_bracket": "((+))."}, "explicit conditioning"),
])
def test_invalid_topology_and_implicit_conditioning_rejected(backend, tmp_path, overrides, match):
    with pytest.raises(ValueError, match=match):
        backend.prepare(request(tmp_path, **overrides))


def test_target_opt_in_is_not_a_multichain_conditioning_source(backend, tmp_path):
    backend.settings["use_target_dot_bracket"] = True
    with pytest.raises(ValueError, match="must be explicit"):
        backend.prepare(request(tmp_path, conditioning_dot_bracket=None, target_dot_bracket="((+))."))


def test_map_conditioning_requires_connected_chain_graph_across_all_channels(backend, tmp_path):
    chains = [{"chain_id": label, "sequence": "AC"} for label in "z7B"]
    first, second = np.zeros((6, 6), dtype=np.float32), np.zeros((6, 6), dtype=np.float32)
    first[0, 3] = 1  # chain z -> 7, but B is disconnected.
    second[5, 2] = 0.01  # B -> 7, in another map/direction; any positive value counts.
    first_path, second_path = tmp_path / "first.npy", tmp_path / "second.npy"
    np.save(first_path, first)
    np.save(second_path, second)
    item = request(tmp_path, chains=chains, conditioning_dot_bracket=None, map_paths=[str(first_path)])
    with pytest.raises(ValueError, match="connect the chain graph"):
        backend.prepare(item)
    prepared = backend.prepare({**item, "map_paths": [str(first_path), str(second_path)]})
    np.testing.assert_array_equal(prepared.ss[..., 0], first)
    np.testing.assert_array_equal(prepared.ss[..., 1], second)
    np.testing.assert_array_equal(prepared.ss[..., 2], first)
    assert prepared.conditioning["interchain_edges"] == [["z", "7"], ["7", "B"]]
    bad_path = tmp_path / "wrong-size.npy"
    np.save(bad_path, np.zeros((4, 4)))
    with pytest.raises(ValueError, match="finite L x L"):
        backend.prepare({**item, "map_paths": [str(bad_path)]})


@pytest.mark.parametrize("fail", [False, True])
def test_real_linear_inputs_joint_sampling_and_scoped_cleanup(backend, upstream, tmp_path, fail):
    model = torch.nn.Module()
    model.node_embedder = upstream.NodeEmbedder(SimpleNamespace(c_pos_emb=4, c_timestep_emb=4, c_s=6))
    model.edge_embedder = upstream.EdgeEmbedder(SimpleNamespace(c_s=6, c_p=5, feat_dim=4))
    observed, calls = {}, []

    def observe(label):
        def hook(module, args, output):
            observed[label] = args[0].detach().clone()
        return hook

    # Forward (not pre-forward) observers see precisely what each real Linear
    # consumed, after the adapter's pre-hooks, not a reimplementation of them.
    node_observer = model.node_embedder.linear.register_forward_hook(observe("node"))
    edge_observer = model.edge_embedder.linear_relpos.register_forward_hook(observe("edge"))
    prepared = backend.prepare(request(tmp_path))
    suffixes = []

    def sample(batch, length, ss, supplied_model, seed, onehot, mapfeat, timesteps):
        calls.append(seed)
        assert batch == 1 and length == 5 and supplied_model is model
        assert timesteps == backend.timesteps
        np.testing.assert_array_equal(ss.numpy()[0], prepared.ss)
        assert ss[0, 0, 3, 0] == 1  # A cross-chain pair survives joint conditioning.
        assert mapfeat["aatype"].shape == (1, 5)
        mask, time = torch.ones((batch, length)), torch.full((batch, 1), 0.25)
        suffixes.append(torch.cat((onehot, model.node_embedder.embed_t(time, mask)), dim=-1))
        node = model.node_embedder(time, mask, onehot)
        model.edge_embedder(node, None, None, mask[:, :, None] * mask[:, None, :], ss)
        if fail:
            raise RuntimeError("deliberate sampler failure")
        return torch.full((1, length, 28, 3), float(seed))

    backend.flow = SimpleNamespace(model=model, interpolant=SimpleNamespace(sample=sample))
    try:
        if fail:
            with pytest.raises(RuntimeError, match="deliberate sampler failure"):
                backend.execute(prepared)
        else:
            payload = backend.execute(prepared)
            assert calls == [21, 22]
            assert all(isinstance(value, np.ndarray) for value in payload["samples"])
            assert isinstance(payload["aatype"], np.ndarray)
            assert payload["samples"][0][0, 0, 0, 0] == 21
        positions = torch.tensor([[0., 1., 0., 1., 2.]])
        torch.testing.assert_close(observed["node"][..., :4], upstream.utils.get_index_embedding(positions, 4))
        torch.testing.assert_close(observed["node"][..., 4:], suffixes[-1])
        assert torch.count_nonzero(observed["edge"][:, :2, 2:]) == 0
        assert torch.count_nonzero(observed["edge"][:, 2:, :2]) == 0
        # Same-chain zero difference has cosine ones; it must NOT be zeroed.
        torch.testing.assert_close(observed["edge"][0, 0, 0], upstream.utils.get_index_embedding(torch.tensor(0.), 4).reshape(-1))
        torch.testing.assert_close(observed["edge"][0, 3, 4], upstream.utils.get_index_embedding(torch.tensor(-1.), 4).reshape(-1))
        assert not model.node_embedder.linear._forward_pre_hooks
        assert not model.edge_embedder.linear_relpos._forward_pre_hooks
        assert node_observer.id in model.node_embedder.linear._forward_hooks
        # Following use has native packed indexing, proving no lingering hook.
        model.node_embedder(torch.full((1, 1), 0.25), torch.ones((1, 5)), torch.ones((1, 5, 4)))
        torch.testing.assert_close(observed["node"][..., :4], upstream.utils.get_index_embedding(torch.arange(5.)[None], 4))
    finally:
        node_observer.remove()
        edge_observer.remove()


def test_actual_positional_features_follow_physical_chain_permutation(backend, upstream, tmp_path):
    original = backend.prepare(request(tmp_path))
    # Swap the unequal-length physical chains, preserving each chain's internal
    # order, coordinates, residue features, and global conditioning map.
    order = torch.tensor([2, 3, 4, 0, 1])
    map_path = tmp_path / "permuted.npy"
    np.save(map_path, original.ss[..., 0][order.numpy()][:, order.numpy()])
    permuted = backend.prepare(request(
        tmp_path,
        chains=list(reversed(request(tmp_path)["chains"])),
        conditioning_dot_bracket=None, conditioning_path=str(map_path),
    ))
    coordinates = torch.arange(15, dtype=torch.float32).reshape(1, 5, 3)
    onehot = torch.as_tensor(original.onehot, dtype=torch.float32)[None]
    np.testing.assert_array_equal(permuted.onehot, original.onehot[order.numpy()])
    model = torch.nn.Module()
    model.node_embedder = upstream.NodeEmbedder(SimpleNamespace(c_pos_emb=4, c_timestep_emb=4, c_s=6))
    model.edge_embedder = upstream.EdgeEmbedder(SimpleNamespace(c_s=6, c_p=5, feat_dim=4))
    backend.flow = SimpleNamespace(model=model)
    observed = {}

    def observe(label):
        def hook(module, args, output):
            observed[label] = args[0].detach().clone()
        return hook

    node_observer = model.node_embedder.linear.register_forward_hook(observe("node"))
    edge_observer = model.edge_embedder.linear_relpos.register_forward_hook(observe("edge"))

    def embed(prepared, positions, residue_features):
        mask = torch.ones((1, 5))
        ss = torch.as_tensor(prepared.ss, dtype=torch.float32)[None]
        with torch.no_grad(), backend._multichain_features(prepared):
            nodes = model.node_embedder(torch.full((1, 1), 0.25), mask, residue_features)
            edges = model.edge_embedder(nodes, positions, positions,
                                        mask[:, :, None] * mask[:, None, :], ss)
        return observed["node"], observed["edge"], nodes, edges

    try:
        native_node_input, native_edge_input, native_nodes, native_edges = embed(original, coordinates, onehot)
        swapped_node_input, swapped_edge_input, swapped_nodes, swapped_edges = embed(
            permuted, coordinates[:, order], onehot[:, order],
        )
        # Observe actual Linear inputs, not reconstructed expected embeddings:
        # changing chain ordinal must produce only the physical residue permutation.
        torch.testing.assert_close(swapped_node_input, native_node_input[:, order])
        torch.testing.assert_close(swapped_edge_input, native_edge_input[:, order][:, :, order])
        torch.testing.assert_close(swapped_nodes, native_nodes[:, order])
        torch.testing.assert_close(swapped_edges, native_edges[:, order][:, :, order])
        assert not model.node_embedder.linear._forward_pre_hooks
        assert not model.edge_embedder.linear_relpos._forward_pre_hooks
    finally:
        node_observer.remove()
        edge_observer.remove()


def test_incompatible_upstream_dimensions_fail_before_sampling(backend, upstream, tmp_path):
    node = upstream.NodeEmbedder(SimpleNamespace(c_pos_emb=4, c_timestep_emb=4, c_s=6))
    edge = upstream.EdgeEmbedder(SimpleNamespace(c_s=6, c_p=5, feat_dim=4))
    edge.linear_relpos = torch.nn.Linear(6, 4)
    backend.flow = SimpleNamespace(model=SimpleNamespace(node_embedder=node, edge_embedder=edge))
    with pytest.raises(ValueError, match="dimensions/layout"):
        backend.execute(backend.prepare(request(tmp_path)))
    assert not node.linear._forward_pre_hooks
    assert not edge.linear_relpos._forward_pre_hooks


def test_single_chain_execute_needs_no_hook_modules(backend, tmp_path):
    prepared = backend.prepare({"sequence": "AC", "chain_id": "z", "output_dir": str(tmp_path),
                                "conditioning_dot_bracket": "()"})
    model = object()  # No node/edge attributes: the single-chain branch cannot inspect them.
    calls = []

    def sample(batch, length, ss, supplied_model, seed, onehot, mapfeat, timesteps):
        calls.append((batch, length, supplied_model, seed))
        return np.zeros((1, length, 28, 3))

    backend.flow = SimpleNamespace(model=model, interpolant=SimpleNamespace(sample=sample))
    payload = backend.execute(prepared)
    assert calls == [(1, 2, model, 420), (1, 2, model, 840)]
    assert len(payload["samples"]) == 2


def test_serialization_keeps_joint_coordinates_ids_and_every_sample(backend, tmp_path, monkeypatch):
    prepared = backend.prepare(request(tmp_path))
    aatype = prepared.mapfeat["aatype"].numpy()
    # Multiples of 0.125 round-trip exactly through the upstream PDB precision.
    coordinates = np.arange(5 * 28 * 3, dtype=float).reshape(1, 5, 28, 3) * 0.125
    samples = [coordinates, coordinates + 100.]
    saved = [array.copy() for array in samples]
    seen_positions = []
    native_from_prediction = backend.nucleicacid.from_prediction

    def capture(features, result):
        seen_positions.append(result["final_atom_positions"])
        return native_from_prediction(features, result)

    monkeypatch.setattr(backend.nucleicacid, "from_prediction", capture)
    result = backend.finalize({"prepared": prepared, "samples": samples, "aatype": aatype})
    assert len(result["artifacts"]["sample_paths"]) == 2
    assert result["structure_path"] == result["artifacts"]["sample_paths"][0]
    assert result["plddt"] is None
    assert result["metadata"]["selection_method"] == "first_sample_no_ranking"
    assert result["metadata"]["accuracy_validation"] == "unvalidated"
    assert result["metadata"]["checkpoint_training"] == "single_chain"
    assert result["metadata"]["chains"] == request(tmp_path)["chains"]
    assert [(entry["start"], entry["stop"]) for entry in result["metadata"]["layout"]] == [(0, 2), (2, 5)]
    for sample_index, path in enumerate(result["artifacts"]["sample_paths"]):
        records = Path(path).read_text().splitlines()
        assert records[-1] == "END" and records.count("END") == 1
        serials = [int(line[6:11]) for line in records if line.startswith(("ATOM  ", "TER   "))]
        assert len(serials) == len(set(serials))
        assert [line[21] for line in records if line.startswith("TER   ")] == ["z", "7"]
        first_ter = next(index for index, line in enumerate(records) if line.startswith("TER   "))
        assert all(line[21] == "z" for line in records[:first_ter])
        assert all(line[21] == "7" for line in records[first_ter + 1:-1])
        for line in records:
            if not line.startswith("ATOM  "):
                continue
            chain_id, residue = line[21], int(line[22:26]) - 1
            assert 0 <= residue < (2 if chain_id == "z" else 3)
            global_residue = residue + (0 if chain_id == "z" else 2)
            atom = backend.constants.atom_types.index(line[12:16].strip())
            actual = [float(line[start:start + 8]) for start in (30, 38, 46)]
            np.testing.assert_array_equal(actual, samples[sample_index][0, global_residue, atom])
        for chain_index, (start, stop) in enumerate(((0, 2), (2, 5))):
            sliced = seen_positions[2 * sample_index + chain_index]
            assert np.shares_memory(sliced, samples[sample_index])
            np.testing.assert_array_equal(sliced, samples[sample_index][:, start:stop])
        np.testing.assert_array_equal(samples[sample_index], saved[sample_index])
