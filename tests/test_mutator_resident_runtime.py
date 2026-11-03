"""Execution-boundary mechanisms; synthetic weights are not backend certification."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("prody")

from astrolabe.evaluators.torch_rng import capture, restore
from astrolabe.mutators.resident import ResidentRunner, request_shape
from astrolabe.mutators.wrappers.ligandmpnn_run import LigandMPNN
from astrolabe.mutators.wrappers.ligandmpnn_utils import ProteinMPNN
from test_ligandmpnn_device_binding import _args
from test_multistate_proteinmpnn import _write_two_chain_pdb


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _equal(a, b)
    else:
        assert left == right


def _checkpoint_case(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.delenv("ASTROLABE_EXECUTION_ENDPOINT", raising=False)
    monkeypatch.delenv("ASTROLABE_EXECUTION_TOKEN", raising=False)
    monkeypatch.delenv("ASTROLABE_ALLOCATED_CUDA_DEVICES", raising=False)
    monkeypatch.delenv("ASTROLABE_MUTATOR_WORKER", raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    checkpoint = tmp_path / "random-proteinmpnn.pt"
    torch.manual_seed(123)
    torch.save({"num_edges": 8, "model_state_dict": ProteinMPNN(k_neighbors=8, model_type="protein_mpnn").state_dict()}, checkpoint)
    args = _args(tmp_path / "outputs")
    args.model_type = "protein_mpnn"
    args.checkpoint_protein_mpnn = str(checkpoint)
    path = tmp_path / "AB.pdb"
    _write_two_chain_pdb(path, "ACD", "EF")
    features = {"pdb_path": str(path), "fixed_residues": "A2 B1 B2", "redesigned_residues": ""}
    return args, features


def test_actual_mpnn_cpu_resident_reuses_weights_and_preserves_two_requests_rng(tmp_path, monkeypatch):
    args, features = _checkpoint_case(tmp_path, monkeypatch)
    torch.set_num_threads(1)
    random.seed(57)
    np.random.seed(57)
    torch.manual_seed(57)
    before = capture("cpu")
    direct = LigandMPNN(args)
    after_init = capture("cpu")
    expected = [direct.design_sequences(**features), direct.design_sequences(**features)]
    expected_rng = capture("cpu")
    direct.close()
    restore(before, "cpu")
    resident = ResidentRunner("ligandmpnn", vars(args))
    process = resident._client.process
    try:
        _equal(after_init, capture("cpu"))
        actual = [resident.design_sequences(**features), resident.design_sequences(**features)]
        _equal(expected, actual)
        _equal(expected_rng, capture("cpu"))
        assert resident.lifecycle["load_count"] == 1
        assert resident.lifecycle["weights_loaded"] is True
        assert resident.lifecycle["gpu_resident"] is False
        assert resident.lifecycle["devices"] == ["cpu"]
    finally:
        resident.close()
    assert process.poll() is not None


def test_mpnn_cpu_cache_is_content_keyed_and_request_local(tmp_path, monkeypatch):
    args, features = _checkpoint_case(tmp_path, monkeypatch)
    runner = LigandMPNN(args)
    path = Path(features["pdb_path"])
    try:
        first = runner._parse_cpu(path, "")
        second = runner._parse_cpu(path, "")
        _equal(first, second)
        second[0]["X"].zero_()
        assert not torch.equal(first[0]["X"], second[0]["X"])
        assert runner.cache_hits == 1 and runner.cache_misses == 1
        _write_two_chain_pdb(path, "ACDE", "EF")
        changed = runner._parse_cpu(path, "")
        assert changed[0]["S"].shape[0] == 6
        assert runner.cache_misses == 2
        runner.prepare_request(**features)
        _write_two_chain_pdb(path, "ACD", "EF")
        with pytest.raises(ValueError, match="changed between prepare and execute"):
            runner.design_sequences(**features)
    finally:
        runner.close()
    assert runner.model is None and runner._parse_cache_bytes == 0


def test_mpnn_request_failure_terminates_private_worker(tmp_path, monkeypatch):
    args, features = _checkpoint_case(tmp_path, monkeypatch)
    resident = ResidentRunner("ligandmpnn", vars(args))
    process = resident._client.process
    with pytest.raises(Exception, match="teacher forcing requires"):
        resident.design_sequences(**features, distribution_semantics="shared_reference_teacher_forcing")
    assert resident._closed and process.poll() is not None


def test_workload_shape_distinguishes_lengths_without_sequence_identity():
    a, _ = request_shape("esm2", [{"sequence": "AAAA"}], {})
    b, _ = request_shape("esm2", [{"sequence": "CCCC"}], {})
    c, _ = request_shape("esm2", [{"sequence": "AAAAA"}], {})
    assert a == b and a != c


def test_child_death_clears_model_readiness_without_erasing_load_history():
    from astrolabe.mutators.wrappers.resident_daemon import lifecycle_snapshot
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait(timeout=5)
    runner = SimpleNamespace(_proc=child, _closed=False, daemon=True,
                             _ready={"weights_loaded": True, "gpu_resident": False,
                                     "device": "cpu", "model_load_count": 1})
    state = lifecycle_snapshot(runner, "dplm")
    assert state["process_alive"] is True
    assert state["weights_loaded"] is False and state["gpu_resident"] is False
    assert state["ready"] is False and state["children"] == []
    assert state["load_count"] == 1 and state["child_exitcode"] == 0


_NATIVE_FIXTURE = '''import argparse
def main(args):
    import torch, random, numpy as np
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    checkpoint = torch.load(args.checkpoint_na_mpnn, weights_only=True)
    model = torch.nn.Linear(4, 4)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to('cpu')
    model.eval()
    values = model(torch.randn(2, 4)) + random.random() + np.random.random()
    sample = torch.multinomial(torch.softmax(values, -1), 1)
    np.savez(args.out_folder, values=values.detach().numpy(), sample=sample.numpy())
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--mode', default='design')
    argparser.add_argument('--model_type', default='na_mpnn')
    argparser.add_argument('--checkpoint_na_mpnn')
    argparser.add_argument('--out_folder')
    argparser.add_argument('--batch_size', type=int, default=1)
    argparser.add_argument('--temperature', type=float, default=0.1)
    argparser.add_argument('--na_shared_tokens', type=int, default=0)
    argparser.add_argument('--k_neighbors', type=int, default=None)
    argparser.add_argument('--seed', type=int, default=0)
    argparser.add_argument('--catch_failed_inferences', type=int, default=0)
    args = argparser.parse_args()
    main(args)
'''


def test_na_native_boundary_keeps_constructor_rng_draws_without_reload(tmp_path, monkeypatch):
    from astrolabe.mutators.wrappers import na_mpnn_native as native
    source = tmp_path / "native.py"
    source.write_text(_NATIVE_FIXTURE)
    # This test admits only its own synthetic architecture fixture, never a
    # user-selected vendor version. Production requires the fixed source SHA.
    monkeypatch.setattr(native, "SOURCE_SHA256", hashlib.sha256(source.read_bytes()).hexdigest())
    checkpoint = tmp_path / "tiny.pt"
    torch.save({"model_state_dict": torch.nn.Linear(4, 4).state_dict()}, checkpoint)
    argv = ["--checkpoint_na_mpnn", str(checkpoint), "--seed", "29", "--out_folder", str(tmp_path / "init.npz")]
    session = native.NativeSession(source, argv)
    direct_namespace = {"__name__": "fixture"}
    exec(compile(_NATIVE_FIXTURE, str(source), "exec"), direct_namespace)
    try:
        for i in range(2):
            direct_path, resident_path = tmp_path / f"direct{i}.npz", tmp_path / f"resident{i}.npz"
            direct_args = session.parse(argv[:-1] + [str(direct_path)])
            direct_namespace["main"](direct_args)
            direct_rng = capture("cpu")
            session.execute(argv[:-1] + [str(resident_path)])
            with np.load(direct_path) as a, np.load(resident_path) as b:
                for key in a.files:
                    np.testing.assert_array_equal(a[key], b[key])
            _equal(direct_rng, capture("cpu"))
        assert session.load_count == 1
        with pytest.raises(ValueError, match="initialization identity"):
            session.execute(argv + ["--seed", "30"])
    finally:
        session.close()
    assert session.model is None


def test_na_native_refuses_unknown_version_and_entropy_seed(tmp_path, monkeypatch):
    from astrolabe.mutators.wrappers import na_mpnn_native as native
    source = tmp_path / "native.py"
    source.write_text(_NATIVE_FIXTURE)
    with pytest.raises(ValueError, match="reviewed source"):
        native.NativeSession(source, [])
    monkeypatch.setattr(native, "SOURCE_SHA256", hashlib.sha256(source.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="nonzero"):
        native.NativeSession(source, ["--checkpoint_na_mpnn", "unused", "--out_folder", str(tmp_path)])
