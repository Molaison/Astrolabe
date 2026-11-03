"""Real CPU torch RNG plus simulated CUDA transport; never launches GPU work."""
import contextlib
import random
import numpy as np
import pytest

torch = pytest.importorskip('torch')
from astrolabe.evaluators import torch_rng as rng


@pytest.fixture
def hidden_owner(monkeypatch):
    monkeypatch.setenv('ASTROLABE_EXECUTION_ENDPOINT', 'fixture-broker')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setenv('ASTROLABE_ALLOCATED_CUDA_DEVICES', '3,4')
    monkeypatch.setenv('ASTROLABE_PREFERRED_CUDA_DEVICE', '3')
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: False)
    monkeypatch.setattr(rng, '_owner_seed', None)
    monkeypatch.setattr(rng, '_owner_cuda_states', {})
    return monkeypatch


def test_opaque_rng_requires_real_seed_registration_and_returns_actual_consumed_state(hidden_owner):
    monkeypatch = hidden_owner
    with pytest.raises(RuntimeError, match='no registered scientific seed'):
        rng.capture('cuda:0')
    rng.register_owner_seed(91)
    initial = rng.capture('cuda:0')
    assert initial['cuda'] == {'kind': 'registered_seed', 'seed': 91, 'physical_device': '3'}
    cuda_calls = []
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '3')
    monkeypatch.setattr(torch.cuda, 'device', lambda device: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, 'manual_seed', lambda seed: cuda_calls.append(('seed', seed)))
    monkeypatch.setattr(torch.cuda, 'get_rng_state', lambda device: torch.tensor([7, 9, 42], dtype=torch.uint8))
    rng.restore(initial, 'cuda:0')
    assert cuda_calls == [('seed', 91)]
    returned = rng.capture('cuda:0')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    rng.restore(returned, 'cuda:0')
    # Another backend receives the already-consumed bytes, never initial_seed.
    next_backend = rng.capture('cuda:0')
    assert torch.equal(next_backend['cuda'], torch.tensor([7, 9, 42], dtype=torch.uint8))
    next_backend['cuda'][0] = 0
    assert rng.capture('cuda:0')['cuda'][0].item() == 7
    assert isinstance(rng.capture('cuda:0', physical_device='4')['cuda'], dict)
    with pytest.raises(ValueError, match='outside the job allocation'):
        rng.capture('cuda:0', physical_device='5')
    rng.register_owner_seed(92)
    assert rng.capture('cuda:0')['cuda']['seed'] == 92


def test_invalid_cuda_reply_never_partially_mutates_owner_rng(hidden_owner):
    rng.register_owner_seed(123)
    first = rng.capture('cuda:0')
    torch.rand(4); random.random(); np.random.random()
    before = rng.capture('cuda:0')
    with pytest.raises(ValueError, match='actual CUDA RNG state'):
        rng.restore(first, 'cuda:0')
    after = rng.capture('cuda:0')
    assert torch.equal(before['cpu'], after['cpu'])
    assert before['python'] == after['python']
    assert np.array_equal(before['numpy'][1], after['numpy'][1])


def test_cpu_rng_roundtrip_preserves_native_interleaving(monkeypatch):
    monkeypatch.delenv('ASTROLABE_EXECUTION_ENDPOINT', raising=False)
    torch.manual_seed(42); random.seed(42); np.random.seed(42)
    state = rng.capture('cpu')
    expected = (torch.rand(4), random.random(), np.random.rand(3))
    rng.restore(state, 'cpu')
    assert torch.equal(torch.rand(4), expected[0])
    assert random.random() == expected[1]
    np.testing.assert_array_equal(np.random.rand(3), expected[2])


def test_worker_preserves_constructor_dtype_cudnn_and_native_precision_side_effect(tmp_path, monkeypatch):
    import os
    import subprocess
    import sys
    from astrolabe.evaluators.tree_codec import dump_tree, load_tree
    monkeypatch.delenv('ASTROLABE_EXECUTION_ENDPOINT', raising=False)
    monkeypatch.delenv('ASTROLABE_EXECUTION_TOKEN', raising=False)
    original = rng.capture('cpu')
    try:
        torch.set_default_dtype(torch.float64)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision('medium')
        initial = rng.capture('cpu')
        dump_tree(tmp_path / 'input', initial)
        # This is the actual direct-construction RNG consumption and the
        # observed Boltz native side effect; no substitute seed is introduced.
        direct_model = torch.nn.Linear(2, 2)
        torch.set_float32_matmul_precision('highest')
        expected_next = torch.rand(3)
        script = '''import sys,torch
from pathlib import Path
from astrolabe.evaluators.torch_rng import capture,restore
from astrolabe.evaluators.tree_codec import load_tree,dump_tree
path=Path(sys.argv[1])
restore(load_tree(path/'input'),'cpu')
model=torch.nn.Linear(2,2)
assert model.weight.dtype == torch.float64
assert torch.backends.cudnn.deterministic is True
assert torch.backends.cudnn.benchmark is False
assert torch.get_float32_matmul_precision() == 'medium'
torch.set_float32_matmul_precision('highest')
dump_tree(path/'output',capture('cpu'))
'''
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
        subprocess.run([sys.executable, '-c', script, str(tmp_path)], check=True, env=env)
        rng.restore(load_tree(tmp_path / 'output'), 'cpu')
        assert torch.get_float32_matmul_precision() == 'highest'
        assert torch.backends.cudnn.deterministic is True
        assert torch.equal(torch.rand(3), expected_next)
        assert direct_model.weight.dtype == torch.float64
    finally:
        rng.restore(original, 'cpu')
