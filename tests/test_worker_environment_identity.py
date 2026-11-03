"""Effective launch environments, not GPU-hidden owner masks, define reuse."""
import os
import sys

import pytest

from astrolabe.evaluators.worker_identity import environment_identity, identity


@pytest.mark.parametrize('key,left,right', [
    ('XLA_FLAGS', '--xla_gpu_enable_triton_gemm=false', '--xla_gpu_enable_triton_gemm=true'),
    ('JAX_ENABLE_X64', '0', '1'),
    ('JAX_DEFAULT_MATMUL_PRECISION', 'high', 'highest'),
    ('CUBLAS_WORKSPACE_CONFIG', ':16:8', ':4096:8'),
    ('NVIDIA_TF32_OVERRIDE', '0', '1'),
    ('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:64', 'max_split_size_mb:128'),
    ('TRITON_ALWAYS_COMPILE', '0', '1'),
])
def test_initialization_environment_changes_model_compatibility(tmp_path, monkeypatch, key, left, right):
    monkeypatch.delenv('ASTROLABE_EXECUTION_ENDPOINT', raising=False)
    monkeypatch.delenv('ASTROLABE_EXECUTION_TOKEN', raising=False)
    code, weights = tmp_path / 'model.py', tmp_path / 'weights'
    code.write_text('local source')
    weights.write_bytes(b'local checkpoint')
    # These values are metadata only. No model/JAX/CUDA initialization occurs.
    first = identity(sys.executable, [code], [weights], 'cpu', env={key: left})
    second = identity(sys.executable, [code], [weights], 'cpu', env={key: right})
    assert first != second
    assert first['environment']['effective_env'][key] == left
    assert second['environment']['effective_env'][key] == right
    assert first['code'] == second['code'] and first['weights'] == second['weights']


def test_environment_probe_uses_worker_overrides_without_mutating_owner(monkeypatch):
    monkeypatch.delenv('ASTROLABE_EXECUTION_ENDPOINT', raising=False)
    monkeypatch.delenv('ASTROLABE_EXECUTION_TOKEN', raising=False)
    monkeypatch.setenv('JAX_PLATFORMS', 'cpu')
    monkeypatch.setenv('ASTROLABE_INNER_BROKER_TOKEN', 'must-not-be-fingerprinted')
    result = environment_identity(sys.executable, env={'JAX_PLATFORMS': 'cuda'})
    assert result['effective_env']['JAX_PLATFORMS'] == 'cuda'
    assert os.environ['JAX_PLATFORMS'] == 'cpu'
    assert not any(key.startswith('ASTROLABE_') for key in result['effective_env'])
    assert 'triton' in result['packages'] and 'nvidia-cudnn-cu12' in result['packages']
    if result['packages']['torch'] is not None and 'probe_error' not in result['native_libraries']:
        assert {'cuda_build', 'cudnn_runtime', 'build_config_sha256'} <= result['native_libraries'].keys()


def test_external_wrapper_unsets_are_observed_without_removing_allocation(monkeypatch):
    monkeypatch.delenv('ASTROLABE_EXECUTION_ENDPOINT', raising=False)
    monkeypatch.delenv('ASTROLABE_EXECUTION_TOKEN', raising=False)
    monkeypatch.setenv('TRITON_CACHE_DIR', '/owner/cache')
    removed = environment_identity(sys.executable, env_remove=('PYTHONPATH', 'TRITON_CACHE_DIR'))
    assert 'PYTHONPATH' not in removed['effective_env']
    assert 'TRITON_CACHE_DIR' not in removed['effective_env']
    assert os.environ['TRITON_CACHE_DIR'] == '/owner/cache'
    with pytest.raises(ValueError, match='unsupported metadata environment removal'):
        environment_identity(sys.executable, env_remove=('CUDA_VISIBLE_DEVICES',))


@pytest.mark.parametrize('device,binding', [
    ({'name': 'cpu', 'visible': ''}, 'cpu'),
    ({'name': 'cuda:0', 'visible': 'allocated-gpu'}, 'allocated-gpu'),
])
def test_hidden_owner_gets_identity_from_admitted_effective_environment(monkeypatch, device, binding):
    from astrolabe.runtime import admission
    observed = []
    monkeypatch.setenv('ASTROLABE_EXECUTION_ENDPOINT', '@fixture')
    monkeypatch.setenv('ASTROLABE_EXECUTION_TOKEN', 'fixture-token')
    monkeypatch.setenv('JAX_PLATFORMS', 'cpu')
    def probe(command, *, env, device):
        observed.append((command, env, device))
        return {'effective_env': {'JAX_PLATFORMS': 'cuda', 'OMP_NUM_THREADS': '1'},
                'prefix': 'worker-environment'}
    monkeypatch.setattr(admission, 'run_probe', probe)
    result = environment_identity(sys.executable, env={'XLA_FLAGS': '--actual-launch-option'}, device=device)
    assert observed[0][1] == {'XLA_FLAGS': '--actual-launch-option'}
    assert observed[0][2] == binding
    assert result['effective_env']['JAX_PLATFORMS'] == 'cuda'
    assert result['prefix'] == 'worker-environment'
