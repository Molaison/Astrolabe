"""Content identities independently recomputed in each worker process."""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import sys
import json
import subprocess


def _digest_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest_path(path):
    path = Path(path).expanduser().resolve(strict=True)
    if path.is_file(): return _digest_file(path)
    if not path.is_dir(): raise ValueError(f'not a file or directory: {path}')
    h = hashlib.sha256()
    for file in sorted(p for p in path.rglob('*') if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith('.pyc')):
        rel = file.relative_to(path).as_posix().encode()
        h.update(len(rel).to_bytes(8, 'big')); h.update(rel)
        h.update(bytes.fromhex(_digest_file(file)))
    return h.hexdigest()


def environment_identity(python, *, env=None, device=None, env_remove=()):
    """Read the environment that will actually launch the model.

    A hidden trajectory owner has intentionally different JAX/CUDA masks.
    Its environment must never stand in for the broker's worker environment.
    The admitted probe uses the same overrides, device binding and thread
    limits as worker startup. The worker independently repeats this metadata
    probe in its own inherited environment before loading any checkpoint.
    """
    removed = tuple(env_remove)
    allowed_removals = {'PYTHONPATH', 'APPTAINERENV_PYTHONPATH', 'SINGULARITYENV_PYTHONPATH', 'TRITON_CACHE_DIR'}
    if any(key not in allowed_removals for key in removed):
        raise ValueError('unsupported metadata environment removal')
    overrides = dict(env or {})
    # Empty PYTHONPATH prevents sitecustomize imports before the probe script
    # can apply the external wrapper's actual unset behavior. Allocation masks
    # are deliberately not removable through this metadata-only facility.
    overrides.update({key: '' for key in removed})
    script = '''import contextlib,hashlib,json,os,sys,importlib.metadata as m
for key in json.loads(sys.argv[1]): os.environ.pop(key, None)
names = ('numpy', 'jax', 'jaxlib', 'dm-haiku', 'torch', 'transformers', 'omegaconf', 'lightning', 'pytorch-lightning', 'biotite', 'atomworks', 'hydra-core', 'safetensors', 'openmm', 'triton', 'pytorch-triton', 'nvidia-cudnn-cu11', 'nvidia-cudnn-cu12', 'nvidia-cuda-runtime-cu11', 'nvidia-cuda-runtime-cu12', 'nvidia-cublas-cu11', 'nvidia-cublas-cu12', 'nvidia-cusolver-cu11', 'nvidia-cusolver-cu12', 'nvidia-nvjitlink-cu12')
versions = {}
for name in names:
 try: versions[name] = m.version(name)
 except m.PackageNotFoundError: versions[name] = None
prefixes = ('XLA_', 'JAX_', 'CUDA_', 'CUBLAS_', 'CUDNN_', 'PYTORCH_', 'TORCH_', 'TRITON_', 'OPENMM_', 'OMP_', 'MKL_', 'OPENBLAS_', 'NUMEXPR_', 'VECLIB_', 'DNNL_', 'ONEDNN_', 'TF_')
keys = {'NVIDIA_VISIBLE_DEVICES', 'NVIDIA_TF32_OVERRIDE', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'PYTHONPATH', 'PYTHONHOME', 'PYTHONHASHSEED', 'PYTHONNOUSERSITE', 'PYTHONSAFEPATH', 'PATH', 'LD_LIBRARY_PATH', 'LD_PRELOAD', 'CC', 'CXX', 'CFLAGS', 'CXXFLAGS', 'LDFLAGS', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'TOKENIZERS_PARALLELISM'}
effective = {key:value for key,value in os.environ.items() if key in keys or key.startswith(prefixes)}
native = {}
if versions['torch'] is not None:
 try:
  with contextlib.redirect_stdout(sys.stderr):
   import torch
   native = {'torch': torch.__version__, 'cuda_build': torch.version.cuda,
             'hip_build': torch.version.hip, 'cudnn_runtime': torch.backends.cudnn.version(),
             'build_config_sha256': hashlib.sha256(torch.__config__.show().encode()).hexdigest()}
 except Exception as exc:
  native = {'probe_error': type(exc).__name__ + ': ' + str(exc)[:512]}
print(json.dumps({'prefix':sys.prefix,'base_prefix':sys.base_prefix,'version':sys.version,
                  'packages':versions,'effective_env':effective,'native_libraries':native},sort_keys=True))'''
    command = [str(python), '-c', script, json.dumps(removed)]
    if os.environ.get('ASTROLABE_EXECUTION_ENDPOINT') or os.environ.get('ASTROLABE_EXECUTION_TOKEN'):
        from astrolabe.runtime.admission import run_probe
        platform = device.get('platform', device.get('name', '')) if isinstance(device, dict) else device
        binding = 'cpu' if platform == 'cpu' else (device.get('visible') if isinstance(device, dict) else None)
        metadata = run_probe(command, env=overrides, device=binding)
    else:
        child_env = os.environ.copy()
        child_env.update(overrides)
        probe = subprocess.run(command, check=True, capture_output=True, text=True, env=child_env)
        metadata = json.loads(probe.stdout)
    return {'python': str(Path(python).resolve(strict=True)), 'sha256': digest_path(python), **metadata}


def identity(python, code_paths, weight_paths, device, model_config=None, *, env=None, env_remove=()):
    return {'environment': environment_identity(python, env=env, device=device, env_remove=env_remove),
            'code': [digest_path(p) for p in code_paths],
            'execution_code': {name: digest_path(Path(__file__).parent / name)
                               for name in ('worker_identity.py', 'lifecycle.py', 'torch_rng.py', 'worker_session.py',
                                            'request_path.py', 'loader_observer.py', 'tree_codec.py', 'artifact_budget.py',
                                            '../runtime/server.py', '../runtime/contracts.py', '../runtime/admission.py',
                                            '../runtime/client.py', '../runtime/broker_client.py', '../runtime/telemetry.py')},
            'weights': [digest_path(p) for p in weight_paths],
            'device': device,
            'model_config': model_config if model_config is not None else {}}


def probe_device(python, backend, requested=None, *, env=None):
    args = [str(python), '-m', 'astrolabe.evaluators.device_probe', backend]
    if requested is not None:
        args.append(requested)
    try:
        from astrolabe.runtime.admission import run_probe
    except ImportError:
        if os.environ.get('ASTROLABE_EXECUTION_ENDPOINT') or os.environ.get('ASTROLABE_EXECUTION_TOKEN'):
            raise RuntimeError('broker device probing requires runtime.admission')
        child_env = os.environ.copy()
        if env:
            child_env.update(env)
        proc = subprocess.run(args, check=True, capture_output=True, text=True, env=child_env)
        return json.loads(proc.stdout)
    return run_probe(args, env=env, device='cpu' if requested == 'cpu' else None)
