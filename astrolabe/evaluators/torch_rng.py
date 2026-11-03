"""Explicit trajectory RNG exchange; never reseed between model requests."""
import os
import random
import threading
import numpy as np
import torch

_owner_seed = None
_owner_cuda_states = {}
_state_lock = threading.RLock()


def capture_settings():
    """Read process-global numerical settings without initializing CUDA."""
    return {'default_dtype': str(torch.get_default_dtype()).split('.')[-1],
            'deterministic': torch.are_deterministic_algorithms_enabled(),
            'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_tf32': torch.backends.cudnn.allow_tf32,
            'float32_matmul_precision': torch.get_float32_matmul_precision()}


def _validate_settings(settings):
    if not isinstance(settings, dict) or set(settings) != set(capture_settings()):
        raise ValueError('invalid torch numerical settings')
    if settings['default_dtype'] not in ('float16', 'float32', 'float64', 'bfloat16'):
        raise ValueError('invalid torch default dtype')
    if settings['float32_matmul_precision'] not in ('highest', 'high', 'medium'):
        raise ValueError('invalid torch matmul precision')
    if any(type(value) is not bool for key, value in settings.items()
           if key not in ('default_dtype', 'float32_matmul_precision')):
        raise ValueError('invalid torch numerical flag')


def apply_settings(settings):
    _validate_settings(settings)
    torch.set_default_dtype(getattr(torch, settings['default_dtype']))
    torch.use_deterministic_algorithms(settings['deterministic'], warn_only=settings['deterministic_warn_only'])
    torch.backends.cudnn.deterministic = settings['cudnn_deterministic']
    torch.backends.cudnn.benchmark = settings['cudnn_benchmark']
    torch.backends.cuda.matmul.allow_tf32 = settings['matmul_tf32']
    # PyTorch's legacy allow_tf32 setter can collapse 'medium' to 'high'.
    # Restore the full precision choice last; both values were captured from
    # the same framework state.
    torch.set_float32_matmul_precision(settings['float32_matmul_precision'])
    torch.backends.cudnn.allow_tf32 = settings['cudnn_tf32']


def _hidden_owner():
    return bool(os.environ.get('ASTROLABE_EXECUTION_ENDPOINT')) and os.environ.get('CUDA_VISIBLE_DEVICES') == ''


def _physical_key(device, physical_device=None):
    if physical_device is not None:
        key = str(physical_device)
        if _hidden_owner() and key not in (os.environ.get('ASTROLABE_ALLOCATED_CUDA_DEVICES') or '').split(','):
            raise ValueError('CUDA RNG device is outside the job allocation')
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        if not _hidden_owner() and visible and ',' not in visible and key != visible:
            raise ValueError('CUDA RNG device differs from worker binding')
        return key
    if _hidden_owner():
        token = os.environ.get('ASTROLABE_PREFERRED_CUDA_DEVICE')
        allocated = (os.environ.get('ASTROLABE_ALLOCATED_CUDA_DEVICES') or '').split(',')
        if token is None and len(allocated) == 1:
            token = allocated[0]
        if not token or token not in allocated:
            raise RuntimeError('opaque CUDA RNG requires an explicitly allocated device')
        return token
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    return visible if visible and ',' not in visible else str(device)


def register_owner_seed(seed):
    """Record the existing scientific seed at its actual owner-side seed call.

    Never infer a fresh state from ``torch.initial_seed``: another operation may
    have consumed the generator.  Legacy CUDA-visible owners remain unchanged.
    """
    global _owner_seed
    if not _hidden_owner():
        return
    if torch.cuda.is_initialized():
        raise RuntimeError('cannot virtualize an already initialized owner CUDA RNG')
    if type(seed) is not int:
        raise TypeError('owner seed must be an integer')
    with _state_lock:
        _owner_seed = seed
        _owner_cuda_states.clear()


def capture(device, *, physical_device=None):
    cpu = torch.get_rng_state().clone()
    cuda = None
    if str(device).startswith('cuda'):
        if _hidden_owner():
            if torch.cuda.is_initialized():
                raise RuntimeError('GPU-hidden trajectory unexpectedly initialized CUDA')
            key = _physical_key(device, physical_device)
            with _state_lock:
                if key in _owner_cuda_states:
                    cuda = _owner_cuda_states[key].clone()
                elif _owner_seed is not None:
                    cuda = {'kind': 'registered_seed', 'seed': _owner_seed, 'physical_device': key}
                else:
                    raise RuntimeError('owner CUDA RNG has no registered scientific seed or captured state')
        else:
            cuda = torch.cuda.get_rng_state(device).clone()
    return {'cpu': cpu, 'cuda': cuda, 'python': random.getstate(), 'numpy': np.random.get_state(),
            'settings': capture_settings()}


def restore(state, device, *, physical_device=None):
    required = {'cpu', 'cuda', 'python', 'numpy'}
    if (not isinstance(state, dict) or set(state) not in (required, required | {'settings'})
            or not isinstance(state['cpu'], torch.Tensor)):
        raise ValueError('invalid random state')
    # Validate the complete message before touching a scientific generator.
    # In particular a stale/wrong-device CUDA marker may not advance CPU RNG.
    cpu = state['cpu']
    if cpu.dtype != torch.uint8 or str(cpu.device) != 'cpu' or cpu.shape != torch.get_rng_state().shape:
        raise ValueError('invalid CPU random state')
    random.Random().setstate(state['python'])
    validator = np.random.RandomState()
    validator.set_state(state['numpy'])
    if 'settings' in state:
        _validate_settings(state['settings'])
    cuda = state['cuda']
    key = None
    if str(device).startswith('cuda'):
        key = _physical_key(device, physical_device)
        if _hidden_owner():
            if torch.cuda.is_initialized():
                raise RuntimeError('GPU-hidden trajectory unexpectedly initialized CUDA')
            if not isinstance(cuda, torch.Tensor):
                raise ValueError('worker must return actual CUDA RNG state, not an initialization seed')
        if isinstance(cuda, torch.Tensor):
            if cuda.dtype != torch.uint8 or str(cuda.device) != 'cpu' or cuda.ndim != 1 or cuda.numel() == 0:
                raise ValueError('invalid CUDA random state')
        elif isinstance(cuda, dict) and set(cuda) == {'kind', 'seed', 'physical_device'}:
            if cuda['kind'] != 'registered_seed' or type(cuda['seed']) is not int or cuda['physical_device'] != key:
                raise ValueError('invalid initial CUDA RNG identity')
        else:
            raise ValueError('missing CUDA random state')
    elif cuda is not None:
        raise ValueError('unexpected CUDA random state')
    if 'settings' in state:
        apply_settings(state['settings'])
    torch.set_rng_state(cpu)
    if str(device).startswith('cuda'):
        if _hidden_owner():
            with _state_lock:
                _owner_cuda_states[key] = cuda.clone()
        elif isinstance(cuda, torch.Tensor):
            torch.cuda.set_rng_state(cuda, device)
        else:
            # This marker is emitted only immediately after the owner's real
            # seed registration and before any device work on this stream.
            with torch.cuda.device(device):
                torch.cuda.manual_seed(cuda['seed'])
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
