"""Explicit trajectory RNG exchange; never reseed between model requests."""
import random
import numpy as np
import torch


def capture(device):
    cpu = torch.get_rng_state().clone()
    cuda = torch.cuda.get_rng_state(device).clone() if str(device).startswith('cuda') else None
    return {'cpu': cpu, 'cuda': cuda, 'python': random.getstate(), 'numpy': np.random.get_state()}


def restore(state, device):
    if set(state) != {'cpu', 'cuda', 'python', 'numpy'} or not isinstance(state['cpu'], torch.Tensor):
        raise ValueError('invalid random state')
    torch.set_rng_state(state['cpu'])
    if str(device).startswith('cuda'):
        if not isinstance(state['cuda'], torch.Tensor): raise ValueError('missing CUDA random state')
        torch.cuda.set_rng_state(state['cuda'], device)
    elif state['cuda'] is not None:
        raise ValueError('unexpected CUDA random state')
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
