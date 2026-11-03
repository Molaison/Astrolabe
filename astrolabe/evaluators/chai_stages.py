"""Chai native API stage boundaries without rewriting sampling or ranking.

The embedding call returns CPU tensors in both the pinned upstream and
Astrolabe's X-mask wrapper. Folding retains all live GPU intermediates until
its native function returns; individual component contexts are not treated as
independent requests and native joint diffusion remains intact.
"""
from contextlib import contextmanager
from functools import wraps
import time

from astrolabe.runtime.admission import operation, stage_event


@contextmanager
def managed_chai_stages(chai, identity, lifecycle):
    originals = {}
    clock = {'cpu_start': time.perf_counter(), 'folded': False}
    def cpu_boundary():
        if clock['cpu_start'] is not None:
            stage_event('cpu_finalize' if clock['folded'] else 'cpu_prepare',
                        duration=time.perf_counter() - clock['cpu_start'], identity=identity)
            clock['cpu_start'] = None
    def wrap(attribute, *, embedding=False):
        original = getattr(chai, attribute)
        originals[attribute] = original
        @wraps(original)
        def invoke(*args, **kwargs):
            cpu_boundary()
            with operation(identity, phase='device') as lease:
                value = original(*args, **kwargs)
                if embedding:
                    tensor = value.esm_embeddings
                    if str(tensor.device) != 'cpu':
                        raise RuntimeError('Chai ESM stage no longer returns CPU embeddings')
                # The native fold computes GPU confidence before moving its
                # outputs to CPU. Keep its entire continuous subgraph intact.
                lease.update_lifecycle(**lifecycle())
            clock['cpu_start'] = time.perf_counter()
            if not embedding:
                clock['folded'] = True
            return value
        setattr(chai, attribute, invoke)
    if hasattr(chai, 'get_esm_embedding_context'):
        wrap('get_esm_embedding_context', embedding=True)
    wrap('run_folding_on_context')
    try:
        yield
    finally:
        cpu_boundary()
        for attribute, original in originals.items():
            setattr(chai, attribute, original)
