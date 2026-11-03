"""CPU fixture for real process staging/identity tests; never a GPU benchmark."""
import json
import os
from astrolabe.runtime.server import serve

stored = {}


def handle(payload):
    method, key = payload['method'], payload['key']
    if method == 'prepare':
        stored[key] = bytes(range(128))
        return {'prepared': len(stored[key])}
    if method == 'execute':
        stored[key] = sum(stored[key])
        return {'executed': True}
    if method == 'finalize':
        return {'value': stored.pop(key)}
    raise ValueError('unknown fixture method')


serve(handle, json.loads(os.environ['RUNTIME_TEST_IDENTITY']), load_count=1,
      lifecycle={'weights_loaded': False, 'gpu_resident': False},
      capabilities={'request_staging': True, 'cpu_phases': ['cpu_prepare', 'cpu_finalize'],
                    'phase_methods': {'prepare': 'cpu_prepare', 'finalize': 'cpu_finalize'}})
