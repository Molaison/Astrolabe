"""CPU-only child for request-owner tests; never initializes a device."""
import json
import os
import time
from astrolabe.runtime.server import serve


def handle(payload):
    if payload.get('exit'):
        os._exit(17)
    if payload.get('fail'):
        raise ValueError('intentional application failure')
    time.sleep(payload.get('sleep', 0))
    return {'value': payload.get('value'), 'pid': os.getpid()}


if __name__ == '__main__':
    serve(handle, json.loads(os.environ['REQUEST_ISOLATION_A_IDENTITY']), load_count=1,
          lifecycle={'weights_loaded': False, 'gpu_resident': False},
          capabilities={'managed_device': True, 'managed_methods': ['managed']})
