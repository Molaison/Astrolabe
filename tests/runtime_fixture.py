import hashlib
import os
import time
import json


def cpu_work(payload):
    start = time.monotonic()
    digest = b'astrolabe'
    for _ in range(payload['rounds']):
        digest = hashlib.sha256(digest).digest()
    time.sleep(payload['delay'])
    return {'start': start, 'end': time.monotonic(), 'pid': os.getpid(),
            'device': os.environ.get('CUDA_VISIBLE_DEVICES'), 'digest': digest.hex()}


def cpu_big_result(payload):
    return 'x' * payload['size']


def daemon_handler(payload):
    def event(kind):
        if payload.get('event_log'):
            with open(payload['event_log'], 'a') as f:
                f.write(json.dumps({'kind': kind, 'time': time.monotonic(),
                                    'pid': os.getpid(), 'owner': payload.get('owner')}) + '\n')
    event('forward_start')
    if payload.get('started_path'):
        from pathlib import Path
        Path(payload['started_path']).touch()
    if payload.get('wait_for_path'):
        from pathlib import Path
        deadline = time.monotonic() + 40
        while not Path(payload['wait_for_path']).exists():
            if time.monotonic() >= deadline:
                raise TimeoutError('independent CPU preparation did not complete while forward was admitted')
            time.sleep(.01)
    if payload.get('sleep'):
        time.sleep(payload['sleep'])
    event('forward_end')
    return {'value': payload.get('value'), 'pid': os.getpid(),
            'artifact_root': os.environ.get('ASTROLABE_EXECUTION_ARTIFACT_ROOT'),
            'visible': os.environ.get('CUDA_VISIBLE_DEVICES')}


if __name__ == '__main__':
    from astrolabe.runtime.server import serve
    identity = json.loads(os.environ.get('RUNTIME_TEST_IDENTITY', 'null'))
    serve(daemon_handler, identity or {'environment': 'test', 'code': 'test', 'weights': 'none', 'device': 'cpu'},
          load_count=1, lifecycle={'weights_loaded': False, 'gpu_resident': False})
