"""Test-only instrumentation of the real DummyEvaluator trajectory."""
import os
if os.environ.get('ASTROLABE_TEST_BROKER') and os.environ.get('ASTROLABE_EXECUTION_ENDPOINT'):
    import hashlib
    import json
    import sys
    import time
    from pathlib import Path
    from astrolabe.evaluators.dummy import DummyEvaluator
    from astrolabe.runtime.client import DaemonClient

    original = DummyEvaluator.evaluate
    event_log = os.environ['ASTROLABE_TEST_EVENTS']
    identity = {'environment': 'test', 'code': 'test', 'weights': 'none', 'device': 'cpu'}
    rendezvous = Path(event_log).parent
    calls = 0

    def wait_for(path):
        deadline = time.monotonic() + 40
        while not path.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError('controller did not supply the independent overlap fixture: ' + str(path))
            time.sleep(.01)
    def event(kind, **kwargs):
        record = {'kind': kind, 'time': time.monotonic(), 'owner': os.getpid(), **kwargs}
        fd = os.open(event_log, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            os.write(fd, (json.dumps(record) + '\n').encode())
        finally:
            os.close(fd)

    def evaluate(self, inputs=None, runtime_config=None):
        global calls
        calls += 1
        owner_path = rendezvous / 'first-owner'
        try:
            fd = os.open(owner_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            # The first owner atomically creates the file, then writes its PID.
            deadline = time.monotonic() + 5
            while not owner_path.read_text().strip():
                if time.monotonic() >= deadline:
                    raise TimeoutError('fixture owner did not publish its identity')
                time.sleep(.001)
        else:
            with os.fdopen(fd, 'w') as output:
                output.write(str(os.getpid()))
        primary = int(owner_path.read_text()) == os.getpid()
        paired = primary and calls == 4
        event('prepare_start')
        if paired:
            # Three completed calls supply one cold and two warm observations.
            # Keep CPU preparation available while the real controller starts
            # the independent owner, regardless of interpreter import latency.
            wait_for(rendezvous / 'peer-forward-started')
        event('cpu_work_start')
        digest = b'cpu-stage'
        for _ in range(150000):
            digest = hashlib.sha256(digest).digest()
        event('cpu_work_end')
        if paired:
            (rendezvous / 'paired-cpu-finished').touch()
        time.sleep(.35)
        event('prepare_end')
        with DaemonClient([sys.executable, os.environ['ASTROLABE_TEST_FIXTURE']], identity) as client:
            event('resident', worker=client.hello['pid'], loads=client.hello['load_count'])
            payload = {'value': digest.hex(), 'sleep': .8, 'event_log': event_log, 'owner': os.getpid()}
            if not primary and calls == 1:
                payload.update(started_path=str(rendezvous / 'peer-forward-started'),
                               wait_for_path=str(rendezvous / 'paired-cpu-finished'))
            client.call(payload, timeout=45,
                        runtime={'phase': 'composite', 'shape_key': 'sha256-rounds=150000;result-hex=64'})
        return original(self, inputs, runtime_config)
    DummyEvaluator.evaluate = evaluate
