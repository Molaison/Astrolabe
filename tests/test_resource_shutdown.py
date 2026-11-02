from types import SimpleNamespace
from astrolabe.mutators.base import Mutator
from astrolabe.structure_processors.base import shutdown_structure_processors
from astrolabe.structure_processors.peptide_docker import RAPiDockPeptideDockingProcessor


def test_mutator_closes_existing_runner_once():
    calls = []
    mutator = Mutator()
    mutator.runner = SimpleNamespace(close=lambda: calls.append('close'), shutdown=lambda: calls.append('shutdown'))
    mutator.shutdown()
    mutator.shutdown()
    assert calls == ['close']


def test_processor_shutdown_deduplicates_and_continues_after_error(caplog):
    calls = []
    processor = RAPiDockPeptideDockingProcessor.__new__(RAPiDockPeptideDockingProcessor)
    processor.runner = SimpleNamespace(shutdown=lambda: calls.append('rapidock'))
    def fail():
        raise RuntimeError('test shutdown failure')
    bad = SimpleNamespace(shutdown=fail)
    shutdown_structure_processors({'bad':bad, 'p':processor, 'alias':processor})
    assert calls == ['rapidock']
    assert "Failed to shutdown structure processor 'bad'" in caplog.text


def test_rapidock_processor_stops_its_existing_daemon_process():
    import subprocess
    import sys
    import threading
    from astrolabe.structure_processors.wrappers.rapidock_runner import RAPiDockRunner
    process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    runner = RAPiDockRunner.__new__(RAPiDockRunner)
    runner._proc = process
    runner._lock = threading.Lock()
    runner._daemon_request = lambda _request: {'type':'shutdown'}
    processor = RAPiDockPeptideDockingProcessor.__new__(RAPiDockPeptideDockingProcessor)
    processor.runner = runner
    try:
        processor.shutdown()
        process.wait(timeout=5)
        assert process.poll() is not None and runner._proc is None
        processor.shutdown()
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
