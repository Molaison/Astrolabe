"""Real Unix RPC, controlled CPU telemetry, no model/device operations."""
import concurrent.futures
import copy
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
import uuid

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = '_astrolabe_readiness_recovery'
spec = importlib.util.spec_from_file_location(
    PACKAGE, ROOT / 'astrolabe/runtime/__init__.py',
    submodule_search_locations=[str(ROOT / 'astrolabe/runtime')])
runtime = importlib.util.module_from_spec(spec)
sys.modules[PACKAGE] = runtime
spec.loader.exec_module(runtime)
a = sys.modules[PACKAGE + '.admission']


def eventually(predicate, timeout=2):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(0.003)
    raise AssertionError('condition did not become true')


class ControlledSampler:
    def __init__(self, gpu):
        self.row = dict(index=0, uuid=gpu, total_bytes=1000,
                        free_bytes=1000, utilization=0, processes=[])
        self.mode = 'valid'
        self.entered = threading.Event()
        self.resume = threading.Event()

    def __call__(self):
        mode = self.mode
        if mode == 'unknown':
            raise ValueError('controlled unknown accounting; no claimed root cause')
        if mode == 'invalid':
            row = copy.deepcopy(self.row)
            del row['processes']
            return [row]
        if mode == 'paused':
            self.entered.set()
            if not self.resume.wait(2):
                raise RuntimeError('controlled sampler pause expired')
        return [copy.deepcopy(self.row)]


class TelemetryReadinessTests(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ)
        self.env.start()
        self.addCleanup(self.env.stop)
        for key in ('CUDA_VISIBLE_DEVICES', 'SLURM_JOB_GPUS', 'SLURM_STEP_GPUS', 'ASTROLABE_RESOURCE_SOCKET'):
            os.environ.pop(key, None)
        self.temp = tempfile.TemporaryDirectory(prefix='arr-')
        self.addCleanup(self.temp.cleanup)
        self.gpu = 'GPU-controlled-' + uuid.uuid4().hex
        self.sampler = ControlledSampler(self.gpu)
        self.events = Path(self.temp.name) / 'events.jsonl'
        self.config = dict(devices=[dict(uuid=self.gpu, index=0, vram_bytes=800)],
                           socket_dir=self.temp.name, telemetry_recovery=True,
                           telemetry_path=str(self.events), ram_bytes=1000,
                           output_bytes=1000, sample_interval=0.01,
                           sample_max_age=0.5, shutdown_timeout=2)
        self.broker = None
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)
        self.addCleanup(self.pool.shutdown, wait=True)
        self.leases = []
        self.addCleanup(self.cleanup)

    def start(self, **config):
        self.broker = a.ResourceBroker(dict(self.config, **config), self.sampler).start()
        self.client = a.ResourceClient(self.broker.socket_path)
        return self.broker

    def cleanup(self):
        self.sampler.mode = 'valid'
        self.sampler.resume.set()
        for lease in self.leases:
            if lease.token:
                lease.finish()
            else:
                lease.cancel()
        if self.broker:
            # Only test-injected fatal flags are cleared for cleanup.
            self.broker._fence_error = None
            self.broker._ram_observation_error = None
            self.broker.close()
            self.assertEqual(self.broker._pending, [])
            self.assertEqual(self.broker._leases, {})

    def unknown(self):
        self.sampler.mode = 'unknown'
        eventually(lambda: self.client.snapshot()['telemetry_error'] == 'ValueError')

    def submit(self, **kwargs):
        lease = self.client.lease(**kwargs)
        self.leases.append(lease)
        return lease, self.pool.submit(lease.__enter__)

    def pending(self, n=1):
        eventually(lambda: self.client.snapshot()['pending'] == n)

    def rows(self):
        return [json.loads(line) for line in self.events.read_text().splitlines()]

    def test_unknown_no_fit_no_model_until_fresh_exactly_one_grant(self):
        self.start()
        self.unknown()
        calls = []
        original = self.broker._fit
        def checked(request):
            self.assertTrue(self.broker._admission_sample_ready())
            self.assertTrue(self.broker._sample)
            return original(request)
        with mock.patch.object(self.broker, '_fit', side_effect=checked) as fit:
            lease = self.client.lease(phase='output_to_pdb', ram_bytes=7, output_bytes=11)
            self.leases.append(lease)
            def workflow():
                with lease:
                    calls.append('controlled model call after grant')
            future = self.pool.submit(workflow)
            self.pending()
            state = self.client.snapshot()
            self.assertEqual((state['pending_bytes'], state['active'], state['resident_models']), (18, 0, 0))
            self.assertEqual(calls, [])
            fit.assert_not_called()
            self.assertFalse(any(r['event'] == 'acquire' for r in self.rows()))
            self.sampler.mode = 'invalid'
            time.sleep(0.04)
            fit.assert_not_called()
            self.assertEqual(calls, [])
            self.sampler.mode = 'valid'
            future.result(2)
            self.assertEqual(len(calls), 1)
            self.pending(0)
        rows = self.rows()
        self.assertEqual(sum(r['event'] == 'acquire' for r in rows), 1)
        wait = next(r for r in rows if r['event'] == 'telemetry_recovery_wait')
        ready = next(r for r in rows if r['event'] == 'telemetry_recovery_ready')
        self.assertEqual(wait['sample_error'], 'ValueError')
        self.assertEqual(wait['sample_error_detail'], 'controlled unknown accounting; no claimed root cause')
        self.assertEqual(wait['error'], 'telemetry unavailable; admission closed')
        self.assertGreater(ready['sample_at'], ready['failed_sample_at'])
        self.assertTrue(any(r['event'] == 'sample' and not r['healthy'] and r['telemetry_error'] == 'ValueError' for r in rows))

    def test_cpu_prepare_and_execute_wait_for_fresh_sample_before_fit(self):
        for phase in ('prepare', 'execute'):
            with self.subTest(phase=phase):
                self.start()
                self.unknown()
                calls = []
                with mock.patch.object(self.broker, '_fit', wraps=self.broker._fit) as fit:
                    lease = self.client.lease(resource='cpu', phase=phase, ram_bytes=7, output_bytes=11)
                    self.leases.append(lease)
                    def work():
                        with lease:
                            calls.append('controlled output_to_pdb')
                    future = self.pool.submit(work)
                    self.pending()
                    state = self.client.snapshot()
                    self.assertEqual((state['pending_bytes'], state['active']), (18, 0))
                    self.assertEqual(calls, [])
                    fit.assert_not_called()
                    self.sampler.mode = 'valid'
                    future.result(2)
                    self.assertEqual(len(calls), 1)
                rows = self.rows()
                acquired = [r for r in rows if r['event'] == 'acquire']
                self.assertEqual(len(acquired), 1)
                self.assertIsNone(acquired[0]['device'])
                self.assertEqual(acquired[0]['reserved_bytes'], 0)
                self.broker.close()
                self.broker = None
                self.events.unlink()

    def test_stale_cached_valid_sample_cannot_fit(self):
        self.start(sample_max_age=0.15)
        self.sampler.mode = 'paused'
        self.assertTrue(self.sampler.entered.wait(1))
        with self.broker._cv:
            self.broker._sample_at = time.monotonic() - 1
        with mock.patch.object(self.broker, '_fit', wraps=self.broker._fit) as fit:
            lease, future = self.submit()
            self.pending()
            fit.assert_not_called()
            self.sampler.mode = 'valid'
            self.sampler.resume.set()
            future.result(2).finish()
            self.assertTrue(fit.called)

    def test_permanent_unknown_bounded_without_timeout_and_keeps_error(self):
        self.start(sample_max_age=0.12)
        self.unknown()
        before = time.monotonic()
        lease, future = self.submit(resource='cpu')
        with self.assertRaisesRegex(a.AdmissionTimeout, 'telemetry unavailable; admission closed; readiness deadline expired'):
            future.result(1)
        elapsed = time.monotonic() - before
        self.assertGreaterEqual(elapsed, 0.10)
        self.assertLess(elapsed, 0.6)
        self.pending(0)
        self.assertIsNone(lease.token)
        self.assertTrue(any(r['event'] == 'telemetry_recovery_timeout' for r in self.rows()))

    def test_original_short_deadline_not_extended(self):
        self.start(sample_max_age=0.6)
        self.unknown()
        before = time.monotonic()
        _, future = self.submit(timeout=0.07)
        with self.assertRaisesRegex(a.AdmissionTimeout, '^admission deadline expired$'):
            future.result(1)
        self.assertLess(time.monotonic() - before, 0.4)
        self.pending(0)

    def test_original_deadline_includes_resource_queue_before_fault(self):
        self.start(sample_max_age=0.6)
        held = self.client.lease().__enter__()
        self.leases.append(held)
        before = time.monotonic()
        _, future = self.submit(timeout=0.18)
        self.pending()
        time.sleep(0.07)
        self.unknown()
        with self.assertRaisesRegex(a.AdmissionTimeout, '^admission deadline expired$'):
            future.result(1)
        self.assertLess(time.monotonic() - before, 0.4)
        held.finish()

    def test_original_deadline_includes_readiness_then_resource_wait(self):
        self.start()
        held = self.client.lease().__enter__()
        self.leases.append(held)
        self.unknown()
        before = time.monotonic()
        _, future = self.submit(timeout=0.16)
        self.pending()
        time.sleep(0.05)
        self.sampler.mode = 'valid'
        with self.assertRaisesRegex(a.AdmissionTimeout, '^admission deadline expired$'):
            future.result(1)
        self.assertLess(time.monotonic() - before, 0.4)
        held.finish()

    def test_pending_count_and_bytes_both_bound_placeholder_requests(self):
        for config, first, rejected in (
            (dict(max_pending=1), dict(ram_bytes=3), dict(ram_bytes=0)),
            (dict(max_pending=4, max_pending_bytes=10), dict(ram_bytes=4, output_bytes=3), dict(resource='cpu', output_bytes=4)),
        ):
            with self.subTest(config=config):
                self.start(**config)
                self.unknown()
                lease, future = self.submit(**first)
                self.pending()
                self.assertEqual(self.client.snapshot()['pending_bytes'], sum(first.values()))
                with self.assertRaisesRegex(a.AdmissionError, 'pending queue budget exceeded'):
                    self.client.lease(**rejected).__enter__()
                self.assertEqual(self.client.snapshot()['pending'], 1)
                lease.cancel()
                with self.assertRaises((a.AdmissionError, OSError, EOFError)):
                    future.result(1)
                self.pending(0)
                self.broker.close()
                self.broker = None
                self.sampler.mode = 'valid'

    def test_equal_pending_requests_cancel_second_without_losing_first(self):
        self.start(max_pending=2, max_pending_bytes=20)
        self.unknown()
        first, granted = self.submit(resource='cpu', ram_bytes=5, output_bytes=5)
        self.pending()
        second, cancelled = self.submit(resource='cpu', ram_bytes=5, output_bytes=5)
        self.pending(2)
        self.assertEqual(self.client.snapshot()['pending_bytes'], 20)
        second.cancel()
        with self.assertRaises((a.AdmissionError, OSError, EOFError)):
            cancelled.result(1)
        self.pending()
        self.assertEqual(self.client.snapshot()['pending_bytes'], 10)
        self.sampler.mode = 'valid'
        granted.result(1).finish()
        self.pending(0)
        self.assertEqual(self.client.snapshot()['active'], 0)
        self.assertEqual(sum(r['event'] == 'acquire' for r in self.rows()), 1)

    def test_repeated_unknown_samples_do_not_reset_recovery_deadline(self):
        self.start(sample_max_age=0.14)
        self.unknown()
        _, future = self.submit(resource='cpu')
        self.pending()
        wait = next(r for r in self.rows() if r['event'] == 'telemetry_recovery_wait')
        time.sleep(0.06)
        self.sampler.mode = 'invalid'
        with self.assertRaisesRegex(a.AdmissionTimeout, 'readiness deadline expired'):
            future.result(0.6)
        self.assertLess(time.monotonic(), wait['recovery_deadline'] + 0.3)
        self.assertEqual(sum(r['event'] == 'telemetry_recovery_wait' for r in self.rows()), 1)
        self.assertFalse(any(r['event'] == 'acquire' for r in self.rows()))

    def test_already_granted_work_is_not_replayed_on_unknown(self):
        self.start()
        lease = self.client.lease(resource='cpu').__enter__()
        self.leases.append(lease)
        self.unknown()
        self.assertEqual(self.client.snapshot()['active'], 1)
        lease.finish()
        self.assertEqual(self.client.snapshot()['active'], 0)
        self.sampler.mode = 'valid'
        eventually(lambda: self.client.snapshot()['healthy'])
        self.assertEqual(sum(r['event'] == 'acquire' for r in self.rows()), 1)
        self.assertFalse(any(r['event'] == 'telemetry_recovery_wait' for r in self.rows()))

    def test_zero_pending_budget_rejects_unknown_immediately(self):
        self.start(max_pending=0)
        self.unknown()
        with self.assertRaisesRegex(a.AdmissionError, 'pending queue budget exceeded'):
            self.client.lease().__enter__()
        self.assertFalse(any(r['event'] == 'telemetry_recovery_wait' for r in self.rows()))

    def test_cancel_disconnect_epoch_and_close_do_not_grant(self):
        for mode in ('cancel', 'disconnect', 'epoch', 'close'):
            with self.subTest(mode=mode):
                self.start()
                self.unknown()
                lease, future = self.submit()
                self.pending()
                if mode == 'cancel':
                    lease.cancel()
                elif mode == 'disconnect':
                    with lease._lock:
                        lease._conn.shutdown(socket.SHUT_RDWR)
                elif mode == 'epoch':
                    original = a._epoch
                    with mock.patch.object(a, '_epoch', side_effect=lambda pid: 'expired' if pid == os.getpid() else original(pid)):
                        with self.broker._cv:
                            self.broker._cv.notify_all()
                        with self.assertRaisesRegex(a.AdmissionError, 'worker epoch expired'):
                            future.result(1)
                else:
                    self.broker.close()
                if mode != 'epoch':
                    with self.assertRaises((a.AdmissionError, OSError, EOFError)):
                        future.result(1)
                eventually(lambda: not self.broker._pending)
                self.assertFalse(self.broker._leases)
                self.assertEqual(self.broker._pending, [])
                self.broker.close()
                self.broker = None
                self.sampler.mode = 'valid'

    def test_external_pid_still_blocks_after_recovery(self):
        self.start()
        child = subprocess.Popen([sys.executable, '-B', '-S', '-c', 'import time; time.sleep(10)'])
        try:
            self.unknown()
            self.sampler.row['processes'] = [dict(pid=child.pid, used_bytes=20)]
            _, future = self.submit(timeout=0.18)
            self.pending()
            self.sampler.mode = 'valid'
            eventually(lambda: self.client.snapshot()['healthy'])
            with self.assertRaises(a.AdmissionTimeout):
                future.result(1)
            self.assertFalse(any(r['event'] == 'acquire' for r in self.rows()))
        finally:
            child.terminate()
            child.wait(2)

    def test_fatal_fence_logger_ram_immediate_even_with_sample_error(self):
        for flag in ('_fence_error', '_ram_observation_error', 'log'):
            with self.subTest(flag=flag):
                self.start()
                self.unknown()
                with self.broker._cv:
                    if flag == 'log':
                        self.broker.log.error = 'telemetry write failed: PermissionError'
                    else:
                        setattr(self.broker, flag, 'persistent fatal test error')
                before = time.monotonic()
                with self.assertRaisesRegex(a.AdmissionError, '^telemetry unavailable; admission closed$'):
                    self.client.lease().__enter__()
                self.assertLess(time.monotonic() - before, 0.3)
                self.assertEqual(self.broker._pending, [])
                self.assertFalse(any(r['event'] == 'telemetry_recovery_wait' for r in self.rows()))
                self.broker.close()
                self.broker = None
                self.sampler.mode = 'valid'

    def test_fatal_errors_interrupt_existing_recovery_wait(self):
        for flag in ('_fence_error', '_ram_observation_error', 'log'):
            with self.subTest(flag=flag):
                self.start()
                self.unknown()
                _, future = self.submit()
                self.pending()
                with self.broker._cv:
                    if flag == 'log':
                        self.broker.log.error = 'telemetry write failed: OSError'
                    else:
                        setattr(self.broker, flag, 'persistent fatal test error')
                    self.broker._cv.notify_all()
                with self.assertRaisesRegex(a.AdmissionError, '^telemetry unavailable; admission closed$'):
                    future.result(1)
                self.assertEqual(self.broker._pending, [])
                self.broker.close()
                self.broker = None
                self.sampler.mode = 'valid'

    def test_actual_logger_append_failure_not_recoverable(self):
        self.start()
        self.unknown()
        self.events.unlink()
        self.events.mkdir()
        with self.assertRaisesRegex(a.AdmissionError, 'telemetry write failed'):
            self.client.lease().__enter__()
        self.assertTrue(self.broker.log.error)
        self.assertEqual(self.broker._pending, [])

    def test_fence_commit_failure_retains_original_error(self):
        self.start()
        with mock.patch.object(self.broker, '_store_fences', side_effect=PermissionError('test')):
            with self.assertRaisesRegex(a.AdmissionError, '^persistent GPU fencing failed: PermissionError$'):
                self.client.lease().__enter__()
        self.assertFalse(self.broker._leases)
        self.assertEqual(self.broker._pending, [])

    def test_recovered_request_still_checks_affinity_and_capacity(self):
        for kwargs, error in ((dict(visible_devices='GPU-absent'), 'ambiguous client CUDA visibility'),
                              (dict(scratch_bytes=801), 'request exceeds safe VRAM budget')):
            with self.subTest(kwargs=kwargs):
                self.start()
                self.unknown()
                _, future = self.submit(**kwargs)
                self.pending()
                self.sampler.mode = 'valid'
                with self.assertRaisesRegex(a.AdmissionError, error):
                    future.result(1)
                self.broker.close()
                self.broker = None

    def test_default_leases_and_registration_remain_fail_fast(self):
        self.start(telemetry_recovery=False)
        self.unknown()
        for resource in ('gpu', 'cpu'):
            with self.assertRaisesRegex(a.AdmissionError, '^telemetry unavailable; admission closed$'):
                self.client.lease(resource=resource).__enter__()
        self.broker.telemetry_recovery = True
        for call in (lambda: self.client.register_model('not-retried'),):
            with self.assertRaisesRegex(a.AdmissionError, '^telemetry unavailable; admission closed$'):
                call()
        self.assertEqual(self.broker._pending, [])

    def test_option_requires_boolean_and_defaults_false(self):
        for value in (1, 'true', None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'telemetry_recovery must be boolean'):
                a.ResourceBroker(dict(self.config, telemetry_recovery=value), self.sampler)
        config = dict(self.config)
        del config['telemetry_recovery']
        self.assertFalse(a.ResourceBroker(config, self.sampler).telemetry_recovery)


if __name__ == '__main__':
    unittest.main(verbosity=2)
