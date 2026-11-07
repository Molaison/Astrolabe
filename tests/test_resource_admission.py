"""CPU-only standard-library acceptance: python3.11 tests/test_resource_admission.py.

Load the runtime as an isolated package so these tests need no scientific stack
and do not change the real astrolabe package's imports for other test modules.
"""
from __future__ import annotations

import concurrent.futures
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import select
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
import uuid


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_astrolabe_resource_acceptance"
spec = importlib.util.spec_from_file_location(
    PACKAGE, ROOT / "astrolabe/runtime/__init__.py",
    submodule_search_locations=[str(ROOT / "astrolabe/runtime")])
runtime = importlib.util.module_from_spec(spec)
sys.modules[PACKAGE] = runtime
spec.loader.exec_module(runtime)
ResourceBroker = runtime.ResourceBroker
ResourceClient = runtime.ResourceClient
AdmissionError = runtime.AdmissionError
AdmissionTimeout = runtime.AdmissionTimeout
BrokerClosed = runtime.BrokerClosed
BrokerBusy = runtime.BrokerBusy
admission = sys.modules[PACKAGE + ".admission"]
telemetry = sys.modules[PACKAGE + ".telemetry"]


class Sampler:
    def __init__(self, device):
        self.row = dict(index=0, uuid=device, total_bytes=1000, free_bytes=1000,
                        utilization=55, processes=[])
        self.error = False

    def __call__(self):
        if self.error:
            raise RuntimeError("injected sample failure")
        return [copy.deepcopy(self.row)]


def eventually(predicate, timeout=3):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition did not become true")


class ResourceAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="arb-")
        self.addCleanup(self.temp.cleanup)
        self.env = mock.patch.dict(os.environ)
        self.env.start()
        self.addCleanup(self.env.stop)
        for key in ("CUDA_VISIBLE_DEVICES", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "ASTROLABE_RESOURCE_SOCKET"):
            os.environ.pop(key, None)
        self.uuid = "GPU-test-" + uuid.uuid4().hex
        self.sampler = Sampler(self.uuid)
        self.config = dict(devices=[dict(uuid=self.uuid, index=0, vram_bytes=800)],
                           socket_dir=self.temp.name, cpu_slots=2, ram_bytes=1000,
                           output_bytes=1000, max_pending_bytes=10000,
                           sample_interval=0.02, sample_max_age=0.5,
                           rpc_timeout=0.5, shutdown_timeout=2,
                           max_concurrency=3, min_observations=2,
                           observation_window=0.02)
        self.brokers = []
        self.addCleanup(self.cleanup_brokers)

    def cleanup_brokers(self):
        for broker in reversed(self.brokers):
            broker.close()
            self.assertFalse(broker._threads)
            self.assertFalse(broker._leases)
            self.assertFalse(broker._models)
            self.assertFalse(broker._sockets)

    def start(self, **overrides):
        config = dict(self.config, **overrides)
        broker = ResourceBroker(config, self.sampler).start()
        self.brokers.append(broker)
        return broker, ResourceClient(broker.socket_path)

    def pending(self, client, count=1):
        eventually(lambda: client.snapshot()["pending"] == count)

    def test_clients_contend_without_blocking_snapshot_or_cpu(self):
        broker, first = self.start()
        other = ResourceClient(broker.socket_path)
        held = first.lease(scratch_bytes=600).__enter__()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(lambda: other.lease(scratch_bytes=600, timeout=2).__enter__())
            self.pending(first)
            self.assertEqual(first.snapshot()["active"], 1)
            with first.lease(resource="cpu", ram_bytes=100):
                self.assertEqual(first.snapshot()["active"], 2)
            held.finish()
            lease = future.result(2)
            self.assertTrue(lease.token)
            self.assertEqual(lease.device, self.uuid)
            lease.finish()
        self.assertEqual(first.snapshot()["active"], 0)

    def test_load_transfer_and_independent_resident_transient_release(self):
        _, client = self.start()
        with client.lease(phase="load", resident_bytes=300, scratch_bytes=100,
                          ram_bytes=20, output_bytes=30):
            registered = client.register_model("model", resident_bytes=300,
                                               metadata={"backend": "test"})
            self.assertEqual(registered["device"], self.uuid)
            self.assertEqual(client.snapshot()["reserved_vram_bytes"], 400)
        state = client.snapshot()
        self.assertEqual(state["reserved_vram_bytes"], 300)
        self.assertEqual(state["ram_bytes"], 0)
        self.assertEqual(state["output_bytes"], 0)
        with client.lease(scratch_bytes=200, ram_bytes=50, output_bytes=60):
            self.assertEqual(client.snapshot()["reserved_vram_bytes"], 500)
            with self.assertRaises(AdmissionError):
                client.unregister_model("model")
        self.assertEqual(client.snapshot()["resident_models"], 1)
        client.unregister_model("model")
        self.assertEqual(client.snapshot()["reserved_vram_bytes"], 0)

    def test_unknown_serial_and_zero_residency_registration(self):
        _, client = self.start()
        first = client.lease().__enter__()
        self.assertEqual(client.snapshot()["reserved_vram_bytes"], 800)
        with self.assertRaises(AdmissionTimeout):
            with client.lease(backend="different", timeout=0.05):
                self.fail("unknown GPU reservation overlapped")
        first.finish()
        with client.lease(phase="load"):
            client.register_model("unknown")
            self.assertEqual(client.snapshot()["reserved_vram_bytes"], 800)
        with client.lease():
            self.assertEqual(client.snapshot()["reserved_vram_bytes"], 800)
        client.unregister_model("unknown")

    def test_noncertified_stays_serial_and_no_ready_work_does_not_grow(self):
        _, client = self.start()
        for _ in range(4):
            with client.lease(scratch_bytes=10, concurrency_certified=True):
                time.sleep(0.012)
        self.assertEqual(client.snapshot()["controllers"][0]["limit"], 1)
        with concurrent.futures.ThreadPoolExecutor() as pool:
            first = client.lease(scratch_bytes=10).__enter__()
            other = pool.submit(lambda: client.lease(scratch_bytes=10, timeout=2).__enter__())
            self.pending(client)
            time.sleep(0.025)
            first.finish()
            second = other.result(2)
            self.assertEqual(client.snapshot()["controllers"][0]["limit"], 1)
            second.finish()

    def train_growth(self, client):
        # Keep actual ready requests behind each completed sample; no artificial
        # throughput or concurrency values are supplied to the controller.
        with client.lease(scratch_bytes=100, concurrency_certified=True):
            time.sleep(0.015)
        first = client.lease(scratch_bytes=100, concurrency_certified=True).__enter__()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(lambda: client.lease(scratch_bytes=100, concurrency_certified=True, timeout=2).__enter__())
            self.pending(client)
            time.sleep(0.015)
            first.finish()
            second = future.result(2)
            time.sleep(0.015)
            second.finish()

    def test_certified_expansion_cap_and_oom_shrink(self):
        _, client = self.start()
        self.train_growth(client)
        before = client.snapshot()["controllers"][0]["limit"]
        self.assertGreaterEqual(before, 2)
        first = client.lease(scratch_bytes=100, concurrency_certified=True).__enter__()
        second = client.lease(scratch_bytes=100, concurrency_certified=True, timeout=0).__enter__()
        self.assertEqual(client.snapshot()["active"], 2)
        first.finish(error="CUDA out of memory", oom=True)
        second.finish()
        self.assertEqual(client.snapshot()["controllers"][0]["limit"], max(1, before // 2))
        self.assertLessEqual(before, 3)

    def test_throughput_regression_shrinks(self):
        _, client = self.start()
        self.train_growth(client)
        before = client.snapshot()["controllers"][0]["limit"]
        self.assertGreaterEqual(before, 2)
        # Same identity/shape/operation, substantially slower measured completions.
        for _ in range(2):
            with client.lease(scratch_bytes=100, concurrency_certified=True):
                time.sleep(0.12)
        state = client.snapshot()["controllers"][0]
        self.assertEqual(state["limit"], max(1, before // 2))
        self.assertGreater(state["completions"], 0)
        self.assertIsNotNone(state["throughput"])

    def test_controller_identity_shape_operation_are_isolated(self):
        _, client = self.start()
        self.train_growth(client)
        first = client.lease(backend="a", identity="w1", shape="small", scratch_bytes=10, concurrency_certified=True).__enter__()
        for changes in (dict(backend="b"), dict(identity="w2"), dict(shape="large"), dict(phase="prepare")):
            args = dict(backend="a", identity="w1", shape="small", scratch_bytes=10, concurrency_certified=True, timeout=0)
            args.update(changes)
            with client.lease(**args):
                pass
        first.finish()
        self.assertEqual(len(client.snapshot()["controllers"]), 6)

    def test_pending_count_and_byte_budgets(self):
        _, client = self.start(max_pending=1, max_pending_bytes=70)
        held = client.lease(resource="cpu").__enter__()
        held2 = client.lease(resource="cpu").__enter__()
        queued = client.lease(resource="cpu", ram_bytes=50, timeout=1)
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(queued.__enter__)
            self.pending(client)
            self.assertEqual(client.snapshot()["pending_bytes"], 50)
            with self.assertRaisesRegex(AdmissionError, "pending queue budget"):
                client.lease(resource="cpu", ram_bytes=30).__enter__()
            queued.cancel()
            with self.assertRaises((AdmissionError, EOFError, OSError)):
                future.result(2)
        self.pending(client, 0)
        with self.assertRaisesRegex(AdmissionError, "pending queue budget"):
            client.lease(resource="cpu", ram_bytes=71).__enter__()
        held.finish()
        held2.finish()

    def test_timeout_cancel_exception_and_double_finish_release(self):
        _, client = self.start()
        first = client.lease(scratch_bytes=800).__enter__()
        with self.assertRaises(AdmissionTimeout):
            client.lease(scratch_bytes=10, timeout=0.03).__enter__()
        self.pending(client, 0)
        first.cancel()
        eventually(lambda: client.snapshot()["fenced"] == 1)
        self.assertEqual(client.snapshot()["active"], 1)
        first.finish()  # Simulation is now actually stopped; reconnect with token.
        eventually(lambda: client.snapshot()["active"] == 0)
        errored = client.lease(scratch_bytes=100)
        with self.assertRaisesRegex(ValueError, "science error"):
            with errored:
                raise ValueError("science error")
        eventually(lambda: client.snapshot()["fenced"] == 1)
        errored.finish(error="science error, device synchronization completed")
        self.assertEqual(client.snapshot()["active"], 0)
        lease = client.lease(scratch_bytes=10).__enter__()
        lease.finish()
        lease.finish()
        lease.cancel()
        with self.assertRaises(AdmissionError):
            lease.__enter__()

    def test_external_busy_distinguished_from_usable_free_vram(self):
        self.sampler.row.update(free_bytes=500, utilization=99,
                                processes=[dict(pid=123456789, used_bytes=500)])
        _, client = self.start()
        with self.assertRaises(AdmissionTimeout):
            client.lease(timeout=0.03).__enter__()
        with client.lease(scratch_bytes=200, timeout=0):
            self.assertEqual(client.snapshot()["active"], 1)
        with self.assertRaises(AdmissionTimeout):
            client.lease(scratch_bytes=600, timeout=0.03).__enter__()
        self.sampler.row.update(free_bytes=1000, processes=[])
        eventually(lambda: client.snapshot()["devices"][self.uuid]["sample"]["free_bytes"] == 1000)
        with client.lease(timeout=0.1):
            pass

    def test_free_sample_does_not_double_reserve_unmaterialized_leases(self):
        _, client = self.start()
        first = client.lease(backend="one", scratch_bytes=500, concurrency_certified=True).__enter__()
        with self.assertRaises(AdmissionTimeout):
            client.lease(backend="two", scratch_bytes=400, concurrency_certified=True, timeout=0.02).__enter__()
        first.finish()

    def test_multiple_owners_same_physical_gpu_rejected(self):
        _, client = self.start()
        other = ResourceBroker(dict(self.config), self.sampler)
        with self.assertRaisesRegex(AdmissionError, "already owned"):
            other.start()
        self.assertTrue(client.snapshot()["healthy"])
        other.close()
        self.brokers[0].close()
        self.start()

    def test_visibility_and_physical_identity_constraints(self):
        for env, mask in (("CUDA_VISIBLE_DEVICES", "1"), ("SLURM_JOB_GPUS", "1"),
                          ("SLURM_STEP_GPUS", ""), ("CUDA_VISIBLE_DEVICES", "MIG-x")):
            with self.subTest(env=env, mask=mask), mock.patch.dict(os.environ, {env: mask}):
                with self.assertRaises(AdmissionError):
                    ResourceBroker(self.config, self.sampler).start()
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": self.uuid}):
            self.start()

    def test_sampler_failure_and_recovery_close_admission(self):
        _, client = self.start()
        held = client.lease(scratch_bytes=800).__enter__()
        queued = client.lease(scratch_bytes=10, timeout=1)
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(queued.__enter__)
            self.pending(client)
            self.sampler.error = True
            with self.assertRaisesRegex(AdmissionError, "telemetry unavailable"):
                future.result(2)
        self.assertFalse(client.snapshot()["healthy"])
        with self.assertRaises(AdmissionError):
            client.lease(resource="cpu").__enter__()
        held.finish()
        self.assertEqual(client.snapshot()["active"], 0)
        self.sampler.error = False
        eventually(lambda: client.snapshot()["healthy"])
        with client.lease(scratch_bytes=1):
            pass

    def test_incomplete_or_stale_async_state_is_not_idle(self):
        broker, client = self.start()
        del self.sampler.row["processes"]
        eventually(lambda: not client.snapshot()["healthy"])
        with self.assertRaises(AdmissionError):
            client.lease(scratch_bytes=1).__enter__()
        self.sampler.row["processes"] = []
        eventually(lambda: client.snapshot()["healthy"])
        with broker._cv:
            broker._sample_at = time.monotonic() - 10
            with self.assertRaises(AdmissionError):
                broker._healthy()

    def test_jsonl_failure_never_leaks_and_latches_closed(self):
        path = Path(self.temp.name) / "telemetry.jsonl"
        broker, client = self.start(telemetry_path=str(path))
        held = client.lease(scratch_bytes=20).__enter__()
        try:
            # Exclude the sampler's append while replacing the file: otherwise
            # it can recreate the file before the intended I/O failure exists.
            with broker.log._lock:
                path.unlink()
                path.mkdir()
        finally:
            held.finish()
        state = client.snapshot()
        self.assertEqual(state["active"], 0)
        self.assertFalse(state["healthy"])
        with self.assertRaises(AdmissionError):
            client.register_model("blocked", resident_bytes=10)

    def test_telemetry_jsonl_contains_lifecycle_evidence(self):
        path = Path(self.temp.name) / "events.jsonl"
        _, client = self.start(telemetry_path=str(path))
        with client.lease(scratch_bytes=10):
            pass
        events = [json.loads(line)["event"] for line in path.read_text().splitlines()]
        self.assertEqual([event for event in events if event != "sample"], ["start", "acquire", "release"])

    def test_close_rejects_queue_but_retains_live_compute_until_completion(self):
        broker, client = self.start(shutdown_timeout=0.1)
        held = client.lease(scratch_bytes=800).__enter__()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(lambda: client.lease(scratch_bytes=1).__enter__())
            self.pending(client)
            with self.assertRaisesRegex(BrokerBusy, "UUID locks.*retained"):
                broker.close()
            with self.assertRaises((AdmissionError, EOFError, OSError)):
                future.result(1)
        self.assertEqual(client.snapshot()["active"], 1)
        self.assertTrue(broker._locks)
        self.assertFalse(broker._stop.is_set())
        held.finish()
        broker.close()
        self.assertFalse(Path(broker.socket_path).exists())
        self.assertFalse(broker._threads)
        self.assertEqual(broker._pending, [])
        self.assertNotIn("ASTROLABE_RESOURCE_SOCKET", os.environ)
        with self.assertRaises(BrokerClosed):
            broker.start()

    def test_process_exit_reclaims_models_and_worker_epoch(self):
        broker, client = self.start()
        code = (
            "import importlib.util,sys,time; "
            f"s=importlib.util.spec_from_file_location('arb_child',{str(ROOT / 'astrolabe/runtime/__init__.py')!r},"
            f"submodule_search_locations={[str(ROOT / 'astrolabe/runtime')]!r}); "
            "m=importlib.util.module_from_spec(s);sys.modules['arb_child']=m;s.loader.exec_module(m); "
            f"c=m.ResourceClient({broker.socket_path!r}); "
            "c.register_model('child',resident_bytes=100); "
            "lease=c.lease(scratch_bytes=100).__enter__(); "
            "print('ready',flush=True); time.sleep(30)"
        )
        process = subprocess.Popen([sys.executable, "-B", "-S", "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            self.assertEqual(process.stdout.readline().strip(), "ready")
            self.assertEqual(client.snapshot()["resident_models"], 1)
            with self.assertRaises(AdmissionError):
                client.unregister_model("child")
            with self.assertRaises(AdmissionError):
                client.register_model("child", resident_bytes=10)
            process.kill()
            process.communicate(timeout=3)
            eventually(lambda: client.snapshot()["resident_models"] == 0 and client.snapshot()["active"] == 0)
            self.assertIsNotNone(process.returncode)
        finally:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=3)

    def test_wrong_uid_frame_limits_and_unsafe_directory(self):
        broker, client = self.start()
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
            conn.connect(broker.socket_path)
            conn.sendall(struct.pack("!I", 1000000))
            reply = admission._receive(conn, 65536)
            self.assertFalse(reply["ok"])
        eventually(lambda: not broker._sockets)
        with mock.patch.object(admission.os, "getuid", return_value=os.getuid() + 1):
            with self.assertRaises(AdmissionError):
                client.snapshot()
        broker.close()
        os.chmod(self.temp.name, 0o755)
        with self.assertRaises(AdmissionError):
            ResourceBroker(self.config, self.sampler).start()
        os.chmod(self.temp.name, 0o700)

    def test_cpu_only_and_env_inheritance(self):
        self.assertIsNone(ResourceClient.from_env())
        broker, client = self.start(devices=[])
        self.assertEqual(ResourceClient.from_env().socket_path, broker.socket_path)
        result = subprocess.run([sys.executable, "-S", "-c", "import os;print(os.environ['ASTROLABE_RESOURCE_SOCKET'])"],
                                check=True, text=True, capture_output=True, timeout=2)
        self.assertEqual(result.stdout.strip(), broker.socket_path)
        self.assertEqual(client.snapshot()["devices"], {})
        with client.lease(resource="cpu"):
            pass
        with self.assertRaises(AdmissionError):
            client.lease().__enter__()

    def test_invalid_requests_do_not_leak(self):
        _, client = self.start()
        for kwargs in (dict(resource="bad"), dict(ram_bytes=-1), dict(output_bytes=1001),
                       dict(scratch_bytes=801), dict(concurrency_certified="true"),
                       dict(timeout=-1)):
            with self.subTest(kwargs=kwargs), self.assertRaises(AdmissionError):
                client.lease(**kwargs).__enter__()
        self.assertEqual(client.snapshot()["active"], 0)
        self.assertEqual(client.snapshot()["pending"], 0)

    def test_idle_graphics_baseline_unknown_uses_actual_free_capacity(self):
        self.sampler.row.update(free_bytes=549, utilization=20, processes=[])
        _, client = self.start()
        with client.lease(phase="load", timeout=0):
            self.assertEqual(client.snapshot()["reserved_vram_bytes"], 549)
            client.register_model(f"weights:{os.getpid()}:epoch", device="cuda:0")
        # The 451-byte driver/Xorg baseline never appeared in compute-apps.
        self.sampler.row.update(free_bytes=349, processes=[dict(pid=os.getpid(), used_bytes=200)])
        eventually(lambda: client.snapshot()["models"][0]["observed"])
        self.assertEqual(client.snapshot()["reserved_vram_bytes"], 200)
        with client.lease(phase="execute", timeout=0.5):
            self.assertEqual(client.snapshot()["reserved_vram_bytes"], 549)
        client.unregister_model(f"weights:{os.getpid()}:epoch")

    def test_pending_host_bytes_exclude_gpu_resident_and_scratch(self):
        _, client = self.start(max_pending_bytes=1)
        held = client.lease().__enter__()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(lambda: client.lease(phase="load", timeout=2).__enter__())
            self.pending(client)
            self.assertEqual(client.snapshot()["pending_bytes"], 0)
            held.finish()
            future.result(2).finish()

    def test_unknown_residency_can_be_observed_during_continuous_compute(self):
        _, client = self.start()
        with client.lease(phase="load"):
            client.register_model(f"continuous:{os.getpid()}:epoch")
        # No idle sampler tick is available before the next execute operation.
        held = client.lease(phase="execute").__enter__()
        self.sampler.row.update(free_bytes=800, processes=[dict(pid=os.getpid(), used_bytes=200)])
        eventually(lambda: client.snapshot()["models"][0]["observed"])
        self.assertEqual(client.snapshot()["models"][0]["resident_bytes"], 200)
        with self.assertRaises(AdmissionTimeout):
            client.lease(backend="other", scratch_bytes=10, concurrency_certified=True, timeout=0.02).__enter__()
        held.finish()
        with client.lease(phase="load", resident_bytes=300, timeout=0.5):
            self.assertEqual(client.snapshot()["resident_models"], 1)
        client.unregister_model(f"continuous:{os.getpid()}:epoch")

    def test_cpu_preparation_with_gpu_declarations_keeps_resident_budget(self):
        _, client = self.start()
        client.register_model("weights", resident_bytes=500)
        with client.lease(resource="cpu", phase="prepare", resident_bytes=500, scratch_bytes=200):
            self.assertEqual(client.snapshot()["reserved_vram_bytes"], 500)
        with client.lease(phase="execute", resident_bytes=500, scratch_bytes=200, timeout=0):
            self.assertEqual(client.snapshot()["reserved_vram_bytes"], 700)
        client.unregister_model("weights")

    def test_client_affinity_cannot_fall_back_to_second_gpu(self):
        second_uuid = "GPU-test-" + uuid.uuid4().hex
        second = dict(index=1, uuid=second_uuid, total_bytes=1000,
                      free_bytes=1000, utilization=0, processes=[])
        self.sampler = lambda: [dict(self.row_copy), dict(second)]
        self.row_copy = Sampler(self.uuid).row
        devices = [dict(index=1, uuid=second_uuid, vram_bytes=800), *self.config["devices"]]
        _, client = self.start(devices=devices)
        held = client.lease().__enter__()
        self.assertEqual(held.device, self.uuid)
        # The same unqualified key on a different physical device is independent.
        with client.lease(device="cuda:1", scratch_bytes=10, timeout=0):
            self.assertEqual(client.snapshot()["active"], 2)
        with self.assertRaises(AdmissionTimeout):
            client.lease(backend="other", timeout=0.03).__enter__()
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": f"{second_uuid},{self.uuid}"}):
            with client.lease(backend="second", phase="load", timeout=0) as load:
                self.assertEqual(load.device, second_uuid)
                client.register_model("mapped", device="cuda:0", resident_bytes=100)
            self.assertEqual(client.snapshot()["models"][0]["device"], second_uuid)
            client.unregister_model("mapped")
        held.finish()
        with client.lease(device="cuda:1", timeout=0) as lease:
            self.assertEqual(lease.device, second_uuid)
        with self.assertRaises(AdmissionError):
            client.lease(visible_devices="MIG-invalid").__enter__()

    def test_model_id_tracks_worker_not_registering_client_and_parent_can_reap(self):
        _, client = self.start()
        child = subprocess.Popen([sys.executable, "-S", "-c", "import time;time.sleep(30)"])
        try:
            model_id = f"kernel:{child.pid}:worker-epoch"
            client.register_model(model_id, resident_bytes=100)
            model = client.snapshot()["models"][0]
            self.assertEqual(model["pid"], child.pid)
            self.assertEqual(model["worker_epoch"], "worker-epoch")
            with self.assertRaises(AdmissionError):
                client.unregister_model(model_id)
            child.kill()
            child.wait(timeout=2)
            client.unregister_model(model_id)
            self.assertEqual(client.snapshot()["resident_models"], 0)
        finally:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=2)

    def test_observed_unknown_workers_can_be_co_resident_and_compute(self):
        _, client = self.start()
        children = [subprocess.Popen([sys.executable, "-S", "-c", "import time;time.sleep(30)"]) for _ in range(2)]
        try:
            first_id = f"first:{children[0].pid}:one"
            second_id = f"second:{children[1].pid}:two"
            client.register_model(first_id)
            with self.assertRaises(AdmissionError):
                client.register_model(second_id)
            self.sampler.row.update(free_bytes=700, processes=[dict(pid=children[0].pid, used_bytes=300)])
            eventually(lambda: client.snapshot()["models"][0]["observed"])
            self.assertEqual(client.snapshot()["reserved_vram_bytes"], 300)
            client.register_model(second_id)
            self.sampler.row.update(free_bytes=500, processes=[dict(pid=children[0].pid, used_bytes=300),
                                                            dict(pid=children[1].pid, used_bytes=200)])
            eventually(lambda: all(model["observed"] for model in client.snapshot()["models"]))
            self.assertEqual(client.snapshot()["reserved_vram_bytes"], 500)
            with client.lease(phase="execute", timeout=0.5):
                self.assertEqual(client.snapshot()["resident_models"], 2)
                self.assertEqual(client.snapshot()["reserved_vram_bytes"], 800)
        finally:
            for child in children:
                child.kill()
                child.wait(timeout=2)

    def test_cancel_race_does_not_return_an_unowned_lease(self):
        _, client = self.start()
        lease = client.lease(scratch_bytes=1)
        original = client._check

        def cancel_on_reply(reply):
            result = original(reply)
            lease.cancel()
            return result

        with mock.patch.object(client, "_check", side_effect=cancel_on_reply):
            with self.assertRaisesRegex(AdmissionError, "cancelled during admission"):
                lease.__enter__()
        eventually(lambda: client.snapshot()["fenced"] == 1)
        lease.finish()  # No simulated GPU work started after the rejected enter.
        eventually(lambda: client.snapshot()["active"] == 0)

    def disconnected_worker(self, socket_path):
        code = f'''
import importlib.util,json,os,sys
s=importlib.util.spec_from_file_location('arb_child', {str(ROOT / 'astrolabe/runtime/__init__.py')!r}, submodule_search_locations={[str(ROOT / 'astrolabe/runtime')]!r})
m=importlib.util.module_from_spec(s)
sys.modules['arb_child']=m
s.loader.exec_module(m)
c=m.ResourceClient({socket_path!r})
model_id=f'live:{{os.getpid()}}:worker-epoch'
with c.lease(phase='load',resident_bytes=100):
    c.register_model(model_id,device='cuda:0',resident_bytes=100)
lease=c.lease(phase='execute',scratch_bytes=100).__enter__()
lease.cancel()
print(json.dumps(dict(token=lease.token,model_id=model_id)),flush=True)
for command in sys.stdin:
    if command.strip()=='step':
        # This represents a device operation that outlives the network request.
        print('gpu-step-after-disconnect',flush=True)
    elif command.strip()=='complete':
        lease.finish()
        c.unregister_model(model_id)
        print('device-synchronized-and-released',flush=True)
    elif command.strip()=='exit':
        os._exit(0)
'''
        process = subprocess.Popen([sys.executable, "-B", "-S", "-c", code],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)
        self.addCleanup(self.stop_worker, process)
        return process

    @staticmethod
    def stop_worker(process):
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=3)

    def worker_line(self, process, command=None):
        if command is not None:
            process.stdin.write(command + "\n")
            process.stdin.flush()
        self.assertTrue(select.select([process.stdout], [], [], 3)[0], "worker response deadline expired")
        line = process.stdout.readline().strip()
        self.assertTrue(line, "worker exited without a response")
        return line

    def assert_second_owner_rejected(self):
        # A fresh interpreter proves physical UUID exclusion is not just an
        # in-process registry or a lock on the original Unix socket pathname.
        code = f'''
import importlib.util,sys
s=importlib.util.spec_from_file_location('arb_contender', {str(ROOT / 'astrolabe/runtime/__init__.py')!r}, submodule_search_locations={[str(ROOT / 'astrolabe/runtime')]!r})
m=importlib.util.module_from_spec(s)
sys.modules['arb_contender']=m
s.loader.exec_module(m)
b=m.ResourceBroker({self.config!r},sampler=lambda:{self.sampler()!r})
try:
    b.start()
except m.AdmissionError as exc:
    if 'already owned' not in str(exc):
        raise
    print('uuid-lock-refused')
else:
    b.close()
    raise RuntimeError('live GPU owner was incorrectly replaced')
'''
        result = subprocess.run([sys.executable, "-B", "-S", "-c", code],
                                capture_output=True, text=True, timeout=3)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "uuid-lock-refused")

    def test_live_disconnected_gpu_worker_keeps_uuid_lock_until_process_exit(self):
        broker, client = self.start(shutdown_timeout=0.1)
        worker = self.disconnected_worker(broker.socket_path)
        json.loads(self.worker_line(worker))
        eventually(lambda: client.snapshot()["fenced"] == 1)
        self.assertEqual(self.worker_line(worker, "step"), "gpu-step-after-disconnect")
        self.assertEqual(client.snapshot()["reserved_vram_bytes"], 200)
        self.assert_second_owner_rejected()
        with self.assertRaises(BrokerBusy):
            broker.close()
        self.assertEqual(self.worker_line(worker, "step"), "gpu-step-after-disconnect")
        self.assertIsNone(worker.poll())
        self.assertEqual(client.snapshot()["active"], 1)
        self.assertEqual(client.snapshot()["resident_models"], 1)
        self.assert_second_owner_rejected()
        with self.assertRaises(BrokerClosed):
            client.lease(scratch_bytes=1).__enter__()
        worker.stdin.write("exit\n")
        worker.stdin.flush()
        worker.wait(timeout=3)
        eventually(lambda: client.snapshot()["active"] == 0 and client.snapshot()["resident_models"] == 0)
        broker.close()
        self.assertFalse(broker._locks)
        self.assertFalse(broker._threads)
        self.start()  # Only after actual worker exit and a successful close retry.

    def test_fenced_completion_requires_original_worker_epoch_and_can_drain(self):
        broker, client = self.start(shutdown_timeout=0.1)
        worker = self.disconnected_worker(broker.socket_path)
        ready = json.loads(self.worker_line(worker))
        eventually(lambda: client.snapshot()["fenced"] == 1)
        with self.assertRaisesRegex(AdmissionError, "another PID/start epoch"):
            client._rpc(dict(op="finish", token=ready["token"]))
        with self.assertRaises(BrokerBusy):
            broker.close()
        self.assertEqual(self.worker_line(worker, "complete"), "device-synchronized-and-released")
        self.assertIsNone(worker.poll())
        self.assertEqual(client.snapshot()["active"], 0)
        self.assertEqual(client.snapshot()["resident_models"], 0)
        broker.close()
        self.assertFalse(broker._locks)

    def test_close_retains_live_resident_model_without_active_compute(self):
        broker, client = self.start(shutdown_timeout=0.05)
        client.register_model("resident", resident_bytes=100)
        with self.assertRaisesRegex(BrokerBusy, "1 resident models"):
            broker.close()
        self.assert_second_owner_rejected()
        client.unregister_model("resident")  # Explicit assertion allocation freed.
        broker.close()
        self.assertFalse(broker._threads)

    def test_close_waits_for_real_completion_and_cpu_cancel_still_releases(self):
        broker, client = self.start()
        cpu = client.lease(resource="cpu").__enter__()
        cpu.cancel()
        eventually(lambda: client.snapshot()["active"] == 0)
        gpu = client.lease(scratch_bytes=100).__enter__()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            drained = pool.submit(broker.close)
            eventually(lambda: client.snapshot()["closed"])
            self.assertFalse(drained.done())
            gpu.finish()
            drained.result(3)
        self.assertFalse(broker._threads)

    def test_unreadable_process_state_is_not_evidence_of_gpu_completion(self):
        broker, client = self.start(shutdown_timeout=0.05)
        held = client.lease(scratch_bytes=100).__enter__()
        held.cancel()
        eventually(lambda: client.snapshot()["fenced"] == 1)
        with mock.patch.object(admission.Path, "read_text", side_effect=PermissionError("proc temporarily unreadable")):
            with self.assertRaises(BrokerBusy):
                broker.close()
            self.assertEqual(len(broker._leases), 1)
            self.assertTrue(broker._locks)
        held.finish()
        broker.close()

    def test_broker_process_crash_preserves_durable_orphan_fencing(self):
        code = f'''
import importlib.util,os,sys
s=importlib.util.spec_from_file_location('arb_crash_owner', {str(ROOT / 'astrolabe/runtime/__init__.py')!r}, submodule_search_locations={[str(ROOT / 'astrolabe/runtime')]!r})
m=importlib.util.module_from_spec(s)
sys.modules['arb_crash_owner']=m
s.loader.exec_module(m)
b=m.ResourceBroker({self.config!r},sampler=lambda:{self.sampler()!r}).start()
print(b.socket_path,flush=True)
sys.stdin.readline()
os._exit(0)
'''
        owner = subprocess.Popen([sys.executable, "-B", "-S", "-c", code],
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True)
        self.addCleanup(self.stop_worker, owner)
        socket_path = self.worker_line(owner)
        worker = self.disconnected_worker(socket_path)
        json.loads(self.worker_line(worker))
        path = Path("/tmp/astrolabe-gpu-" + hashlib.sha256(self.uuid.encode()).hexdigest() + ".state.json")
        before = json.loads(path.read_text())
        self.assertEqual({r["kind"] for r in before["records"]}, {"lease", "model"})
        self.assertEqual({r["pid"] for r in before["records"]}, {worker.pid})
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        owner.kill()  # No close(), no finally, no surviving broker threads/FDs.
        owner.wait(timeout=3)
        self.assertEqual(self.worker_line(worker, "step"), "gpu-step-after-disconnect")
        self.assertIsNone(worker.poll())
        replacement = ResourceBroker(self.config, self.sampler)
        with self.assertRaisesRegex(BrokerBusy, "orphan GPU owners.*startup refused"):
            replacement.start()
        replacement.close()
        self.assertEqual(json.loads(path.read_text()), before)
        worker.stdin.write("exit\n")
        worker.stdin.flush()
        worker.wait(timeout=3)
        broker, client = self.start()
        self.assertEqual(client.snapshot()["active"], 0)
        self.assertEqual(client.snapshot()["resident_models"], 0)
        self.assertEqual(json.loads(path.read_text())["records"], [])
        broker.close()

    def test_durable_fence_tracks_lease_and_resident_lifetimes(self):
        broker, client = self.start()
        path = Path(broker._fence_paths[self.uuid])
        with client.lease(phase="load", resident_bytes=100) as load:
            self.assertEqual({r["id"] for r in json.loads(path.read_text())["records"]}, {load.token})
            client.register_model("durable-model", resident_bytes=100)
            self.assertEqual(len(json.loads(path.read_text())["records"]), 2)
        records = json.loads(path.read_text())["records"]
        self.assertEqual([r["kind"] for r in records], ["model"])
        self.assertEqual(records[0]["pid"], os.getpid())
        self.assertEqual(records[0]["start_epoch"], admission._epoch(os.getpid()))
        with client.lease(scratch_bytes=100):
            self.assertEqual(len(json.loads(path.read_text())["records"]), 2)
        self.assertEqual(len(json.loads(path.read_text())["records"]), 1)
        client.unregister_model("durable-model")
        self.assertEqual(json.loads(path.read_text())["records"], [])
        self.assertEqual(list(path.parent.glob(path.name + ".tmp-*")), [])

    def test_durable_fence_write_failure_cannot_grant_or_forget_gpu_work(self):
        broker, client = self.start()
        path = Path(broker._fence_paths[self.uuid])
        with mock.patch.object(admission.os, "replace", side_effect=OSError("journal unavailable")):
            with self.assertRaisesRegex(AdmissionError, "persistent GPU fencing failed"):
                client.lease(scratch_bytes=100).__enter__()
        self.assertEqual(client.snapshot()["active"], 0)
        self.assertFalse(client.snapshot()["healthy"])
        self.assertFalse(path.exists())
        broker.close()
        broker, client = self.start()
        held = client.lease(scratch_bytes=100).__enter__()
        with mock.patch.object(admission.os, "replace", side_effect=OSError("journal unavailable")):
            with self.assertRaisesRegex(AdmissionError, "cleanup failed"):
                held.finish()
        self.assertEqual(client.snapshot()["active"], 1)
        self.assertEqual(len(json.loads(path.read_text())["records"]), 1)
        held.finish()  # Actual simulated completion; durable removal can now retry.
        self.assertEqual(json.loads(path.read_text())["records"], [])
        broker.close()

    def test_cpu_model_resident_ram_transfer_prevents_two_80_byte_loads(self):
        _, client = self.start(ram_bytes=100)
        with client.lease(resource="cpu", phase="load", ram_bytes=80):
            client.register_model("cpu-weights", device="cpu", resident_ram_bytes=80)
            snapshot = client.snapshot()
            self.assertEqual(snapshot["ram_bytes"], 80)
            self.assertEqual(snapshot["resident_ram_bytes"], 80)
            self.assertEqual(snapshot["transient_ram_bytes"], 0)
        self.assertEqual(client.snapshot()["active"], 0)
        self.assertEqual(client.snapshot()["ram_bytes"], 80)
        with self.assertRaises(AdmissionTimeout):
            client.lease(resource="cpu", phase="load", ram_bytes=80, timeout=0.03).__enter__()
        with self.assertRaisesRegex(AdmissionError, "resident RAM exceeds"):
            client.register_model("second", device="cpu", resident_ram_bytes=80)
        with client.lease(resource="cpu", phase="prepare", ram_bytes=20):
            self.assertEqual(client.snapshot()["ram_bytes"], 100)
        client.unregister_model("cpu-weights")
        with client.lease(resource="cpu", phase="load", ram_bytes=80):
            client.register_model("second", device="cpu", resident_ram_bytes=80)
        client.unregister_model("second")
        self.assertEqual(client.snapshot()["ram_bytes"], 0)

    def test_gpu_model_can_reserve_both_resident_vram_and_host_ram(self):
        _, client = self.start(ram_bytes=100)
        with client.lease(phase="load", resident_bytes=300, ram_bytes=80):
            client.register_model("mixed", resident_bytes=300, resident_ram_bytes=80)
        snapshot = client.snapshot()
        self.assertEqual(snapshot["reserved_vram_bytes"], 300)
        self.assertEqual(snapshot["ram_bytes"], 80)
        with client.lease(phase="execute", scratch_bytes=100, ram_bytes=20):
            self.assertEqual(client.snapshot()["ram_bytes"], 100)
            with self.assertRaises(AdmissionTimeout):
                client.lease(resource="cpu", ram_bytes=1, timeout=0.02).__enter__()
        client.unregister_model("mixed")
        self.assertEqual(client.snapshot()["ram_bytes"], 0)

    def test_resident_ram_is_owned_by_actual_worker_until_process_exit(self):
        _, client = self.start(ram_bytes=100)
        worker = subprocess.Popen([sys.executable, "-B", "-S", "-c", "import time;time.sleep(30)"])
        try:
            model_id = f"cpu-weights:{worker.pid}:worker-epoch"
            client.register_model(model_id, device="cpu", resident_ram_bytes=80)
            with self.assertRaises(AdmissionError):
                client.unregister_model(model_id)
            self.assertEqual(client.snapshot()["ram_bytes"], 80)
            worker.kill()
            worker.wait(timeout=3)
            client.unregister_model(model_id)
            self.assertEqual(client.snapshot()["ram_bytes"], 0)
        finally:
            if worker.poll() is None:
                worker.kill()
            worker.wait(timeout=3)

    def test_cpu_resident_ram_keeps_close_busy_until_explicit_release(self):
        broker, client = self.start(devices=[], ram_bytes=100, shutdown_timeout=0.05)
        client.register_model("cpu", device="cpu", resident_ram_bytes=80)
        with self.assertRaises(BrokerBusy):
            broker.close()
        self.assertEqual(client.snapshot()["ram_bytes"], 80)
        client.unregister_model("cpu")
        broker.close()

    def rss_worker(self):
        code = f'''
import gc,importlib.util,sys
s=importlib.util.spec_from_file_location('arb_rss_worker', {str(ROOT / 'astrolabe/runtime/__init__.py')!r}, submodule_search_locations={[str(ROOT / 'astrolabe/runtime')]!r})
m=importlib.util.module_from_spec(s)
sys.modules['arb_rss_worker']=m
s.loader.exec_module(m)
client=m.ResourceClient.from_env()
buffers=[]
print('rss-ready',flush=True)
for command in sys.stdin:
    command=command.strip()
    if command=='grow':
        block=bytearray(8*1024**2)
        for offset in range(0,len(block),4096):
            block[offset]=1
        buffers.append(block)
        del block
        print('grown',flush=True)
    elif command=='shrink':
        buffers.clear()
        gc.collect()
        print('shrunk',flush=True)
    elif command.startswith('unregister:'):
        client.unregister_model(command.split(':',1)[1])
        print('unregistered',flush=True)
'''
        worker = subprocess.Popen([sys.executable, "-B", "-S", "-c", code],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True)
        self.addCleanup(self.stop_worker, worker)
        self.assertEqual(self.worker_line(worker), "rss-ready")
        return worker

    def test_observe_resident_ram_reads_actual_worker_rss_and_deduplicates_models(self):
        broker, client = self.start(devices=[], ram_bytes=128 * 1024**2, observe_resident_ram=True)
        worker = self.rss_worker()
        first = f"rss-a:{worker.pid}:epoch-a"
        second = f"rss-b:{worker.pid}:epoch-b"
        registered = client.register_model(first, device="cpu")
        self.assertGreater(registered["resident_ram_bytes"], 0)
        self.assertEqual(registered["resident_ram_source"], "proc_status:VmRSS")
        client.register_model(second, device="cpu")
        snapshot = client.snapshot()
        self.assertEqual(snapshot["resident_models"], 2)
        self.assertEqual(len(snapshot["resident_ram_observations"]), 1)
        self.assertEqual(snapshot["resident_ram_observations"][0]["pid"], worker.pid)
        self.assertEqual(snapshot["resident_ram_bytes"], snapshot["resident_ram_observations"][0]["bytes"])
        baseline = snapshot["resident_ram_bytes"]
        self.assertEqual(self.worker_line(worker, "grow"), "grown")
        eventually(lambda: client.snapshot()["resident_ram_bytes"] > baseline)
        peak = client.snapshot()["resident_ram_bytes"]
        self.assertEqual(self.worker_line(worker, "shrink"), "shrunk")
        observed_at = client.snapshot()["resident_ram_observations"][0]["observed_at"]
        eventually(lambda: client.snapshot()["resident_ram_observations"][0]["observed_at"] > observed_at)
        self.assertGreaterEqual(client.snapshot()["resident_ram_bytes"], peak)
        self.assertEqual(self.worker_line(worker, "unregister:" + first), "unregistered")
        self.assertEqual(self.worker_line(worker, "unregister:" + second), "unregistered")
        self.assertEqual(client.snapshot()["resident_models"], 0)
        self.assertGreaterEqual(client.snapshot()["ram_bytes"], peak)
        broker.shutdown_timeout = 0.05
        with self.assertRaisesRegex(BrokerBusy, "1 RSS owners"):
            broker.close()
        worker.kill()
        worker.wait(timeout=3)
        eventually(lambda: client.snapshot()["ram_bytes"] == 0)
        broker.close()

    def test_observe_resident_ram_explicit_value_keeps_toy_budget_and_transfer(self):
        broker, client = self.start(devices=[], ram_bytes=100, observe_resident_ram=True)
        with mock.patch.object(broker, "_read_resident_rss", side_effect=AssertionError("explicit RAM must not use RSS")):
            with client.lease(resource="cpu", phase="load", ram_bytes=80):
                registered = client.register_model("explicit", device="cpu", resident_ram_bytes=80)
                self.assertEqual(registered["resident_ram_source"], "declared")
                self.assertEqual(client.snapshot()["ram_bytes"], 80)
                self.assertEqual(client.snapshot()["transient_ram_bytes"], 0)
            with self.assertRaises(AdmissionTimeout):
                client.lease(resource="cpu", ram_bytes=21, timeout=0.02).__enter__()
        self.assertEqual(client.snapshot()["resident_ram_observations"], [])
        client.unregister_model("explicit")

    def test_observe_resident_ram_missing_rss_is_not_zero_or_certified(self):
        broker, client = self.start(devices=[], ram_bytes=128 * 1024**2, observe_resident_ram=True)
        worker = self.rss_worker()
        with mock.patch.object(broker, "_read_resident_rss", side_effect=AdmissionError("RSS unavailable")):
            with self.assertRaisesRegex(AdmissionError, "RSS unavailable"):
                client.register_model(f"unknown:{worker.pid}:epoch", device="cpu")
            snapshot = client.snapshot()
            self.assertFalse(snapshot["healthy"])
            self.assertEqual(snapshot["ram_bytes"], broker.ram_bytes)
            self.assertEqual(snapshot["resident_models"], 0)
            with self.assertRaises(AdmissionError):
                client.lease(resource="cpu", timeout=0).__enter__()
            worker.kill()
            worker.wait(timeout=3)
            eventually(lambda: client.snapshot()["healthy"] and client.snapshot()["ram_bytes"] == 0)

    def test_observe_resident_ram_budget_failure_retains_live_owner_accounting(self):
        _, client = self.start(devices=[], ram_bytes=100, observe_resident_ram=True)
        worker = self.rss_worker()
        with self.assertRaisesRegex(AdmissionError, "resident RAM exceeds"):
            client.register_model(f"too-large:{worker.pid}:epoch", device="cpu")
        snapshot = client.snapshot()
        self.assertFalse(snapshot["healthy"])
        self.assertGreater(snapshot["ram_bytes"], 100)
        self.assertEqual(snapshot["resident_models"], 0)
        worker.kill()
        worker.wait(timeout=3)
        eventually(lambda: client.snapshot()["healthy"] and client.snapshot()["ram_bytes"] == 0)

    def test_observe_resident_ram_unknown_initializers_are_exclusive(self):
        _, client = self.start(devices=[], ram_bytes=100, observe_resident_ram=True)
        with client.lease(resource="cpu", phase="load"):
            for ram in (0, 20):
                with self.assertRaises(AdmissionTimeout):
                    client.lease(resource="cpu", phase="load", ram_bytes=ram, timeout=0.02).__enter__()
            with client.lease(resource="cpu", phase="prepare"):
                self.assertEqual(client.snapshot()["active"], 2)
        with client.lease(resource="cpu", phase="load", ram_bytes=20):
            with client.lease(resource="cpu", phase="load", ram_bytes=20, timeout=0):
                self.assertEqual(client.snapshot()["ram_bytes"], 40)

    def test_observe_resident_ram_config_defaults_to_explicit_accounting(self):
        broker, client = self.start(devices=[], ram_bytes=100)
        self.assertFalse(broker.observe_resident_ram)
        with mock.patch.object(broker, "_read_resident_rss", side_effect=AssertionError("observation disabled")):
            client.register_model("legacy-zero", device="cpu")
        self.assertEqual(client.snapshot()["ram_bytes"], 0)
        client.unregister_model("legacy-zero")

    def test_uncertified_is_device_exclusive_across_keys_and_load_is_exclusive(self):
        _, client = self.start()
        for first_certified in (False, True):
            with client.lease(backend="one", scratch_bytes=10, concurrency_certified=first_certified):
                with self.assertRaises(AdmissionTimeout):
                    client.lease(backend="two", scratch_bytes=10, timeout=0.02).__enter__()
                if not first_certified:
                    with self.assertRaises(AdmissionTimeout):
                        client.lease(backend="two", scratch_bytes=10, concurrency_certified=True, timeout=0.02).__enter__()
                with self.assertRaises(AdmissionTimeout):
                    client.lease(backend="loader", phase="load", scratch_bytes=10,
                                 concurrency_certified=True, timeout=0.02).__enter__()
        with client.lease(phase="load", scratch_bytes=10, concurrency_certified=True):
            with self.assertRaises(AdmissionTimeout):
                client.lease(backend="two", scratch_bytes=10, concurrency_certified=True, timeout=0.02).__enter__()

    def test_device_cap_is_shared_by_different_certified_keys(self):
        _, client = self.start(max_concurrency=2)
        self.train_growth(client)
        first = client.lease(backend="one", scratch_bytes=10, concurrency_certified=True).__enter__()
        second = client.lease(backend="two", scratch_bytes=10, concurrency_certified=True, timeout=0).__enter__()
        with self.assertRaises(AdmissionTimeout):
            client.lease(backend="three", scratch_bytes=10, concurrency_certified=True, timeout=0.02).__enter__()
        self.assertEqual(client.snapshot()["active"], 2)
        first.finish()
        second.finish()

    def test_high_activity_does_not_probe_more_concurrency(self):
        self.sampler.row["utilization"] = 100
        _, client = self.start()
        self.train_growth(client)
        state = client.snapshot()
        self.assertEqual(state["devices"][self.uuid]["concurrency"], 1)
        self.assertEqual(state["controllers"][0]["limit"], 1)
        self.assertIn(state["controllers"][0]["last_decision"],
                      {"activity_saturated", "conservative_start"})

    def test_live_memory_pressure_retracts_admission(self):
        _, client = self.start()
        self.train_growth(client)
        self.assertGreaterEqual(client.snapshot()["devices"][self.uuid]["concurrency"], 2)
        held = client.lease(scratch_bytes=10, concurrency_certified=True).__enter__()
        self.sampler.row.update(free_bytes=50,
                                processes=[dict(pid=os.getpid(), used_bytes=950)])
        eventually(lambda: client.snapshot()["devices"][self.uuid]["concurrency"] == 1)
        self.assertEqual(client.snapshot()["controllers"][0]["last_decision"], "memory_pressure")
        held.finish()

    def test_unknown_keys_cannot_bypass_conservative_device_start(self):
        _, client = self.start(max_concurrency=3)
        with client.lease(backend="first", scratch_bytes=10, concurrency_certified=True):
            with self.assertRaises(AdmissionTimeout):
                client.lease(backend="second", scratch_bytes=10,
                             concurrency_certified=True, timeout=0.02).__enter__()
        state = client.snapshot()["devices"][self.uuid]
        self.assertEqual(state["concurrency"], 1)
        self.assertEqual(state["activity_reason"], "no_ready_device_request")

    def test_nvidia_sampler_uses_bounded_subprocess_without_gpu_execution(self):
        output = [subprocess.CompletedProcess([], 0, "0, GPU-fake, 100, 90, 75\n", ""),
                  subprocess.CompletedProcess([], 0, f"GPU-fake, {os.getpid()}, 10\n", "")]
        with mock.patch.object(telemetry.subprocess, "run", side_effect=output) as run:
            result = telemetry.validate_sample(telemetry.NvidiaSMISampler(timeout=0.2)())
            self.assertEqual(result["GPU-fake"]["free_bytes"], 90 * 1048576)
            self.assertEqual(result["GPU-fake"]["processes"], {os.getpid(): 10 * 1048576})
            self.assertTrue(all(call.kwargs["timeout"] == 0.2 for call in run.call_args_list))
        malformed = Sampler(self.uuid)()
        malformed[0]["free_bytes"] = float("nan")
        with self.assertRaises(ValueError):
            telemetry.validate_sample(malformed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
