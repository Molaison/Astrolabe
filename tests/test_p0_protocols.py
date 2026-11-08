"""Real process faults at wrapper, collection and durable recovery boundaries."""
from __future__ import annotations

import json
from dataclasses import dataclass
import os
from pathlib import Path
import random
import sys
import threading
import socket
from concurrent.futures import ThreadPoolExecutor

import pytest

from astrolabe.design.collection import (
    CollectionConfig, CompletionPool, InlineExecutor, Job,
)
from astrolabe.design.runtime import OutcomeUnknown
from astrolabe.model_daemon import RequestOutcomeUnknown


def make_runner(tmp_path, backend, fault="disconnect"):
    script = tmp_path / "daemon.py"
    script.write_text(
        """import json, os, sys, time
from pathlib import Path
root = Path(__file__).parent
for line in sys.stdin:
    request = json.loads(line)
    if request["type"] == "ping":
        print(json.dumps({"type":"pong", "id":request["id"]}), flush=True)
        continue
    if request["type"] == "shutdown":
        print(json.dumps({"type":"ok"}), flush=True)
        break
    with (root / "effects.jsonl").open("a") as out:
        out.write(json.dumps(request) + "\\n")
        out.flush()
        os.fsync(out.fileno())
    fault = FAULT
    if fault == "disconnect":
        os._exit(12)
    if fault == "malformed":
        print("{broken", flush=True)
        continue
    if fault == "timeout":
        time.sleep(60)
    if fault == "error":
        print(json.dumps({"type":"error", "id":request["id"],
                         "error_type":"RuntimeError", "message":"inference failed"}),
              flush=True)
        continue
    output = root / "candidate.pdb"
    output.write_text("MODEL 1\\nEND\\n")
    result = {"name":"sample", "samples":["ACD"],
              "candidates":[{"peptide_pdb":str(output)}]}
    print(json.dumps({"type":"result",
                      "id":"foreign" if fault == "identity" else request["id"],
                      "results":[result]}), flush=True)
""".replace("FAULT", repr(fault))
    )
    args = dict(python_bin=sys.executable, repo_path=str(tmp_path),
                work_dir=str(tmp_path / "work"),
                response_cache_dir=str(tmp_path / "cache"),
                response_cache=False, request_timeout=0.2)
    if backend == "dplm":
        from astrolabe.mutators.wrappers.dplm_runner import DPLMRunner
        runner = DPLMRunner(args)
        runner.daemon_script_path = script
    else:
        from astrolabe.structure_processors.wrappers.rapidock_runner import RAPiDockRunner
        (tmp_path / "rapidock_global.pt").write_bytes(b"fixture")
        runner = RAPiDockRunner(dict(
            args, model_dir=str(tmp_path), daemon_script_path=str(script)))
    return runner


def infer(runner, backend, tmp_path):
    if backend == "dplm":
        return runner.design_sequences([{"name": "sample", "masked_sequence": "A<mask>D"}])
    protein, peptide = tmp_path / "protein.pdb", tmp_path / "peptide.pdb"
    protein.write_text("ATOM protein\n")
    peptide.write_text("ATOM peptide\n")
    return runner.dock_candidates(
        name="sample", protein_pdb=str(protein), peptide_pdb=str(peptide))


def effects(tmp_path):
    path = tmp_path / "effects.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


@pytest.mark.parametrize("backend", ["dplm", "rapidock"])
@pytest.mark.parametrize("fault", ["disconnect", "malformed", "identity", "error", "timeout"])
def test_issued_stdio_fault_is_unknown_and_never_reissued(tmp_path, backend, fault):
    runner = make_runner(tmp_path, backend, fault)
    try:
        with pytest.raises(RequestOutcomeUnknown) as caught:
            infer(runner, backend, tmp_path)
        first = effects(tmp_path)
        assert len(first) == 1
        assert caught.value.request_id == first[0]["id"]
        after = random.getstate()
        with pytest.raises(RequestOutcomeUnknown):
            infer(runner, backend, tmp_path)
        assert effects(tmp_path) == first
        assert random.getstate() == after
    finally:
        runner.shutdown()
    assert runner._proc is None


@pytest.mark.parametrize("wrapped", [False, True])
def test_collection_retains_transport_unknown_instead_of_failed_result(wrapped):
    observed, consumed = [], []

    def worker(job):
        error = RequestOutcomeUnknown("response lost", request_id=job.job_id)
        if wrapped:
            raise RuntimeError("adapter context") from error
        raise error

    with CompletionPool(
        CollectionConfig(num_jobs=1, min_completed=1, max_wait_s=1),
        worker, executor_factory=InlineExecutor,
        terminal_observed=lambda *args: observed.append(args),
        collection_commit=consumed.append,
    ) as pool:
        pool.submit_generation(1, [Job("candidate", generation=1)])
        with pytest.raises(OutcomeUnknown):
            pool.collect_until_ready()
        assert observed == consumed == []


@pytest.mark.parametrize("backend", ["dplm", "rapidock"])
def test_stdio_shutdown_failure_retains_owned_process_for_retry(tmp_path, backend, monkeypatch):
    runner = make_runner(tmp_path, backend, "success")
    infer(runner, backend, tmp_path)
    child = runner._proc
    module = sys.modules[type(runner).__module__]
    stop = module.stop_owned_stdio_process

    def failed_stop(proc):
        assert proc is child
        raise OSError("injected cleanup failure")

    monkeypatch.setattr(module, "stop_owned_stdio_process", failed_stop)
    try:
        with pytest.raises(OSError, match="injected"):
            runner.shutdown()
        assert runner._proc is child and child.poll() is None
        with pytest.raises(RuntimeError, match="cleanup"):
            infer(runner, backend, tmp_path)
        assert len(effects(tmp_path)) == 1
    finally:
        monkeypatch.setattr(module, "stop_owned_stdio_process", stop)
        runner.shutdown()
    assert child.poll() is not None and runner._proc is None


@pytest.mark.parametrize("backend", ["dplm", "rapidock"])
def test_stdio_success_keeps_original_result_and_seed_consumption(tmp_path, backend):
    runner = make_runner(tmp_path, backend, "success")
    before = random.getstate()
    rng = random.Random()
    rng.setstate(before)
    if backend == "dplm":
        expected_seed = rng.randint(0, 1_000_000)
    try:
        result = infer(runner, backend, tmp_path)
        assert len(effects(tmp_path)) == 1
        assert random.getstate() == rng.getstate()
        if backend == "dplm":
            assert result == {"sample": ["ACD"]}
            assert effects(tmp_path)[0]["seed"] == expected_seed
        else:
            assert result == [str(tmp_path / "candidate.pdb")]
    finally:
        runner.shutdown()


@dataclass(frozen=True)
class Count:
    value: int


@pytest.mark.parametrize("backend", ["dplm", "rapidock"])
def test_stdio_unknown_survives_durable_reopen_and_keeps_charge(tmp_path, backend):
    from astrolabe.design.operations import INVOKE, operation
    from astrolabe.design.runtime import BudgetExhausted, Runtime
    from astrolabe.design.types import History, Requirement, key_for

    runner = make_runner(tmp_path, backend)
    calls = []

    def invoke(request):
        calls.append(request)
        return Count(len(infer(runner, backend, tmp_path)))

    component = operation("sample", Count, Count, invoke, version="p0-test",
                          paid_boundary="predict", terminal_errors=(Exception,))
    try:
        for _ in range(2):
            with Runtime((component,), root=tmp_path / "journal",
                         budgets={"predict": 1}) as runtime:
                runtime.preflight((Requirement("sample", INVOKE, "required"),))
                with runtime.bind("candidate", History(key_for(()), ())):
                    with pytest.raises(OutcomeUnknown):
                        runtime.call("propose", component, Count(1))
                assert runtime.attempts()[0].status == "unknown"
                with runtime.bind("other", History(key_for(()), ())):
                    with pytest.raises(BudgetExhausted):
                        runtime.call("propose", component, Count(1))
        assert calls == [Count(1)]
        assert len(effects(tmp_path)) == 1
    finally:
        runner.shutdown()


@pytest.mark.parametrize("backend", ["native", "na_mpnn"])
def test_inner_owner_survives_failed_shutdown_and_cannot_be_reused(backend):
    class Inner:
        attempts = 0

        def shutdown(self):
            self.attempts += 1
            if self.attempts == 1:
                raise OSError("retained inner owner")

    inner = Inner()
    if backend == "native":
        from astrolabe.design.native_support import LazyNativeOwner
        from astrolabe.design.store import DataStore
        from astrolabe.design.types import key_for
        owner = LazyNativeOwner(lambda: inner, DataStore(), key_for("fixture"),
                                rng_modules=())
        ready = owner._initialize(owner.initialization_request)
        owner._bind_ready(ready)
        get = owner.get
        field = "_value"
    else:
        from astrolabe.mutators.wrappers.na_mpnn_runner import NAMPNNRunner
        owner = object.__new__(NAMPNNRunner)
        owner._daemon_lock = threading.RLock()
        owner._owned_daemon = inner
        get = lambda: owner._request_daemon({})
        field = "_owned_daemon"
    with pytest.raises(OSError, match="retained"):
        owner.shutdown()
    assert getattr(owner, field) is inner
    with pytest.raises(RuntimeError, match="cleanup"):
        get()
    owner.shutdown()
    owner.shutdown()
    assert inner.attempts == 2
    assert getattr(owner, field) is None


@pytest.mark.parametrize("field", [
    "protocol", "request_id", "generation", "identity", "frame", "serialization",
])
def test_invalid_socket_response_retains_unknown_request_identity(tmp_path, field):
    from astrolabe import model_daemon as transport

    path = tmp_path / "daemon.sock"
    requests = []
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(path))
        path.chmod(0o600)
        server.listen(1)

        def serve():
            connection, _ = server.accept()
            with connection:
                request = transport._read_frame(connection)
                requests.append(request)
                response = {key: request[key] for key in (
                    "protocol", "request_id", "generation", "identity")}
                response.update(type="result", result=1)
                if field == "frame":
                    connection.sendall(b"\xff" * 8)
                else:
                    if field == "serialization":
                        response.update(type="error", code="response_not_materializable")
                    else:
                        response[field] = "foreign"
                    transport._send_frame(connection, response)

        owner = transport.OwnedModelDaemon(
            transport.DaemonSpec(backend="fixture", factory="unused:factory", settings={}),
            socket_path=path,
        )
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(serve)
            with pytest.raises(RequestOutcomeUnknown) as caught:
                owner._exchange({"type": "infer", "request_id": "original"}, timeout=2)
            future.result(timeout=2)
        assert caught.value.request_id == "original"
        assert caught.value.generation == owner.generation
        assert len(requests) == 1
        owner.shutdown()  # attached: no remote stop request


@pytest.mark.parametrize("backend", ["dplm", "rapidock"])
def test_cache_publication_failure_after_compute_does_not_reissue(tmp_path, backend, monkeypatch):
    runner = make_runner(tmp_path, backend, "success")
    runner.response_cache = True

    def fail(*args):
        raise OSError("publication failure")

    if backend == "dplm":
        # Ask for one sample from a two-sample bank so the surplus is published.
        script = tmp_path / "daemon.py"
        script.write_text(script.read_text().replace('"samples":["ACD"]', '"samples":["ACD","ACE"]'))
        monkeypatch.setattr(runner, "_append_cached_samples", fail)
    else:
        monkeypatch.setattr(runner, "_store_cached_candidates", fail)
    try:
        with pytest.raises(RequestOutcomeUnknown, match="publication"):
            infer(runner, backend, tmp_path)
        with pytest.raises(RequestOutcomeUnknown):
            infer(runner, backend, tmp_path)
        assert len(effects(tmp_path)) == 1
        assert (tmp_path / "candidate.pdb").exists()
    finally:
        runner.shutdown()


def test_native_cleanup_callback_can_join_reentrant_thread():
    from astrolabe.design.native_support import LazyNativeOwner
    from astrolabe.design.store import DataStore
    from astrolabe.design.types import key_for

    errors = []
    owner = None

    class Inner:
        def shutdown(self):
            def reenter():
                try:
                    owner.shutdown()
                except RuntimeError as error:
                    errors.append(error)
            child = threading.Thread(target=reenter)
            child.start()
            child.join(timeout=2)
            assert not child.is_alive(), "shutdown held the lifecycle lock across callback"

    inner = Inner()
    owner = LazyNativeOwner(lambda: inner, DataStore(), key_for("reentrant"), rng_modules=())
    owner._initialize(owner.initialization_request)
    owner.shutdown()
    assert len(errors) == 1 and "in progress" in str(errors[0])
    assert owner._value is None


def test_real_refinement_owner_chain_retries_only_failed_child():
    from astrolabe.design.native_refinement_runtime import _LoadedModels, DurableNativeRefinementOwner
    from astrolabe.design.native_support import LazyNativeOwner
    from astrolabe.design.store import DataStore
    from astrolabe.design.types import key_for

    calls = []

    class Child:
        def __init__(self, name):
            self.name = name

        def shutdown(self):
            calls.append(self.name)
            if self.name == "evaluator" and calls.count(self.name) == 1:
                raise OSError("child cleanup failed")

    models = _LoadedModels(Child("mutator"), Child("evaluator"))
    lazy = LazyNativeOwner(lambda: models, DataStore(), key_for("composite"), rng_modules=())
    lazy._initialize(lazy.initialization_request)
    durable = object.__new__(DurableNativeRefinementOwner)
    durable.models, durable._shutdown = lazy, False
    with pytest.raises(OSError, match="child"):
        durable.shutdown()
    assert not durable._shutdown and not models._shutdown
    assert lazy._value is models
    durable.shutdown()
    durable.shutdown()
    assert durable._shutdown and models._shutdown and lazy._value is None
    assert calls == ["evaluator", "mutator", "evaluator"]


@pytest.mark.parametrize("science_failed", [False, True])
def test_pipeline_cleanup_failure_retains_raw_and_original_science_error(
    tmp_path, monkeypatch, science_failed,
):
    from types import SimpleNamespace
    from astrolabe.app import _load_run_module
    from astrolabe.core import global_config

    run = _load_run_module()
    original = ValueError("scientific failure")
    raw_cleanup, provenance = [], []

    class Resource:
        def shutdown(self):
            raise OSError("cleanup failure")

    def initialize(args):
        global_config.init_const("mutators", {"fixture": Resource()})
        return object(), SimpleNamespace(current_evaluator="fixture"), {}

    def optimize(*args, **kwargs):
        if science_failed:
            raise original

    args = SimpleNamespace(out_dir=tmp_path, task_name="cleanup", mutator=None)
    monkeypatch.setattr(run, "parse_arguments", lambda cfg: args)
    monkeypatch.setattr(run, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(run, "initialize_pipeline_components", initialize)
    monkeypatch.setattr(run, "run_optimization_loop", optimize)
    monkeypatch.setattr(run, "_finish_run_provenance",
                        lambda args, status, **kw: provenance.append(status))
    monkeypatch.setattr(run, "_cleanup_auto_managed_raw_dir",
                        lambda *, args, succeeded: raw_cleanup.append(succeeded))
    global_config.reset()
    try:
        with pytest.raises(ValueError if science_failed else RuntimeError) as caught:
            run._run_pipeline({})
        assert raw_cleanup == [False]
        assert provenance[-1] == "failed"
        if science_failed:
            assert caught.value is original
            assert caught.value.cleanup_error.cleanup_report.failed
        else:
            assert caught.value.cleanup_report.failed
    finally:
        global_config.reset()


def test_partial_stdio_write_timeout_retains_unknown():
    import subprocess
    from astrolabe.stdio_daemon import json_line_request, stop_owned_stdio_process

    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, start_new_session=True,
    )
    try:
        with pytest.raises(RequestOutcomeUnknown) as caught:
            json_line_request(
                proc, {"type": "infer", "id": "partial", "input": "x" * 2_000_000},
                timeout=0.1,
            )
        assert caught.value.request_id == "partial"
        assert isinstance(caught.value.__cause__, TimeoutError)
    finally:
        stop_owned_stdio_process(proc)


def test_worker_cleanup_failure_retains_resource_and_blocks_execution():
    from astrolabe.design.refinement_worker import _Worker
    from astrolabe.runtime_support import ResourceOwner

    calls = []

    class Evaluator:
        def shutdown(self):
            calls.append("shutdown")
            if len(calls) == 1:
                raise OSError("worker cleanup")

    evaluator = Evaluator()
    worker = object.__new__(_Worker)
    worker.closed, worker._cleanup_requested = False, False
    worker._cleanup = ResourceOwner()
    worker.evaluators = {"a": evaluator, "b": evaluator}
    worker.pid = os.getpid()
    with pytest.raises(OSError, match="worker cleanup"):
        worker.shutdown()
    assert not worker.closed
    assert worker._cleanup.report.failed[0].resource is evaluator
    with pytest.raises(RuntimeError, match="initialized"):
        worker.execute({})
    worker.shutdown()
    worker.shutdown()
    assert worker.closed and calls == ["shutdown", "shutdown"]


@pytest.mark.parametrize("fault", ["disconnect", "success"])
@pytest.mark.parametrize("outer_exception", [False, True])
def test_descriptor_restore_error_cannot_remove_unknown_fence(
    tmp_path, monkeypatch, fault, outer_exception,
):
    runner = make_runner(tmp_path, "rapidock", fault)
    original = os.set_blocking

    def fail_restore(fd, blocking):
        if blocking:
            raise OSError("descriptor closed concurrently")
        original(fd, blocking)

    # Complete the startup ping first, so the fault belongs to a scientific RPC.
    runner._ensure_daemon()
    monkeypatch.setattr(os, "set_blocking", fail_restore)
    try:
        def check():
            with pytest.raises(RequestOutcomeUnknown):
                infer(runner, "rapidock", tmp_path)
        if outer_exception:
            try:
                raise ValueError("unrelated caller exception")
            except ValueError:
                check()
        else:
            check()
        with pytest.raises(RequestOutcomeUnknown):
            infer(runner, "rapidock", tmp_path)
        assert len(effects(tmp_path)) == 1
    finally:
        runner.shutdown()


@pytest.mark.parametrize("failure_count", [1, 2])
def test_registered_parent_and_child_do_not_retry_during_one_unwind(failure_count):
    from astrolabe.design.native_refinement_runtime import _LoadedModels, DurableNativeRefinementOwner
    from astrolabe.design.native_support import LazyNativeOwner
    from astrolabe.design.store import DataStore
    from astrolabe.design.types import key_for
    from astrolabe.model_execution import model_run_scope, register_model_owner

    calls = []

    class Child:
        def shutdown(self):
            calls.append("shutdown")
            if len(calls) <= failure_count:
                raise OSError("first cleanup failed")

    models = _LoadedModels(None, Child())
    lazy = LazyNativeOwner(lambda: models, DataStore(), key_for("registered"), rng_modules=())
    durable = object.__new__(DurableNativeRefinementOwner)
    durable.models, durable._shutdown = lazy, False
    with pytest.raises(RuntimeError, match="failed to stop"):
        with model_run_scope() as scope:
            lazy._initialize(lazy.initialization_request)
            register_model_owner(durable)
    assert calls == ["shutdown"]
    assert scope.cleanup_report.failed and lazy._value is models
    scope.cancel(retry_failed=True)
    assert calls == ["shutdown", "shutdown"]
    if failure_count == 2:
        assert scope.cleanup_report.failed
        scope.cancel(retry_failed=True)
    assert scope.cleanup_report.complete
    assert len(calls) == failure_count + 1


def test_partial_refinement_load_preserves_original_error_and_retry_owner():
    from types import SimpleNamespace
    from astrolabe.design.native_refinement_runtime import DurableNativeRefinementOwner
    from astrolabe.model_execution import model_run_scope, register_model_owner

    original, calls = ValueError("evaluator constructor"), []

    class Mutator:
        def shutdown(self):
            calls.append("shutdown")
            if len(calls) == 1:
                raise OSError("mutator cleanup")

    def fail():
        raise original

    owner = object.__new__(DurableNativeRefinementOwner)
    owner._shutdown, owner._partial_cleanup = False, None
    owner.models = SimpleNamespace(shutdown=lambda: None)
    owner.spec = SimpleNamespace(load_mutator=Mutator, load_evaluator=fail)
    with pytest.raises(ValueError) as caught:
        with model_run_scope() as scope:
            register_model_owner(owner)
            owner._load_models()
    assert caught.value is original and calls == ["shutdown"]
    assert original.cleanup_owner is owner
    scope.cancel(retry_failed=True)
    assert scope.cleanup_report.complete and calls == ["shutdown", "shutdown"]


def test_refinement_unwind_preserves_unknown_and_exposes_failed_cleanup(monkeypatch):
    from astrolabe.design import native_refinement_runtime as refinement
    original = OutcomeUnknown("native request unknown")
    calls = []

    class Owner:
        def __init__(self, *args, **kwargs):
            pass

        def bootstrap(self):
            raise original

        def shutdown(self):
            calls.append("shutdown")
            raise OSError("owner cleanup")

    monkeypatch.setattr(refinement, "DurableNativeRefinementOwner", Owner)
    with pytest.raises(OutcomeUnknown) as caught:
        refinement.run_durable_native_refinement(None, policy=None, protocol=None, runtime=None)
    assert caught.value is original and calls == ["shutdown"]
    assert original.cleanup_report.failed
