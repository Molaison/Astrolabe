"""CPU-only pool contracts; real bounded transport, fake scientific loading."""

from copy import deepcopy
import builtins
import os
import random
import threading

import pytest

from astrolabe.model_daemon import (
    DaemonSpec, FrameTooLarge, LocalModelOwner, RequestOutcomeUnknown, model_identity,
)
from astrolabe.model_execution import model_run_scope, register_model_owner
from astrolabe.runtime.model_pool import (
    LifecycleDependencyError, get_local_model_pool, import_lifecycle_dependency,
    local_model_pool, pool_owns_registration,
)


def initial_state():
    # Serialized capture shapes, without importing NumPy/Torch or probing CUDA.
    return {"python": random.Random(71).getstate(),
            "numpy": ["MT19937", [17, 29, 31], 3, 0, 0.0],
            "torch": [2, 4, 6, 8], "torch_seed": 71, "cuda_seed": 71}


def full_state(state=None, *, cuda=None, visibility="GPU-A,GPU-B"):
    state = deepcopy(initial_state() if state is None else state)
    state.pop("cuda_seed", None)
    state["cuda"] = deepcopy([[11, 12], [21, 22]] if cuda is None else cuda)
    state["cuda_visible_devices"] = visibility
    return state


def spec_for(state=None, **settings):
    return DaemonSpec("esmfold", "cpu_pool_fixture:make",
                      dict(settings, _initial_random_state=deepcopy(
                          initial_state() if state is None else state)),
                      max_frame_bytes=32768)


class FakeOwner(LocalModelOwner):
    """Reuse LocalModelOwner.request/_copy_message; fake only its loader."""

    def __init__(self, spec, factory, **kwargs):
        self.spec = spec
        self.pid = os.getpid()
        self.identity = model_identity(spec)
        self._closed = False
        self.shutdown_calls = 0
        self.request_calls = 0
        self.describe_calls = 0
        self.fail = None
        self.factory = factory
        self.output_buffer = [0, 0]
        self.registered_to_pool = pool_owns_registration(self)
        if not self.registered_to_pool:
            register_model_owner(self)
        factory.owners.append(self)
        if factory.construction_error:
            raise RuntimeError("construction failed after registration")
        self.description = {
            "ok": True,
            "metadata": {"runtime_lifecycle": {
                "loader_count": 1, "model_id": f"cpu-model-{len(factory.owners)}",
                "pid": self.pid}},
            "random_state": deepcopy(factory.postinit),
        }
        if factory.transition is not None:
            self.description["metadata"]["initialization_transition"] = deepcopy(
                factory.transition)
        self.handle = self._handle

    def _handle(self, request):
        self.request_calls += 1
        if self.fail is not None:
            if isinstance(self.fail, BaseException):
                raise self.fail
            return self.fail
        if request["operation"] == "describe":
            self.describe_calls += 1
            if self.factory.describe_error:
                raise RequestOutcomeUnknown("describe outcome unknown")
            return self.description
        self.last_request = request
        payload = request["payload"]
        payload["nested"]["labels"].append("kernel")
        self.output_buffer[:] = [payload["value"], payload["value"] + 1]
        return {"ok": True, "result": self.output_buffer,
                "random_state": request["random_state"]}

    def shutdown(self):
        self.shutdown_calls += 1
        if self.factory.shutdown_error:
            raise RuntimeError("shutdown failed")
        super().shutdown()


class Factory:
    def __init__(self):
        self.owners = []
        self.transition = None
        self.postinit = full_state(cuda=[[91, 92], [93, 94]])
        self.postinit["torch"] = [5, 7, 9, 11]
        self.construction_error = False
        self.describe_error = False
        self.shutdown_error = False

    def __call__(self, spec, **kwargs):
        return FakeOwner(spec, self, **kwargs)


def acquire(pool, factory, spec=None, *, identity="identity-A", **kwargs):
    return pool.acquire(spec or spec_for(), factory, kwargs, identity)


@pytest.fixture
def factory(monkeypatch):
    # Environment names are scope data only. No socket or GPU is touched.
    monkeypatch.setenv("ASTROLABE_RESOURCE_SOCKET", "/cpu-pool-fixture/broker.sock")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-A,GPU-B")
    assert get_local_model_pool() is None
    result = Factory()
    yield result
    assert get_local_model_pool() is None


def test_reuse_across_run_scopes_and_borrow_shutdown(factory):
    with local_model_pool(max_models=1, backends={"esmfold"}) as pool:
        assert get_local_model_pool() is pool
        with model_run_scope():
            first = acquire(pool, factory)
            real = factory.owners[0]
            assert first.attached and first.release_requires_process_exit
            assert real.registered_to_pool
            assert not pool_owns_registration(real)  # only during construction
            register_model_owner(first)  # attached handles cannot own residency
            first.shutdown()
            first.shutdown()
            with pytest.raises(RuntimeError, match="Borrowed.*closed"):
                first.request({"operation": "describe"})
        assert real.shutdown_calls == 0
        with model_run_scope():
            second = acquire(pool, factory)
            assert second is not first
            description = second.request({"operation": "describe"})
            assert description["random_state"] == factory.postinit
            assert description["metadata"]["runtime_lifecycle"]["loader_count"] == 1
        assert real.shutdown_calls == 0
        assert pool.metadata["real_load_count"] == 1
        assert pool.metadata["hits"] == 1
        assert pool.metadata["models"][0]["borrowers"] == 1
    assert real.shutdown_calls == 1
    pool.close()
    second.shutdown()
    assert real.shutdown_calls == 1
    assert pool.metadata["release_requires_process_exit"]
    with pytest.raises(RuntimeError, match="closed"):
        second.request({"operation": "describe"})


def test_dependency_cpu_rng_replayed_before_model_acquisition(factory, monkeypatch):
    from astrolabe import model_execution
    state = initial_state()
    module = object()
    imports = []
    monkeypatch.setattr(model_execution, "capture_random_state", lambda: deepcopy(state))
    def restore(value):
        state.clear()
        state.update(deepcopy(value))
    monkeypatch.setattr(model_execution, "restore_random_state", restore)
    def imported(name):
        imports.append(name)
        if len(imports) == 1:
            state["numpy"] = ["MT19937", [31, 29, 17], 2, 0, 0.]
        return module
    monkeypatch.setattr("astrolabe.runtime.model_pool.importlib.import_module", imported)
    with local_model_pool() as pool:
        assert import_lifecycle_dependency("cpu_library") is module
        postimport = deepcopy(state)
        restore(full_state())
        actual_cuda = deepcopy(state["cuda"])
        assert import_lifecycle_dependency("cpu_library") is module
        assert state["numpy"] == postimport["numpy"]
        assert state["cuda"] == actual_cuda
        assert pool.metadata["dependency_initializations"][0]["hits"] == 1
        # A seed alone cannot authorize replay from a different CPU state.
        restore(full_state())
        state["numpy"][2] = 99
        with pytest.raises(LifecycleDependencyError, match="different complete CPU"):
            import_lifecycle_dependency("cpu_library")


def test_dependency_changing_cuda_is_rejected(factory, monkeypatch):
    from astrolabe import model_execution
    state = initial_state()
    monkeypatch.setattr(model_execution, "capture_random_state", lambda: deepcopy(state))
    def imported(name):
        state["cuda_seed"] += 1
        return object()
    monkeypatch.setattr("astrolabe.runtime.model_pool.importlib.import_module", imported)
    with local_model_pool() as pool:
        with pytest.raises(LifecycleDependencyError, match="changed CUDA"):
            import_lifecycle_dependency("cpu_library")
        assert pool.metadata["quarantined"]
        with pytest.raises(LifecycleDependencyError, match="quarantined"):
            import_lifecycle_dependency("cpu_library")
        with pytest.raises(RuntimeError, match="quarantined"):
            acquire(pool, factory)


@pytest.mark.parametrize("damage", ["module", "cached_rng", "broker"])
def test_dependency_replay_cannot_skip_live_guards(factory, monkeypatch, damage):
    from astrolabe import model_execution
    state = initial_state()
    module = object()
    calls = []
    monkeypatch.setattr(model_execution, "capture_random_state", lambda: deepcopy(state))
    monkeypatch.setattr(model_execution, "restore_random_state", lambda value: pytest.fail(
        "Guard failure must precede RNG restoration"))
    def imported(name):
        calls.append(name)
        if len(calls) > 1:
            if damage == "module":
                return object()
            if damage == "cached_rng":
                state["numpy"][2] += 1
        return module
    monkeypatch.setattr("astrolabe.runtime.model_pool.importlib.import_module", imported)
    with local_model_pool():
        import_lifecycle_dependency("cpu_library")
        if damage == "broker":
            monkeypatch.setenv("ASTROLABE_RESOURCE_SOCKET", "/changed")
        with pytest.raises(LifecycleDependencyError):
            import_lifecycle_dependency("cpu_library")


def test_partial_dependency_restore_propagates_through_real_logging(factory, monkeypatch):
    import ast
    import logging
    from pathlib import Path
    from astrolabe import model_execution
    state = initial_state()
    module = object()
    monkeypatch.setattr(model_execution, "capture_random_state", lambda: deepcopy(state))
    monkeypatch.setattr("astrolabe.runtime.model_pool.importlib.import_module", lambda name: module)
    def broken_restore(value):
        state["numpy"][2] = 88  # Identifies a deliberately partial CPU write.
        raise ValueError("restore stopped after partial state write")
    monkeypatch.setattr(model_execution, "restore_random_state", broken_restore)
    source = Path(__file__).resolve().parents[1] / "run.py"
    node = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "_suppress_external_logging")
    namespace = {"absl_logging": None, "logging": logging, "os": os}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    with local_model_pool() as pool:
        borrowed = acquire(pool, factory)
        import_lifecycle_dependency("prody")
        with pytest.raises(LifecycleDependencyError, match="partial state"):
            namespace["_suppress_external_logging"]()
        assert state["numpy"][2] == 88
        assert pool.metadata["quarantined"]
        with pytest.raises(RuntimeError, match="quarantined"):
            acquire(pool, factory)
        with pytest.raises(RuntimeError, match="quarantined"):
            borrowed.request({"operation": "describe"})
        assert factory.owners[0].shutdown_calls == 0
    assert factory.owners[0].shutdown_calls == 1


@pytest.mark.parametrize("field", ["python", "numpy", "torch", "torch_seed", "cuda_seed"])
def test_same_identity_different_initial_rng_is_incompatible(factory, field):
    changed = initial_state()
    changed[field] = 999
    with local_model_pool(max_models=2) as pool:
        original = acquire(pool, factory)
        assert acquire(pool, factory, spec_for(changed)) is None
        assert len(factory.owners) == 1  # no spare-capacity reload of this identity
        assert original.request({"operation": "describe"})["ok"]


def test_full_cuda_requires_exact_all_devices_and_visibility(factory):
    expected = full_state()
    with local_model_pool() as pool:
        acquire(pool, factory, spec_for(expected))
        assert acquire(pool, factory, spec_for(expected)) is not None
        for state in (full_state(cuda=[[11, 12], [21, 99]]),
                      full_state(visibility="GPU-B,GPU-A"), initial_state()):
            assert acquire(pool, factory, spec_for(state)) is None


def transition_proof(factory):
    return {"before": initial_state(), "cuda_before_place": full_state(),
            "after": deepcopy(factory.postinit)}


def test_lazy_to_full_uses_captured_preplacement_cuda_and_returns_postinit(factory):
    factory.transition = transition_proof(factory)
    # CPU loading is allowed to consume RNG between before and preplacement.
    factory.transition["cuda_before_place"]["torch"] = [80, 81]
    with local_model_pool() as pool:
        first = acquire(pool, factory)
        second = acquire(pool, factory, spec_for(full_state()))
        assert second is not None
        assert second.request({"operation": "describe"}) == first.initrecord
        assert second.initrecord["random_state"] == factory.postinit
        assert pool.metadata["real_load_count"] == 1
        assert pool.metadata["hits"] == 1


@pytest.mark.parametrize("damage", [
    "no_proof", "missing_before", "missing_after", "missing_preplacement",
    "missing_visibility", "bad_before", "bad_after", "bad_cuda", "bad_visibility",
    "bad_seed", "caller_cpu", "caller_numpy", "caller_python", "caller_seed",
    "postinit_cuda_instead_of_preplacement",
])
def test_lazy_full_rejects_inexact_or_missing_proof(factory, damage):
    proof = transition_proof(factory)
    actual = full_state()
    if damage == "no_proof":
        proof = None
    elif damage.startswith("missing_"):
        key = damage.removeprefix("missing_")
        if key == "visibility":
            proof["cuda_before_place"].pop("cuda_visible_devices")
        else:
            proof.pop("cuda_before_place" if key == "preplacement" else key)
    elif damage == "bad_before":
        proof["before"]["numpy"][2] += 1
    elif damage == "bad_after":
        proof["after"]["cuda"][0][0] += 1
    elif damage == "bad_cuda":
        proof["cuda_before_place"]["cuda"][1][0] += 1
    elif damage == "bad_visibility":
        proof["cuda_before_place"]["cuda_visible_devices"] = "GPU-B,GPU-A"
    elif damage == "bad_seed":
        proof["cuda_before_place"]["torch_seed"] += 1
    elif damage == "caller_cpu":
        actual["torch"][0] += 1
    elif damage == "caller_numpy":
        actual["numpy"][2] += 1
    elif damage == "caller_python":
        actual["python"] = random.Random(72).getstate()
    elif damage == "caller_seed":
        actual["torch_seed"] += 1
    else:
        actual["cuda"] = deepcopy(factory.postinit["cuda"])
    factory.transition = proof
    with local_model_pool() as pool:
        acquire(pool, factory)
        assert acquire(pool, factory, spec_for(actual)) is None
        assert len(factory.owners) == 1


def test_initrecord_and_describe_are_detached_and_preserve_real_rng(factory):
    with local_model_pool() as pool:
        first = acquire(pool, factory)
        original = first.initrecord
        exposed = first.initrecord
        exposed["random_state"]["torch"][0] = -1
        pool.metadata["models"][0]["initrecord"]["random_state"].clear()
        real = factory.owners[0]
        real.description["random_state"]["torch"][0] = -2
        assert first.request({"operation": "describe"}) == original
        borrowed = acquire(pool, factory)
        assert borrowed.initrecord == original


def test_capacity_has_no_eviction_even_after_handle_shutdown(factory):
    with local_model_pool(max_models=2) as pool:
        first = acquire(pool, factory)
        acquire(pool, factory, identity="identity-B")
        assert acquire(pool, factory, identity="identity-C") is None
        first.shutdown()
        assert acquire(pool, factory, identity="identity-C") is None
        assert acquire(pool, factory) is not None
        assert pool.metadata["real_load_count"] == 2
        assert all(owner.shutdown_calls == 0 for owner in factory.owners)
    assert [owner.shutdown_calls for owner in factory.owners] == [1, 1]


def test_changed_asset_content_changes_real_identity_and_cannot_reuse(factory, tmp_path):
    asset = tmp_path / "dummy-weights.txt"
    asset.write_text("synthetic asset A")
    spec = spec_for(weight_path=str(asset))
    before = model_identity(spec)
    with local_model_pool() as pool:
        acquire(pool, factory, spec, identity=before)
        asset.write_text("synthetic asset B")
        after = model_identity(spec)
        assert before != after
        assert acquire(pool, factory, spec, identity=after) is None
        assert len(factory.owners) == 1


@pytest.mark.parametrize("key,value", [
    ("ASTROLABE_RESOURCE_SOCKET", "/different/broker.sock"),
    ("CUDA_VISIBLE_DEVICES", "GPU-B,GPU-A"),
    ("CUDA_VISIBLE_DEVICES", "GPU-A"),
    ("CUDA_VISIBLE_DEVICES", ""),
    ("CUDA_VISIBLE_DEVICES", None),
])
def test_scope_boundary_rejects_acquire_and_existing_requests(factory, monkeypatch, key, value):
    with local_model_pool() as pool:
        borrowed = acquire(pool, factory)
        if value is None:
            monkeypatch.delenv(key)
        else:
            monkeypatch.setenv(key, value)
        assert acquire(pool, factory) is None
        with pytest.raises(RuntimeError, match="scope changed"):
            borrowed.request({"operation": "describe"})
        assert len(factory.owners) == 1
    assert factory.owners[0].shutdown_calls == 1


@pytest.mark.parametrize("change", ["backend", "provider", "python", "cwd", "env", "socket", "profile"])
def test_incompatible_runtime_and_resource_profile(factory, change, tmp_path):
    with local_model_pool() as pool:
        acquire(pool, factory)
        candidate = spec_for()
        kwargs = {}
        if change == "backend":
            candidate.backend = "other"
        elif change == "provider":
            candidate.provider = {"mode": "remote"}
        elif change == "python":
            candidate.python_bin = "/unavailable/python"
        elif change == "cwd":
            candidate.cwd = str(tmp_path)
        elif change == "env":
            candidate.env = {"POOL_TEST_UNSET_ENV": "new"}
        elif change == "socket":
            kwargs["socket_path"] = "/attached/owner.sock"
        else:
            candidate.settings["_runtime_profile"] = {"resident_bytes": 999}
        assert acquire(pool, factory, candidate, **kwargs) is None
        assert len(factory.owners) == 1


def infer_request(value):
    return {"operation": "infer", "payload": {"value": value, "nested": {"labels": ["caller"]}},
            "random_state": initial_state()}


def test_real_owner_keeps_request_output_and_rng_isolation(factory):
    with local_model_pool() as pool:
        first = acquire(pool, factory)
        request = infer_request(3)
        expected = deepcopy(request)
        result = first.request(request)
        assert request == expected
        assert factory.owners[0].last_request["payload"]["nested"]["labels"] == ["caller", "kernel"]
        first.shutdown()
        second = acquire(pool, factory)
        next_request = infer_request(8)
        next_request["random_state"]["torch"] = [0, 0]
        later = second.request(next_request)
        assert factory.owners[0].last_request["random_state"] == next_request["random_state"]
        assert later["result"] == [8, 9]
        assert result["result"] == [3, 4]
        later["result"][0] = -1
        assert factory.owners[0].output_buffer == [8, 9]
        assert result["result"] == [3, 4]


@pytest.mark.parametrize("direction", ["input", "output"])
def test_real_owner_frame_bound_is_not_bypassed(factory, direction):
    with local_model_pool() as pool:
        borrowed = acquire(pool, factory)
        request = infer_request(1)
        if direction == "input":
            request["payload"]["oversized"] = "x" * 40000
        else:
            request["payload"]["value"] = "x" * 15000
            # Keep input bounded but force the handler's output over the bound.
            factory.owners[0].handle = lambda message: {"ok": True, "result": "x" * 40000}
        with pytest.raises(FrameTooLarge):
            borrowed.request(request)
        assert pool.metadata["quarantined"]


@pytest.mark.parametrize("failure", [
    RuntimeError("request failed"), RequestOutcomeUnknown("outcome unknown"),
    {"ok": False, "error": "failed"}, None,
])
def test_failed_or_unknown_request_never_retries_or_replaces(factory, failure):
    with local_model_pool(max_models=2) as pool:
        borrowed = acquire(pool, factory)
        real = factory.owners[0]
        if failure is None:
            real.handle = lambda request: None
        else:
            real.fail = failure
        calls = real.request_calls
        if isinstance(failure, BaseException) or failure is None:
            error_type = type(failure) if failure is not None else TypeError
            with pytest.raises(error_type):
                borrowed.request(infer_request(1))
        else:
            assert borrowed.request(infer_request(1)) == failure
        assert real.request_calls == calls + (failure is not None)
        with pytest.raises(RuntimeError, match="quarantined"):
            acquire(pool, factory, identity="another-identity")
        with pytest.raises(RuntimeError, match="quarantined"):
            borrowed.request(infer_request(2))
        assert len(factory.owners) == 1
    assert real.shutdown_calls == 1


@pytest.mark.parametrize("phase", ["construction", "describe"])
def test_failed_load_keeps_registered_owner_until_pool_finally(factory, phase):
    setattr(factory, f"{phase}_error", True)
    with local_model_pool() as pool:
        with pytest.raises(RuntimeError):
            acquire(pool, factory)
        assert factory.owners[0].shutdown_calls == 0
        with pytest.raises(RuntimeError, match="quarantined"):
            acquire(pool, factory)
    assert factory.owners[0].shutdown_calls == 1


def test_main_thread_and_process_boundaries(factory):
    with local_model_pool() as pool:
        borrowed = acquire(pool, factory)
        errors = []
        def background():
            for operation in (get_local_model_pool, lambda: acquire(pool, factory),
                              lambda: borrowed.request(infer_request(1)), borrowed.shutdown,
                              pool.close):
                try:
                    operation()
                except ValueError as error:
                    errors.append(str(error))
        thread = threading.Thread(target=background)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert len(errors) == 5 and all("main thread" in message for message in errors)
        pid = pool._pid
        pool._pid = -1  # simulate an inherited pool in a different process
        try:
            assert get_local_model_pool() is None
            with pytest.raises(RuntimeError, match="another process"):
                borrowed.request(infer_request(1))
            with pytest.raises(RuntimeError, match="another process"):
                pool.close()
        finally:
            pool._pid = pid
        assert factory.owners[0].shutdown_calls == 0


def test_context_exception_closes_once_and_does_not_unregister(factory, monkeypatch):
    from astrolabe.runtime.admission import ResourceClient
    def forbidden(*args, **kwargs):
        raise AssertionError("pool cannot acknowledge resident release")
    monkeypatch.setattr(ResourceClient, "unregister_model", forbidden)
    with pytest.raises(LookupError, match="scientific error"):
        with local_model_pool() as pool:
            acquire(pool, factory)
            raise LookupError("scientific error")
    assert factory.owners[0].shutdown_calls == 1


def test_close_attempts_all_owners_and_preserves_original_error(factory):
    with pytest.raises(LookupError, match="original"):
        with local_model_pool(max_models=2) as pool:
            acquire(pool, factory)
            acquire(pool, factory, identity="identity-B")
            factory.shutdown_error = True
            raise LookupError("original")
    assert [owner.shutdown_calls for owner in factory.owners] == [1, 1]
    pool.close()


def test_shutdown_failure_propagates_after_detaching_and_is_not_retried(factory):
    with pytest.raises(RuntimeError, match="failed to close"):
        with local_model_pool() as pool:
            acquire(pool, factory)
            factory.shutdown_error = True
    assert get_local_model_pool() is None
    pool.close()
    assert factory.owners[0].shutdown_calls == 1


def test_quarantine_cannot_be_bypassed_by_changing_scope(factory, monkeypatch):
    with local_model_pool() as pool:
        borrowed = acquire(pool, factory)
        factory.owners[0].fail = RequestOutcomeUnknown("unknown outcome")
        with pytest.raises(RequestOutcomeUnknown):
            borrowed.request(infer_request(1))
        monkeypatch.setenv("ASTROLABE_RESOURCE_SOCKET", "/changed/broker.sock")
        with pytest.raises(RuntimeError, match="quarantined"):
            acquire(pool, factory)
        assert len(factory.owners) == 1


def test_reentrant_execution_cannot_acquire_or_close_residency(factory):
    with local_model_pool() as pool:
        borrowed = acquire(pool, factory)
        real = factory.owners[0]
        def reentrant(request):
            with pytest.raises(RuntimeError, match="reentrant"):
                acquire(pool, factory)
            with pytest.raises(RuntimeError, match="during model execution"):
                pool.close()
            assert real.shutdown_calls == 0
            return {"ok": True, "result": 5}
        real.handle = reentrant
        assert borrowed.request(infer_request(1))["result"] == 5
        assert not pool.metadata["quarantined"]


def test_registration_outside_construction_still_belongs_to_run_scope(factory):
    assert not pool_owns_registration(object())
    with model_run_scope():
        real = factory(spec_for())
        assert not real.registered_to_pool
        assert real.shutdown_calls == 0
    assert real.shutdown_calls == 1


@pytest.mark.parametrize("state", [None, {}, [], "not-a-captured-state"])
def test_missing_initial_rng_declines_without_loading(factory, state):
    spec = spec_for()
    spec.settings["_initial_random_state"] = state
    with local_model_pool() as pool:
        assert acquire(pool, factory, spec) is None
        assert not factory.owners


def test_no_framework_import_or_rng_change_and_no_nested_context(factory, monkeypatch):
    original_import = builtins.__import__
    def guard(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "numpy"}:
            raise AssertionError("pool imported a scientific framework")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guard)
    before = random.getstate()
    with local_model_pool() as pool:
        acquire(pool, factory)
        assert acquire(pool, factory) is not None
        with pytest.raises(RuntimeError, match="nested"):
            with local_model_pool():
                pass
    assert random.getstate() == before


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_capacity(value):
    with pytest.raises(ValueError, match="positive integer"):
        local_model_pool(max_models=value)
