import random

import numpy as np
import pytest

from astrolabe.model_execution import (
    capture_random_state, host_result, kernel_handler, restore_random_state,
    model_run_scope, register_model_owner,
    reset_remote_random_state,
)


def test_kernel_preserves_constructor_and_interleaved_request_random_streams():
    random.seed(77)
    np.random.seed(77)
    initial = capture_random_state()
    loads = []

    def loader(settings):
        loads.append((random.random(), np.random.random()))
        def infer(payload):
            result = (random.random(), float(np.random.random()))
            if payload.get("fail"):
                raise ValueError("request failed after random draws")
            return result
        return infer, {"model": "fixture"}

    random.seed(77)
    np.random.seed(77)
    reference_constructor = (random.random(), np.random.random())
    expected_first = (random.random(), float(np.random.random()))
    expected_second = (random.random(), float(np.random.random()))
    handler = kernel_handler({"_initial_random_state": initial}, loader)
    initialized = handler({"operation": "describe"})["random_state"]
    assert loads == [reference_constructor]
    first = handler({"operation": "infer", "payload": {}, "random_state": initialized})
    random.seed(999)
    np.random.seed(999)
    other = handler({"operation": "infer", "payload": {}, "random_state": capture_random_state()})
    assert other["result"] != first["result"]
    second = handler({"operation": "infer", "payload": {}, "random_state": first["random_state"]})
    assert first["result"] == expected_first
    assert second["result"] == expected_second
    assert len(loads) == 1


def test_failure_returns_advanced_random_state_without_repeating_the_kernel():
    np.random.seed(13)
    state = capture_random_state()
    handler = kernel_handler({}, lambda settings: (lambda payload: (_ for _ in ()).throw(ValueError("bad input")), {}))
    failed = handler({"operation": "infer", "payload": {}, "random_state": state})
    assert failed["ok"] is False and failed["error_type"] == "ValueError"
    restore_random_state(failed["random_state"])


def test_host_materialization_preserves_array_dtype_and_shape():
    values = np.arange(6, dtype=np.int32).reshape(2, 3)
    result = host_result({"features": values})
    np.testing.assert_array_equal(result["features"], values)
    assert result["features"].dtype == np.int32


def test_kernel_rejects_unknown_operations():
    handler = kernel_handler({}, lambda settings: (lambda payload: payload, {}))
    with pytest.raises(ValueError, match="Unknown kernel operation"):
        handler({"operation": "arbitrary_method"})


def test_run_scope_releases_partial_startup_and_keeps_outer_owners_alive():
    stopped = []
    class Owner:
        def __init__(self, name, attached=False):
            self.name = name
            self.attached = attached
        def shutdown(self):
            stopped.append(self.name)

    with model_run_scope():
        outer = Owner("outer")
        register_model_owner(outer)
        with pytest.raises(ValueError, match="later component failed"):
            with model_run_scope():
                inner = Owner("inner")
                register_model_owner(inner)
                register_model_owner(inner)
                register_model_owner(Owner("external", attached=True))
                raise ValueError("later component failed")
        assert stopped == ["inner"]
    assert stopped == ["inner", "outer"]


def test_remote_cuda_state_stays_opaque_and_does_not_queue_device_callbacks(monkeypatch):
    import torch

    reset_remote_random_state()
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    def forbidden(*args):
        pytest.fail("The CPU coordinator touched a CUDA generator")
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", forbidden)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", forbidden)
    state = capture_random_state()
    state["cuda"] = [[1, 2, 3]]
    try:
        restore_random_state(state)
        assert capture_random_state()["cuda"] == [[1, 2, 3]]
        state["cuda"] = [[4, 5, 6]]
        restore_random_state(state)
        assert capture_random_state()["cuda"] == [[4, 5, 6]]
        assert torch.initial_seed() == state["torch_seed"]
    finally:
        reset_remote_random_state()


def test_same_seed_restarts_the_remote_cuda_stream(monkeypatch):
    import torch
    from astrolabe.argparser import set_random_seeds

    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)
    set_random_seeds(17)
    state = capture_random_state()
    state["cuda"] = [[1, 2, 3]]
    restore_random_state(state)
    assert "cuda" in capture_random_state()
    set_random_seeds(17)
    assert "cuda" not in capture_random_state()


def test_remote_cuda_continues_after_constructor_interleaving_and_failure(monkeypatch):
    import torch

    reset_remote_random_state()
    active = {"process": "parent"}
    device_generators = {name: torch.Generator() for name in ("one", "two")}
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: active["process"] != "parent")
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [
        device_generators[active["process"]].get_state()
    ])
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda states:
        device_generators[active["process"]].set_state(states[0]))
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed:
        device_generators[active["process"]].manual_seed(seed))
    torch.set_rng_state(torch.Generator().manual_seed(17).get_state())
    reference = torch.Generator().manual_seed(17)
    def draw():
        return torch.rand(1, generator=device_generators[active["process"]]).item()
    def loader(settings):
        assert draw() == torch.rand(1, generator=reference).item()
        def infer(payload):
            value = draw()
            if payload.get("fail"):
                raise ValueError("after CUDA draw")
            return value
        return infer, {}
    handlers = {}
    try:
        for name in ("one", "two"):
            state = capture_random_state()
            active["process"] = name
            handlers[name] = kernel_handler({"_initial_random_state": state}, loader)
            description = handlers[name]({"operation": "describe"})
            active["process"] = "parent"
            restore_random_state(description["random_state"])
        for name, fail in (("one", False), ("two", True), ("one", False)):
            state = capture_random_state()
            active["process"] = name
            response = handlers[name]({"operation": "infer", "random_state": state,
                                       "payload": {"fail": fail}})
            expected = torch.rand(1, generator=reference).item()
            assert response["ok"] is (not fail)
            if not fail:
                assert response["result"] == expected
            active["process"] = "parent"
            restore_random_state(response["random_state"])
    finally:
        reset_remote_random_state()


def test_cpu_owner_preserves_incoming_cuda_and_observes_same_seed_constructor_reset(monkeypatch):
    import torch

    reset_remote_random_state()
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    seed_calls = []
    monkeypatch.setattr(torch.cuda, "manual_seed_all", seed_calls.append)
    incoming = capture_random_state()
    incoming["cuda"] = [[1, 2, 3]]
    try:
        handler = kernel_handler({"_initial_random_state": incoming},
                                 lambda settings: (lambda payload: "cpu", {}))
        description = handler({"operation": "describe"})
        assert description["random_state"]["cuda"] == incoming["cuda"]
        response = handler({"operation": "infer", "payload": {},
                            "random_state": description["random_state"]})
        assert response["random_state"]["cuda"] == incoming["cuda"]
        assert seed_calls == []
        original_seed_function = torch.cuda.manual_seed_all
        def reseeded_loader(settings):
            torch.manual_seed(incoming["torch_seed"])
            def fail(payload):
                torch.manual_seed(incoming["torch_seed"])
                raise ValueError("after same-seed reset")
            return fail, {}
        reset_handler = kernel_handler({"_initial_random_state": incoming}, reseeded_loader)
        description = reset_handler({"operation": "describe"})
        assert "cuda" not in description["random_state"]
        assert description["random_state"]["cuda_seed"] == incoming["torch_seed"]
        failed = reset_handler({"operation": "infer", "payload": {}, "random_state": incoming})
        assert failed["ok"] is False
        assert "cuda" not in failed["random_state"]
        assert failed["random_state"]["cuda_seed"] == incoming["torch_seed"]
        assert torch.cuda.manual_seed_all is original_seed_function
    finally:
        reset_remote_random_state()


def test_cuda_child_initializes_before_restoring_advanced_state(monkeypatch):
    import torch

    reset_remote_random_state()
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    initialized = {"value": False}
    device_generator = torch.Generator().manual_seed(17)
    torch.rand(3, generator=device_generator)
    advanced = device_generator.get_state()
    expected = torch.rand(1, generator=device_generator).item()
    incoming = capture_random_state()
    incoming["cuda"] = [advanced.tolist()]
    incoming["cuda_visible_devices"] = None
    events = []
    def initialize():
        events.append("init")
        device_generator.manual_seed(999)
        initialized["value"] = True
    def restore(states):
        events.append("restore")
        assert initialized["value"]
        device_generator.set_state(states[0])
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: initialized["value"])
    monkeypatch.setattr(torch.cuda, "init", initialize)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", restore)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [device_generator.get_state()])
    def loader(settings):
        assert torch.rand(1, generator=device_generator).item() == expected
        return lambda payload: None, {}
    try:
        kernel_handler({"_initial_random_state": incoming}, loader, torch_device="cuda")
        assert events == ["init", "restore"]
    finally:
        reset_remote_random_state()
