"""AF3 factory integration with vendor doubles; no model/GPU qualification."""
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace
import sys
import random
from omegaconf import OmegaConf
import threading
import numpy as np
import pytest
from astrolabe import model_execution as execution
from astrolabe.evaluators import native_daemons as native
from astrolabe.inference_pipeline import Pending, Stage

class Broker:
    def __init__(self):
        self.socket_path = "/fake/broker"
        self.active = []
        self.events = []
        self.models = []
        self.leases = []

    @contextmanager
    def lease(self, *, resource, phase, **kwargs):
        self.leases.append(dict(resource=resource, phase=phase, **kwargs))
        entry = (resource, phase)
        self.active.append(entry)
        self.events.append(("enter", entry))
        try:
            yield
        finally:
            self.active.remove(entry)
            self.events.append(("exit", entry))

    def register_model(self, model_id, **kwargs):
        self.models.append((model_id, kwargs))

@pytest.fixture
def broker(monkeypatch):
    broker = Broker()
    monkeypatch.delenv("ASTROLABE_RESOURCE_SOCKET", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    torch = sys.modules.get("torch")
    if torch is not None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(execution, "_resource_client", lambda: broker)
    return broker

def vendor(monkeypatch, name, **attributes):
    monkeypatch.setitem(sys.modules, name, SimpleNamespace(**attributes))

def call(handler, payload):
    return handler(dict(operation="infer", payload=payload,
                        random_state=execution.capture_random_state()))

def lifecycle(handler):
    return handler({"operation": "describe"})["metadata"]["runtime_lifecycle"]

def af3_capacity_broker(monkeypatch, broker):
    """The AF3 stage contract borrows a real token from its memory lease."""
    original = broker.lease

    @contextmanager
    def lease(**fields):
        with original(**fields):
            yield SimpleNamespace(token="af3-request-capacity", device=fields.get("device"))

    monkeypatch.setattr(broker, "lease", lease)

def af3_factory(monkeypatch, broker, completion, *, config=None):
    from astrolabe.evaluators import af3_service
    af3_capacity_broker(monkeypatch, broker)

    dispatched = threading.Event()
    materialized = []

    def stages(key):
        def prepare(features):
            assert broker.active == [("gpu", "evaluation_memory"), ("gpu", "prepare")]
            return {"features": features}

        def upload(value):
            assert broker.active == [("gpu", "evaluation_memory"), ("gpu", "prepare")]
            return Pending(value, lambda: None)

        def dispatch(value):
            assert broker.active == [("gpu", "evaluation_memory"), ("gpu", "infer")]
            dispatched.set()
            return Pending(value, completion, keepalive=value["features"])

        def materialize(value):
            assert broker.active == [("gpu", "evaluation_memory"), ("gpu", "finalize")]
            materialized.append(key)
            return {"coordinates": np.asarray(value["features"], dtype=np.float32), "__identifier__": b"native"}

        return [Stage("af3_features", "cpu", prepare), Stage("af3_upload", "transfer", upload),
                Stage("af3_model", "gpu", dispatch), Stage("af3_materialize", "cpu", materialize)]

    loads = []
    runner = SimpleNamespace(_device="fake-jax-gpu", inference_stages=stages,
                             _model_config=SimpleNamespace(global_config=SimpleNamespace()))
    monkeypatch.setattr(af3_service, "load_runner", lambda config: loads.append(config) or runner)
    vendor(monkeypatch, "astrolabe.evaluators.wrappers.alphafold3", service_model_settings=lambda config: {"fixed": True})
    handler = native._af3_factory({"config": config or {}, "service_settings": {"fixed": True}})
    request = dict(model="alphafold3", settings={"fixed": True}, features=[1, 2], rng_key=7,
                   cyclic_asym_ids=[], cyclic_relative_patches=[])
    return handler, request, dispatched, materialized, loads

def test_af3_native_pending_keeps_inference_until_completion_and_capacity_until_finalize(monkeypatch, broker):
    completion = Future()
    handler, request, dispatched, materialized, loads = af3_factory(monkeypatch, broker, completion)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(call, handler, request)
        try:
            assert dispatched.wait(3)
            assert not pending.done() and not materialized
            assert broker.active == [("gpu", "evaluation_memory"), ("gpu", "infer")]
        finally:
            completion.set_result(None)
        result = pending.result(timeout=3)
    assert result["result"]["__identifier__"] == b"native"
    assert result["result"]["coordinates"].dtype == np.float32
    assert materialized == [7] and len(loads) == 1 and not broker.active
    invalid = call(handler, dict(request, settings={"fixed": False}))
    assert not invalid["ok"] and "settings" in invalid["error"]

def test_af3_native_pending_failure_releases_slot_without_materializing(monkeypatch, broker):
    completion = Future()
    completion.set_exception(RuntimeError("device failed"))
    handler, request, _, materialized, loads = af3_factory(monkeypatch, broker, completion)
    result = call(handler, request)
    assert not result["ok"] and "device failed" in result["error"]
    assert not materialized and not broker.active and len(loads) == 1
    assert lifecycle(handler)["failures"] == 1


@pytest.mark.parametrize("changed", [
    {"model": "other"}, {"settings": {"fixed": False}},
    {"cyclic_asym_ids": [1]}, {"cyclic_relative_patches": [[1, 2]]},
])
def test_af3_protocol_rejection_does_not_admit_or_quarantine_owner(monkeypatch, broker, changed):
    completion = Future()
    completion.set_result(None)
    handler, request, _, materialized, loads = af3_factory(monkeypatch, broker, completion)
    before = list(broker.leases)
    state = execution.capture_random_state()
    rejected = handler({
        "operation": "infer", "payload": dict(request, **changed), "random_state": state,
    })
    assert not rejected["ok"]
    assert rejected["random_state"] == state
    assert broker.leases == before and not materialized
    assert call(handler, request)["ok"]
    assert len(loads) == 1 and materialized == [7]
    handler.close()


def test_af3_configured_gpu_ordinal_is_admitted_for_load_and_execution(monkeypatch, broker):
    complete = Future()
    complete.set_result(None)
    handler, request, _, _, _ = af3_factory(monkeypatch, broker, complete, config={"gpu_device": 2})
    assert call(handler, request)["ok"]
    leases = [lease for lease in broker.leases if lease["resource"] == "gpu"]
    assert [lease["phase"] for lease in leases] == ["load", "evaluation_memory", "prepare", "infer", "finalize"]
    assert all(lease["device"] == "cuda:2" and not lease["concurrency_certified"] for lease in leases)


def af3_private_protocol(monkeypatch, broker):
    """Public facade -> real factory/kernel protocol -> original sync adapter."""
    from astrolabe.evaluators import af3_service
    af3_capacity_broker(monkeypatch, broker)
    loaded = []

    def tree_map(function, value):
        if isinstance(value, dict):
            return {key: tree_map(function, item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return type(value)(tree_map(function, item) for item in value)
        return function(value)

    vendor(monkeypatch, "jax", tree=SimpleNamespace(map=tree_map))
    vendor(monkeypatch, "astrolabe.evaluators.wrappers.alphafold3",
           service_model_settings=lambda config: {"fixed": True})

    def draw():
        return np.array([random.random(), np.random.random()])

    class Runner:
        def __init__(self, config):
            assert broker.active == [("gpu", "load")]
            self._device = "TFRT_GPU_0"
            self._model_config = SimpleNamespace(global_config=SimpleNamespace(
                cyclic_asym_ids=tuple(config.get("cyclic_asym_ids", [])),
                cyclic_relative_patches=tuple(tuple(patch) for patch in config.get("cyclic_relative_patches", [])),
            ))
            self.weights = object()
            self.initial_draw = draw()
            self.stage_builds = 0
            self.calls = []
            self.staged = False
            loaded.append(self)

        def set_cyclic_asym_ids(self, ids):
            assert broker.active == [("gpu", "evaluation_memory"), ("gpu", "infer")]
            self._model_config.global_config.cyclic_asym_ids = tuple(int(value) for value in ids)

        def set_cyclic_relative_patches(self, patches):
            assert broker.active == [("gpu", "evaluation_memory"), ("gpu", "infer")]
            self._model_config.global_config.cyclic_relative_patches = tuple(tuple(int(value) for value in patch) for patch in patches)

        def run_inference(self, features, rng_key):
            assert broker.active == [("gpu", "evaluation_memory"), ("gpu", "infer")]
            config = self._model_config.global_config
            self.calls.append((config.cyclic_asym_ids, config.cyclic_relative_patches,
                               "staged" if self.staged else "monolithic", self.weights))
            values = draw()
            if features.get("fail", False):
                raise ValueError("fake forward failure")
            return dict(random=values, cyclic=np.asarray(config.cyclic_asym_ids, dtype=np.int32),
                        patches=np.asarray(config.cyclic_relative_patches, dtype=np.int32),
                        features=np.asarray(features["value"], dtype=np.float32),
                        rng_key=np.asarray(rng_key), __identifier__=b"native")

        def extract_structures(self, batch, result, target_name):
            from astrolabe.evaluators.wrappers.alphafold3 import ModelRunner
            return ModelRunner.extract_structures(self, batch, result, target_name)

        def inference_stages(self, key):
            self.stage_builds += 1

            def prepare(features):
                assert broker.active == [("gpu", "evaluation_memory"), ("gpu", "prepare")]
                return features

            def execute(features):
                self.staged = True
                try:
                    return self.run_inference(features, key)
                finally:
                    self.staged = False

            return (Stage("af3_features", "cpu", prepare), Stage("af3_model", "gpu", execute),
                    Stage("af3_materialize", "cpu", lambda result: result))

    monkeypatch.setattr(af3_service, "load_runner", Runner)

    class ProtocolClient:
        def __init__(self, backend, factory, settings, **kwargs):
            assert factory == "astrolabe.evaluators.native_daemons:_af3_factory"
            settings = dict(settings, _initial_random_state=execution.capture_random_state())
            # A supplied profile cannot certify unqualified AF3 inference.
            settings["_runtime_profile"] = {"identity": "alphafold3", "concurrency_certified": True}
            self.handler = native._af3_factory(settings)
            description = self.handler({"operation": "describe"})
            self.metadata = description["metadata"]
            execution.restore_random_state(description["random_state"])

        def request(self, payload):
            response = call(self.handler, execution.host_result(payload))
            execution.restore_random_state(response["random_state"])
            if not response["ok"]:
                raise RuntimeError(f"{response['error_type']}: {response['error']}")
            return response["result"]

    monkeypatch.setattr(native, "_KernelClient", ProtocolClient)
    return af3_service, loaded


def test_af3_remote_seed_and_extraction_do_not_initialize_caller_jax(monkeypatch):
    requests = []
    # Backend enumeration and PRNG generation are deliberately unavailable in
    # this coordinator. Tree traversal alone does not initialize a device.
    vendor(monkeypatch, "jax", tree=SimpleNamespace(map=lambda f, d: {
        key: f(value) for key, value in d.items()}))
    runner = native.NativeAF3Runner.__new__(native.NativeAF3Runner)
    runner._client = SimpleNamespace(request=lambda request: requests.append(request))
    runner._settings = {"fixed": True}
    runner._cyclic_asym_ids = ()
    runner._cyclic_relative_patches = ()
    runner.run_inference_for_seed({"value": np.ones(3)}, 17)
    runner.extract_structures({"batch": 1}, {"result": 2}, "target")
    runner.run_inference_and_extract_for_seed({"value": np.ones(3)}, 19, "target")
    assert requests[0]["rng_seed"] == 17 and "rng_key" not in requests[0]
    assert requests[1] == dict(operation="extract_structures", batch={"batch": 1},
                               result={"result": 2}, target_name="target")
    assert requests[2]["rng_seed"] == 19 and requests[2]["target_name"] == "target"
    assert "rng_key" not in requests[2]


def test_af3_seed_key_and_native_extraction_belong_to_loaded_owner(monkeypatch, broker):
    _, loaded = af3_private_protocol(monkeypatch, broker)
    jax = sys.modules["jax"]

    @contextmanager
    def cpu_scope(device):
        assert device == "cpu" and broker.active == [("gpu", "evaluation_memory"), ("gpu", "prepare")]
        assert len(loaded) == 1
        yield

    jax.local_devices = lambda backend: ["cpu"] if backend == "cpu" else []
    jax.default_device = cpu_scope
    jax.random = SimpleNamespace(PRNGKey=lambda seed: np.asarray([0, seed], dtype=np.uint32))
    def extract(owner, batch, result, target_name):
        assert owner is loaded[0] and broker.active == [("gpu", "evaluation_memory"), ("gpu", "finalize")]
        return [{"batch": batch, "result": result, "target": target_name}]
    sys.modules["astrolabe.evaluators.wrappers.alphafold3"].ModelRunner = SimpleNamespace(
        extract_structures=extract)
    runner = native.NativeAF3Runner(OmegaConf.create({}))
    initial = execution.capture_random_state()
    seeded = runner.run_inference_for_seed({"value": [1, 2]}, 17)
    execution.restore_random_state(initial)
    explicit = runner.run_inference({"value": [1, 2]}, np.asarray([0, 17], dtype=np.uint32))
    for key in explicit:
        np.testing.assert_equal(seeded[key], explicit[key])
    result = runner.extract_structures({"x": 1}, {"y": 2}, "A")
    assert result == [{"batch": {"x": 1}, "result": {"y": 2}, "target": "A"}]
    assert len(loaded) == 1 and not broker.active


def test_native_af3_cyclic_mismatch_rejects_without_mutation_or_fallback(monkeypatch, broker):
    _, loaded = af3_private_protocol(monkeypatch, broker)
    runner = native.NativeAF3Runner(OmegaConf.create({}))
    runner.set_cyclic_asym_ids([1])
    runner.set_cyclic_relative_patches([[1, 2]])
    with pytest.raises(RuntimeError, match="immutable owner cyclic"):
        runner.run_inference({"value": [1, 2]}, np.asarray([0, 7], dtype=np.uint32))
    model = loaded[0]
    assert model.stage_builds == 0 and model.calls == []
    assert model._model_config.global_config.cyclic_asym_ids == ()
    metadata = runner._client.handler({"operation": "describe"})["metadata"]
    assert metadata["owner_configuration"] == "immutable_cyclic"
    assert "cyclic_fallback" not in metadata and not broker.active


def test_native_af3_compatible_prebound_cyclic_owner_retains_stages(monkeypatch, broker):
    _, loaded = af3_private_protocol(monkeypatch, broker)
    runner = native.NativeAF3Runner(OmegaConf.create({"cyclic_asym_ids": [1], "cyclic_relative_patches": [[1, 2]]}))
    assert runner._cyclic_asym_ids == (1,)
    assert runner._cyclic_relative_patches == ((1, 2),)
    for ids, patches in [([1], [[1, 2]]), ([1], [[1, 2]])]:
        runner.set_cyclic_asym_ids(ids)
        runner.set_cyclic_relative_patches(patches)
        result = runner.run_inference({"value": [1, 2]}, np.asarray([0, 7], dtype=np.uint32))
        np.testing.assert_array_equal(result["cyclic"], ids)
        np.testing.assert_array_equal(result["patches"], patches)
    assert [call[2] for call in loaded[0].calls] == ["staged", "staged"]
    assert lifecycle(runner._client.handler)["loader_count"] == 1 and len(loaded) == 1
    runner.set_cyclic_asym_ids([2])
    with pytest.raises(RuntimeError, match="immutable owner cyclic"):
        runner.run_inference({"value": [1, 2]}, np.asarray([0, 7], dtype=np.uint32))
    assert loaded[0]._model_config.global_config.cyclic_asym_ids == (1,)
    assert not broker.active


def test_af3_service_load_finishes_weight_placement_before_readiness(monkeypatch):
    from astrolabe.evaluators import af3_service

    events = []

    class Runner:
        _device = SimpleNamespace(platform="gpu")

        @property
        def model_params(self):
            events.append("params")
            return {}

        @property
        def _model(self):
            events.append("placed_and_ready")
            return object()

    runner = Runner()
    vendor(monkeypatch, "jax", local_devices=lambda **kwargs: [runner._device], __version__="test")
    vendor(monkeypatch, "jaxlib", __version__="test")
    vendor(monkeypatch, "astrolabe.evaluators.wrappers.alphafold3",
           alphafold3_setup=lambda _: runner)
    monkeypatch.setattr(af3_service, "mark", lambda event, **kwargs: events.append(event))
    assert af3_service.load_runner(SimpleNamespace()) is runner
    assert events == ["model_load_start", "params", "placed_and_ready", "model_load_end"]
