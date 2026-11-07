"""Bounded independent review: CPU invariants, synthetic GPU telemetry only."""
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import uuid

import pytest

from astrolabe import model_execution as execution
from astrolabe.runtime import admission, session


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    import torch

    for name in ("ASTROLABE_RESOURCE_SOCKET", "ASTROLABE_RESOURCE_PROFILES",
                 "SLURM_JOB_GPUS", "SLURM_STEP_GPUS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "init", lambda: pytest.fail("GPU initialization forbidden"))
    execution.reset_remote_random_state()
    yield
    execution.reset_remote_random_state()


@pytest.fixture
def broker_factory(tmp_path, monkeypatch):
    brokers = []

    def start(*, gpu=True, ram_bytes=1000):
        rows = [dict(index=i, uuid="GPU-review-" + uuid.uuid4().hex,
                     total_bytes=1000, free_bytes=1000, utilization=0,
                     processes=[]) for i in range(2)] if gpu else []
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(r["uuid"] for r in rows))
        directory = tmp_path / str(len(brokers))
        directory.mkdir(mode=0o700)
        owner = admission.ResourceBroker({
            "devices": [dict(index=r["index"], uuid=r["uuid"], vram_bytes=800)
                        for r in rows],
            "socket_dir": str(directory), "cpu_slots": 2,
            "ram_bytes": ram_bytes, "output_bytes": 1000,
            "sample_interval": 0.02, "sample_max_age": 2,
            "shutdown_timeout": 0.5,
        }, sampler=lambda: copy.deepcopy(rows)).start()
        brokers.append(owner)
        return owner, admission.ResourceClient(owner.socket_path), rows

    yield start
    for owner in reversed(brokers):
        # All work in this module is CPU simulation and has already returned.
        # Explicitly acknowledge completion if a failed invariant left a lease.
        client = admission.ResourceClient(owner.socket_path)
        for token in list(owner._leases):
            client._rpc(dict(op="finish", token=token))
        for model_id in list(owner._models):
            client.unregister_model(model_id)
        owner.close()


@pytest.mark.parametrize("explicit_budget", [None, 800])
def test_allocated_default_starts_broker_with_bounded_capacity(tmp_path, monkeypatch, explicit_budget):
    device = "GPU-review-" + uuid.uuid4().hex
    rows = [dict(index=0, uuid=device, total_bytes=1000, free_bytes=1000,
                 utilization=0, processes=[])]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", device)
    monkeypatch.setattr(admission, "NvidiaSMISampler", lambda *args: lambda: rows)
    tmp_path.chmod(0o700)
    config = dict(mode="adaptive", devices="allocated", socket_dir=str(tmp_path))
    if explicit_budget is not None:
        config["vram_bytes"] = explicit_budget
    with session.resource_session(config) as client:
        state = client.snapshot()
        assert state["healthy"]
        budget = state["devices"][device]["vram_bytes"]
        assert 0 < budget <= rows[0]["free_bytes"]
        if explicit_budget is not None:
            assert budget == explicit_budget


def test_default_esmfold_factory_can_reach_loader_with_real_broker(broker_factory, monkeypatch):
    from astrolabe.evaluators import native_daemons

    broker_factory()
    loaded = []

    class Evaluator:
        def __init__(self, **kwargs):
            self._device = "cuda:0"
            self._load_metadata = {"model_state": {}}

        def _prepare_model_cpu(self):
            loaded.append(True)
            return SimpleNamespace(infer=lambda payload: payload,
                                   output_to_pdb=lambda payload: payload)

        def _place_model(self, prepared):
            return prepared

    monkeypatch.setitem(sys.modules, "astrolabe.evaluators.esmfold",
                        SimpleNamespace(ESMFoldEvaluator=Evaluator))
    native_daemons._esmfold_factory(dict(model_name="fixture"))
    assert loaded


def test_local_evaluator_nonzero_device_fails_closed_before_wrong_lease(broker_factory):
    from astrolabe.evaluators.runtime_stages import local_evaluator_guard

    owner, _, _ = broker_factory()

    class Evaluator:
        _device = "cuda:1"
        _model = object()

        @local_evaluator_guard("esmfold", "_model")
        def run_function(self, payload):
            # Stand in for the real model's configured cuda:1 execution, while
            # inspecting the physical device actually admitted by the broker.
            return next(iter(owner._leases.values()))["device"]

    with pytest.raises(ValueError, match="nonzero GPU"):
        Evaluator().run_function({})
    assert not owner._leases


@pytest.fixture
def profile_port(tmp_path, monkeypatch):
    """Exercise real certificate/client/kernel code with CPU model/lease stubs."""
    from astrolabe import model_daemon

    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"selected-weight-contents")
    settings = dict(checkpoint=str(weights), device="cuda:0")
    used_device = SimpleNamespace(value="cuda:0")
    payload = dict(features={"length": 8})
    gpu = "GPU-review-" + uuid.uuid4().hex
    second_gpu = "GPU-review-" + uuid.uuid4().hex
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", f"{gpu},{second_gpu}")
    hardware_rows = f"0, {gpu}, review-driver\n1, {second_gpu}, review-driver\n"

    def gpu_inventory(command, **kwargs):
        assert command[:2] == ["nvidia-smi", "--query-gpu=index,uuid,driver_version"]
        return hardware_rows

    monkeypatch.setattr(execution.subprocess, "check_output", gpu_inventory)
    events = []
    owners = []
    leases = []
    admitted_device = SimpleNamespace(value=None)
    backend_calls = {"load": 0, "execute": 0, "register": 0}

    class CPUResourceClient:
        socket_path = "/cpu-review-profile"

        @contextmanager
        def lease(self, **request):
            events.append(dict(request))
            acquired = SimpleNamespace(
                device=(admitted_device.value or
                        (second_gpu if request.get("device") == "cuda:1" else gpu)),
                finished=False, finish_calls=0, fenced=False,
                finished_before_exception=False,
            )
            def finish():
                acquired.finish_calls += 1
                acquired.finished = True
            acquired.finish = finish
            leases.append(acquired)
            try:
                yield acquired
            except BaseException:
                # Model the real Lease contract: an unfinished exceptional GPU
                # lease fences; completion before rejection must avoid that.
                acquired.finished_before_exception = acquired.finished
                acquired.fenced = request["resource"] == "gpu" and not acquired.finished
                raise
            else:
                acquired.finish()

        def register_model(self, *args, **kwargs):
            backend_calls["register"] += 1

        def unregister_model(self, *args, **kwargs):
            pass

    resource_client = CPUResourceClient()
    monkeypatch.setattr(execution, "_resource_client", lambda: resource_client)

    class CPUOwner:
        attached = False

        def __init__(self, spec, **kwargs):
            owners.append(self)
            self.spec = spec
            def load(config):
                backend_calls["load"] += 1
                def infer(request):
                    backend_calls["execute"] += 1
                    return request
                return infer, {"device": used_device.value}
            with execution.model_kernel_context(
                    spec.backend, model_daemon.model_identity(spec)):
                self.handle = execution.kernel_handler(
                    spec.settings, load,
                    lifecycle={"device": used_device.value})

        def request(self, request):
            return self.handle(request)

        def shutdown(self):
            pass

    monkeypatch.setattr(model_daemon, "OwnedModelDaemon", CPUOwner)
    factory = "astrolabe.model_execution:kernel_handler"
    spec = model_daemon.DaemonSpec(
        backend="fixture", factory=factory, settings=settings,
        python_bin=sys.executable, max_frame_bytes=execution.DEFAULT_MODEL_FRAME_BYTES)
    identity = model_daemon.model_identity(spec)
    source_paths = [Path(execution.__file__), Path(model_daemon.__file__),
                    Path(admission.__file__), Path(session.__file__)]

    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    profile = dict(
        identity=identity, verified=True, resident_bytes=10, scratch_bytes=10,
        concurrency_certified=True, shapes=[execution.resource_shape(payload)],
        asset_sha256={str(path): digest(path) for path in source_paths},
        weights_sha256={str(weights): digest(weights)},
        hardware=dict(gpu_uuid=gpu, driver_version="review-driver"),
        environment=dict(python_abi=sys.implementation.cache_tag,
                         python_version=sys.version.split()[0],
                         distributions={"torch": importlib.metadata.version("torch")}))
    profiles = tmp_path / "profiles.json"

    def construct(*, certificate_identity=None, **kwargs):
        selected_spec = model_daemon.DaemonSpec(
            backend="fixture", factory=factory, settings=settings,
            python_bin=kwargs.get("python_bin") or sys.executable,
            env=kwargs.get("env"), cwd=kwargs.get("cwd"), provider=kwargs.get("provider"),
            max_frame_bytes=execution.DEFAULT_MODEL_FRAME_BYTES)
        selected_identity = model_daemon.model_identity(
            model_daemon._prepare_provider_spec(selected_spec))
        profile["identity"] = certificate_identity or selected_identity
        profiles.write_text(json.dumps(dict(profiles={profile["identity"]: profile})))
        monkeypatch.setenv("ASTROLABE_RESOURCE_PROFILES", str(profiles))
        return execution.NativeModelClient("fixture", factory, settings, **kwargs)

    return SimpleNamespace(profile=profile, construct=construct, payload=payload,
                           weights=weights, events=events, owners=owners,
                           leases=leases, admitted_device=admitted_device,
                           backend_calls=backend_calls,
                           gpu=gpu, second_gpu=second_gpu, settings=settings,
                           used_device=used_device, spec=spec)


def test_valid_automatic_profile_cannot_certify_a_different_request_shape(profile_port):
    port = profile_port
    client = port.construct()
    assert client.request(port.payload) == port.payload
    known = [event for event in port.events if event["phase"] == "execute"]
    assert known[-1]["concurrency_certified"] is True
    assert known[-1]["scratch_bytes"] == 10
    before = len(port.events)
    outside = dict(features={"length": 128})
    try:
        client.request(outside)
    except ValueError:
        assert not any(event["phase"] == "execute" for event in port.events[before:])
    else:
        fallback = [event for event in port.events[before:]
                    if event["resource"] == "gpu" and event["phase"] == "execute"]
        assert fallback, "Outside-scope work ran without GPU admission"
        assert fallback[-1]["concurrency_certified"] is False
        assert fallback[-1]["scratch_bytes"] == 0  # Unknown-size exclusive fallback.
    client.shutdown()


@pytest.mark.parametrize("change", ["weights", "uuid", "driver", "source",
                                   "python_abi", "framework"])
def test_automatic_profile_rejects_changed_assets_or_hardware_before_owner(profile_port, change):
    port = profile_port
    if change == "weights":
        port.weights.write_bytes(b"different-weights-at-the-same-path")
    elif change == "uuid":
        port.profile["hardware"]["gpu_uuid"] = "GPU-unallocated"
    elif change == "driver":
        port.profile["hardware"]["driver_version"] = "another-driver"
    elif change == "source":
        port.profile["asset_sha256"][str(Path(execution.__file__))] = "0" * 64
    elif change == "python_abi":
        port.profile["environment"]["python_abi"] = "another-python-abi"
    else:
        port.profile["environment"]["distributions"]["torch"] = "another-framework-version"
    with pytest.raises(ValueError):
        port.construct()
    assert not port.owners


def test_automatic_profile_requires_actual_selected_source_evidence(profile_port):
    port = profile_port
    # A nonempty hash map for unrelated data is not implementation evidence.
    port.profile["asset_sha256"] = dict(port.profile["weights_sha256"])
    with pytest.raises(ValueError):
        port.construct()
    assert not port.owners


def test_profile_hardware_must_match_used_gpu_not_merely_another_visible_gpu(profile_port):
    port = profile_port
    # Both GPUs are in the allocation, but settings and the stub model use cuda:0.
    port.profile["hardware"]["gpu_uuid"] = port.second_gpu
    with pytest.raises(ValueError):
        port.construct()
    assert not port.owners


@pytest.mark.parametrize("nested,device_key,device", [
    ("factory_kwargs", "gpu_device", 1),
    ("config", "device", "cuda:1"),
    ("constructor", "device", "cuda:1"),
])
def test_profile_hardware_tracks_native_backend_nested_device(profile_port, nested, device_key, device):
    port = profile_port
    port.settings.clear()
    port.settings.update(checkpoint=str(port.weights), **{nested: {device_key: device}})
    port.used_device.value = "cuda:1"
    # This models AF2/AF3/ESM3's actual device lookup. A cuda:0 certificate
    # must not authorize a backend whose configured execution uses cuda:1.
    with pytest.raises(ValueError):
        port.construct()
    assert not port.owners


@pytest.mark.parametrize("scope", ["config", "factory_kwargs"])
@pytest.mark.parametrize("certificate_device", [0, 1])
def test_profile_rejects_conflicting_device_fields_before_owner(
        profile_port, scope, certificate_device):
    port = profile_port
    port.settings.clear()
    port.settings.update(checkpoint=str(port.weights), **{
        scope: {"device": "cuda:0", "gpu_device": 1}})
    port.used_device.value = "cuda:1"
    port.profile["hardware"]["gpu_uuid"] = (
        port.gpu if certificate_device == 0 else port.second_gpu)
    # Neither certificate may make an explicitly contradictory scope safe.
    with pytest.raises(ValueError):
        port.construct()
    assert not port.owners
    assert not port.events


@pytest.mark.parametrize("scope", ["config", "factory_kwargs"])
@pytest.mark.parametrize("certificate_device", [0, 1])
def test_profile_none_device_follows_explicit_gpu_device(
        profile_port, scope, certificate_device):
    from astrolabe.evaluators.runtime_stages import configured_device_hint

    port = profile_port
    config = {"device": None, "gpu_device": 1}
    port.settings.clear()
    port.settings.update(checkpoint=str(port.weights), **{scope: config})
    port.used_device.value = configured_device_hint(config)
    assert port.used_device.value == "cuda:1"
    port.profile["hardware"]["gpu_uuid"] = (
        port.gpu if certificate_device == 0 else port.second_gpu)
    if certificate_device == 0:
        with pytest.raises(ValueError):
            port.construct()
        assert not port.owners
        assert not port.events
        return
    client = port.construct()
    try:
        assert client.request(port.payload) == port.payload
        executed = [event for event in port.events if event["phase"] == "execute"]
        assert executed[-1]["device"] == "cuda:1"
        assert executed[-1]["concurrency_certified"] is True
    finally:
        client.shutdown()


@pytest.mark.parametrize("scope", ["config", "factory_kwargs"])
def test_profile_accepts_consistent_normalized_device_fields(profile_port, scope):
    port = profile_port
    port.settings.clear()
    port.settings.update(checkpoint=str(port.weights), **{
        scope: {"device": "gpu:1", "gpu_device": 1}})
    port.used_device.value = "cuda:1"
    port.profile["hardware"]["gpu_uuid"] = port.second_gpu
    client = port.construct()
    try:
        assert client.request(port.payload) == port.payload
        executed = [event for event in port.events if event["phase"] == "execute"]
        assert executed[-1]["device"] == "cuda:1"
        assert executed[-1]["concurrency_certified"] is True
    finally:
        client.shutdown()


@pytest.mark.parametrize("gpu_device", ["auto", "none"])
def test_profile_gpu_auto_cannot_certify_a_shadowed_device(profile_port, gpu_device):
    from astrolabe.evaluators.runtime_stages import configured_device_hint

    port = profile_port
    config = {"device": "cuda:1", "gpu_device": gpu_device}
    port.settings.clear()
    port.settings.update(checkpoint=str(port.weights), config=config)
    # A non-None gpu_device wins in the real helper, even when it asks for auto.
    # AF3's default is cuda:0; the shadowed device field cannot certify cuda:1.
    port.used_device.value = configured_device_hint(config, default="cuda:0")
    assert port.used_device.value == "cuda:0"
    port.profile["hardware"]["gpu_uuid"] = port.second_gpu
    with pytest.raises(ValueError):
        port.construct()
    assert not port.owners
    assert not port.events


def test_profile_lease_mismatch_rejects_before_model_load(profile_port):
    port = profile_port
    assert port.settings["device"] == "cuda:0"
    assert port.profile["hardware"]["gpu_uuid"] == port.gpu
    # Static settings and the certificate agree; the factory actually requests
    # another card. Only the acquired physical device can catch this mismatch.
    port.used_device.value = "cuda:1"
    with pytest.raises(ValueError, match="admitted physical GPU"):
        port.construct()
    assert len(port.owners) == 1, "must reach the owner after static validation"
    assert port.backend_calls == {"load": 0, "execute": 0, "register": 0}
    assert len(port.events) == len(port.leases) == 1
    assert port.events[0]["phase"] == "load"
    acquired = port.leases[0]
    assert acquired.device == port.second_gpu
    assert acquired.finish_calls == 1
    assert acquired.finished_before_exception is True
    assert acquired.fenced is False


def test_profile_lease_mismatch_rejects_before_model_compute(profile_port):
    port = profile_port
    client = port.construct()
    try:
        assert port.backend_calls == {"load": 1, "execute": 0, "register": 1}
        # Keep the statically validated settings/certificate unchanged, but make
        # the subsequent compute admission return another physical GPU.
        port.admitted_device.value = port.second_gpu
        with pytest.raises(RuntimeError, match="admitted physical GPU"):
            client.request(port.payload)
        assert port.backend_calls == {"load": 1, "execute": 0, "register": 1}
        assert port.events[-1]["phase"] == "execute"
        acquired = port.leases[-1]
        assert acquired.device == port.second_gpu
        assert acquired.finish_calls == 1
        assert acquired.finished_before_exception is True
        assert acquired.fenced is False
    finally:
        client.shutdown()


@pytest.mark.parametrize("outer,inner", [(None, "config"), ("config", "factory_kwargs")])
def test_profile_rejects_explicit_device_conflicts_across_scopes(
        profile_port, outer, inner):
    port = profile_port
    port.settings.clear()
    port.settings["checkpoint"] = str(port.weights)
    if outer is None:
        port.settings["device"] = "cuda:0"
    else:
        port.settings[outer] = {"device": "cuda:0"}
    port.settings[inner] = {"gpu_device": 1}
    port.used_device.value = "cuda:1"
    port.profile["hardware"]["gpu_uuid"] = port.second_gpu
    with pytest.raises(ValueError):
        port.construct()
    assert not port.owners
    assert not port.events


def test_profile_hardware_tracks_child_cuda_visibility_override(profile_port):
    port = profile_port
    # OwnedModelDaemon._child_env overlays spec.env: child cuda:0 is GPU #2.
    with pytest.raises(ValueError):
        port.construct(env={"CUDA_VISIBLE_DEVICES": port.second_gpu})
    assert not port.owners


def test_same_path_weight_replacement_invalidates_identity_and_old_certificate(profile_port):
    from astrolabe import model_daemon

    port = profile_port
    previous = model_daemon.model_identity(port.spec)
    port.weights.write_bytes(b"replacement-weights-at-the-same-path")
    assert model_daemon.model_identity(port.spec) != previous
    client = port.construct(certificate_identity=previous)
    assert client.request(port.payload) == port.payload
    gpu_events = [event for event in port.events if event["resource"] == "gpu"]
    assert gpu_events
    assert all(event["concurrency_certified"] is False for event in gpu_events)
    client.shutdown()


@pytest.mark.parametrize("change", ["checkpoint", "cuda_mask"])
def test_automatic_profile_checks_prepared_provider_settings_and_env(profile_port, monkeypatch, change):
    from astrolabe import model_daemon

    port = profile_port
    replacement = port.weights.with_name("provider-selected.bin")
    replacement.write_bytes(b"different-provider-selected-weights")

    def prepare(spec):
        if change == "checkpoint":
            return replace(spec, settings=dict(spec.settings, checkpoint=str(replacement)))
        return replace(spec, env=dict(spec.env or {}, CUDA_VISIBLE_DEVICES=port.second_gpu))

    monkeypatch.setattr(model_daemon, "_prepare_provider_spec", prepare)
    # The manifest is indexed by the PREPARED identity, so this is not a stale
    # identity cache miss. Its assets/GPU still certify the original request.
    with pytest.raises(ValueError):
        port.construct()
    assert not port.owners


def test_automatic_profile_resolves_relative_weights_in_actual_worker_cwd(profile_port, monkeypatch):
    port = profile_port
    worker_dir = port.weights.parent / "worker"
    worker_dir.mkdir()
    (worker_dir / port.weights.name).write_bytes(b"worker-cwd-has-different-weights")
    monkeypatch.chdir(port.weights.parent)
    port.settings["checkpoint"] = port.weights.name
    with pytest.raises(ValueError):
        port.construct(cwd=str(worker_dir))
    assert not port.owners


@pytest.mark.parametrize("change", ["python", "provider", "env"])
def test_production_profile_requires_attestation_in_actual_child_runtime(profile_port, monkeypatch, change):
    from astrolabe import model_daemon

    port = profile_port
    options = {}
    if change == "python":
        options["python_bin"] = str(port.weights.parent / "unattested-python")
    elif change == "provider":
        monkeypatch.setattr(model_daemon, "_prepare_provider_spec",
                            lambda spec: replace(spec, provider={"provider": "stub"}))
    else:
        options["env"] = {"OMP_NUM_THREADS": "unattested-thread-limit"}
    with pytest.raises(ValueError):
        port.construct(**options)
    assert not port.owners


def test_automatic_qualification_flag_does_not_bypass_production_checks(profile_port):
    port = profile_port
    port.profile["qualification"] = True
    port.profile["asset_sha256"] = {}
    with pytest.raises(ValueError):
        port.construct()
    assert not port.owners


def test_incomplete_automatic_profile_is_rejected_before_owner_constructor(tmp_path, monkeypatch):
    from astrolabe import model_daemon

    owners = []
    def forbidden_owner(*args, **kwargs):
        owners.append(True)
        pytest.fail("Invalid certificate reached the model owner constructor")
    monkeypatch.setattr(model_daemon, "OwnedModelDaemon", forbidden_owner)
    checkpoint = tmp_path / "weights.bin"
    checkpoint.write_bytes(b"qualification-weights")
    settings = dict(checkpoint=str(checkpoint))
    spec = model_daemon.DaemonSpec(backend="fixture", factory="astrolabe.model_execution:kernel_handler",
                                 settings=settings, python_bin=sys.executable,
                                 max_frame_bytes=execution.DEFAULT_MODEL_FRAME_BYTES)
    identity = model_daemon.model_identity(spec)
    profile = dict(identity=identity, verified=True, resident_bytes=10,
                   scratch_bytes=10, concurrency_certified=True,
                   shape="length=8", hardware={"uuid": "GPU-other-hardware"})
    profiles = tmp_path / "profiles.json"
    profiles.write_text(json.dumps(dict(profiles={identity: profile})))
    monkeypatch.setenv("ASTROLABE_RESOURCE_PROFILES", str(profiles))
    with pytest.raises(ValueError):
        execution.NativeModelClient("fixture", spec.factory, settings)
    assert not owners


@pytest.mark.parametrize("profiled", [False, True])
def test_two_resident_cpu_models_do_not_exceed_ram_budget(broker_factory, monkeypatch, profiled):
    _, client, _ = broker_factory(gpu=False, ram_bytes=100)
    retained = []
    original_lease = client.lease

    def bounded_lease(*args, **kwargs):
        # Bound only the queue wait; retain the genuine broker capacity checks.
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = 0.1
        return original_lease(*args, **kwargs)

    monkeypatch.setattr(client, "lease", bounded_lease)
    monkeypatch.setattr(execution, "_resource_client", lambda: client)

    def loader(settings):
        weights = bytearray(80)
        retained.append(weights)
        return lambda payload: len(weights), {"device": "cpu"}

    settings = {"_runtime_profile": {"identity": "ram-review", "ram_bytes": 80}} if profiled else {}
    with execution.model_kernel_context("fixture", "ram-review"):
        first = execution.kernel_handler(settings, loader, torch_device="cpu",
                                         lifecycle=dict(ram_bytes=80))
    model_id = first({"operation": "describe"})["metadata"]["runtime_lifecycle"]["model_id"]
    state = client.snapshot()
    assert state["resident_ram_bytes"] == state["ram_bytes"] == 80
    assert state["active"] == 0  # No CPU slot remains held by residency.
    with pytest.raises(admission.AdmissionError):
        with execution.model_kernel_context("fixture", "ram-review"):
            execution.kernel_handler(settings, loader, torch_device="cpu",
                                     lifecycle=dict(ram_bytes=80))
    assert len(retained) == 1, "Second loader ran before RAM admission"
    assert client.snapshot()["ram_bytes"] == 80
    retained.clear()
    del first
    client.unregister_model(model_id)  # The fixture explicitly freed its weights.
    assert client.snapshot()["ram_bytes"] == 0


def test_real_broker_cpu_stages_release_gpu_and_preserve_request_rng(broker_factory):
    """Positive control: genuine RPC broker, CPU stages and RNG continuation."""
    import random

    _, client, _ = broker_factory(gpu=False)
    random.seed(71)
    initial = execution.capture_random_state()
    expected = [random.random() for _ in range(3)]

    def stage(value):
        assert client.snapshot()["active"] == 1
        assert client.snapshot()["reserved_vram_bytes"] == 0
        return value + [random.random()]

    handler = execution.kernel_handler(
        {"_initial_random_state": initial},
        lambda settings: (execution.KernelStages(stage, stage, stage, resource="cpu"),
                          {"device": "cpu"}), torch_device="cpu")
    response = handler(dict(operation="infer", payload=[], random_state=initial,
                            runtime_request_id="cpu-review"))
    assert response["ok"] and response["result"] == expected
    assert response["runtime"]["request_id"] == "cpu-review"
    assert client.snapshot()["active"] == 0
