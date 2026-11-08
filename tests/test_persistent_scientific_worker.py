"""CPU-only protocol/evidence tests; no real-model or GPU acceptance claims."""
from __future__ import annotations

import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import textwrap
import threading
from types import SimpleNamespace

import pytest


SPEC = importlib.util.spec_from_file_location(
    "_persistent_scientific_fixtures", Path(__file__).with_name("test_scientific_workflow_acceptance.py"))
F = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(F)
H = F.H


def acquisition(*, reused=False, attached=True):
    state = F.state()
    internal = {"python": state["python"], "numpy": state["numpy"],
                "torch": state["torch_cpu"].tolist(), "torch_seed": 17, "cuda_seed": 17}
    return {"backend": "esmfold", "ok": True, "pid": os.getpid(), "owner_pid": os.getpid(),
            "model_id": "model:epoch1", "identity": "model", "pool_attached": attached, "pool_reused": reused,
            "rng_before": copy.deepcopy(state), "rng_after": copy.deepcopy(state),
            "start": 1., "end": 2., "seconds": 1.,
            "actual_runtime_lifecycle": {"pid": os.getpid(), "model_id": "model:epoch1",
                                         "identity": "model", "loader_count": 1},
            "initialization_transition": {key: copy.deepcopy(internal)
                for key in ("before", "cuda_before_place", "after")}}


def evidence(root, *, reused=False, binding=None, source="/synthetic/source"):
    ev, report = F.fixture(root)
    ev.update(model_loads=[], model_acquisitions=[acquisition(reused=reused)],
              native_requests=F.native_requests(), native_clients=[{"profile": {"qualification": True}}],
              torch_loads=[] if reused else [{"resolved": "/models/synthetic.pt", "ok": True, "pid": os.getpid()}])
    report.update(lane="resident_serial", persistent=True, pid=os.getpid(), source=str(source),
                  source_identity={"sha256": "frozen-source"})
    if binding is not None:
        ev["model_acquisitions"][0]["initialization_binding"] = copy.deepcopy(binding)
    F.persist(root, ev, report)
    return ev, report


def cold_binding(root):
    evidence(root)
    return {"worker_path": str(root / "worker.json"), "sha256": H.sha256(root / "worker.json"),
            "model_id": "model:epoch1", "owner_pid": os.getpid(), "identity": "model"}


@pytest.fixture
def service(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "run.py").write_text("# synthetic source\n")
    (source / "astrolabe").mkdir()
    (source / "astrolabe/__init__.py").write_text("# synthetic package\n")
    config, envelope = tmp_path / "input.yaml", tmp_path / "envelope.json"
    config.write_text("# frozen scientific fixture\n")
    envelope.write_text(json.dumps({"profiles_by_backend": {"esmfold": {
        "qualification": True, "concurrency_certified": True, "shapes": ["fixed-shape"],
        "upper_bound": True}}}))
    root = tmp_path / "service"
    events, calls = [], []

    class Pool:
        metadata = {"closed": False}

        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, *exc):
            self.metadata = {"closed": True}
            events.append("close")

    def pool_factory(**kwargs):
        assert kwargs == {"max_models": 1, "backends": {"esmfold"}}
        return Pool()

    monkeypatch.setattr(H, "select_source", lambda path: events.append(("source", path)))
    monkeypatch.setitem(sys.modules, "astrolabe.runtime.model_pool", SimpleNamespace(local_model_pool=pool_factory))
    # Admission uses real file hashing but no Astrolabe import or execution.
    def fake_worker(args, **kwargs):
        accepted = list((root / "accepted").glob("*.json"))
        assert any(json.loads(p.read_text())["output"] == str(args.output) for p in accepted)
        assert threading.current_thread() is threading.main_thread()
        assert kwargs["persistent"] is True and kwargs["allow_existing_source"] is True
        assert os.environ["ASTROLABE_MODEL_EXECUTION"] == "daemon"
        assert os.environ["ASTROLABE_MODEL_OWNER"] == "trajectory"
        assert os.environ["ASTROLABE_REQUEST_CONCURRENCY"] == ("0" if args.lane == "resident_serial" else "1")
        calls.append((args, kwargs))
        print("synthetic public Astrolabe application log")
        binding = kwargs["init_bindings"].get("model:epoch1")
        evidence(args.output, reused=binding is not None, binding=binding, source=source)
        return 0

    monkeypatch.setattr(H, "worker", fake_worker)
    # Restore the environment after the real service mutates its own process.
    for key in ("ASTROLABE_MODEL_EXECUTION", "ASTROLABE_MODEL_OWNER", "ASTROLABE_REQUEST_CONCURRENCY"):
        monkeypatch.setenv(key, "fixture-parent")

    def request(name, lane="resident_serial", **changes):
        return dict({"request_id": name, "lane": lane, "source": str(source), "config": str(config),
                     "envelope": str(envelope), "output": str(tmp_path / name)}, **changes)

    def run(requests):
        monkeypatch.setattr(sys, "stdin", io.StringIO("".join(json.dumps(r) + "\n" for r in requests)))
        return H.main(["--serve-worker", "--source", str(source), "--output", str(root)])

    return SimpleNamespace(root=root, source=source, request=request, run=run, calls=calls, events=events,
                           fake_worker=fake_worker)


def test_sequential_service_binds_cold_evidence_and_closes_pool(service, capsys):
    first = service.request("first")
    second = service.request("second", lane="adaptive")
    assert service.run([first, second]) == 0
    assert len(service.calls) == 2
    assert service.events[-1] == "close"
    assert stat.S_IMODE(service.root.stat().st_mode) == 0o700
    ready = json.loads((service.root / "ready.json").read_text())
    assert ready["status"] == "ready" and ready["pid"] == os.getpid() and "exitcode" not in ready
    cold = Path(first["output"]) / "worker.json"
    final = json.loads((service.root / "final.json").read_text())
    assert final["completed"] == 2 and final["stop_reason"] == "stdin_eof" and final["pool"]["closed"]
    assert final["init_bindings"]["model:epoch1"]["sha256"] == H.sha256(cold)
    for request in (first, second):
        result = json.loads((service.root / "results" / (request["request_id"] + ".json")).read_text())
        assert result["pid"] == os.getpid() and result["exitcode"] == 0
        assert result["workerpath"] == str(Path(request["output"]) / "worker.json")
    assert H.compare_workers(first["output"], second["output"])["equal"]
    assert capsys.readouterr().out == "synthetic public Astrolabe application log\n" * 2


def test_duplicate_ids_and_overlapping_paths_are_never_replayed(service):
    first = service.request("first")
    duplicate = service.request("first", output=str(service.root.parent / "other"))
    same_output = service.request("new-id", output=first["output"])
    nested_output = service.request("nested", output=first["output"] + "/nested")
    assert service.run([first, duplicate, same_output, nested_output, service.request("last")]) == 0
    assert len(service.calls) == 2
    assert json.loads((service.root / "results/first.json").read_text())["status"] == "completed"
    assert len(list((service.root / "rejections").glob("*.json"))) == 3


@pytest.mark.parametrize("kind", ["source", "relative", "root_overlap", "traversal_id", "direct", "unknown_field"])
def test_invalid_admission_does_not_call_worker(service, kind):
    request = service.request("invalid")
    if kind == "source":
        request["source"] = str(service.source.parent / "other-source")
    elif kind == "relative":
        request["config"] = "relative.yaml"
    elif kind == "root_overlap":
        request["output"] = str(service.root / "evidence")
    elif kind == "traversal_id":
        request["request_id"] = "../escape"
    elif kind == "direct":
        request["lane"] = "direct"
    else:
        request["new_science"] = True
    assert service.run([request, {"stop": True}]) == 0
    assert not service.calls
    assert json.loads((service.root / "final.json").read_text())["stop_reason"] == "explicit_stop"


def test_changed_source_is_rejected_before_acceptance(service, monkeypatch):
    def mutate_source(args, **kwargs):
        code = service.fake_worker(args, **kwargs)
        (service.source / "run.py").write_text("# changed source\n")
        return code
    monkeypatch.setattr(H, "worker", mutate_source)
    assert service.run([service.request("first"), service.request("second")]) == 0
    assert len(service.calls) == 1
    assert not (service.root / "accepted/second.json").exists()
    assert "Source contents changed" in (service.root / "results/second.json").read_text()


@pytest.mark.parametrize("raise_error", [False, True])
def test_scientific_failure_stops_without_accepting_next(service, monkeypatch, raise_error):
    def fail(args, **kwargs):
        if raise_error:
            raise RuntimeError("unknown scientific outcome")
        return 7
    monkeypatch.setattr(H, "worker", fail)
    assert service.run([service.request("failed"), service.request("never")]) == 1
    result = json.loads((service.root / "results/failed.json").read_text())
    assert result["exitcode"] == (1 if raise_error else 7)
    assert not (service.root / "accepted/never.json").exists()
    assert service.events[-1] == "close"
    assert json.loads((service.root / "final.json").read_text())["stop_reason"] == "scientific_failure"


def test_exit_zero_without_completed_evidence_is_a_service_failure(service, monkeypatch):
    monkeypatch.setattr(H, "worker", lambda *args, **kwargs: 0)
    assert service.run([service.request("empty"), service.request("never")]) == 1
    assert not (service.root / "accepted/never.json").exists()
    assert json.loads((service.root / "results/empty.json").read_text())["status"] == "failed"


def test_existing_service_root_is_never_overwritten(service):
    service.root.mkdir()
    sentinel = service.root / "sentinel"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError):
        service.run([])
    assert sentinel.read_text() == "keep" and not service.calls


def test_source_reuse_is_explicit_and_origin_checked(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setitem(sys.modules, "astrolabe", SimpleNamespace(__file__=str(source / "astrolabe/__init__.py")))
    with pytest.raises(RuntimeError, match="fresh interpreter"):
        H.select_source(source)
    # Isolate just the origin list from unrelated imports in the test runner.
    monkeypatch.setattr(H, "module_origins", lambda path: {"astrolabe": str(path / "astrolabe/__init__.py")})
    assert H.select_source(source, allow_existing_source=True) is sys.modules["astrolabe"]
    with pytest.raises(RuntimeError, match="cannot change"):
        H.select_source(tmp_path / "other", allow_existing_source=True)
    with pytest.raises(ValueError, match="only in persistent"):
        H.worker(SimpleNamespace(), allow_existing_source=True)


def test_persistent_source_rejects_mixed_imports(tmp_path, monkeypatch):
    package = SimpleNamespace(__file__=str(tmp_path / "astrolabe/__init__.py"))
    monkeypatch.setattr(H, "sys", SimpleNamespace(modules={"astrolabe": package,
        "astrolabe.foreign": SimpleNamespace(__file__="/foreign/astrolabe/foreign.py")}))
    with pytest.raises(RuntimeError, match="Mixed source import"):
        H.select_source(tmp_path, allow_existing_source=True)


def test_acquisition_boundary_overrides_loads_without_weakening_rng(tmp_path):
    root = tmp_path / "logical"
    ev, report = evidence(root)
    assert all(H.validate_evidence(ev, report).values())
    assert H.compare_workers(root, root)["equal"]
    ev["model_loads"] = [{"rng_before": {"bogus": True}}]
    F.persist(root, ev, report)
    assert H.compare_workers(root, root)["equal"]
    for key in ("rng_before", "rng_after"):
        changed = copy.deepcopy(ev)
        changed["model_acquisitions"][0][key]["torch_cpu"][0] = 99
        other = tmp_path / key
        other_ev, other_report = evidence(other)
        changed["artifacts"] = other_ev["artifacts"]
        F.persist(other, changed, other_report)
        result = H.compare_workers(root, other)
        assert not result["equal"]
        assert not result["checks"]["loader_before_rng_equal" if key == "rng_before" else "loader_after_full_rng_equal"]


@pytest.mark.parametrize("change", ["empty", "two", "identity", "owner", "count", "transition", "rng"])
def test_incomplete_acquisition_never_falls_back_to_old_loads(tmp_path, change):
    ev, report = evidence(tmp_path / "worker")
    ev["model_loads"] = [{"rng_before": F.state(), "rng_after": F.state()}]
    row = ev["model_acquisitions"][0]
    if change == "empty":
        ev["model_acquisitions"] = []
    elif change == "two":
        ev["model_acquisitions"].append(copy.deepcopy(row))
    elif change == "identity":
        row.pop("identity")
    elif change == "owner":
        row["owner_pid"] += 1
    elif change == "count":
        row["actual_runtime_lifecycle"]["loader_count"] = 2
    elif change == "transition":
        row["initialization_transition"] = True
    else:
        del row["rng_before"]["torch_cpu"]
    assert not all(H.validate_evidence(ev, report).values())


@pytest.mark.parametrize("damage", ["missing_binding", "hash", "deleted", "pid", "model_id", "not_completed",
                                    "no_reads", "loader_count", "not_cold", "evidence_digest", "artifact",
                                    "transition", "foreign_read_pid", "nonpersistent"])
def test_warm_binding_damage_fails_closed(tmp_path, damage):
    cold = tmp_path / "cold"
    binding = cold_binding(cold)
    warm = tmp_path / "warm"
    ev, report = evidence(warm, reused=True, binding=binding)
    assert all(H.validate_evidence(ev, report).values())
    if damage == "missing_binding":
        ev["model_acquisitions"][0].pop("initialization_binding")
    elif damage == "hash":
        (cold / "worker.json").write_text((cold / "worker.json").read_text() + "\n")
    elif damage == "deleted":
        (cold / "worker.json").unlink()
    elif damage in {"pid", "model_id"}:
        ev["model_acquisitions"][0]["owner_pid" if damage == "pid" else "model_id"] = "different-owner"
    elif damage == "transition":
        ev["model_acquisitions"][0]["initialization_transition"]["after"]["torch"][0] = 55
    elif damage == "nonpersistent":
        report["persistent"] = False
    elif damage == "evidence_digest":
        (cold / "evidence.pkl").write_bytes(b"corrupt")
    elif damage == "artifact":
        next((cold / "trajectory").rglob("*.pdb")).write_text("corrupt")
    else:
        first, receipt = H.load_worker(cold)
        if damage == "not_completed":
            receipt["status"] = "failed"
        elif damage == "no_reads":
            first["torch_loads"] = []
        elif damage == "loader_count":
            first["model_acquisitions"][0]["actual_runtime_lifecycle"]["loader_count"] = 2
        elif damage == "foreign_read_pid":
            first["torch_loads"][0]["pid"] += 1
        else:
            first["model_acquisitions"][0]["pool_reused"] = True
        F.persist(cold, first, receipt)
        ev["model_acquisitions"][0]["initialization_binding"]["sha256"] = H.sha256(cold / "worker.json")
    assert not H.validate_evidence(ev, report)["warm_initialization_bound"]


def test_private_native_acquisition_requires_current_reads(tmp_path):
    ev, report = evidence(tmp_path / "private")
    report["persistent"] = False
    ev["model_acquisitions"][0]["pool_attached"] = False
    assert all(H.validate_evidence(ev, report).values())
    ev["torch_loads"] = []
    assert not H.validate_evidence(ev, report)["checkpoint_loads_recorded"]


@pytest.mark.parametrize("boundary", ["first_eval", "preloader_cuda"])
def test_logical_boundary_preserves_rng_diagnostics(tmp_path, boundary):
    left, right = tmp_path / "left", tmp_path / "right"
    evidence(left)
    ev, report = evidence(right)
    if boundary == "first_eval":
        ev["evaluations"][0]["rng_before"]["torch_cpu"][0] = 99
    else:
        ev["model_acquisitions"][0]["rng_before"]["cuda_initialized"] = True
    F.persist(right, ev, report)
    result = H.compare_workers(left, right)
    assert result["equal"] and result["requires_review"]


def test_client_init_records_real_boundary_and_binding_without_inventing_loads(monkeypatch):
    binding = {"worker_path": "/trusted/worker.json", "sha256": "digest"}
    row = acquisition(reused=True)
    cap = H.Capture(None, "adaptive", {"profiles_by_backend": {"esmfold": {"qualification": True}}},
                    init_bindings={"model:epoch1": binding})
    before, after = F.state(), F.state()
    after["torch_cpu"][0] = 19
    states = iter([before, after])
    monkeypatch.setattr(H, "rng_state", lambda torch: next(states))

    def original(client, backend, factory, settings, *, resource_profile=None):
        client.pid = os.getpid()
        client.metadata = {"runtime_lifecycle": row["actual_runtime_lifecycle"],
                           "initialization_transition": row["initialization_transition"]}
        client.owner = SimpleNamespace(_entry=SimpleNamespace(hits=1))

    cap.client_init(original)(SimpleNamespace(), "esmfold", "factory", {})
    captured = cap.model_acquisitions[0]
    assert captured["rng_before"]["torch_cpu"][0] != captured["rng_after"]["torch_cpu"][0]
    assert captured["initialization_binding"] == binding
    assert captured["pool_reused"] and captured["owner_pid"] == os.getpid()
    assert H.acquisition_complete(captured)
    assert cap.model_loads == [] and cap.loads == []


def test_capture_with_actual_client_and_pool_reuses_only_one_fake_owner():
    # The offline harness also runs on Python 3.9; native runtime dataclasses
    # require >=3.10. Use an existing compatible interpreter, never install one.
    executable = sys.executable if sys.version_info >= (3, 10) else (
        shutil.which("python3.11") or shutil.which("python3.12"))
    if executable is None:
        pytest.skip("Native runtime interface requires an existing Python >=3.10")
    script = textwrap.dedent('''
        import copy, importlib.util, os, random, sys
        from pathlib import Path
        root = Path(sys.argv[1])
        sys.path.insert(0, str(root))
        spec = importlib.util.spec_from_file_location("harness", root / "scripts/validation/scientific_workflow_acceptance.py")
        H = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(H)
        from astrolabe import model_daemon, model_execution
        from astrolabe.runtime.model_pool import local_model_pool
        initial = {"python": random.Random(17).getstate(), "numpy": ["MT19937", [1, 2], 0, 0, 0.],
                   "torch": [1, 2], "torch_seed": 17, "cuda_seed": 17}
        owners = []
        class Owner:
            def __init__(self, spec, **kwargs):
                self.spec, self.pid = spec, os.getpid()
                self.identity = model_daemon.model_identity(spec)
                self.closed = False
                owners.append(self)
            def request(self, request):
                assert request == {"operation": "describe"}
                return {"ok": True, "metadata": {"runtime_lifecycle": {
                    "pid": self.pid, "identity": self.identity, "model_id": "model:epoch1", "loader_count": 1},
                    "initialization_transition": {key: copy.deepcopy(initial)
                        for key in ("before", "cuda_before_place", "after")}},
                    "random_state": copy.deepcopy(initial)}
            def shutdown(self):
                self.closed = True
        model_daemon.LocalModelOwner = Owner
        model_execution.capture_random_state = lambda: copy.deepcopy(initial)
        model_execution.restore_random_state = lambda state: None
        H.rng_state = lambda torch: copy.deepcopy(initial)
        H.freeze = copy.deepcopy  # list-only fake interface; no NumPy/Torch import
        os.environ["ASTROLABE_MODEL_OWNER"] = "trajectory"
        os.environ.pop("ASTROLABE_MODEL_PROVIDERS", None)
        envelope = {"profiles_by_backend": {"esmfold": {"qualification": True,
            "concurrency_certified": True, "shapes": ["fixed-shape"]}}}
        captures = []
        with local_model_pool(max_models=1, backends={"esmfold"}) as pool:
            for _ in range(2):
                cap = H.Capture(None, "resident_serial", envelope)
                with H.patch(model_execution.NativeModelClient, "__init__", cap.client_init):
                    client = model_execution.NativeModelClient("esmfold", "cpu_fixture:make", {})
                    client.shutdown()
                captures.append(cap)
            assert pool.metadata["real_load_count"] == 1 and len(owners) == 1
            assert not owners[0].closed
        assert owners[0].closed
        assert [cap.model_acquisitions[0]["pool_reused"] for cap in captures] == [False, True]
        assert all(H.acquisition_complete(cap.model_acquisitions[0]) for cap in captures)
        assert all(not cap.model_loads and not cap.loads for cap in captures)
        print("actual client/pool CPU interface passed")
    ''')
    result = subprocess.run([executable, "-c", script, str(F.ROOT)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "actual client/pool CPU interface passed"


def test_warm_owner_uses_its_profile_after_same_identity_private_fallback(monkeypatch):
    cap = H.Capture(None, "adaptive", {"profiles_by_backend": {"esmfold": {"shapes": ["infer-shape"]}}})
    retained_scopes, private_scopes = [], []
    cap.runtime_shapes.update({"same-identity": private_scopes, "retained-model-id": retained_scopes})
    client = SimpleNamespace(metadata={"runtime_lifecycle": {
        "identity": "same-identity", "model_id": "retained-model-id"}},
        _resource_profile={"qualification": True}, last_runtime_receipt={"profile_matched": True})
    cap.backend_by_client[id(client)] = "esmfold"
    monkeypatch.setitem(sys.modules, "astrolabe.model_execution", SimpleNamespace(resource_shape=lambda p: "infer-shape"))
    cap.client_request(lambda client, payload: None)(client, {
        "operation": "infer", "sequences": [F.SEQ], "return_residue_plddt": True})
    assert retained_scopes == ["infer-shape"] and private_scopes == []
