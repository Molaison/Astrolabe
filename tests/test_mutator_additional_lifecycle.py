from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

from astrolabe.mutators.wrappers.dplm_runner import DPLMRunner
from astrolabe.mutators.wrappers.dynamicmpnn_run import DynamicMPNNRunner
from astrolabe.runtime.client import OutcomeUnknown


def _dplm(tmp_path, **options):
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    return DPLMRunner({"python_bin": sys.executable, "repo_path": str(repo),
                      "work_dir": str(tmp_path / "work"),
                      "response_cache_dir": str(tmp_path / "cache"), **options})


def test_dplm_unknown_outcome_is_not_replayed_or_reseeded(tmp_path, monkeypatch):
    runner = _dplm(tmp_path, response_cache=False)
    calls = []

    def submitted(tasks):
        calls.append(tasks)
        raise TimeoutError("response timed out after execution")

    monkeypatch.setattr(runner, "_daemon_infer", submitted)
    with pytest.raises(OutcomeUnknown, match="not replayed"):
        runner.design_sequences([{"name": "A", "masked_sequence": "AC<mask>"}])
    assert len(calls) == 1 and runner._closed
    with pytest.raises(RuntimeError, match="closed"):
        runner.design_sequences([{"name": "A", "masked_sequence": "AC<mask>"}])
    assert len(calls) == 1


def test_dplm_pool_cache_invalidates_changed_local_weights(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    weights = snapshot / "weights.bin"
    weights.write_bytes(b"weights A")
    task = {"name": "A", "masked_sequence": "A<mask>"}
    first = _dplm(tmp_path, model_name=str(snapshot))
    before = first._cache_key(task)
    first.close()
    weights.write_bytes(b"weights B")
    second = _dplm(tmp_path, model_name=str(snapshot))
    try:
        assert second._cache_key(task) != before
    finally:
        second.close()


def test_dynamic_factory_reuses_owned_backend_and_closes_it(tmp_path):
    events = []

    class Backend:
        def __init__(self):
            self.model_load_count = 1
            events.append("load")

        def design_sequences(self, *, manifest, features):
            assert len(manifest["pdb_paths"]) == 2
            events.append("execute")
            return np.full((1, 2, 20), 0.05)

        def close(self):
            events.append("close")

    paths = tmp_path / "inputs.json"
    paths.write_text(json.dumps(["open.pdb", "closed.pdb"]))
    runner = DynamicMPNNRunner({"backend_factory": lambda *, config: Backend(),
                                "output_dir": str(tmp_path / "out")})
    first = runner.design_sequences(pdb_path_multi=str(paths))
    second = runner.design_sequences(pdb_path_multi=str(paths))
    np.testing.assert_array_equal(first["sampling_probs"], second["sampling_probs"])
    assert first["dynamicmpnn_backend_mode"] == "resident_factory"
    runner.close()
    runner.close()
    assert events == ["load", "execute", "execute", "close"]


def test_dynamic_cli_cannot_reuse_a_stale_probability_file(tmp_path):
    stale = tmp_path / "run_0000" / "probabilities.npy"
    stale.parent.mkdir()
    np.save(stale, np.full((1, 2, 20), 0.05))
    paths = tmp_path / "inputs.json"
    paths.write_text(json.dumps(["open.pdb", "closed.pdb"]))
    runner = DynamicMPNNRunner({"output_dir": str(tmp_path), "command_template": "true"})
    with pytest.raises(FileNotFoundError, match="raw probability file"):
        runner.design_sequences(pdb_path_multi=str(paths))


def test_dynamic_cannot_override_job_device_allocation():
    with pytest.raises(ValueError, match="resource ownership"):
        DynamicMPNNRunner({"env": {"CUDA_VISIBLE_DEVICES": "unallocated-device"}})


def test_resident_release_unknown_survives_repeated_close_and_collection(monkeypatch):
    """A wrapper cannot convert failed release into success or delete live input."""
    import gc
    import shutil
    from astrolabe.evaluators.worker_session import WorkerArtifacts
    from astrolabe.mutators.resident import ResidentRunner

    for name in ("ASTROLABE_EXECUTION_ENDPOINT", "ASTROLABE_EXECUTION_TOKEN",
                 "ASTROLABE_EXECUTION_ARTIFACT_ROOT"):
        monkeypatch.delenv(name, raising=False)
    artifacts = WorkerArtifacts()
    root = artifacts.root
    request = root / "unknown-request-input.json"
    request.write_text('{"request":"in flight"}')

    class UncertainClient:
        def close(self, *, release):
            assert release is True
            raise OutcomeUnknown("worker release could not be confirmed")

    runner = ResidentRunner.__new__(ResidentRunner)
    runner._closed = False
    runner._client = UncertainClient()
    runner._artifacts = artifacts
    try:
        with pytest.raises(OutcomeUnknown, match="could not be confirmed") as first:
            runner.close()
        with pytest.raises(OutcomeUnknown, match="could not be confirmed") as second:
            runner.close()
        assert second.value is first.value
        assert second.value.retained_artifact_root == str(root)
        assert runner._client is not None
        assert request.is_file()
        del first, second, runner, artifacts
        gc.collect()
        assert request.read_text() == '{"request":"in flight"}'
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.mark.parametrize("failure", ["startup", "release"])
def test_resident_startup_unknown_keeps_bootstrap(monkeypatch, failure):
    """A startup timeout may leave a child using initial/ready artifacts."""
    import gc
    import shutil
    from types import SimpleNamespace
    from astrolabe.mutators import resident
    from astrolabe.runtime import client

    for name in ("ASTROLABE_EXECUTION_ENDPOINT", "ASTROLABE_EXECUTION_TOKEN",
                 "ASTROLABE_EXECUTION_ARTIFACT_ROOT"):
        monkeypatch.delenv(name, raising=False)

    def dump(path, value):
        path.mkdir()
        (path / "state.json").write_text(json.dumps(value))

    def read_ready(_):
        assert failure == "release"
        raise ValueError("invalid bootstrap response")

    monkeypatch.setitem(sys.modules, "astrolabe.evaluators.tree_codec",
                        SimpleNamespace(dump_tree=dump, load_tree=read_ready))
    monkeypatch.setitem(sys.modules, "astrolabe.evaluators.torch_rng",
                        SimpleNamespace(capture=lambda _: {"initial": True},
                                        restore=lambda *_: None, capture_settings=lambda: {}))
    monkeypatch.setitem(sys.modules, "astrolabe.evaluators.worker_identity",
                        SimpleNamespace(identity=lambda *_, **__: {},
                                        probe_device=lambda *_: pytest.fail("CPU only")))
    monkeypatch.setattr(resident, "_pin_inputs", lambda _, config: (config, [], []))
    bootstrap = []

    class UncertainClient:
        def close(self, *, release):
            assert release is True
            raise OutcomeUnknown("startup worker termination was not confirmed")

    def start(command, *_args, **_kwargs):
        spec = json.loads(Path(command[-1]).read_text())
        boot = Path(spec["bootstrap"])
        assert (boot / "initial" / "state.json").is_file()
        bootstrap.append(boot)
        if failure == "startup":
            raise OutcomeUnknown("startup worker termination was not confirmed")
        return UncertainClient()

    monkeypatch.setattr(client, "DaemonClient", start)
    try:
        with pytest.raises(OutcomeUnknown, match="termination was not confirmed") as error:
            resident.ResidentRunner("dynamicmpnn", {"device": "cpu"})
        boot, = bootstrap
        assert error.value.retained_request_dir == str(boot)
        del error
        gc.collect()
        assert (boot / "initial" / "state.json").read_text() == '{"initial": true}'
    finally:
        for boot in bootstrap:
            shutil.rmtree(boot.parent, ignore_errors=True)
