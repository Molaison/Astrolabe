"""Framework-free contract tests; no claim of real model/GPU certification."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from astrolabe.evaluators.base import Evaluator
from astrolabe.evaluators.lifecycle import model_lifecycle, request_runtime, shape_key
from astrolabe.evaluators.loader_observer import LoaderObserver
from astrolabe.evaluators.worker_session import request_artifacts
from astrolabe.runtime.client import OutcomeUnknown


def test_close_cleans_all_owned_aliases_even_when_one_cleanup_fails():
    events = []
    class Handle:
        def __init__(self, name, fail=False):
            self.name, self.fail = name, fail
        def close(self):
            events.append(self.name)
            if self.fail:
                raise OSError('injected cleanup error')
    first, second, model = Handle('first', True), Handle('second'), Handle('model')
    evaluator = Evaluator()
    evaluator.resident = first
    evaluator._resident = first
    evaluator.model_runners = {'a': first, 'b': second}
    evaluator._model = model
    with pytest.raises(RuntimeError, match='1 evaluator resource release'):
        evaluator.close()
    assert events.count('first') == events.count('second') == events.count('model') == 1
    assert evaluator.model_runners == {} and evaluator._model is None
    with pytest.raises(RuntimeError, match='remain unconfirmed'):
        evaluator.close()
    assert len(events) == 3
    with pytest.raises(RuntimeError, match='closed'):
        evaluator.evaluate({})


def test_request_artifacts_survive_unknown_execution_and_only_known_work_is_deleted(tmp_path):
    with request_artifacts(tmp_path) as value:
        first = Path(value)
        (first / 'input').write_text('live input')
    assert not first.exists()
    with pytest.raises(OutcomeUnknown) as error:
        with request_artifacts(tmp_path) as value:
            uncertain = Path(value)
            (uncertain / 'input').write_text('must survive until broker fence')
            raise OutcomeUnknown('transport disconnected after submit')
    assert error.value.retained_request_dir == str(uncertain)
    assert (uncertain / 'input').read_text() == 'must survive until broker fence'
    with pytest.raises(ValueError):
        with request_artifacts(tmp_path) as value:
            failed = Path(value)
            raise ValueError('validation failed before submit')
    assert not failed.exists()


def test_private_artifacts_survive_uncertain_startup_close_and_collection(monkeypatch):
    import gc
    import shutil
    from astrolabe.evaluators.worker_session import WorkerArtifacts
    from astrolabe.runtime import client as runtime_client
    for key in ('ASTROLABE_EXECUTION_ENDPOINT', 'ASTROLABE_EXECUTION_TOKEN'):
        monkeypatch.delenv(key, raising=False)
    artifacts = WorkerArtifacts()
    root = artifacts.root
    (root / 'bootstrap').write_text('in use by an unconfirmed worker')
    def failed_start(*args, **kwargs):
        raise OutcomeUnknown('startup termination not confirmed')
    monkeypatch.setattr(runtime_client, 'DaemonClient', failed_start)
    try:
        with pytest.raises(OutcomeUnknown) as error:
            artifacts.start_client(['fixture'], {})
        assert error.value.retained_artifact_root == str(root)
        # A failed constructor's destructor must not turn absence of a client
        # handle into permission to remove bootstrap files.
        artifacts.close()
        assert root.exists()
        del artifacts, error
        gc.collect()
        assert (root / 'bootstrap').is_file()
    finally:
        shutil.rmtree(root)

    class Client:
        def close(self, *, release):
            raise OutcomeUnknown('release not confirmed')
    artifacts = WorkerArtifacts()
    root = artifacts.root
    (root / 'request').write_text('still in use')
    try:
        with pytest.raises(OutcomeUnknown):
            artifacts.close_client(Client(), release=True)
        artifacts.close()
        del artifacts
        gc.collect()
        assert (root / 'request').is_file()
    finally:
        shutil.rmtree(root)


def test_retained_private_artifacts_can_be_removed_after_explicit_confirmation(monkeypatch):
    from astrolabe.evaluators.worker_session import WorkerArtifacts
    for key in ('ASTROLABE_EXECUTION_ENDPOINT', 'ASTROLABE_EXECUTION_TOKEN'):
        monkeypatch.delenv(key, raising=False)
    artifacts = WorkerArtifacts()
    root = artifacts.root
    artifacts.retain()
    artifacts.close()
    assert root.is_dir()
    artifacts.close(release_confirmed=True)
    assert not root.exists()


def test_live_process_does_not_manufacture_loaded_or_resident_weights():
    empty = model_lifecycle(load_count=0, loaded=False)
    assert empty['process_alive'] and empty['ready']
    assert empty['weights_loaded'] is False and empty['gpu_resident'] is False
    unknown = model_lifecycle(SimpleNamespace(), load_count=1, loaded=True)
    assert unknown['weights_loaded'] and unknown['gpu_resident'] is None
    assert unknown['resident_bytes'] is None


def test_successful_loader_calls_only_are_counted_and_original_is_restored():
    class Loader:
        @staticmethod
        def read(value):
            if value == 'bad':
                raise ValueError('bad checkpoint')
            return {'weight': value}
    original = Loader.read
    observed = LoaderObserver(Loader, 'read')
    assert Loader.read('first') == {'weight': 'first'}
    with pytest.raises(ValueError):
        Loader.read('bad')
    assert Loader.read('second') == {'weight': 'second'}
    assert (observed.count, observed.failures) == (2, 1)
    observed.close()
    assert Loader.read is original


def test_request_resources_include_file_input_and_do_not_guess_cold_or_output(tmp_path):
    (tmp_path / 'arrays.npz').write_bytes(b'1' * 123)
    (tmp_path / 'input.json').write_bytes(b'{}')
    runtime = request_runtime(tmp_path, 'device', {'tokens': np.zeros((3, 2)), 'sequence': 'AAA'})
    assert runtime['resources']['prepared_bytes'] == 125
    assert runtime['resources']['output_bytes'] is None
    assert 'cold' not in runtime
    assert shape_key({'sequence': 'AAA'}) == shape_key({'sequence': 'CCC'})
    assert shape_key({'tokens': np.zeros((2, 3))}) != shape_key({'tokens': np.zeros((3, 2))})
    assert request_runtime(tmp_path, 'managed', 'AAA', execution_settings={'precision': 'high'})['shape_key'] != request_runtime(
        tmp_path, 'managed', 'AAA', execution_settings={'precision': 'medium'})['shape_key']


def test_esmfold2_keeps_native_fold_and_materialization_inside_device_lease(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from astrolabe.evaluators.wrappers.esmfold2 import _worker_run
    events, active = [], [False]
    @contextmanager
    def device_context():
        assert not active[0]
        active[0] = True
        events.append('device_enter')
        try:
            yield
        finally:
            active[0] = False
            events.append('device_exit')
    def protein(**values):
        assert not active[0]
        events.append('cpu_prepare')
        return values
    class Complex:
        def to_mmcif(self):
            assert active[0]
            events.append('materialize')
            return 'data_native_result'
    class Builder:
        def fold(self, model, inputs, **kwargs):
            assert active[0]
            assert kwargs['seed'] == 19 and kwargs['num_diffusion_samples'] == 5
            events.append('fold')
            return SimpleNamespace(complex=Complex(), plddt=np.array([85.]), ptm=.8, iptm=.7)
    original_write = Path.write_text
    def write(path, *args, **kwargs):
        if path.suffix == '.cif':
            assert not active[0]
            events.append('cpu_write')
        return original_write(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'write_text', write)
    result = _worker_run({'output_dir': str(tmp_path), 'seed': 19, 'num_diffusion_samples': 5,
                         'jobs': [{'name': 'A', 'chains': [{'id': 'A', 'sequence': 'ACD'}]}]},
                        session=(object(), Builder(), 'fixed-model', 'fixture', protein,
                                 lambda **values: values), execution_context=device_context)
    assert events == ['cpu_prepare', 'device_enter', 'fold', 'materialize', 'device_exit', 'cpu_write']
    assert result['jobs'][0]['plddt'] == [85.]
    assert (tmp_path / 'A_model.cif').read_text() == 'data_native_result'


def test_codec_uses_admitted_output_allowance(monkeypatch, tmp_path):
    from astrolabe.runtime import server
    from astrolabe.evaluators.tree_codec import dump_tree
    monkeypatch.setattr(server, 'current_request', lambda: {'output_limit_bytes': 64})
    with pytest.raises(ValueError, match='budget'):
        dump_tree(tmp_path / 'output', np.ones(1024, dtype=np.float64))
    assert not (tmp_path / 'output').exists()


def test_chai_lazy_component_reuse_and_failure_offload():
    from astrolabe.evaluators.chai_components import ChaiComponents
    changes, loads = [], []
    class Module:
        def to(self, device):
            changes.append(str(device)); return self
    original = object()
    def loader(key, device):
        loads.append(key)
        return SimpleNamespace(jit_module=Module())
    chai = SimpleNamespace(_component_cache={}, _component_moved_to=original, load_exported=loader)
    manager = ChaiComponents(chai, residency='device', use_esm=False)
    assert manager.lifecycle()['load_count'] == 0
    try:
        for _ in range(2):
            with chai._component_moved_to('trunk', 'fixture-device'):
                pass
        assert loads == ['trunk'] and changes == ['fixture-device', 'fixture-device']
        with pytest.raises(ValueError):
            with chai._component_moved_to('trunk', 'fixture-device'):
                raise ValueError('inference failed')
        assert changes[-1] == 'cpu'
        assert manager.lifecycle()['load_count'] == 1
        assert manager.lifecycle()['resident_bytes'] is None
    finally:
        manager.close()
    assert chai._component_moved_to is original and chai._component_cache == {}


def test_opencl_indices_are_checked_against_allocated_physical_gpu(monkeypatch):
    from astrolabe.evaluators import openmm_binding as binding
    monkeypatch.setattr(binding, '_opencl_device_pci', lambda p, d: {'kind': 'gpu', 'pci': (0, 65, 0, 0)})
    monkeypatch.setattr(binding, '_cuda_device_pci', lambda token: {'uuid': 'GPU-fixture', 'pci': (0, 65, 0, 0)})
    properties = {'OpenCLPlatformIndex': '0', 'DeviceIndex': '1'}
    assert binding.verify_opencl_binding(properties, '3')['uuid'] == 'GPU-fixture'
    monkeypatch.setattr(binding, '_cuda_device_pci', lambda token: {'uuid': 'GPU-other', 'pci': (0, 66, 0, 0)})
    with pytest.raises(ValueError, match='outside the admitted physical GPU'):
        binding.verify_opencl_binding(properties, '3')
    with pytest.raises(ValueError, match='one explicit'):
        binding.verify_opencl_binding({'OpenCLPlatformIndex': '0', 'DeviceIndex': '0,1'}, '3')


def test_rosetta_reference_cache_tracks_files_and_computation_config(tmp_path, monkeypatch):
    import sys
    from astrolabe.evaluators.rosetta_crystal import RosettaCrystalEvaluator
    reference = tmp_path / 'reference.pdb'; reference.write_text('first')
    params = tmp_path / 'ligand.params'; params.write_text('params-v1')
    reads = []
    class Pose:
        def clone(self): return Pose()
    fake = SimpleNamespace(is_initialized=lambda: True,
                           pose_from_file=lambda path: reads.append(Path(path).read_text()) or Pose())
    monkeypatch.setitem(sys.modules, 'pyrosetta', fake)
    evaluator = RosettaCrystalEvaluator(states={'minus2': {'reference': str(reference), 'params': str(params)}}, raw_model_results_dir=tmp_path)
    monkeypatch.setattr(evaluator, '_validate_pose', lambda *a, **kw: None)
    evaluator._reference_pose('minus2', 'AB')
    evaluator._reference_pose('minus2', 'AB')
    assert reads == ['first']
    evaluator._reference_metrics[('minus2', 'AB')] = {'interaction_energy': 1}
    reference.write_text('second')
    evaluator._reference_pose('minus2', 'AB')
    assert reads == ['first', 'second'] and not evaluator._reference_metrics
    evaluator._reference_metrics[('minus2', 'AB')] = {'interaction_energy': 1}
    evaluator.packer_nloop += 1
    evaluator._reference_pose('minus2', 'AB')
    assert not evaluator._reference_metrics
    assert evaluator.lifecycle()['components']['owned_pyrosetta_initializations'] == 0
    params.write_text('params-v2')
    with pytest.raises(RuntimeError, match='requires a new process'):
        evaluator._reference_pose('minus2', 'AB')
    evaluator.close()
    assert evaluator.lifecycle()['ready'] is False
