"""Regression checks at actual entry and coordinator boundaries."""
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from astrolabe import launcher, startup_dependencies
from astrolabe.af3_featurisation_settings import featurisation_environment

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('document', [[], {'profiles': {}}, {'profiles': [None]},
    {'profiles': [{'command': 'run', 'engine': 'a', 'image': 'b'}]},
    {'profiles': [{'command': ['run'], 'engine': 'a', 'image': 'b', 'env': []}]}])
def test_malformed_runtime_config_has_actionable_cli_error(tmp_path, document):
    config = tmp_path / 'runtime.json'
    config.write_text(json.dumps(dict(version=1, **document) if isinstance(document, dict) else document))
    result = subprocess.run([sys.executable, '-m', 'astrolabe', 'run'], cwd=ROOT,
        env=dict(os.environ, ASTROLABE_RUNTIME_ACTIVE='', ASTROLABE_RUNTIME_CONFIG=str(config)),
        capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
    assert 'runtime setup:' in result.stderr
    assert 'expected' in result.stderr or 'must be a list' in result.stderr
    assert 'Traceback' not in result.stderr


def test_missing_cli_dependency_reports_interpreter():
    result = subprocess.run([sys.executable, '-S', '-m', 'astrolabe', '--version'], cwd=ROOT,
        env=dict(os.environ, ASTROLABE_RUNTIME_ACTIVE='1', PYTHONPATH=str(ROOT)),
        capture_output=True, text=True, timeout=10)
    assert result.returncode == 1
    assert 'missing CLI dependency' in result.stderr and sys.executable in result.stderr
    assert 'Traceback' not in result.stderr


def test_refine_invalid_input_never_initializes_cuda(monkeypatch, tmp_path, capsys):
    from astrolabe import cli
    monkeypatch.setenv('GH_CUDA_PREINITIALIZE', '1')
    monkeypatch.setattr(ctypes, 'CDLL', lambda *a, **kw: pytest.fail('CUDA touched before input error'))
    with pytest.raises(SystemExit) as caught:
        cli.main(['refine', 'channel', '--task-name', 'bad', '--out-dir', str(tmp_path),
            '--structure', str(tmp_path / 'missing.pdb'), '--protein-chain', 'A',
            '--ligand-residue', 'L:LIG:101', '--mutable-positions', 'all', '--spheres', 'missing.spheres'])
    assert caught.value.code == 2
    assert 'missing.pdb' in capsys.readouterr().err


def test_legacy_script_rejects_invalid_config_before_heavy_imports(tmp_path):
    (tmp_path / 'bad.yaml').write_text('steps: -1\n')
    code = '''import sys, runpy
class Block:
    def find_spec(self, fullname, *a):
        if fullname.split('.')[0] in {'torch', 'jax', 'pyrosetta'}:
            raise AssertionError('HEAVY_IMPORT_BEFORE_VALIDATION:' + fullname)
sys.meta_path.insert(0, Block())
sys.argv = ['run.py', '--config-path', sys.argv[1], '--config-name', 'bad']
runpy.run_path('run.py', run_name='__main__')
'''
    result = subprocess.run([sys.executable, '-c', code, str(tmp_path)], cwd=ROOT,
        env=dict(os.environ, GH_CUDA_PREINITIALIZE='1'), capture_output=True, text=True, timeout=15)
    assert result.returncode != 0
    assert 'steps' in result.stderr
    assert 'HEAVY_IMPORT_BEFORE_VALIDATION' not in result.stderr


@pytest.mark.parametrize('backend', ['apptainer', 'docker-apptainer'])
@pytest.mark.parametrize('override', ['', 'external binary'])
def test_generated_runner_forwards_authoritative_dalphaball(tmp_path, monkeypatch, backend, override):
    spec = importlib.util.spec_from_file_location('runner_builder', ROOT / 'scripts/containers/apptainer_build.py')
    builder = importlib.util.module_from_spec(spec); spec.loader.exec_module(builder)
    runner_path = tmp_path / 'runner.py'
    builder.write_python_runner(runner_path, apptainer='/fake/apptainer', sif_path='/fake.sif',
        runner_backend=backend, docker_bin='/fake/docker', use_nv=False)
    spec = importlib.util.spec_from_file_location('generated_runner', runner_path)
    runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    binary = tmp_path / override if override else None
    if binary is not None:
        binary.write_text('fixture')
    value = str(binary) if binary else ''
    monkeypatch.setenv('GH_DALPHABALL', value)
    monkeypatch.setattr(sys, 'argv', [str(runner_path), '-c', 'pass'])
    captured = {}
    def execute(command, **kwargs):
        captured.update(command=command, **kwargs)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(runner.subprocess, 'check_call', execute)
    runner.main()
    if backend == 'apptainer':
        assert captured['env']['APPTAINERENV_GH_DALPHABALL'] == value
    else:
        assert 'APPTAINERENV_GH_DALPHABALL' in captured['command']
        assert captured['env']['APPTAINERENV_GH_DALPHABALL'] == value
    if binary:
        assert f'{tmp_path}:{tmp_path}:ro' in captured['command']


def test_dalphaball_init_does_not_fallback_or_chmod(monkeypatch, tmp_path):
    binary = tmp_path / 'no execute'
    binary.write_text('not executable')
    monkeypatch.setenv('GH_DALPHABALL', str(binary))
    before = binary.stat().st_mode
    assert startup_dependencies.dalphaball_init_options('-mute all') == '-mute all'
    assert binary.stat().st_mode == before
    binary.chmod(0o700)
    options = startup_dependencies.dalphaball_init_options('-mute all')
    assert shlex.split(options)[-2:] == ['-holes:dalphaball', str(binary)]
    explicit = '-holes:dalphaball "/other explicit path" -mute all'
    assert startup_dependencies.dalphaball_init_options(explicit) == explicit
    monkeypatch.setenv('GH_DALPHABALL', '')
    assert startup_dependencies.dalphaball_init_options('-mute all') == '-mute all'
    with pytest.raises(startup_dependencies.StartupDependencyError):
        startup_dependencies.dalphaball_init_options('-holes:dalphaball')


def test_af3_coordinator_settings_are_scoped_and_restored_on_failure(monkeypatch):
    key = 'ALPHAFOLD3_REFSTRUCTURE_WORKERS'
    monkeypatch.setenv(key, '7')
    def feature(workers):
        with featurisation_environment(OmegaConf.create({'refstructure_workers': workers})):
            assert os.environ[key] == str(workers)
            with featurisation_environment(OmegaConf.create({'refstructure_workers': 99})):
                assert os.environ[key] == '99'
            assert os.environ[key] == str(workers)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(feature, [2, 3]))
    assert os.environ[key] == '7'
    with pytest.raises(RuntimeError), featurisation_environment(OmegaConf.create({'refstructure_workers': 4})):
        raise RuntimeError('featurisation failed')
    assert os.environ[key] == '7'


def test_lazy_concurrent_first_lookup_constructs_exactly_once(tmp_path):
    import ast
    import threading
    import time
    from collections.abc import Mapping
    tree = ast.parse((ROOT / 'run.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LazyEvaluatorMap')
    calls = []
    instance = object()
    def construct(*args, **kwargs):
        calls.append(kwargs['enabled_aliases'])
        time.sleep(0.05)
        return {'one': instance}
    namespace = {'Mapping': Mapping, 'threading': threading, 'construct_evaluators': construct}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / 'run.py'), 'exec'), namespace)
    mapping = namespace['LazyEvaluatorMap'](OmegaConf.create({'evaluator_terms': {'one': {'_type': 'sequence'}}}), tmp_path, ['A'])
    gate = threading.Barrier(8)
    def lookup(_):
        gate.wait(timeout=5)
        return mapping['one']
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert all(value is instance for value in pool.map(lookup, range(8)))
    assert calls == [['one']]
    assert list(mapping.constructed_items()) == [('one', instance)]
