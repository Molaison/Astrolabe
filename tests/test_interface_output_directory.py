"""Output reuse must accommodate shell logs while protecting scientific state."""
import fcntl
import hashlib
import json
from pathlib import Path

import pytest

from astrolabe import interface


@pytest.fixture
def prepare(monkeypatch):
    artifacts = {'prepared/example/input.cif': 'data_example\n',
                 'prepared/example/config.yaml': 'task_name: example\n'}
    plan = {'schema': 1, 'cases': [{'name': 'example'}],
            'artifacts': {k: hashlib.sha256(v.encode()).hexdigest() for k, v in artifacts.items()}}
    monkeypatch.setattr(interface, '_prepare', lambda *args: (plan, artifacts))
    import astrolabe.protein_beam as beam
    monkeypatch.setattr(beam, 'run_beam', lambda *a, **k: pytest.fail('No predictions during preparation'))
    def run(out, **kwargs):
        return interface.run_interface(input_path='input.pdb', target_homology='target.json',
                                       output_root=out, prepare_only=True, **kwargs)
    return run


@pytest.mark.parametrize('name', ['nohup.out', 'run.log', 'stdout', 'run with spaces.log'])
def test_precreated_redirected_log_is_preserved(tmp_path, prepare, name):
    output = tmp_path / 'out'
    output.mkdir()
    log = output / name
    log.write_bytes(b'nohup: ignoring input\nprevious log bytes\n')
    before, inode = log.read_bytes(), log.stat().st_ino
    with log.open('a') as open_log:
        assert prepare(output) == 0
        open_log.write('still open after initialization\n')
    assert log.stat().st_ino == inode
    assert log.read_bytes() == before + b'still open after initialization\n'
    assert (output / 'interface_plan.json').is_file()


def test_unrelated_files_and_subdirectories_are_not_overwritten(tmp_path, prepare):
    (tmp_path / 'logs').mkdir()
    files = {tmp_path / 'README.md': b'notes', tmp_path / 'logs/old.log': b'old log',
             tmp_path / 'another_model.cif': b'existing unrelated structure'}
    for file, value in files.items():
        file.write_bytes(value)
    assert prepare(tmp_path) == 0
    assert all(file.read_bytes() == value for file, value in files.items())


@pytest.mark.parametrize('name', ['prepared', 'cases', 'services', 'interface_results.json',
                                 'beam_state.json', 'af3_calls.json', 'candidates', 'scores.csv',
                                 'interface_plan.json.tmp', 'interface_results.json.tmp',
                                 'owner.lock', 'root_calibration.json', 'root_parity.json'])
def test_existing_run_paths_are_rejected_without_altering_them(tmp_path, prepare, name):
    existing = tmp_path / name
    existing.write_bytes(b'preserve this existing result\n')
    with pytest.raises(FileExistsError, match='existing Astrolabe run paths'):
        prepare(tmp_path)
    assert existing.read_bytes() == b'preserve this existing result\n'
    assert not (tmp_path / 'interface_plan.json').exists()


@pytest.mark.parametrize('name', ['interface_plan.json.tmp', 'interface_results.json.tmp'])
def test_atomic_write_temporary_symlink_cannot_overwrite_an_existing_file(tmp_path, prepare, name):
    output = tmp_path / 'out'
    output.mkdir()
    original = tmp_path / 'valuable.json'
    original.write_bytes(b'keep this data\n')
    temporary = output / name
    temporary.symlink_to(original)
    with pytest.raises(FileExistsError, match='existing Astrolabe run paths'):
        prepare(output)
    assert temporary.is_symlink()
    assert original.read_bytes() == b'keep this data\n'
    assert not (output / 'interface_plan.json').exists()


@pytest.mark.parametrize('kind', ['empty_directory', 'dangling_symlink'])
def test_reserved_names_still_conflict_when_empty_or_symlinked(tmp_path, prepare, kind):
    path = tmp_path / 'cases'
    if kind == 'empty_directory':
        path.mkdir()
    else:
        path.symlink_to(tmp_path / 'missing')
    with pytest.raises(FileExistsError, match='cases'):
        prepare(tmp_path)
    assert path.is_symlink() if kind == 'dangling_symlink' else path.is_dir()
    assert not (tmp_path / 'interface_plan.json').exists()


def test_existing_interface_plan_requires_explicit_resume(tmp_path, prepare):
    (tmp_path / 'run.log').write_text('log\n')
    assert prepare(tmp_path) == 0
    before = (tmp_path / 'interface_plan.json').read_bytes()
    with pytest.raises(FileExistsError, match='--resume'):
        prepare(tmp_path)
    assert prepare(tmp_path, resume=True) == 0
    assert (tmp_path / 'interface_plan.json').read_bytes() == before
    (tmp_path / 'prepared/example/input.cif').write_text('modified\n')
    with pytest.raises(ValueError, match='changed or missing'):
        prepare(tmp_path, resume=True)


def test_output_lock_still_prevents_concurrent_initialization(tmp_path, prepare):
    (tmp_path / 'nohup.out').write_text('log\n')
    with (tmp_path / 'interface.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            prepare(tmp_path)
    assert not (tmp_path / 'interface_plan.json').exists()
