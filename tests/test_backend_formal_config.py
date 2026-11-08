"""Standalone backend YAML corpus: one deliberate defect per invalid file."""
import json
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf
import pytest

from astrolabe.startup_validation import StartupValidationError, validate_startup_config
from astrolabe.backend_startup import validate_backend_settings


SOURCE = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / 'fixtures' / 'formal_config' / 'backend'
CASES = json.loads((FIXTURES / 'manifest.json').read_text())


@pytest.mark.parametrize('case', CASES, ids=lambda case: case['id'])
def test_backend_standalone_fixture(case, monkeypatch):
    # Fixture assets are startup-only file-stat placeholders, never model inputs.
    monkeypatch.chdir(SOURCE)
    monkeypatch.delenv('ASTROLABE_MODEL_EXECUTION', raising=False)
    before = set(sys.modules)

    def forbidden_spawn(*args, **kwargs):
        pytest.fail('startup validation attempted to spawn a child')

    monkeypatch.setattr(subprocess, 'Popen', forbidden_spawn)
    cfg = OmegaConf.load(FIXTURES / case['file'])
    if case['expected_error'] is None:
        assert validate_startup_config(cfg, check_dependencies=False) == {}
    else:
        with pytest.raises(StartupValidationError) as caught:
            validate_startup_config(cfg, check_dependencies=False)
        assert case['expected_error'] in str(caught.value)
        # The corresponding standalone control must succeed without a model.
        control = OmegaConf.load(FIXTURES / case['valid_control'])
        assert validate_startup_config(control, check_dependencies=False) == {}
    assert not {'torch', 'jax', 'pyrosetta', 'run'} & (set(sys.modules) - before)


def test_corpus_in_clean_process():
    # An existing test session might already contain torch; this verifies the
    # stronger cold-process guarantee without substituting scientific modules.
    program = '''
import json, subprocess, sys
from pathlib import Path
from omegaconf import OmegaConf
from astrolabe.startup_validation import StartupValidationError, validate_startup_config
def no_spawn(*args, **kwargs):
    raise AssertionError('validation spawned a child')
subprocess.Popen = no_spawn
root = Path(sys.argv[1])
for case in json.loads((root / 'manifest.json').read_text()):
    try:
        validate_startup_config(OmegaConf.load(root / case['file']), check_dependencies=False)
    except StartupValidationError as error:
        assert case['expected_error'] and case['expected_error'] in str(error), (case['id'], str(error))
    else:
        assert case['expected_error'] is None, case['id']
assert not {'torch', 'jax', 'pyrosetta', 'run'} & sys.modules.keys()
'''
    result = subprocess.run([sys.executable, '-c', program, str(FIXTURES.resolve())],
                            cwd=SOURCE, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


def test_manifest_has_controls_consumers_and_single_defects():
    assert len({case['id'] for case in CASES}) == len(CASES)
    assert {case['file'] for case in CASES} == {path.name for path in FIXTURES.glob('*.yaml')}
    for case in CASES:
        assert case['parameters'] and case['source']
        assert all(isinstance(item, str) and item for item in case['parameters'] + case['source'])
        assert not Path(case['file']).is_absolute()
        assert (FIXTURES / case['valid_control']).is_file()
        if case['expected_error'] is None:
            continue
        broken = OmegaConf.to_container(OmegaConf.load(FIXTURES / case['file']))
        control = OmegaConf.to_container(OmegaConf.load(FIXTURES / case['valid_control']))
        # Undo only the declared defect; all other fields must be identical.
        parts = case['parameters'][0].split('.')
        dst, src = broken, control
        for part in parts[:-1]:
            dst, src = dst[part], src[part]
        if parts[-1] in src:
            dst[parts[-1]] = src[parts[-1]]
        else:
            dst.pop(parts[-1])
        assert broken == control, case['id']


def test_unresolved_permission_is_separate_from_path_checks():
    options = {'model_type': 'protein_mpnn', 'checkpoint_protein_mpnn': 'weights.pt',
               'seed': '${seed}', 'bias_mode': '${oc.select:aa_bias.mode,additive}'}
    kwargs = {'path': 'mutator.m', 'check_paths': False}
    errors = validate_backend_settings('mutator', 'ligandmpnn', options, **kwargs)
    assert any('mutator.m.seed:' in error for error in errors)
    assert any('mutator.m.bias_mode:' in error for error in errors)
    assert validate_backend_settings('mutator', 'ligandmpnn', options,
                                     allow_unresolved=True, **kwargs) == []


@pytest.mark.parametrize('mode', ['daemon', 'in_process'])
def test_initial_guess_and_cache_use_effective_child_cwd(tmp_path, monkeypatch, mode):
    parent = tmp_path / 'parent'
    child = tmp_path / 'child'
    parent.mkdir()
    child.mkdir()
    monkeypatch.chdir(parent)
    reference = child / 'reference.pdb'
    reference.write_text('Not parsed by startup validation.\n')
    (parent / 'cache').write_text('file in parent; not a child directory')
    options = {'execution_mode': mode, 'execution_cwd': str(child),
               'initial_guess': {'enabled': True, 'path': 'reference.pdb',
                                 'chain_map': {'A': None}, 'allow_missing_chains': True}}
    original_open = Path.open

    def no_structure_read(path, *args, **kwargs):
        if path.name == 'reference.pdb':
            pytest.fail('startup validation read structure contents')
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', no_structure_read)
    errors = validate_backend_settings('evaluator', 'alphafold2', options,
                                       path='evaluator.af2', check_paths=True)
    if mode == 'daemon':
        assert errors == []
    else:
        assert any('evaluator.af2.initial_guess.path:' in error for error in errors)
    cache_options = {'execution_mode': mode, 'execution_cwd': str(child),
                     'jax_compilation_cache_dir': 'cache'}
    errors = validate_backend_settings('evaluator', 'alphafold3', cache_options,
                                       path='evaluator.af3', check_paths=True)
    if mode == 'daemon':
        assert errors == []
        assert not (child / 'cache').exists()  # No mkdir for intended output dirs.
    else:
        assert any('evaluator.af3.jax_compilation_cache_dir:' in error for error in errors)


def test_disabled_initial_guess_never_stats_irrelevant_path(monkeypatch):
    def forbidden_stat(*args, **kwargs):
        pytest.fail('disabled initial guess accessed its irrelevant structure path')

    monkeypatch.setattr(Path, 'is_file', forbidden_stat)
    assert validate_backend_settings('evaluator', 'alphafold2', {
        'initial_guess': {'enabled': False, 'path': ['malformed', 'unused'],
                          'chain_map': {'A': ['also', 'unused']}},
    }, path='evaluator.af2', check_paths=True) == []


def test_unresolved_initial_guess_and_cache_are_only_deferred_explicitly():
    options = {'initial_guess': {'enabled': True, 'path': '${reference}',
                                 'chain_map': '${chains}'}}
    kwargs = {'path': 'evaluator.af2', 'check_paths': False}
    assert validate_backend_settings('evaluator', 'alphafold2', options,
                                     allow_unresolved=True, **kwargs) == []
    assert any('.chain_map:' in error for error in
               validate_backend_settings('evaluator', 'alphafold2', options, **kwargs))


def test_boltz_cache_output_directory_is_not_created(tmp_path, monkeypatch):
    cfg = OmegaConf.load(FIXTURES / 'valid_boltz.yaml')
    existing_file = tmp_path / 'cache-file'
    existing_file.write_text('not a directory')
    intended_cache = tmp_path / 'new-parent' / 'new-cache'

    def forbidden_mkdir(*args, **kwargs):
        pytest.fail('startup validation created a cache directory')

    monkeypatch.setattr(Path, 'mkdir', forbidden_mkdir)
    cfg.evaluator.evaluator_terms.backend.cache = str(intended_cache)
    assert validate_startup_config(cfg, check_dependencies=False) == {}
    assert not intended_cache.exists()
    cfg.evaluator.evaluator_terms.backend.cache = str(tmp_path)
    assert validate_startup_config(cfg, check_dependencies=False) == {}
    cfg.evaluator.evaluator_terms.backend.cache = str(existing_file)
    with pytest.raises(StartupValidationError, match=r'backend\.cache: existing path is not a directory'):
        validate_startup_config(cfg, check_dependencies=False)
