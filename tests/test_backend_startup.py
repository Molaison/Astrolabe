"""Selected-backend contracts, without loading any scientific backend."""
import os
import sys
from pathlib import Path

import pytest

from astrolabe.backend_startup import validate_backend_settings


EVALUATORS = (
    'alphafold3 rf3 alphafold2 esmfold esmfold2 chai1 boltz opendde protenix '
    'rhofold dummy sequence rosetta_crystal drfold2 numonomer zfold rnabpflow'
).split()
MUTATORS = (
    'mcmc pore_blocker_mcmc ligandmpnn grouped_ligandmpnn ProteinMPNN '
    'MultistateProteinMPNN multistate_proteinmpnn FarApartTiedMultitargetLigandMPNN '
    'far_apart_tied_multitarget_ligandmpnn dynamicmpnn surfdesign surfpro '
    'carbondesign esmif esm2 esm3 motifbreaker dplm xmix nampnn'
).split()
INITIALIZERS = 'random dplm aa_set mask_mix all_a pssm x_mix structure rfd3'.split()


def check(family, kind, options, *, check_paths=False):
    return validate_backend_settings(family, kind, options, path=f'{family}.{kind}', check_paths=check_paths)


@pytest.mark.parametrize('kind', EVALUATORS)
def test_each_registered_evaluator_has_a_boundary(kind):
    options = {'repo_path': '/repo', 'weights_path': '/weights', 'checkpoint': '/ckpt',
               'python_bin': sys.executable, 'ckpt_path': 'model_identifier',
               'lm_weights_path': '/lm'}
    assert check('evaluator', kind, options) == []


@pytest.mark.parametrize('kind', MUTATORS)
def test_each_registered_mutator_has_a_boundary(kind):
    options = {'model_type': 'protein_mpnn', 'checkpoint_protein_mpnn': '/ckpt',
               'backend_factory': 'module:factory', 'python_bin': sys.executable,
               'repo_path': '/repo', 'experiment_path': '/experiment',
               'checkpoint_dir': '/checkpoints', 'msms_bin': sys.executable,
               'pdb_to_xyzrn_bin': sys.executable, 'model_ckpt': 'weights.pt',
               'model_config': 'config.json', 'model_features': 'features.json'}
    assert check('mutator', kind, options) == []


@pytest.mark.parametrize('kind', INITIALIZERS)
def test_each_registered_initializer_has_a_boundary(kind):
    options = {'repo_path': '/repo', 'python_bin': sys.executable, 'ckpt_path': 'model_name'}
    assert check('initializer', kind, options) == []


@pytest.mark.parametrize('kind,options,field', [
    ('drfold2', {'experiments': ['bad'], 'repo_path': '/r', 'weights_path': '/w'}, 'experiments'),
    ('drfold2', {'num_cycles': 9, 'repo_path': '/r', 'weights_path': '/w'}, 'num_cycles'),
    ('numonomer', {'ftype': 'xyz', 'repo_path': '/r', 'weights_path': '/w'}, 'ftype'),
    ('numonomer', {'seed': 2**32, 'repo_path': '/r', 'weights_path': '/w'}, 'seed'),
    ('zfold', {'window': 49, 'repo_path': '/r', 'weights_path': '/w'}, 'window'),
    ('zfold', {'stru_feat_type': 'none', 'repo_path': '/r', 'weights_path': '/w'}, 'stru_feat_type'),
    ('zfold', {'repo_path': '/r', 'weights_path': '/w'}, 'lm_weights_path'),
    ('rnabpflow', {'num_timesteps': 1, 'repo_path': '/r', 'weights_path': '/w'}, 'num_timesteps'),
    ('rnabpflow', {'samples_per_sequence': 0, 'repo_path': '/r', 'weights_path': '/w'}, 'samples_per_sequence'),
    ('rnabpflow', {'device': 'cuda', 'repo_path': '/r', 'weights_path': '/w'}, 'device'),
    ('drfold2', {'multichain_mode': 'experimental', 'repo_path': '/r', 'weights_path': '/w'}, 'multichain_mode'),
    ('rf3', {}, 'repo_path'),
    ('rhofold', {}, 'checkpoint'),
    ('esmfold2', {'execution_mode': 'other'}, 'execution_mode'),
    ('opendde', {'execution_mode': 'other'}, 'execution_mode'),
])
def test_evaluator_invalid_field(kind, options, field):
    assert any(f'.{field}:' in e for e in check('evaluator', kind, options))


@pytest.mark.parametrize('kind,options,field', [
    ('ligandmpnn', {'model_type': 'other'}, 'model_type'),
    ('ligandmpnn', {'model_type': 'ligand_mpnn'}, 'checkpoint_ligand_mpnn'),
    ('ProteinMPNN', {'model_type': 'ligand_mpnn'}, 'model_type'),
    ('dynamicmpnn', {'execution_mode': 'daemon', 'backend_callable': 'x:y'}, 'backend_factory'),
    ('surfdesign', {'surface_mode': 'xyz'}, 'surface_mode'),
    ('surfdesign', {'logit_temperature': 0}, 'logit_temperature'),
    ('surfpro', {'device': 'cpu'}, 'device'),
    ('surfpro', {'max_surface_vertices': 1}, 'max_surface_vertices'),
    ('nampnn', {'execution_mode': 'in_process'}, 'execution_mode'),
    ('carbondesign', {'repo_path': '/r', 'model_ckpt': 'm'}, 'model_config'),
])
def test_mutator_invalid_field(kind, options, field):
    assert any(f'.{field}:' in e for e in check('mutator', kind, options))


def test_explicit_assets_and_child_controls(tmp_path):
    file = tmp_path / 'file'; file.write_text('x')
    directory = tmp_path / 'repo'; directory.mkdir()
    assert check('evaluator', 'rf3', {'repo_path': str(directory), 'ckpt_path': str(file),
                                     'python_bin': sys.executable, 'execution_mode': 'daemon'}, check_paths=True) == []
    assert any('.ckpt_path:' in x for x in check('evaluator', 'rf3',
               {'repo_path': str(directory), 'ckpt_path': str(tmp_path / 'missing')}, check_paths=True))
    assert check('initializer', 'rfd3', {'repo_path': str(directory), 'python_bin': sys.executable,
                                        'ckpt_path': 'downloaded_model_name'}, check_paths=True) == []
    bad = check('evaluator', 'dummy', {'execution_python_bin': str(file),
               'execution_cwd': str(file), 'execution_request_timeout': 0,
               'execution_max_frame_bytes': 0, 'execution_env': {'A': 1}}, check_paths=True)
    for field in ('execution_python_bin', 'execution_cwd', 'execution_request_timeout',
                  'execution_max_frame_bytes', 'execution_env'):
        assert any(f'.{field}:' in e for e in bad)


def test_attached_af3_has_no_local_model_requirement(tmp_path):
    assert check('evaluator', 'alphafold3', {'service_socket': str(tmp_path / 'remote.sock')},
                 check_paths=True) == []


def test_ligandmpnn_checks_only_selected_checkpoint(tmp_path):
    good = tmp_path / 'good.pt'; good.write_bytes(b'placeholder')
    assert check('mutator', 'ligandmpnn', {'model_type': 'ligand_mpnn',
        'checkpoint_ligand_mpnn': str(good),
        'checkpoint_protein_mpnn': str(tmp_path / 'not_used')}, check_paths=True) == []


def test_backend_specific_default_resolution(tmp_path):
    repo = tmp_path / 'repo'; repo.mkdir()
    for name in ('config.json', 'features.json', 'weights.pt'):
        (repo / name).write_text('placeholder')
    assert check('mutator', 'carbondesign', {'repo_path': str(repo),
        'model_config': 'config.json', 'model_features': 'features.json',
        'model_ckpt': 'weights.pt'}, check_paths=True) == []
    assert any('.checkpoint:' in e for e in check('mutator', 'nampnn',
        {'repo_path': str(repo)}, check_paths=True))
    assert check('evaluator', 'rf3', {'repo_path': str(repo),
        'ckpt_path': 'autodownload_name', 'execution_mode': 'daemon'}, check_paths=True) == []
    assert check('evaluator', 'opendde', {'repo_path': str(repo),
        'use_msa': False, 'kalign_binary_path': str(repo / 'unused')}, check_paths=True) == []


def test_registered_mutator_case_and_malformed_enum():
    assert any('.model_type:' in e for e in check('mutator', 'MULTISTATEPROTEINMPNN',
        {'model_type': 'ligand_mpnn', 'checkpoint_ligand_mpnn': '/ckpt'}))
    errors = check('evaluator', 'drfold2', {'repo_path': '/r', 'weights_path': '/w',
        'experiments': [['bad']], 'multichain_mode': []})
    assert any('.experiments:' in e for e in errors)
    assert any('.multichain_mode:' in e for e in errors)


def test_import_is_cheap():
    import subprocess
    result = subprocess.run([sys.executable, "-c", "import astrolabe.backend_startup, sys; assert not {'torch', 'jax', 'pyrosetta'} & sys.modules.keys()"], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert 'pyrosetta' not in sys.modules


@pytest.mark.parametrize('options,field', [
    ({'num_samples': 0}, 'num_samples'),
    ({'num_recycles': -1}, 'num_recycles'),
    ({'refstructure_workers': 'bad'}, 'refstructure_workers'),
    ({'msa_cache': 'no'}, 'msa_cache'),
    ({'buckets': [512, 256]}, 'buckets'),
    ({'flash_attention_implementation': 'bad'}, 'flash_attention_implementation'),
])
def test_af3_coordinator_settings_fail_before_setup(options, field):
    assert any('.' + field + ':' in error for error in check('evaluator', 'alphafold3', options))


def test_af3_service_does_not_require_local_model_directory(tmp_path):
    assert check('evaluator', 'alphafold3', {'service_socket': '/service.sock',
        'model_dir': str(tmp_path / 'not-local')}, check_paths=True) == []



def test_native_child_checkpoint_uses_execution_cwd_not_parent(tmp_path, monkeypatch):
    parent = tmp_path / 'parent'; parent.mkdir()
    child = tmp_path / 'child'; child.mkdir()
    monkeypatch.chdir(parent)
    cfg = {'checkpoint': 'weights.pt', 'execution_cwd': str(child)}
    (child / 'weights.pt').touch()
    assert check('evaluator', 'rhofold', cfg, check_paths=True) == []
    (child / 'weights.pt').unlink()
    (parent / 'weights.pt').touch()
    assert any('.checkpoint:' in error for error in check('evaluator', 'rhofold', cfg, check_paths=True))
    cfg['execution_mode'] = 'in_process'
    assert check('evaluator', 'rhofold', cfg, check_paths=True) == []



def test_native_asset_checks_use_effective_mode_and_default_child_directory(tmp_path, monkeypatch):
    from astrolabe.model_execution import execution_directory
    parent = tmp_path / 'parent'; parent.mkdir()
    child = tmp_path / 'child'; child.mkdir()
    monkeypatch.chdir(parent)
    (parent / 'only_parent.pt').touch()
    cfg = {'checkpoint': 'only_parent.pt', 'execution_mode': None, 'execution_cwd': str(child)}
    monkeypatch.setenv('ASTROLABE_MODEL_EXECUTION', 'in_process')
    assert check('evaluator', 'rhofold', cfg, check_paths=True) == []
    monkeypatch.setenv('ASTROLABE_MODEL_EXECUTION', 'daemon')
    cfg.pop('execution_cwd')
    errors = check('evaluator', 'rhofold', cfg, check_paths=True)
    assert any(str(Path(execution_directory()) / 'only_parent.pt') in error for error in errors)
