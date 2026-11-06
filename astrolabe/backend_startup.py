"""Cheap, selected-backend startup checks; never imports or starts a model.

``options`` are the already merged and resolved settings for one selected backend.
Path checks reflect only assets visible to this process, not child imports or
remote service readiness.
"""
from __future__ import annotations

import os
import re
import math
import shutil
from collections.abc import Mapping
from pathlib import Path


_RNA = frozenset({'drfold2', 'numonomer', 'zfold', 'rnabpflow'})
_MPNN = frozenset({'ligandmpnn', 'grouped_ligandmpnn', 'ProteinMPNN',
                  'MultistateProteinMPNN', 'multistate_proteinmpnn',
                  'FarApartTiedMultitargetLigandMPNN', 'far_apart_tied_multitarget_ligandmpnn'})
_MPNN_CHECKPOINTS = {
    'protein_mpnn': 'checkpoint_protein_mpnn',
    'ligand_mpnn': 'checkpoint_ligand_mpnn',
    'per_residue_label_membrane_mpnn': 'checkpoint_per_residue_label_membrane_mpnn',
    'global_label_membrane_mpnn': 'checkpoint_global_label_membrane_mpnn',
    'soluble_mpnn': 'checkpoint_soluble_mpnn',
}
_PROTEIN_MPNN_ONLY = frozenset({'proteinmpnn', 'multistateproteinmpnn',
                               'faraparttiedmultitargetligandmpnn',
                               'far_apart_tied_multitarget_ligandmpnn'})
_MISSING = (None, '', 'None', 'null', '???')


def validate_backend_settings(family, kind, options, *, path, check_paths) -> list[str]:
    """Return field-specific errors for one selected backend, without side effects."""
    root = str(path).rstrip('.')
    errors: list[str] = []
    if not isinstance(options, Mapping):
        return [f'{root}: expected a mapping of resolved backend settings']
    cfg = options
    child_base = None
    if family == 'evaluator' and kind in {'alphafold2', 'alphafold3', 'esmfold', 'rhofold'}:
        from astrolabe.model_execution import execution_directory, execution_mode
        try:
            selected_mode = cfg.get('execution_mode')
            if selected_mode is None:
                selected_mode = cfg.get('execution')
            mode = execution_mode(selected_mode)
        except (TypeError, ValueError) as error:
            return [f'{root}.execution_mode: {error}']
        if mode.value == 'daemon':
            try:
                child_base = Path(execution_directory(cfg.get('execution_cwd')))
            except (TypeError, ValueError) as error:
                return [f'{root}.execution_cwd: {error}']

    def fail(field, reason):
        errors.append(f'{root}.{field}: {reason}')

    def value(field):
        return cfg.get(field)

    def asset(field, *, required=False, kind='file', explicit_only=False,
              base=None, executable=False):
        raw = value(field)
        if raw in _MISSING:
            if required:
                fail(field, 'is required')
            return None
        if not isinstance(raw, (str, os.PathLike)):
            fail(field, 'must be a path')
            return None
        target = Path(raw).expanduser()
        if explicit_only and not target.is_absolute() and target.parent == Path('.'):
            return target  # model/checkpoint name resolved by the child
        if base is not None and not target.is_absolute():
            target = Path(base).expanduser() / target
        if check_paths:
            found = shutil.which(str(target)) if executable and target.parent == Path('.') else None
            good = bool(found) or (target.is_dir() if kind == 'dir' else target.is_file() if kind == 'file' else target.exists())
            if not good:
                fail(field, f'{kind} does not exist: {target}')
            elif executable and not found and not os.access(target, os.X_OK):
                fail(field, f'is not executable: {target}')
        return target

    def choice(field, allowed, default=None):
        selected = cfg.get(field, default)
        if not isinstance(selected, str) or selected not in allowed:
            fail(field, f'must be one of {sorted(allowed)!r}')

    def number(field, default, low, *, integer=False, high=None):
        selected = cfg.get(field, default)
        try:
            parsed = int(selected) if integer else float(selected)
            if isinstance(selected, bool) or (integer and (isinstance(selected, float) and not selected.is_integer())):
                raise ValueError
            if not math.isfinite(parsed) or parsed < low or (high is not None and parsed > high):
                raise ValueError
        except (TypeError, ValueError, OverflowError):
            fail(field, f'must be {"an integer" if integer else "a number"} in [{low}, {high if high is not None else "infinity"}]')

    # These controls are consumed by NativeModelClient or explicit subprocess
    # runners. No absent/default control is inferred from another namespace.
    for field in ('execution_python_bin', 'python_bin'):
        if value(field) not in _MISSING:
            asset(field, executable=True)
    if value('execution_cwd') not in _MISSING:
        asset('execution_cwd', kind='dir')
    for field in ('execution_request_timeout', 'request_timeout', 'startup_timeout', 'shutdown_timeout'):
        if value(field) not in _MISSING:
            number(field, None, 0.000000001)
    if value('execution_max_frame_bytes') not in _MISSING:
        number('execution_max_frame_bytes', None, 1, integer=True)
    for field in ('execution_env', 'env'):
        if value(field) is not None:
            env = value(field)
            if not isinstance(env, Mapping) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in env.items()):
                fail(field, 'must map string names to string values')

    if family == 'evaluator':
        if kind == 'alphafold3':
            if not value('service_socket') and value('model_dir') not in _MISSING:
                asset('model_dir', kind='dir', base=child_base)
            if value('flash_attention_implementation') not in _MISSING:
                choice('flash_attention_implementation', {'cudnn', 'triton', 'xla'})
            for field, minimum in (('num_samples', 1), ('num_seeds', 1), ('num_recycles', 0),
                                   ('refstructure_workers', 0), ('refstructure_cache_max_entries', 0),
                                   ('msa_cache_max_entries', 0), ('bucket_pad_tolerance', 0)):
                if value(field) is not None:
                    number(field, None, minimum, integer=True)
            for field in ('refstructure_cache', 'msa_cache'):
                if value(field) is not None and value(field) not in (True, False, 0, 1):
                    fail(field, 'must be a boolean or 0/1')
            buckets = value('buckets')
            if buckets is not None and (not isinstance(buckets, (list, tuple)) or not buckets
                    or any(type(x) is not int or x <= 0 for x in buckets)
                    or list(buckets) != sorted(set(buckets))):
                fail('buckets', 'must be a strictly increasing list of positive integers')
        elif kind == 'alphafold2':
            if value('data_dir') not in _MISSING:
                asset('data_dir', kind='dir', base=child_base)
        elif kind == 'esmfold':
            if value('model_name') not in _MISSING:
                asset('model_name', explicit_only=True, base=child_base)
        elif kind in _RNA:
            choice('multichain_mode', {'reject', 'experimental'}, 'reject')
            if kind in {'drfold2', 'zfold'} and value('multichain_mode') == 'experimental':
                fail('multichain_mode', f'{kind} has no implemented multichain mode')
            for field in ('execution_mode', 'execution'):
                if value(field) not in _MISSING and value(field) != 'daemon':
                    fail(field, 'RNA backends require daemon')
            device = cfg.get('device', 'cuda:0')
            if not isinstance(device, str) or (device != 'cpu' and re.fullmatch(r'cuda:\d+', device) is None):
                fail('device', "must be 'cpu' or 'cuda:N'")
            repo = asset('repo_path', required=True, kind='dir')
            weights = asset('weights_path', required=True,
                            kind='dir' if kind == 'drfold2' else 'any')
            if kind == 'drfold2':
                experiments = cfg.get('experiments', ('cfg_95', 'cfg_96', 'cfg_97', 'cfg_99'))
                if (not isinstance(experiments, (list, tuple)) or not experiments
                        or any(not isinstance(x, str) or x not in {'cfg_95', 'cfg_96', 'cfg_97', 'cfg_99'} for x in experiments)
                        or len(set(experiments)) != len(experiments)):
                    fail('experiments', 'must be a nonempty unique subset of native experiments')
                for field, default, high in (('num_ensemble', 3, 3), ('num_cycles', 8, 8)):
                    number(field, default, 1, integer=True, high=high)
                selected = cfg.get('checkpoints', {})
                if not isinstance(selected, Mapping) or (isinstance(experiments, (list, tuple)) and all(isinstance(x, str) for x in experiments) and set(selected) - set(experiments)):
                    fail('checkpoints', 'must map selected experiments to checkpoint filenames')
                elif any(not isinstance(names, (list, tuple)) or not names or any(not isinstance(n, str) or Path(n).name != n for n in names) or len(set(names)) != len(names) for names in selected.values()):
                    fail('checkpoints', 'must contain nonempty unique native filenames')
                for field in ('rclm_path', 'selection_config', 'folding_config', 'arena_path'):
                    if value(field) not in _MISSING:
                        asset(field, executable=field == 'arena_path')
                if check_paths and repo is not None and repo.is_dir() and weights is not None and weights.is_dir():
                    for field, target in (
                        ('rclm_path', weights / 'RCLM/epoch_67000'),
                        ('selection_config', repo / 'cfg_for_selection.json'),
                        ('folding_config', repo / 'cfg_for_folding.json'),
                        ('arena_path', repo / 'Arena/Arena'),
                    ):
                        if value(field) in _MISSING and not target.is_file():
                            fail(field, f'default file does not exist: {target}')
                    for name in experiments if isinstance(experiments, (list, tuple)) else ():
                        if not isinstance(name, str) or name not in {'cfg_95', 'cfg_96', 'cfg_97', 'cfg_99'}:
                            continue
                        for filename in ('EvoMSA2XYZ.py', 'data.py', 'util.py', 'base.npy'):
                            if not (repo / name / filename).is_file():
                                fail('repo_path', f'missing {name}/{filename}')
                        weight_dir = weights / name
                        if not weight_dir.is_dir():
                            fail('weights_path', f'missing checkpoint directory {weight_dir}')
                        elif isinstance(selected, Mapping) and name in selected and isinstance(selected[name], (list, tuple)):
                            for filename in selected[name]:
                                if isinstance(filename, str) and not (weight_dir / filename).is_file():
                                    fail('checkpoints', f'missing {name}/{filename}')
                        elif not any(p.is_file() and 'model' in p.name and 'opt' not in p.name for p in weight_dir.iterdir()):
                            fail('weights_path', f'no native checkpoint in {weight_dir}')
                    for filename in ('PotentialFold/Selection.py', 'PotentialFold/Optimization.py'):
                        if not (repo / filename).is_file():
                            fail('repo_path', f'missing {filename}')
                if value('subprocess_timeout') is not None:
                    number('subprocess_timeout', None, 0.000000001)
            elif kind == 'numonomer':
                for field, default, low in (('num_iter', 8, 1), ('clamp_plddt', 512, 1), ('split_seq', 0, 0), ('split_atom', 0, 0), ('seed', 42, 0)):
                    number(field, default, low, integer=True, high=(2**32 - 1 if field == 'seed' else None))
                choice('ftype', {'pdb', 'cif'}, 'cif')
                choice('topology', {'linear', 'circular'}, 'linear')
                if not isinstance(cfg.get('save_last', False), bool):
                    fail('save_last', 'must be a boolean')
            elif kind == 'zfold':
                choice('stru_feat_type', {'LM', 'both'}, 'LM')
                for field, default, low, high in (('window', 100, 50, 256), ('nrows', 200, 1, None), ('nmodels', 5, 1, None), ('cpu', 5, 1, None)):
                    number(field, default, low, integer=True, high=high)
                for field in ('zfold_path', 'config_path', 'lm_weights_path', 'structrfm_weights_path', 'fold_script_path'):
                    if value(field) not in _MISSING:
                        asset(field, kind='dir' if field == 'zfold_path' else 'any' if field in {'lm_weights_path', 'structrfm_weights_path'} else 'file')
                if value('lm_weights_path') in _MISSING and value('structrfm_weights_path') in _MISSING:
                    fail('lm_weights_path', 'LM weights path is required (or structrfm_weights_path)')
            elif kind == 'rnabpflow':
                for field, low in (('samples_per_sequence', 1), ('num_timesteps', 2)):
                    if value(field) is not None:
                        number(field, None, low, integer=True)
                if value('config_path') not in _MISSING:
                    asset('config_path')
        elif kind == 'rf3':
            asset('repo_path', required=True, kind='dir')
            asset('ckpt_path', explicit_only=True)
            # rf3_setup fills an absent daemon interpreter with sys.executable.
        elif kind == 'rhofold':
            asset('checkpoint', required=True, base=child_base)
        elif kind == 'esmfold2':
            if value('python_bin') not in _MISSING:
                asset('python_bin', executable=True)
            choice('execution_mode', {'daemon', 'subprocess'}, 'daemon')
            for field in ('esm_repo_path', 'transformers_repo_path'):
                if value(field) not in _MISSING:
                    asset(field, kind='dir')
        elif kind == 'opendde':
            for field, asset_kind in (('repo_path', 'dir'), ('opendde_root_dir', 'dir'), ('load_checkpoint_path', 'file')):
                if value(field) not in _MISSING:
                    asset(field, kind=asset_kind)
            choice('execution_mode', {'daemon', 'subprocess'}, 'daemon')
            # Optional tools/databases are needed only for the selected feature.
            if cfg.get('use_msa'):
                for field in ('kalign_binary_path', 'hmmsearch_binary_path', 'hmmbuild_binary_path', 'seqres_database_path'):
                    if value(field) not in _MISSING:
                        asset(field, executable=field.endswith('binary_path'))
            if cfg.get('use_rna_msa'):
                for field in ('nhmmer_binary_path', 'hmmalign_binary_path', 'hmmbuild_rna_binary_path', 'ntrna_database_path', 'rfam_database_path', 'rna_central_database_path'):
                    if value(field) not in _MISSING:
                        asset(field, executable=field.endswith('binary_path'))
        # alphafold3 service_socket is attached-service identity, not local weights.
    elif family == 'mutator':
        # MutatorRegistry.get lowercases names; honor that native behavior.
        normalized_kind = str(kind).strip().lower()
        if normalized_kind in {name.lower() for name in _MPNN}:
            subtype = str(cfg.get('model_type') or 'protein_mpnn').lower()
            if subtype not in _MPNN_CHECKPOINTS:
                fail('model_type', f'must be one of {sorted(_MPNN_CHECKPOINTS)!r}')
            elif normalized_kind in _PROTEIN_MPNN_ONLY and subtype != 'protein_mpnn':
                fail('model_type', 'this ProteinMPNN family requires protein_mpnn')
            else:
                asset(_MPNN_CHECKPOINTS[subtype], required=True)
        elif normalized_kind == 'dynamicmpnn':
            mode = cfg.get('execution_mode') or 'daemon'
            if mode == 'daemon' and not isinstance(value('backend_factory'), str):
                fail('backend_factory', 'is required as an import path for daemon mode')
            if mode not in {'daemon', 'in_process'}:
                fail('execution_mode', 'must be daemon or in_process')
        elif normalized_kind == 'surfdesign':
            for field, asset_kind in (('python_bin', 'file'), ('repo_path', 'dir'), ('experiment_path', 'dir')):
                asset(field, required=True, kind=asset_kind, executable=field == 'python_bin')
            experiment = value('experiment_path')
            if experiment not in _MISSING:
                checkpoint = cfg.get('checkpoint') or 'best.ckpt'
                target = Path(checkpoint).expanduser()
                if not target.is_absolute():
                    target = Path(experiment).expanduser() / 'checkpoints' / target
                if check_paths and not target.is_file():
                    fail('checkpoint', f'file does not exist: {target}')
                if check_paths and not (Path(experiment).expanduser() / '.hydra/config.yaml').is_file():
                    fail('experiment_path', 'missing .hydra/config.yaml')
            choice('surface_mode', {'pymol'}, 'pymol')
            number('logit_temperature', 1.0, 0.000000001)
            if value('daemon') is False:
                fail('daemon', 'SurfDesign requires persistent daemon')
            if value('daemon_script_path') not in _MISSING:
                asset('daemon_script_path')
        elif normalized_kind == 'surfpro':
            asset('python_bin', required=True, executable=True)
            repo = asset('repo_path', required=True, kind='dir')
            if repo is not None and check_paths and not (repo / 'fairseq/models/surface_protein_model.py').is_file():
                fail('repo_path', 'missing fairseq/models/surface_protein_model.py')
            checkpoint = cfg.get('checkpoint') or 'checkpoint_best.pt'
            if Path(checkpoint).is_absolute():
                asset('checkpoint')
            else:
                directory = asset('checkpoint_dir', required=True, kind='dir')
                if directory is not None and check_paths and not (directory / checkpoint).is_file():
                    fail('checkpoint', f'file does not exist: {directory / checkpoint}')
            for field in ('msms_bin', 'pdb_to_xyzrn_bin'):
                asset(field, required=True, executable=True)
            device = str(cfg.get('device') or 'cuda')
            if re.fullmatch(r'cuda(?::\d+)?', device) is None:
                fail('device', 'must be cuda or cuda:N')
            number('logit_temperature', 1.0, 0.000000001)
            number('max_surface_vertices', 5000, 2, integer=True)
            if value('daemon') is False:
                fail('daemon', 'SurfPro requires persistent daemon')
            if value('daemon_script_path') not in _MISSING:
                asset('daemon_script_path')
        elif normalized_kind == 'nampnn':
            repo = asset('repo_path', required=True, kind='dir')
            if repo is not None:
                checkpoint = value('checkpoint') or 'models/design_model/s_19137.pt'
                target = Path(checkpoint).expanduser()
                if not target.is_absolute():
                    target = repo / target
                if check_paths and not target.is_file():
                    fail('checkpoint', f'file does not exist: {target}')
            choice('execution_mode', {'daemon', 'subprocess'}, 'daemon')
        elif normalized_kind == 'carbondesign':
            repo = asset('repo_path', required=True, kind='dir')
            for field in ('model_ckpt', 'model_config', 'model_features'):
                asset(field, required=True, base=repo)
    elif family == 'initializer' and kind == 'rfd3':
        asset('repo_path', required=True, kind='dir')
        asset('python_bin', required=True, executable=True)
        asset('ckpt_path', required=True, explicit_only=True)
        choice('execution_mode', {'daemon', 'subprocess'}, 'daemon')
    return errors
