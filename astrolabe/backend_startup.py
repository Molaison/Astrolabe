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


def _validate_common_options(family, kind, cfg, root, *, allow_unresolved=False):
    """Check consumed controls, not the vendor's entire scientific schema.

    Only explicitly supplied values are checked. In particular, null remains
    meaningful for optional assets, attached services and inherited execution
    policy. Numeric YAML strings and booleans are not numeric controls.
    """
    errors = []
    absent = object()

    def get(field):
        node = cfg
        for part in field.split('.'):
            if not isinstance(node, Mapping) or part not in node:
                return absent
            node = node[part]
        if allow_unresolved and isinstance(node, str) and '${' in node:
            return absent
        return node

    def fail(field, message):
        errors.append(f'{root}.{field}: {message}')

    def numeric(field, low=0, *, integer=True, high=None, positive=False, nullable=False):
        raw = get(field)
        if raw is absent or (raw is None and nullable):
            return
        correct_type = type(raw) is int if integer else type(raw) in (int, float)
        try:
            good = (correct_type and (type(raw) is int or math.isfinite(raw)) and raw >= low
                    and (not positive or raw > 0) and (high is None or raw <= high))
        except (OverflowError, TypeError, ValueError):
            good = False
        if not good:
            bound = '> 0' if positive else f'>= {low}'
            if high is not None:
                bound += f' and <= {high}'
            fail(field, f'expected {"integer" if integer else "finite number"} {bound}')

    def boolean(field, *, binary=False):
        raw = get(field)
        if raw is not absent and not (type(raw) is bool or (binary and type(raw) is int and raw in (0, 1))):
            fail(field, 'expected boolean' + (' or integer 0/1' if binary else ''))

    def enum(field, allowed, *, nullable=False):
        raw = get(field)
        if raw is not absent and not (raw is None and nullable):
            if not isinstance(raw, str) or raw not in allowed:
                fail(field, f'must be one of {sorted(allowed)!r}')

    def text(field, *, nullable=False):
        raw = get(field)
        if raw is not absent and not (nullable and raw in (None, '')):
            if not isinstance(raw, (str, os.PathLike)) or not str(raw).strip():
                fail(field, 'expected nonempty name or path')

    def mapping(field):
        raw = get(field)
        if raw is not absent and not isinstance(raw, Mapping):
            fail(field, 'expected mapping')

    def seed(field='seed', *, nullable=True):
        numeric(field, high=2**32 - 1, nullable=nullable)

    # NativeModelClient consumes these for AF2/AF3/ESMFold/RhoFold.
    native = family == 'evaluator' and kind in {'alphafold2', 'alphafold3', 'esmfold', 'rhofold'}
    mpnn = family == 'mutator' and str(kind).lower() in {name.lower() for name in _MPNN}
    if native or mpnn:
        enum('execution_mode', {'daemon', 'in_process'}, nullable=True)
        if native:
            enum('execution', {'daemon', 'in_process'}, nullable=True)
    for field in ('execution_request_timeout', 'request_timeout', 'startup_timeout', 'shutdown_timeout'):
        numeric(field, integer=False, positive=True, nullable=True)
    numeric('execution_max_frame_bytes', 1)

    if family == 'evaluator':
        if kind == 'bioemu2':
            for field in ('num_samples', 'batch_size', 'propagation_steps'):
                numeric(field, 1)
            numeric('lag_time_ns', integer=False, positive=True)
            seed()
            enum('mode', {'equilibrium', 'structure', 'propagation'})
            enum('sampler', {'edm', 'edm_gd', 'edm_fk', 'edm_fkc'})
            enum('matmul_precision', {'highest', 'high', 'medium'})
            for field in ('device', 'precision'):
                text(field)
        elif kind == 'alphafold2':
            numeric('model_id', 1, high=5)
            enum('suffix', {'ptm', 'multimer', 'multimer_v2', 'multimer_v3'})
            numeric('recycles')
            numeric('msa_clusters', 1)
            mapping('initial_guess')
            for field in ('enabled', 'use_initial_prev_pos', 'use_template_features',
                          'allow_missing_chains', 'allow_missing_residues'):
                # enabled:null means auto-enable when a path is provided.
                if field != 'enabled' or get('initial_guess.enabled') is not None:
                    boolean('initial_guess.' + field)
        elif kind == 'alphafold3':
            for field in ('num_samples', 'num_seeds'):
                numeric(field, 1)
            for field in ('num_recycles', 'refstructure_workers', 'refstructure_cache_max_entries',
                          'msa_cache_max_entries', 'bucket_pad_tolerance'):
                numeric(field)
            for field in ('refstructure_cache', 'msa_cache'):
                boolean(field, binary=True)
            boolean('save_embeddings')
            text('service_socket', nullable=True)
            buckets = get('buckets')
            if buckets is not absent and buckets is not None and (
                    not isinstance(buckets, (list, tuple)) or not buckets
                    or any(type(x) is not int or x <= 0 for x in buckets)
                    or list(buckets) != sorted(set(buckets))):
                fail('buckets', 'expected strictly increasing positive integer list or null')
        elif kind == 'esmfold':
            text('model_name')  # Both cache identifiers and local .pt files work.
            text('device', nullable=True)
        elif kind == 'protenix':
            text('model_name', nullable=True)  # May be inferred from checkpoint stem.
            enum('dtype', {'fp32', 'fp16', 'bf16'})
            for field in ('model', 'sample_diffusion', 'sample_diffusion.guidance',
                          'infer_setting', 'data', 'data.msa', 'data.template',
                          'model.constraint_embedder', 'skip_amp'):
                mapping(field)
            numeric('model.N_cycle')  # Native trunk initializes s/z before the recycle loop.
            for field in ('N_sample', 'N_step', 'N_sample_mini_rollout', 'N_step_mini_rollout'):
                numeric('sample_diffusion.' + field, 1)
            for field in ('gamma0', 'gamma_min', 'noise_scale_lambda', 'step_scale_eta'):
                numeric('sample_diffusion.' + field, integer=False)
            for field in ('chunk_size', 'lddt_metrics_chunk_size', 'sample_diffusion_chunk_size'):
                numeric('infer_setting.' + field, 1, nullable=True)
            for field in ('num_workers', 'data.num_dl_workers'):
                numeric(field)
            for field in ('need_atom_confidence', 'sorted_by_ranking_score', 'load_strict',
                          'use_msa', 'enable_tf32', 'enable_efficient_fusion',
                          'enable_diffusion_shared_vars_cache', 'msa_pair_as_unpair',
                          'use_template', 'use_rna_msa', 'use_seeds_in_json',
                          'use_deepspeed_evo_attention', 'data.msa.enable',
                          'data.msa.enable_rna_msa', 'data.template.enable',
                          'sample_diffusion.guidance.enable'):
                boolean(field)
            seed()
            seeds = get('seeds')
            if seeds is not absent:
                if not isinstance(seeds, (list, tuple)) or not seeds:
                    fail('seeds', 'expected nonempty integer seed list')
                else:
                    for index, item in enumerate(seeds):
                        if type(item) is not int or not 0 <= item <= 2**32 - 1:
                            fail(f'seeds.{index}', 'expected integer seed in [0, 4294967295]')
        elif kind == 'boltz':
            numeric('recycling_steps')
            numeric('num_workers')  # DataLoader supports zero workers.
            for field in ('sampling_steps', 'diffusion_samples'):
                numeric(field, 1)
            devices = get('devices')
            if devices is not absent and not (
                    type(devices) is int and devices > 0 or devices == 'auto'
                    or isinstance(devices, (list, tuple)) and devices
                    and all(type(x) is int and x >= 0 for x in devices)
                    and len(set(devices)) == len(devices)):
                fail('devices', 'expected positive count, auto, or unique nonnegative device indices')
            enum('accelerator', {'auto', 'cpu', 'gpu', 'cuda', 'mps', 'tpu'})
            enum('output_format', {'mmcif', 'pdb'})
            enum('msa_pairing_strategy', {'greedy', 'complete'})
            for field in ('write_full_pae', 'write_full_pde', 'use_msa_server'):
                boolean(field)
            text('msa_server_url', nullable=True)
        elif kind == 'chai1':
            numeric('num_trunk_recycles')
            numeric('recycle_msa_subsample')
            for field in ('num_diffn_timesteps', 'num_diffn_samples', 'num_trunk_samples'):
                numeric(field, 1)
            for field in ('use_esm_embeddings', 'use_msa_server', 'use_templates_server',
                          'low_memory', 'fasta_names_as_cif_chains'):
                boolean(field)
            seed()
        elif kind == 'rf3':
            numeric('n_recycles')
            for field in ('diffusion_batch_size', 'num_steps'):
                numeric(field, 1)
            numeric('template_noise_scale', integer=False)
            numeric('early_stopping_plddt_threshold', integer=False, nullable=True)
            for field in ('fallback_conformer_to_input_coords', 'compress_outputs'):
                # Null fallback deliberately removes the override in _engine_config.
                if field != 'fallback_conformer_to_input_coords' or get(field) is not None:
                    boolean(field)
    elif family == 'mutator':
        normalized = str(kind).lower()
        if normalized == 'rfd4':
            numeric('max_fixed_atom_rmsd', integer=False)
            numeric('request_timeout', integer=False, positive=True)
            seed(nullable=False)
        if mpnn:
            numeric('number_of_batches', 1)
            numeric('temperature', integer=False, positive=True)
            seed()
            for field in ('homo_oligomer', 'ligand_mpnn_use_atom_context',
                          'ligand_mpnn_use_side_chain_context', 'parse_atoms_with_zero_occupancy',
                          'save_stats', 'verbose', 'zero_indexed'):
                boolean(field, binary=True)
            enum('bias_mode', {'additive', 'multiplicative'})
            numeric('ligand_mpnn_cutoff_for_score', integer=False)
            for field in ('fallback_to_all_mutable_if_selector_empty', 'enforce_mutation_num',
                          'map_sequence_positions_to_pdb_residues'):
                boolean(field)
            enum('enforce_mutation_distribution', {'raw_log_probs', 'sampling_probs'})
            numeric('supplemental_structure_weight', integer=False)
            numeric('entropy_cutoff', integer=False, nullable=True)
        elif normalized == 'dplm':
            for field in ('max_iter', 'num_samples'):
                numeric(field, 1)
            numeric('temperature', integer=False, positive=True)
            text('model_name')  # HF names and local directories are both supported.
            text('sampling_strategy')  # The installed DPLM defines the strategy names.
            text('device')
            for field in ('daemon', 'response_cache', 'prefetch_save',
                          'prefetch_invalidate_on_sequence_change', 'quality_filter_enable',
                          'quality_filter_fallback_to_raw'):
                boolean(field)
            for field in ('quality_filter_max_rounds', 'quality_filter_attempt_num_samples'):
                numeric(field, nullable=True)
        elif normalized in {'mcmc', 'pore_blocker_mcmc'}:
            # mutation_method is stored but not dispatched on; do not invent an enum.
            numeric('max_mutations', low=-math.inf, nullable=True)
            boolean('avoid_current_residue')
            for field in ('allowed_residues', 'disallowed_residues'):
                residues = get(field)
                if residues is not absent and residues is not None and (
                        not isinstance(residues, (str, list, tuple))
                        or any(not isinstance(x, str) or len(x) != 1 or x.upper() not in 'ACDEFGHIKLMNPQRSTVWY' for x in residues)):
                    fail(field, 'expected protein residue letters or null')
    return errors


def _validate_af2_initial_guess(guess, *, path, base, check_paths, allow_unresolved):
    """Mirror AF2's per-oligomer overlay/path precedence without loading a structure."""
    if not isinstance(guess, Mapping):
        return []  # The common mapping check already reports this.
    errors = []

    def unresolved(raw):
        return allow_unresolved and isinstance(raw, str) and '${' in raw

    def fail(field, reason):
        errors.append(f'{path}.{field}: {reason}')

    def check(block, origins):
        enabled = block.get('enabled')
        if unresolved(enabled):
            return
        # The consumer's auto-enable rule uses path, not source or legacy alias.
        if enabled is False or (enabled is None and not block.get('path')):
            return
        for flag in ('enabled', 'use_initial_prev_pos', 'use_template_features',
                     'allow_missing_chains', 'allow_missing_residues'):
            if flag in block and not unresolved(block[flag]) and not (
                    type(block[flag]) is bool or flag == 'enabled' and block[flag] is None):
                fail(origins.get(flag, flag), 'expected boolean' + (' or null' if flag == 'enabled' else ''))
        field = 'path' if block.get('path') else 'init_guess_path' if 'init_guess_path' in block else 'path'
        raw = block.get(field)
        label = origins.get(field, field)
        if not unresolved(raw):
            if raw is None or raw == '':
                fail(label, 'is required when initial_guess is enabled (path or init_guess_path)')
            elif not isinstance(raw, (str, os.PathLike)) or not str(raw).strip():
                fail(label, 'must be a nonempty structure file path')
            else:
                target = Path(raw).expanduser()
                if base is not None and not target.is_absolute():
                    target = base / target
                if check_paths and not target.is_file():
                    fail(label, f'file does not exist: {target}')
        chain_map = block.get('chain_map')
        if chain_map is None or unresolved(chain_map):
            return
        label = origins.get('chain_map', 'chain_map')
        if not isinstance(chain_map, Mapping):
            fail(label, 'must map input chain or protomer names to source chain names or null')
            return
        for chain, source_chain in chain_map.items():
            if not isinstance(chain, str) or not chain:
                fail(label, 'keys must be nonempty input chain or protomer names')
            elif not unresolved(source_chain):
                if source_chain is not None and not isinstance(source_chain, str):
                    fail(f'{label}.{chain}', 'must be a source chain name or null')
                elif source_chain in (None, '', '-') and not (
                        block.get('allow_missing_chains', False) is True
                        or unresolved(block.get('allow_missing_chains'))):
                    fail(f'{label}.{chain}', 'omitted chain requires allow_missing_chains=true')

    # These are settings, not the consumer's inline per-oligomer blocks.
    fields = {'enabled', 'path', 'source', 'init_guess_path', 'chain_map',
              'allow_missing_chains', 'allow_missing_residues', 'use_initial_prev_pos',
              'use_template_features', 'query_residue_index_jumps', 'residue_index_jumps'}
    tables = {}
    for table in ('references', 'paths', 'chain_maps'):
        raw = guess.get(table)
        if isinstance(raw, Mapping):
            tables[table] = raw
        elif raw is not None and not unresolved(raw) and guess.get('enabled') is not False:
            fail(table, 'expected per-oligomer mapping')
    inline = {key: val for key, val in guess.items()
              if key not in fields | {'references', 'paths', 'chain_maps'} and isinstance(val, Mapping)}
    names = set(inline)
    for table in tables.values():
        names.update(table)
    global_block = {key: val for key, val in guess.items() if key in fields}
    origins = {key: key for key in global_block}
    if not names or guess.get('path'):
        check(global_block, origins)
    for name in names:
        block, locations = dict(global_block), dict(origins)
        override = inline.get(name)
        prefix = str(name)
        if override is None and name in tables.get('references', {}):
            override = tables['references'][name]
            prefix = f'references.{name}'
            if not isinstance(override, Mapping):
                if guess.get('enabled') is not False:
                    fail(prefix, 'expected initial-guess mapping')
                continue
        if override is not None:
            block.update(override)
            locations.update({key: f'{prefix}.{key}' for key in override})
        else:
            for table, field in (('paths', 'path'), ('chain_maps', 'chain_map')):
                if name in tables.get(table, {}):
                    block[field] = tables[table][name]
                    locations[field] = f'{table}.{name}'
        check(block, locations)
    return errors


def validate_backend_settings(family, kind, options, *, path, check_paths,
                              allow_unresolved=False) -> list[str]:
    """Return field-specific errors without side effects.

    ``allow_unresolved`` is solely for the caller's intentional unresolved
    dry-run pass; disabling filesystem checks alone does not enable it.
    """
    root = str(path).rstrip('.')
    errors: list[str] = []
    if not isinstance(options, Mapping):
        return [f'{root}: expected a mapping of resolved backend settings']
    cfg = options
    errors.extend(_validate_common_options(family, kind, cfg, root,
                                           allow_unresolved=allow_unresolved))
    child_base = None
    if family == 'evaluator' and kind in {'alphafold2', 'alphafold3', 'esmfold', 'rhofold'}:
        from astrolabe.model_execution import execution_directory, execution_mode
        try:
            selected_mode = cfg.get('execution_mode')
            if selected_mode is None:
                selected_mode = cfg.get('execution')
            mode = execution_mode(None if allow_unresolved and isinstance(selected_mode, str)
                                  and '${' in selected_mode else selected_mode)
        except (TypeError, ValueError) as error:
            field = 'execution_mode' if cfg.get('execution_mode') is not None else 'execution'
            errors.append(f'{root}.{field}: {error}')
            return errors
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
        if allow_unresolved and isinstance(selected, str) and '${' in selected:
            return
        if not isinstance(selected, str) or selected not in allowed:
            fail(field, f'must be one of {sorted(allowed)!r}')

    def number(field, default, low, *, integer=False, high=None):
        selected = cfg.get(field, default)
        if allow_unresolved and isinstance(selected, str) and '${' in selected:
            return
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
            if allow_unresolved and isinstance(env, str) and '${' in env:
                continue
            if not isinstance(env, Mapping) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in env.items()):
                fail(field, 'must map string names to string values')

    if family == 'evaluator':
        if kind == 'bioemu2':
            asset('python_bin', required=True, executable=True)
            asset('cache_dir', required=True, kind='dir')
            choice('mode', {'equilibrium', 'structure', 'propagation'}, 'equilibrium')
            if value('mode') == 'propagation':
                asset('initial_cif', required=True)
            elif value('initial_cif') not in _MISSING and not (
                allow_unresolved and isinstance(value('mode'), str) and '${' in value('mode')
            ):
                fail('initial_cif', 'is supported only in propagation mode')
            for field in ('execution_mode', 'execution'):
                if value(field) not in _MISSING and value(field) != 'subprocess':
                    fail(field, 'BioEmu2 loads in a subprocess for every request, not a staged daemon')
        elif kind == 'generated_structure':
            structures = value('initial_structures')
            if not (allow_unresolved and isinstance(structures, str) and '${' in structures):
                if not isinstance(structures, Mapping) or not structures:
                    fail('initial_structures', 'must map assembly names to initial generated coordinate files')
                else:
                    for name, filename in structures.items():
                        if not isinstance(filename, (str, os.PathLike)) or not str(filename).strip():
                            fail('initial_structures', f'invalid path for {name!r}')
                        elif check_paths and not Path(filename).expanduser().is_file():
                            fail('initial_structures', f'file does not exist: {filename}')
        elif kind == 'alphafold3':
            if not value('service_socket') and value('model_dir') not in _MISSING:
                asset('model_dir', kind='dir', base=child_base)
            if value('flash_attention_implementation') not in _MISSING:
                choice('flash_attention_implementation', {'cudnn', 'triton', 'xla'})
            # alphafold3_setup returns before local cache setup for attached services.
            cache = value('jax_compilation_cache_dir')
            if not value('service_socket') and cache is not None and cache != '':
                if allow_unresolved and isinstance(cache, str) and '${' in cache:
                    pass
                elif not isinstance(cache, (str, os.PathLike)) or not str(cache).strip():
                    fail('jax_compilation_cache_dir', 'must be a directory path or null')
                else:
                    target = Path(cache).expanduser()
                    if child_base is not None and not target.is_absolute():
                        target = child_base / target
                    if check_paths and target.exists() and not target.is_dir():
                        fail('jax_compilation_cache_dir', f'existing path is not a directory: {target}')
        elif kind == 'alphafold2':
            if value('data_dir') not in _MISSING:
                asset('data_dir', kind='dir', base=child_base)
            errors.extend(_validate_af2_initial_guess(
                value('initial_guess'), path=f'{root}.initial_guess', base=child_base,
                check_paths=check_paths, allow_unresolved=allow_unresolved,
            ))
        elif kind == 'esmfold':
            if value('model_name') not in _MISSING:
                asset('model_name', explicit_only=not str(value('model_name')).endswith('.pt'), base=child_base)
        elif kind == 'protenix':
            for field, asset_kind in (('load_checkpoint_path', 'file'), ('load_checkpoint_dir', 'dir')):
                if value(field) not in _MISSING:
                    asset(field, kind=asset_kind)
        elif kind == 'boltz':
            # Boltz creates its cache before download; this is an output path,
            # not an existing model asset. Validation must never create it.
            cache = value('cache')
            if cache not in _MISSING:
                if allow_unresolved and isinstance(cache, str) and '${' in cache:
                    pass
                elif not isinstance(cache, (str, os.PathLike)) or not str(cache).strip():
                    fail('cache', 'must be a directory path')
                else:
                    target = Path(cache).expanduser()
                    if check_paths and target.exists() and not target.is_dir():
                        fail('cache', f'existing path is not a directory: {target}')
            if value('checkpoint') not in _MISSING:
                asset('checkpoint')
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
        if normalized_kind == 'rfd4':
            asset('python_bin', required=True, executable=True)
            source = asset('source_dir', required=True, kind='dir')
            model = asset('model_dir', required=True, kind='dir')
            cache = value('cache_dir')
            if not (allow_unresolved and isinstance(cache, str) and '${' in cache):
                if not isinstance(cache, (str, os.PathLike)) or not str(cache).strip():
                    fail('cache_dir', 'is required as a directory path')
                elif check_paths and Path(cache).exists() and not Path(cache).is_dir():
                    fail('cache_dir', 'existing path is not a directory')
            if check_paths:
                for directory, field, files in (
                    (source, 'source_dir', ('src/rfproteina/generate.py', 'configs/inference.yaml')),
                    (model, 'model_dir', ('model.safetensors', 'config.json')),
                ):
                    if directory is not None and directory.is_dir():
                        for filename in files:
                            if not (directory / filename).is_file():
                                fail(field, f'missing {filename}')
            number('max_fixed_atom_rmsd', 0.5, 0)
            number('request_timeout', 1800, 0.000000001)
            generation = cfg.get('generation')
            if not (allow_unresolved and '${' in str(generation)):
                from astrolabe.mutators.rfd4_backend import validate_generation
                try:
                    validate_generation(generation)
                except (TypeError, ValueError) as error:
                    fail('generation', str(error))
        elif normalized_kind in {name.lower() for name in _MPNN}:
            if allow_unresolved and isinstance(value('model_type'), str) and '${' in value('model_type'):
                return errors  # The selected checkpoint cannot yet be determined.
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
        elif normalized_kind == 'dplm':
            if value('repo_path') not in _MISSING:
                asset('repo_path', kind='dir')
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
