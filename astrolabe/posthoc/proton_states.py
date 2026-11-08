"""Bounded audit of explicit His states on retained Potts contexts."""
from __future__ import annotations
import hashlib
import json
import math
import shutil
from pathlib import Path


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _network_input(item, output_dir, config):
    """Use unchanged ph_core geometry and persist its exact scoring input."""
    from astrolabe.losses import ph_core
    source = Path(item['pdb_path'])
    stat = source.stat()
    term = config.get('loss', {}).get('loss_terms', {}).get('neutral_preference', {})
    defaults = dict(hbond_scale=.5, depth_scale=4., probe_radius=1.4, resolution=.25)
    params = ph_core._network_parameters(**{k: term.get(k, v) for k, v in defaults.items()})
    evidence = ph_core._network_cached(str(source), item['input_sha256'],
        stat.st_mtime_ns, stat.st_size, 'A', item['binder_chain'], params.cache_tuple)
    if evidence['binder_sequence'] != item['binder_sequence']:
        raise ValueError('Network normalization changed the declared binder sequence.')
    if evidence['histidine_positions_zero_based'] != sorted(item['histidine_positions']):
        raise ValueError('Network scoring requires explicit coverage of every binder His.')
    normalized = Path(evidence['normalized_apo_path'])
    retained = output_dir/'normalized_inputs'/(_sha(normalized)+'.pdb')
    retained.parent.mkdir(parents=True, exist_ok=True)
    if not retained.exists():
        shutil.copyfile(normalized, retained)
    features = dict(pdb_path=str(retained), binder_chain='A',
        binder_sequence=evidence['binder_sequence'],
        chain_sequences={'A': evidence['binder_sequence']},
        histidine_positions=item['histidine_positions'])
    return evidence, features, dict(gap_scale=float(term.get('gap_scale', 1.)),
                                   gap_reference=float(term.get('gap_reference', 0.)))


def _network_metrics(evidence, result, parameters):
    """Do not substitute a whole-set gap for a different selected component."""
    values = dict(N=float(evidence['neutral']), G=None, R=0., U=int(evidence['unpaired']))
    if evidence['status'] == 'no_network':
        return values
    centers = [p-1 for p in evidence['selected_component']['histidine_positions']]
    if result is None or set(centers) != set(result['histidine_positions']):
        raise ValueError('Selected component differs from the explicitly audited His set; separate scoring required.')
    gap = float(result['gap'])
    scale, reference = parameters['gap_scale'], parameters['gap_reference']
    if not math.isfinite(gap) or not math.isfinite(scale) or scale <= 0 or not math.isfinite(reference):
        raise ValueError('Invalid release score parameters or gap.')
    values.update(G=gap, R=values['N']*math.tanh((gap/len(centers)-reference)/scale))
    return values


def run_from_config(*, config_path, inputs_path, output_dir, dry_run=False, network_scores=False):
    from omegaconf import OmegaConf
    config_path, inputs_path = Path(config_path).resolve(), Path(inputs_path).resolve()
    output_dir = Path(output_dir).resolve()
    manifest = json.loads(inputs_path.read_text())
    if manifest.get('schema') != 'astrolabe_proton_state_audit_v1':
        raise ValueError('Expected astrolabe_proton_state_audit_v1 manifest.')
    items = manifest.get('items')
    if not isinstance(items, list) or not items or len(items) > 16:
        raise ValueError('Audit requires 1..16 explicit items.')
    names = set()
    for item in items:
        name = item.get('item_id')
        if not isinstance(name, str) or not name or name in names:
            raise ValueError('Audit item IDs must be nonempty and unique.')
        names.add(name)
        source = Path(item['pdb_path'])
        if not source.is_file() or _sha(source) != item['input_sha256']:
            raise ValueError(f'Input identity/hash mismatch for {name}.')
        if not isinstance(item.get('histidine_positions'), list):
            raise ValueError(f'Explicit center positions required for {name}.')
    summary = dict(scope='fixed geometry Potts state audit; no probability, pKa, AF3, or geometry relaxation',
                   config_sha256=_sha(config_path), input_manifest_sha256=_sha(inputs_path),
                   dry_run=bool(dry_run), items=[], state_calls=0, AF3_calls=0)
    if dry_run:
        summary['items'] = [dict(item_id=x['item_id'], status='validated_inputs_only') for x in items]
        return summary
    output_dir.mkdir(parents=True, exist_ok=False)
    cfg = OmegaConf.load(config_path)
    if network_scores:
        summary['network_protocol'] = dict(definition='unchanged ph_core_network_v1',
            geometry='binder-only normalized coordinates; loaded neutral His tautomer; no repack',
            states='HIS-S/HIS-P are model labels, not pH probabilities',
            score_expression='R=N*tanh((G/nHis-gap_reference)/gap_scale)',
            parent_G='not applicable for zero His', F=None, AF3_calls=0)
    mutator = None
    cache = {}
    try:
        for item in items:
            record = dict(item_id=item['item_id'], input_sha256=item['input_sha256'])
            network = None
            if network_scores:
                network, normalized_features, release_parameters = _network_input(item, output_dir, cfg)
                record.update(network_evidence=network,
                    scoring_input_sha256=_sha(Path(normalized_features['pdb_path'])),
                    scoring_input=normalized_features,
                    release_parameters=release_parameters)
            if not item['histidine_positions']:
                # The parent has no switch centers. An empty state gap is undefined.
                record.update(status='not_applicable_zero_selected_His', result=None)
            else:
                features = {k: item[k] for k in ('pdb_path', 'binder_chain', 'binder_sequence',
                            'chain_sequences', 'histidine_positions')}
                if network_scores:
                    features = normalized_features
                for key in ('state_patterns', 'encoded_binder_tokens'):
                    if key in item:
                        features[key] = item[key]
                cache_key = json.dumps(features, sort_keys=True)
                if cache_key in cache:
                    record.update(status='reused_exact_audit', same_as=cache[cache_key]['item_id'],
                                  result=cache[cache_key]['result'])
                else:
                    if mutator is None:
                        from astrolabe.mutators.base import _build_proton_pottsmpnn_mutator
                        mutator = _build_proton_pottsmpnn_mutator({}, alias='state_audit',
                            mutator_conf=cfg.mutator, raw_model_results_dir=output_dir/'raw')
                        runner = mutator.runner
                        summary['model'] = dict(checkpoint_path=str(runner.checkpoint_path),
                            checkpoint_sha256=_sha(runner.checkpoint_path),
                            repo_path=str(runner.repo_path), extended_vocab=runner.extended_vocab,
                            device=runner.device, seed=runner.seed)
                    result = mutator.runner.score_states(**features)
                    summary['state_calls'] += 1
                    record.update(status='scored', result=result)
                    cache[cache_key] = record
            if network_scores:
                record['network_scores'] = _network_metrics(network, record['result'], release_parameters)
            summary['items'].append(record)
            (output_dir/'result.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    except Exception as exc:
        summary['failure'] = dict(type=type(exc).__name__, error=str(exc),
                                  item_id=item.get('item_id'))
        (output_dir/'result.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
        raise
    finally:
        if mutator is not None:
            mutator.shutdown()
    return summary
