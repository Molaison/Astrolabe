"""Freeze pose limits from a newly predicted unmutated root, once per case."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path


def calibration_enabled(cfg):
    return bool((cfg.protein_beam.get('root_calibration') or {}).get('enabled', False))


def geometry_root(cfg, output, task, predicted):
    if not calibration_enabled(cfg):
        return cfg.case_input.get('root_structure')
    if task['id'] == 0:
        return predicted
    path = Path(output) / 'root_calibration.json'
    if not path.is_file():
        raise ValueError('Native root must be calibrated before child evaluation')
    return json.loads(path.read_text())['root_structure']


def calibrate_root(cfg, output, state, root):
    """Freeze measured geometry and verify immutable context on completed resume.

    The configured policy remains unchanged. Derived gates live in the state and
    in a separate frozen artifact, rather than changing the resume config hash.
    """
    from omegaconf import OmegaConf
    from astrolabe.protein_beam import atomic_json
    from astrolabe.protein_beam_contract import content_sha256
    from astrolabe.protein_beam_root_parity import _atoms, _confidence, _digest

    baseline = copy.deepcopy(root['geometry'])
    required = {'target_ca_rmsd', 'binder_pose_ca_rmsd', 'binder_self_ca_rmsd', 'interface_bb_rmsd',
                'root_binder_pose_ca_rmsd', 'root_interface_bb_rmsd', 'root_complex_bb_rmsd',
                'root_interface_binder_bb_rmsd'}
    if not required <= set(baseline) or any(not math.isfinite(float(baseline[k])) or float(baseline[k]) < 0 for k in required):
        raise ValueError('Root calibration requires all finite measured pose metrics')
    policy = cfg.protein_beam.root_calibration
    gates = OmegaConf.to_container(cfg.protein_beam.gates, resolve=True)
    gates = copy.deepcopy(gates)
    for metric in ('target_ca_rmsd', 'binder_pose_ca_rmsd', 'binder_self_ca_rmsd'):
        gates['geometry_max'][metric] = baseline[metric] + float(policy.original_ca_delta)
    gates['geometry_max']['interface_bb_rmsd'] = baseline['interface_bb_rmsd'] + float(policy.original_interface_bb_delta)
    fingerprint = {
        'record': _digest({k: root[k] for k in ('sequence', 'seed', 'metrics', 'holes', 'geometry')}),
        'atoms': _digest(_atoms(root['structure_path'])),
        'full_native_confidence': _digest(_confidence(root)),
        'proposal_snapshot': content_sha256(root['snapshot']),
        'native_terminal': content_sha256(Path(output) / 'candidates/000000/result.json'),
    }
    evidence = dict(root_structure=root['structure_path'], baseline_geometry=baseline,
                    gates=gates, fingerprint=fingerprint,
                    boundary='Frozen before proposals and verified on every resume')
    path = Path(output) / 'root_calibration.json'
    if path.exists():
        if json.loads(path.read_text()) != evidence:
            raise ValueError('Frozen native root calibration changed; retained original calibration')
    else:
        # A missing artifact after a recorded calibration is evidence of damage.
        if ('calibrated_gates' in state or state.get('root_calibration_completed') or
                state['status'] == 'complete' or state.get('completed_at') is not None or
                state.get('generation', 0) != 0 or state.get('proposals', 0) != 0 or
                state.get('generation_work') or any(r['id'] != 0 for r in state['records'])):
            raise ValueError('Frozen native root calibration artifact is missing')
        atomic_json(path, evidence)
    if 'calibrated_gates' in state and state['calibrated_gates'] != gates:
        raise ValueError('Checkpoint gates differ from frozen native root calibration')
    state['calibrated_gates'] = gates
    state['root_calibration_completed'] = True
    return evidence


def validate_native_result(task, result, output):
    """Check dispatched identity and local ownership before importing terminal data."""
    required = {'id', 'kind', 'generation', 'parent_id', 'sequence', 'seed', 'status',
                'metrics', 'holes', 'geometry', 'score', 'snapshot', 'structure_path'}
    if not required <= set(result):
        raise ValueError('Native terminal result is missing required fields')
    for key in ('id', 'kind', 'generation', 'parent_id', 'sequence', 'seed', 'af3_call_token'):
        if result.get(key) != task.get(key):
            raise ValueError(f'Native terminal result changed dispatched {key}')
    if result.get('attempt', 1) != task.get('attempt', 1) or result.get('status') != 'done':
        raise ValueError('Native terminal result has an invalid attempt/status')
    root = Path(output).resolve()
    expected = root / 'candidates' / f"{task['id']:06d}" / 'scores.pkl'
    if Path(result['snapshot']).resolve() != expected:
        raise ValueError('Native result snapshot is outside its candidate output')
    if not Path(result['structure_path']).resolve().is_relative_to(root / 'workers'):
        raise ValueError('Native result structure is outside its worker output')


def verify_calibrated_state(cfg, output, state):
    """Verify root identity, its original terminal record and saved proposal context."""
    sequence = ''.join(str(cfg.seqs.A).replace('+', '').split())
    def valid_identity(record):
        return (record['sequence'] == sequence and record['seed'] == int(cfg.seed) and
                record.get('parent_id') is None and record.get('generation') == 0 and record.get('kind') == 'evaluate')
    roots = [r for r in state['records'] if r['id'] == 0]
    if not roots and not state['records'] and state['status'] != 'complete' and state.get('generation', 0) == 0:
        if len(state['pending']) == 1 and state['pending'][0]['id'] == 0 and valid_identity(state['pending'][0]):
            return None
    if len(roots) != 1:
        raise ValueError('Calibrated beam requires exactly one native root')
    root = roots[0]
    if not valid_identity(root):
        raise ValueError('Native root identity differs from the original input')
    if root['status'] != 'done':
        if state['status'] != 'complete' and not any(r['id'] != 0 for r in state['records']):
            return None
        raise ValueError('Calibrated beam has no completed native root')
    raw = json.loads((Path(output) / 'candidates/000000/result.json').read_text())
    validate_native_result(root, raw, output)
    if any(root.get(key) != value for key, value in raw.items()):
        raise ValueError('Checkpoint root differs from its native terminal result')
    from astrolabe.protein_beam_geometry import verify_prediction_sequences
    verify_prediction_sequences(root['structure_path'], sequence, ''.join(str(cfg.seqs.B).replace('+', '').split()))
    records = {r['id']: r for r in state['records'] if r['status'] == 'done'}
    work = state.get('generation_work') or {}
    fields = ('sequence', 'snapshot', 'structure_path', 'seed', 'metrics', 'holes', 'geometry')
    for parent in work.get('parents', []):
        canonical = records.get(parent['id'])
        if canonical is None or any(parent.get(k) != canonical.get(k) for k in fields):
            raise ValueError('Saved generation parent differs from its verified record')
    for task in [*work.get('tasks', []), *state.get('pending', [])]:
        if task.get('parent_id') is None or task['id'] in records and task['kind'] != 'proposal':
            continue
        parent = records.get(task['parent_id'])
        if parent is None or task.get('snapshot') != parent['snapshot']:
            raise ValueError('Saved task uses a different parent snapshot')
        if task['kind'] == 'proposal' and (task['sequence'] != parent['sequence'] or
                task['seed'] != ((int(cfg.seed)*100003 + task['id']) % 2147483647 or 1)):
            raise ValueError('Saved proposal identity changed')
    return root
