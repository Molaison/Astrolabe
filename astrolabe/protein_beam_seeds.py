"""Explicit reuse of an existing root prediction under an unchanged AF3 contract."""
from copy import deepcopy
import json
from pathlib import Path


def load_root(cfg, source_state, contract):
    from omegaconf import OmegaConf
    from astrolabe.protein_beam_contract import content_sha256
    path=Path(source_state).resolve()
    prior=json.loads(path.read_text())
    if prior['status']!='complete' or not prior['budget_complete']:
        raise ValueError('Root import requires a completed source beam')
    previous=prior['scientific_contract']
    old=OmegaConf.create(previous['config_yaml'])
    # Proposal/selection policies may change, prediction and raw metric definitions may not.
    def plain(value):
        return OmegaConf.to_container(value,resolve=True) if OmegaConf.is_config(value) else value
    for key in ('seqs','types','homologies','seed','loss'):
        if plain(old[key]) != plain(cfg[key]):
            raise ValueError(f'Root prediction contract differs: {key}')
    def prediction_settings(config):
        value=OmegaConf.to_container(config.evaluator.alphafold3,resolve=True)
        value.pop('service_socket',None)
        return value
    if prediction_settings(old)!=prediction_settings(cfg):
        raise ValueError('Root prediction AF3 settings differ')
    if previous['readonly_weight_asset_stat']!=contract['readonly_weight_asset_stat']:
        raise ValueError('Root prediction model assets differ')
    identity=cfg.protein_beam.get('initial_prediction_identity') if cfg.get('protein_beam') else None
    if identity:
        audit=json.loads(Path(str(cfg.protein_beam.initial_state_audit)).read_text())
        if audit['status']!='passed' or audit['base_commit']!=str(identity.base_commit) or audit['current_source_sha256']!=previous['source_sha256']:
            raise ValueError('Root historical source identity is not established')
        package=Path(__file__).resolve().parent.parent
        required={'astrolabe/losses/confidence.py','astrolabe/evaluators/wrappers/alphafold3.py',
                  'astrolabe/evaluators/af3_service.py','astrolabe/core/folding_input.py'}
        if not required<=set(identity.file_sha256):
            raise ValueError('Incomplete root prediction source identity')
        for filename,digest in identity.file_sha256.items():
            if content_sha256(package/filename)!=str(digest):
                raise ValueError('Root prediction implementation changed')
        if previous['source_sha256']['protein_beam_inputs.py']!=contract['source_sha256']['protein_beam_inputs.py']:
            raise ValueError('Root fixed-target input adapter changed')
    else:
        for name in ('interface_holes.py','interaction.py'):
            if previous['source_sha256'][name]!=contract['source_sha256'][name]:
                raise ValueError('Root native metric implementation differs without audited prediction provenance')
    for filename,digest in previous['data_sha256'].items():
        if contract['data_sha256'].get(filename)!=digest or content_sha256(filename)!=digest:
            raise ValueError('Root input data differ')
    record=deepcopy(prior['records'][0])
    expected=''.join(str(cfg.seqs.A).replace('+','').split())
    if record['id']!=0 or record['parent_id'] is not None or record['sequence']!=expected or record['seed']!=int(cfg.seed) or record['status']!='done':
        raise ValueError('Source record is not the exact requested root')
    for key in ('snapshot','structure_path'):
        if not Path(record[key]).is_file():
            raise FileNotFoundError(f'Root native {key} unavailable')
    native=path.parent/'candidates/000000/result.json'
    original=json.loads(native.read_text())
    if any(original[k]!=record[k] for k in ('sequence','seed','metrics','snapshot','structure_path','score')):
        raise ValueError('Root checkpoint differs from native terminal result')
    record.pop('selection',None)
    record.update(evaluation_origin='reused_native',
                  imported_from=dict(beam_state=str(path),config_hash=prior['config_hash'],
                                     native_result=str(native),native_result_sha256=content_sha256(native),
                                     original_metrics=deepcopy(record['metrics']),
                                     snapshot_sha256=content_sha256(record['snapshot']),
                                     structure_sha256=content_sha256(record['structure_path'])))
    return record
