"""Posthoc coupling must retain geometry gating and undefined parent gaps."""
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from astrolabe.posthoc import proton_states as states


def test_positive_explicit_gap_cannot_bypass_missing_geometry():
    geometry = dict(status='no_network', neutral=0., unpaired=2)
    assert states._network_metrics(geometry, {'gap': 100.}, {}) == dict(N=0., G=None, R=0., U=2)


@pytest.mark.parametrize('gap', [-4., 0., 4.])
def test_native_release_formula_retains_direction_and_center_normalization(gap):
    geometry = dict(status='network', neutral=.6, unpaired=0,
                    selected_component={'histidine_positions':[2, 5]})
    result = states._network_metrics(geometry, dict(gap=gap,histidine_positions=[4,1]),
                                    dict(gap_scale=1.,gap_reference=0.))
    assert result['R'] == pytest.approx(.6*math.tanh(gap/2))
    assert result['G'] == gap


def test_whole_set_gap_is_not_reused_for_a_different_component():
    geometry=dict(status='network',neutral=.8,unpaired=1,
                  selected_component={'histidine_positions':[2]})
    with pytest.raises(ValueError,match='separate scoring required'):
        states._network_metrics(geometry,dict(gap=5.,histidine_positions=[1,4]),
                                dict(gap_scale=1.,gap_reference=0.))


def test_opt_in_parent_keeps_no_model_and_undefined_gap(tmp_path,monkeypatch):
    src=tmp_path/'source.pdb';src.write_text('retained parent')
    cfg=tmp_path/'config.yaml';cfg.write_text('mutator: {}\n')
    item=dict(item_id='parent',pdb_path=str(src),input_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),histidine_positions=[])
    manifest=tmp_path/'input.json';manifest.write_text(json.dumps(dict(schema='astrolabe_proton_state_audit_v1',items=[item])))
    evidence=dict(status='no_network',neutral=0.,unpaired=0)
    monkeypatch.setattr(states,'_network_input',lambda *a:(evidence,dict(pdb_path=str(src)),{}))
    import astrolabe.mutators.base as base
    factory=Mock(side_effect=AssertionError('parent must not invoke Proton'))
    monkeypatch.setattr(base,'_build_proton_pottsmpnn_mutator',factory)
    out=states.run_from_config(config_path=cfg,inputs_path=manifest,output_dir=tmp_path/'out',network_scores=True)
    factory.assert_not_called()
    assert out['state_calls']==out['AF3_calls']==0
    assert out['items'][0]['network_scores']==dict(N=0.,G=None,R=0.,U=0)


def test_opt_in_cli_flag_reaches_existing_native_posthoc(monkeypatch):
    import astrolabe.cli as cli
    call=Mock(return_value={'items':[],'state_calls':0})
    monkeypatch.setattr(states,'run_from_config',call)
    ns=cli._build_parser().parse_args(['posthoc','proton-states','--config','c','--inputs','i','--out-dir','o','--network-scores'])
    assert ns._handler(ns)==0
    assert call.call_args.kwargs['network_scores'] is True


def test_actual_normalized_input_is_used_by_model(tmp_path,monkeypatch):
    source=tmp_path/'preview';source.write_text('unmodified preview')
    normalized=tmp_path/'normalized';normalized.write_text('ph_core normalized binder')
    evidence=dict(status='network',neutral=.5,unpaired=0,
                  selected_component={'histidine_positions':[1,3]})
    features=dict(pdb_path=str(normalized),binder_chain='A',binder_sequence='HAH',
                  chain_sequences={'A':'HAH'},histidine_positions=[0,2])
    item=dict(item_id='proposal',pdb_path=str(source),input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),histidine_positions=[0,2],binder_chain='A',binder_sequence='HAH',chain_sequences={'A':'HAH'},state_patterns=['SS','SP','PS','PP'])
    config=tmp_path/'config';config.write_text('mutator: {}\n')
    manifest=tmp_path/'manifest';manifest.write_text(json.dumps(dict(schema='astrolabe_proton_state_audit_v1',items=[item])))
    weights=tmp_path/'weights';weights.write_text('test only')
    runner=SimpleNamespace(checkpoint_path=weights,repo_path=tmp_path,extended_vocab='v6',device='cpu',seed=10,
        score_states=Mock(return_value=dict(gap=2.,histidine_positions=[0,2])))
    close=Mock()
    import astrolabe.mutators.base as base
    monkeypatch.setattr(base,'_build_proton_pottsmpnn_mutator',lambda *a,**k:SimpleNamespace(runner=runner,shutdown=close))
    monkeypatch.setattr(states,'_network_input',lambda *a:(evidence,features,dict(gap_scale=1.,gap_reference=0.)))
    out=states.run_from_config(config_path=config,inputs_path=manifest,output_dir=tmp_path/'out',network_scores=True)
    assert runner.score_states.call_args.kwargs['pdb_path']==str(normalized)
    assert out['items'][0]['input_sha256']==item['input_sha256']
    assert out['items'][0]['scoring_input_sha256']==hashlib.sha256(normalized.read_bytes()).hexdigest()
    assert out['items'][0]['network_scores']['R']==pytest.approx(.5*math.tanh(1.))
    close.assert_called_once()
