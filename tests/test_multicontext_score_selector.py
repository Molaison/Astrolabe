from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from omegaconf import OmegaConf

from astrolabe.core import global_config
from astrolabe.mutators.base import build_mutators
from astrolabe.score_evidence import ScoreRecord, ScoreSpec
from astrolabe.selectors.utils import construct_position_selector
from test_mpnn_sync_policy import setup_case


def pair_record(context, labels, target_label, values, snapshot):
    matrix = np.zeros((len(labels), len(labels)), dtype=float)
    target = labels.index(target_label)
    for i, value in enumerate(values):
        matrix[i, target] = value
        matrix[target, i] = 99.0  # Reverse direction is deliberately different.
    return ScoreRecord(ScoreSpec('directed_pair','evaluator',('position','target_position'),context,'oligo'),
                       snapshot, matrix, (tuple(labels),tuple(labels)))


def config_for(context, labels, target, n, weight):
    return {'context':context,'weight':weight,
            'query':{'name':'directed_pair','source':'evaluator','entity':context,'entity_scope':'oligo',
                     'axes':['position'],'select':{'position':labels[:n], 'target_position':[target]}},
            'mapping':'auto'}


@pytest.mark.parametrize('producer', ['stored_records', 'standard_scores', 'merged_member_records', 'merged_member_standard'])
def test_weighted_directed_pair_selector_reaches_real_builder_and_mutate(tmp_path, monkeypatch, producer):
    protomers, scores, _ = setup_case(tmp_path, monkeypatch, 'probability_mean')
    ab = ['AB:A:1:A','AB:A:2:A','AB:A:3:A','AB:B:1:B','AB:B:2:B']
    ac = ['AC:A:1:A','AC:A:2:A','AC:A:3:A','AC:B:1:C','AC:B:2:C','AC:B:3:C','AC:B:4:C']
    records = (pair_record('AB',ab,ab[3],[1,9,2],scores.assessment_id),
               pair_record('AC',ac,ac[3],[7,3,2],scores.assessment_id))
    if producer == 'stored_records':
        scores.score_records = records
    elif producer == 'standard_scores':
        scores.pae = {'AB':[records[0].value], 'AC':[records[1].value]}
    else:
        from astrolabe.evaluators.utils import StandardScores
        from astrolabe.score_merge import merge_standard_scores
        members = []
        for context, labels, target, values in [('AB',ab,ab[3],[1,9,2]), ('AC',ac,ac[3],[7,3,2])]:
            member = StandardScores(names=[context],plddt={context:[.8]},complex_paths={context:scores.complex_paths[context]})
            record = pair_record(context,labels,target,values,member.assessment_id)
            if producer == 'merged_member_standard':
                member.pae = {context: [record.value]}
            else:
                member.score_records = (record,)
            members.append(member)
        scores = merge_standard_scores(members)
        if producer == 'merged_member_records':
            assert [record.snapshot for record in scores.score_records] == [member.assessment_id for member in members]
    global_config.init_const('decisions', SimpleNamespace(last_accepted=0,evaluation_results={0:scores}))
    global_config.init_const('stage_manager', SimpleNamespace(current_stage='default', transition_consistency_gate_context=None))
    spec = {'observations':[config_for('AB',ab,ab[3],3,1),config_for('AC',ac,ac[3],3,3)],
            'aggregation':{'reduction':'mean','count':'context','missing':'error'},
            'comparison':'lt','threshold':5}
    if producer == 'standard_scores':
        for observation in spec['observations']:
            observation['query']['name'] = 'pae'
    elif producer in {'merged_member_records', 'merged_member_standard'}:
        for observation in spec['observations']:
            observation['query']['snapshot'] = 'member'
            if producer == 'merged_member_standard':
                observation['query']['name'] = 'pae'
    runtime = {'score_selector_queries':OmegaConf.create({'two':spec}),'oligo':['AB','AC'],
               'ignored_complexes':[], 'negative_design_complexes':[]}
    cfg = OmegaConf.create({'enabled_aliases':['random'],
        'mutator_terms':{'random':{'_type':'MCMC', 'position_selector':'score("two")',
                                   'allowed_residues':['C'], 'avoid_current_residue':True}},
        'MCMC':{'mutation_method':'random','select_positions':'random'}})
    mutator = build_mutators(cfg,tmp_path)['random']; mutator.mutation_weight=1
    np.random.seed(21)
    mutator.mutate(protomers, scores, runtime)
    manager = global_config.segment_manager
    assert not manager.segments['A_1'].mutations
    assert not manager.segments['A_2'].mutations  # selected but physically fixed
    assert manager.segments['A_3'].mutations[0][2] == 'C'
    manager.apply_mutations()
    assert protomers['A'].sequence == 'ACC'
    global_config.reset()


def test_missing_position_requires_explicit_policy(tmp_path, monkeypatch):
    protomers, scores, _ = setup_case(tmp_path, monkeypatch, 'probability_mean')
    ab = ['AB:A:1:A','AB:A:2:A','AB:A:3:A','AB:B:1:B','AB:B:2:B']
    ac = ['AC:A:1:A','AC:A:2:A','AC:A:3:A','AC:B:1:C','AC:B:2:C','AC:B:3:C','AC:B:4:C']
    scores.score_records = (pair_record('AB',ab,ab[3],[1,9,2],scores.assessment_id),
                            pair_record('AC',ac,ac[3],[7,3,2],scores.assessment_id))
    spec = {'observations':[config_for('AB',ab,ab[3],3,1),config_for('AC',ac,ac[3],2,1)],
            'aggregation':{'reduction':'mean','count':'context','missing':'error'}, 'threshold':5,'comparison':'lt'}
    runtime={'score_selector_queries':{'two':spec},'score_capture_snapshot':scores.assessment_id,
             'score_snapshots':{'last_accepted':scores.assessment_id}}
    with pytest.raises(ValueError, match='missing from an active context'):
        construct_position_selector('score("two")',scores,protomers,runtime_config=runtime)
    spec['aggregation']['missing']='ignore'
    masks=construct_position_selector('score("two")',scores,protomers,runtime_config=runtime)
    assert masks['A'].tolist() == [True,False,True]
    global_config.reset()


def test_repeat_chain_context_vs_physical_count_and_explicit_weights():
    global_config.reset()
    protomers={'A':SimpleNamespace(length=1),'B':SimpleNamespace(length=1)}
    aab=['AAB:A:1:A','AAB:B:1:A','AAB:C:1:B']
    ab=['AB:A:1:A','AB:B:1:B']
    records=[pair_record('AAB',aab,aab[2],[0,10], 's'), pair_record('AB',ab,ab[1],[9], 's')]
    scores=SimpleNamespace(score_records=records, names=[], complex_poses=None)
    first=config_for('AAB',aab,aab[2],2,1); second=config_for('AB',ab,ab[1],1,1)
    spec={'observations':[first,second], 'aggregation':{'reduction':'mean','missing':'error','count':'context'},
          'threshold':6.8,'comparison':'ge'}
    runtime={'score_selector_queries':{'two':spec}, 'score_snapshots':{'last_accepted':'s'}}
    with pytest.raises(ValueError, match='explicit position_weights'):
        construct_position_selector('score("two")',scores,protomers,runtime_config=runtime)
    first['position_weights']='equal'
    assert construct_position_selector('score("two")',scores,protomers,runtime_config=runtime)['A'].tolist() == [True]
    spec['aggregation']['count']='physical'
    assert construct_position_selector('score("two")',scores,protomers,runtime_config=runtime)['A'].tolist() == [False]


def test_context_name_cannot_relabel_other_oligomer():
    global_config.reset()
    protomers={'A':SimpleNamespace(length=1),'B':SimpleNamespace(length=1),'C':SimpleNamespace(length=1)}
    ab=['AB:A:1:A','AB:B:1:B']; ac=['AC:A:1:A','AC:B:1:C']
    records=[pair_record('AB',ab,ab[1],[1],'s'),pair_record('AC',ac,ac[1],[2],'s')]
    scores=SimpleNamespace(score_records=records,names=[])
    wrong=config_for('AB',ab,ab[1],1,1); wrong['context']='AC'
    spec={'observations':[wrong,config_for('AB',ac,ac[1],1,1)],
          'aggregation':{'reduction':'mean','missing':'error','count':'context'},'threshold':3,'comparison':'lt'}
    with pytest.raises(ValueError, match='context differs'):
        construct_position_selector('score("two")',scores,protomers,
            runtime_config={'score_selector_queries':{'two':spec},'score_snapshots':{'last_accepted':'s'}})


def test_shared_segment_aliases_project_selection_to_each_protomer(tmp_path, monkeypatch):
    from astrolabe.core.protomer_config_parser import Protomer
    from astrolabe.evaluators.utils import StandardScores
    from pathlib import Path
    protomers, old_scores, _ = setup_case(tmp_path, monkeypatch, 'probability_mean')
    segment = global_config.segments['A_1']
    global_config.segments['shared_alias'] = segment
    protomers['X'] = Protomer('X','shared_alias','protein')
    scores = StandardScores(names=['AB','XC'],plddt={'AB':[.8],'XC':[.8]},complex_paths={'AB':Path('a'),'XC':Path('b')})
    ab=['AB:A:1:A','AB:A:2:A','AB:A:3:A','AB:B:1:B','AB:B:2:B']
    xc=['XC:A:1:X','XC:B:1:C','XC:B:2:C','XC:B:3:C','XC:B:4:C']
    scores.score_records = (pair_record('AB',ab,ab[3],[1,9,9],scores.assessment_id),
                            pair_record('XC',xc,xc[1],[3],scores.assessment_id))
    first=config_for('AB',ab,ab[3],1,1); second=config_for('XC',xc,xc[1],1,1)
    spec={'observations':[first,second], 'aggregation':{'reduction':'mean','missing':'error','count':'context'},
          'threshold':3,'comparison':'lt'}
    runtime={'score_selector_queries':{'two':spec},'score_capture_snapshot':scores.assessment_id,
             'score_snapshots':{'last_accepted':scores.assessment_id}}
    masks=construct_position_selector('score("two")',scores,protomers,runtime_config=runtime)
    assert masks['A'].tolist() == [True,False,False]
    assert masks['X'].tolist() == [True]
    global_config.reset()


def test_member_snapshot_cannot_be_used_against_stale_aggregate():
    global_config.reset()
    protomers={'A':SimpleNamespace(length=1),'B':SimpleNamespace(length=1),'C':SimpleNamespace(length=1)}
    ab=['AB:A:1:A','AB:B:1:B']; ac=['AC:A:1:A','AC:B:1:C']
    scores=SimpleNamespace(assessment_id='candidate',member_assessment_ids={'AB':'a','AC':'b'}, names=[],
        score_records=[pair_record('AB',ab,ab[1],[1],'a'),pair_record('AC',ac,ac[1],[2],'b')])
    first=config_for('AB',ab,ab[1],1,1); second=config_for('AC',ac,ac[1],1,1)
    first['query']['snapshot']=second['query']['snapshot']='member'
    spec={'observations':[first,second],'aggregation':{'reduction':'mean','missing':'error','count':'context'},
          'threshold':3,'comparison':'lt'}
    runtime={'score_selector_queries':{'two':spec},'score_snapshots':{'last_accepted':'other'}}
    with pytest.raises(ValueError, match='actual last accepted assessment'):
        construct_position_selector('score("two")',scores,protomers,runtime_config=runtime)
    runtime['score_snapshots']['last_accepted']='candidate'
    assert construct_position_selector('score("two")',scores,protomers,runtime_config=runtime)['A'].tolist() == [True]
    del scores.member_assessment_ids['AC']
    with pytest.raises(ValueError, match='no member identity'):
        construct_position_selector('score("two")',scores,protomers,runtime_config=runtime)
