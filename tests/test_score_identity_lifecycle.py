from pathlib import Path
from types import SimpleNamespace
import csv
import subprocess
import sys
import numpy as np
import pytest
import yaml

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import ProtomerTypes
from astrolabe.decision_context import passes_stage_entry_acceptance_gate
from astrolabe.decision_maker import Decisions
from astrolabe.evaluators.utils import StandardScores
from astrolabe.losses.base import LossEngine
from astrolabe.mutators.utils import prepare_exclude_sites


def scores(value, *, name='A', pae=None):
    return StandardScores(names=[name], plddt={name:[value]}, complex_paths={name:Path('x.pdb')},
                          pae=None if pae is None else {name:[pae]})


def init_decisions(tmp_path):
    global_config.reset()
    proto = SimpleNamespace(type=SimpleNamespace(value='p'), sequence='AAA', mutated_sequence='AAA', length=3)
    global_config.init_const('protomers', {'A':proto})
    global_config.init_const('stage_manager', SimpleNamespace(current_stage='default'))
    global_config.init_const('out_dir', tmp_path)
    global_config.init_const('initial_temperature', 1.0)
    global_config.init_const('use_metropolis_acceptance', False)
    calls = []
    global_config.init_const('segment_manager', SimpleNamespace(apply_mutations=lambda: calls.append('apply'), clear_mutations=lambda: calls.append('clear')))
    engine = LossEngine({'loss_terms':{}, 'score_queries':{
        'current':{'name':'plddt','source':'evaluator','entity':'A'},
        'accepted':{'name':'plddt','source':'evaluator','entity':'A','snapshot':'last_accepted'}}})
    engine.setup('current', None, None, acceptance_gate='current >= 0.3', validate_config=False)
    initial = scores(.6)
    decisions = Decisions.create(initial, {}, engine.compute_loss)
    global_config.init_const('decisions', decisions)
    return decisions, engine, initial, calls


def test_initial_accept_reject_and_rescore_follow_decision_owner(tmp_path):
    decisions, engine, initial, calls = init_decisions(tmp_path)
    assert decisions.scores_detail[0]['current'] == pytest.approx(.6)
    assert decisions.scores_detail[0]['accepted'] == pytest.approx(.6)
    proposal = scores(.8)
    assert proposal.assessment_id != initial.assessment_id
    _, returned, accepted = decisions.make_mutation_decision(1, proposal, engine, {}, force_accept=True)
    assert accepted and returned is proposal and calls == ['apply']
    assert decisions.scores_detail[1]['current'] == pytest.approx(.8)
    assert decisions.scores_detail[1]['accepted'] == pytest.approx(.6)
    rejected = scores(.2)
    _, returned, accepted = decisions.make_mutation_decision(2, rejected, engine, {})
    assert not accepted and returned is proposal and calls == ['apply','clear']
    assert decisions.last_accepted == 1
    assert decisions.scores_detail[2]['current'] == pytest.approx(.2)
    assert decisions.scores_detail[2]['accepted'] == pytest.approx(.8)
    rescored = scores(.75)
    decisions.recompute_last_score(engine, {}, rescored)
    assert decisions.evaluation_results[decisions.last_accepted] is rescored
    assert decisions.scores_detail[1]['accepted'] == pytest.approx(.75)
    assert rescored.clone().assessment_id == rescored.assessment_id
    assert rescored.filter_by_oligomer(['A']).assessment_id == rescored.assessment_id
    global_config.reset()


def test_direct_stage_rejection_preserves_already_accepted_mutation(tmp_path):
    from astrolabe.core.protomer_config_parser import Protomer, SegmentManager
    from astrolabe.stage_transaction import (plan_stage_transaction, begin_direct_stage_entry,
        finish_direct_stage_entry_decision, resolve_already_accepted_direct_stage_entry_decision)
    from test_stage_transaction import _StageManager
    global_config.reset()
    for key, value in [('segments', {}), ('named_files', {}), ('predefined_statement', {}),
                       ('exclude_AA', []), ('segment_manager', SegmentManager()),
                       ('out_dir', tmp_path), ('initial_temperature', 1.0),
                       ('use_metropolis_acceptance', False)]:
        global_config.init_const(key, value)
    proto = Protomer('A', 'AAA', 'protein')
    global_config.init_const('protomers', {'A':proto})
    manager = _StageManager()
    global_config.init_const('stage_manager', manager)
    engine = LossEngine({'loss_terms':{}, 'score_queries':{
        'current':{'name':'plddt','source':'evaluator','entity':'A'},
        'accepted':{'name':'plddt','source':'evaluator','entity':'A','snapshot':'last_accepted'}}})
    engine.setup('current', None, None, acceptance_gate='current >= 0.3', validate_config=False)
    decisions = Decisions.create(scores(.6), {}, engine.compute_loss)
    global_config.init_const('decisions', decisions)
    global_config.segments['A_1'].add_mutations((0, 'A', 'C', 0.0, 0.0))
    accepted_result = scores(.8)
    _, _, accepted = decisions.make_mutation_decision(1, accepted_result, engine, {}, force_accept=True)
    assert accepted and proto.sequence == 'CAA'
    plan = plan_stage_transaction(manager, 'stage_1')
    target = scores(.1)
    attempt = begin_direct_stage_entry(manager, plan, previous_stage_result=accepted_result,
                                       evaluate_target=lambda: target, clone_result=lambda x:x)
    passed = passes_stage_entry_acceptance_gate(eval_result=target, loss_computer=engine, runtime_config={})
    decision = finish_direct_stage_entry_decision(manager, attempt, previous_stage_result=accepted_result,
                                                   evaluation_result=target, is_accepted=passed)
    result = resolve_already_accepted_direct_stage_entry_decision(decision, recompute_last_score=lambda r: decisions.recompute_last_score(engine, {}, r))
    assert not passed and result is accepted_result
    assert decisions.last_accepted == 1 and decisions.evaluation_results[1] is accepted_result
    assert proto.sequence == 'CAA' and manager.current_stage == 'default'
    global_config.reset()


def test_pair_selector_uses_actual_input_identity_and_explicit_reference():
    global_config.reset()
    class Proto:
        type = ProtomerTypes.PROTEIN
        def __init__(self, length): self.length=length; self.mutable_mask=np.ones(length, dtype=bool)
        def aggregate_bool(self, value): return value
    protomers = {'A':Proto(2), 'B':Proto(1)}
    global_config.init_const('protomers', protomers)
    accepted = scores(.8, name='AB', pae=np.array([[0,2,3],[2,0,7],[9,9,0]]))
    proposal = scores(.8, name='AB', pae=np.array([[0,2,8],[2,0,1],[9,9,0]]))
    owner = SimpleNamespace(last_accepted=0, evaluation_results={0:accepted})
    global_config.init_const('decisions', owner)
    manager = SimpleNamespace(current_stage='default', transition_consistency_gate_context=None, current_transition_input_scores=None)
    global_config.init_const('stage_manager', manager)
    cfg = yaml.safe_load('''
score_selector_queries:
  directed:
    query:
      name: pae
      source: evaluator
      entity: AB
      axes: [position]
      select: {target_position: ["AB:B:1:B"]}
    mapping: auto
    threshold: 5
    comparison: lt
''')
    result = prepare_exclude_sites('score("directed")', accepted, protomers, runtime_config=cfg)
    assert result['A'] == [1]
    with pytest.raises(ValueError, match='expected one current score record'):
        prepare_exclude_sites('score("directed")', proposal, protomers, runtime_config=cfg)
    manager.transition_consistency_gate_context = object()
    manager.current_transition_input_scores = proposal
    cfg['score_selector_queries']['directed']['query']['snapshot'] = 'reference'
    result = prepare_exclude_sites('score("directed")', proposal, protomers, runtime_config=cfg)
    assert result['A'] == [0]
    global_config.reset()


def test_typed_score_real_cli(tmp_path):
    config = {
        'task_name':'typed_score_cli', 'out_dir':str(tmp_path/'output'), 'raw_model_results_dir':str(tmp_path/'raw'),
        'repeats':1, 'seed':17, 'steps':2, 'ananas_path':None, 'seqs':{'A':'AAAA'},
        'mutator':{'MCMC':{'mutation_method':'random','select_positions':'random'}, 'mutator_terms':{'MCMC':{'_type':'MCMC'}}},
        'evaluator':{'evaluator_terms':{'Dummy':{'_type':'dummy'}}},
        'loss':{'loss_terms':{}, 'score_queries':{'model_value':{'name':'plddt','source':'evaluator','entity':'A','snapshot':'current_proposal'}}},
        'stages':{'default':{'oligo':['A'],'evaluator':'Dummy','mutation_num':1,'mutators':['MCMC'], 'mutator_weights':[1.0], 'weighted_score_expression':'model_value','early_stopping':None}},
    }
    path = tmp_path/'typed_score.yaml'
    path.write_text(yaml.safe_dump(config))
    result = subprocess.run([sys.executable,'-m','astrolabe','run','--standalone',str(path)], text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout+'\n'+result.stderr
    assert 'Pipeline completed successfully' in result.stdout + result.stderr
    paths = list((tmp_path/'output'/'typed_score_cli').glob('*/scores.csv'))
    assert len(paths) == 1
    with paths[0].open() as handle: rows = list(csv.DictReader(handle))
    assert [int(row['step']) for row in rows] == [0,1]
    assert all(float(row['model_value']) == pytest.approx(.9) for row in rows)


def test_runtime_contribution_is_frozen_with_accepted_assessment(tmp_path):
    from astrolabe.score_contributions import ScoreContribution
    decisions, engine, initial, _ = init_decisions(tmp_path)
    engine.score_catalog = [{'name':'energy', 'source':'predocker.vina',
                             'axes':['position'], 'coordinates':[['p0','p1']]}]
    value = [.2, .8]
    accepted = scores(.8)
    runtime = {'score_contributions':[ScoreContribution('predocker.vina','energy',value,aggregation='mean', measurement_snapshot=accepted.assessment_id, input_snapshot=accepted.assessment_id, subject='predicted_structure', relation='measured_on')]}
    decisions.make_mutation_decision(1, accepted, engine, runtime, force_accept=True)
    assert accepted.assessment_id in decisions.score_records_by_assessment
    value[0] = 100
    engine.score_queries['accepted_energy'] = {'name':'energy','source':'predocker.vina','snapshot':'last_accepted'}
    rejected = scores(.1)
    decisions.make_mutation_decision(2, rejected, engine, {})
    assert decisions.scores_detail[2]['accepted_energy'] == pytest.approx(.5)
    assert decisions.evaluation_results[decisions.last_accepted] is accepted
    global_config.reset()


def test_decision_query_can_explicitly_use_transition_reference(tmp_path):
    decisions, engine, initial, _ = init_decisions(tmp_path)
    reference = scores(.45)
    global_config.stage_manager.transition_consistency_gate_context = SimpleNamespace(reference_scores=reference)
    engine.score_queries['gate_reference'] = {'name':'plddt','source':'evaluator','entity':'A','snapshot':'reference'}
    proposal = scores(.8)
    decisions.make_mutation_decision(1, proposal, engine, {}, force_accept=True)
    assert decisions.scores_detail[1]['gate_reference'] == pytest.approx(.45)
    assert decisions.scores_detail[1]['accepted'] == pytest.approx(.6)
    global_config.reset()


def test_reference_producer_is_never_retagged_as_new_prediction(tmp_path):
    from astrolabe.score_contributions import ScoreContribution, score_contributions_to_snapshot
    decisions, engine, a, _ = init_decisions(tmp_path)
    engine.score_catalog = [{'name':'likelihood','source':'mutator.mpnn'},
                            {'name':'dock','source':'predocker.vina'}]
    measured = [
        ScoreContribution('mutator.mpnn','likelihood',.7,measurement_snapshot=a.assessment_id,
                          input_snapshot=a.assessment_id,subject='proposal',relation='proposal_on_reference'),
        ScoreContribution('predocker.vina','dock',.4,measurement_snapshot=a.assessment_id,
                          input_snapshot=a.assessment_id,subject='reference_structure',relation='measured_on'),
    ]
    runtime = {'score_contributions':measured}
    b = scores(.8)
    engine.score_queries['bad'] = {'name':'likelihood','source':'mutator.mpnn'}
    with pytest.raises(ValueError, match='expected one current score record'):
        decisions.make_mutation_decision(1, b, engine, runtime, force_accept=True)
    engine.score_queries.pop('bad')
    engine.score_queries['proposal_on_reference'] = {'name':'likelihood','source':'mutator.mpnn',
        'snapshot':'last_accepted','subject':'proposal','relation':'proposal_on_reference'}
    engine.score_queries['reference_dock'] = {'name':'dock','source':'predocker.vina',
        'snapshot':'last_accepted','subject':'reference_structure','relation':'measured_on'}
    _, _, accepted = decisions.make_mutation_decision(1, b, engine, runtime, force_accept=True)
    assert accepted
    assert decisions.scores_detail[1]['proposal_on_reference'] == pytest.approx(.7)
    assert decisions.scores_detail[1]['reference_dock'] == pytest.approx(.4)
    assert a.assessment_id in decisions.score_records_by_assessment
    assert b.assessment_id not in decisions.score_records_by_assessment
    c = scores(.9)
    engine.score_queries = {'stale':{'name':'likelihood','source':'mutator.mpnn'}}
    with pytest.raises(ValueError, match='expected one current score record'):
        decisions.make_mutation_decision(2, c, engine, runtime, force_accept=True)
    assert score_contributions_to_snapshot([ScoreContribution('mutator.mpnn','legacy',.6)]) == {'legacy':.6}
    global_config.reset()


def test_strict_unlabelled_non_evaluator_contribution_fails_closed(tmp_path):
    from astrolabe.score_contributions import ScoreContribution
    decisions, engine, _, _ = init_decisions(tmp_path)
    engine.score_catalog = [{'name':'entropy','source':'mutator.mpnn'}]
    engine.score_queries['entropy_strict'] = {'name':'entropy','source':'mutator.mpnn'}
    runtime = {'score_contributions':[ScoreContribution('mutator.mpnn','entropy',.2)]}
    with pytest.raises(ValueError, match='requires measurement_snapshot'):
        decisions.make_mutation_decision(1, scores(.8), engine, runtime)
    global_config.reset()


@pytest.mark.parametrize("execution", [False, True])
def test_canonical_protomer_selector_real_cli(tmp_path, execution):
    config = {
        'task_name':'position_score_cli', 'out_dir':str(tmp_path/'output'), 'raw_model_results_dir':str(tmp_path/'raw'),
        'repeats':1, 'seed':17, 'steps':3, 'ananas_path':None, 'seqs':{'A':'AAAA'},
        'mutator':{'MCMC':{'mutation_method':'random','select_positions':'random'},
                   'mutator_terms':{'MCMC':{'_type':'MCMC','position_selector':'score("positions")'}}},
        'evaluator':{'evaluator_terms':{'Dummy':{'_type':'dummy'}}},
        'loss':{'loss_terms':{}, 'score_queries':{'model_value':{'name':'plddt','source':'evaluator','entity':'A'}},
                'score_selector_queries':{'positions':{'query':{'name':'per_proto_score','source':'evaluator',
                   'entity':'A','axes':['design_position']}, 'mapping':'auto','threshold':.5,'comparison':'ge'}}},
        'stages':{'default':{'oligo':['A'],'evaluator':'Dummy','mutation_num':1,'mutators':['MCMC'],
                             'mutator_weights':[1.0], 'weighted_score_expression':'model_value','early_stopping':None,
                             'acceptance_gate':'step < 2', 'acceptance_threshold':'model_value > 0.8'}},
    }
    config['execution'] = {'enabled': execution, 'max_active': 'auto', 'max_resident_models': 'auto'}
    path = tmp_path/'position_score.yaml'
    path.write_text(yaml.safe_dump(config))
    result = subprocess.run([sys.executable,'-m','astrolabe','run','--standalone',str(path)], text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout+'\n'+result.stderr
    csv_path = next((tmp_path/'output'/'position_score_cli').glob('*/scores.csv'))
    with csv_path.open() as handle: rows = list(csv.DictReader(handle))
    assert [int(row['step']) for row in rows] == [0,1,2]
    assert [row['accepted'].lower() for row in rows] == ['true', 'true', 'false']
    assert float(rows[1]['mutator_selector_MCMC_pA_selected_count']) == 4


def test_multicontext_loss_to_accepted_selector_real_cli(tmp_path):
    def observation(context, chain):
        return {'context': context, 'weight': 1.0,
                'query': {'name': 'quality', 'source': 'loss', 'entity': context,
                          'entity_scope': 'oligo'},
                'broadcast': True,
                'mapping': {f'{context}:A:{i}:A': ['A', i - 1] for i in range(1, 5)}}
    config = {
        'task_name': 'loss_selector_cli', 'out_dir': str(tmp_path/'out'),
        'raw_model_results_dir': str(tmp_path/'raw'), 'repeats': 1, 'seed': 17,
        'steps': 3, 'ananas_path': None, 'seqs': {'A': 'AAAA', 'B': 'CCCC', 'C': 'DDDD'},
        'mutator': {'MCMC': {'mutation_method': 'random', 'select_positions': 'random'},
                    'mutator_terms': {'MCMC': {'_type': 'MCMC', 'position_selector': 'score("from_loss")'}}},
        'evaluator': {'evaluator_terms': {'Dummy': {'_type': 'dummy'}}},
        'loss': {'loss_terms': {'quality': {'_type': 'plddt'}},
                 'score_catalog': [{'source': 'loss', 'name': 'quality', 'entity': c, 'entity_scope': 'oligo'} for c in ['AB','AC']],
                 'score_queries': {'typed_quality': {'source': 'loss', 'name': 'quality', 'entity': 'AB'}},
                 'score_selector_queries': {'from_loss': {
                     'observations': [observation('AB', 'B'), observation('AC', 'C')],
                     'aggregation': {'reduction': 'mean', 'missing': 'error', 'count': 'context'},
                     'threshold': .5, 'comparison': 'ge'}}},
        'stages': {'default': {'oligo': ['AB','AC'], 'evaluator': 'Dummy', 'mutation_num': 1,
                               'mutators': ['MCMC'], 'mutator_weights': [1.0],
                               'weighted_score_expression': 'typed_quality', 'early_stopping': None,
                               'acceptance_gate': 'step < 2', 'acceptance_threshold': 'typed_quality > 0.8'}}}
    path = tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(config))
    result = subprocess.run([sys.executable, '-m', 'astrolabe', 'run', '--standalone', str(path)],
                            text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + '\n' + result.stderr
    csv_path = next((tmp_path/'out'/'loss_selector_cli').glob('*/scores.csv'))
    with csv_path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row['step']) for row in rows] == [0, 1, 2]
    assert [row['accepted'].lower() for row in rows] == ['true', 'true', 'false']
    assert all(float(row['typed_quality']) == pytest.approx(.9) for row in rows)
    assert all(float(row['mutator_selector_MCMC_pA_selected_count']) == 4 for row in rows[1:])


def test_loss_evidence_tracks_accepted_identity_across_rejection(tmp_path):
    decisions, _, initial, _ = init_decisions(tmp_path)
    engine = LossEngine({'loss_terms': {'quality': {'_type': 'plddt'}},
                         'score_catalog': [{'source': 'loss', 'name': 'quality', 'entity': 'A', 'entity_scope': 'oligo'}]},
                        validate_config=False)
    engine.setup('quality', None, None, acceptance_gate='quality >= 0.3', validate_config=False)
    decisions.recompute_last_score(engine, {}, initial)
    selector = {'score_catalog': engine.score_catalog,
                'score_selector_queries': {'quality': {'query': {'source': 'loss', 'name': 'quality', 'entity': 'A'},
                   'broadcast': True, 'mapping': {'A:A:1:A': ['A', 0]}, 'threshold': .7, 'comparison': 'ge'}}}
    protomers = global_config.protomers
    from astrolabe.selectors.utils import construct_position_selector
    from astrolabe.score_identity import score_runtime_view
    def selected(result):
        view = score_runtime_view(selector, result, decisions=decisions, selector=True)
        return construct_position_selector('score("quality")', result, protomers, runtime_config=view)['A'].tolist()
    assert selected(initial) == [False, False, False]
    accepted = scores(.8)
    assert decisions.make_mutation_decision(1, accepted, engine, {}, force_accept=True)[2]
    assert selected(accepted) == [True, False, False]
    rejected = scores(.2)
    assert not decisions.make_mutation_decision(2, rejected, engine, {})[2]
    assert selected(accepted) == [True, False, False]
    assert {initial.assessment_id, accepted.assessment_id, rejected.assessment_id} <= set(decisions.score_records_by_assessment)
    with pytest.raises(ValueError, match='expected one current score record'):
        selected(rejected)
    global_config.reset()
