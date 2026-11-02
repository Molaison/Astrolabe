from types import SimpleNamespace
import numpy as np
import pytest
from astrolabe.score_evidence import ScoreSpec, ScoreRecord, ScoreQuery, query_score
from astrolabe.score_contributions import ScoreContribution
from astrolabe.losses.base import LossEngine
from astrolabe.selectors.utils import construct_position_selector


def test_directed_pair_selector_and_shared_chain_mapping():
    # Direction matters: position -> target. Two physical copies map to A:0.
    values = np.array([[1., 3.], [7., 9.], [4., 6.]])
    record = ScoreRecord(ScoreSpec('pae', 'evaluator.af3', ('position', 'target')), 's1', values,
                         (('A0', 'B0', 'A1'), ('X0', 'X1')), input_snapshot='s1', subject='predicted_structure', relation='measured_on')
    values[:] = 99
    cfg = {'query': {'name': 'pae', 'source': 'evaluator.af3', 'axes': ['position'],
                     'select': {'target': ['X1']}},
           'mapping': {'A0': ('A', 0), 'B0': ('A', 0), 'A1': ('A', 1)},
           'position_weights': {'A0': 3, 'B0': 1, 'A1': 1},
           'threshold': 5, 'comparison': 'lt'}
    result = SimpleNamespace(score_records=[record], score_snapshots={'last_accepted': 's1'},
                             score_selector_queries={'directed': cfg})
    masks = construct_position_selector('score("directed")', result, {'A': SimpleNamespace(length=2)})
    assert masks['A'].tolist() == [True, False]
    assert record.value[0, 1] == 3
    with pytest.raises(ValueError):
        record.value[0, 0] = 8


def test_stale_duplicate_invalid_and_weighted_context():
    spec = ScoreSpec('x', 'mutator.m', ('context', 'residue'))
    rec = ScoreRecord(spec, 'old', [[1, 2], [5, 6]], (('c1', 'c2'), (0, 1)), input_snapshot='old', subject='predicted_structure', relation='measured_on')
    q = ScoreQuery('x', 'mutator.m', axes=('residue',), weights={'context': {'c1': 3, 'c2': 1}})
    np.testing.assert_allclose(query_score([rec], q, snapshots={'last_accepted':'old'}), [2, 3])
    with pytest.raises(ValueError): query_score([rec], q, snapshots={'last_accepted':'new'})
    with pytest.raises(ValueError): query_score([rec, rec], q, snapshots={'last_accepted':'old'})
    with pytest.raises(ValueError): query_score([rec], ScoreQuery('x','mutator.m',axes=('residue',)), snapshots={'last_accepted':'old'})
    bad = ScoreRecord(spec, 'old', [[1, np.nan], [2, 3]], (('c1', 'c2'), (0, 1)), input_snapshot='old', subject='predicted_structure', relation='measured_on')
    with pytest.raises(ValueError): query_score([bad], q, snapshots={'last_accepted':'old'})
    with pytest.raises(ValueError): query_score([rec], ScoreQuery('x','mutator.m',snapshot='current_proposal'), generating_selector=True)


def test_real_loss_engine_multi_producer_catalog():
    engine = LossEngine({'loss_terms': {}, 'score_catalog': [
        {'name':'q', 'source':'evaluator.a'}, {'name':'q', 'source':'predocker.b'}],
        'score_queries': {'a': {'name':'q','source':'evaluator.a'}, 'b': {'name':'q','source':'predocker.b'}}}, validate_config=False)
    engine.setup('a + b', None, None, validate_config=False)
    engine._extract_scores_with_registry = lambda _: ({}, {})
    engine._default_expression_symbols = lambda: {}
    scores = SimpleNamespace(score_contributions=[ScoreContribution(source='evaluator.a',metric_name='q',value=2,measurement_snapshot='s1',input_snapshot='s1',subject='predicted_structure',relation='measured_on')])
    runtime = {'score_contributions':[ScoreContribution(source='predocker.b',metric_name='q',value=3,flatten_as='legacy_b',measurement_snapshot='s1',input_snapshot='s1',subject='predicted_structure',relation='measured_on')],
               'score_capture_snapshot':'s1', 'score_snapshots':{'current_proposal':'s1'}}
    score, details = engine.compute_loss(scores, runtime)
    assert score == 5 and details['a'] == 2 and details['b'] == 3


def test_yaml_real_standard_scores_and_three_producers(tmp_path):
    from pathlib import Path
    from omegaconf import OmegaConf
    from astrolabe.evaluators.utils import StandardScores
    from astrolabe.core import global_config
    from astrolabe.selectors.utils import construct_position_selector
    config = OmegaConf.create('''
score_catalog:
  - {name: confidence, source: mutator.mpnn}
  - {name: energy, source: predocker.vina}
score_queries:
  model_value: {name: iptm, source: evaluator, entity: AB, snapshot: current_proposal}
  mutation_value: {name: confidence, source: mutator.mpnn, snapshot: last_accepted, subject: proposal, relation: proposal_on_reference}
  docking_value: {name: energy, source: predocker.vina, snapshot: last_accepted, subject: reference_structure, relation: measured_on}
score_selector_queries:
  directed:
    query:
      name: pae
      source: evaluator
      entity: AB
      axes: [position]
      select:
        target_position: ["AB:B:1:B"]
    mapping: auto
    threshold: 5
    comparison: lt
''')
    protomers = {'A': SimpleNamespace(length=2), 'B': SimpleNamespace(length=1)}
    global_config.reset()
    global_config.init_const('protomers', protomers)
    scores = StandardScores(names=['AB'], plddt={'AB':[0.8]}, iptm={'AB':[0.7]},
                            pae={'AB':[np.array([[0, 2, 3], [2, 0, 7], [9, 9, 0]])]},
                            complex_paths={'AB':Path('ab.cif')})
    reference = StandardScores(names=['AB'], plddt={'AB':[0.8]}, complex_paths={'AB':Path('reference.cif')})
    a, b = reference.assessment_id, scores.assessment_id
    contributions = [ScoreContribution('mutator.mpnn','confidence',0.2, measurement_snapshot=a, input_snapshot=a, subject='proposal', relation='proposal_on_reference'),
                     ScoreContribution('predocker.vina','energy',0.3, measurement_snapshot=a, input_snapshot=a, subject='reference_structure', relation='measured_on')]
    engine = LossEngine(config, validate_config=False)
    engine.setup('model_value + mutation_value + docking_value', None, None, validate_config=False)
    current = {'score_contributions': contributions, 'score_capture_snapshot':b,
               'score_snapshots': {'current_proposal':b, 'last_accepted':a}}
    value, _ = engine.compute_loss(scores, current)
    assert value == pytest.approx(1.2)
    accepted = {'score_selector_queries': config.score_selector_queries, 'score_catalog':config.score_catalog,
                'score_capture_snapshot':b, 'score_snapshots':{'last_accepted':b}}
    masks = construct_position_selector('score("directed")', scores, protomers, runtime_config=accepted)
    assert masks['A'].tolist() == [True, False]
    assert masks['B'].tolist() == [True]
    accepted['score_snapshots']['last_accepted'] = 'other'
    with pytest.raises(ValueError):
        construct_position_selector('score("directed")', scores, protomers, runtime_config=accepted)
    global_config.reset()


@pytest.mark.parametrize('source', ['mutator.mpnn', 'predocker.vina'])
def test_yaml_selector_consumes_runtime_producer_evidence(source):
    from omegaconf import OmegaConf
    scores = SimpleNamespace(names=[], score_contributions=[])
    subject, relation = ('proposal','proposal_on_reference') if source.startswith('mutator') else ('reference_structure','measured_on')
    cfg = OmegaConf.create({'score_selector_queries': {'p': {
        'query': {'name':'per_position','source':source,'axes':['position'],'subject':subject,'relation':relation},
        'mapping': {'p0':['A',0], 'p1':['A',1]}, 'threshold':0.5}},
        'score_catalog':[{'name':'per_position','source':source,
                          'axes':['position'], 'coordinates':[['p0','p1']]}]})
    runtime = {'score_selector_queries':cfg.score_selector_queries, 'score_catalog':cfg.score_catalog,
               'score_contributions':[ScoreContribution(source,'per_position',[0.3,0.8],measurement_snapshot='s0',input_snapshot='s0',subject=subject,relation=relation)],
               'score_capture_snapshot':'s0', 'score_snapshots':{'last_accepted':'s0'}}
    masks = construct_position_selector('score("p")', scores, {'A':SimpleNamespace(length=2)}, runtime_config=runtime)
    assert masks['A'].tolist() == [False, True]


def test_cross_chain_reduction_needs_explicit_physical_weights():
    record = ScoreRecord(ScoreSpec('pae','evaluator',('position',)), 's', [1, 9],
                         (('AB:A:1:A','AB:B:1:B'),))
    query = ScoreQuery('pae', 'evaluator')
    with pytest.raises(ValueError, match='cross-chain'):
        query_score([record], query, snapshots={'last_accepted':'s'})
    weighted = ScoreQuery('pae','evaluator', weights={'position':{'AB:A:1:A':3, 'AB:B:1:B':1}})
    assert query_score([record], weighted, snapshots={'last_accepted':'s'}) == 3


def test_non_evaluator_record_requires_explicit_subject_identity():
    with pytest.raises(ValueError, match='explicit input_snapshot'):
        ScoreRecord(ScoreSpec('probability','mutator.mpnn'), 's1', .5)
    with pytest.raises(ValueError, match='actual input assessment'):
        ScoreRecord(ScoreSpec('dock','predocker.vina'), 's2', .5,
                    input_snapshot='s1', subject='predicted_structure', relation='measured_on')
