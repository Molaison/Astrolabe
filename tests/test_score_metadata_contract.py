from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest

from astrolabe.evaluators.utils import StandardScores
from astrolabe.score_evidence import ScoreRecord, ScoreSpec
from astrolabe.score_merge import merge_standard_scores


def one(name, value, *, candidate_key=None, path=None):
    return StandardScores(names=[name], plddt={name:[np.asarray(value, dtype=float)]},
                          complex_paths={name:Path(path or f'{name}.pdb')}, candidate_key=candidate_key)


def test_clone_filter_and_stage_replace_preserve_scoped_metadata_without_copying_arrays():
    ab = one('AB', [.3,.7], candidate_key=('seed',3,0))
    ac = one('AC', [.4,.8], candidate_key=('seed',3,0))
    data = np.arange(8.).reshape(2,2,2)
    record = ScoreRecord(ScoreSpec('directed_pae','evaluator',('context','position','target_position')),
        'assessment', data, (('AB','AC'), ('AB:A:1:A','AC:A:1:A'), ('AB:B:1:B','AC:B:1:C')))
    source = StandardScores(names=['AB','AC'], plddt={'AB':[np.array([.8,.9])], 'AC':[np.array([.6,.7])]},
        complex_paths={'AB':Path('ab.pdb'),'AC':Path('ac.pdb')}, candidate_scores=[ab,ac],
        pre_eval_hook_reports={'AB':[{'path':'ab_input.json'}], 'AC':[{'path':'ac_input.json'}]},
        post_eval_hook_reports={'AB':{'path':'ab_relaxed.pdb'}, 'AC':{'path':'ac_relaxed.pdb'}},
        member_assessment_ids={'AB':'member-ab','AC':'member-ac'}, score_records=[record])
    clone = source.clone()
    snap = replace(source, complex_paths={'AB':Path('snap_ab.pdb'),'AC':Path('snap_ac.pdb')})
    assert clone.assessment_id == snap.assessment_id == source.assessment_id
    assert clone.candidate_scores[0].assessment_id == source.candidate_scores[0].assessment_id
    clone.complex_paths['AB'] = Path('clone.pdb')
    clone.candidate_scores[0].complex_paths['AB'] = Path('clone_candidate.pdb')
    clone.post_eval_hook_reports['AB']['path'] = 'clone_relaxed.pdb'
    snap.pre_eval_hook_reports['AB'][0]['path'] = 'snapshot_input.json'
    assert source.complex_paths['AB'] == Path('ab.pdb')
    assert source.candidate_scores[0].complex_paths['AB'] == Path('AB.pdb')
    assert source.post_eval_hook_reports['AB']['path'] == 'ab_relaxed.pdb'
    assert source.pre_eval_hook_reports['AB'][0]['path'] == 'ab_input.json'
    assert np.shares_memory(clone.plddt['AB'][0], source.plddt['AB'][0])
    clone.plddt['AB'][0][0] = .123
    assert source.plddt['AB'][0][0] == .123  # explicitly not a deep immutable snapshot
    with pytest.raises(ValueError): clone.score_records[0].value[0,0,0] = 9
    filtered = source.filter_by_oligomer(['AB'])
    assert filtered.assessment_id == source.assessment_id
    assert [candidate.names for candidate in filtered.candidate_scores] == [['AB']]
    assert filtered.candidate_scores[0].assessment_id == source.candidate_scores[0].assessment_id
    assert set(filtered.pre_eval_hook_reports) == set(filtered.post_eval_hook_reports) == {'AB'}
    assert filtered.member_assessment_ids == {'AB':'member-ab'}
    selected = filtered.score_records[0]
    assert selected.spec.axes == record.spec.axes
    assert selected.coordinates == (('AB',),('AB:A:1:A',),('AB:B:1:B',))
    np.testing.assert_array_equal(selected.value, data[:1,:1,:1])


def test_merge_uses_candidate_keys_and_preserves_member_evidence():
    a2, a1 = one('AB',[.2],candidate_key=('seed',2,0)), one('AB',[.1],candidate_key=('seed',1,0))
    b1, b2 = one('AC',[.7],candidate_key=('seed',1,0)), one('AC',[.8],candidate_key=('seed',2,0))
    a = one('AB',[.5]); b = one('AC',[.6])
    a.candidate_scores, b.candidate_scores = [a2,a1], [b1,b2]
    a.post_eval_hook_reports={'AB':{'path':'ab_relaxed.pdb'}}
    b.post_eval_hook_reports={'AC':{'path':'ac_relaxed.pdb'}}
    a.score_records=(ScoreRecord(ScoreSpec('x','evaluator',entity='AB'),a.assessment_id,.5),)
    merged = merge_standard_scores([a,b])
    assert merged.assessment_id not in {a.assessment_id,b.assessment_id}
    assert merged.member_assessment_ids == {'AB':a.assessment_id,'AC':b.assessment_id}
    assert [x.candidate_key for x in merged.candidate_scores] == [('seed',2,0),('seed',1,0)]
    assert merged.candidate_scores[0].member_assessment_ids == {'AB':a2.assessment_id,'AC':b2.assessment_id}
    assert merged.candidate_scores[0].assessment_id not in {a2.assessment_id,b2.assessment_id}
    assert merged.score_records[0].snapshot == a.assessment_id
    assert set(merged.post_eval_hook_reports) == {'AB','AC'}
    a.candidate_scores[0].complex_paths['AB'] = Path('later.pdb')
    assert merged.candidate_scores[0].complex_paths['AB'] != Path('later.pdb')


def test_merge_never_pairs_candidates_by_list_position():
    a, b = one('AB',[.5]), one('AC',[.6])
    a.candidate_scores=[one('AB',[.2])]
    b.candidate_scores=[one('AC',[.2])]
    with pytest.raises(ValueError, match='candidate_key'):
        merge_standard_scores([a,b])
    a.candidate_scores=[one('AB',[.2],path='jobAB_candidate_seed42_sample0_model.cif')]
    b.candidate_scores=[one('AC',[.3],path='jobAC_candidate_seed42_sample0_model.cif')]
    merged = merge_standard_scores([a,b])
    assert merged.candidate_scores[0].candidate_key == ('af3',42,0)


def test_filter_requires_explicit_scope_for_ambiguous_monomer_entity():
    score = StandardScores(names=['A','AB'], plddt={'A':[.8],'AB':[.7]},
                           complex_paths={'A':Path('a.pdb'),'AB':Path('ab.pdb')})
    score.score_records = (ScoreRecord(ScoreSpec('x','evaluator',entity='A'), 's', .5),)
    with pytest.raises(ValueError, match='ambiguous score record entity scope'):
        score.filter_by_oligomer(['AB'])
    score.score_records = (ScoreRecord(ScoreSpec('x','evaluator',entity='A',entity_scope='oligo'), 's', .5),
                           ScoreRecord(ScoreSpec('x','evaluator',entity='A',entity_scope='protomer'), 's', .7))
    filtered = score.filter_by_oligomer(['AB'])
    assert len(filtered.score_records) == 1 and filtered.score_records[0].spec.entity_scope == 'protomer'


def test_candidate_key_is_an_immutable_identity_not_oligo_data():
    score = StandardScores(names=['AB','AC'],plddt={'AB':[.8],'AC':[.8]},
                           complex_paths={'AB':Path('a'),'AC':Path('b')}, candidate_key=('seed',1,0))
    assert score.filter_by_oligomer(['AB']).candidate_key == ('seed',1,0)
    with pytest.raises(ValueError, match='immutable and hashable'):
        StandardScores(names=['AB'],plddt={'AB':[.8]},complex_paths={'AB':Path('a')},candidate_key=['seed',1])
