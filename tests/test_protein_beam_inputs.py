import json
from types import SimpleNamespace
import numpy as np
import pytest
from astrolabe.protein_beam_inputs import preserve_fixed_target_homology


def test_fixed_target_retains_insertions_headers_gaps_and_templates(tmp_path):
    raw=dict(id='B',sequence='ACD',unpairedMsa='>query full header\nACD\n>hit species\nAqCD\n>gap\nA--\n',
             pairedMsa='>query\nACD\n',templates=[dict(mmcif='template bytes',queryIndices=[0,2],templateIndices=[4,8])])
    p=tmp_path/'input.json';p.write_text(json.dumps(dict(sequences=[dict(protein=raw)])))
    target=SimpleNamespace(sequence='ACD',length=3,mutable_mask=np.zeros(3,dtype=bool))
    cfg=SimpleNamespace(homologies={'B':str(p)+':B'})
    preserve_fixed_target_homology(cfg,target)
    assert target._homology['unpairedMsa']==raw['unpairedMsa']
    assert target._homology['pairedMsa']==raw['pairedMsa']
    assert target._homology['templates'][0].query_to_template_map=={0:4,2:8}
    assert target._homology_seq=='ACD'


def test_target_sequence_or_mutability_mismatch_fails(tmp_path):
    p=tmp_path/'input.json';p.write_text(json.dumps(dict(sequences=[dict(protein=dict(id='B',sequence='ACD'))])))
    cfg=SimpleNamespace(homologies={'B':str(p)+':B'})
    target=SimpleNamespace(sequence='AAA',length=3,mutable_mask=np.zeros(3,dtype=bool))
    with pytest.raises(ValueError,match='sequence differs'):preserve_fixed_target_homology(cfg,target)
    target.mutable_mask[0]=True
    with pytest.raises(ValueError,match='fully fixed'):preserve_fixed_target_homology(cfg,target)


def test_unmapped_target_ptms_cannot_be_silently_omitted():
    target=SimpleNamespace(sequence='ACD',length=3,mutable_mask=np.zeros(3,dtype=bool))
    cfg=SimpleNamespace(homologies={'B':dict(modifications=[dict(ptmType='SEP',ptmPosition=1)])})
    with pytest.raises(ValueError,match='PTMs'):preserve_fixed_target_homology(cfg,target)
