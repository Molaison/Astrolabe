import json
from pathlib import Path
import numpy as np
import pytest
from astrolabe.mutators.utils import _condition_non_wt_probabilities
from astrolabe.protein_beam_geometry import pose_geometry


def test_sampling_condition_preserves_temperature_distribution_and_zero_mask():
    p=np.asarray([[[.75,0.,.25],[.5,.1,.4]]])
    log=np.log(np.asarray([[[.1,.8,.1],[.1,.8,.1]]]))
    result=_condition_non_wt_probabilities(p,log,'AA',['A','C','D'],protomer_name='A',distribution='sampling_probs')
    np.testing.assert_allclose(result,[[[0.,0.,1.],[0.,.2,.8]]])
    legacy=_condition_non_wt_probabilities(p,log,'AA',['A','C','D'],protomer_name='A')
    assert legacy[0,0,1]>0


def test_sampling_condition_rejects_no_legal_alternative():
    with pytest.raises(ValueError,match='No legal non-WT'):
        _condition_non_wt_probabilities(np.asarray([[[1.,0.]]]),np.zeros((1,1,2)),'A',['A','C'],protomer_name='A',distribution='sampling_probs')


def test_sampling_condition_leaves_inactive_positions_zero():
    r=_condition_non_wt_probabilities(np.zeros((1,1,2)),np.zeros((1,1,2)),'A',['A','C'],protomer_name='A',distribution='sampling_probs')
    assert not r.any()


def test_root_context_gate_stops_before_any_proposal(monkeypatch,tmp_path):
    from concurrent.futures import Future
    from omegaconf import open_dict
    import astrolabe.protein_beam as beam
    import astrolabe.protein_beam_root_parity as parity
    cfg=beam.load_config('/case/config/001_binder_interface/01_astrolabe_beam/02_implement_shared_beam/02_fixture.yaml')
    with open_dict(cfg.protein_beam):
        cfg.protein_beam.root_parity_reference='/case/runs/001_binder_interface/01_astrolabe_beam/04_calibrate_search/search/D5056_D15/candidates/000000/result.json'
    calls=[]
    class Pool:
        def __init__(self,**kw):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def submit(self,func,t):
            calls.append(t['kind']);f=Future()
            f.set_result(dict(t,status='done',score=0.,metrics={m:.2 for m in beam.METRICS},snapshot='unused',structure_path='unused'))
            return f
    monkeypatch.setattr(beam,'ProcessPoolExecutor',Pool)
    monkeypatch.setattr(parity,'root_parity',lambda *a:dict(passed=False,different=['all_atom_coordinates']))
    with pytest.raises(ValueError,match='before proposal'):
        beam.run_beam(cfg,tmp_path/'native','unused')
    assert calls==['evaluate']
    assert not json.loads((tmp_path/'native/root_parity.json').read_text())['passed']


def test_completed_resume_rechecks_root_parity_without_replaying(monkeypatch,tmp_path):
    from concurrent.futures import Future
    from omegaconf import open_dict
    import astrolabe.protein_beam as beam
    import astrolabe.protein_beam_root_parity as parity
    cfg=beam.load_config('/case/config/001_binder_interface/01_astrolabe_beam/02_implement_shared_beam/02_fixture.yaml')
    cfg.protein_beam.max_evaluations=1
    with open_dict(cfg.protein_beam):cfg.protein_beam.root_parity_reference='/case/runs/001_binder_interface/01_astrolabe_beam/04_calibrate_search/search/D5056_D15/candidates/000000/result.json'
    calls=[]
    class Pool:
        def __init__(self,**kw):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def submit(self,func,t):
            calls.append(t['kind']);f=Future()
            f.set_result(dict(t,status='done',score=0.,metrics={m:.2 for m in beam.METRICS},snapshot='unused',structure_path='unused'))
            return f
    monkeypatch.setattr(beam,'ProcessPoolExecutor',Pool)
    monkeypatch.setattr(parity,'root_parity',lambda *a:dict(passed=True,different=[]))
    assert beam.run_beam(cfg,tmp_path/'native','unused')['budget_complete']
    monkeypatch.setattr(parity,'root_parity',lambda *a:dict(passed=False,different=['all_atom_coordinates']))
    with pytest.raises(ValueError,match='before proposal'):beam.run_beam(cfg,tmp_path/'native','unused',resume=True)
    assert calls==['evaluate']


def test_root_parity_detects_sidechain_motion_with_unchanged_backbone(monkeypatch,tmp_path):
    import shlex
    import astrolabe.protein_beam_root_parity as parity
    reference=Path('/case/runs/001_binder_interface/01_astrolabe_beam/04_calibrate_search/search/D5056_D15/candidates/000000/result.json')
    old=json.loads(reference.read_text());assert parity.root_parity(old,reference)['passed']
    full=parity._confidence(old)
    lines=Path(old['structure_path']).read_text().splitlines();columns=[]
    for index,line in enumerate(lines):
        if line.startswith('_atom_site.'):columns.append(line.strip().split('.',1)[1])
        elif columns and line.startswith('ATOM '):
            fields=shlex.split(line)
            if fields[columns.index('label_atom_id')]=='CB':
                ix=columns.index('Cartn_x');fields[ix]=str(float(fields[ix])+1.)
                lines[index]=' '.join(fields);break
    else:raise AssertionError('No sidechain atom in test native root')
    moved=tmp_path/'moved_model.cif';moved.write_text('\n'.join(lines)+'\n')
    monkeypatch.setattr(parity,'_confidence',lambda record:full)
    evidence=parity.root_parity(dict(old,structure_path=str(moved)),reference)
    assert not evidence['passed'] and evidence['different']==['all_atom_coordinates']


@pytest.mark.parametrize('case,candidate',[('D6998_D19',8),('D6998_D19',11),('D6542_D32',3),('D6542_D32',15),('D5056_D15',9),('D5056_D15',11)])
def test_native_root_and_fixed_interface_geometry_matches_independent_review(case,candidate):
    root=Path('/case')
    if not (root/'analysis').exists():pytest.skip('Retained case files unavailable')
    rev='001_binder_interface/01_astrolabe_beam/04_calibrate_search'
    state=json.loads((root/'runs'/rev/'search'/case/'beam_state.json').read_text())
    review=json.loads((root/'analysis'/rev/'07_geometry_review.json').read_text())
    row=next(r for r in review['representatives'] if r['case']==case and int(r['id'])==candidate)
    record=next(r for r in state['records'] if r['id']==candidate)
    original=root/'inputs/001_binder_interface/01_astrolabe_beam/03_bind_target_homologies'/(case+'.pdb')
    g=pose_geometry(original,record['structure_path'],state['records'][0]['structure_path'])
    for a,b in [('root_binder_pose_ca_rmsd','vs_root_target_aligned_binder_ca_rmsd'),('root_interface_bb_rmsd','vs_root_interface_bb_rmsd'),('interface_bb_rmsd','vs_input_interface_bb_rmsd'),('root_complex_bb_rmsd','vs_root_complex_bb_rmsd'),('root_interface_binder_bb_rmsd','vs_root_target_aligned_interface_binder_bb_rmsd')]:
        assert abs(g[a]-row[b])<1e-8
    own=pose_geometry(original,state['records'][0]['structure_path'],state['records'][0]['structure_path'])
    assert max(own[k] for k in own if k.startswith('root_'))<1e-10
