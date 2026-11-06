from pathlib import Path
from copy import deepcopy
import json
import math
import pytest
from branching_test_support import native_control_assets
from astrolabe.protein_beam import METRICS, select_beam, load_config, validate_config

pytestmark = pytest.mark.usefixtures("native_control_assets")


def candidate(i,seq,score):
    return dict(id=i,sequence=seq,score=score,status='done',metrics={m:0.2 for m in METRICS})


def completed_native(output,cfg,task):
    from astrolabe.protein_beam_budget import claim_call
    from astrolabe.protein_beam_contract import native_artifact_manifest

    if cfg.protein_beam.get('af3_call_budget') and task.get('af3_call_token'):
        claim_call(cfg.protein_beam.af3_call_budget, task)
    directory=Path(output)/'candidates'/f"{task['id']:06d}"
    directory.mkdir(parents=True,exist_ok=True)
    snapshot=directory/'scores.pkl'
    snapshot.write_bytes(f"scores-{task['id']}-{task.get('attempt',1)}".encode())
    structure=Path(output)/'workers'/f"fixture-{task['id']}"/'structures'/f"step_{task['id']:03d}_AB_model.cif"
    structure.parent.mkdir(parents=True,exist_ok=True)
    structure.write_text(f"data_fixture_{task['id']}_{task.get('attempt',1)}\n")
    value=dict(task,status='done',score=float(task['id']),
               metrics={m:0.2 for m in METRICS},holes={},geometry={},
               snapshot=str(snapshot),structure_path=str(structure))
    value['native_artifacts']=native_artifact_manifest(value)
    (directory/'result.json').write_text(json.dumps(value))
    return value


def test_elitist_selection_dedup_and_stable_ties():
    parent=candidate(0,'AA',1)
    best=candidate(2,'BB',2)
    assert select_beam([parent,candidate(1,'CC',-1),best,candidate(3,'BB',3)],2)==[candidate(3,'BB',3),parent]
    assert [c['id'] for c in select_beam([candidate(4,'DD',1),parent],2)]==[0,4]


def test_invalid_or_incomplete_metrics_fail():
    c=candidate(0,'AA',1)
    c['metrics']['interface_holes']=math.nan
    with pytest.raises(ValueError): select_beam([c],1)
    del c['metrics']['interface_holes']
    with pytest.raises(KeyError): select_beam([c],1)


def test_engineering_config_and_target_mask():
    cfg=load_config(str(Path(__file__).parent / 'data/protein_beam/02_fixture.yaml'))
    validate_config(cfg)
    broken=deepcopy(cfg)
    broken.seqs.B=str(broken.seqs.B).lstrip('+')
    with pytest.raises(ValueError,match='fixed'): validate_config(broken)


def test_budget_lineage_resume_and_contract_preservation(monkeypatch,tmp_path):
    from concurrent.futures import Future
    import astrolabe.protein_beam as beam
    cfg=load_config(str(Path(__file__).parent / 'data/protein_beam/02_fixture.yaml'))
    calls=[]
    def task(t):
        calls.append(t.copy())
        if t['kind']=='proposal':
            # Fixture stand-in to exercise coordinator identity, not a protein method.
            s=list(t['sequence']);s[t['id']%len(s)]='ACDEFGHIKLMNPQRSTVWY'[t['id']%20]
            return dict(t,sequence=''.join(s))
        return completed_native(tmp_path/'beam',cfg,t)
    class Pool:
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def submit(self,func,t):
            f=Future();f.set_result(task(t));return f
    monkeypatch.setattr(beam,'ProcessPoolExecutor',Pool)
    state=beam.run_beam(cfg,tmp_path/'beam','unused',workers=3,in_flight=3)
    assert state['budget_complete'] and len(state['records'])==7
    assert [r['id'] for r in state['records']]==list(range(7))
    assert all(r['parent_id']<r['id'] for r in state['records'][1:])
    assert len({r['sequence'] for r in state['records']})==7
    count=len(calls)
    assert beam.run_beam(cfg,tmp_path/'beam','unused',workers=3,in_flight=1,resume=True)['status']=='complete'
    assert len(calls)==count
    changed=deepcopy(cfg);changed.seed+=1
    with pytest.raises(ValueError,match='changed'): beam.run_beam(changed,tmp_path/'beam','unused',resume=True)


def test_unknown_native_attempt_is_not_replayed_and_cost_is_retained(monkeypatch,tmp_path):
    from concurrent.futures import Future
    import json
    import astrolabe.protein_beam as beam
    from astrolabe.design.runtime import OutcomeUnknown
    from omegaconf import open_dict
    cfg=load_config(str(Path(__file__).parent / 'data/protein_beam/02_fixture.yaml'))
    cfg.protein_beam.max_evaluations=3
    output=tmp_path/'search';old_socket=tmp_path/'old-owner.sock';old_socket.write_text('live marker')
    with open_dict(cfg.protein_beam):
        cfg.protein_beam.af3_call_budget=dict(
            ledger=str(output/'af3_calls.json'),scope='fixture',max_total=3,max_per_case=3)
    attempts=[];fail=[True]
    def task(t):
        if t['kind']=='proposal':
            sequence=list(t['sequence']);sequence[t['id']]='ACDEFGHIKLMNPQRSTVWY'[t['id']]
            return dict(t,sequence=''.join(sequence))
        attempts.append(t['id'])
        if t['id']==2 and fail[0]:
            from astrolabe.protein_beam_budget import claim_call
            claim_call(cfg.protein_beam.af3_call_budget,t)
            fail[0]=False;raise RuntimeError('injected interruption')
        return completed_native(output,cfg,t)
    class Pool:
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def submit(self,func,t):
            f=Future()
            try:f.set_result(task(t))
            except Exception as e:f.set_exception(e)
            return f
    monkeypatch.setattr(beam,'ProcessPoolExecutor',Pool)
    with pytest.raises(RuntimeError,match='injected'):beam.run_beam(cfg,output,str(old_socket))
    with pytest.raises(RuntimeError,match='explicit'):beam.run_beam(cfg,output,'new-owner',resume=True)
    with pytest.raises(RuntimeError,match='still exists'):beam.run_beam(cfg,output,'new-owner',resume=True,retry_interrupted=True)
    old_socket.unlink()
    with pytest.raises(OutcomeUnknown,match='no verified terminal'):
        beam.run_beam(cfg,output,'new-owner',resume=True,retry_interrupted=True)
    assert attempts==[0,1,2]
    state=json.loads((output/'beam_state.json').read_text())
    assert state['records'][2]['status'] in {'pending','running'}
    assert not (output/'candidates/000002/result.json').exists()
    ledger=json.loads(Path(cfg.protein_beam.af3_call_budget.ledger).read_text())
    assert {entry['status'] for entry in ledger['entries'].values()}=={'issued'}
    assert {entry['candidate_id'] for entry in ledger['entries'].values()}=={0,1,2}


def test_unknown_attempt_ignores_retry_flag_without_owner_terminal(monkeypatch,tmp_path):
    from concurrent.futures import Future
    import json
    import astrolabe.protein_beam as beam
    from astrolabe.design.runtime import OutcomeUnknown
    from omegaconf import open_dict
    cfg=load_config(str(Path(__file__).parent / 'data/protein_beam/02_fixture.yaml'))
    cfg.protein_beam.max_evaluations=3
    output=tmp_path/'search';old=tmp_path/'owner1.sock';new=tmp_path/'owner2.sock'
    with open_dict(cfg.protein_beam):
        cfg.protein_beam.af3_call_budget=dict(
            ledger=str(output/'af3_calls.json'),scope='fixture',max_total=3,max_per_case=3)
    old.write_text('owner1');attempts=[];failures=[2]
    def task(t):
        if t['kind']=='proposal':
            sequence=list(t['sequence']);sequence[t['id']]='ACDEFGHIKLMNPQRSTVWY'[t['id']]
            return dict(t,sequence=''.join(sequence))
        attempts.append(t['id'])
        if t['id']==2 and failures[0]:
            from astrolabe.protein_beam_budget import claim_call
            claim_call(cfg.protein_beam.af3_call_budget,t)
            failures[0]-=1;raise RuntimeError('injected repeated interruption')
        return completed_native(output,cfg,t)
    class Pool:
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def submit(self,func,t):
            f=Future()
            try:f.set_result(task(t))
            except Exception as e:f.set_exception(e)
            return f
    monkeypatch.setattr(beam,'ProcessPoolExecutor',Pool)
    with pytest.raises(RuntimeError,match='interruption'):beam.run_beam(cfg,output,str(old))
    old.unlink();new.write_text('owner2')
    with pytest.raises(OutcomeUnknown,match='no verified terminal'):
        beam.run_beam(cfg,output,str(new),resume=True,retry_interrupted=True)
    state=json.loads((output/'beam_state.json').read_text())
    assert state['records'][2]['status'] in {'pending','running'}
    assert not (output/'candidates/000002/result.json').exists()
    with pytest.raises(OutcomeUnknown,match='no verified terminal'):
        beam.run_beam(cfg,output,'owner3',resume=True)
    with pytest.raises(OutcomeUnknown,match='no verified terminal'):
        beam.run_beam(cfg,output,'owner3',resume=True,retry_interrupted=True)
    assert attempts==[0,1,2]
    ledger=json.loads(Path(cfg.protein_beam.af3_call_budget.ledger).read_text())
    assert {entry['status'] for entry in ledger['entries'].values()}=={'issued'}
    assert {entry['candidate_id'] for entry in ledger['entries'].values()}=={0,1,2}
