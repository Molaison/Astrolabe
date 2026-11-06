"""Dispatch order, bounded refill and failure behavior without GPU/timing claims."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,Future
import threading
import pytest
import astrolabe.protein_beam as beam
from astrolabe.design import branch_execution
from astrolabe.design.branch_execution import BranchExecutor
from astrolabe.design.branch_types import (
    BranchAction,
    BranchEvent,
    BranchTerminal,
)
from branching_test_support import (
    native_branch_config,
    native_control_assets,
    prepare_one_generation,
)


pytestmark = pytest.mark.usefixtures("native_control_assets")


def test_free_slot_refills_while_first_task_is_still_blocked(monkeypatch):
    released=threading.Event();fourth_started=threading.Event()
    lock=threading.Lock();active=[0];maximum=[0]
    def task(item):
        with lock:
            active[0]+=1;maximum[0]=max(maximum[0],active[0])
        try:
            if item['id']==0:
                assert fourth_started.wait(5),'A finished slot was not refilled'
                released.set()
            if item['id']==3:
                assert not released.is_set()
                fourth_started.set()
            return item
        finally:
            with lock:active[0]-=1
    monkeypatch.setattr(beam,'worker_task',task)
    inputs=[dict(id=i) for i in range(7)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        assert beam.execute_batch(pool,inputs,3)==inputs
    assert fourth_started.is_set() and maximum[0]<=3


def test_known_failure_aborts_further_submission_and_does_not_hide_terminal_results():
    submitted=[]
    class Pool:
        def submit(self,func,item):
            submitted.append(item['id']);future=Future()
            if item['id']==1:future.set_exception(RuntimeError('failure injection'))
            else:future.set_result(item)
            return future
    with pytest.raises(RuntimeError,match='injection'):
        beam.execute_batch(Pool(),[dict(id=i) for i in range(8)],3)
    assert submitted==[0,1,2]


def test_empty_remainder_and_identity_guard(monkeypatch):
    assert beam.execute_batch(object(),[],1)==[]
    class Pool:
        def submit(self,func,item):
            future=Future();future.set_result(dict(id=999));return future
    with pytest.raises(ValueError,match='identity'):
        beam.execute_batch(Pool(),[dict(id=1)],2)
    with pytest.raises(ValueError,match='positive'):
        beam.execute_batch(Pool(),[],0)


def test_execute_batch_is_thin_gather_ordered_wrapper(monkeypatch):
    calls=[]
    expected=object()
    def gather(pool,tasks,width,worker,identity):
        calls.append((pool,tasks,width,worker,identity))
        return expected
    monkeypatch.setattr(branch_execution,'gather_ordered',gather)
    pool=object();tasks=[dict(id=1)]
    assert beam.execute_batch(pool,tasks,2) is expected
    assert calls==[(pool,tasks,2,beam.worker_task,calls[0][4])]
    assert calls[0][4]({'id':7})==7


def test_accepted_prefix_overlaps_next_proposal_phase():
    prepared=prepare_one_generation(
        native_branch_config(),
        proposals_per_parent=6,
        max_evaluations=7,
        proposal_batch_size=3,
        evaluation_width=3,
        pipeline=True,
    )
    first=prepared.policy.ask(prepared.state)
    assert tuple(job.id for job in first.jobs)==(1,2,3)
    state=prepared.policy.tell(
        prepared.policy.admit(prepared.state,first),
        BranchEvent(tuple(
            prepared.proposal_terminal(job,f"child-{job.id}")
            for job in first.jobs
        )),
    )
    assert len(state.queued)==3

    evaluation=prepared.policy.ask(state)
    assert tuple(job.phase for job in evaluation.jobs)==('evaluate',)*3
    admitted=prepared.policy.admit(state,evaluation)
    assert len(admitted.pending)==3
    second=prepared.policy.ask(admitted)
    assert tuple(job.id for job in second.jobs)==(4,5,6)
    assert all(job.phase=='propose' for job in second.jobs)


def _fake_pool(output,eval_calls,fail_proposal=None):
    import json
    class Pool:
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def submit(self,func,t):
            future=Future()
            try:
                if t['kind']=='proposal':
                    if fail_proposal and t['id']==fail_proposal[0]:
                        fail_proposal[0]=-1;raise RuntimeError('CPU proposal interruption')
                    seq=list(t['sequence']);seq[t['id']%len(seq)]='ACDEFGHIKLMNPQRSTVWY'[t['id']%20]
                    value=dict(t,sequence=''.join(seq),valid=t['id']%5!=0,
                               proposal_rejections=[] if t['id']%5 else ['excluded_amino_acid'])
                else:
                    eval_calls.append(t['id'])
                    if t.get('af3_call_token'):
                        from astrolabe.protein_beam_budget import claim_call
                        claim_call(dict(scope=t['af3_call_token'].rsplit(':',2)[0],
                                        ledger=output/'af3_calls.json'),t)
                    p=output/'candidates'/f"{t['id']:06d}";p.mkdir(parents=True,exist_ok=True)
                    snapshot=p/'scores.pkl';snapshot.write_bytes(f"scores-{t['id']}".encode())
                    structure=output/'workers'/f"fixture-{t['id']}"/'structures'/f"step_{t['id']:03d}_AB_model.cif"
                    structure.parent.mkdir(parents=True,exist_ok=True);structure.write_text(f"data_fixture_{t['id']}\n")
                    value=dict(t,status='done',score=float(t['id']),metrics={m:.2 for m in beam.METRICS},
                               holes={},geometry={},snapshot=str(snapshot),structure_path=str(structure))
                    from astrolabe.protein_beam_contract import native_artifact_manifest
                    value['native_artifacts']=native_artifact_manifest(value)
                    (p/'result.json').write_text(json.dumps(value))
                future.set_result(value)
            except Exception as e:future.set_exception(e)
            return future
    return Pool


def test_logical_proposals_and_multigeneration_results_do_not_depend_on_worker_count(monkeypatch,tmp_path):
    from copy import deepcopy
    cfg=beam.load_config(str(Path(__file__).parent/'data/protein_beam/02_fixture.yaml'))
    cfg.protein_beam.proposal_batch_size=3
    results=[]
    for workers,pipeline in [(3,False),(3,True),(4,True)]:
        config=deepcopy(cfg);config.protein_beam.pipeline_proposals=pipeline
        output=tmp_path/f'run{workers}-{pipeline}';calls=[]
        monkeypatch.setattr(beam,'ProcessPoolExecutor',_fake_pool(output,calls))
        state=beam.run_beam(config,output,'unused',workers=workers,in_flight=workers)
        assert state['budget_complete'] and len(calls)==len(set(calls))==7
        results.append([(r['id'],r['parent_id'],r['generation'],r['sequence'],r['seed'],r['score']) for r in state['records']])
    assert results[0]==results[1]==results[2]


def test_midgeneration_cpu_interruption_reuses_completed_native_evaluations(monkeypatch,tmp_path):
    cfg=beam.load_config(str(Path(__file__).parent/'data/protein_beam/02_fixture.yaml'))
    cfg.protein_beam.proposal_batch_size=3;cfg.protein_beam.pipeline_proposals=True
    output=tmp_path/'search';calls=[];fail=[4]
    monkeypatch.setattr(beam,'ProcessPoolExecutor',_fake_pool(output,calls,fail))
    with pytest.raises(RuntimeError,match='CPU proposal interruption'):
        beam.run_beam(cfg,output,'old-owner',workers=4,in_flight=4)
    final=beam.run_beam(cfg,output,'new-owner',workers=4,in_flight=4,resume=True)
    assert final['budget_complete'] and len(calls)==len(set(calls))==7


def test_ready_group_failure_keeps_atomic_admission_for_explicit_recovery():
    prepared=prepare_one_generation(
        native_branch_config(),
        max_evaluations=3,
        proposals_per_parent=2,
        proposal_batch_size=2,
        pipeline=False,
    )
    action=prepared.policy.ask(prepared.state)
    admitted=prepared.policy.admit(prepared.state,action)
    terminals=tuple(
        prepared.proposal_terminal(job,f"child-{job.id}")
        for job in action.jobs
    )
    submitted=[];errors=[]
    def submit(pool,job):
        submitted.append(job.identity)
        future=Future()
        if job==action.jobs[1]:
            future.set_exception(RuntimeError('native job interrupted'))
        else:
            future.set_result(terminals[0])
        return future
    executor=BranchExecutor(
        object(),
        submit=submit,
        on_error=lambda job,error: errors.append((job.identity,str(error))),
    )
    with pytest.raises(RuntimeError,match='native job interrupted'):
        executor.perform(action)
    assert admitted.batch==action.jobs and admitted.batch_results==()
    assert len(executor.one_pending)==2 and executor.observed=={}
    assert submitted==[job.identity for job in action.jobs]
    assert errors==[(action.jobs[1].identity,'native job interrupted')]

    recovered=[]
    def recovered_submit(pool,job):
        recovered.append(job.identity)
        future=Future();future.set_result(terminals[action.jobs.index(job)])
        return future
    resumed=BranchExecutor(object(),submit=recovered_submit)
    resumed.restore(
        action.jobs[0],
        next(
            future
            for future,job in executor.one_pending.items()
            if job==action.jobs[0]
        ),
    )
    recovered_other=Future();recovered_other.set_result(terminals[1])
    resumed.restore(action.jobs[1],recovered_other)
    event=resumed.perform(BranchAction(admitted,jobs=action.jobs,wait='all'))
    assert recovered==[]
    assert tuple(terminal.job.id for terminal in event.terminals)==(1,2)
    completed=prepared.policy.tell(admitted,event)
    assert completed.counts==((0,2),)
    assert tuple(terminal.job.id for terminal in completed.batch_results)==(1,2)


def test_existing_cpu_batch_is_drained_after_native_quota_refusal():
    prepared=prepare_one_generation(
        native_branch_config(),
        max_evaluations=4,
        proposals_per_parent=2,
        proposal_batch_size=2,
        evaluation_width=2,
        pipeline=False,
    )
    state=prepared.state
    proposal_action=prepared.policy.ask(state)
    state=prepared.policy.tell(
        prepared.policy.admit(state,proposal_action),
        BranchEvent(tuple(
            prepared.proposal_terminal(job,f"child-{job.id}")
            for job in proposal_action.jobs
        )),
    )
    evaluation_action=prepared.policy.ask(state)
    state=prepared.policy.admit(state,evaluation_action)
    first,second=evaluation_action.jobs
    state=prepared.policy.tell(
        state,
        BranchEvent((
            BranchTerminal(first,'refused'),
            prepared.evaluation_terminal(second),
        )),
    )
    assert state.stop_reason=='native_call_budget'
    assert state.refused==(first,)
    assert state.queued==()
    assert tuple(member.id for member in state.records)==(0,second.id)

    drain=prepared.policy.ask(state)
    assert drain.jobs==()
    drained=prepared.policy.admit(state,drain)
    final=drained if prepared.policy.finished(drained) else prepared.policy.admit(
        drained,prepared.policy.ask(drained)
    )
    assert final.finished and final.stop_reason=='native_call_budget'
    assert tuple(member.id for member in final.records)==(0,second.id)
