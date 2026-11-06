import json
import pytest
from astrolabe.protein_beam_budget import reserve_calls,claim_call


def task(i,attempt=1):
    return dict(id=i,kind='evaluate',seed=5,attempt=attempt)


def test_retry_consumes_new_request_quota_before_dispatch(tmp_path):
    p=dict(ledger=str(tmp_path/'calls.json'),scope='caseA',max_total=2,max_per_case=2)
    admitted,refused=reserve_calls(p,[task(1)])
    assert len(admitted)==1 and not refused
    claim_call(p,admitted[0])
    with pytest.raises(RuntimeError):claim_call(p,admitted[0])
    with pytest.raises(RuntimeError):reserve_calls(p,[task(1)])
    retry,refused=reserve_calls(p,[task(1,2)])
    claim_call(p,retry[0])
    admitted,refused=reserve_calls(p,[task(2)])
    assert not admitted and len(refused)==1
    ledger=json.loads((tmp_path/'calls.json').read_text())
    assert len(ledger['entries'])==2 and all(e['status']=='issued' for e in ledger['entries'].values())


def test_case_and_global_limits_preserve_cpu_rescoring_and_uncertain_cost(tmp_path):
    p=dict(ledger=str(tmp_path/'calls.json'),scope='caseA',max_total=3,max_per_case=2)
    a,_=reserve_calls(p,[task(1),task(2)])
    # Even unissued reservations retain cost; a crashed coordinator cannot overbook.
    again,_=reserve_calls(p,[task(1)])
    assert again[0]['af3_call_token']==a[0]['af3_call_token']
    p['scope']='caseB'
    b,refused=reserve_calls(p,[task(1),task(2)])
    assert len(b)==len(refused)==1
    root=dict(id=0,kind='rescore',seed=5)
    assert reserve_calls(p,[root])==([root],[])


def test_quota_contract_and_claim_identity_cannot_change(tmp_path):
    p=dict(ledger=str(tmp_path/'calls.json'),scope='caseA',max_total=3,max_per_case=2)
    admitted,_=reserve_calls(p,[task(1)])
    admitted[0]['seed']=6
    with pytest.raises(RuntimeError):claim_call(p,admitted[0])
    p['max_total']=4
    with pytest.raises(ValueError,match='changed'):reserve_calls(p,[task(2)])
