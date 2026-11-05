"""Real adapter, fake device completion: test async ownership without importing AF3."""
from concurrent.futures import Future,ThreadPoolExecutor
from contextlib import ExitStack
from types import SimpleNamespace
import sys,threading,time
import pytest
from astrolabe.inference_pipeline import Pending,PipelineExecutor,Stage
from astrolabe.evaluators.af3_pipeline import stage_factory
from astrolabe.evaluators.inference_service import call,serve_staged

def factory(monkeypatch,dispatch):
 name='astrolabe.evaluators.wrappers.alphafold3'
 monkeypatch.setitem(sys.modules,name,SimpleNamespace(service_model_settings=lambda c:{'fixed':True}))
 runner=SimpleNamespace(_model_config=SimpleNamespace(global_config=SimpleNamespace()),
   inference_stages=lambda key:[Stage('af3_model','gpu',lambda value:dispatch(value,key)),Stage('af3_materialize','cpu',lambda value:value+1)])
 return stage_factory(runner,object())
def request(value,key=1):return dict(model='alphafold3',settings={'fixed':True},cyclic_asym_ids=[],cyclic_relative_patches=[],features=value,rng_key=key,identity={'seed':key})
def test_two_predictions_pending_on_one_owner_without_early_materialization(monkeypatch):
 completed=[Future(),Future()];dispatched=threading.Event();threads=[];keys=[]
 def dispatch(value,key):
  threads.append(threading.get_ident());keys.append(key)
  if len(keys)==2:dispatched.set()
  return Pending(value*2,completed[value-1],keepalive=object())
 stages=factory(monkeypatch,dispatch)
 with ExitStack() as stack:
  pools={name:stack.enter_context(ThreadPoolExecutor(max_workers=n)) for name,n in [('cpu',2),('gpu',1),('io',1)]}
  pipeline=stack.enter_context(PipelineExecutor(pools,max_pending=2,on_event=None))
  items=[dict(value=None,payload=r,request_id=i) for i,r in enumerate([request(1,7),request(2,11)])]
  results=[pipeline.submit(stages(item['payload']),item) for item in items]
  assert dispatched.wait(3),'Second prediction must dispatch before first completion'
  assert not any(r.done() for r in results)
  completed[1].set_result(None);assert results[1].result(timeout=3)['value']==5
  assert not results[0].done();completed[0].set_result(None);assert results[0].result(timeout=3)['value']==3
  assert len(set(threads))==1 and keys==[7,11]
def test_fixed_owner_rejects_mutable_cyclic_and_wrong_settings(monkeypatch):
 stages=factory(monkeypatch,lambda value,key:value)
 for fields in [dict(model='other'),dict(settings={'fixed':False}),dict(cyclic_asym_ids=[1]),dict(cyclic_relative_patches=[[1,2]])]:
  with pytest.raises(ValueError):stages(dict(request(1),**fields))
def test_rpc_validation_error_recovery_and_shutdown_waits_for_device(monkeypatch,tmp_path):
 completion=Future();dispatched=threading.Event();socket=tmp_path/'service.sock'
 def dispatch(value,key):dispatched.set();return Pending(value,completion)
 stages=factory(monkeypatch,dispatch)
 with ExitStack() as stack:
  pools={name:stack.enter_context(ThreadPoolExecutor(max_workers=n)) for name,n in [('cpu',2),('gpu',1),('io',2)]}
  callers=stack.enter_context(ThreadPoolExecutor(max_workers=3));pipeline=PipelineExecutor(pools,max_pending=2,on_event=None)
  owner=callers.submit(serve_staged,stages,socket,pipeline,max_pending=2)
  deadline=time.monotonic()+3
  while not socket.exists() and not owner.done() and time.monotonic()<deadline:time.sleep(.005)
  try:
   with pytest.raises(RuntimeError,match='settings'):call(socket,dict(request(2),settings={}))
   good=callers.submit(call,socket,request(5));assert dispatched.wait(3)
   shutdown=callers.submit(call,socket,{'type':'shutdown'})
   assert not shutdown.done() and not good.done()
  finally:completion.set_result(None)
  assert good.result(timeout=3)==6;assert shutdown.result(timeout=3) is None;owner.result(timeout=3)
  assert not socket.exists()


def test_disconnected_validation_client_does_not_stop_owner(tmp_path):
 import pickle
 from multiprocessing.connection import Client
 closed=threading.Event();checked=threading.Event();socket=tmp_path/'service.sock'
 def stages(payload):
  if payload.get('bad'):
   checked.set();assert closed.wait(3);raise ValueError('invalid request')
  return [Stage('result','cpu',lambda item:dict(item,value=7))]
 with ExitStack() as stack:
  pools={name:stack.enter_context(ThreadPoolExecutor(max_workers=1)) for name in ['cpu','io']}
  owner_pool=stack.enter_context(ThreadPoolExecutor(max_workers=1));pipeline=PipelineExecutor(pools,on_event=None)
  owner=owner_pool.submit(serve_staged,stages,socket,pipeline)
  deadline=time.monotonic()+3
  while not socket.exists() and not owner.done() and time.monotonic()<deadline:time.sleep(.005)
  connection=Client(str(socket),family='AF_UNIX');connection.send_bytes(pickle.dumps({'bad':True},protocol=5))
  assert checked.wait(3);connection.close();closed.set()
  try:assert call(socket,{})==7
  finally:assert call(socket,{'type':'shutdown'}) is None
  owner.result(timeout=3)
