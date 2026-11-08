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
   _device=SimpleNamespace(platform='gpu'),
   inference_stages=lambda key:[Stage('af3_model','gpu',lambda value:dispatch(value,key)),Stage('af3_materialize','cpu',lambda value:value+1)])
 return stage_factory(runner,object())
def request(value,key=1):return dict(model='alphafold3',settings={'fixed':True},cyclic_asym_ids=[],cyclic_relative_patches=[],features=value,rng_key=key,identity={'seed':key})
def test_two_predictions_pending_on_one_owner_without_early_materialization(monkeypatch):
 completed=[Future(),Future()];first=threading.Event();second=threading.Event();threads=[];keys=[]
 def dispatch(value,key):
  threads.append(threading.get_ident());keys.append(key)
  (first if len(keys)==1 else second).set()
  return Pending(value*2,completed[value-1],keepalive=object())
 stages=factory(monkeypatch,dispatch)
 with ExitStack() as stack:
  pools={name:stack.enter_context(ThreadPoolExecutor(max_workers=n)) for name,n in [('cpu',2),('gpu',1),('io',1)]}
  pipeline=stack.enter_context(PipelineExecutor(pools,max_pending=2,on_event=None))
  items=[dict(value=None,payload=r,request_id=i) for i,r in enumerate([request(1,7),request(2,11)])]
  results=[pipeline.submit(stages(item['payload']),item) for item in items]
  try:
   assert first.wait(3) and not second.wait(.05)
   assert not any(r.done() for r in results)
  finally:
   completed[0].set_result(None)
  assert results[0].result(timeout=3)['value']==3
  assert second.wait(3)
  completed[1].set_result(None);assert results[1].result(timeout=3)['value']==5
  assert keys==[7,11]
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


@pytest.fixture
def af3_broker(monkeypatch):
 """CPU admission double, including shared request-capacity lifetime."""
 from contextlib import contextmanager
 from astrolabe import model_execution
 class Broker:
  socket_path='/synthetic/af3-broker'
  def __init__(self):self.active=[];self.events=[]
  @contextmanager
  def lease(self,**fields):
   entry=SimpleNamespace(token=str(len(self.events)),**fields)
   self.active.append(entry);self.events.append(('enter',entry))
   try:yield entry
   finally:self.active.remove(entry);self.events.append(('exit',entry))
  def register_model(self,*args,**kwargs):pass
  @property
  def phase(self):return self.active[-1].phase
 broker=Broker()
 monkeypatch.setattr(model_execution,'_resource_client',lambda:broker)
 return broker


def synthetic_runner(broker, completion=None, *, fail=None):
 from astrolabe.evaluators.wrappers.alphafold3 import ModelRunner
 import numpy as np
 calls=[]
 class DeviceOnly:
  __module__='jax.synthetic_af3'
  shape=(2,)
  def __init__(self,value):self.value=value
  def __array__(self,dtype=None):
   assert broker.phase=='finalize','raw output was materialized before finalize'
   return np.asarray(self.value,dtype=dtype)
 class Runner(ModelRunner):
  def __init__(self):
   super().__init__(SimpleNamespace(global_config=SimpleNamespace()),
                    SimpleNamespace(platform='gpu',id=2),'unused')
   self.__dict__['model_params']={'__meta__':{'__identifier__':np.asarray([1],dtype=np.uint8)}}
   self.__dict__['_model']=object()
  def prepare_features(self,batch):
   assert broker.phase=='prepare';calls.append('features')
   if fail=='prepare':raise ValueError('prepare failed')
   return batch
  def upload_features(self,batch):
   assert broker.phase=='prepare';calls.append('upload');return batch
  def run_model(self,batch,key):
   assert broker.phase=='infer';calls.append(('model',tuple(key)))
   if fail=='infer':raise ValueError('infer failed')
   return {'raw':DeviceOnly(batch)}
  def materialize_result(self,raw):
   assert broker.phase=='finalize';calls.append('materialize')
   if fail=='finalize':raise ValueError('finalize failed')
   return {'coordinates':np.asarray(raw['raw'],dtype=np.float32),'__identifier__':b'native'}
  def extract_structures(self,batch,result,target_name):
   assert broker.phase=='finalize';calls.append(('extract',target_name))
   return [dict(sample=0,target=target_name),dict(sample=1,target=target_name)]
 return Runner(),calls


def patch_transfers(monkeypatch,broker,completion=None):
 from astrolabe.evaluators.wrappers import alphafold3 as wrapper
 import numpy as np
 def upload(value,device):
  assert broker.phase=='prepare'
  return value
 monkeypatch.setattr(wrapper.jax,'device_put',upload)
 def wait(value):
  if isinstance(value,dict) and 'raw' in value:
   assert broker.phase=='infer'
   if completion is not None:completion.result(timeout=5)
  else:assert broker.phase=='prepare'
  return value
 monkeypatch.setattr(wrapper.jax,'block_until_ready',wait)
 monkeypatch.setattr(wrapper.jax.random,'PRNGKey',lambda seed:np.asarray([0,seed],dtype=np.uint32))


def test_local_modelrunner_real_stages_keep_raw_on_device_until_finalize(monkeypatch,af3_broker):
 import numpy as np
 runner,calls=synthetic_runner(af3_broker)
 patch_transfers(monkeypatch,af3_broker)
 output=runner.run_inference([1,2],np.asarray([0,17],dtype=np.uint32))
 np.testing.assert_array_equal(output['coordinates'],[1,2])
 assert output['coordinates'].dtype==np.float32 and output['__identifier__']==b'native'
 assert calls==['features','upload',('model',(0,17)),'materialize']
 entered=[entry for event,entry in af3_broker.events if event=='enter']
 assert [entry.phase for entry in entered]==['load','evaluation_memory','prepare','infer','finalize']
 memory=entered[1]
 assert memory.capacity_only
 assert all(entry.reservation_token==memory.token for entry in entered[2:])
 assert [entry.evaluator_inference for entry in entered[2:]]==[False,True,False]
 assert not af3_broker.active


def test_pending_model_cannot_materialize_or_release_infer_early(monkeypatch,af3_broker):
 import numpy as np
 from astrolabe.evaluators.runtime_stages import DirectKernel
 from astrolabe.evaluators.af3_pipeline import evaluator_stages
 completion=Future();dispatched=threading.Event()
 runner,calls=synthetic_runner(af3_broker)
 patch_transfers(monkeypatch,af3_broker,completion)
 run=runner.run_model
 def dispatch(batch,key):
  value=run(batch,key);dispatched.set();return value
 runner.run_model=dispatch
 kernel=DirectKernel({},lambda _: (evaluator_stages(runner),{}))
 payload=dict(model='alphafold3',features=[1,2],rng_key=np.asarray([0,7]))
 with ThreadPoolExecutor(max_workers=1) as pool:
  future=pool.submit(kernel,payload)
  try:
   assert dispatched.wait(3) and not future.done()
   assert af3_broker.phase=='infer' and 'materialize' not in calls
   assert [item.phase for item in af3_broker.active]==['evaluation_memory','infer']
  finally:completion.set_result(None)
  assert future.result(timeout=3)['__identifier__']==b'native'
 assert not af3_broker.active


@pytest.mark.parametrize('failure',['prepare','infer','finalize'])
def test_af3_stage_failure_does_not_leak_capacity(monkeypatch,af3_broker,failure):
 import numpy as np
 runner,calls=synthetic_runner(af3_broker,fail=failure)
 patch_transfers(monkeypatch,af3_broker)
 with pytest.raises(ValueError,match=failure+' failed'):
  runner.run_inference([1,2],np.asarray([0,7]))
 assert not af3_broker.active
 if failure!='finalize':assert 'materialize' not in calls


def test_native_factory_and_both_service_modes_use_the_same_stages(monkeypatch,af3_broker):
 from omegaconf import OmegaConf
 from astrolabe.evaluators import af3_service,af3_pipeline,native_daemons
 from astrolabe.evaluators.runtime_stages import complete_stage
 from astrolabe.model_execution import capture_random_state
 from astrolabe.evaluators.wrappers.alphafold3 import service_model_settings
 import numpy as np
 config=OmegaConf.create(dict(model_dir='unused',flash_attention_implementation='xla',
                             num_samples=2,num_recycles=3,save_embeddings=False,gpu_device=2))
 runner,calls=synthetic_runner(af3_broker)
 patch_transfers(monkeypatch,af3_broker)
 monkeypatch.setattr(af3_service,'load_runner',lambda config:runner)
 handler=native_daemons._af3_factory({'config':OmegaConf.to_container(config),
                                     'service_settings':service_model_settings(config)})
 payload=dict(model='alphafold3',settings=service_model_settings(config),features=[1,2],
              cyclic_asym_ids=[],cyclic_relative_patches=[],rng_key=np.asarray([0,19]))
 result=handler(dict(operation='infer',payload=payload,random_state=capture_random_state()))
 assert result['ok'] and set(result['runtime']['timings'])=={'prepare','infer','finalize'}
 assert calls==['features','upload',('model',(0,19)),'materialize']
 lease_count=len(af3_broker.events)
 invalid=handler(dict(operation='infer',payload=dict(payload,cyclic_asym_ids=[2]),
                      random_state=capture_random_state()))
 assert not invalid['ok'] and 'immutable' in invalid['error']
 assert len(af3_broker.events)==lease_count
 assert handler({'operation':'describe'})['metadata']['runtime_lifecycle']['state']=='ready'
 assert handler(dict(operation='infer',payload=payload,random_state=capture_random_state()))['ok']
 calls.clear()
 kernel=af3_service.inference_handler(runner,config)
 np.testing.assert_array_equal(kernel(payload)['coordinates'],[1,2])
 stages=af3_pipeline.stage_factory(runner,config,device_kernel=kernel)(payload)
 assert [stage.name for stage in stages]==['af3_evaluate']
 item=dict(payload=payload,request_id=1)
 for stage in stages:item=complete_stage(stage,item)
 np.testing.assert_array_equal(item['value']['coordinates'],[1,2])
 assert calls==['features','upload',('model',(0,19)),'materialize']*2
 assert not af3_broker.active


def test_integrated_extraction_and_legacy_extract_rpc_are_finalize_work(monkeypatch,af3_broker):
 runner,calls=synthetic_runner(af3_broker)
 patch_transfers(monkeypatch,af3_broker)
 result=runner.run_inference_and_extract_for_seed([1,2],29,'target')
 assert result==[dict(sample=0,target='target'),dict(sample=1,target='target')]
 assert calls==['features','upload',('model',(0,29)),'materialize',('extract','target')]
 result=runner._inference_kernel(dict(operation='extract_structures',batch=[1,2],
                                      result={'coordinates':[1,2]},target_name='legacy'))
 assert result[0]['target']=='legacy'
 assert calls[-1]==('extract','legacy') and len([c for c in calls if isinstance(c,tuple) and c[0]=='model'])==1
 assert not af3_broker.active


def test_bound_local_cyclic_owner_cannot_be_mutated_after_first_request(monkeypatch,af3_broker):
 import numpy as np
 runner,calls=synthetic_runner(af3_broker)
 patch_transfers(monkeypatch,af3_broker)
 runner.set_cyclic_asym_ids([1]);runner.set_cyclic_relative_patches([[1,2]])
 runner.__dict__['_model']=object()  # The setters invalidate the native JIT cache.
 runner.run_inference([1,2],np.asarray([0,7]))
 runner.set_cyclic_asym_ids([1])
 for setter,value in [(runner.set_cyclic_asym_ids,[2]),(runner.set_cyclic_relative_patches,[[2,3]])]:
  with pytest.raises(ValueError,match='immutable'):setter(value)
 assert runner._model_config.global_config.cyclic_asym_ids==(1,)


def test_facade_uses_scoped_inputs_kwargs_buckets_and_hook_cache(monkeypatch,tmp_path):
 from astrolabe.evaluators.alphafold3 import AlphaFold3Evaluator
 evaluator=AlphaFold3Evaluator(SimpleNamespace(_client=object()),tmp_path)
 evaluator._af3_antibody_msa_cache={'calls':[]}
 source={'A':{'values':[1]}}
 runtime={'labels':[]}
 def prepare(self,inputs,config):
  assert self is not evaluator and self.model_runner is not evaluator.model_runner
  inputs['A']['values'].append(2)
  self.kwargs['A']['flag']=len(self._af3_antibody_msa_cache['calls'])
  self.buckets['A']=(256,)
  self._pre_eval_hook_reports={'A':['report']}
  self._af3_antibody_msa_cache['calls'].append('request')
  config['labels'].append('frozen')
  return inputs
 def run(self,inputs):
  return inputs['A']['values'],self.kwargs['A']['flag']
 def output(self,outputs,inputs,config):
  assert config['labels']==['frozen'] and self._pre_eval_hook_reports=={'A':['report']}
  assert inputs['A']['values']==[1,2]
  return outputs
 monkeypatch.setattr(AlphaFold3Evaluator,'prepare_input',prepare)
 monkeypatch.setattr(AlphaFold3Evaluator,'run_function',run)
 monkeypatch.setattr(AlphaFold3Evaluator,'prepare_output',output)
 assert evaluator.evaluate(source,runtime)==([1,2],0)
 assert evaluator.evaluate(source,runtime)==([1,2],1)
 assert source=={'A':{'values':[1]}} and runtime=={'labels':[]}
 assert not evaluator.kwargs and not evaluator.buckets and not evaluator._pre_eval_hook_reports
 assert evaluator._af3_antibody_msa_cache=={'calls':['request','request']}


def test_facade_preserves_seed_and_sample_order_and_extracts_in_owner(monkeypatch):
 from contextlib import nullcontext
 from astrolabe.evaluators import alphafold3 as facade
 calls=[]
 monkeypatch.setattr(facade.chemical_components,'Ccd',lambda **kw:object())
 monkeypatch.setattr(facade.featurisation,'featurise_input',lambda **kw:[{'seed':19},{'seed':7}])
 monkeypatch.setattr(facade,'allow_all_x_protein_chains_during_featurisation',lambda _:nullcontext())
 monkeypatch.setattr(facade.jax,'local_devices',lambda **kw:pytest.fail('coordinator initialized a device'))
 def infer(batch,seed,name):
  calls.append((batch['seed'],seed,name))
  return [f'{seed}-sample0',f'{seed}-sample1']
 runner=SimpleNamespace(run_inference_and_extract_for_seed=infer)
 fold=SimpleNamespace(rng_seeds=[19,7],user_ccd=None,name='ordered')
 result=facade.AlphaFold3Evaluator.predict_structure(fold,runner,None)
 assert calls==[(19,19,'ordered'),(7,7,'ordered')]
 assert [(r.seed,r.inference_results) for r in result]==[(19,['19-sample0','19-sample1']),(7,['7-sample0','7-sample1'])]


@pytest.mark.parametrize('pipeline',[False,True])
def test_service_cli_modes_share_native_stage_boundary(monkeypatch,af3_broker,pipeline):
 from omegaconf import OmegaConf
 from astrolabe.evaluators import af3_service,af3_pipeline,af3_lifecycle
 from astrolabe.evaluators.runtime_stages import complete_stage
 from astrolabe.evaluators.wrappers.alphafold3 import service_model_settings
 import numpy as np
 config=OmegaConf.create(dict(model_dir='unused',flash_attention_implementation='xla',
                             num_samples=2,num_recycles=3,save_embeddings=False,gpu_device=2))
 runner,calls=synthetic_runner(af3_broker)
 patch_transfers(monkeypatch,af3_broker)
 monkeypatch.setattr(af3_lifecycle,'guard_owner_parent',lambda:None)
 monkeypatch.setattr(af3_service,'load_service_config',lambda _:config)
 monkeypatch.setattr(af3_service,'load_runner',lambda _:runner)
 payload=dict(model='alphafold3',settings=service_model_settings(config),features=[1,2],
              cyclic_asym_ids=[],cyclic_relative_patches=[],rng_key=np.asarray([0,23]))
 results=[]
 monkeypatch.setattr(af3_service,'serve',lambda handler,*args:results.append(handler(payload)))
 def serve_staged(factory,*args,**kwargs):
  item=dict(request_id=1,payload=payload)
  for stage in factory(payload):item=complete_stage(stage,item)
  results.append(item['value'])
 monkeypatch.setattr(af3_pipeline,'serve_staged',serve_staged)
 af3_service.main(['--socket','unused','--config','unused']+(['--pipeline'] if pipeline else []))
 assert calls==['features','upload',('model',(0,23)),'materialize']
 assert results[0]['__identifier__']==b'native' and not af3_broker.active


def test_model_weights_are_uploaded_and_ready_before_request_inference(monkeypatch, af3_broker):
    from astrolabe.evaluators.wrappers import alphafold3 as wrapper

    device = SimpleNamespace(platform="gpu", id=0)
    host_params, device_params = {"weights": object()}, {"weights": object()}
    runner = wrapper.ModelRunner(
        SimpleNamespace(global_config=SimpleNamespace()), device, "unused")
    runner.__dict__["model_params"] = host_params
    observations = []

    def upload(params, selected):
        assert af3_broker.phase == "load"
        assert params is host_params and selected is device
        observations.append("upload")
        return device_params

    def ready(params):
        assert af3_broker.phase == "load" and params is device_params
        observations.append("ready")
        return params

    monkeypatch.setattr(wrapper.jax, "device_put", upload)
    monkeypatch.setattr(wrapper.jax, "block_until_ready", ready)
    monkeypatch.setattr(wrapper.jax, "jit", lambda function, **kwargs: function)
    monkeypatch.setattr(wrapper.hk, "transform",
                        lambda function: SimpleNamespace(apply=lambda *args: None))
    runner._inference_kernel
    assert runner._model.args[0] is device_params
    assert runner.model_params is host_params
    assert observations == ["upload", "ready"]


def test_public_shared_runner_adaptive_entrypoints_are_disabled(monkeypatch):
    from astrolabe.evaluators.af3_service import inference_handler, load_runner
    from astrolabe.evaluators.af3_pipeline import evaluator_stages
    monkeypatch.delenv('ASTROLABE_RESOURCE_SOCKET', raising=False)
    config = {'_service_adaptive': {'enabled': True, 'max_concurrency': 3}}
    with pytest.raises(RuntimeError, match='Shared-runner.*disabled'):
        inference_handler(object(), config)
    with pytest.raises(RuntimeError, match='parent must not load a runner'):
        load_runner(config)
    with pytest.raises(RuntimeError, match='Shared-runner.*disabled'):
        evaluator_stages(object(), adaptive=config['_service_adaptive'])


def test_native_infer_marker_survives_forwarded_transport_id(monkeypatch):
    from astrolabe.evaluators import af3_pipeline
    events = []
    monkeypatch.setattr(af3_pipeline, 'mark', lambda event, **fields: events.append((event, fields)))
    monkeypatch.setitem(sys.modules, 'astrolabe.evaluators.wrappers.alphafold3',
                        SimpleNamespace(service_model_settings=lambda _: {'fixed': True}))
    completion = Future()
    runner = SimpleNamespace(_device=SimpleNamespace(platform='gpu'),
        _model_config=SimpleNamespace(global_config=SimpleNamespace()),
        inference_stages=lambda key: [Stage('af3_model', 'gpu', lambda value: Pending(value, completion)),
                                     Stage('af3_materialize', 'cpu', lambda value: value + 1)])
    stages = af3_pipeline.evaluator_stages(runner, {})
    item = stages.prepare(dict(request(4, 7), request_id=19))
    pending = stages.execute(item)
    assert [event for event, _ in events] == ['gpu_start']
    completion.set_result(None)
    pending.completion()
    assert stages.finalize(pending.value) == 5
    assert [event for event, _ in events] == ['gpu_start', 'gpu_end']
    assert all(fields['stage'] == 'infer' and fields['request_id'] == 19 for _, fields in events)


@pytest.mark.parametrize('pipeline', [False, True])
def test_adaptive_service_with_broker_rejects_before_model_loading(monkeypatch, pipeline):
    from astrolabe.evaluators import af3_service, af3_lifecycle, runtime_stages

    monkeypatch.setenv('ASTROLABE_RESOURCE_SOCKET', '/unused/resource.sock')
    monkeypatch.setattr(af3_lifecycle, 'guard_owner_parent', lambda: None)
    monkeypatch.setattr(af3_service, 'load_service_config', lambda _: {
        '_service_adaptive': {'enabled': True, 'max_concurrency': 6}})
    monkeypatch.setattr(af3_service, 'load_runner',
                        lambda _: pytest.fail('Rejected service must not load its model'))
    monkeypatch.setattr(runtime_stages, 'DirectKernel',
                        lambda *a, **kw: pytest.fail('Rejected service must not construct a kernel'))
    with pytest.raises(RuntimeError, match='resource broker.*not qualified'):
        af3_service.main(['--socket', 'unused', '--config', 'unused'] +
                         (['--pipeline'] if pipeline else []))


def test_adaptive_handler_with_broker_rejects_before_adopting_runner(monkeypatch):
    from astrolabe.evaluators.af3_service import inference_handler

    monkeypatch.setenv('ASTROLABE_RESOURCE_SOCKET', '/unused/resource.sock')
    with pytest.raises(RuntimeError, match='resource broker.*not qualified'):
        inference_handler(object(), {'_service_adaptive': {'enabled': True}})


def test_numeric_bundle_roundtrip_and_unlink_lifetime(tmp_path):
    import gc
    import mmap
    import shutil
    import struct
    import numpy as np
    from astrolabe.evaluators.numeric_buffers import read_bundle, write_bundle

    source = np.arange(12, dtype='>i4').reshape(3, 4)[:, ::2]
    nan_bits = bytes.fromhex('7ff8000000000042')
    payload_nan = struct.unpack('>d', nan_bits)[0]
    payload = {'array': source, 'nested': [np.array([], dtype='<f8'),
               (np.float32(-0.0), np.complex128(complex(float('nan'), -1.0)),
                b'\0\xff', None, True, 17, 'hello', -0.0, payload_nan)]}
    bundle = tmp_path / 'bundle'
    stats = write_bundle(bundle, payload)
    assert stats['array_count'] == 2
    assert stats['data_bytes'] == (source.nbytes + 4 + 16 + 2)
    assert stats['manifest_bytes'] == (bundle / 'manifest.json').stat().st_size
    assert not ((bundle / 'data.bin').stat().st_mode & 0o077)
    restored = read_bundle(bundle)
    array = restored['array']
    assert array.dtype == source.dtype and array.shape == source.shape
    assert array.tobytes() == np.ascontiguousarray(source).tobytes()
    assert array.flags.writeable
    assert restored['nested'][0].shape == (0,)
    scalar, complex_scalar, raw, missing, flag, integer, string, python_float, nan = restored['nested'][1]
    assert type(scalar) is np.float32 and scalar.tobytes() == np.float32(-0.0).tobytes()
    assert type(complex_scalar) is np.complex128 and np.isnan(complex_scalar.real)
    assert complex_scalar.imag == -1.0
    assert (raw, missing, flag, integer, string) == (b'\0\xff', None, True, 17, 'hello')
    assert struct.pack('>d', python_float) == struct.pack('>d', -0.0)
    assert struct.pack('>d', nan) == nan_bits
    view = array[:, 1:]
    base = view
    while base is not None and not isinstance(base, mmap.mmap):
        base = base.base
    assert base is not None
    shutil.rmtree(bundle)
    del restored, array
    gc.collect()
    assert view.tolist() == [[2], [6], [10]]
    view[0, 0] = 99
    assert view[0, 0] == 99
    republished = tmp_path / 'republished'
    write_bundle(republished, {'view': view})
    assert read_bundle(republished)['view'].tolist() == [[99], [6], [10]]


def test_numeric_bundle_rejects_existing_path_and_unsupported_leaves(tmp_path):
    import numpy as np
    from astrolabe.evaluators.numeric_buffers import write_bundle

    existing = tmp_path / 'existing'
    existing.mkdir()
    with pytest.raises(FileExistsError):
        write_bundle(existing, np.arange(2))
    for index, value in enumerate((object(), np.array([object()], dtype=object),
                                   np.array([(1,)], dtype=[('field', 'i4')]),
                                   np.array(['2020-01-01'], dtype='datetime64[D]'))):
        with pytest.raises(TypeError):
            write_bundle(tmp_path / f'bad-{index}', value)


def test_numeric_bundle_rejects_malformed_descriptor_and_symlink(tmp_path):
    import json
    import numpy as np
    from astrolabe.evaluators.numeric_buffers import read_bundle, write_bundle

    bundle = tmp_path / 'bundle'
    write_bundle(bundle, [np.arange(3, dtype='i4'), np.arange(2, dtype='i4')])
    manifest_path = bundle / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['value']['items'][1]['offset'] = 4
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        read_bundle(bundle)
    link = tmp_path / 'linked'
    link.symlink_to(bundle, target_is_directory=True)
    with pytest.raises(ValueError):
        read_bundle(link)


def test_numeric_bundle_empty_array_and_invalid_shape(tmp_path):
    import gc
    import json
    import shutil
    import numpy as np
    from astrolabe.evaluators.numeric_buffers import read_bundle, write_bundle

    bundle = tmp_path / 'empty'
    stats = write_bundle(bundle, np.empty((0, 3), dtype='>f4'))
    assert stats['array_count'] == 1 and stats['data_bytes'] == 1
    array = read_bundle(bundle)
    assert array.shape == (0, 3) and array.dtype == np.dtype('>f4')
    shutil.rmtree(bundle)
    gc.collect()
    assert array.copy().shape == (0, 3)
    republished = tmp_path / 'empty-republished'
    write_bundle(republished, array)
    assert read_bundle(republished).shape == (0, 3)

    bad = tmp_path / 'bad-shape'
    write_bundle(bad, np.arange(2, dtype='i4'))
    manifest_path = bad / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['value']['shape'] = [3]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        read_bundle(bad)
