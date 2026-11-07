from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
import pickle,subprocess,sys,threading,time
import pytest
from astrolabe.evaluators.af3_pool import DaemonPool,serve_pool,process_pool_handler
from astrolabe.evaluators.inference_service import call,serve


_loads = []


def _restore_probe(label):
    _loads.append(label)
    return label


class _Probe:
    def __init__(self, label):
        self.label = label

    def __reduce__(self):
        return _restore_probe, (self.label,)


def _ready(*sockets):
    deadline = time.monotonic() + 3
    while not all(socket.exists() for socket in sockets) and time.monotonic() < deadline:
        time.sleep(.005)
    assert all(socket.exists() for socket in sockets)


def test_numeric_reference_pool_never_uses_pickle_and_preserves_views(tmp_path, monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    from astrolabe.evaluators import inference_service as rpc
    from astrolabe.evaluators.af3_buffers import BufferHandler, infer, prepare_namespace, cleanup_namespace

    def forbidden(*args, **kwargs):
        raise AssertionError("JSON buffer path must not use pickle")

    monkeypatch.setattr(rpc, 'pickle', SimpleNamespace(dumps=forbidden, loads=forbidden))
    child, broker = tmp_path / 'child' / 'owner.sock', tmp_path / 'broker' / 'owner.sock'
    root = prepare_namespace(broker)
    assert prepare_namespace(child, shared_root=root) == root
    assert str(root).startswith('/dev/shm/')
    seen = []

    def predict(request):
        seen.append(request)
        assert 'buffers' not in request and 'target_name' not in request
        return {'coordinates': request['features']['x'] * 2, '__identifier__': b'\x00native\xff'}

    pool = DaemonPool([child], invoke=rpc.json_call, on_event=lambda *args, **kwargs: None)
    with ThreadPoolExecutor(2) as workers:
        child_owner = workers.submit(rpc.serve, BufferHandler(predict, root), child, 1, transport='json')
        _ready(child)
        broker_owner = workers.submit(rpc.serve, pool, broker, 2, transport='json')
        _ready(broker)
        try:
            values = np.arange(12, dtype='>f8').reshape(3, 4)
            result = infer(broker, {'model': 'alphafold3', 'rng_seed': 10079},
                           {'features': {'x': values}})
            assert result['__identifier__'] == b'\x00native\xff'
            np.testing.assert_array_equal(result['coordinates'], values * 2)
            assert not list(root.iterdir())
            result['coordinates'][0, 0] = 42  # Writable after unlink.
            assert result['coordinates'][0, 0] == 42
            assert len(seen) == 1 and seen[0]['rng_seed'] == 10079
            assert rpc.json_call(broker, {'type': 'describe_snapshot'})['completed'] == [1]
        finally:
            rpc.json_call(broker, {'type': 'shutdown'})
            broker_owner.result(5)
            rpc.json_call(child, {'type': 'shutdown'})
            child_owner.result(5)
            assert cleanup_namespace(child) is None
            assert root.exists()  # A drained child does not own the shared namespace.
            assert cleanup_namespace(broker) is None


def test_numeric_failure_retains_bundle_and_quarantines_without_replay(tmp_path):
    import numpy as np
    from astrolabe.evaluators.inference_service import json_call
    from astrolabe.evaluators.af3_buffers import BufferHandler, infer, prepare_namespace, cleanup_namespace

    socket = tmp_path / 'owner.sock'
    root = prepare_namespace(socket)
    seen = []
    def predict(request):
        seen.append(request)
        raise RuntimeError('simulated model failure')

    with ThreadPoolExecutor(1) as workers:
        owner = workers.submit(serve, BufferHandler(predict, root), socket, 1, transport='json')
        _ready(socket)
        try:
            for message in ('simulated model failure', 'quarantined'):
                with pytest.raises(RuntimeError, match=message):
                    infer(socket, {'model': 'alphafold3', 'rng_seed': 7},
                          {'features': {'x': np.arange(3)}})
            assert len(seen) == 1
            assert len(list(root.iterdir())) == 2
        finally:
            json_call(socket, {'type': 'shutdown'})
            owner.result(5)
            assert cleanup_namespace(socket) == str(root)
            # Test-only disposal, after all simulated work is confirmed ended.
            import shutil
            shutil.rmtree(root)


def test_json_ingress_rejects_bad_frames_without_stopping_owner(tmp_path, monkeypatch):
    import json
    from multiprocessing.connection import Client
    from types import SimpleNamespace
    from astrolabe.evaluators import inference_service as rpc

    def forbidden(*args, **kwargs):
        raise AssertionError('JSON ingress must not unpickle')

    monkeypatch.setattr(rpc, 'pickle', SimpleNamespace(loads=forbidden, dumps=forbidden))
    socket = tmp_path / 'owner.sock'
    seen = []

    def predict(request):
        seen.append(request)
        return {'accepted': request['model']}

    bad_frames = (
        pickle.dumps({'model': 'alphafold3'}, protocol=5),
        b'{"model":',
        b'[]',
        b'{"model":NaN}',
        b'[' * 1100 + b'0' + b']' * 1100,
        b'x' * (rpc._JSON_LIMIT + 1),
    )
    with ThreadPoolExecutor(1) as workers:
        owner = workers.submit(rpc.serve, predict, socket, 1, transport='json')
        _ready(socket)
        try:
            for frame in bad_frames:
                with Client(str(socket), family='AF_UNIX') as connection:
                    try:
                        connection.send_bytes(frame)
                    except BrokenPipeError:
                        assert len(frame) > rpc._JSON_LIMIT
                    else:
                        assert json.loads(connection.recv_bytes()) == {
                            'ok': False, 'error': 'Invalid inference control frame'}
                assert not seen and not owner.done()
            assert rpc.json_call(socket, {'model': 'alphafold3'}) == {'accepted': 'alphafold3'}
            assert len(seen) == 1
        finally:
            if not owner.done():
                rpc.json_call(socket, {'type': 'shutdown'})
            owner.result(5)


def test_opaque_rpc_materializes_science_only_at_child_and_caller(tmp_path):
    _loads.clear()
    child, broker = tmp_path / 'child.sock', tmp_path / 'broker.sock'
    seen = []

    def predict(request):
        seen.append(request.get('request_id'))
        if request.get('request_id') is not None:
            assert request['features'] == 'request'
        return {'features': _Probe('response'), 'request_id': request.get('request_id')}

    pool = DaemonPool([child], on_event=lambda *args, **kwargs: None)
    with ThreadPoolExecutor(2) as workers:
        child_owner = workers.submit(serve, predict, child, 1)
        _ready(child)
        broker_owner = workers.submit(serve, pool, broker, 2)
        _ready(broker)
        try:
            result = call(broker, {'model': 'alphafold3', 'identity': {'case': 1},
                                   'features': _Probe('request')}, opaque=True)
            assert result == {'features': 'response', 'request_id': 1}
            assert _loads == ['request', 'response']
            assert seen == [1]
            assert call(broker, {'type': 'describe_snapshot'})['completed'] == [1]
            assert call(child, {'model': 'alphafold3', 'features': 7}) == {
                'features': 'response', 'request_id': None}
        finally:
            call(broker, {'type': 'shutdown'})
            broker_owner.result(5)
            call(child, {'type': 'shutdown'})
            child_owner.result(5)


@pytest.mark.parametrize('mismatch', [False, True])
def test_opaque_child_failure_or_header_mismatch_quarantines_without_replay(tmp_path, mismatch):
    from astrolabe.evaluators.inference_service import _OPAQUE, _opaque_exchange

    child, broker = tmp_path / 'child.sock', tmp_path / 'broker.sock'
    seen = []

    def predict(request):
        seen.append(request)
        raise ValueError('child failed')

    pool = DaemonPool([child], on_event=lambda *args, **kwargs: None)
    with ThreadPoolExecutor(2) as workers:
        child_owner = workers.submit(serve, predict, child, 1)
        _ready(child)
        broker_owner = workers.submit(serve, pool, broker, 2)
        _ready(broker)
        try:
            request = {'model': 'alphafold3', 'identity': {'case': 2 if mismatch else 1}}
            frame = {_OPAQUE: 1, 'kind': 'predict', 'model': 'alphafold3',
                     'operation': None, 'identity': {'case': 1}, 'root_request_id': None,
                     'body': pickle.dumps(request, protocol=5)}
            with pytest.raises(RuntimeError, match='header/body mismatch' if mismatch else 'child failed'):
                _opaque_exchange(broker, frame)
            with pytest.raises(RuntimeError, match='quarantined'):
                call(broker, {'model': 'alphafold3', 'identity': {'case': 3}}, opaque=True)
            snapshot = call(broker, {'type': 'describe_snapshot'})
            assert snapshot['quarantined'] and snapshot['completed'] == [0]
            assert snapshot['failed'] == [1] and len(seen) == (0 if mismatch else 1)
        finally:
            call(broker, {'type': 'shutdown'})
            broker_owner.result(5)
            call(child, {'type': 'shutdown'})
            child_owner.result(5)

def test_two_daemons_work_before_either_finishes_and_release_on_failure():
    release=threading.Event();drain=threading.Event();both=threading.Event();seen=[];lock=threading.Lock()
    def invoke(backend,request):
        with lock:
            seen.append((backend,request['seed']))
            if len(seen)==2:both.set()
        if request.get('fail'):
            assert release.wait(3)
            raise ValueError('injected daemon failure')
        assert drain.wait(3)
        return request['seed']
    pool=DaemonPool(['a','b'],invoke=invoke,on_event=lambda *a,**k:None)
    with ThreadPoolExecutor(3) as executor:
        first=executor.submit(pool.predict,{'seed':7});second=executor.submit(pool.predict,{'seed':11,'fail':True})
        try:
            assert both.wait(2) and len({x[0] for x in seen})==2
            assert pool.pending==[1,1] and not first.done() and not second.done()
            release.set()
            with pytest.raises(ValueError):second.result(timeout=2)
            assert not first.done()  # Failure fences new work, not an active sibling.
            with pytest.raises(RuntimeError,match='quarantined'):pool.predict({'seed':13})
        finally:release.set();drain.set()
        assert first.result(timeout=2)==7
    assert pool.pending==[0,0] and sum(pool.failed)==1 and sum(pool.completed)==1
    assert len(seen)==2  # Failed inference was not retried on another backend.

@pytest.mark.parametrize('capacity',[1,2])
def test_round_robin_ties_and_logging_failure_do_not_change_requests(capacity):
    seen=[]
    def record(backend,request):seen.append((backend,request));return request
    def broken(*a,**k):raise OSError('log unavailable')
    pool=DaemonPool(['a','b','c'],capacity=capacity,invoke=record,on_event=broken)
    for seed in range(6):
        request={'model':'alphafold3','seed':seed,'identity':{'candidate':seed}}
        assert pool.predict(request) is request
    assert [Path(p).name for p,r in seen]==['a','b','c']*2 and pool.pending==[0,0,0]
    assert pool.describe_snapshot()['capacity_per_owner']==capacity

@pytest.mark.parametrize('backends,capacity,adaptive',[([],1,False),(['a','a'],1,False),(['a'],0,False),(['a'],2,True)])
def test_invalid_topology(backends,capacity,adaptive):
    from types import SimpleNamespace
    controller=SimpleNamespace(settings=SimpleNamespace(max_concurrency=len(backends))) if adaptive else None
    with pytest.raises(ValueError):DaemonPool(backends,capacity,controller=controller)

@pytest.mark.parametrize('count',[2,3])
def test_real_rpc_pool_uses_all_daemons_and_drains(tmp_path,count):
    sockets=[tmp_path/f'af3_{i}.sock' for i in range(count)];broker=tmp_path/'pool.sock';both=threading.Barrier(count);completed=[]
    def handler(index):
        def predict(request):
            both.wait(timeout=3);completed.append(index);return {'seed':request['seed'],'backend':index}
        return predict
    with ExitStack() as stack:
        workers=stack.enter_context(ThreadPoolExecutor(2*count+1))
        owners=[workers.submit(serve,handler(i),socket,max_pending=1) for i,socket in enumerate(sockets)]
        deadline=time.monotonic()+3
        while not all(p.exists() for p in sockets) and time.monotonic()<deadline:time.sleep(.005)
        owner=workers.submit(serve_pool,sockets,broker,max_pending=count)
        while not broker.exists() and time.monotonic()<deadline:time.sleep(.005)
        try:
            results=[workers.submit(call,broker,dict(model='alphafold3',seed=seed,identity={'seed':seed})) for seed in [7,11,13][:count]]
            values=[r.result(timeout=5) for r in results]
            assert {v['backend'] for v in values}==set(range(count)) and {v['seed'] for v in values}==set([7,11,13][:count])
        finally:
            call(broker,{'type':'shutdown'});owner.result(timeout=5)
            for socket in sockets:call(socket,{'type':'shutdown'})
            for future in owners:future.result(timeout=5)
    assert sorted(completed)==list(range(count)) and not broker.exists()

def test_imports_do_not_load_model_runtimes():
    subprocess.run([sys.executable,'-c','import sys;import astrolabe.evaluators.af3_pool;assert not any(x in sys.modules for x in ["torch","jax","numpy"])'],check=True,timeout=5)


@pytest.mark.parametrize('startup_failure',[False,True])
def test_lazy_expansion_keeps_resident_owner_available_and_never_replays(monkeypatch,startup_failure):
    from astrolabe.runtime import evaluation
    monkeypatch.delenv('ASTROLABE_RESOURCE_SOCKET',raising=False)
    monkeypatch.setattr(evaluation.EvaluatorInferController,'start',lambda self:None)
    starting=threading.Event();release=threading.Event();seen=[];loads=[]
    def load(index):
        loads.append(index);starting.set()
        assert release.wait(5)
        if startup_failure:raise RuntimeError('startup failed')
        return dict(index=index,socket=str(Path('b').resolve()),pid=2)
    def invoke(backend,request):
        seen.append((Path(backend).name,request['seed']))
        return request['seed']
    pool=process_pool_handler(['a','deferred-b','deferred-c'],
        adaptive=dict(enabled=True,max_concurrency=3),owners=[{}, {}, {}],
        backend_loader=load,invoke=invoke,on_event=lambda *a,**kw:None)
    try:
        assert pool.describe_snapshot()['resident_owner_count']==1
        assert pool.predict({'seed':0})==0 and loads==[]
        pool.controller.limit=2
        with ThreadPoolExecutor(2) as threads:
            expanding=threads.submit(pool.predict,{'seed':1})
            try:
                assert starting.wait(3) and pool.controller.warming==1
                # Capacity2 is not reported as two loaded models during startup.
                assert pool.describe_snapshot()['resident_owner_count']==1
                assert threads.submit(pool.predict,{'seed':2}).result(3)==2
            finally:release.set()
            if startup_failure:
                with pytest.raises(RuntimeError,match='startup failed'):expanding.result(3)
                with pytest.raises(RuntimeError,match='quarantined'):pool.predict({'seed':3})
            else:
                assert expanding.result(3)==1
                assert pool.describe_snapshot()['resident_owner_count']==2
        assert loads==[1] and pool.controller.warming==0
        assert seen==[('a',0),('a',2)]+([] if startup_failure else [('b',1)])
    finally:release.set();pool.close()


def test_aliases_are_not_independent_capacity_slots(tmp_path):
    socket=tmp_path/'a.sock'
    with pytest.raises(ValueError):DaemonPool([str(socket),str(tmp_path)+'/./a.sock'])

def test_broker_cannot_forward_to_itself(tmp_path):
    socket=tmp_path/'pool.sock'
    with pytest.raises(ValueError,match='itself'):serve_pool([socket],socket)
    assert not socket.exists()


def test_process_pool_fixed_owner_adaptive_distinct_and_drained_control(monkeypatch):
    from astrolabe.runtime import evaluation
    monkeypatch.delenv('ASTROLABE_RESOURCE_SOCKET',raising=False)
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
    now=[0.0];events=[];seen=[];lock=threading.Lock();expanded=threading.Event()
    entered={i:threading.Event() for i in range(4)}
    release={i:threading.Event() for i in range(4)}
    cls=evaluation.EvaluatorInferController
    monkeypatch.setattr(cls,'start',lambda self:None)
    monkeypatch.setattr(evaluation,'EvaluatorInferController',
                        lambda settings,**kw:cls(settings,clock=lambda:now[0],**kw))
    def invoke(backend,request):
        with lock:seen.append((Path(backend).name,request['seed']))
        if request.get('block'):
            entered[request['seed']].set()
            if request['seed'] in (2,3):expanded.set()
            assert release[request['seed']].wait(5)
        return request['seed']
    handler=process_pool_handler(['a','b','c'],adaptive=dict(enabled=True,max_concurrency=3,
        memory_per_request_bytes=100,memory_reserve_bytes=10),invoke=invoke,
        on_event=lambda event,**fields:events.append((event,fields)))
    controller=handler.controller
    sample=[dict(index=0,uuid='GPU-pool-test',total_bytes=1000,free_bytes=900,
                 utilization=10,processes=[dict(pid=123,used_bytes=100)])]
    def observe():
        now[0]+=10
        controller.observe(sample,sampled_at=now[0])
    try:
        for enabled in (False,True):
            assert handler.adaptive_control(enabled=enabled,reset=True)['limit']==1
            assert handler.cursor==0
            assert handler.predict({'seed':0})==0
            assert handler.predict({'seed':0})==0
            assert seen[-2:]==[('a',0),('a',0)]
            with ThreadPoolExecutor(3) as workers:
                expanded.clear()
                for i in (1,2,3):entered[i].clear();release[i].clear()
                futures=[workers.submit(handler.predict,{'seed':1,'block':True})]
                try:
                    assert entered[1].wait(3)
                    futures.extend(workers.submit(handler.predict,{'seed':i,'block':True}) for i in (2,3))
                    with controller._cv:assert controller._cv.wait_for(lambda:controller.ready==2,timeout=3)
                    with pytest.raises(RuntimeError,match='drain'):handler.adaptive_control(enabled=not enabled,reset=True)
                    observe()
                    if enabled:
                        assert expanded.wait(3)
                        assert sum(entered[i].is_set() for i in (2,3))==1
                        observe()
                        assert entered[2].wait(3) and entered[3].wait(3)
                        assert handler.pending==[1,1,1]
                    else:
                        assert controller.limit==1 and not entered[2].is_set()
                finally:
                    for event in release.values():event.set()
                assert [f.result(3) for f in futures]==[1,2,3]
            assert handler.pending==[0,0,0]
            if enabled:assert {b for b,s in seen[-3:]}=={'a','b','c'}
            else:assert {b for b,s in seen[-3:]}=={'a'}
        samples=[fields for event,fields in events if event=='adaptive_sample']
        assert samples and all(s['process_ids']==[123] and s['gpu_uuid']=='GPU-pool-test' for s in samples)
        assert all(s['stage']=='evaluation_gate' for s in samples)
        assert not any(event.startswith('gpu_') for event,_ in events)
    finally:
        for event in release.values():event.set()
        handler.close()


def test_process_pool_waits_for_first_completion_and_never_replays_unknown_failure(monkeypatch):
    from astrolabe.runtime import evaluation
    monkeypatch.delenv('ASTROLABE_RESOURCE_SOCKET',raising=False)
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
    cls=evaluation.EvaluatorInferController;now=[0.0]
    monkeypatch.setattr(cls,'start',lambda self:None)
    monkeypatch.setattr(evaluation,'EvaluatorInferController',lambda settings,**kw:cls(settings,clock=lambda:now[0],**kw))
    entered=threading.Event();release=threading.Event();seen=[]
    def invoke(backend,request):
        seen.append(backend);entered.set();assert release.wait(5)
        raise EOFError('outcome unknown')
    handler=process_pool_handler(['a','b','c'],adaptive=dict(enabled=True,max_concurrency=3,
        memory_per_request_bytes=100,memory_reserve_bytes=10),invoke=invoke,on_event=lambda *a,**kw:None)
    try:
        with ThreadPoolExecutor(2) as workers:
            first=workers.submit(handler.predict,{'seed':7})
            assert entered.wait(3)
            second=workers.submit(handler.predict,{'seed':11})
            try:
                with handler.controller._cv:assert handler.controller._cv.wait_for(lambda:handler.controller.ready==1,timeout=3)
                now[0]=10
                handler.controller.observe([dict(index=0,uuid='GPU-pool-test',total_bytes=1000,
                    free_bytes=900,utilization=10,processes=[])],sampled_at=10)
                assert handler.controller.limit==1 and handler.controller.completed==0 and len(seen)==1
            finally:release.set()
            with pytest.raises(EOFError):first.result(3)
            with pytest.raises(RuntimeError,match='quarantined'):second.result(3)
        with pytest.raises(RuntimeError,match='quarantined'):handler.adaptive_control(enabled=False,reset=True)
        assert handler.pending==[0,0,0] and sum(handler.failed)==1 and len(seen)==1
    finally:release.set();handler.close()


def test_process_pool_rpc_control_requires_empty_pending_and_shutdown_drains(tmp_path,monkeypatch):
    from astrolabe.runtime import evaluation
    monkeypatch.delenv('ASTROLABE_RESOURCE_SOCKET',raising=False)
    monkeypatch.setattr(evaluation.EvaluatorInferController,'start',lambda self:None)
    socket=tmp_path/'pool.sock';entered=threading.Event();release=threading.Event();seen=[]
    def invoke(backend,payload):
        seen.append((backend,payload['request_id'],payload['identity']))
        entered.set();assert release.wait(5)
        return payload['seed']
    handler=process_pool_handler(['a','b','c'],adaptive=dict(enabled=True,max_concurrency=3),
                                 invoke=invoke,on_event=lambda *a,**kw:None)
    with ThreadPoolExecutor(4) as workers:
        owner=workers.submit(serve,handler,socket,4)
        deadline=time.monotonic()+3
        while not socket.exists() and not owner.done() and time.monotonic()<deadline:time.sleep(.005)
        shutdown=None
        try:
            assert call(socket,dict(type='adaptive_control',enabled=False,reset=True))['limit']==1
            first=workers.submit(call,socket,dict(model='alphafold3',seed=7,identity={'case':1}))
            assert entered.wait(3)
            second=workers.submit(call,socket,dict(model='alphafold3',seed=11,identity={'case':2}))
            with handler.controller._cv:assert handler.controller._cv.wait_for(lambda:handler.controller.ready==1,timeout=3)
            snapshot=call(socket,{'type':'describe_snapshot'})
            assert snapshot['pending']==2 and snapshot['resident_owner_count']==3
            assert snapshot['runtime_lifecycle']['state']=='busy'
            assert snapshot['runtime_lifecycle']['loader_count']==0 and not snapshot['runtime_lifecycle']['weights_loaded']
            assert snapshot['pool_pending']==[1,0,0] and snapshot['adaptive']['limit']==1
            with pytest.raises(RuntimeError,match='pending=0'):call(socket,dict(type='adaptive_control',enabled=True,reset=True))
            shutdown=workers.submit(call,socket,{'type':'shutdown'})
            assert not shutdown.done() and not first.done() and not second.done()
        finally:
            release.set()
            if shutdown is None:shutdown=workers.submit(call,socket,{'type':'shutdown'})
            assert shutdown.result(5) is None
            owner.result(5)
        assert first.result(3)==7 and second.result(3)==11
    assert [Path(b).name for b,_,_ in seen]==['a','a']
    assert [i for _,i,_ in seen]==[1,2] and [identity for _,_,identity in seen]==[{'case':1},{'case':2}]
    assert handler.closed and handler.pending==[0,0,0]
