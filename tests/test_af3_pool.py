from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
import subprocess,sys,threading,time
import pytest
from astrolabe.evaluators.af3_pool import DaemonPool,serve_pool
from astrolabe.evaluators.inference_service import call,serve

def test_two_daemons_work_before_either_finishes_and_release_on_failure():
    release=threading.Event();both=threading.Event();seen=[];lock=threading.Lock()
    def invoke(backend,request):
        with lock:
            seen.append((backend,request['seed']))
            if len(seen)==2:both.set()
        assert release.wait(3)
        if request.get('fail'):raise ValueError('injected daemon failure')
        return request['seed']
    pool=DaemonPool(['a','b'],invoke=invoke,on_event=lambda *a,**k:None)
    with ThreadPoolExecutor(3) as executor:
        first=executor.submit(pool.predict,{'seed':7});second=executor.submit(pool.predict,{'seed':11,'fail':True})
        try:
            assert both.wait(2) and len({x[0] for x in seen})==2
            assert pool.pending==[1,1] and not first.done() and not second.done()
            third=executor.submit(pool.predict,{'seed':13});time.sleep(.05);assert len(seen)==2
        finally:release.set()
        assert first.result(timeout=2)==7
        with pytest.raises(ValueError):second.result(timeout=2)
        assert third.result(timeout=2)==13
    assert pool.pending==[0,0] and sum(pool.failed)==1 and sum(pool.completed)==2
    assert len(seen)==3  # Failed inference was not retried on another backend.

def test_round_robin_ties_and_logging_failure_do_not_change_requests():
    seen=[]
    def record(backend,request):seen.append((backend,request));return request
    def broken(*a,**k):raise OSError('log unavailable')
    pool=DaemonPool(['a','b','c'],invoke=record,on_event=broken)
    for seed in range(6):
        request={'model':'alphafold3','seed':seed,'identity':{'candidate':seed}}
        assert pool.predict(request) is request
    assert [Path(p).name for p,r in seen]==['a','b','c']*2 and pool.pending==[0,0,0]

@pytest.mark.parametrize('backends,capacity',[([],1),(['a','a'],1),(['a'],0)])
def test_invalid_topology(backends,capacity):
    with pytest.raises(ValueError):DaemonPool(backends,capacity)

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


def test_aliases_are_not_independent_capacity_slots(tmp_path):
    socket=tmp_path/'a.sock'
    with pytest.raises(ValueError):DaemonPool([str(socket),str(tmp_path)+'/./a.sock'])

def test_broker_cannot_forward_to_itself(tmp_path):
    socket=tmp_path/'pool.sock'
    with pytest.raises(ValueError,match='itself'):serve_pool([socket],socket)
    assert not socket.exists()
