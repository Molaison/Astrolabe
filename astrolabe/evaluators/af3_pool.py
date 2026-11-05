"""Bounded pool of independently resident AF3 daemons; no numeric/model changes."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import argparse
from pathlib import Path
import threading
import time
from astrolabe.evaluators.inference_service import call,mark,serve_staged
from astrolabe.inference_pipeline import PipelineExecutor,Stage

class DaemonPool:
    def __init__(self,backends,capacity=1,*,invoke=call,on_event=mark):
        self.backends=tuple(str(Path(path).resolve()) for path in backends)
        if not self.backends or len(set(self.backends))!=len(self.backends):raise ValueError('Distinct AF3 backends required')
        if capacity<1:raise ValueError('Positive per-daemon capacity required')
        self.capacity=capacity;self.invoke=invoke;self.on_event=on_event
        self.pending=[0]*len(self.backends);self.completed=[0]*len(self.backends);self.failed=[0]*len(self.backends)
        self.cursor=0;self.condition=threading.Condition()
    def emit(self,event,**fields):
        try:self.on_event(event,**fields)
        except Exception:pass  # Logging cannot lose a charged inference or a capacity slot.
    def predict(self,request):
        queued=time.time()
        with self.condition:
            self.condition.wait_for(lambda:any(n<self.capacity for n in self.pending))
            available=[i for i,n in enumerate(self.pending) if n<self.capacity]
            index=min(available,key=lambda i:(self.pending[i],(i-self.cursor)%len(self.backends)))
            self.pending[index]+=1;self.cursor=(index+1)%len(self.backends)
            current=list(self.pending)
        fields=dict(backend=index,socket=self.backends[index],identity=request.get('identity'))
        self.emit('pool_dispatch',queue_s=time.time()-queued,pending=current,**fields)
        ok=False
        try:
            # Exactly one forward attempt. Ambiguous failures are never replayed.
            result=self.invoke(self.backends[index],request);ok=True;return result
        finally:
            with self.condition:
                self.pending[index]-=1
                (self.completed if ok else self.failed)[index]+=1
                self.condition.notify_all()
            self.emit('pool_complete',ok=ok,**fields)

def serve_pool(backends,socket_path,*,max_pending=6,capacity=1):
    pool=DaemonPool(backends,capacity)
    if str(Path(socket_path).resolve()) in pool.backends:raise ValueError('AF3 broker cannot forward to itself')
    def factory(request):
        if request.get('model')!='alphafold3':raise ValueError('AF3 pool accepts only AF3 predictions')
        def forward(item):item['value']=pool.predict(item['payload']);return item
        return (Stage('af3_daemon_forward','forward',forward),)
    with ExitStack() as stack:
        resources={name:stack.enter_context(ThreadPoolExecutor(max_workers=count,thread_name_prefix='af3-pool-'+name))
                   for name,count in [('forward',max_pending),('io',2)]}
        pipeline=stack.enter_context(PipelineExecutor(resources,max_pending=max_pending))
        mark('pool_starting',backends=pool.backends,capacity=capacity,max_pending=max_pending)
        serve_staged(factory,socket_path,pipeline,max_pending=max_pending)
    mark('pool_drained',completed=pool.completed,failed=pool.failed,pending=pool.pending)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--socket',required=True);p.add_argument('--backends',nargs='+',required=True)
    p.add_argument('--max-pending',type=int,default=6);p.add_argument('--capacity',type=int,default=1);a=p.parse_args()
    serve_pool(a.backends,a.socket,max_pending=a.max_pending,capacity=a.capacity)
if __name__=='__main__':main()
