"""Bounded pool of independently resident AF3 daemons; no numeric/model changes."""
from contextlib import nullcontext
import argparse
import os
from pathlib import Path
import threading
import time
from astrolabe.evaluators.inference_service import call,json_call,mark,serve,_OPAQUE,_opaque_exchange

class DaemonPool:
    opaque_proxy=True
    # This flag bypasses only the RPC envelope lock. There is no runner here;
    # each child still serializes its entire prepare/model/finalize request.
    request_local_rng=True

    def __init__(self,backends,capacity=1,*,invoke=call,on_event=mark,controller=None,owners=(),backend_loader=None):
        self.backends=tuple(str(Path(path).resolve()) for path in backends)
        if not self.backends or len(set(self.backends))!=len(self.backends):raise ValueError('Distinct AF3 backends required')
        if capacity<1:raise ValueError('Positive per-daemon capacity required')
        if controller is not None and capacity!=1:
            raise ValueError('Adaptive AF3 process isolation requires per-daemon capacity 1')
        if controller is not None and controller.settings.max_concurrency!=len(self.backends):
            raise ValueError('Adaptive capacity must match the resident child count')
        self.capacity=capacity;self.invoke=invoke;self.on_event=on_event
        self.controller=controller;self.owners=list(owners)
        if backend_loader is not None and (controller is None or len(self.owners)!=len(self.backends)):
            raise ValueError('Lazy backends require an adaptive controller and one owner record per slot')
        self.backend_loader=backend_loader
        self.resident=[backend_loader is None or i==0 for i in range(len(self.backends))]
        self.pending=[0]*len(self.backends);self.completed=[0]*len(self.backends);self.failed=[0]*len(self.backends)
        self.cursor=0;self.condition=threading.Condition()
        self.requests=0;self.quarantined=False;self.closed=False
    def emit(self,event,**fields):
        try:self.on_event(event,**fields)
        except Exception:pass  # Logging cannot lose a charged inference or a capacity slot.
    def predict(self,request):
        queued=time.time()
        with self.condition:
            if self.closed or self.quarantined:raise RuntimeError('AF3 pool is quarantined or stopping; request not forwarded')
            self.requests+=1
        try:
            gate=self.controller.inference_slot() if self.controller is not None else nullcontext()
            with gate:
                with self.condition:
                    def available():
                        count=(self.controller.limit if self.controller.enabled else 1) if self.controller is not None else len(self.backends)
                        return [i for i in range(count) if self.pending[i]<self.capacity]
                    self.condition.wait_for(lambda:self.quarantined or bool(available()))
                    if self.quarantined:raise RuntimeError('AF3 pool is quarantined; request not forwarded')
                    index=min(available(),key=lambda i:(self.pending[i],(i-self.cursor)%len(self.backends)))
                    self.pending[index]+=1;self.cursor=(index+1)%len(self.backends)
                    current=list(self.pending)
                fields=dict(backend=index,socket=self.backends[index],identity=request.get('identity'),
                            request_id=request.get('root_request_id',request.get('request_id')),stage='evaluation_rpc')
                self.emit('pool_dispatch',queue_s=time.time()-queued,pending=current,**fields)
                started=time.monotonic();ok=False
                warming=not self.resident[index]
                try:
                    if warming:
                        # This request owns the new serial slot. Other requests
                        # can keep using an already resident owner during startup.
                        with self.controller._cv:
                            self.controller.warming+=1
                        self.emit('pool_expansion_start',backend=index)
                        loaded=self.backend_loader(index)
                        with self.condition:
                            backends=list(self.backends);backends[index]=loaded['socket']
                            self.backends=tuple(backends)
                            self.owners[index]=loaded
                            self.resident[index]=True
                        fields['socket']=self.backends[index]
                        self.emit('pool_expansion_ready',backend=index,owner=loaded)
                    # Exactly one forward attempt, including ambiguous OOM/EOF.
                    result=(_opaque_exchange(self.backends[index],request) if _OPAQUE in request
                            else self.invoke(self.backends[index],request))
                    ok=True;return result
                except BaseException:
                    with self.condition:
                        self.quarantined=True
                        if self.controller is not None:self.controller.quarantine()
                        self.condition.notify_all()
                    self.emit('pool_quarantined',**fields)
                    raise
                finally:
                    if warming:
                        with self.controller._cv:
                            self.controller.warming-=1
                            self.controller._cv.notify_all()
                    with self.condition:
                        self.pending[index]-=1
                        (self.completed if ok else self.failed)[index]+=1
                        self.condition.notify_all()
                    self.emit('pool_complete',ok=ok,rpc_elapsed_s=time.monotonic()-started,**fields)
        finally:
            with self.condition:
                self.requests-=1
                self.condition.notify_all()

    def __call__(self,request):
        if self.invoke is json_call:
            from astrolabe.evaluators.af3_buffers import check_reference
            check_reference(request)
        if request.get('model')!='alphafold3' and request.get('operation')!='extract_structures':
            raise ValueError('AF3 pool accepts only AF3 predictions/extraction')
        return self.predict(request)

    def adaptive_control(self,*,enabled,reset=False):
        if self.controller is None:raise RuntimeError('adaptive_control requires an adaptive process pool')
        with self.condition:
            if self.requests:raise RuntimeError('adaptive_control requires pending=0 and a drained pool')
            if self.closed or self.quarantined:raise RuntimeError('AF3 pool is quarantined or stopping')
            result=self.controller.control(enabled=enabled,reset=reset)
            if reset or not enabled:self.cursor=0
            return result

    def describe_snapshot(self):
        with self.condition:
            snapshot=dict(backend='alphafold3',execution_mode='independent_process_pool',
                          admission_scope='complete_evaluator_request',inner_infer_source='child gpu_* stage=infer',
                          resident_owner_count=sum(self.resident),max_owner_count=len(self.backends),capacity_per_owner=self.capacity,
                          backends=[path if ready else None for path,ready in zip(self.backends,self.resident)],
                          owners=list(self.owners),cursor=self.cursor,
                          pool_pending=list(self.pending),completed=list(self.completed),failed=list(self.failed),
                          quarantined=self.quarantined,active_requests=self.requests,closed=self.closed,
                          concurrency_certified=False)
            if self.controller is not None:snapshot['adaptive']=self.controller.snapshot()
            snapshot['runtime_lifecycle']=dict(
                backend='alphafold3',pid=os.getpid(),residency='forwarding_only',
                state=('stopped' if self.closed else 'quarantined' if self.quarantined else
                       'busy' if self.requests else 'ready'),
                loader_count=0,weights_loaded=False,gpu_resident=False,
                active_requests=self.requests,
                device_execution_slots=snapshot['adaptive']['limit'] if self.controller is not None else len(self.backends))
            return snapshot

    def close(self):
        with self.condition:
            if self.closed:return
            self.closed=True
            self.condition.wait_for(lambda:self.requests==0)
        if self.controller is not None:self.controller.close()
        self.emit('pool_drained',completed=self.completed,failed=self.failed,pending=self.pending,
                  stage='evaluation_rpc',quarantined=self.quarantined)


def process_pool_handler(backends,*,adaptive,device='cuda:0',owners=(),invoke=call,on_event=mark,backend_loader=None):
    from astrolabe.runtime.evaluation import EvaluatorInferController,require_private_evaluator_owner
    require_private_evaluator_owner(True)
    def emit(event,**fields):on_event(event,stage='evaluation_gate',admission_scope='complete_evaluator_request',**fields)
    controller=EvaluatorInferController(adaptive,device=device,emit=emit)
    pool=DaemonPool(backends,invoke=invoke,on_event=on_event,controller=controller,owners=owners,backend_loader=backend_loader)
    controller.start()
    pool.emit('pool_starting',**pool.describe_snapshot())
    return pool

def serve_pool(backends,socket_path,*,max_pending=6,capacity=1):
    pool=DaemonPool(backends,capacity)
    if str(Path(socket_path).resolve()) in pool.backends:raise ValueError('AF3 broker cannot forward to itself')
    try:
        mark('pool_starting',backends=pool.backends,capacity=capacity,max_pending=max_pending)
        serve(pool,socket_path,max_pending=max_pending)
    finally:pool.close()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--socket',required=True);p.add_argument('--backends',nargs='+',required=True)
    p.add_argument('--max-pending',type=int,default=6);p.add_argument('--capacity',type=int,default=1);a=p.parse_args()
    serve_pool(a.backends,a.socket,max_pending=a.max_pending,capacity=a.capacity)
if __name__=='__main__':main()
