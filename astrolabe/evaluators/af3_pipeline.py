"""AF3 assembly using the existing GH stage chain and one immutable model owner."""
from concurrent.futures import Future,ThreadPoolExecutor
from contextlib import ExitStack
from astrolabe.inference_pipeline import Pending,PipelineExecutor,Stage
from astrolabe.evaluators.inference_service import mark,serve_staged


def stage_factory(runner, config):
    from astrolabe.evaluators.wrappers.alphafold3 import service_model_settings
    settings = service_model_settings(config)
    global_config = runner._model_config.global_config
    cyclic_ids = tuple(int(x) for x in getattr(global_config,'cyclic_asym_ids',()))
    cyclic_patches = tuple(tuple(int(y) for y in x) for x in getattr(global_config,'cyclic_relative_patches',()))

    def stages(request):
        if request.get('model') != 'alphafold3':
            raise ValueError('Wrong model for this AlphaFold3 service')
        if request['settings'] != settings:
            raise ValueError('Shared AlphaFold3 model settings differ from owner')
        if tuple(request['cyclic_asym_ids']) != cyclic_ids or tuple(tuple(x) for x in request['cyclic_relative_patches']) != cyclic_patches:
            raise ValueError('Staged AF3 requires the immutable owner cyclic configuration')
        native = runner.inference_stages(request['rng_key'])
        result = [Stage('af3_request','cpu',lambda item: dict(item,value=request['features']))]
        for native_stage in native:
            def run(item, stage=native_stage):
                fields = dict(request_id=item['request_id'],identity=item['payload'].get('identity'))
                is_model = stage.name == 'af3_model'
                if is_model:mark('gpu_start',**fields)
                try:
                    output = stage.run(item['value'])
                except BaseException:
                    if is_model:mark('gpu_end',ok=False,**fields)
                    raise
                if isinstance(output,Pending):
                    item['value'] = output.value
                    def complete(pending=output):
                        ok = False
                        try:
                            if isinstance(pending.completion,Future):pending.completion.result()
                            else:pending.completion()
                            ok = True
                        finally:
                            if is_model:mark('gpu_end',ok=ok,**fields)
                    return Pending(item,complete,output.keepalive)
                item['value'] = output
                if is_model:mark('gpu_end',ok=True,**fields)
                return item
            result.append(Stage(native_stage.name,native_stage.executor,run))
        return tuple(result)
    return stages


def serve_pipeline(runner, config, socket_path, max_pending=4):
    with ExitStack() as stack:
        pools = {name:stack.enter_context(ThreadPoolExecutor(max_workers=count,
                    thread_name_prefix='af3-'+name))
                 for name,count in [('cpu',2),('transfer',1),('gpu',1),('io',2)]}
        pipeline = stack.enter_context(PipelineExecutor(pools,max_pending=max_pending))
        serve_staged(stage_factory(runner,config),socket_path,pipeline,max_pending=max_pending)
