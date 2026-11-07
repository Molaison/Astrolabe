"""One immutable AF3 owner and the native prepare / model / finalize boundary."""
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack

from astrolabe.inference_pipeline import Pending, PipelineExecutor, Stage
from astrolabe.evaluators.inference_service import mark, serve_staged


def validate_request(runner, request, settings=None):
    """Reject protocol/configuration errors before device admission."""
    if request.get("operation") == "extract_structures":
        return
    if request.get("model") != "alphafold3":
        raise ValueError("Wrong model for this AlphaFold3 service")
    if settings is not None and request["settings"] != settings:
        raise ValueError("Shared AlphaFold3 model settings differ from owner")
    owner = runner._model_config.global_config
    current = (tuple(int(x) for x in getattr(owner, "cyclic_asym_ids", ())),
               tuple(tuple(int(y) for y in x) for x in
                     getattr(owner, "cyclic_relative_patches", ())))
    bound = getattr(runner, "_inference_owner_settings", current)
    requested = (tuple(int(x) for x in request.get("cyclic_asym_ids", ())),
                 tuple(tuple(int(y) for y in x) for x in
                       request.get("cyclic_relative_patches", ())))
    if requested != bound or current != bound:
        raise ValueError("Staged AF3 requires the immutable owner cyclic configuration")


def split_inference_stages(runner, rng_key, *, target_name=None):
    """Group native stages without moving upload or materialization into infer."""
    from astrolabe.evaluators.runtime_stages import complete_stage

    kwargs = {} if target_name is None else {"target_name": target_name}
    native = tuple(runner.inference_stages(rng_key, **kwargs))
    models = [i for i, stage in enumerate(native) if stage.name == "af3_model"]
    if len(models) != 1:
        raise ValueError("AF3 requires exactly one complete af3_model stage")
    index = models[0]
    if not native[index + 1:]:
        raise ValueError("AF3 requires native output materialization after af3_model")

    def prepare(value):
        for stage in native[:index]:
            value = complete_stage(stage, value)
        return value

    def finalize(value):
        for stage in native[index + 1:]:
            value = complete_stage(stage, value)
        return value

    return (Stage("af3_prepare", "gpu", prepare), native[index],
            Stage("af3_finalize", "gpu", finalize))


def evaluator_stages(runner, config=None, *, allow_extraction=False, adaptive=None):
    """The shared implementation for local, native-daemon and RPC callers.

    Request RNG, native stages and buffers belong to that request. The model's
    cyclic configuration is fixed for this owner; mismatches never fall back to
    an unclassified, monolithic model call.
    """
    from astrolabe.model_execution import EvaluatorStages
    from astrolabe.evaluators.runtime_stages import complete_stage
    from astrolabe.runtime.evaluation import adaptive_settings

    adaptive = adaptive_settings(adaptive)
    resident = (hasattr(config, "get")
                and config.get("_service_settings", {}).get("concurrency_mode") == "resident")
    if adaptive.enabled and not resident:
        raise RuntimeError("Shared-runner AF3 adaptive concurrency is disabled; use OwnedAF3 process isolation")

    if config is not None:
        from astrolabe.evaluators.wrappers.alphafold3 import service_model_settings
        settings = service_model_settings(config)
    else:
        settings = None
    owner = runner._model_config.global_config
    cyclic_ids = tuple(int(x) for x in getattr(owner, "cyclic_asym_ids", ()))
    cyclic_patches = tuple(tuple(int(y) for y in x) for x in
                           getattr(owner, "cyclic_relative_patches", ()))
    # Native ModelRunner setters also enforce immutability after this binding.
    runner._inference_owner_settings = (cyclic_ids, cyclic_patches)

    def prepare(request):
        request = dict(request)
        if allow_extraction and request.get("operation") == "extract_structures":
            return {"extract": request}
        validate_request(runner, request, settings)
        if (adaptive.enabled or resident) and "rng_seed" not in request and "rng_key" not in request:
            raise ValueError("Concurrent AF3 requires an explicit per-request RNG seed/key")
        if "rng_seed" in request:
            import jax
            with jax.default_device(jax.local_devices(backend="cpu")[0]):
                request["rng_key"] = jax.random.PRNGKey(request["rng_seed"])
        stages = split_inference_stages(
            runner, request["rng_key"], target_name=request.get("target_name"))
        return {"stages": stages, "value": complete_stage(stages[0], request["features"]),
                "fields": {"request_id": request.get("request_id"),
                           "identity": request.get("identity")}}

    def execute(item):
        if "extract" in item:
            # Compatibility RPC: extraction is finalize work, not inference.
            return item
        fields = dict(item["fields"], stage="infer")
        mark("gpu_start", **fields)
        try:
            output = item["stages"][1].run(item["value"])
        except BaseException:
            mark("gpu_end", ok=False, **fields)
            raise
        if isinstance(output, Pending):
            item["value"] = output.value

            def complete():
                ok = False
                try:
                    if isinstance(output.completion, Future):
                        output.completion.result()
                    else:
                        output.completion()
                    ok = True
                finally:
                    mark("gpu_end", ok=ok, **fields)

            return Pending(item, complete, (output, output.keepalive))
        item["value"] = output
        mark("gpu_end", ok=True, **fields)
        return item

    def finalize(item):
        if "extract" in item:
            request = item["extract"]
            return runner.extract_structures(request["batch"], request["result"],
                                             request["target_name"])
        return complete_stage(item["stages"][2], item["value"])

    device_resource = "cpu" if getattr(getattr(runner, "_device", None), "platform", None) == "cpu" else "gpu"
    resource = (lambda request: "cpu" if request.get("operation") == "extract_structures"
                else device_resource) if allow_extraction else device_resource
    return EvaluatorStages(prepare, execute, finalize, resource=resource,
                           prepare_resource=device_resource, finalize_resource=device_resource, pipeline_slots=1,
                           request_local_rng=adaptive.enabled or resident, adaptive=adaptive)


def stage_factory(runner, config, *, admit_device=True, device_kernel=None):
    """RPC envelope only; the shared kernel owns all three admitted stages."""
    if not admit_device:
        raise ValueError("Use evaluator_stages inside an existing native owner")
    if device_kernel is None:
        from astrolabe.evaluators.af3_service import inference_handler
        device_kernel = inference_handler(runner, config)
    from astrolabe.evaluators.wrappers.alphafold3 import service_model_settings
    settings = service_model_settings(config)

    def stages(request):
        validate_request(runner, request, settings)
        def run(item):
            return dict(item, value=device_kernel(dict(request, request_id=item.get("request_id"))))
        # Waiting for the owner is not GPU dispatch work. Do not collapse the
        # native upload/model/materialize chain into an af3_device GPU stage.
        return (Stage("af3_evaluate", "cpu", run),)
    return stages


def serve_pipeline(runner, config, socket_path, max_pending=4, *, device_kernel=None):
    with ExitStack() as stack:
        pools = {name: stack.enter_context(ThreadPoolExecutor(
            max_workers=count, thread_name_prefix="af3-" + name))
            for name, count in [("cpu", 2), ("io", 2)]}
        pipeline = stack.enter_context(PipelineExecutor(pools, max_pending=max_pending))
        serve_staged(stage_factory(runner, config, device_kernel=device_kernel),
                     socket_path, pipeline, max_pending=max_pending)
