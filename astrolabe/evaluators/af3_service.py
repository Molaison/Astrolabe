"""AF3 serial model owner or adaptive independent-process pool owner.

python -m astrolabe.evaluators.af3_service --socket /af3-service/model.sock \
    --config /path/to/af3_conf.yaml

All clients use the same model paths/settings. Stop after workers finish using
inference_service.call(socket, {"type": "shutdown"}); accepted work drains first.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import sys

from astrolabe.evaluators.inference_service import mark, serve


def resident_mode(config):
    from astrolabe.evaluators.af3_lifecycle import service_settings
    raw = config.get("_service_settings") if hasattr(config, "get") else None
    return service_settings({}, explicit_settings=raw).concurrency_mode == "resident"


def load_service_config(path):
    """Use the same inherited AF3 settings as clients, or a legacy mapping."""
    from omegaconf import OmegaConf
    config = OmegaConf.load(path)
    if "defaults" in config or "evaluator" in config:
        from astrolabe.configuration import load_composed_config as load_config
        from astrolabe.evaluators.af3_lifecycle import service_settings
        task = load_config(path)
        adaptive = service_settings(task).adaptive
        lifecycle = asdict(service_settings(task))
        config = task.evaluator.alphafold3
        config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        # This config also serves CPU clients. Only the owner copy must use
        # the local model; keep the original client transport untouched.
        config.service_socket = None
        config._service_adaptive = asdict(adaptive)
        config._service_settings = lifecycle
        return config
    return OmegaConf.create(OmegaConf.to_container(config, resolve=True))


def inference_handler(runner, config, *, device_kernel=None):
    """Adopt a ready runner using the same stages as the native daemon."""
    from astrolabe.runtime.evaluation import adaptive_settings, require_private_evaluator_owner

    adaptive = adaptive_settings(config.get("_service_adaptive") if hasattr(config, "get") else None)
    resident = resident_mode(config)
    require_private_evaluator_owner(adaptive.enabled or resident)
    if adaptive.enabled and not resident:
        raise RuntimeError("Shared-runner AF3 adaptive concurrency is disabled; use OwnedAF3 process isolation")
    from astrolabe.evaluators.af3_pipeline import evaluator_stages, validate_request
    from astrolabe.evaluators.runtime_stages import DirectKernel, configured_device_hint
    from astrolabe.evaluators.wrappers.alphafold3 import service_model_settings

    settings = service_model_settings(config)
    kernel = device_kernel if device_kernel is not None else DirectKernel({}, lambda settings: (
        evaluator_stages(runner, config, allow_extraction=True, adaptive=adaptive),
        {"backend": "alphafold3", "device": str(runner._device)},
    ), lifecycle={
        "backend": "alphafold3", "residency": "resident_model",
        "device": configured_device_hint(config, default="cuda:0"),
        "weights_loaded": True, "concurrency_certified": False,
        "request_local_rng": adaptive.enabled or resident,
    })

    def predict(request):
        validate_request(runner, request, settings)
        return kernel(request)

    predict.request_local_rng = kernel.request_local_rng
    predict.describe_snapshot = kernel.describe_snapshot
    predict.adaptive_control = kernel.adaptive_control
    predict.close = kernel.close
    return predict


def load_runner(config):
    import os
    from astrolabe.runtime.pjrt_cache_hotfix import validate_pjrt_hotfix
    validate_pjrt_hotfix(os.environ)
    from astrolabe.runtime.evaluation import adaptive_settings, require_private_evaluator_owner
    adaptive = adaptive_settings(config.get("_service_adaptive") if hasattr(config, "get") else None)
    resident = resident_mode(config)
    require_private_evaluator_owner(adaptive.enabled or resident)
    if adaptive.enabled and not resident:
        raise RuntimeError("Adaptive AF3 parent must not load a runner; use OwnedAF3 process isolation")
    import jax
    import jaxlib
    from astrolabe.evaluators.wrappers.alphafold3 import alphafold3_setup

    if getattr(config, "service_socket", None):
        raise ValueError("Owner config must not itself select a remote service_socket")
    # Unlike the legacy local path, an explicit GPU owner never falls back to CPU.
    devices = jax.local_devices(backend="gpu")
    if not devices or devices[0].platform != "gpu":
        raise RuntimeError("AlphaFold3 service requires a real JAX GPU device")
    runner = alphafold3_setup(config)
    if runner._device.platform != "gpu":
        raise RuntimeError("AlphaFold3 service requires a real JAX GPU device")
    mark("model_load_start", device=str(runner._device),
         jax_version=jax.__version__, jaxlib_version=jaxlib.__version__,
         gpu_plugin_module=getattr(sys.modules.get("jax_plugins.xla_cuda12"), "__file__", None),
         cache_dir=str(getattr(config, "jax_compilation_cache_dir", None)))
    runner.model_params
    runner._model
    if resident:
        runner.enable_resident_execution()
    mark("model_load_end", device=str(runner._device))
    return runner


def main(argv=None):
    from astrolabe.evaluators.af3_lifecycle import guard_owner_parent
    guard_owner_parent()
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--config", required=True,
                        help="Task/case YAML (including defaults) or a resolved AF3 parameter mapping")
    parser.add_argument("--pipeline", action="store_true", help="Use the existing multi-stage asynchronous AF3 path")
    parser.add_argument("--max-pending", type=int, default=4)
    args = parser.parse_args(argv)
    config = load_service_config(args.config)
    resident = resident_mode(config)
    transport = str(config.get("service_transport", "pickle"))
    if transport not in {"pickle", "buffers"}:
        raise ValueError("AF3 service_transport must be pickle or buffers")
    if transport == "buffers":
        if args.pipeline:
            raise ValueError("Buffer transport uses the native kernel, not the legacy staged RPC")
        from astrolabe.evaluators.af3_buffers import prepare_namespace
        # Children inherit the parent's data namespace, but have their own
        # model, socket, admission and RNG state. Parent never opens bundles.
        buffer_root = prepare_namespace(args.socket, config.get("_service_buffer_root"))
        config._service_buffer_root = str(buffer_root)
    wire_options = {"transport": "json"} if transport == "buffers" else {}
    from astrolabe.runtime.evaluation import adaptive_settings, require_private_evaluator_owner
    adaptive = adaptive_settings(config.get("_service_adaptive"))
    require_private_evaluator_owner(adaptive.enabled or resident)
    if resident and args.pipeline:
        raise ValueError("Resident AF3 uses its native request scheduler, not legacy --pipeline")
    if adaptive.enabled and not resident:
        from pathlib import Path
        from astrolabe.evaluators.af3_lifecycle import owned_af3_children
        from astrolabe.evaluators.af3_pool import process_pool_handler
        from astrolabe.evaluators.runtime_stages import configured_device_hint
        from astrolabe.evaluators.inference_service import call, json_call
        # No DirectKernel/JAX/model here. Owner0 serves immediately; new serial
        # owners load only on admitted expansion, without synthetic warmups.
        with owned_af3_children(config, Path(args.config).resolve().parent, lazy=True) as children:
            handler = process_pool_handler(
                [child.socket_path if child.ready else child.config_path for child in children], adaptive=adaptive,
                device=configured_device_hint(config, default="cuda:0"),
                owners=[dict(index=i, pid=child.process.pid if child.process else None, lifecycle=str(child.state_path),
                             config=str(child.config_path), log=str(child.log_path),
                             stderr=str(child.diagnostics_path)) for i, child in enumerate(children)],
                backend_loader=children.ensure,
                invoke=json_call if transport == "buffers" else call)
            try:
                serve(handler, args.socket, args.max_pending, **wire_options)
            finally:
                handler.close()  # Drain the pool before ExitStack closes children.
        return
    from astrolabe.evaluators.runtime_stages import DirectKernel, configured_device_hint
    from astrolabe.evaluators.af3_pipeline import evaluator_stages
    loaded = []

    def loader(settings):
        runner = load_runner(config)
        loaded.append(runner)

        return evaluator_stages(runner, config, allow_extraction=True, adaptive=adaptive), {
            "backend": "alphafold3", "device": str(runner._device),
            "owner_configuration": "immutable_cyclic",
            "inference_concurrency": "resident_adaptive" if adaptive.enabled and resident else "fixed1",
            "cold_shape_jit": "inside_native_model_call",
        }

    kernel = DirectKernel({}, loader, lifecycle={
        "backend": "alphafold3", "residency": "resident_model",
        "device": configured_device_hint(config, default="cuda:0"),
        "weights_loaded": True, "concurrency_certified": False,
        "request_local_rng": adaptive.enabled or resident,
    })
    runner = loaded[0]
    try:
        if args.pipeline and not adaptive.enabled:
            from astrolabe.evaluators.af3_pipeline import serve_pipeline
            serve_pipeline(runner, config, args.socket, args.max_pending, device_kernel=kernel)
        else:
            # The native kernel still runs all three stages. Do not funnel the
            # opt-in owner through the legacy two-thread RPC envelope pool.
            handler = inference_handler(runner, config, device_kernel=kernel)
            if transport == "buffers":
                from astrolabe.evaluators.af3_buffers import BufferHandler
                handler = BufferHandler(handler, buffer_root)
            serve(handler, args.socket, args.max_pending, **wire_options)
    finally:
        kernel.close()


if __name__ == "__main__":
    main()
