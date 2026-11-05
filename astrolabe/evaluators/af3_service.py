"""Single AF3 GPU owner; --config accepts an AF3 mapping or a task config.

python -m astrolabe.evaluators.af3_service --socket /af3-service/model.sock \
    --config /path/to/af3_conf.yaml

All clients use the same model paths/settings. Stop after workers finish using
inference_service.call(socket, {"type": "shutdown"}); accepted work drains first.
"""
from __future__ import annotations

import argparse
import sys

from astrolabe.evaluators.inference_service import mark, serve


def load_service_config(path):
    """Use the same inherited AF3 settings as clients, or a legacy mapping."""
    from omegaconf import OmegaConf
    config = OmegaConf.load(path)
    if "defaults" in config or "evaluator" in config:
        from astrolabe.configuration import load_composed_config as load_config
        config = load_config(path).evaluator.alphafold3
        config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        # This config also serves CPU clients. Only the owner copy must use
        # the local model; keep the original client transport untouched.
        config.service_socket = None
        return config
    return OmegaConf.create(OmegaConf.to_container(config, resolve=True))


def inference_handler(runner, config):
    import jax
    import numpy as np
    from astrolabe.evaluators.wrappers.alphafold3 import service_model_settings

    settings = service_model_settings(config)

    def predict(request):
        if request.get("model") != "alphafold3":
            raise ValueError("Wrong model for this AlphaFold3 service")
        if request["settings"] != settings:
            raise ValueError("Shared AlphaFold3 model settings differ from owner")
        runner.set_cyclic_asym_ids(request["cyclic_asym_ids"])
        runner.set_cyclic_relative_patches(request["cyclic_relative_patches"])
        result = runner.run_inference(request["features"], request["rng_key"])
        # Materialize all device results before serve releases the GPU lock.
        # ModelRunner already does the original bfloat16 -> float32 conversion.
        return jax.tree.map(lambda x: x if isinstance(x, bytes) else np.asarray(x), result)

    return predict


def load_runner(config):
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
    runner = load_runner(config)
    if args.pipeline:
        from astrolabe.evaluators.af3_pipeline import serve_pipeline
        serve_pipeline(runner, config, args.socket, args.max_pending)
    else:
        serve(inference_handler(runner, config), args.socket, args.max_pending)


if __name__ == "__main__":
    main()
