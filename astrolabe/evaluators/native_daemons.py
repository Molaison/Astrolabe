"""Typed persistent-model facades for native evaluators."""
from __future__ import annotations

import copy
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
import inspect
from pathlib import Path
from typing import Any, Callable

from astrolabe.model_execution import (
    DEFAULT_MODEL_FRAME_BYTES,
    NativeModelClient as _KernelClient,
)


SCIENTIFIC_FRAME_BYTES = DEFAULT_MODEL_FRAME_BYTES


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _factory_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in settings.items()
        if key not in {"_initial_random_state", "_provider_request"}
    }


def _af2_factory(settings: dict[str, Any]) -> Callable[[dict[str, Any]], Any]:
    from astrolabe.model_execution import kernel_handler

    def loader(raw_settings: dict[str, Any]):
        clean = _factory_settings(raw_settings)
        from astrolabe.evaluators.utils import _ensure_alphafold2_import_path

        _ensure_alphafold2_import_path(clean["factory_kwargs"].get("data_dir"))
        from astrolabe.evaluators.wrappers.alphafold import alphafold_setup

        runners = alphafold_setup(**dict(clean["factory_kwargs"]))
        runner = runners[str(clean["oligo_names"][0])]
        initial_guess = _plain(getattr(runner, "_gh_initial_guess_cfg", None))

        def infer(request: Mapping[str, Any]) -> Any:
            operation = request.get("operation")
            if operation == "process_features":
                return runner.process_features(
                    copy.deepcopy(request["features"]),
                    random_seed=request["random_seed"],
                )
            if operation == "predict":
                return runner.predict(
                    request["features"],
                    random_seed=request["random_seed"],
                )
            raise ValueError(f"Unknown AlphaFold2 operation {operation!r}")

        return infer, {
            "multimer_mode": bool(getattr(runner, "multimer_mode", False)),
            "initial_guess": initial_guess,
        }

    return kernel_handler(settings, loader, torch_device=None)


class NativeAF2Runner:
    def __init__(
        self,
        oligo_names: list[str],
        factory_kwargs: Mapping[str, Any],
        *,
        python_bin: str | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        request_timeout: float | None = None,
        max_frame_bytes: int = SCIENTIFIC_FRAME_BYTES,
    ) -> None:
        self._client = _KernelClient(
            "alphafold2",
            "astrolabe.evaluators.native_daemons:_af2_factory",
            {"oligo_names": list(oligo_names), "factory_kwargs": dict(factory_kwargs)},
            python_bin=python_bin,
            cwd=cwd,
            env=env,
            request_timeout=request_timeout,
            max_frame_bytes=max_frame_bytes,
        )
        self.multimer_mode = bool(self._client.metadata["multimer_mode"])
        self._gh_initial_guess_cfg = self._client.metadata["initial_guess"]

    def process_features(self, features: Mapping[str, Any], *, random_seed: int) -> Any:
        return self._client.request(
            {
                "operation": "process_features",
                "features": features,
                "random_seed": int(random_seed),
            }
        )

    def predict(self, features: Mapping[str, Any], *, random_seed: int) -> Any:
        return self._client.request(
            {
                "operation": "predict",
                "features": features,
                "random_seed": int(random_seed),
            }
        )

    def shutdown(self) -> None:
        self._client.shutdown()


def build_af2_daemon_runners(
    oligo_names: list[str],
    factory_kwargs: Mapping[str, Any],
    **kwargs: Any,
) -> dict[str, NativeAF2Runner]:
    runner = NativeAF2Runner(oligo_names, factory_kwargs, **kwargs)
    return {str(name): runner for name in oligo_names}


def _af3_factory(settings: dict[str, Any]) -> Callable[[dict[str, Any]], Any]:
    from astrolabe.model_execution import kernel_handler

    def loader(raw_settings: dict[str, Any]):
        clean = _factory_settings(raw_settings)
        from omegaconf import OmegaConf
        from astrolabe.evaluators.af3_service import inference_handler, load_runner

        config = OmegaConf.create(clean["config"])
        config.service_socket = None
        runner = load_runner(config)
        infer = inference_handler(runner, config)
        return infer, {
            "device": str(runner._device),
            "settings": _plain(clean["service_settings"]),
        }

    return kernel_handler(settings, loader, torch_device=None)


class NativeAF3Runner:
    extract_structures = None
    extract_embeddings = None

    def __init__(
        self,
        config: Any,
        *,
        python_bin: str | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        request_timeout: float | None = None,
        max_frame_bytes: int = SCIENTIFIC_FRAME_BYTES,
    ) -> None:
        from omegaconf import OmegaConf
        from astrolabe.evaluators.wrappers.alphafold3 import service_model_settings

        container = OmegaConf.to_container(config, resolve=True)
        if not isinstance(container, dict):
            raise TypeError("AlphaFold3 config must resolve to a mapping")
        self._settings = service_model_settings(config)
        self._cyclic_asym_ids: tuple[int, ...] = ()
        self._cyclic_relative_patches: tuple[tuple[int, ...], ...] = ()
        self._client = _KernelClient(
            "alphafold3",
            "astrolabe.evaluators.native_daemons:_af3_factory",
            {"config": container, "service_settings": self._settings},
            python_bin=python_bin,
            cwd=cwd,
            env=env,
            request_timeout=request_timeout,
            max_frame_bytes=max_frame_bytes,
        )

    def set_cyclic_asym_ids(self, ids) -> None:
        self._cyclic_asym_ids = tuple(int(item) for item in (ids or ()))

    def set_cyclic_relative_patches(self, patches) -> None:
        self._cyclic_relative_patches = tuple(
            tuple(int(item) for item in patch) for patch in (patches or ())
        )

    def run_inference(self, featurised_example, rng_key):
        import jax
        import numpy as np

        return self._client.request(
            {
                "model": "alphafold3",
                "settings": self._settings,
                "features": jax.tree.map(np.asarray, featurised_example),
                "rng_key": np.asarray(rng_key),
                "cyclic_asym_ids": self._cyclic_asym_ids,
                "cyclic_relative_patches": self._cyclic_relative_patches,
            }
        )

    def extract_structures(self, batch, result, target_name):
        from astrolabe.evaluators.wrappers.alphafold3 import ModelRunner

        return ModelRunner.extract_structures(self, batch, result, target_name)

    def extract_embeddings(self, result):
        from astrolabe.evaluators.wrappers.alphafold3 import ModelRunner

        return ModelRunner.extract_embeddings(self, result)

    def shutdown(self) -> None:
        self._client.shutdown()


def _boltz_factory(settings: dict[str, Any]) -> Callable[[dict[str, Any]], Any]:
    from astrolabe.model_execution import kernel_handler

    def loader(raw_settings: dict[str, Any]):
        clean = _factory_settings(raw_settings)
        from omegaconf import OmegaConf
        from astrolabe.evaluators.wrappers.boltz import Boltz1Runner

        runner = Boltz1Runner(OmegaConf.create(clean["config"]))

        def infer(request: Mapping[str, Any]) -> Any:
            operation = request.get("operation")
            if operation == "featurize_data":
                return runner.featurize_data(request["data"], request["out_dir"])
            if operation == "run_inference":
                runner.run_inference(request["data"], request["out_dir"])
                return {"completed": True}
            raise ValueError(f"Unknown Boltz1 operation {operation!r}")

        return infer, {"backend": "boltz1"}

    accelerator = str(dict(settings["config"]).get("accelerator", "auto")).lower()
    torch_device = "cpu" if accelerator == "cpu" else "auto"
    return kernel_handler(settings, loader, torch_device=torch_device)


class NativeBoltz1Runner:
    def __init__(
        self,
        config: Any,
        *,
        python_bin: str | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        request_timeout: float | None = None,
        max_frame_bytes: int = SCIENTIFIC_FRAME_BYTES,
    ) -> None:
        from omegaconf import OmegaConf

        container = OmegaConf.to_container(config, resolve=True)
        if not isinstance(container, dict):
            raise TypeError("Boltz1 config must resolve to a mapping")
        self._client = _KernelClient(
            "boltz1",
            "astrolabe.evaluators.native_daemons:_boltz_factory",
            {"config": container},
            python_bin=python_bin,
            cwd=cwd,
            env=env,
            request_timeout=request_timeout,
            max_frame_bytes=max_frame_bytes,
        )

    def featurize_data(self, data, out_dir):
        return self._client.request(
            {"operation": "featurize_data", "data": data, "out_dir": Path(out_dir)}
        )

    def run_inference(self, data, out_dir):
        return self._client.request(
            {"operation": "run_inference", "data": data, "out_dir": Path(out_dir)}
        )

    def shutdown(self) -> None:
        self._client.shutdown()


def _protenix_factory(settings: dict[str, Any]) -> Callable[[dict[str, Any]], Any]:
    from astrolabe.model_execution import kernel_handler

    def loader(raw_settings: dict[str, Any]):
        clean = _factory_settings(raw_settings)
        from omegaconf import OmegaConf
        from astrolabe.evaluators.wrappers.protenix import ProtenixRunner

        runner = ProtenixRunner(OmegaConf.create(clean["config"]))

        def infer(request: Mapping[str, Any]) -> Any:
            operation = request.get("operation")
            if operation == "predict_and_dump":
                runner.update_model_configs(OmegaConf.create(request["configs"]))
                prediction = runner.predict(request["data"])
                runner.dumper.dump(
                    dataset_name=request["dataset_name"],
                    pdb_id=request["pdb_id"],
                    seed=request["seed"],
                    pred_dict=prediction,
                    atom_array=request["atom_array"],
                    entity_poly_type=request["entity_poly_type"],
                )
                return prediction
            raise ValueError(f"Unknown Protenix operation {operation!r}")

        return infer, {
            "backend": "protenix",
            "device": str(runner.device),
            "dump_dir": str(runner.dump_dir),
        }

    return kernel_handler(settings, loader, torch_device="auto")


class NativeProtenixRunner:
    def __init__(
        self,
        config: Any,
        *,
        python_bin: str | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        request_timeout: float | None = None,
        max_frame_bytes: int = SCIENTIFIC_FRAME_BYTES,
    ) -> None:
        from omegaconf import OmegaConf

        container = OmegaConf.to_container(config, resolve=True)
        if not isinstance(container, dict):
            raise TypeError("Protenix config must resolve to a mapping")
        self.configs = OmegaConf.create(container)
        self._pending_configs = None
        self._client = _KernelClient(
            "protenix",
            "astrolabe.evaluators.native_daemons:_protenix_factory",
            {"config": container},
            python_bin=python_bin,
            cwd=cwd,
            env=env,
            request_timeout=request_timeout,
            max_frame_bytes=max_frame_bytes,
        )

    def update_model_configs(self, configs) -> None:
        from omegaconf import OmegaConf

        self._pending_configs = OmegaConf.to_container(configs, resolve=True)

    def predict_and_dump(
        self,
        data,
        *,
        atom_array,
        dataset_name: str,
        pdb_id: str,
        seed: int,
        entity_poly_type,
    ):
        configs = self._pending_configs
        if configs is None:
            from omegaconf import OmegaConf

            configs = OmegaConf.to_container(self.configs, resolve=True)
        self._pending_configs = None
        return self._client.request(
            {
                "operation": "predict_and_dump",
                "configs": configs,
                "data": data,
                "atom_array": atom_array,
                "dataset_name": dataset_name,
                "pdb_id": pdb_id,
                "seed": int(seed),
                "entity_poly_type": entity_poly_type,
            }
        )

    def shutdown(self) -> None:
        self._client.shutdown()


def _esmfold_factory(settings: dict[str, Any]) -> Callable[[dict[str, Any]], Any]:
    from astrolabe.model_execution import kernel_handler

    def loader(raw_settings: dict[str, Any]):
        clean = _factory_settings(raw_settings)
        from astrolabe.evaluators.esmfold import ESMFoldEvaluator

        evaluator = ESMFoldEvaluator(
            model_name=clean["model_name"],
            device=clean.get("device"),
        )
        model = evaluator._ensure_model()

        def infer(request: Mapping[str, Any]) -> Any:
            operation = request.get("operation")
            if operation == "infer":
                return model.infer(request["sequences"])
            if operation == "output_to_pdb":
                return model.output_to_pdb(request["output"])
            raise ValueError(f"Unknown ESMFold operation {operation!r}")

        return infer, {"backend": "esmfold", "device": str(evaluator._device)}

    torch_device = str(settings.get("device") or "auto")
    return kernel_handler(settings, loader, torch_device=torch_device)


class NativeESMFoldModel:
    def __init__(
        self,
        *,
        model_name: str = "esmfold_3B_v1",
        device: str | None = None,
        python_bin: str | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        request_timeout: float | None = None,
        max_frame_bytes: int = SCIENTIFIC_FRAME_BYTES,
    ) -> None:
        self._client = _KernelClient(
            "esmfold",
            "astrolabe.evaluators.native_daemons:_esmfold_factory",
            {"model_name": model_name, "device": device},
            python_bin=python_bin,
            cwd=cwd,
            env=env,
            request_timeout=request_timeout,
            max_frame_bytes=max_frame_bytes,
        )

    def infer(self, sequences):
        return self._client.request({"operation": "infer", "sequences": list(sequences)})

    def output_to_pdb(self, output):
        return self._client.request({"operation": "output_to_pdb", "output": output})

    def shutdown(self) -> None:
        self._client.shutdown()


def _rhofold_factory(settings: dict[str, Any]) -> Callable[[dict[str, Any]], Any]:
    from astrolabe.model_execution import kernel_handler

    def loader(raw_settings: dict[str, Any]):
        clean = _factory_settings(raw_settings)
        config = dict(clean["config"])
        checkpoint = config.get("checkpoint")
        if not checkpoint:
            raise ValueError("RhoFold checkpoint is missing")
        import torch
        from rhofold.config import rhofold_config
        from rhofold.rhofold import RhoFold

        model = RhoFold(rhofold_config)
        state = torch.load(str(checkpoint), map_location=torch.device("cpu"))
        model.load_state_dict(state["model"])
        model = model.to(config.get("device", "cpu")).eval()

        def infer(request: Mapping[str, Any]) -> Any:
            operation = request.get("operation")
            if operation == "forward":
                return _rhofold_forward(model, request, config.get("device", "cpu"))
            if operation == "export_pdb":
                model.structure_module.converter.export_pdb_file(
                    request["sequence"],
                    request["coordinates"],
                    path=request["path"],
                    chain_id=request["chain_id"],
                    confidence=request["confidence"],
                )
                return {"path": str(request["path"])}
            raise ValueError(f"Unknown RhoFold operation {operation!r}")

        return infer, {"backend": "rhofold", "device": str(config.get("device", "cpu"))}

    torch_device = str(dict(settings["config"]).get("device") or "cpu")
    return kernel_handler(settings, loader, torch_device=torch_device)


def _rhofold_forward(model: Any, request: Mapping[str, Any], device: Any) -> Any:
    import torch

    tokens = request["tokens"]
    rna_fm_tokens = request["rna_fm_tokens"]
    if getattr(tokens, "is_cuda", True) or getattr(rna_fm_tokens, "is_cuda", True):
        raise ValueError("RhoFold daemon requires CPU input tensors")
    with torch.no_grad():
        if torch.is_grad_enabled():
            raise RuntimeError("RhoFold child inference must run with gradients disabled")
        return model(
            tokens=tokens.to(device),
            rna_fm_tokens=rna_fm_tokens.to(device),
            seq=request["seq"],
        )[-1]


class NativeRhoFoldRunner:
    native_daemon = True

    def __init__(
        self,
        config: Any,
        *,
        python_bin: str | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        request_timeout: float | None = None,
        max_frame_bytes: int = SCIENTIFIC_FRAME_BYTES,
    ) -> None:
        from omegaconf import OmegaConf

        container = OmegaConf.to_container(config, resolve=True)
        if not isinstance(container, dict):
            raise TypeError("RhoFold config must resolve to a mapping")
        self._client = _KernelClient(
            "rhofold",
            "astrolabe.evaluators.native_daemons:_rhofold_factory",
            {"config": container},
            python_bin=python_bin,
            cwd=cwd,
            env=env,
            request_timeout=request_timeout,
            max_frame_bytes=max_frame_bytes,
        )

    def to(self, device):
        return self

    def forward(self, *, tokens, rna_fm_tokens, seq):
        return self._client.request(
            {
                "operation": "forward",
                "tokens": tokens,
                "rna_fm_tokens": rna_fm_tokens,
                "seq": seq,
            }
        )

    def export_pdb_file(self, sequence, coordinates, *, path, chain_id, confidence):
        return self._client.request(
            {
                "operation": "export_pdb",
                "sequence": sequence,
                "coordinates": coordinates,
                "path": Path(path),
                "chain_id": chain_id,
                "confidence": confidence,
            }
        )

    def shutdown(self) -> None:
        self._client.shutdown()


def _install_chai_component_cache(module: Any, *, provider_identity: str | None):
    run_inference = getattr(module, "run_inference", None)
    load_exported = getattr(module, "load_exported", None)
    if not callable(run_inference) or not callable(load_exported):
        raise RuntimeError(
            "Chai1 does not expose the required run_inference/load_exported API."
        )
    try:
        source = inspect.getsource(run_inference)
    except (OSError, TypeError) as error:
        raise RuntimeError(
            "Chai1 run_inference source is unavailable; refusing an unverified "
            "component cache."
        ) from error
    if "_component_moved_to(" not in source:
        raise RuntimeError(
            "Chai1 run_inference does not use _component_moved_to; refusing to "
            "claim persistent exported-component weights."
        )

    module_file = str(getattr(module, "__file__", "<unknown>"))
    package_name = str(module.__package__ or module.__name__).split(".", 1)[0]
    try:
        package = __import__(package_name)
        version = str(getattr(package, "__version__", "unknown"))
    except ImportError:
        version = "unknown"
    model_identity = provider_identity or f"chai1-module:{module_file}@{version}"
    cache: dict[tuple[str, str, str], Any] = {}

    @contextmanager
    def component_moved_to(comp_key: str, device):
        import torch

        exact_device = torch.device(device)
        key = (str(comp_key), str(exact_device), model_identity)
        component = cache.get(key)
        if component is None:
            component = load_exported(key[0], exact_device)
            cache[key] = component
        component.jit_module.to(exact_device)
        try:
            yield component
        finally:
            component.jit_module.to("cpu")

    module._component_moved_to = component_moved_to
    return cache


def _chai1_factory(settings: dict[str, Any]) -> Callable[[dict[str, Any]], Any]:
    from astrolabe.model_execution import kernel_handler

    def loader(raw_settings: dict[str, Any]):
        clean = _factory_settings(raw_settings)
        from astrolabe.evaluators.chai1 import _load_chai1_module
        import astrolabe.evaluators.chai1 as chai_evaluator
        from astrolabe.evaluators.x_masking import patch_chai1_get_esm_embedding_context

        module = _load_chai1_module()
        provider_request = raw_settings.get("_provider_request") or {}
        provider_identity = (
            provider_request.get("model_identity")
            if isinstance(provider_request, Mapping)
            else None
        )
        _install_chai_component_cache(
            module,
            provider_identity=str(provider_identity) if provider_identity else None,
        )

        def infer(request: Mapping[str, Any]) -> Any:
            with patch_chai1_get_esm_embedding_context(module):
                return module.run_inference(**dict(clean["model_config"]), **dict(request["kwargs"]))

        return infer, {
            "backend": "chai1",
            "supports_constraints": chai_evaluator._CHAI1_SUPPORTS_CONSTRAINTS is True,
            "module_path": chai_evaluator._CHAI1_MODULE_PATH,
        }

    torch_device = str(dict(settings["model_config"]).get("device") or "auto")
    return kernel_handler(settings, loader, torch_device=torch_device)


class NativeChai1Runner:
    def __init__(
        self,
        model_config: Any,
        *,
        python_bin: str | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        request_timeout: float | None = None,
        max_frame_bytes: int = SCIENTIFIC_FRAME_BYTES,
    ) -> None:
        from omegaconf import OmegaConf

        container = OmegaConf.to_container(model_config, resolve=True)
        if not isinstance(container, dict):
            raise TypeError("Chai1 config must resolve to a mapping")
        self._client = _KernelClient(
            "chai1",
            "astrolabe.evaluators.native_daemons:_chai1_factory",
            {"model_config": container},
            python_bin=python_bin,
            cwd=cwd,
            env=env,
            request_timeout=request_timeout,
            max_frame_bytes=max_frame_bytes,
        )
        self.supports_constraints = bool(
            self._client.metadata["supports_constraints"]
        )

    def run_inference(self, **kwargs):
        return self._client.request({"operation": "chai1", "kwargs": kwargs})

    def shutdown(self) -> None:
        self._client.shutdown()
