"""Typed persistent-model facades for native evaluators."""
from __future__ import annotations

import copy
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
import inspect
import os
from pathlib import Path
from typing import Any, Callable

from astrolabe.model_execution import (
    DEFAULT_MODEL_FRAME_BYTES,
    NativeModelClient as _KernelClient,
)


from astrolabe.evaluators.runtime_stages import (
    configured_device_hint, identity, jax_device_scope, operation_resource,
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
    device_hint = configured_device_hint(settings["factory_kwargs"])

    def loader(raw_settings: dict[str, Any]):
        clean = _factory_settings(raw_settings)
        from astrolabe.evaluators.utils import _ensure_alphafold2_import_path

        _ensure_alphafold2_import_path(clean["factory_kwargs"].get("data_dir"))
        from astrolabe.evaluators.wrappers.alphafold import alphafold_setup

        factory_kwargs = dict(clean["factory_kwargs"])
        factory_kwargs.pop("gpu_device", None)
        factory_kwargs.pop("device", None)
        with jax_device_scope(device_hint) as actual_device:
            runners = alphafold_setup(
                oligo_names=list(clean["oligo_names"]), **factory_kwargs,
            )
        runner = runners[str(clean["oligo_names"][0])]
        initial_guess = _plain(getattr(runner, "_gh_initial_guess_cfg", None))

        from astrolabe.evaluators.af2_stages import evaluator_stages

        return evaluator_stages(runner, device_hint=device_hint), {
            "device": str(actual_device) if actual_device is not None else None,
            "multimer_mode": bool(getattr(runner, "multimer_mode", False)),
            "initial_guess": initial_guess,
        }

    return kernel_handler(settings, loader, torch_device=None, lifecycle={
        "backend": "alphafold2", "residency": "resident_model",
        "device": device_hint, "weights_loaded": True,
        "resource": "cpu" if device_hint == "cpu" else "gpu",
    })


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

    def pair_and_merge(self, all_chain_features: Mapping[str, Any]) -> Any:
        return self._client.request(
            {"operation": "pair_and_merge", "all_chain_features": all_chain_features}
        )

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
    from astrolabe.evaluators.af3_pipeline import validate_request

    loaded = []

    # AF3 has not been qualified for concurrent scientific execution.
    profile = settings.get("_runtime_profile")
    if isinstance(profile, Mapping) and profile.get("concurrency_certified") is True:
        settings = dict(settings, _runtime_profile=dict(profile, concurrency_certified=False))

    def loader(raw_settings: dict[str, Any]):
        clean = _factory_settings(raw_settings)
        from omegaconf import OmegaConf
        from astrolabe.evaluators.af3_service import load_runner
        from astrolabe.evaluators.af3_pipeline import evaluator_stages
        from astrolabe.evaluators.wrappers.alphafold3 import service_model_settings

        config = OmegaConf.create(clean["config"])
        config.service_socket = None
        runner = load_runner(config)
        loaded.append((runner, service_model_settings(config)))
        return evaluator_stages(runner, config, allow_extraction=True), {
            "backend": "alphafold3", "device": str(runner._device),
            "settings": _plain(clean["service_settings"]),
            "residency": "resident_model", "owner_configuration": "immutable_cyclic",
            "inference_concurrency": "fixed1",
            "cold_shape_jit": "inside_native_model_call",
        }

    handler = kernel_handler(settings, loader, torch_device=None, lifecycle={
        "backend": "alphafold3", "residency": "resident_model",
        "device": configured_device_hint(settings["config"], default="cuda:0"),
        "weights_loaded": True, "concurrency_certified": False,
    })

    def handle(request):
        if request.get("operation") == "infer":
            try:
                validate_request(loaded[0][0], request["payload"], loaded[0][1])
            except (ValueError, KeyError, TypeError) as error:
                # Bad protocol/configuration is not a failed device operation
                # and must not quarantine a healthy resident model.
                return {"ok": False, "error": str(error),
                        "error_type": type(error).__name__, "random_state": request["random_state"]}
        return handler(request)

    handle.close = handler.close
    handle.max_concurrency = handler.max_concurrency
    return handle


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
        self._cyclic_asym_ids = tuple(int(x) for x in getattr(config, "cyclic_asym_ids", ()))
        self._cyclic_relative_patches = tuple(tuple(int(y) for y in x) for x in
                                             getattr(config, "cyclic_relative_patches", ()))
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

    def run_inference_for_seed(self, featurised_example, seed):
        import jax
        import numpy as np

        return self._client.request({
            "model": "alphafold3", "settings": self._settings,
            "features": jax.tree.map(np.asarray, featurised_example),
            "rng_seed": int(seed),
            "cyclic_asym_ids": self._cyclic_asym_ids,
            "cyclic_relative_patches": self._cyclic_relative_patches,
        })

    def run_inference_and_extract_for_seed(self, batch, seed, target_name):
        import jax
        import numpy as np

        return self._client.request({
            "model": "alphafold3", "settings": dict(self._settings),
            "features": jax.tree.map(np.asarray, batch), "rng_seed": int(seed),
            "target_name": target_name,
            "cyclic_asym_ids": self._cyclic_asym_ids,
            "cyclic_relative_patches": self._cyclic_relative_patches,
        })

    def extract_structures(self, batch, result, target_name):
        # Native extraction may initialize a JAX context. Keep the original
        # function in the device owner, so the scientific coordinator stays CPU.
        return self._client.request({
            "operation": "extract_structures", "batch": batch, "result": result,
            "target_name": target_name,
        })

    def extract_embeddings(self, result):
        from astrolabe.evaluators.wrappers.alphafold3 import ModelRunner

        return ModelRunner.extract_embeddings(self, result)

    def shutdown(self) -> None:
        self._client.shutdown()


def _boltz_factory(settings: dict[str, Any]) -> Callable[[dict[str, Any]], Any]:
    from astrolabe.model_execution import (
        KernelStages, capture_random_state, kernel_handler, resource_shape,
    )

    accelerator = str(dict(settings["config"]).get("accelerator", "auto")).lower()
    devices = dict(settings["config"]).get("devices", 1)
    multi_device = (isinstance(devices, (list, tuple)) and len(devices) > 1
                    or isinstance(devices, int) and devices > 1)
    if multi_device and os.environ.get("ASTROLABE_RESOURCE_SOCKET"):
        raise ValueError("Boltz multi-device prediction requires multi-device admission")
    device_hint = ("cpu" if accelerator == "cpu" else
                   f"cuda:{int(devices[0])}" if isinstance(devices, (list, tuple)) and len(devices) == 1
                   else None)
    torch_device = device_hint or "auto"
    owner = {"runner": None, "prediction": None, "failed": False,
             "scientific_requests": 0}
    model_state = {"weights_loaded": False, "model_load_count": 0,
                   "placement": "lightning_managed"}

    def cpu_loader(raw_settings: dict[str, Any]):
        clean = _factory_settings(raw_settings)
        from omegaconf import OmegaConf
        from astrolabe.evaluators.wrappers.boltz import Boltz1Runner

        owner["runner"] = Boltz1Runner(OmegaConf.create(clean["config"]))

        def featurize(request: Mapping[str, Any]) -> Any:
            if request.get("operation") != "featurize_data":
                raise ValueError(f"Unknown Boltz1 operation {request.get('operation')!r}")
            return owner["runner"].featurize_data(request["data"], request["out_dir"])

        return KernelStages(identity, featurize, resource="cpu"), {
            "backend": "boltz1", "residency": "cpu_wrapper", "device": "cpu",
            "precision": 32, "placement": "lightning_managed", "model_state": model_state,
        }

    # Constructing the wrapper and featurizing do not load model weights. They
    # must not initialize CUDA or reserve an unknown whole-device resident.
    cpu = kernel_handler(settings, cpu_loader, torch_device=None, lifecycle={
        "backend": "boltz1", "residency": "cpu_wrapper", "resource": "cpu",
        "device": "cpu", "precision": 32,
    })

    def predict(payload: Mapping[str, Any]) -> Any:
        try:
            owner["runner"].run_inference(payload["data"], payload["out_dir"])
        finally:
            if getattr(owner["runner"], "model_module", None) is not None:
                model_state.update(weights_loaded=True, model_load_count=1)
        return {"completed": True}

    def handle(request: dict[str, Any]) -> Any:
        if owner["failed"]:
            raise RuntimeError("Boltz prediction outcome_unknown; close this owner before explicit recovery")
        if request.get("operation") != "infer":
            handler = owner["prediction"] or cpu
            return handler(request)
        payload = request["payload"]
        if payload.get("operation") == "featurize_data":
            return cpu(request)
        if payload.get("operation") != "run_inference":
            raise ValueError(f"Unknown Boltz1 operation {payload.get('operation')!r}")
        try:
            first = owner["prediction"] is None
            if first:
                def prediction_loader(raw_settings: dict[str, Any]):
                    # Preserve the genuine wrapper's ordering: featurization
                    # first, then its datamodule, lazy checkpoint load and
                    # Trainer.predict. This first request runs UNDER the load
                    # lease, so registration transfers that lease atomically.
                    predict(payload)
                    return KernelStages(identity, predict, resource=(
                        "cpu" if device_hint == "cpu" else "gpu")), {
                            "backend": "boltz1", "residency": "resident_model",
                            "device": device_hint, "precision": 32,
                            "placement": "lightning_managed", "model_state": model_state,
                        }

                pending = dict(settings, _initial_random_state=request["random_state"])
                owner["prediction"] = kernel_handler(
                    pending, prediction_loader, torch_device=torch_device, lifecycle={
                        "backend": "boltz1", "residency": "resident_model",
                        "device": device_hint, "precision": 32, "weights_loaded": True,
                        "resource": "cpu" if device_hint == "cpu" else "gpu",
                        "placement": "lightning_managed",
                    })
                description = owner["prediction"]({"operation": "describe"})
                lifecycle = description["metadata"]["runtime_lifecycle"]
                try:
                    shape = resource_shape(payload)
                except (TypeError, ValueError, OSError):
                    shape = "unclassified"
                response = {"ok": True, "result": {"completed": True},
                    "random_state": capture_random_state(),
                    "runtime": {"request_id": request.get("runtime_request_id"),
                        "model_id": lifecycle["model_id"], "lifecycle": lifecycle,
                        "shape": shape, "profile_matched": (
                            not settings.get("_runtime_profile") or shape in
                            settings["_runtime_profile"].get("shapes", ())),
                        "timings": {"load_and_first_inference": {
                            "seconds": lifecycle["load_seconds"]}},
                        "first_request_executed_during_load": True}}
            else:
                response = owner["prediction"](request)
            if not response["ok"]:
                owner["failed"] = True
            else:
                owner["scientific_requests"] += 1
            response["runtime"]["scientific_requests"] = owner["scientific_requests"]
            return response
        except BaseException:
            owner["failed"] = True
            raise

    return handle


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
        from astrolabe.evaluators.protenix_stages import evaluator_stages
        import tempfile

        model_config = dict(clean["config"])
        temporary = tempfile.TemporaryDirectory(prefix="astrolabe-protenix-")
        model_config["dump_dir"] = temporary.name
        runner = ProtenixRunner(OmegaConf.create(model_config))
        runner._astrolabe_output_temp = temporary

        return evaluator_stages(runner), {
            "residency": "resident_model",
            "precision": str(runner.configs.dtype),
            "backend": "protenix",
            "device": str(runner.device),
            "dump_dir": str(runner.dump_dir),
        }

    return kernel_handler(settings, loader, torch_device="auto", lifecycle={
        "backend": "protenix", "residency": "resident_model",
        "precision": dict(settings["config"]).get("dtype", "native_config"),
        "device": f"cuda:{int(os.environ['LOCAL_RANK'])}" if "LOCAL_RANK" in os.environ else None,
        "weights_loaded": True,
        "staging": "prepare_complete_model_finalize",
    })


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
        model_config = {key: value for key, value in container.items() if key != "dump_dir"}
        self._client = _KernelClient(
            "protenix",
            "astrolabe.evaluators.native_daemons:_protenix_factory",
            {"config": model_config},
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
    from astrolabe.runtime.loading import StagedLoader

    def prepare_loader(raw_settings: dict[str, Any]):
        clean = _factory_settings(raw_settings)
        from astrolabe.evaluators.esmfold import ESMFoldEvaluator

        evaluator = ESMFoldEvaluator(
            model_name=clean["model_name"],
            device=clean.get("device"),
        )
        return evaluator, evaluator._prepare_model_cpu()

    def place_loader(prepared):
        evaluator, prepared_model = prepared
        model = evaluator._place_model(prepared_model)
        from astrolabe.evaluators.esmfold_stages import evaluator_stages

        return evaluator_stages(model, device=evaluator._device), {
            "backend": "esmfold", "device": str(evaluator._device),
            "residency": "resident_model",
            "load_metadata": evaluator._load_metadata,
            "model_state": evaluator._load_metadata["model_state"],
            "cpu_encoding": "verified_native_infer"}

    torch_device = str(settings.get("device") or "auto")
    return kernel_handler(settings, StagedLoader(prepare_loader, place_loader), torch_device=torch_device, lifecycle={
        "backend": "esmfold", "residency": "resident_model",
        "device": configured_device_hint(settings),
        "weights_loaded": None, "gpu_resident": None,
    })


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

    def infer_with_residue_scores(self, sequences):
        result = self._client.request({
            "operation": "infer", "sequences": list(sequences),
            "return_residue_plddt": True,
        })
        return result["output"], result["residue_plddt"]

    def infer_with_outputs(self, sequences):
        return self._client.request({
            "operation": "infer", "sequences": list(sequences),
            "return_residue_plddt": True, "return_pdb": True,
        })

    def output_to_pdb(self, output):
        return self._client.request({"operation": "output_to_pdb", "output": output})

    def shutdown(self) -> None:
        self._client.shutdown()


def _rhofold_factory(settings: dict[str, Any]) -> Callable[[dict[str, Any]], Any]:
    from astrolabe.model_execution import KernelStages, host_result, kernel_handler

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

        def execute(request):
            return host_result(infer(request))

        return KernelStages(
            identity, execute,
            resource=operation_resource({"export_pdb"}, device=config.get("device", "cpu")),
        ), {"backend": "rhofold", "device": str(config.get("device", "cpu")),
            "residency": "resident_model"}

    torch_device = str(dict(settings["config"]).get("device") or "cpu")
    return kernel_handler(settings, loader, torch_device=torch_device, lifecycle={
        "backend": "rhofold", "residency": "resident_model",
        "device": torch_device, "weights_loaded": True,
    })


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
        try:
            component.jit_module.to(exact_device)
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
        component_cache = _install_chai_component_cache(
            module,
            provider_identity=str(provider_identity) if provider_identity else None,
        )
        component_state = {"cached_components": 0, "weights_loaded": False}

        def infer(request: Mapping[str, Any]) -> Any:
            try:
                with patch_chai1_get_esm_embedding_context(module):
                    return module.run_inference(**dict(clean["model_config"]), **dict(request["kwargs"]))
            finally:
                component_state.update(cached_components=len(component_cache),
                                       weights_loaded=bool(component_cache))

        return infer, {
            "residency": "resident_model",
            "placement": "component_cpu_offload",
            "device": torch_device,
            "backend": "chai1",
            "supports_constraints": chai_evaluator._CHAI1_SUPPORTS_CONSTRAINTS is True,
            "module_path": chai_evaluator._CHAI1_MODULE_PATH,
            "component_state": component_state,
        }

    torch_device = str(dict(settings["model_config"]).get("device") or "auto")
    return kernel_handler(settings, loader, torch_device=torch_device, lifecycle={
        "backend": "chai1", "residency": "resident_model",
        "device": configured_device_hint(dict(settings["model_config"])),
        "placement": "component_cpu_offload",
        "staging": "native_run_inference",
    })


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
