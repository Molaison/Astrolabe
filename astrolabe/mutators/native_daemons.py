from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from astrolabe.model_execution import NativeModelClient, KernelStages, host_result
from astrolabe.mutators.runtime_stages import mutator_kernel_handler, device_resource, admitted_kernel, torch_load_device


LIGANDMPNN_DAEMON_FACTORY = (
    "astrolabe.mutators.native_daemons:ligandmpnn_daemon_factory"
)
CARBONDESIGN_DAEMON_FACTORY = (
    "astrolabe.mutators.native_daemons:carbondesign_daemon_factory"
)
ESM2_DAEMON_FACTORY = "astrolabe.mutators.native_daemons:esm2_daemon_factory"
ESM_IF1_DAEMON_FACTORY = (
    "astrolabe.mutators.native_daemons:esm_if1_daemon_factory"
)
ESM3_DAEMON_FACTORY = "astrolabe.mutators.native_daemons:esm3_daemon_factory"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _json_config(value: Any) -> Any:
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_config(item) for item in value]
    return value


def _alphabet_metadata(alphabet: Any) -> dict[str, Any]:
    return {
        "standard_toks": list(alphabet.standard_toks),
        "prepend_toks": list(alphabet.prepend_toks),
        "append_toks": list(alphabet.append_toks),
        "prepend_bos": bool(alphabet.prepend_bos),
        "append_eos": bool(alphabet.append_eos),
        "use_msa": bool(alphabet.use_msa),
    }


def _alphabet_from_metadata(metadata: Mapping[str, Any]) -> Any:
    import esm2.data

    return esm2.data.Alphabet(
        standard_toks=metadata["standard_toks"],
        prepend_toks=metadata["prepend_toks"],
        append_toks=metadata["append_toks"],
        prepend_bos=bool(metadata["prepend_bos"]),
        append_eos=bool(metadata["append_eos"]),
        use_msa=bool(metadata["use_msa"]),
    )


def _metadata_device(metadata: Mapping[str, Any], backend: str) -> str:
    value = metadata.get("device")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{backend} daemon metadata must report its actual execution device."
        )
    return value.strip()


def _model_device(model: Any) -> str:
    parameter = next(iter(model.parameters()), None)
    if parameter is None:
        raise ValueError("Native mutator model reports no execution device.")
    return str(parameter.device)


class NativeSequenceRunner:
    def __init__(
        self,
        *,
        backend: str,
        factory: str,
        runner_config: Any,
    ) -> None:
        config = _json_config(runner_config)
        if not isinstance(config, dict):
            raise TypeError("Native sequence runner config must resolve to a mapping.")
        self.client = NativeModelClient(
            backend,
            factory,
            {"runner_config": config},
            cwd=str(_repo_root()),
        )
        try:
            output_dir = self.client.metadata.get("output_dir")
            if not output_dir:
                raise ValueError(
                    f"{backend} daemon did not report an output directory."
                )
            self.output_dir = Path(str(output_dir))
            self.device = _metadata_device(self.client.metadata, backend)
        except BaseException:
            self.client.shutdown()
            raise

    def design_sequences(self, **features: Any) -> dict[str, Any]:
        return self.client.request({"features": features})

    def close(self) -> None:
        self.shutdown()

    def shutdown(self) -> None:
        self.client.shutdown()

    @property
    def pid(self) -> int | None:
        return self.client.pid


class NativeESMRunner:
    def __init__(self, *, backend: str, factory: str) -> None:
        self.client = NativeModelClient(
            backend,
            factory,
            {},
            cwd=str(_repo_root()),
        )
        try:
            self.alphabet = _alphabet_from_metadata(
                self.client.metadata["alphabet"]
            )
            self.device = _metadata_device(self.client.metadata, backend)
        except BaseException:
            self.client.shutdown()
            raise

    def compute_mutation(
        self, mutation_input_features: Mapping[str, Any]
    ) -> dict[str, Any]:
        return self.client.request({"features": mutation_input_features})

    def shutdown(self) -> None:
        self.client.shutdown()

    @property
    def pid(self) -> int | None:
        return self.client.pid


def _serialize_prompt(prompt: Any) -> dict[str, Any]:
    if isinstance(prompt, Mapping):
        sequence = prompt.get("sequence")
        coordinates = prompt.get("coordinates")
    else:
        sequence = getattr(prompt, "sequence", None)
        coordinates = getattr(prompt, "coordinates", None)
    if sequence is None:
        raise ValueError("ESM3 protein prompt must contain a sequence.")
    payload = {"sequence": sequence}
    if coordinates is not None:
        if hasattr(coordinates, "detach"):
            coordinates = coordinates.detach().cpu()
        payload["coordinates"] = coordinates
    return payload


class NativeESM3Runner:
    def __init__(self, *, model_name: str, device: str | None, temperature: float):
        self.client = NativeModelClient(
            "esm3",
            ESM3_DAEMON_FACTORY,
            {
                "constructor": {
                    "model_name": model_name,
                    "device": device,
                    "temperature": temperature,
                }
            },
            cwd=str(_repo_root()),
        )
        try:
            self.aa_order = list(self.client.metadata["aa_order"])
            self.device = _metadata_device(self.client.metadata, "esm3")
        except BaseException:
            self.client.shutdown()
            raise

    def compute_mutation(
        self, mutation_input_features: Mapping[str, Any]
    ) -> dict[str, Any]:
        prompts = {
            str(name): _serialize_prompt(input_feat["protein_prompt"])
            for name, input_feat in mutation_input_features.items()
        }
        return self.client.request({"prompts": prompts})

    def shutdown(self) -> None:
        self.client.shutdown()

    @property
    def pid(self) -> int | None:
        return self.client.pid


def _sequence_stages(runner):
    # Older external runner replacements stay a conservative device callable.
    if not all(callable(getattr(runner, name, None)) for name in (
        "prepare_design", "execute_design", "finalize_design"
    )):
        return lambda payload: runner.design_sequences(**payload["features"])
    return KernelStages(
        lambda payload: runner.prepare_design(**payload["features"]),
        runner.execute_design, runner.finalize_design,
        resource=device_resource(runner.device),
    )


def _load_ligandmpnn(settings: dict[str, Any]):
    from astrolabe.mutators.wrappers.ligandmpnn_run import LigandMPNN
    # This loader is also consumed by the real throughput harness inside its
    # own kernel_handler; keep both its callable closure and single lease owner.
    with admitted_kernel():
        runner = LigandMPNN(OmegaConf.create(settings["runner_config"]))
    def infer(payload):
        with admitted_kernel():
            return runner.design_sequences(**payload["features"])
    infer.runtime_stages = _sequence_stages(runner)
    return infer, {
        "output_dir": str(runner.base_folder), "device": str(runner.device),
        "model_type": str(getattr(runner, "model_type", "unknown")),
    }


def ligandmpnn_daemon_factory(settings: dict[str, Any]):
    def load(config):
        infer, metadata = _load_ligandmpnn(config)
        return getattr(infer, "runtime_stages", infer), metadata
    return mutator_kernel_handler(settings, load,
                                  backend="ligandmpnn", torch_device=torch_load_device(cpu_fallback=True))


def _load_carbondesign(settings: dict[str, Any]):
    from astrolabe.mutators.wrappers.carbondesign_runner import CarbonDesignRunner
    runner = CarbonDesignRunner(OmegaConf.create(settings["runner_config"]))
    return _sequence_stages(runner), {
        "output_dir": str(runner.output_dir), "device": str(runner.device),
        "stage_limitations": "device-configured dataset transforms remain in execute",
    }


def carbondesign_daemon_factory(settings: dict[str, Any]):
    return mutator_kernel_handler(
        settings, _load_carbondesign, backend="carbondesign",
        torch_device=torch_load_device(settings.get("runner_config", {}).get("device"), cpu_fallback=True),
    )


def _finalize_esm(rows):
    return {name: {**output, "name": name, "feature_dict": features}
            for name, features, output in rows}


def _load_esm2(settings: dict[str, Any]):
    from astrolabe.mutators.esm2 import ESM2Mutator, extract_esm2_probs
    mutator = ESM2Mutator(execution_mode="in_process")
    def prepare(payload):
        rows = []
        for name, features in payload["features"].items():
            tokens = features["batch_tokens"]
            if tokens.ndim != 2:
                raise ValueError("ESM2 batch_tokens must have shape (B,L).")
            rows.append((name, features))
        return rows
    def execute(rows):
        return [(name, features, host_result(extract_esm2_probs(mutator.runner, **features)))
                for name, features in rows]
    device = _model_device(mutator.runner)
    return KernelStages(prepare, execute, _finalize_esm,
                        resource=device_resource(device)), {
        "alphabet": _alphabet_metadata(mutator.alphabet), "device": device,
    }


def esm2_daemon_factory(settings: dict[str, Any]):
    return mutator_kernel_handler(settings, _load_esm2,
                                  backend="esm2", torch_device="auto")


def _load_esm_if1(settings: dict[str, Any]):
    from astrolabe.mutators.esm_if1 import (
        ESMIF1Mutator, prepare_if1_coords, esm_extract_probs,
    )
    mutator = ESMIF1Mutator(execution_mode="in_process")
    def prepare(payload):
        return [(name, features, prepare_if1_coords(
            mutator.runner, features["coords"], features.get("confidence")
        )) for name, features in payload["features"].items()]
    def execute(rows):
        return [(name, features, esm_extract_probs(
            mutator.runner, **features, prepared=prepared
        )) for name, features, prepared in rows]
    device = _model_device(mutator.runner)
    return KernelStages(prepare, execute, _finalize_esm,
                        resource=device_resource(device)), {
        "alphabet": _alphabet_metadata(mutator.alphabet), "device": device,
    }


def esm_if1_daemon_factory(settings: dict[str, Any]):
    return mutator_kernel_handler(settings, _load_esm_if1,
                                  backend="esm_if1", torch_device="auto")


def _load_esm3(settings: dict[str, Any]):
    from esm3.sdk.api import ESMProtein
    from astrolabe.mutators.esm3 import ESM3Mutator
    mutator = ESM3Mutator(execution_mode="in_process", **dict(settings["constructor"]))
    def prepare(payload):
        features = {}
        for name, prompt in payload["prompts"].items():
            kwargs = {"sequence": prompt["sequence"]}
            if prompt.get("coordinates") is not None:
                kwargs["coordinates"] = prompt["coordinates"]
            features[str(name)] = {"protein_prompt": ESMProtein(**kwargs)}
        return features
    def execute(features):
        # encode includes learned structure/function encoders: device work.
        return host_result(mutator._execute_prompts(features))
    return KernelStages(prepare, execute, _finalize_esm,
                        resource=device_resource(mutator.device)), {
        "aa_order": list(mutator.aa_order), "device": str(mutator.device),
    }


def esm3_daemon_factory(settings: dict[str, Any]):
    return mutator_kernel_handler(
        settings, _load_esm3, backend="esm3",
        torch_device=settings.get("constructor", {}).get("device") or "auto",
    )
