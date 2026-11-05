"""Single-chain RNA adapters using the existing resident model owner."""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import importlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from astrolabe.core.folding_input import RnaChain
from astrolabe.evaluators.base import Evaluator


@dataclass(frozen=True)
class RnaBackend:
    name: str
    class_name: str
    source: str
    revision: str


RNA_BACKENDS = {
    spec.name: spec
    for spec in (
        RnaBackend("drfold2", "DRfold2Backend", "leeyang/DRfold2",
                   "3667d6ac44c5b14cff94c36e0371224acb2cdae5"),
        RnaBackend("numonomer", "NUMonomerBackend", "yunda-si/NUMonomer",
                   "af38434e7ee3f82c6948841cce1ca4d65840742a"),
        RnaBackend("zfold", "ZfoldBackend", "heqin-zhu/structRFM",
                   "f0c1add19f8dfc786f16bc4abb7aa775dbadddbf"),
        RnaBackend("rnabpflow", "RNAbpFlowBackend", "Bhattacharya-Lab/RNAbpFlow",
                   "e8b1c077f7951d4d6ebe00c851d22c37777dfc9f"),
    )
}


def _host_only(value: Any) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _host_only(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _host_only(item)
    elif is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _host_only(getattr(value, field.name))
    elif hasattr(value, "device") and hasattr(value, "detach"):
        if str(value.device) != "cpu":
            raise RuntimeError("RNA execute must complete device-to-host transfer")


def _backend_kernel(settings: dict):
    from astrolabe import model_execution

    name = settings["backend"]
    spec = RNA_BACKENDS[name]
    device = str(settings.get("device", "cuda:0"))
    native_stages = getattr(model_execution, "EvaluatorStages", None)
    if native_stages is None and os.environ.get("ASTROLABE_RESOURCE_SOCKET"):
        raise RuntimeError(
            "RNA evaluators require the staged model runtime when a resource "
            "broker is active. The legacy runtime cannot admit stages separately."
        )

    def prepare_loader(raw_settings):
        module = importlib.import_module(
            f"astrolabe.evaluators.wrappers.{name}"
        )
        options = {
            key: value for key, value in raw_settings.items()
            if key not in {"_initial_random_state", "_provider_request"}
        }
        return getattr(module, spec.class_name)(options)

    def place_loader(backend):
        backend.place(device)

        def execute(prepared):
            result = backend.execute(prepared)
            _host_only(result)
            return result

        if native_stages is not None:
            infer = native_stages(
                prepare=backend.prepare,
                execute=execute,
                finalize=backend.finalize,
                resource="cpu" if device == "cpu" else "gpu",
                prepare_resource="cpu",
                finalize_resource="cpu",
            )
        else:
            def infer(request):
                return backend.finalize(execute(backend.prepare(request)))

        return infer, {
            "backend": name,
            "device": device,
            "upstream_source": spec.source,
            "adapter_source_revision": spec.revision,
            "stage_contract": (
                "native_cpu_device_cpu" if native_stages is not None
                else "serial_cpu_device_cpu"
            ),
        }

    if native_stages is not None:
        from astrolabe.runtime.loading import StagedLoader

        loader = StagedLoader(prepare_loader, place_loader)
    else:
        def loader(raw_settings):
            return place_loader(prepare_loader(raw_settings))

    return model_execution.kernel_handler(settings, loader, torch_device=device)


class RnaEvaluator(Evaluator):
    """Validate requests before starting an isolated, lazily loaded backend."""

    _staged_evaluation = True

    def __init__(self, backend: str, output_dir, settings: dict):
        if backend not in RNA_BACKENDS:
            raise ValueError(f"Unknown RNA backend {backend!r}")
        self.backend = backend
        self.settings = dict(settings)
        self.settings["backend"] = backend
        self.output_dir = Path(output_dir).resolve()
        self.model_runner = None
        self._client_options = {}
        for key, option in (
            ("execution_python_bin", "python_bin"),
            ("execution_cwd", "cwd"),
            ("execution_env", "env"),
            ("execution_request_timeout", "request_timeout"),
            ("execution_max_frame_bytes", "max_frame_bytes"),
        ):
            if key in self.settings:
                self._client_options[option] = self.settings.pop(key)
        mode = self.settings.pop("execution_mode", "daemon")
        legacy_mode = self.settings.pop("execution", mode)
        if mode != "daemon" or legacy_mode != "daemon":
            raise ValueError("RNA backends require execution_mode=daemon for import isolation")
        device = str(self.settings.setdefault("device", "cuda:0"))
        if device != "cpu" and not (
            device.startswith("cuda:") and device[5:].isdigit()
        ):
            raise ValueError("RNA device must be 'cpu' or an explicit 'cuda:N'")

    def _runner(self):
        if self.model_runner is None:
            from astrolabe.model_execution import NativeModelClient

            self.model_runner = NativeModelClient(
                self.backend,
                "astrolabe.evaluators.rna:_backend_kernel",
                self.settings,
                max_pending=1,
                **self._client_options,
            )
        return self.model_runner

    def prepare_input(self, inputs, runtime_config=None):
        if not inputs:
            raise ValueError("RNA evaluation requires at least one input")
        validated = {}
        for name, folding_input in inputs.items():
            chains = folding_input.chains
            if len(chains) != 1:
                raise ValueError(
                    f"{self.backend} supports a single RNA chain, not oligomer {name!r}"
                )
            chain = chains[0]
            native_rna = (
                type(chain).__name__ == "RnaChain"
                and type(chain).__module__ == "alphafold3.common.folding_input"
            )
            if not isinstance(chain, RnaChain) and not native_rna:
                raise ValueError(f"{self.backend} requires an RNA chain")
            sequence = str(chain.sequence).upper()
            if not sequence or set(sequence) - set("ACGU"):
                raise ValueError("RNA prediction requires an unmodified A/C/G/U sequence")
            if getattr(chain, "modifications", None):
                raise ValueError("Modified RNA is not supported by this adapter")
            if (
                getattr(folding_input, "bonded_atom_pairs", None)
                or getattr(folding_input, "cyclic_chain_ids", None)
                or getattr(folding_input, "cyclic_relative_patches", None)
            ):
                raise ValueError("RNA adapters currently support linear, unbonded chains")
            seeds = list(folding_input.rng_seeds)
            if len(seeds) != 1:
                raise ValueError("RNA adapters require exactly one input seed per evaluation")
            validated[str(name)] = {
                "name": str(name),
                "sequence": sequence,
                "chain_id": str(chain.id),
                "seed": int(seeds[0]),
            }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        step = int((runtime_config or {}).get("step", 0))
        for request in validated.values():
            request["output_dir"] = tempfile.mkdtemp(
                prefix=f"step_{step:06d}_", dir=self.output_dir
            )
        return validated

    def run_function(self, input_feat):
        runner = self._runner()
        return {
            name: {"request": request, "prediction": runner.request(request)}
            for name, request in input_feat.items()
        }

    def prepare_output(self, output, inputs, runtime_config=None):
        import numpy as np

        from astrolabe.evaluators.utils import StandardScores

        structures, confidence, per_proto = {}, {}, {}
        for name, item in output.items():
            request, prediction = item["request"], item["prediction"]
            path = Path(prediction["structure_path"]).resolve()
            root = Path(request["output_dir"]).resolve()
            if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"{self.backend} did not produce a request-local structure")
            structures[name] = path
            plddt = prediction.get("plddt")
            if plddt is not None:
                values = np.asarray(plddt, dtype=float)
                if values.shape != (len(request["sequence"]),) or not np.isfinite(values).all():
                    raise ValueError(f"{self.backend} returned invalid per-residue confidence")
                confidence[name] = [values]
                per_proto[name] = [values]
            receipt = {
                "backend": self.backend,
                "request": request,
                "owner": self.model_runner.metadata,
                "structure_path": str(path),
                "artifacts": prediction.get("artifacts", {}),
                "metadata": prediction.get("metadata", {}),
                "confidence_available": plddt is not None,
            }
            (root / "astrolabe_prediction.json").write_text(
                json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
            )
        return StandardScores(
            names=list(output), plddt=confidence, complex_paths=structures,
            per_proto_scores=per_proto or None,
        )

    def shutdown(self):
        if self.model_runner is not None:
            self.model_runner.shutdown()
            self.model_runner = None
