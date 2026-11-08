from collections import defaultdict
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any

import torch

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import ProtomerTypes
from astrolabe.evaluators.runtime_stages import local_evaluator_guard
from astrolabe.evaluators.base import Evaluator
from astrolabe.evaluators.structure_output import project_structure, structure_directory
from astrolabe.evaluators.utils import StandardScores, shutdown_model_handles


@contextmanager
def _record_load_stage(metadata, name):
    """Time actual host calls only; no CUDA events, probes or synchronization."""
    started = time.perf_counter()
    process_started = time.process_time()
    stage = {"start_perf_counter_s": started, "status": "failed"}
    metadata["stages"][name] = stage
    try:
        yield
        stage["status"] = "completed"
    finally:
        ended = time.perf_counter()
        stage.update(
            end_perf_counter_s=ended,
            wall_s=ended - started,
            process_time_s=time.process_time() - process_started,
        )


def _model_tensor_inventory(model):
    """Read tensor descriptors, without copying data or querying a device.

    Bytes are logical numel * element_size, not allocator/storage measurements.
    Repeated tensor objects are counted once per kind; distinct storage views
    are separate tensors. Meta tensors have logical size but no loaded weights.
    gpu_resident means any nonempty CUDA parameter/buffer is present, not that
    every tensor is on CUDA. Without nonempty parameters both facts are unknown.
    """
    state = {"weights_loaded": None, "gpu_resident": None,
             "byte_accounting": "logical_numel_times_element_size"}
    parameter_numel = 0
    meta_parameter_numel = 0
    cuda_numel = 0
    for kind, method_name in (("parameters", "named_parameters"),
                              ("buffers", "named_buffers")):
        method = getattr(model, method_name, None)
        summary = {"available": callable(method), "tensor_count": 0,
                   "numel": 0, "bytes": 0, "by_device_dtype": []}
        groups = {}
        seen = set()
        if callable(method):
            for _, tensor in method():
                if id(tensor) in seen:
                    continue
                seen.add(id(tensor))
                device = str(tensor.device)
                dtype = str(tensor.dtype)
                numel = tensor.numel()
                size = numel * tensor.element_size()
                group = groups.setdefault((device, dtype), {
                    "device": device, "dtype": dtype, "tensor_count": 0,
                    "numel": 0, "bytes": 0,
                })
                for counts in (summary, group):
                    counts["tensor_count"] += 1
                    counts["numel"] += numel
                    counts["bytes"] += size
                device_type = device.split(":", 1)[0]
                if device_type == "cuda":
                    cuda_numel += numel
                if kind == "parameters":
                    parameter_numel += numel
                    if device_type == "meta":
                        meta_parameter_numel += numel
        summary["by_device_dtype"] = [groups[key] for key in sorted(groups)]
        state[kind] = summary
    if parameter_numel:
        state["weights_loaded"] = meta_parameter_numel == 0
        state["gpu_resident"] = cuda_numel > 0
    return state


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_esmfold_vendor_paths() -> None:
    for path in (
        _repo_root() / "vendor" / "openfold_esm2",
        _repo_root() / "vendor" / "esm2",
    ):
        if path.exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)


def _load_renamed_esmfold_model_class():
    repo_vendor = str(_repo_root() / "vendor")
    original_sys_path = list(sys.path)
    namespace_module = sys.modules.get("esm2")
    if namespace_module is not None and getattr(namespace_module, "__file__", None) is None:
        sys.modules.pop("esm2", None)
    try:
        sys.path = [path for path in sys.path if path != repo_vendor]
        return importlib.import_module("esm2.esmfold.v1.esmfold").ESMFold
    finally:
        sys.path = original_sys_path


def _load_esmfold_model_class():
    try:
        return _load_renamed_esmfold_model_class()
    except ImportError as esm2_exc:
        _ensure_esmfold_vendor_paths()
        try:
            return _load_renamed_esmfold_model_class()
        except ImportError:
            try:
                return importlib.import_module("esm.esmfold.v1.esmfold").ESMFold
            except ImportError:
                raise RuntimeError(
                    "Failed to import ESMFold model implementation."
                ) from esm2_exc


def _adapt_esmfold_state_dict_for_openfold(
    model_state: dict[str, Any],
) -> dict[str, Any]:
    adapted = dict(model_state)
    for base_key in (
        "trunk.structure_module.ipa.linear_q_points",
        "trunk.structure_module.ipa.linear_kv_points",
    ):
        for suffix in ("weight", "bias"):
            old_key = f"{base_key}.{suffix}"
            new_key = f"{base_key}.linear.{suffix}"
            if old_key in adapted and new_key not in adapted:
                adapted[new_key] = adapted[old_key]
    return adapted


def renumber_each_chain(pdb_str):
    """Renumber each chain in a PDB string to start from 1."""
    output_lines = []
    current_chain = None
    residue_number = 0
    last_residue_id = None

    for line in pdb_str.splitlines():
        if line.startswith("ATOM") or line.startswith("HETATM"):
            chain_id = line[21]
            residue_id = line[22:27].strip()
            if chain_id != current_chain:
                current_chain = chain_id
                residue_number = 1
                last_residue_id = residue_id
            elif residue_id != last_residue_id:
                residue_number += 1
                last_residue_id = residue_id
            new_line = line[:22] + f"{residue_number:>4}" + line[26:]
            output_lines.append(new_line)
        else:
            output_lines.append(line)

    return "\n".join(output_lines) + "\n"


class ESMFoldEvaluator(Evaluator):
    _staged_evaluation = True

    def __init__(
        self,
        *,
        model: Any | None = None,
        model_name: str = "esmfold_3B_v1",
        device: str | None = None,
    ):
        self._model = model
        self._model_name = model_name
        self._device = device
        self._load_metadata = {}
        self._loader_attempts = 0
        self._successful_loads = 0
        self._placements = 0
        self._load_identity = f"esmfold:{os.getpid()}:{id(self)}"
        self._evaluation_lock = threading.RLock()
        self._inference_kernel = None

    def evaluate(self, inputs=None, runtime_config=None):
        with self._evaluation_lock:
            return super().evaluate(inputs, runtime_config)

    def _local_inference(self, request):
        from astrolabe.evaluators.esmfold_stages import evaluator_stages
        from astrolabe.evaluators.runtime_stages import DirectKernel
        from astrolabe.runtime.loading import StagedLoader

        with self._evaluation_lock:
            if self._inference_kernel is None:
                def place(prepared):
                    model = self._place_model(prepared)
                    return evaluator_stages(model, device=self._device), {
                        "backend": "esmfold", "device": str(self._device),
                        "load_metadata": self._load_metadata,
                    }

                self._inference_kernel = DirectKernel(
                    {}, StagedLoader(lambda _: self._prepare_model_cpu(), place),
                    torch_device=self._device or "auto",
                    lifecycle={"backend": "esmfold", "residency": "resident_model",
                               "device": self._device},
                )
            return self._inference_kernel(request)

    def _emit_load_evidence(self, event):
        """Optional host-only trace of actual loading, never constructor counts.

        An injected model has no local loading evidence. Shutdown is only a
        request; process-exit/resource-release evidence belongs to its owner.
        """
        target = os.environ.get("ASTROLABE_MODEL_LIFECYCLE_PATH")
        if not target or not self._loader_attempts:
            return
        record = {
            "schema": "astrolabe.esmfold-load-evidence/v1",
            "event": event, "time": time.time(), "pid": os.getpid(),
            "backend": "esmfold", "model_id": self._load_identity,
            "model_name": self._model_name, "device": self._device,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "loader_attempt_count": self._loader_attempts,
            "loader_count": self._successful_loads,
            "placement_count": self._placements,
            "load_metadata": self._load_metadata,
        }
        content = (json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode()
        descriptor = os.open(
            target, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600,
        )
        try:
            if os.write(descriptor, content) != len(content):
                raise OSError("Incomplete ESMFold lifecycle trace write")
        finally:
            os.close(descriptor)

    def shutdown(self) -> None:
        self._emit_load_evidence("shutdown_requested")
        if self._inference_kernel is not None:
            self._inference_kernel.close()
        shutdown_model_handles(self._model)

    def _ensure_model(self):
        if self._model is not None:
            if "model_state" not in self._load_metadata and any(
                callable(getattr(self._model, name, None))
                for name in ("named_parameters", "named_buffers")
            ):
                self._load_metadata["model_state"] = _model_tensor_inventory(self._model)
            self._emit_load_evidence("model_reused")
            return self._model
        return self._place_model(self._prepare_model_cpu())

    def _prepare_model_cpu(self):
        """Read weights, construct and populate the model in the original order."""
        if self._model is not None:
            return self._ensure_model()
        self._load_metadata = {
            "clock_anchor": {
                "time_s": time.time(),
                "perf_counter_s": time.perf_counter(),
                "process_time_s": time.process_time(),
            },
            "timing_scope": "host_calls",
            "stages": {},
        }
        with _record_load_stage(self._load_metadata, "cpu_prepare"):
            model_name = self._model_name
            if model_name.endswith(".pt"):  # local, treat as filepath
                model_path = Path(model_name).expanduser()
                if not model_path.exists():
                    raise FileNotFoundError(
                        "ESMFold weights not found at "
                        f"{model_path}. Provide a local .pt file path for `model_name`."
                    )
            else:
                cache_dir = Path(torch.hub.get_dir()) / "checkpoints"
                model_path = cache_dir / f"{model_name}.pt"
                if not model_path.exists():
                    raise FileNotFoundError(
                        "ESMFold weights are missing and auto-download is disabled. "
                        f"Expected cached checkpoint at {model_path}. "
                        "Download it ahead of time or set `model_name` to an explicit local .pt path."
                    )

            with _record_load_stage(self._load_metadata, "torch_load"):
                self._loader_attempts += 1
                model_data = torch.load(str(model_path), map_location="cpu", weights_only=False)
            cfg = model_data["cfg"]["model"]
            model_state = _adapt_esmfold_state_dict_for_openfold(model_data["model"])

            ESMFold = _load_esmfold_model_class()
            with _record_load_stage(self._load_metadata, "model_constructor"):
                model = ESMFold(esmfold_config=cfg)
            expected_keys = set(model.state_dict().keys())
            found_keys = set(model_state.keys())

            missing_essential_keys = []
            for missing_key in expected_keys - found_keys:
                if not missing_key.startswith("esm."):
                    missing_essential_keys.append(missing_key)
            if missing_essential_keys:
                raise RuntimeError(
                    f"Keys '{', '.join(missing_essential_keys)}' are missing."
                )

            with _record_load_stage(self._load_metadata, "load_state_dict"):
                model.load_state_dict(model_state, strict=False)
            self._load_metadata["model_state"] = _model_tensor_inventory(model)
            self._load_metadata["checkpoint_path"] = str(model_path.resolve())
            self._successful_loads += 1
        self._emit_load_evidence("cpu_weights_loaded")
        return model

    def _place_model(self, model):
        """Keep legacy device selection and eval().to() order at placement."""
        if self._model is not None:
            return self._ensure_model()
        with _record_load_stage(self._load_metadata, "device_placement"):
            device = self._device
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self._device = device

            placed_model = model.eval().to(device)
            self._load_metadata["model_state"] = _model_tensor_inventory(placed_model)
            self._model = placed_model
            self._placements += 1
        self._emit_load_evidence("device_placement_completed")
        return self._model

    def prepare_input(self, inputs, runtime_config):
        input_feat = {}
        for k, v in inputs.items():
            if not all(
                global_config.protomers[c].type == ProtomerTypes.PROTEIN for c in k
            ):
                raise ValueError("ESMFold can only handle protein sequences")
            input_feat[k] = ":".join(c.sequence for c in v.protein_chains)
        return input_feat

    def prepare_output(self, output, inputs, runtime_config):
        return output

    @local_evaluator_guard("esmfold", "_model")
    def run_function(self, input_feat):
        with torch.no_grad():
            names = list(input_feat.keys())
            model = self._model
            sequences = [input_feat[n] for n in names]
            infer_with_outputs = getattr(model, "infer_with_outputs", None)
            if getattr(model, "_client", None) is not None and callable(infer_with_outputs):
                result = infer_with_outputs(sequences)
            else:
                result = self._local_inference({
                    "operation": "infer", "sequences": sequences,
                    "return_residue_plddt": True, "return_pdb": True,
                })
            output, residue_plddt = result["output"], result["residue_plddt"]
            pdbs = result["pdbs"]
        structure_path = structure_directory(global_config.out_dir, getattr(self, "alias", None))
        results = defaultdict(dict)
        for i, oligo_name in enumerate(names):
            jobname = f"step_{global_config.step:03d}_{oligo_name}"
            with open(structure_path / f"{jobname}_model.pdb", "w") as f:
                f.write(renumber_each_chain(pdbs[i]))
            project_structure(structure_path / f"{jobname}_model.pdb", global_config.out_dir, getattr(self, "alias", None))
            results["complex_paths"][oligo_name] = (
                structure_path / f"{jobname}_model.pdb"
            )
            mask = output["chain_index"][i] != -1
            per_residue = (
                residue_plddt[i] if residue_plddt is not None
                else output["plddt"][i, mask].mean(-1)
            ).cpu().numpy() / 100
            results["per_proto_scores"][oligo_name] = per_residue
            res_idx = 0
            results["per_proto_scores"] = defaultdict(list)
            for chain_id in oligo_name:
                chain_len = global_config.protomers[chain_id].length
                results["per_proto_scores"][chain_id].append(
                    per_residue[res_idx : res_idx + chain_len]
                )
                res_idx += chain_len
            results["plddt"][oligo_name] = output["mean_plddt"][i].cpu().numpy() / 100
            results["ptm"][oligo_name] = output["ptm"][i].cpu().numpy()
            results["pae"][oligo_name] = (
                output["predicted_aligned_error"][i][mask][:, mask].cpu().numpy()
            )

        return StandardScores(names=names, **results)
