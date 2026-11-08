# SPDX-License-Identifier: Apache-2.0
# Native inference loop adapted from OpenDDE, Copyright (c) 2026 Aureka AI Research.
from __future__ import annotations

import importlib
import json
import os
import random
import subprocess
import tempfile
import time
import traceback
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping


OPENDDE_PIN = "266ce4c49d492ad1077866000d83704999985f46"


def verify_vendor(repo_path: str | os.PathLike[str]) -> Path:
    repo = Path(repo_path).expanduser().resolve()
    required = (
        repo / "runner" / "batch_inference.py",
        repo / "runner" / "inference.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"OpenDDE source is missing: {missing}")
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"OpenDDE checkout is not a Git repository: {repo}")
    top_level = Path(result.stdout.strip()).resolve()
    if top_level != repo:
        raise RuntimeError(
            f"OpenDDE path is not a Git root: {repo}; resolved to {top_level}"
        )
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != OPENDDE_PIN:
        raise RuntimeError(f"OpenDDE checkout is {head}, expected {OPENDDE_PIN}.")
    batch_text = required[0].read_text(encoding="utf-8")
    inference_text = required[1].read_text(encoding="utf-8")
    if "def get_default_runner(" not in batch_text or "def preprocess_input(" not in batch_text:
        raise RuntimeError("Pinned OpenDDE batch inference API is unavailable.")
    if "class InferenceRunner" not in inference_text or "def infer_predict(" not in inference_text:
        raise RuntimeError("Pinned OpenDDE resident inference API is unavailable.")
    return repo


def _runner_kwargs(
    settings: Mapping[str, Any], *, dump_dir: str
) -> dict[str, Any]:
    return {
        "seeds": settings.get("seeds"),
        "dump_dir": dump_dir,
        "n_cycle": int(settings.get("num_cycles", 10)),
        "n_step": int(settings.get("num_steps", 200)),
        "n_sample": int(settings.get("num_samples", 5)),
        "dtype": str(settings.get("dtype", "fp32")),
        "model_name": str(settings.get("model_name", "opendde_v1")),
        "load_checkpoint_path": str(settings.get("load_checkpoint_path") or ""),
        "use_msa": bool(settings.get("use_msa", False)),
        "trimul_kernel": str(settings.get("triangle_multiplicative", "auto")),
        "triatt_kernel": str(settings.get("triangle_attention", "auto")),
        "enable_cache": bool(settings.get("enable_cache", True)),
        "enable_fusion": bool(settings.get("enable_fusion", True)),
        "enable_tf32": bool(settings.get("enable_tf32", True)),
        "deterministic": bool(settings.get("deterministic", False)),
        "use_template": bool(settings.get("use_template", False)),
        "use_rna_msa": bool(settings.get("use_rna_msa", False)),
        "need_atom_confidence": bool(settings.get("need_atom_confidence", False)),
        "kalign_binary_path": settings.get("kalign_binary_path"),
        "use_tfg_guidance": bool(settings.get("use_tfg_guidance", False)),
        "foldcp_mode": str(settings.get("foldcp_mode", "single")),
        "foldcp_size_dp": int(settings.get("foldcp_size_dp", 1)),
        "foldcp_size_cp": int(settings.get("foldcp_size_cp", 1)),
        "foldcp_devices": str(settings.get("foldcp_devices", "") or ""),
        "foldcp_metrics_jsonl": "",
    }


@dataclass
class _OpenDDESample:
    batches: Any
    seed: int
    num_data: int
    sample_name: str = "unknown"
    data: Any = None
    atom_array: Any = None
    device_data: Any = None
    prediction: Any = None
    n_token: Any = None
    data_error: str = ""
    started: float = 0.0


class OpenDDEHandler:
    def __init__(
        self,
        settings: Mapping[str, Any],
        *,
        batch_inference: Any | None = None,
        inference: Any | None = None,
    ) -> None:
        self.settings = dict(settings)
        if batch_inference is None or inference is None:
            source = verify_vendor(self.settings["repo_path"])
            batch_inference = importlib.import_module("runner.batch_inference")
            inference = importlib.import_module("runner.inference")
            for module in (batch_inference, inference):
                if not Path(module.__file__).resolve().is_relative_to(source / "runner"):
                    raise RuntimeError("OpenDDE imported unexpected runner source")
        self.batch_inference = batch_inference
        self.inference = inference
        self._initial_dump = tempfile.TemporaryDirectory(prefix="astrolabe-opendde-")
        from astrolabe.evaluators.runtime_stages import DirectKernel

        self._kernel = DirectKernel(
            self.settings, self._load, torch_device="auto", lifecycle={
                "backend": "opendde", "residency": "resident_model",
                "device": f"cuda:{int(os.environ['LOCAL_RANK'])}" if "LOCAL_RANK" in os.environ else None,
                "staging": "prepare_complete_model_finalize",
            },
        )

    def _load(self, settings):
        batch = self.batch_inference
        download = getattr(batch, "download_inference_cache", None)
        if download is not None:
            batch.download_inference_cache = self._verify_local_assets
        try:
            self.runner = batch.get_default_runner(
                **_runner_kwargs(settings, dump_dir=self._initial_dump.name)
            )
        finally:
            if download is not None:
                batch.download_inference_cache = download
        return self.stages(), {
            "backend": "opendde", "residency": "resident_model",
            "device": str(self.runner.device),
        }

    def _verify_local_assets(self, configs):
        checkpoint = self.inference.resolve_checkpoint_path(configs)
        if configs.model_name != self.settings.get("model_name", "opendde_v1"):
            raise RuntimeError("OpenDDE model identity changed")
        expected = self.settings.get("load_checkpoint_path")
        if expected and Path(checkpoint).resolve() != Path(expected).expanduser().resolve():
            raise RuntimeError("OpenDDE checkpoint identity changed")
        paths = [checkpoint, configs.data.ccd_components_file,
                 configs.data.ccd_components_rdkit_mol_file]
        if configs.use_template:
            paths.extend([configs.data.template.obsolete_pdbs_path,
                          configs.data.template.release_dates_path])
        for path in paths:
            if not Path(path).expanduser().is_file():
                raise FileNotFoundError(f"OpenDDE requires a local asset: {path}")

    @property
    def metadata(self):
        return self._kernel.metadata

    def stages(self):
        from astrolabe.model_execution import EvaluatorStages

        resource = "cpu" if str(self.runner.device) == "cpu" else "gpu"
        return EvaluatorStages(
            self._prepare_sample, self._infer_sample, self._finalize_sample,
            resource=resource, prepare_resource=resource, finalize_resource=resource,
            pipeline_slots=1,
        )

    def _pending(self, value, sample):
        from astrolabe.inference_pipeline import Pending

        torch = self.inference.torch
        event = None
        completion = lambda: None
        if str(self.runner.device) != "cpu":
            with torch.cuda.device(self.runner.device):
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream(self.runner.device))
            completion = event.synchronize
        return Pending(value, completion, keepalive=(self.runner, sample, event))

    def _prepare_sample(self, sample):
        native = self.inference
        batch = next(sample.batches)
        sample.started = time.time()
        sample.data, sample.atom_array, sample.data_error = batch[0]
        data = sample.data
        sample.sample_name = data["sample_name"]
        if not sample.data_error:
            native.logger.info(
                f"[Rank {native.DIST_WRAPPER.rank} ({data['sample_index'] + 1}/{sample.num_data})] "
                f"{sample.sample_name} [seed:{sample.seed}]: "
                f"N_asym {data['N_asym'].item()}, N_token {data['N_token'].item()}, "
                f"N_atom {data['N_atom'].item()}, N_msa {data['N_msa'].item()}"
            )
            configs = native.update_inference_configs(
                self.runner.configs, data["N_token"].item()
            )
            data["input_feature_dict"]["inference_seed"] = native.torch.tensor(
                int(sample.seed), dtype=native.torch.long
            )
            self.runner.update_model_configs(configs)
            sample.n_token = native.infer_n_token(data)
            with native.torch.no_grad():
                sample.device_data = native.to_device(data, self.runner.device)
        return self._pending(sample, sample)

    def _infer_sample(self, sample):
        native = self.inference
        torch = native.torch
        if not sample.data_error:
            precision = {"fp32": torch.float32, "bf16": torch.bfloat16,
                         "fp16": torch.float16}[self.runner.configs.dtype]
            amp = (torch.autocast(device_type="cuda", dtype=precision)
                   if torch.cuda.is_available() else nullcontext())
            with torch.no_grad(), amp, native.measure_foldcp_stage(
                task_id="task0", stage_name="model_forward",
                foldcp_config=self.runner.foldcp_config,
                recorder=self.runner.foldcp_recorder,
                sample_name=sample.sample_name,
                n_token=sample.n_token,
            ):
                sample.prediction, _, _ = self.runner.model(
                    input_feature_dict=sample.device_data["input_feature_dict"],
                    label_full_dict=None, label_dict=None, mode="inference",
                )
        return self._pending(sample, sample)

    def _finalize_sample(self, sample):
        native = self.inference
        if sample.data_error:
            native.logger.error(f"Data error for {sample.sample_name}: {sample.data_error}")
            self._write_error(sample.sample_name, sample.data_error)
        else:
            if not (self.runner.foldcp_config.enabled and native.DIST_WRAPPER.rank != 0):
                self.runner.dumper.dump(
                    group_name="", pdb_id=sample.sample_name, seed=sample.seed,
                    pred_dict=sample.prediction, atom_array=sample.atom_array,
                    entity_poly_type={k: v for k, v in sample.data["entity_poly_type"].items()
                                      if v != "non-polymer"},
                )
            native.logger.info(
                f"[Rank {native.DIST_WRAPPER.rank}] {sample.sample_name} [seed:{sample.seed}] succeeded. "
                f"Model forward time: {time.time() - sample.started:.2f}s. "
                f"Results saved to {self.runner.configs.dump_dir}"
            )
        return self._pending(None, sample)

    def _write_error(self, sample_name, message):
        with open(Path(self.runner.error_dir) / f"{sample_name}.txt", "a", encoding="utf-8") as handle:
            handle.write(message)

    def _predict(self):
        native = self.inference
        configs = self.runner.configs
        native.logger.info(f"Loading data from {configs.input_json_path}")
        with open(configs.input_json_path, encoding="utf-8") as handle:
            jobs = json.load(handle)
        if not isinstance(jobs, list) or not jobs:
            raise ValueError(
                f"Input JSON must be a non-empty top-level list, got {type(jobs).__name__} "
                f"from {configs.input_json_path}"
            )
        for job in jobs:
            name = job.get("name")
            if not isinstance(name, str) or not name or Path(name).name != name or name in (".", ".."):
                raise ValueError("invalid OpenDDE job name")
        if configs.seeds:
            seeds = [int(seed) for seed in configs.seeds]
        elif jobs[0].get("modelSeeds"):
            seeds = [int(seed) for seed in jobs[0]["modelSeeds"]]
        else:
            seeds = [random.randint(1, 65536)]
        try:
            dataloader = native.get_inference_dataloader(configs=configs)
        except Exception as error:
            message = f"Dataloader initialization failed: {error}\n{traceback.format_exc()}"
            native.logger.error(message)
            self._write_error("error", message)
            raise RuntimeError("OpenDDE reported inference errors") from error

        started = time.time()
        with native.disable_cudnn_benchmark():
            for seed in seeds:
                native.seed_everything(seed=seed, deterministic=configs.deterministic)
                native.cleanup_cuda_memory()
                seed_started = time.time()
                batches = iter(dataloader)
                # The pinned loader has batch_size=1 and a sized distributed sampler.
                # Pull each batch inside prepare, after the preceding dump and RNG use.
                for _ in range(len(dataloader)):
                    sample = _OpenDDESample(batches, seed, len(dataloader.dataset))
                    try:
                        self._kernel(sample)
                    except Exception as error:
                        message = (f"[Rank {native.DIST_WRAPPER.rank}] {sample.sample_name} failed: "
                                   f"{error}\n{traceback.format_exc()}")
                        native.logger.error(message)
                        self._write_error(sample.sample_name, message)
                        if self.metadata["runtime_lifecycle"]["state"] == "quarantined":
                            raise
                    finally:
                        del sample
                        native.cleanup_cuda_memory(collect_garbage=False)
                del batches
                native.cleanup_cuda_memory()
                native.logger.info(
                    f"[Rank {native.DIST_WRAPPER.rank}] Seed {seed} completed in "
                    f"{time.time() - seed_started:.2f}s."
                )
        error_dir = Path(self.runner.error_dir)
        try:
            if error_dir.exists() and not any(error_dir.iterdir()):
                error_dir.rmdir()
        except OSError:
            pass
        native.logger.info(
            f"[Rank {native.DIST_WRAPPER.rank}] Job completed in {time.time() - started:.2f}s."
        )
        self._validate_outputs(jobs, seeds)

    def _validate_outputs(self, jobs, seeds):
        # Reuse a60721d's all-requested-samples check, not just one successful CIF.
        if self.runner.foldcp_config.enabled and self.inference.DIST_WRAPPER.rank != 0:
            return
        for job in jobs:
            name = job["name"]
            for seed in seeds:
                directory = Path(self.runner.dump_dir) / name / f"seed_{int(seed)}" / "predictions"
                summaries = sorted(directory.glob(f"{name}_summary_confidence_sample_*.json"))
                if len(summaries) != int(self.settings.get("num_samples", 5)) or any(
                    not path.with_name(path.name.replace("_summary_confidence", "")).with_suffix(".cif").is_file()
                    for path in summaries
                ):
                    raise RuntimeError(f"OpenDDE missing outputs for {name} seed {seed}")
        error_dir = Path(self.runner.error_dir)
        if error_dir.exists() and any(error_dir.iterdir()):
            raise RuntimeError("OpenDDE reported inference errors")

    def _bind_output_dir(self, out_dir: str, metrics_jsonl: str) -> None:
        configs = self.runner.configs
        configs.dump_dir = out_dir
        self.runner.dump_dir = out_dir
        self.runner.error_dir = str(Path(out_dir) / "ERR")
        Path(self.runner.dump_dir).mkdir(parents=True, exist_ok=True)
        Path(self.runner.error_dir).mkdir(parents=True, exist_ok=True)
        self.runner.init_dumper(
            need_atom_confidence=bool(
                getattr(configs, "need_atom_confidence", False)
            ),
            sorted_by_ranking_score=bool(
                getattr(configs, "sorted_by_ranking_score", True)
            ),
        )
        previous = self.runner.foldcp_recorder
        if metrics_jsonl or getattr(previous, "jsonl_path", ""):
            self.runner.foldcp_config = replace(
                self.runner.foldcp_config,
                metrics_jsonl=metrics_jsonl,
            )
            recorder_type = importlib.import_module(
                "opendde.distributed.foldcp.metrics"
            ).FoldCPBenchmarkRecorder
            self.runner.foldcp_recorder = recorder_type(
                metrics_jsonl,
                rank=previous.rank,
                write_rank_sidecar=previous.write_rank_sidecar,
            )
        os.environ["OPENDDE_FOLDCP_METRICS_JSONL"] = metrics_jsonl

    def __call__(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if payload.get("operation") == "describe":
            return self._kernel.handle(payload)
        from astrolabe.model_execution import _KERNEL_RANDOM_LOCK

        # Seeding, loader workers, config updates and dumping share the owner's RNG.
        with _KERNEL_RANDOM_LOCK:
            return self._run_request(payload)

    def _run_request(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if self.metadata["runtime_lifecycle"]["state"] == "quarantined":
            raise RuntimeError("Kernel is quarantined after device failure; rebuild its owner")
        input_json = str(Path(payload["input_json"]).expanduser().resolve())
        out_dir = str(Path(payload["out_dir"]).expanduser().resolve())
        metrics_jsonl = str(payload.get("foldcp_metrics_jsonl") or "")
        self._bind_output_dir(out_dir, metrics_jsonl)
        preprocessed = self.batch_inference.preprocess_input(
            input_json,
            out_dir=out_dir,
            use_msa=bool(self.settings.get("use_msa", False)),
            use_template=bool(self.settings.get("use_template", False)),
            use_rna_msa=bool(self.settings.get("use_rna_msa", False)),
            hmmsearch_binary_path=self.settings.get("hmmsearch_binary_path"),
            hmmbuild_binary_path=self.settings.get("hmmbuild_binary_path"),
            seqres_database_path=self.settings.get("seqres_database_path"),
            nhmmer_binary_path=self.settings.get("nhmmer_binary_path"),
            hmmalign_binary_path=self.settings.get("hmmalign_binary_path"),
            hmmbuild_rna_binary_path=self.settings.get("hmmbuild_rna_binary_path"),
            ntrna_database_path=self.settings.get("ntrna_database_path"),
            rfam_database_path=self.settings.get("rfam_database_path"),
            rna_central_database_path=self.settings.get("rna_central_database_path"),
            nhmmer_n_cpu=self.settings.get("nhmmer_n_cpu"),
        )
        configs = self.runner.configs
        configs.dump_dir = out_dir
        configs["input_json_path"] = preprocessed
        self._predict()
        return {
            "input_json": preprocessed,
            "out_dir": out_dir,
        }


def create_handler(settings: Mapping[str, Any]):
    if os.environ.get("ASTROLABE_RESOURCE_SOCKET") and (
        int(settings.get("foldcp_size_cp", 1)) > 1
        or int(settings.get("foldcp_size_dp", 1)) > 1
        or settings.get("foldcp_devices")
    ):
        raise ValueError("OpenDDE FoldCP device routing requires explicit multi-device admission")

    return OpenDDEHandler(settings)


def main():
    batch = importlib.import_module("runner.batch_inference")
    original = batch.inference_jsons

    def staged_inference_jsons(json_file, out_dir="./output", use_msa=True, **options):
        names = {"n_cycle": "num_cycles", "n_step": "num_steps", "n_sample": "num_samples",
                 "trimul_kernel": "triangle_multiplicative", "triatt_kernel": "triangle_attention"}
        settings = {names.get(key, key): value for key, value in options.items()}
        settings.update(repo_path=str(Path(batch.__file__).resolve().parents[1]), use_msa=use_msa)
        handler = create_handler(settings)
        path = Path(json_file)
        files = list(path.rglob("*.json")) if path.is_dir() else [path]
        for input_json in files:
            handler({"input_json": str(input_json), "out_dir": str(out_dir),
                     "foldcp_metrics_jsonl": options.get("foldcp_metrics_jsonl", "")})

    batch.inference_jsons = staged_inference_jsons
    try:
        batch.opendde_cli()
    finally:
        batch.inference_jsons = original


if __name__ == "__main__":
    main()
