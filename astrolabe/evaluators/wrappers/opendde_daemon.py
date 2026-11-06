from __future__ import annotations

import importlib
import os
import subprocess
import tempfile
from dataclasses import replace
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
            verify_vendor(self.settings["repo_path"])
            batch_inference = importlib.import_module("runner.batch_inference")
            inference = importlib.import_module("runner.inference")
        self.batch_inference = batch_inference
        self.inference = inference
        self._initial_dump = tempfile.TemporaryDirectory(prefix="astrolabe-opendde-")
        self.runner = batch_inference.get_default_runner(
            **_runner_kwargs(self.settings, dump_dir=self._initial_dump.name)
        )

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
        self.inference.infer_predict(self.runner, configs)
        return {
            "input_json": preprocessed,
            "out_dir": out_dir,
        }


def create_handler(settings: Mapping[str, Any]) -> OpenDDEHandler:
    return OpenDDEHandler(settings)
