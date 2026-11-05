from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from io import StringIO
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


_DAEMON_FACTORY = "astrolabe.evaluators.wrappers.esmfold2:daemon_factory"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _path_with_optional_src(path: str | Path | None) -> str | None:
    if path is None:
        return None
    root = Path(path).expanduser().resolve()
    src = root / "src"
    if src.exists():
        return str(src)
    return str(root)


def _prepend_path(path: str | None) -> None:
    if path and path not in sys.path:
        sys.path.insert(0, path)


def _resolve_model_source(model_name: str) -> tuple[str, bool]:
    pinned_snapshot = os.environ.get("GH_ESMFOLD2_MODEL_SNAPSHOT")
    if pinned_snapshot and model_name == "biohub/ESMFold2":
        return pinned_snapshot, True
    return model_name, False


def _resolve_esmc_snapshot_path() -> Path | None:
    esmc_snapshot = os.environ.get("GH_ESMFOLD2_ESMC_SNAPSHOT")
    if not esmc_snapshot:
        return None
    esmc_path = Path(esmc_snapshot).expanduser().resolve()
    for required in ("config.json", "model.safetensors.index.json"):
        if not (esmc_path / required).is_file():
            raise FileNotFoundError(
                f"Pinned ESMFold2 ESMC snapshot is missing {required}: {esmc_path}"
            )
    return esmc_path


def _model_load_kwargs(payload: dict[str, Any], *, pinned_snapshot: bool) -> dict[str, Any]:
    model_kwargs: dict[str, Any] = {
        "esmc_precision": payload.get("esmc_precision") or "bf16",
    }
    cache_dir = payload.get("cache_dir")
    if cache_dir:
        model_kwargs["cache_dir"] = str(Path(cache_dir).expanduser())
    offline = bool(payload.get("local_files_only", False)) or pinned_snapshot
    if offline:
        model_kwargs["local_files_only"] = True
    if _resolve_esmc_snapshot_path() is not None:
        model_kwargs["local_files_only"] = True
    elif offline:
        raise FileNotFoundError(
            "Offline ESMFold2 requires GH_ESMFOLD2_ESMC_SNAPSHOT; the nested "
            "ESMC-6B loader does not inherit local_files_only"
        )
    return model_kwargs


def _load_esmfold2_model(
    model_cls: Any,
    model_name: str,
    model_kwargs: dict[str, Any],
    *,
    esmc_snapshot_path: Path | None,
) -> Any:
    if esmc_snapshot_path is None:
        return model_cls.from_pretrained(model_name, **model_kwargs).eval()

    esmc_precision = model_kwargs.get("esmc_precision") or "bf16"
    model = model_cls.from_pretrained(
        model_name,
        load_esmc=False,
        **model_kwargs,
    )
    model.config.esmc_id = str(esmc_snapshot_path)
    model.load_esmc(str(esmc_snapshot_path), precision=esmc_precision)
    return model.eval()


def _ccd_cache_dir(
    model_source: str,
    *,
    pinned_snapshot: bool,
    local_files_only: bool = False,
) -> Path | None:
    """Return the local model directory that owns the pinned CCD pickle."""
    source = Path(model_source).expanduser()
    if not pinned_snapshot and not source.is_dir():
        if local_files_only:
            explicit_ccd = os.environ.get("ESMCFOLD_CCD_PATH")
            if not explicit_ccd:
                raise FileNotFoundError(
                    "Offline ESMFold2 requires GH_ESMFOLD2_MODEL_SNAPSHOT or "
                    "ESMCFOLD_CCD_PATH"
                )
            ccd = Path(explicit_ccd).expanduser().resolve()
            if not ccd.is_file() or ccd.stat().st_size == 0:
                raise FileNotFoundError(
                    f"Explicit ESMFold2 CCD pickle is missing or empty: {ccd}"
                )
        return None
    source = source.resolve()
    ccd = source / "ccd.pkl"
    if not ccd.is_file() or ccd.stat().st_size == 0:
        raise FileNotFoundError(f"Pinned ESMFold2 CCD pickle is missing or empty: {ccd}")
    return source


def _to_python(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def _mean_float(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().float().cpu()
    if hasattr(value, "numel") and value.numel() != 1:
        value = value.mean()
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def _model_identity(payload: dict[str, Any]) -> dict[str, Any]:
    model_name, pinned_snapshot = _resolve_model_source(payload["model_name"])
    esmc_snapshot_path = _resolve_esmc_snapshot_path()
    model_kwargs = _model_load_kwargs(payload, pinned_snapshot=pinned_snapshot)
    ccd_cache_dir = _ccd_cache_dir(
        model_name,
        pinned_snapshot=pinned_snapshot,
        local_files_only=bool(payload.get("local_files_only", False)),
    )
    return {
        "model_name": model_name,
        "requested_model_name": payload["model_name"],
        "pinned_snapshot": pinned_snapshot,
        "device": payload.get("device"),
        "model_kwargs": model_kwargs,
        "esmc_snapshot_path": (
            str(esmc_snapshot_path) if esmc_snapshot_path is not None else None
        ),
        "ccd_cache_dir": str(ccd_cache_dir) if ccd_cache_dir is not None else None,
        "chunk_size": payload.get("chunk_size"),
        "esm_repo_path": payload.get("esm_repo_path"),
        "transformers_repo_path": payload.get("transformers_repo_path"),
    }


def _fold_request(payload: dict[str, Any], identity: dict[str, Any]) -> dict[str, Any]:
    fold_kwargs = {
        "num_loops": int(payload.get("num_loops", 3)),
        "num_sampling_steps": int(payload.get("num_sampling_steps", 50)),
        "num_diffusion_samples": int(payload.get("num_diffusion_samples", 1)),
        "seed": payload.get("seed"),
        "lm_dropout": payload.get("lm_dropout"),
        "msa_max_depth": payload.get("msa_max_depth"),
        "msa_column_mask_rate": float(payload.get("msa_column_mask_rate", 0.1)),
    }
    for key in ("noise_scale", "step_scale", "max_inference_sigma", "lm_mask_pct"):
        if payload.get(key) is not None:
            fold_kwargs[key] = payload[key]
    return {
        "model_identity": identity,
        "output_dir": payload["output_dir"],
        "jobs": payload["jobs"],
        "fold_kwargs": fold_kwargs,
    }


def _fold_jobs(
    model: Any,
    builder: Any,
    request: dict[str, Any],
) -> dict[str, Any]:
    from esm.models.esmfold2 import ProteinInput, StructurePredictionInput

    output_dir = Path(request["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for job in request["jobs"]:
        name = str(job["name"])
        chains = [
            ProteinInput(id=str(chain["id"]), sequence=str(chain["sequence"]))
            for chain in job["chains"]
        ]
        prediction_input = StructurePredictionInput(sequences=chains)
        result = builder.fold(
            model,
            prediction_input,
            complex_id=name,
            **request["fold_kwargs"],
        )
        if isinstance(result, list):
            result = result[0]

        structure_path = output_dir / f"{name}_model.cif"
        structure_path.write_text(result.complex.to_mmcif(), encoding="utf-8")
        results.append(
            {
                "name": name,
                "structure_path": str(structure_path),
                "plddt": _to_python(result.plddt),
                "plddt_mean": _mean_float(result.plddt),
                "ptm": _mean_float(result.ptm),
                "iptm": _mean_float(result.iptm),
                "pae": _to_python(getattr(result, "pae", None)),
                "pde": _to_python(getattr(result, "pde", None)),
            }
        )

    identity = request["model_identity"]
    return {
        "model_name": identity["model_name"],
        "device": identity.get("device"),
        "jobs": results,
    }


@dataclass(frozen=True)
class _ESMFold2Worker:
    model: Any
    builder: Any
    identity: dict[str, Any]
    device: str

    def __call__(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.infer(payload)

    def infer(
        self, payload: dict[str, Any], *, capture_logs: bool = True
    ) -> dict[str, Any] | Any:
        if payload.get("model_identity") != self.identity:
            raise ValueError(
                "ESMFold2 request model settings do not match the loaded model owner"
            )
        if not capture_logs:
            result = _fold_jobs(self.model, self.builder, payload)
            result["device"] = self.device
            return result

        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            result = _fold_jobs(self.model, self.builder, payload)
        result["device"] = self.device
        return {
            "result": result,
            "stdout": stdout.getvalue(),
            "stderr": stderr.getvalue(),
        }


def daemon_factory(settings: dict[str, Any]) -> _ESMFold2Worker:
    identity = dict(settings["identity"])
    _prepend_path(_path_with_optional_src(identity["transformers_repo_path"]))
    _prepend_path(_path_with_optional_src(identity["esm_repo_path"]))

    import torch
    from esm.models.esmfold2 import ESMFold2InputBuilder
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

    device = identity.get("device") or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    identity["model_kwargs"] = dict(identity["model_kwargs"])
    esmc_snapshot_path = identity.get("esmc_snapshot_path")

    model = _load_esmfold2_model(
        ESMFold2Model,
        identity["model_name"],
        identity["model_kwargs"],
        esmc_snapshot_path=(
            Path(esmc_snapshot_path) if esmc_snapshot_path is not None else None
        ),
    )
    if device != "cpu":
        model = model.to(device)
    if identity.get("chunk_size") is not None and hasattr(model, "set_chunk_size"):
        model.set_chunk_size(identity["chunk_size"])

    ccd_cache_dir = identity.get("ccd_cache_dir")
    builder = ESMFold2InputBuilder(
        ccd_cache=Path(ccd_cache_dir) if ccd_cache_dir is not None else None
    )
    return _ESMFold2Worker(
        model=model,
        builder=builder,
        identity=identity,
        device=device,
    )


def _worker_run(payload: dict[str, Any]) -> dict[str, Any]:
    identity = _model_identity(payload)
    worker = daemon_factory({"identity": identity})
    return worker.infer(_fold_request(payload, identity), capture_logs=False)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Biohub ESMFold2 inference jobs.")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)

    payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    result = _worker_run(payload)
    Path(args.output_json).write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return 0


class ESMFold2Runner:
    def __init__(
        self,
        *,
        python_bin: str | None = None,
        esm_repo_path: str | None = None,
        transformers_repo_path: str | None = None,
        model_name: str = "biohub/ESMFold2",
        cache_dir: str | None = None,
        device: str | None = None,
        esmc_precision: str = "bf16",
        num_loops: int = 3,
        num_sampling_steps: int = 50,
        num_diffusion_samples: int = 1,
        seed: int | None = None,
        lm_dropout: float | None = 0.0,
        msa_max_depth: int | None = 1024,
        msa_column_mask_rate: float = 0.1,
        chunk_size: int | None = 64,
        local_files_only: bool = False,
        timeout_seconds: float | None = None,
        execution_mode: str = "daemon",
    ) -> None:
        if execution_mode not in {"daemon", "subprocess"}:
            raise ValueError(
                "ESMFold2 execution_mode must be 'daemon' or 'subprocess'."
            )
        self.python_bin = python_bin
        self.esm_repo_path = esm_repo_path
        self.transformers_repo_path = transformers_repo_path
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device
        self.esmc_precision = esmc_precision
        self.num_loops = num_loops
        self.num_sampling_steps = num_sampling_steps
        self.num_diffusion_samples = num_diffusion_samples
        self.seed = seed
        self.lm_dropout = lm_dropout
        self.msa_max_depth = msa_max_depth
        self.msa_column_mask_rate = msa_column_mask_rate
        self.chunk_size = chunk_size
        self.local_files_only = local_files_only
        self.timeout_seconds = timeout_seconds
        self.execution_mode = execution_mode
        self._daemon: Any = None
        self._daemon_settings: dict[str, Any] | None = None

    @classmethod
    def from_config(cls, config: Any) -> "ESMFold2Runner":
        if hasattr(config, "items"):
            container = dict(config)
        else:
            from omegaconf import OmegaConf

            container = OmegaConf.to_container(config, resolve=True)
        if not isinstance(container, dict):
            raise TypeError("ESMFold2 config must resolve to a mapping")
        allowed = {
            "python_bin",
            "esm_repo_path",
            "transformers_repo_path",
            "model_name",
            "cache_dir",
            "device",
            "esmc_precision",
            "num_loops",
            "num_sampling_steps",
            "num_diffusion_samples",
            "seed",
            "lm_dropout",
            "msa_max_depth",
            "msa_column_mask_rate",
            "chunk_size",
            "local_files_only",
            "timeout_seconds",
            "execution_mode",
        }
        return cls(**{k: v for k, v in container.items() if k in allowed})

    def _payload(
        self, jobs: list[dict[str, Any]], output_dir: Path
    ) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "device": self.device,
            "cache_dir": self.cache_dir,
            "local_files_only": self.local_files_only,
            "esm_repo_path": self.esm_repo_path,
            "transformers_repo_path": self.transformers_repo_path,
            "esmc_precision": self.esmc_precision,
            "num_loops": self.num_loops,
            "num_sampling_steps": self.num_sampling_steps,
            "num_diffusion_samples": self.num_diffusion_samples,
            "seed": self.seed,
            "lm_dropout": self.lm_dropout,
            "msa_max_depth": self.msa_max_depth,
            "msa_column_mask_rate": self.msa_column_mask_rate,
            "chunk_size": self.chunk_size,
            "output_dir": str(output_dir / "structures"),
            "jobs": jobs,
        }

    def _daemon_spec(self, identity: dict[str, Any]) -> Any:
        from astrolabe.model_daemon import DaemonSpec

        env = os.environ.copy()
        import_roots = [str(_repo_root())]
        if env.get("PYTHONPATH"):
            import_roots.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(import_roots)
        return DaemonSpec(
            backend="esmfold2",
            factory=_DAEMON_FACTORY,
            settings={"identity": identity},
            python_bin=self.python_bin or sys.executable,
            cwd=str(_repo_root()),
            env=env,
        )

    def _ensure_daemon(self, identity: dict[str, Any]) -> Any:
        if self._daemon is not None and self._daemon_settings == identity:
            return self._daemon
        self.shutdown()

        from astrolabe.model_daemon import OwnedModelDaemon

        self._daemon = OwnedModelDaemon(
            self._daemon_spec(identity),
            startup_timeout=300,
            request_timeout=self.timeout_seconds,
            shutdown_timeout=30,
            max_pending=4,
        )
        self._daemon_settings = identity
        return self._daemon

    @property
    def pid(self) -> int | None:
        return self._daemon.pid if self._daemon is not None else None

    def shutdown(self) -> None:
        daemon = self._daemon
        self._daemon = None
        self._daemon_settings = None
        if daemon is not None:
            daemon.shutdown()

    def _run_daemon(
        self,
        payload: dict[str, Any],
        identity: dict[str, Any],
        output_json: Path,
        stdout_log: Path,
        stderr_log: Path,
    ) -> dict[str, Any]:
        response = self._ensure_daemon(identity).request(
            {
                **payload,
                "model_identity": identity,
                "output_dir": payload["output_dir"],
                "jobs": payload["jobs"],
                "fold_kwargs": _fold_request(payload, identity)["fold_kwargs"],
            }
        )
        if not isinstance(response, dict) or "result" not in response:
            raise RuntimeError(f"Unexpected ESMFold2 daemon response: {response!r}")
        stdout_log.write_text(str(response.get("stdout") or ""), encoding="utf-8")
        stderr_log.write_text(str(response.get("stderr") or ""), encoding="utf-8")
        result = response["result"]
        output_json.write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
        return result

    def _run_subprocess(
        self,
        payload: dict[str, Any],
        *,
        input_json: Path,
        output_json: Path,
        stdout_log: Path,
        stderr_log: Path,
    ) -> dict[str, Any]:
        python_bin = self.python_bin or sys.executable
        if not shutil.which(python_bin) and not Path(python_bin).exists():
            raise FileNotFoundError(f"ESMFold2 python_bin not found: {python_bin}")
        env = os.environ.copy()
        pythonpath = [str(_repo_root())]
        if env.get("PYTHONPATH"):
            pythonpath.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pythonpath)
        cmd = [
            python_bin,
            "-m",
            "astrolabe.evaluators.wrappers.esmfold2",
            "--input-json",
            str(input_json),
            "--output-json",
            str(output_json),
        ]
        completed = subprocess.run(
            cmd,
            cwd=str(_repo_root()),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=self.timeout_seconds,
        )
        stdout_log.write_text(completed.stdout, encoding="utf-8")
        stderr_log.write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            tail = "\n".join(completed.stderr.splitlines()[-30:])
            raise RuntimeError(
                f"ESMFold2 subprocess failed with exit code {completed.returncode}. "
                f"See {stderr_log}. Stderr tail:\n{tail}"
            )
        if not output_json.exists():
            raise FileNotFoundError(f"ESMFold2 output JSON was not written: {output_json}")
        return json.loads(output_json.read_text(encoding="utf-8"))

    def run(self, jobs: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = self._payload(jobs, output_dir)
        identity = _model_identity(payload)
        input_json = output_dir / "esmfold2_input.json"
        output_json = output_dir / "esmfold2_output.json"
        stdout_log = output_dir / "esmfold2.stdout.log"
        stderr_log = output_dir / "esmfold2.stderr.log"
        input_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

        if self.execution_mode == "subprocess":
            return self._run_subprocess(
                payload,
                input_json=input_json,
                output_json=output_json,
                stdout_log=stdout_log,
                stderr_log=stderr_log,
            )
        return self._run_daemon(
            payload,
            identity,
            output_json,
            stdout_log,
            stderr_log,
        )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
