from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import os
import pathlib
import re
import subprocess
import sys
from argparse import Namespace
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from importlib import metadata as importlib_metadata
from functools import wraps
from typing import Any, Dict, Iterator, List, Mapping

from astrolabe.model_execution import device_operation


def _admitted(phase):
    def decorate(function):
        @wraps(function)
        def run(self, *args, **kwargs):
            if (os.environ.get("ASTROLABE_RESOURCE_SOCKET") and
                    self.device not in (None, "", "null", "None", "auto", "cpu", "cuda", "cuda:0")):
                # Both official ESM and legacy helpers use default .cuda(). A
                # score-model-only cuda:N hint cannot admit that whole graph.
                raise ValueError(
                    "Broker-admitted RAPiDock requires logical cuda:0; use "
                    "CUDA_VISIBLE_DEVICES to select the physical GPU"
                )
            # The official prepare_data path also runs ESM on CUDA. Its device
            # graph is not CPU-only even when the score model requests CPU.
            # Inside the JSONL kernel this reuses the existing local lease.
            with device_operation("rapidock", phase=phase):
                return function(self, *args, **kwargs)
        return run
    return decorate


OFFICIAL_RAPIDOCK_REVISION = "36b2f780cf60b6cd80c2ed128b79a7d5c3b29469"
OFFICIAL_INFERENCE_SHA256 = (
    "83f4a3dfe16b29d6f2d99755a265aaefbd82a4c5bf8aa59d4c9beca86b4c796d"
)
ESM2_MODEL_NAME = "esm2_t33_650M_UR50D"
ESM2_CONSTRUCTOR_PARAMETERS = {
    "num_layers",
    "embed_dim",
    "attention_heads",
    "alphabet",
    "token_dropout",
}
ESM2_EXPECTED_SHAPE = (33, 1280, 20)


@contextmanager
def backend_output_to_stderr() -> Iterator[None]:
    """Keep library output away from the JSONL stdout protocol."""

    stdout_fd: int | None = None
    saved_stdout: int | None = None
    try:
        stdout_fd = sys.stdout.fileno()
        stderr_fd = sys.stderr.fileno()
        sys.stdout.flush()
        sys.stderr.flush()
        saved_stdout = os.dup(stdout_fd)
        os.dup2(stderr_fd, stdout_fd)
    except (AttributeError, OSError, ValueError):
        if saved_stdout is not None:
            os.close(saved_stdout)
        saved_stdout = None

    try:
        with redirect_stdout(sys.stderr), redirect_stderr(sys.stderr):
            yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        if saved_stdout is not None and stdout_fd is not None:
            os.dup2(saved_stdout, stdout_fd)
            os.close(saved_stdout)


def _as_int(value: Any, default: int) -> int:
    if value in (None, "", "null", "None"):
        return int(default)
    return int(value)


def _as_bool(value: Any, default: bool) -> bool:
    if value in (None, "", "null", "None"):
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def _load_inference_class(repo_path: pathlib.Path):
    api_path = repo_path / "RAPiDock" / "API.py"
    if not api_path.is_file():
        raise FileNotFoundError(
            f"RAPiDock Python API not found at '{api_path}'. "
            "The configured source must provide RAPiDock/API.py with "
            "RAPiDockInference."
        )

    repo_text = os.fspath(repo_path)
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)
    importlib.invalidate_caches()
    try:
        module = importlib.import_module("RAPiDock.API")
    except Exception as exc:
        raise RuntimeError(
            f"Unable to import RAPiDockInference from '{api_path}': "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    inference_class = getattr(module, "RAPiDockInference", None)
    if not callable(inference_class):
        raise RuntimeError(
            f"'{api_path}' does not define a callable RAPiDockInference."
        )
    return inference_class


def _has_maintained_api_capability(repo_path: pathlib.Path) -> bool:
    api_path = repo_path / "RAPiDock" / "API.py"
    if not api_path.is_file():
        return False
    try:
        source = api_path.read_text(encoding="utf-8")
    except OSError:
        return False
    return re.search(
        r"(?m)^\s*(?:class\s+RAPiDockInference\b|RAPiDockInference\s*=)",
        source,
    ) is not None


def _read_git_revision(repo_path: pathlib.Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    revision = completed.stdout.strip()
    return revision or None


def _verify_official_source(repo_path: pathlib.Path) -> str:
    inference_path = repo_path / "inference.py"
    revision = _read_git_revision(repo_path)
    if revision is None:
        raise RuntimeError(
            f"Cannot verify RAPiDock source revision at '{repo_path}'. "
            "The original-pin runtime requires the official "
            "https://github.com/huifengzhao/RAPiDock checkout at "
            f"{OFFICIAL_RAPIDOCK_REVISION}."
        )
    if revision != OFFICIAL_RAPIDOCK_REVISION:
        raise RuntimeError(
            f"RAPiDock source at '{repo_path}' is revision {revision}. "
            "The original function runtime is restricted to official revision "
            f"{OFFICIAL_RAPIDOCK_REVISION}; use a maintained source with "
            "RAPiDock/API.py or add an explicit runtime for the new revision."
        )

    inference_sha256 = hashlib.sha256(inference_path.read_bytes()).hexdigest()
    if inference_sha256 != OFFICIAL_INFERENCE_SHA256:
        raise RuntimeError(
            f"RAPiDock inference.py at '{inference_path}' does not match "
            f"official revision {OFFICIAL_RAPIDOCK_REVISION} "
            f"(expected sha256 {OFFICIAL_INFERENCE_SHA256})."
        )

    try:
        dirty = subprocess.run(
            [
                "git",
                "-C",
                str(repo_path),
                "status",
                "--porcelain",
                "--untracked-files=no",
                "--",
                "inference.py",
                "utils",
                "dataset",
                "models",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise RuntimeError(
            f"Cannot verify tracked RAPiDock source files at '{repo_path}': {exc}"
        ) from exc
    if dirty.returncode != 0:
        detail = (dirty.stderr or dirty.stdout).strip()
        raise RuntimeError(
            f"Cannot verify tracked RAPiDock source files at '{repo_path}': {detail}"
        )
    if dirty.stdout.strip():
        raise RuntimeError(
            f"RAPiDock source at '{repo_path}' has modified tracked inference "
            "files. Restore the official pinned checkout before using the "
            "original function runtime."
        )
    return revision


def _load_official_inference_module(repo_path: pathlib.Path):
    inference_path = repo_path / "inference.py"
    repo_text = os.fspath(repo_path)
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)
    module_name = f"_astrolabe_rapidock_official_{OFFICIAL_RAPIDOCK_REVISION}"
    spec = importlib.util.spec_from_file_location(module_name, inference_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load RAPiDock inference module from '{inference_path}'.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    with backend_output_to_stderr():
        spec.loader.exec_module(module)
    return module


def _dock_options(params: Mapping[str, Any]) -> Dict[str, Any]:
    actual_steps = params.get("actual_steps")
    return {
        "N": _as_int(params.get("num_samples_bank"), 5),
        "batch_size": _as_int(params.get("batch_size"), 10),
        "no_final_step_noise": _as_bool(
            params.get("no_final_step_noise"), False
        ),
        "inference_steps": _as_int(params.get("inference_steps"), 20),
        "actual_steps": (
            None
            if actual_steps in (None, "", "null", "None")
            else int(actual_steps)
        ),
        "save_visualisation": _as_bool(
            params.get("save_visualisation"), False
        ),
        "scoring_function": params.get("scoring_function"),
        "fastrelax": _as_bool(params.get("fastrelax"), False),
        "cpu": _as_int(params.get("cpu"), 1),
        "conformation_type": str(params.get("conformation_type") or "E"),
        "conformation_partial": params.get("conformation_partial"),
    }


def _validate_task(task: Mapping[str, Any]) -> Dict[str, str]:
    fields = {
        key: task.get(key)
        for key in ("name", "protein_pdb", "peptide_pdb", "output_dir")
    }
    if any(not isinstance(value, str) or not value for value in fields.values()):
        raise ValueError(
            "Dock task must include non-empty name/protein_pdb/"
            "peptide_pdb/output_dir."
        )
    return {key: str(value) for key, value in fields.items()}


class RAPiDockRuntime:
    """Own the retained RAPiDock API object for one daemon process."""

    def __init__(
        self,
        *,
        repo_path: pathlib.Path,
        model_dir: pathlib.Path,
        ckpt: str,
        confidence_model_dir: str | None,
        confidence_ckpt: str,
        device: str | None,
    ) -> None:
        self.repo_path = repo_path
        self.model_dir = model_dir
        self.ckpt = ckpt
        self.confidence_model_dir = confidence_model_dir
        self.confidence_ckpt = confidence_ckpt
        self.device = device
        self._inference: Any | None = None
        self.load_count = 0
        self.load_attempts = 0
        self.load_error: str | None = None
        self.confidence_load_count = 0
        self.confidence_load_attempts = 0

    @property
    def model_ready(self) -> bool:
        return self._inference is not None

    def status(self) -> Dict[str, Any]:
        confidence_ready: bool | None = None
        if self.confidence_model_dir not in (None, "", "null", "None"):
            confidence_ready = (
                self._inference is not None
                and getattr(self._inference, "confidence_model", None) is not None
            )
            if confidence_ready and self.confidence_load_count == 0:
                self.confidence_load_count = 1
                self.confidence_load_attempts = max(
                    self.confidence_load_attempts, 1
                )
        status = {
            "runtime": "maintained-rapidock-api",
            "model_ready": self.model_ready,
            "load_count": self.load_count,
            "load_attempts": self.load_attempts,
            "confidence_model_ready": confidence_ready,
            "confidence_load_count": self.confidence_load_count,
            "confidence_load_attempts": self.confidence_load_attempts,
        }
        if self.load_error is not None:
            status["message"] = self.load_error
        return status

    @_admitted("load")
    def load(self) -> Any:
        if self._inference is not None:
            return self._inference

        self.load_attempts += 1
        try:
            inference_class = _load_inference_class(self.repo_path)
            with backend_output_to_stderr():
                inference = inference_class(
                    model_dir=str(self.model_dir),
                    ckpt=self.ckpt,
                    confidence_model_dir=self.confidence_model_dir,
                    confidence_ckpt=self.confidence_ckpt,
                    device=self.device,
                )
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"
            raise

        self._inference = inference
        self.load_count += 1
        self.load_error = None
        return inference

    def _require_inference(self) -> Any:
        if self._inference is not None:
            return self._inference
        detail = self.load_error or "model initialization has not completed"
        raise RuntimeError(f"RAPiDock model is not ready: {detail}")

    @staticmethod
    def _candidate_paths(
        *, name: str, output_dir: str, result: Mapping[str, Any]
    ) -> List[str]:
        payload = result.get(name)
        if not isinstance(payload, Mapping):
            raise RuntimeError(
                f"RAPiDock API returned no result payload for complex '{name}'."
            )
        raw_candidates = payload.get("candidate_paths")
        if not isinstance(raw_candidates, (list, tuple)) or not raw_candidates:
            raise RuntimeError(
                f"RAPiDock API returned no candidate paths for complex '{name}'."
            )

        candidates: List[str] = []
        for item in raw_candidates:
            if not isinstance(item, (str, os.PathLike)) or not str(item):
                raise RuntimeError(
                    f"RAPiDock API returned an invalid candidate path for '{name}'."
                )
            path = pathlib.Path(item).expanduser()
            if not path.is_absolute():
                path = pathlib.Path(output_dir) / path
            path = path.resolve()
            if not path.is_file():
                raise RuntimeError(
                    f"RAPiDock candidate does not exist for '{name}': {path}"
                )
            candidates.append(str(path))
        return candidates

    @_admitted("compute")
    def run(
        self, tasks: List[Mapping[str, Any]], params: Mapping[str, Any]
    ) -> List[Dict[str, Any]]:
        inference = self._require_inference()
        results: List[Dict[str, Any]] = []
        for raw_task in tasks:
            task = _validate_task(raw_task)
            name = task["name"]
            output_dir = task["output_dir"]
            options = _dock_options(params)
            if (
                options["scoring_function"] == "confidence"
                and self.confidence_model_dir not in (None, "", "null", "None")
                and getattr(inference, "confidence_model", None) is None
            ):
                self.confidence_load_attempts += 1

            with backend_output_to_stderr():
                api_result = inference.run_inference_strict(
                    output_dir=output_dir,
                    complex_name=name,
                    protein_description=task["protein_pdb"],
                    peptide_description=task["peptide_pdb"],
                    **options,
                )

            if not isinstance(api_result, Mapping):
                raise RuntimeError(
                    f"RAPiDock API returned an invalid result for complex '{name}'."
                )
            candidates = self._candidate_paths(
                name=name, output_dir=output_dir, result=api_result
            )
            results.append(
                {
                    "name": name,
                    "candidates": [
                        {"rank": rank, "peptide_pdb": path}
                        for rank, path in enumerate(candidates, start=1)
                    ],
                }
            )
        return results


def _rank_sort_key(path: pathlib.Path) -> tuple[int, str]:
    match = re.search(r"rank(\d+)", path.name)
    if match is None:
        return (sys.maxsize, path.name)
    return (int(match.group(1)), path.name)


def _collect_official_candidates(output_dir: str) -> List[str]:
    root = pathlib.Path(output_dir)
    preferred = [
        path for path in root.rglob("rank*_confidence.pdb") if path.is_file()
    ]
    fallback = [
        path
        for path in root.rglob("rank*.pdb")
        if path.is_file() and re.fullmatch(r"rank\d+\.pdb", path.name)
    ]
    candidates = sorted(preferred or fallback, key=_rank_sort_key)
    if not candidates:
        raise RuntimeError(
            f"No final rank PDB outputs found for official RAPiDock run under '{root}'."
        )
    return [str(path.resolve()) for path in candidates]


def _validate_esm2_constructor(esm2_class: Any) -> None:
    try:
        signature = inspect.signature(esm2_class)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "Unable to inspect the pinned fair-esm ESM2 constructor."
        ) from exc
    parameters = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    if parameters != ESM2_CONSTRUCTOR_PARAMETERS:
        raise RuntimeError(
            "Unsupported fair-esm ESM2 constructor parameters. "
            f"Expected {sorted(ESM2_CONSTRUCTOR_PARAMETERS)}, got "
            f"{sorted(parameters)}."
        )


def _load_pinned_esm2_class() -> Any:
    try:
        version = importlib_metadata.version("fair-esm")
    except importlib_metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "The official RAPiDock runtime requires fair-esm==2.0.0 "
            "for RNG-preserving ESM reuse."
        ) from exc
    if version != "2.0.0":
        raise RuntimeError(
            "RNG-preserving ESM reuse is restricted to fair-esm==2.0.0; "
            f"found fair-esm=={version}."
        )
    module = importlib.import_module("esm.model.esm2")
    esm2_class = getattr(module, "ESM2", None)
    if not inspect.isclass(esm2_class):
        raise RuntimeError("fair-esm does not expose esm.model.esm2.ESM2.")
    _validate_esm2_constructor(esm2_class)
    return esm2_class


class _ScopedESMCache:
    def __init__(
        self,
        *,
        pretrained: Any,
        torch_module: Any,
        esm2_class: Any,
        require_pinned_class: bool,
    ) -> None:
        self.pretrained = pretrained
        self.torch = torch_module
        self.esm2_class = esm2_class
        self.require_pinned_class = require_pinned_class
        self._model: Any | None = None
        self._alphabet: Any | None = None
        self._constructor_kwargs: Dict[str, Any] | None = None
        self.loader_count = 0
        self.replay_constructor_count = 0
        self.offload_count = 0
        _validate_esm2_constructor(esm2_class)

    @property
    def model_ready(self) -> bool:
        return self._model is not None

    def _require_cpu_default(self) -> None:
        get_default_device = getattr(self.torch, "get_default_device", None)
        if get_default_device is None:
            return
        default_device = str(get_default_device())
        if not default_device.startswith("cpu"):
            raise RuntimeError(
                "RNG-preserving ESM replay requires Torch's default device "
                f"to be CPU; found {default_device}."
            )

    def _validate_loaded_model(self, model: Any, alphabet: Any) -> None:
        if model is None or alphabet is None:
            raise RuntimeError(
                "fair-esm returned an incomplete ESM2 model/alphabet pair."
            )
        if self.require_pinned_class:
            model_class = type(model)
            if (
                model_class.__module__ != "esm.model.esm2"
                or model_class.__name__ != "ESM2"
            ):
                raise RuntimeError(
                    "Unsupported ESM model class. Expected "
                    "esm.model.esm2.ESM2 from fair-esm==2.0.0."
                )
        missing = [
            name
            for name in (
                "num_layers",
                "embed_dim",
                "attention_heads",
                "alphabet",
                "token_dropout",
            )
            if not hasattr(model, name)
        ]
        if missing:
            raise RuntimeError(
                "Unsupported ESM2 model configuration. Missing attributes "
                f"{missing}."
            )
        shape = (
            model.num_layers,
            model.embed_dim,
            model.attention_heads,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in shape
        ):
            raise RuntimeError(
                f"Unsupported pinned ESM2 shape {shape}; expected integer "
                f"dimensions {ESM2_EXPECTED_SHAPE}."
            )
        if shape != ESM2_EXPECTED_SHAPE:
            raise RuntimeError(
                f"Unsupported pinned ESM2 shape {shape}; expected "
                f"{ESM2_EXPECTED_SHAPE} for {ESM2_MODEL_NAME}."
            )
        if not isinstance(model.token_dropout, bool):
            raise RuntimeError(
                "Unsupported ESM2 token_dropout configuration; expected bool."
            )
        if model.alphabet is not alphabet:
            raise RuntimeError(
                "ESM2 model alphabet does not match the loader alphabet."
            )
        self._constructor_kwargs = {
            "num_layers": model.num_layers,
            "embed_dim": model.embed_dim,
            "attention_heads": model.attention_heads,
            "alphabet": alphabet,
            "token_dropout": model.token_dropout,
        }

    def load_model_and_alphabet(self, model_name: str) -> tuple[Any, Any]:
        if model_name != ESM2_MODEL_NAME:
            raise RuntimeError(
                f"Unsupported ESM model '{model_name}'. "
                f"The pinned runtime supports {ESM2_MODEL_NAME} only."
            )
        if self._model is None:
            self.loader_count += 1
            model, alphabet = self.pretrained.load_model_and_alphabet(model_name)
            self._validate_loaded_model(model, alphabet)
            self._model = model
            self._alphabet = alphabet
            return model, alphabet

        self._require_cpu_default()
        assert self._constructor_kwargs is not None
        self.replay_constructor_count += 1
        replay = self.esm2_class(**self._constructor_kwargs)
        del replay
        return self._model, self._alphabet

    def offload_to_cpu(self) -> None:
        if self._model is None:
            return
        self._model.cpu()
        self.offload_count += 1


class _ScopedPretrained:
    def __init__(self, original: Any, cache: _ScopedESMCache) -> None:
        self._original = original
        self._cache = cache

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)

    def load_model_and_alphabet(self, model_name: str) -> tuple[Any, Any]:
        return self._cache.load_model_and_alphabet(model_name)


@contextmanager
def _scoped_pretrained(
    dataset_module: Any, cache: _ScopedESMCache
) -> Iterator[None]:
    original = dataset_module.pretrained
    dataset_module.pretrained = _ScopedPretrained(original, cache)
    try:
        yield
    finally:
        dataset_module.pretrained = original
        cache.offload_to_cpu()


class OfficialRAPiDockRuntime:
    """Retain the official pinned inference functions in one daemon process."""

    def __init__(
        self,
        *,
        repo_path: pathlib.Path,
        model_dir: pathlib.Path,
        ckpt: str,
        confidence_model_dir: str | None,
        confidence_ckpt: str,
        device: str | None,
        official_module: Any | None = None,
        verify_source: bool = True,
        esm2_class: Any | None = None,
        dataset_module: Any | None = None,
    ) -> None:
        self.repo_path = repo_path
        self.model_dir = model_dir
        self.ckpt = ckpt
        self.confidence_model_dir = confidence_model_dir
        self.confidence_ckpt = confidence_ckpt
        self.device = device
        self._official_module = official_module
        self._verify_source = verify_source
        self._esm2_class = esm2_class
        self._dataset_module = dataset_module
        self._esm_cache: _ScopedESMCache | None = None
        self._score_model: Any | None = None
        self._confidence_model: Any | None = None
        self._score_model_args: Any | None = None
        self._resolved_device: Any | None = None
        self.source_revision: str | None = None
        self.load_count = 0
        self.load_attempts = 0
        self.load_error: str | None = None
        self.confidence_load_count = 0
        self.confidence_load_attempts = 0
        self.confidence_load_error: str | None = None

    @property
    def model_ready(self) -> bool:
        return self._score_model is not None and self._score_model_args is not None

    @property
    def confidence_model_ready(self) -> bool | None:
        if self.confidence_model_dir in (None, "", "null", "None"):
            return None
        return self._confidence_model is not None

    def status(self) -> Dict[str, Any]:
        esm_cache = self._esm_cache
        status = {
            "runtime": "official-rapidock-functions",
            "source_revision": self.source_revision,
            "model_ready": self.model_ready,
            "load_count": self.load_count,
            "load_attempts": self.load_attempts,
            "confidence_model_ready": self.confidence_model_ready,
            "confidence_load_count": self.confidence_load_count,
            "confidence_load_attempts": self.confidence_load_attempts,
            "esm_model_ready": bool(esm_cache and esm_cache.model_ready),
            "esm_loader_count": esm_cache.loader_count if esm_cache else 0,
            "esm_replay_constructor_count": (
                esm_cache.replay_constructor_count if esm_cache else 0
            ),
            "esm_offload_count": esm_cache.offload_count if esm_cache else 0,
        }
        if self.load_error is not None:
            status["message"] = self.load_error
        if self.confidence_load_error is not None:
            status["confidence_message"] = self.confidence_load_error
        return status

    @_admitted("load")
    def load(self) -> Any:
        if self.model_ready:
            return self._score_model

        self.load_attempts += 1
        try:
            if self._verify_source:
                self.source_revision = _verify_official_source(self.repo_path)
            else:
                self.source_revision = OFFICIAL_RAPIDOCK_REVISION

            official = self._official_module
            if official is None:
                official = _load_official_inference_module(self.repo_path)
            self._official_module = official
            for name in ("load_model", "prepare_data", "DataListLoader", "process_complex"):
                if not hasattr(official, name):
                    raise RuntimeError(
                        f"Official RAPiDock revision {OFFICIAL_RAPIDOCK_REVISION} "
                        f"does not expose required function '{name}'."
                    )

            yaml = importlib.import_module("yaml")
            with (self.model_dir / "model_parameters.yml").open(encoding="utf-8") as handle:
                score_config = yaml.full_load(handle)
            if not isinstance(score_config, Mapping):
                raise RuntimeError(
                    f"RAPiDock model config is not a mapping: "
                    f"{self.model_dir / 'model_parameters.yml'}"
                )
            self._score_model_args = Namespace(**dict(score_config))
            if self.device not in (None, "", "null", "None"):
                self._resolved_device = official.torch.device(str(self.device))
            else:
                self._resolved_device = official.torch.device(
                    "cuda" if official.torch.cuda.is_available() else "cpu"
                )
            with backend_output_to_stderr():
                self._score_model = official.load_model(
                    self._score_model_args,
                    str(self.model_dir / self.ckpt),
                    self._resolved_device,
                )
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"
            raise

        self.load_count += 1
        self.load_error = None
        return self._score_model

    def _require_score_model(self) -> Any:
        if not self.model_ready:
            detail = self.load_error or "score model initialization has not completed"
            raise RuntimeError(f"RAPiDock model is not ready: {detail}")
        return self._score_model

    def _require_confidence_model(self, scoring_function: Any) -> Any | None:
        if scoring_function != "confidence":
            return None
        if self._confidence_model is not None:
            return self._confidence_model
        if self.confidence_model_dir in (None, "", "null", "None"):
            raise RuntimeError(
                "RAPiDock confidence scoring was requested without "
                "confidence_model_dir."
            )

        self.confidence_load_attempts += 1
        try:
            yaml = importlib.import_module("yaml")
            confidence_dir = pathlib.Path(str(self.confidence_model_dir))
            with (confidence_dir / "model_parameters.yml").open(
                encoding="utf-8"
            ) as handle:
                confidence_config = yaml.full_load(handle)
            if not isinstance(confidence_config, Mapping):
                raise RuntimeError(
                    f"RAPiDock confidence model config is not a mapping: "
                    f"{confidence_dir / 'model_parameters.yml'}"
                )
            official = self._official_module
            confidence_args = Namespace(**dict(confidence_config))
            with backend_output_to_stderr():
                confidence_model = official.get_model(
                    confidence_args, no_parallel=True, confidence_mode=True
                )
                state_dict = official.torch.load(
                    str(confidence_dir / self.confidence_ckpt),
                    map_location=official.torch.device("cpu"),
                )
                confidence_model.load_state_dict(state_dict["model"], strict=True)
                confidence_model = confidence_model.to(self._resolved_device)
                confidence_model.eval()
        except Exception as exc:
            self.confidence_load_error = f"{type(exc).__name__}: {exc}"
            raise

        self._confidence_model = confidence_model
        self.confidence_load_count += 1
        self.confidence_load_error = None
        return confidence_model

    @staticmethod
    def _task_args(
        *,
        task: Mapping[str, str],
        options: Mapping[str, Any],
        model_dir: pathlib.Path,
        ckpt: str,
        confidence_model_dir: str | None,
        confidence_ckpt: str,
    ) -> Namespace:
        return Namespace(
            config=None,
            protein_peptide_csv=None,
            complex_name=task["name"],
            protein_description=task["protein_pdb"],
            peptide_description=task["peptide_pdb"],
            output_dir=task["output_dir"],
            model_dir=str(model_dir),
            ckpt=ckpt,
            confidence_model_dir=confidence_model_dir,
            confidence_ckpt=confidence_ckpt,
            **options,
        )

    def _resolve_dataset_module(self, official: Any) -> Any:
        if self._dataset_module is not None:
            module = self._dataset_module
        else:
            dataset_class = getattr(official, "InferenceDataset", None)
            if dataset_class is None:
                prepare_globals = getattr(
                    getattr(official, "prepare_data", None), "__globals__", {}
                )
                dataset_class = prepare_globals.get("InferenceDataset")
            if dataset_class is None:
                raise RuntimeError(
                    "Official RAPiDock source does not expose "
                    "utils.inference_utils.InferenceDataset."
                )
            module = sys.modules.get(dataset_class.__module__)
            if module is None:
                try:
                    module = importlib.import_module(dataset_class.__module__)
                except Exception as exc:
                    raise RuntimeError(
                        "Unable to import the official RAPiDock dataset module "
                        f"'{dataset_class.__module__}'."
                    ) from exc
        if not hasattr(module, "pretrained"):
            raise RuntimeError(
                "Official RAPiDock dataset module does not expose the "
                "'pretrained' ESM loader."
            )
        return module

    def _get_esm_cache(self, official: Any, dataset_module: Any) -> _ScopedESMCache:
        if self._esm_cache is not None:
            return self._esm_cache
        esm2_class = self._esm2_class
        if esm2_class is None:
            esm2_class = _load_pinned_esm2_class()
        self._esm_cache = _ScopedESMCache(
            pretrained=dataset_module.pretrained,
            torch_module=official.torch,
            esm2_class=esm2_class,
            require_pinned_class=self._esm2_class is None,
        )
        return self._esm_cache

    @_admitted("compute")
    def run(
        self, tasks: List[Mapping[str, Any]], params: Mapping[str, Any]
    ) -> List[Dict[str, Any]]:
        score_model = self._require_score_model()
        official = self._official_module
        dataset_module = self._resolve_dataset_module(official)
        esm_cache = self._get_esm_cache(official, dataset_module)
        options = _dock_options(params)
        confidence_model = self._require_confidence_model(
            options["scoring_function"]
        )
        results: List[Dict[str, Any]] = []

        for raw_task in tasks:
            task = _validate_task(raw_task)
            pathlib.Path(task["output_dir"]).mkdir(parents=True, exist_ok=True)
            args = self._task_args(
                task=task,
                options=options,
                model_dir=self.model_dir,
                ckpt=self.ckpt,
                confidence_model_dir=self.confidence_model_dir,
                confidence_ckpt=self.confidence_ckpt,
            )
            with backend_output_to_stderr():
                with _scoped_pretrained(dataset_module, esm_cache):
                    dataset = official.prepare_data(args, self._score_model_args)
                loader = official.DataListLoader(
                    dataset=dataset, batch_size=1, shuffle=False
                )
                complexes = list(enumerate(loader))
                for index, original_complex_graph in complexes:
                    name = dataset.complex_names[index]
                    if not original_complex_graph[0].success:
                        raise RuntimeError(
                            f"Official RAPiDock dataset preparation failed for "
                            f"complex '{name}'."
                        )
                    write_dir = str(pathlib.Path(task["output_dir"]) / name)
                    official.process_complex(
                        score_model,
                        confidence_model,
                        self._score_model_args,
                        args,
                        original_complex_graph[0],
                        write_dir,
                    )

            candidates = _collect_official_candidates(task["output_dir"])
            results.append(
                {
                    "name": task["name"],
                    "candidates": [
                        {"rank": rank, "peptide_pdb": path}
                        for rank, path in enumerate(candidates, start=1)
                    ],
                }
            )
        return results


class UnavailableRAPiDockRuntime:
    def __init__(self, *, message: str) -> None:
        self.message = message

    def load(self) -> Any:
        raise FileNotFoundError(self.message)

    def status(self) -> Dict[str, Any]:
        return {
            "runtime": "unavailable",
            "model_ready": False,
            "load_count": 0,
            "load_attempts": 1,
            "confidence_model_ready": None,
            "confidence_load_count": 0,
            "confidence_load_attempts": 0,
            "message": f"FileNotFoundError: {self.message}",
        }

    def run(
        self, tasks: List[Mapping[str, Any]], params: Mapping[str, Any]
    ) -> List[Dict[str, Any]]:
        raise FileNotFoundError(self.message)


def build_rapidock_runtime(
    *,
    repo_path: pathlib.Path,
    model_dir: pathlib.Path,
    ckpt: str,
    confidence_model_dir: str | None,
    confidence_ckpt: str,
    device: str | None,
) -> RAPiDockRuntime | OfficialRAPiDockRuntime | UnavailableRAPiDockRuntime:
    if _has_maintained_api_capability(repo_path):
        return RAPiDockRuntime(
            repo_path=repo_path,
            model_dir=model_dir,
            ckpt=ckpt,
            confidence_model_dir=confidence_model_dir,
            confidence_ckpt=confidence_ckpt,
            device=device,
        )

    inference_path = repo_path / "inference.py"
    if inference_path.is_file():
        return OfficialRAPiDockRuntime(
            repo_path=repo_path,
            model_dir=model_dir,
            ckpt=ckpt,
            confidence_model_dir=confidence_model_dir,
            confidence_ckpt=confidence_ckpt,
            device=device,
        )

    return UnavailableRAPiDockRuntime(
        message=(
            f"RAPiDock source at '{repo_path}' provides neither maintained "
            "RAPiDock/API.py with RAPiDockInference nor official inference.py."
        )
    )
