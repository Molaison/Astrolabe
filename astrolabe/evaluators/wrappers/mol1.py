"""Mol-1 facade; the existing native model owner owns execution and recovery."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile

from astrolabe.evaluators.mol1_contract import (
    SOURCE_COMMIT, checkpoint_paths, configuration_errors, local_assets, selected_model,
)


class Mol1Runner:
    def __init__(self, settings, *, constraint_hint=False):
        self.settings = deepcopy(dict(settings))
        self._constraint_hint = constraint_hint
        self._client = None
        self._owned_settings = None
        self._frozen_settings = None
        self._cleanup_pending = False
        self._request_failure = None

    @classmethod
    def from_config(cls, conf, *, constraint_hint=False):
        from omegaconf import OmegaConf
        settings = OmegaConf.to_container(conf, resolve=True) if OmegaConf.is_config(conf) else dict(conf)
        settings.pop("_type", None)
        errors = configuration_errors(settings)
        if errors:
            raise ValueError("; ".join(errors))
        return cls(settings, constraint_hint=constraint_hint)

    @property
    def native_client(self):
        """Currently held native client, or None; observation never loads/switches."""
        return self._client

    def prepare_shared_model(self, job=None):
        from astrolabe.model_daemon import _content_identity
        from astrolabe.model_execution import NativeModelClient

        if self._cleanup_pending:
            raise RuntimeError("Mol-1 owner cleanup is unresolved")
        if self._request_failure is not None:
            raise RuntimeError("Mol-1 native request failed; model switching cannot clear its fence") from self._request_failure
        settings = deepcopy(self.settings)
        # Resolve against the host cwd, before the child switches to repo_path.
        for key in ("repo_path", "mol_root_dir"):
            settings[key] = str(Path(settings[key]).expanduser().resolve(strict=True))
        checkpoints = {name: path.resolve(strict=True) for name, path in checkpoint_paths(settings).items()}
        # Normalize the legacy override to its model-specific key before freezing.
        settings.pop("load_checkpoint_path", None)
        settings.pop("base_checkpoint_path", None)
        settings.pop("constraint_checkpoint_path", None)
        for name, path in checkpoints.items():
            settings["base_checkpoint_path" if name == "mol-1-base" else "constraint_checkpoint_path"] = str(path)
        extra = settings.get("extra_python_paths") or []
        if isinstance(extra, str):
            extra = [path for path in extra.split(os.pathsep) if path]
        settings["extra_python_paths"] = [str(Path(path).expanduser().resolve()) for path in extra]
        # Expand an explicit executable path lexically; resolving its symlink
        # would escape the venv and lose that environment's site-packages.
        python_bin = os.path.expanduser(settings["python_bin"])
        settings["python_bin"] = os.path.abspath(python_bin) if os.path.dirname(python_bin) else python_bin
        settings["source_commit"] = SOURCE_COMMIT
        paths, references = local_assets(settings)
        settings["reference_assets"] = references
        settings["asset_sha256"] = {str(path): _content_identity(path) for path in paths}
        if self._frozen_settings is not None and settings != self._frozen_settings:
            raise ValueError("Mol-1 configuration/assets changed after native owner initialization")
        self._frozen_settings = deepcopy(settings)
        model_name = selected_model(settings, job, constraint_hint=self._constraint_hint)
        checkpoint = checkpoints[model_name]
        # The durable/config identity above freezes *both* switchable checkpoints.
        # NativeModelClient identity contains only the model/assets actually used.
        for key in ("base_checkpoint_path", "constraint_checkpoint_path", "constraints", "dynamic_binding_site"):
            settings.pop(key, None)
        settings["model_name"] = model_name
        settings["load_checkpoint_path"] = str(checkpoint)
        settings["asset_sha256"] = {path: identity for path, identity in settings["asset_sha256"].items()
                                    if Path(path) not in checkpoints.values() or Path(path) == checkpoint}
        if self._client is not None and settings != self._owned_settings:
            # Do not retain a second client or clear an unresolved cleanup failure.
            self.shutdown()
        if self._client is None:
            source = settings["repo_path"]
            repo = str(Path(__file__).resolve().parents[3])
            inherited = os.environ.get("PYTHONPATH")
            inherited_paths = ([str(Path(path or ".").expanduser().resolve())
                                for path in inherited.split(os.pathsep)] if inherited else [])
            env = {"MOL_ROOT_DIR": settings["mol_root_dir"],
                   "LAYERNORM_TYPE": "torch",
                   "PYTHONPATH": os.pathsep.join((source, repo, *settings["extra_python_paths"], *inherited_paths))}
            try:
                self._client = NativeModelClient(
                    "mol1", "astrolabe.evaluators.wrappers.mol1_daemon:create_handler", settings,
                    python_bin=settings["python_bin"], cwd=source, env=env,
                    request_timeout=settings.get("timeout_seconds"), startup_timeout=600,
                    poolable=False,
                )
            except BaseException as error:
                # Construction can leave failed cleanup in the native owner scope.
                # Ordinary access/switching is not permission to retry initialization.
                self._request_failure = error
                raise
            self._owned_settings = settings
        return self._client

    def run_inference(self, *, input_json, out_dir):
        jobs = json.loads(Path(input_json).read_text())
        if not isinstance(jobs, list) or len(jobs) != 1:
            raise ValueError("Mol-1 native request requires exactly one job")
        job = jobs[0]
        client = self.prepare_shared_model(job)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        target = Path(tempfile.mkdtemp(prefix="request-", dir=out_dir))
        try:
            # No local retry or exception conversion: retain the native outcome.
            reply = client.request({"input_json": str(Path(input_json).resolve()), "out_dir": str(target.resolve())})
            if reply != {"output_dir": str(target.resolve()), "job": job["name"], "modelSeeds": job["modelSeeds"],
                         "model_name": self._owned_settings["model_name"]}:
                from astrolabe.model_daemon import RequestOutcomeUnknown

                raise RequestOutcomeUnknown("Mol-1 returned mismatched output identity; native request must not be replayed")
        except BaseException as error:
            self._request_failure = error
            raise
        return target

    def shutdown(self):
        if self._client is None:
            return
        self._cleanup_pending = True
        self._client.shutdown()
        self._client = self._owned_settings = None
        self._cleanup_pending = False
