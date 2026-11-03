"""Trajectory-owned mutator models behind the shared execution broker.

Only runner DTOs cross this boundary.  Selection, shared segments, acceptance,
and stage transactions stay in the trajectory process.  Initialization has its
own RNG exchange because constructing a torch module consumes random numbers,
even when its parameters are immediately overwritten by a checkpoint.
"""
from __future__ import annotations

import importlib
import hashlib
from contextlib import contextmanager
from contextvars import ContextVar
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import uuid

from omegaconf import OmegaConf


# This is an implementation allowlist, not a numerical certification claim.
# DynamicMPNN callbacks remain private opaque bridges unless they expose an
# explicit model-owning factory. They are never advertised as shared models.
RUNNER_TYPES = {
    "ligandmpnn": ("astrolabe.mutators.wrappers.ligandmpnn_run", "LigandMPNN"),
    "carbondesign": ("astrolabe.mutators.wrappers.carbondesign_runner", "CarbonDesignRunner"),
    "dplm": ("astrolabe.mutators.wrappers.dplm_runner", "DPLMRunner"),
    "surfdesign": ("astrolabe.mutators.wrappers.surfdesign_runner", "SurfDesignRunner"),
    "surfpro": ("astrolabe.mutators.wrappers.surfpro_runner", "SurfProRunner"),
    "nampnn": ("astrolabe.mutators.wrappers.na_mpnn_runner", "NAMPNNRunner"),
    "dynamicmpnn": ("astrolabe.mutators.wrappers.dynamicmpnn_run", "DynamicMPNNRunner"),
    "esm2": ("astrolabe.mutators.esm2", "ESM2Mutator"),
    "esmif": ("astrolabe.mutators.esm_if1", "ESMIF1Mutator"),
    "esm3": ("astrolabe.mutators.esm3", "ESM3Mutator"),
}
BROKER_MUTATORS = frozenset({
    "ligandmpnn", "grouped_ligandmpnn", "proteinmpnn", "multistateproteinmpnn",
    "multistate_proteinmpnn", "faraparttiedmultitargetligandmpnn",
    "far_apart_tied_multitarget_ligandmpnn", "carbondesign", "dplm", "surfdesign",
    "surfpro", "nampnn", "esm2", "esmif", "esm3", "dynamicmpnn",
})
CPU_MUTATORS = frozenset({"mcmc", "pore_blocker_mcmc", "motifbreaker", "xmix"})
_EXECUTION_MODE = ContextVar("mutator_execution_mode", default="auto")
_EXECUTION_RESOURCES = ContextVar("mutator_execution_resources", default=None)


@contextmanager
def execution_mode(mode, *, resources=None):
    if mode not in ("legacy", "daemon", "auto"):
        raise ValueError("mutator execution must be legacy, daemon, or auto")
    if mode == "legacy" and uses_broker():
        raise ValueError("GPU-hidden trajectory owners cannot execute legacy GPU mutators")
    token = _EXECUTION_MODE.set(mode)
    resource_token = _EXECUTION_RESOURCES.set(_plain(resources) if resources is not None else None)
    try:
        yield
    finally:
        _EXECUTION_MODE.reset(token)
        _EXECUTION_RESOURCES.reset(resource_token)


def uses_resident():
    if os.environ.get("ASTROLABE_MUTATOR_WORKER") == "1":
        return False
    return _EXECUTION_MODE.get() == "daemon" or uses_broker()


def uses_broker() -> bool:
    endpoint = os.environ.get("ASTROLABE_EXECUTION_ENDPOINT")
    token = os.environ.get("ASTROLABE_EXECUTION_TOKEN")
    if bool(endpoint) != bool(token):
        raise RuntimeError("incomplete execution broker credentials")
    return bool(endpoint) and os.environ.get("ASTROLABE_MUTATOR_WORKER") != "1"


def _plain(value):
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_plain(v) for v in value)
    if isinstance(value, list):
        return [_plain(v) for v in value]
    return value


def request_shape(kind, args, kwargs):
    """Describe workload size without adding sequence contents to identities."""
    import numpy as np
    import torch
    shapes, external_bytes = [], 0
    seen_files = set()

    def visit(value, key="", depth=0):
        nonlocal external_bytes
        if depth > 32:
            raise ValueError("mutator request nesting exceeds shape budget")
        if isinstance(value, (torch.Tensor, np.ndarray)):
            shapes.append([key, str(value.dtype), list(value.shape)])
        elif isinstance(value, dict):
            for name, child in sorted(value.items()):
                visit(child, str(name), depth + 1)
        elif isinstance(value, (tuple, list)):
            shapes.append([key, "items", len(value)])
            for child in value:
                visit(child, key, depth + 1)
        elif isinstance(value, (str, Path)):
            text = str(value)
            if "path" in key or isinstance(value, Path):
                path = Path(text)
                if path.is_file():
                    path = path.resolve()
                    if path not in seen_files:
                        seen_files.add(path)
                        size = path.stat().st_size
                        external_bytes += size
                        shapes.append([key, path.suffix.lower(), size])
                    return
            shapes.append([key, "text", len(text)])
    visit(args, "args")
    visit(kwargs, "kwargs")
    encoded = json.dumps(shapes, separators=(",", ":"), sort_keys=True).encode()
    return kind + ":" + hashlib.sha256(encoded).hexdigest()[:24], external_bytes


def _pin_inputs(kind, config):
    """Pin the actual local weights/configuration consumed by each constructor."""
    import torch
    cfg = dict(config)
    paths = []
    code_paths = []
    repo = Path(str(cfg.get("repo_path") or ".")).expanduser().resolve()

    def pin(value, *, base=None):
        path = Path(value).expanduser()
        if not path.is_absolute() and base is not None:
            path = base / path
        path = path.resolve(strict=True)
        paths.append(str(path))
        return str(path)

    if cfg.get("repo_path"):
        code_paths.append(str(repo))
    if kind == "ligandmpnn":
        key = "checkpoint_" + str(cfg.get("model_type", "ligand_mpnn"))
        if not cfg.get(key):
            raise ValueError(f"resident MPNN requires {key}")
        cfg[key] = pin(cfg[key])
    elif kind == "carbondesign":
        for key in ("model_ckpt", "model_config", "model_features"):
            cfg[key] = pin(cfg[key], base=repo)
        model_config = json.loads(Path(cfg["model_config"]).read_text())
        lm = model_config.get("model", {}).get("embeddings_and_seqformer", {}).get("lm", {})
        if lm.get("model_path"):
            pin(lm["model_path"], base=repo)
    elif kind == "nampnn":
        extra = dict(cfg.get("extra_args") or {})
        selected = extra.get("checkpoint_na_mpnn") or extra.get("checkpoint-na-mpnn") or cfg.get("checkpoint")
        cfg["checkpoint"] = pin(selected or "models/design_model/s_19137.pt", base=repo)
        for key in ("checkpoint_na_mpnn", "checkpoint-na-mpnn"):
            if key in extra:
                extra[key] = cfg["checkpoint"]
        cfg["extra_args"] = extra
    elif kind == "dynamicmpnn":
        if cfg.get("checkpoint") or cfg.get("model_ref"):
            cfg["checkpoint"] = pin(cfg.get("checkpoint") or cfg.get("model_ref"), base=repo)
        from importlib.machinery import PathFinder
        # Locate source without importing the callback in the CPU owner. Its
        # first-import RNG consumption still belongs to the actual request.
        for key in ("backend_callable", "backend_factory"):
            value = cfg.get(key)
            if not value:
                continue
            if not isinstance(value, str):
                raise TypeError("resident DynamicMPNN requires an import path, not a live callable")
            module_name = value.split(":", 1)[0] if ":" in value else value.rsplit(".", 1)[0]
            search, name = list(sys.path), ""
            for component in module_name.split("."):
                name = component if not name else name + "." + component
                found = PathFinder.find_spec(name, search)
                if found is None or not found.origin or not Path(found.origin).is_file():
                    raise ValueError("resident DynamicMPNN callback source cannot be pinned: " + name)
                code_paths.append(str(Path(found.origin).resolve()))
                search = list(found.submodule_search_locations or [])
        if cfg.get("command_template") and not cfg.get("repo_path"):
            raise ValueError("resident DynamicMPNN CLI bridge requires repo_path to pin external source")
    elif kind == "surfdesign":
        experiment = Path(cfg["experiment_path"]).expanduser().resolve(strict=True)
        pin(experiment / ".hydra/config.yaml")
        cfg["checkpoint"] = pin(cfg.get("checkpoint") or "best.ckpt", base=experiment / "checkpoints")
    elif kind == "surfpro":
        checkpoint = cfg.get("checkpoint") or "checkpoint_best.pt"
        checkpoint_base = Path(str(cfg.get("checkpoint_dir") or repo)).expanduser()
        cfg["checkpoint"] = pin(checkpoint, base=checkpoint_base)
        for key in ("msms_bin", "pdb_to_xyzrn_bin"):
            if cfg.get(key):
                cfg[key] = pin(cfg[key])
    elif kind == "dplm":
        # Resolve the same pinned snapshot as the external inference script;
        # never allow a model-hub download as an implicit task side effect.
        from astrolabe.mutators.wrappers.dplm_infer import _resolve_model_name
        source = _resolve_model_name(str(cfg.get("model_name", "airkingbd/dplm_150m")))
        if not Path(source).expanduser().exists():
            raise ValueError("resident DPLM requires a local model snapshot or GH_DPLM_MODEL_SNAPSHOT")
        cfg["model_name"] = pin(source)
    elif kind in ("esm2", "esmif"):
        stem = "esm2_t33_650M_UR50D" if kind == "esm2" else "esm_if1_gvp4_t16_142M_UR50"
        hub = Path(torch.hub.get_dir()) / "checkpoints"
        pin(hub / (stem + ".pt"))
        regression = hub / (stem + "-contact-regression.pt")
        if regression.exists():
            pin(regression)
        module = importlib.import_module("esm2")
        code_paths.append(str(Path(module.__file__).resolve().parent))
    elif kind == "esm3":
        snapshot = os.environ.get("GH_ESM3_MODEL_SNAPSHOT")
        if not snapshot:
            raise ValueError("resident ESM3 requires GH_ESM3_MODEL_SNAPSHOT")
        pin(snapshot)
        module = importlib.import_module("esm3")
        code_paths.append(str(Path(module.__file__).resolve().parent))
    for key in ("daemon_script_path", "script_path"):
        if cfg.get(key):
            code_paths.append(str(Path(cfg[key]).expanduser().resolve(strict=True)))
    return cfg, list(dict.fromkeys(paths)), list(dict.fromkeys(code_paths))


def build_runner(kind, config, **kwargs):
    """Keep direct execution as the default; use the broker in isolated owners."""
    if uses_resident():
        return ResidentRunner(kind, config, kwargs=kwargs)
    module, name = RUNNER_TYPES[kind]
    cls = getattr(importlib.import_module(module), name)
    return cls(config, **kwargs)


def external_environment_identity(kind, cfg, device, *, worker_env=None):
    """Probe the environment actually launched by each private subprocess."""
    from astrolabe.evaluators.worker_identity import environment_identity
    overrides = dict(worker_env or {})
    removed = ()
    if kind in ("dplm", "surfdesign", "surfpro"):
        removed = ("PYTHONPATH", "APPTAINERENV_PYTHONPATH")
    if kind == "dplm":
        work = Path(cfg.get("work_dir") or Path.cwd() / "dplm_cache").expanduser().resolve()
        cache = Path(cfg.get("triton_cache_dir") or work / "triton_cache").expanduser().resolve()
        overrides["TRITON_CACHE_DIR"] = str(cache)
    if kind == "nampnn":
        inherited = overrides.get("PYTHONPATH", os.environ.get("PYTHONPATH", ""))
        repo = str(Path(cfg["repo_path"]).expanduser().resolve())
        overrides["PYTHONPATH"] = repo + (os.pathsep + inherited if inherited else "")
    return environment_identity(str(cfg["python_bin"]), env=overrides,
                                env_remove=removed, device=device)


class ResidentRunner:
    """A private model instance; requests reuse it without consuming init RNG."""
    supported_distribution_semantics = frozenset({
        "legacy_order_conditioned_marginals", "shared_reference_teacher_forcing",
    })

    def __init__(self, kind, config, *, kwargs=None, keyword_constructor=False):
        if kind not in RUNNER_TYPES:
            raise ValueError(f"unsupported resident mutator backend: {kind}")
        from astrolabe.evaluators.tree_codec import dump_tree, load_tree
        from astrolabe.evaluators.torch_rng import capture, restore, capture_settings
        from astrolabe.evaluators.worker_identity import identity, probe_device
        from astrolabe.evaluators.worker_session import WorkerArtifacts
        from astrolabe.runtime.client import DaemonClient, OutcomeUnknown

        self.kind = kind
        self._closed = False
        self._release_error = None
        self._client = None
        self._artifacts = WorkerArtifacts()
        self._lock = __import__("threading").RLock()
        native_config = _plain(config)
        self._resource_hints = dict(native_config.pop("execution_resources", {}) or {})
        self._resource_hints.update(_EXECUTION_RESOURCES.get() or {})
        allowed_resources = {"resident_bytes", "scratch_bytes", "prepared_bytes", "ram_bytes",
                             "output_bytes", "load_ram_bytes", "cpu_slots"}
        if set(self._resource_hints) - allowed_resources:
            raise ValueError("unknown mutator execution_resources fields")
        cfg, weights, vendor_code = _pin_inputs(kind, native_config)
        kwargs = _plain(kwargs or {})
        instance = uuid.uuid4().hex
        # The model is private and so are mutable native workspaces. Shared
        # DPLM response pools keep their original protocol and directory.
        for key in ("output_dir", "out_folder", "work_dir"):
            if cfg.get(key):
                cfg[key] = str(Path(cfg[key]).expanduser().resolve() / ("resident-" + instance))
        if kwargs.get("output_root"):
            kwargs["output_root"] = str(Path(kwargs["output_root"]).resolve() / ("resident-" + instance))
        # Load the same module in the owner before capturing RNG; imports are
        # not allowed to masquerade as model construction on the worker side.
        module_name, class_name = RUNNER_TYPES[kind]
        module = importlib.import_module(module_name)
        code = [str(Path(__file__).resolve()), str(Path(module.__file__).resolve()),
                str(Path(__file__).with_name("wrappers") / "resident_daemon.py")]
        code.extend(vendor_code)
        code.extend(str(Path(__file__).with_name("wrappers") / name) for name in (
            "ligandmpnn_utils.py", "dplm_daemon.py", "dplm_infer.py",
            "surfdesign_daemon.py", "surfpro_daemon.py", "na_mpnn_daemon.py",
            "na_mpnn_native.py", "subprocess_protocol.py"))
        env = {"PYTHONPATH": os.pathsep.join(str(Path(p).resolve()) for p in sys.path if p),
               "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        requested = cfg.get("device")
        if requested in (None, "auto", "cuda"):
            requested = "cuda:0" if os.environ.get("ASTROLABE_ALLOCATED_CUDA_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES")) else "cpu"
        if requested == "cpu":
            device = {"name": "cpu", "device_kind": "cpu", "visible": ""}
        else:
            device = probe_device(sys.executable, "protenix", requested, env=env)
        self._device = device["name"]
        env["CUDA_VISIBLE_DEVICES"] = device["visible"] or ""
        if self._device == "cpu":
            env["NVIDIA_VISIBLE_DEVICES"] = "void"
        if kind in ("esm2", "esmif", "esm3", "dplm", "surfdesign", "surfpro", "carbondesign"):
            cfg["device"] = self._device
        if kind == "carbondesign" and self._device.startswith("cuda:"):
            cfg["gpu_idx"] = [int(self._device.split(":", 1)[1])]
        if cfg.get("python_bin"):
            external_environment = external_environment_identity(kind, cfg, device, worker_env=env)
        else:
            external_environment = None
        # Explicit constructor settings are part of compatibility identity.
        # This random instance is a residency owner, not a scientific seed.
        torch_settings = capture_settings()
        settings = {"kind": kind, "config": cfg, "kwargs": kwargs,
                    "keyword_constructor": bool(keyword_constructor),
                    "external_environment": external_environment,
                    "torch_settings": torch_settings,
                    "instance": instance}
        expected = identity(sys.executable, code, weights, device, settings, env=env)
        boot = Path(tempfile.mkdtemp(prefix="request-", dir=self._artifacts.root))
        try:
            dump_tree(boot / "initial", capture(self._device))
            spec = {"settings": settings, "code_paths": code, "weight_paths": weights,
                    "device": device, "bootstrap": str(boot)}
            spec_path = self._artifacts.specification(expected, spec)
            self._client = DaemonClient(
                [sys.executable, "-m", "astrolabe.mutators.wrappers.resident_daemon", str(spec_path)],
                expected, env=env, startup_timeout=float(cfg.get("startup_timeout", 600)),
                resources={"resident_bytes": self._resource_hints.get("resident_bytes"),
                           "scratch_bytes": self._resource_hints.get("scratch_bytes"),
                           "ram_bytes": self._resource_hints.get("load_ram_bytes", 64 * 1024**2)},
            )
            initialized = load_tree(boot / "ready")
            restore(initialized["rng"], self._device)
            self.metadata = initialized["metadata"]
            self.model_type = self.metadata.get("model_type")
            for key in ("num_samples", "na_shared_tokens", "job_name"):
                if key in self.metadata:
                    setattr(self, key, self.metadata[key])
            for key in ("output_dir", "work_dir"):
                if key in self.metadata:
                    setattr(self, key, Path(self.metadata[key]))
            self.lifecycle = initialized["lifecycle"]
        except BaseException as exc:
            if isinstance(exc, OutcomeUnknown) and self._client is None:
                # Startup can fail before the client constructor returns. No
                # handle is not evidence that its worker has stopped reading
                # the bootstrap; keep both it and the private artifact root.
                self._closed = True
                self._release_error = exc
                self._artifacts.retain()
                exc.retained_request_dir = str(boot)
                exc.retained_artifact_root = str(self._artifacts.root)
            else:
                try:
                    self.close()
                except OutcomeUnknown as release_error:
                    release_error.retained_request_dir = str(boot)
                    raise
            raise
        finally:
            if self._release_error is None:
                shutil.rmtree(boot, ignore_errors=True)

    def call(self, method, *args, **kwargs):
        if method not in ("design_sequences", "compute_mutation"):
            raise ValueError("unsupported resident mutator method")
        from astrolabe.evaluators.tree_codec import dump_tree, load_tree
        from astrolabe.evaluators.torch_rng import capture, restore
        from astrolabe.evaluators.worker_session import request_artifacts
        with self._lock:
            if self._closed:
                raise RuntimeError("resident mutator runner is closed")
            with request_artifacts(self._artifacts.root) as tmp:
                root = Path(tmp)
                shape, external_bytes = request_shape(self.kind, args, kwargs)
                random_state = capture(self._device)
                execution_settings = random_state.get("settings", {})
                shape += ":" + hashlib.sha256(json.dumps(execution_settings, sort_keys=True).encode()).hexdigest()[:12]
                dump_tree(root / "input", {"method": method, "args": _plain(args),
                                          "kwargs": _plain(kwargs), "rng": random_state})
                artifact_bytes = sum(p.stat().st_size for p in (root / "input").iterdir())
                # These are reservations, not measured peaks. Absent certified
                # backend profiles the broker reserves unknown output/scratch
                # conservatively and observes actual process-tree RAM/VRAM.
                prepared_bytes = artifact_bytes + external_bytes
                ram_hint = self._resource_hints.get("ram_bytes", max(4 * prepared_bytes, 64 * 1024**2))
                resources = {
                    "prepared_bytes": max(prepared_bytes, self._resource_hints.get("prepared_bytes") or 0),
                    "ram_bytes": max(4 * prepared_bytes, ram_hint) if ram_hint is not None else None,
                    "output_bytes": self._resource_hints.get("output_bytes"),
                    "scratch_bytes": self._resource_hints.get("scratch_bytes"),
                    "cpu_slots": self._resource_hints.get("cpu_slots", 1),
                }
                metadata = {"shape_key": shape, "scientific_request_id": root.name,
                            "resources": resources}
                try:
                    self._client.call({"method": "prepare", "request_dir": str(root)},
                                      runtime={**metadata, "phase": "cpu_prepare"})
                    self._client.call({"method": "execute", "request_dir": str(root)},
                                      runtime={**metadata, "phase": "composite"})
                    reply = self._client.call({"method": "finalize", "request_dir": str(root)},
                                             runtime={**metadata, "phase": "cpu_finalize"})
                    if reply != {"request": root.name}:
                        raise ValueError("mutator response request mismatch")
                    result = load_tree(root / "output")
                    restore(result["rng"], self._device)
                    self.lifecycle = result["lifecycle"]
                    return result["result"]
                except BaseException:
                    # No fallback to direct execution and no request replay.
                    self.close()
                    raise

    def design_sequences(self, *args, **kwargs):
        return self.call("design_sequences", *args, **kwargs)

    def compute_mutation(self, features):
        return self.call("compute_mutation", features)

    def close(self):
        if self._closed:
            error = getattr(self, "_release_error", None)
            if error is not None:
                raise error
            return
        self._closed = True
        if self._client is not None:
            try:
                self._client.close(release=True)
            except BaseException as exc:
                from astrolabe.runtime.client import OutcomeUnknown
                self._release_error = (exc if isinstance(exc, OutcomeUnknown) else
                                       OutcomeUnknown("mutator worker release outcome unknown; artifacts retained"))
                if self._artifacts is not None:
                    self._artifacts.retain()
                    self._release_error.retained_artifact_root = str(self._artifacts.root)
                if self._release_error is exc:
                    raise
                raise self._release_error from exc
            self._client = None
        if self._artifacts is not None:
            self._artifacts.close()
            self._artifacts = None

    shutdown = close

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
