"""Allowlisted mutator runner worker. No trajectory state is imported or sent."""
from __future__ import annotations

import contextlib
import importlib
import json
import os
from pathlib import Path
import sys
import time


def _metadata(runner):
    result = {}
    for name in ("model_type", "aa_order", "num_samples", "na_shared_tokens", "job_name", "output_dir", "work_dir"):
        value = getattr(runner, name, None)
        if value is not None:
            result[name] = value
    alphabet = getattr(runner, "alphabet", None)
    if alphabet is not None:
        fields = ("standard_toks", "prepend_toks", "append_toks", "prepend_bos", "append_eos", "use_msa")
        if all(hasattr(alphabet, key) for key in fields):
            result["alphabet"] = {key: getattr(alphabet, key) for key in fields}
    return result


def lifecycle_snapshot(runner, kind):
    """Report observations, not existence of an adapter object as model loading."""
    import torch
    model = getattr(runner, "model", None)
    if model is None and isinstance(getattr(runner, "runner", None), torch.nn.Module):
        model = runner.runner
    params = list(model.parameters()) if isinstance(model, torch.nn.Module) else []
    devices = sorted({str(p.device) for p in params})
    dtypes = sorted({str(p.dtype) for p in params})
    remote = getattr(runner, "_ready", None)
    child = getattr(runner, "_proc", None)
    child_exit = child.poll() if child is not None else None
    ready = not bool(getattr(runner, "_closed", False))
    if isinstance(remote, dict):
        count = remote.get("load_count", remote.get("model_load_count"))
        loaded = remote.get("weights_loaded", count is not None and count > 0)
        resident = remote.get("gpu_resident")
        devices = [remote["device"]] if remote.get("device") else devices
        dtypes = list(remote.get("dtypes") or [])
        if child is None or child_exit is not None:
            loaded = resident = ready = False
            devices = dtypes = []
    else:
        count = getattr(runner, "model_load_count", None)
        loaded = bool(params) if count is not None and count > 0 else None
        resident = any(p.is_cuda for p in params) if params else None
    mode = "resident_model" if loaded else "resident_bridge"
    if kind == "dynamicmpnn" and not loaded:
        # An arbitrary callback/CLI is not evidence of model residency. Private
        # execution still receives the same conservative GPU admission lease.
        loaded = resident = count = None
        if getattr(runner, "command_template", None):
            mode = "one_shot"
    if kind in ("nampnn", "dplm") and not runner.daemon:
        mode = "one_shot"
    outputs = []
    for name in ("output_dir", "output_root", "work_dir", "base_folder"):
        path = getattr(runner, name, None)
        if path and Path(path).is_dir():
            item = {"path": str(Path(path).resolve()), "scope": "job"}
            if item not in outputs:
                outputs.append(item)
    cache = getattr(runner, "response_cache_dir", None)
    if cache and Path(cache).is_dir():
        outputs.append({"path": str(Path(cache).resolve()), "scope": "shared_cache"})
    return {"mode": mode, "process_alive": True, "pid": os.getpid(),
            "weights_loaded": loaded, "gpu_resident": resident,
            "ready": ready,
            "load_count": count, "model_load_count": count, "devices": devices, "dtypes": dtypes,
            "output_roots": outputs,
            "children": [child.pid] if child is not None and child_exit is None else [],
            "child_exitcode": child_exit}


def main():
    protocol_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = os.fdopen(protocol_fd, "w", buffering=1)
    spec = json.loads(Path(sys.argv[1]).read_text())
    settings = spec["settings"]
    from astrolabe.mutators.resident import RUNNER_TYPES, _plain, external_environment_identity
    from astrolabe.evaluators.worker_identity import identity
    from astrolabe.evaluators.tree_codec import dump_tree, load_tree
    from astrolabe.evaluators.torch_rng import capture, restore
    from astrolabe.evaluators.request_path import request_directory
    from astrolabe.runtime.server import serve
    from omegaconf import OmegaConf
    import torch

    kind = settings["kind"]
    module_name, class_name = RUNNER_TYPES[kind]
    actual = identity(sys.executable, spec["code_paths"], spec["weight_paths"], spec["device"], settings)
    if settings.get("external_environment") is not None:
        observed = external_environment_identity(kind, settings["config"], spec["device"])
        if observed != settings["external_environment"]:
            raise RuntimeError("mutator external interpreter environment changed")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != spec["device"]["visible"]:
        raise RuntimeError("mutator worker visible device mismatch")
    device = spec["device"]["name"]
    if device.startswith("cuda"):
        if torch.cuda.get_device_name(torch.device(device)) != spec["device"]["device_kind"]:
            raise RuntimeError("mutator worker device kind mismatch")
        torch.cuda.set_device(torch.device(device))
    os.environ["ASTROLABE_MUTATOR_WORKER"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    # All weights were resolved and hashed by the owner. Runtime installation
    # and downloads are outside the scientific request and its resource budget.
    def no_download(*_args, **_kwargs):
        raise RuntimeError("resident mutator may only load preinstalled local weights")
    torch.hub.download_url_to_file = no_download
    boot = request_directory(spec["bootstrap"])
    with contextlib.redirect_stdout(sys.stderr):
        cls = getattr(importlib.import_module(module_name), class_name)
        restore(load_tree(boot / "initial"), device)
        cfg = dict(settings["config"])
        kwargs = dict(settings["kwargs"])
        if settings["keyword_constructor"]:
            runner = cls(**cfg, **kwargs)
        else:
            runner = cls(OmegaConf.create(cfg), **kwargs)
        # Private subprocess wrappers must finish loading while the broker's
        # startup lease is held. Their children inherit this process group.
        if kind in ("surfdesign", "surfpro") or (kind in ("dplm", "nampnn") and runner.daemon):
            runner._ensure_daemon()
        initial = {"rng": capture(device), "metadata": _plain(_metadata(runner)),
                   "lifecycle": lifecycle_snapshot(runner, kind)}
        dump_tree(boot / "ready", initial)

    pending = {}

    def cpu_tree(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {key: cpu_tree(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(cpu_tree(item) for item in value)
        return _plain(value)

    def handle(payload):
        request = request_directory(payload["request_dir"])
        phase = payload.get("method")
        if phase == "prepare":
            if pending:
                raise ValueError("mutator already has a prepared or unfinished request")
            task = load_tree(request / "input")
            allowed = "compute_mutation" if kind in ("esm2", "esmif", "esm3") else "design_sequences"
            if task["method"] != allowed:
                raise ValueError("method is not supported by this mutator runner")
            cpu_state = dict(task["rng"], cuda=None)
            with contextlib.redirect_stdout(sys.stderr):
                restore(cpu_state, "cpu")
                prepare = getattr(runner, "prepare_request", None)
                if prepare is not None:
                    prepare(*task["args"], **task["kwargs"])
                prepared_rng = capture("cpu")
                prepared_rng["cuda"] = task["rng"]["cuda"]
            pending.update(request=str(request), task=task, rng=prepared_rng, phase="prepared")
            return {"request": request.name, "prepared": True}
        if str(request) != pending.get("request"):
            raise ValueError("mutator request does not own prepared state")
        if phase == "finalize":
            if pending.get("phase") != "executed":
                raise ValueError("mutator request has no executed result")
            dump_tree(request / "output", pending["output"])
            pending.clear()
            return {"request": request.name}
        if phase != "execute" or pending.get("phase") != "prepared":
            raise ValueError("invalid mutator request phase")
        task = pending["task"]
        with contextlib.redirect_stdout(sys.stderr):
            restore(pending["rng"], device)
            started = time.perf_counter()
            result = getattr(runner, task["method"])(*task["args"], **task["kwargs"])
            from astrolabe.evaluators.artifact_budget import validate_dump, DEFAULT_MAX_BYTES, DEFAULT_MAX_METADATA_BYTES
            from astrolabe.runtime.server import current_request
            budget = (current_request() or {}).get("output_limit_bytes") or DEFAULT_MAX_BYTES
            validate_dump(result, budget, min(budget, DEFAULT_MAX_METADATA_BYTES))
            result = cpu_tree(result)
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            state = capture(device)
            lifecycle = lifecycle_snapshot(runner, kind)
            lifecycle["request_seconds"] = elapsed
            pending["output"] = {"result": result, "rng": state, "lifecycle": lifecycle}
            pending["phase"] = "executed"
            del pending["task"], pending["rng"]
        return {"request": request.name, "executed": True}

    try:
        count = initial["lifecycle"]["model_load_count"]
        def current_lifecycle():
            state = lifecycle_snapshot(runner, kind)
            state["ready"] = state["ready"] and not bool(pending)
            state["active_request"] = Path(pending["request"]).name if pending else None
            return state
        serve(handle, actual, load_count=count,
              capabilities={"cpu_phases": ["cpu_prepare", "cpu_finalize"],
                            "phase_methods": {"prepare": "cpu_prepare", "finalize": "cpu_finalize"},
                            "request_staging": True},
              lifecycle=current_lifecycle)
    finally:
        closer = getattr(runner, "close", None) or getattr(runner, "shutdown", None)
        if closer:
            with contextlib.redirect_stdout(sys.stderr):
                closer()


if __name__ == "__main__":
    main()
