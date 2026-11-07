#!/usr/bin/env python3
"""Full public Astrolabe trajectory worker and offline exact comparison.

Run this file by absolute path in a fresh interpreter. --serve-worker explicitly
retains one ESMFold owner across sequential public trajectories in that process;
it is a distinct bounded protocol, not evidence of fresh-process isolation.
The parent owns resource
admission, trial profiles, cohorts and launch-to-real-exit timing. This module
never installs packages or downloads assets. Offline comparison imports neither
Astrolabe nor Torch. Only load locally trusted worker pickle files.

Worker::

    python /ABS/scripts/validation/scientific_workflow_acceptance.py --worker \
      --lane direct --source /tmp/as5m7-baseline-src --config /ABS/sequence.yaml \
      --output /ABS/direct

Persistent service::

    python /ABS/scripts/validation/scientific_workflow_acceptance.py --serve-worker \
      --source /ABS/source --output /ABS/new-owner-only-service-root

stdin is JSON lines of request_id/lane/config/source/envelope/output (absolute
paths); only resident_serial/adaptive lanes are accepted. ready.json, accepted/,
results/ and final.json carry the protocol; stdout remains application logs.
EOF or {"stop": true} closes the pool. Rejections are recorded in rejections/;
duplicate IDs never overwrite an existing result. Scientific failure stops the
service without replay. The parent owns process-group timeout and real exit.

Native workers additionally accept --envelope /ABS/trial.json, containing
``profiles_by_backend`` with explicit ``qualification: true`` profiles. The
parent sets ASTROLABE_MODEL_EXECUTION, ASTROLABE_MODEL_OWNER and resource env.
``execution.mode`` is recorded as legacy/direct, resident_serial or adaptive;
the parent owns the broker, so no resource session is created here.

``compare_workers(expected_root, actual_root)`` returns ``equal`` and ``checks``.
First-before RNG differences are reported for parent review, independently of
scientific equality. Pre-loader Python/NumPy/Torch CPU/seed states are exact;
CUDA initialization and state availability there are lifecycle diagnostics.
CUDA values are also exact when observed on both pre-loader sides. Every
post-loader, after-eval, warm before-eval and final RNG state is compared exactly.
Paths and explicitly named timestamp columns are excluded from science, while
all original evidence remains in evidence.pkl.
"""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
from contextlib import contextmanager, ExitStack
import copy
import csv
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import pickle
import platform
import random
import re
import resource
import subprocess
import sys
import threading
import time
import traceback
from types import SimpleNamespace
from typing import Any


SCHEMA = "astrolabe.scientific-workflow-acceptance/v1"
REFERENCE = Path(__file__).resolve().with_name("model_gpu_throughput.py")
EXPECTED_STEPS = [0, 1, 2]
INITIALIZER_ALIAS = "RuntimeInitializerProteinMPNN"
TIMESTAMP_COLUMNS = frozenset({"timestamp", "started_at", "ended_at", "created_at", "updated_at"})
PATH_FIELDS = frozenset({"path", "paths", "complex_paths", "selected_structure", "source_path",
                         "structure_path", "pdb_path", "output_path", "out_dir", "manifest_path"})
AA = dict(zip(("ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
               "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"),
              "ARNDCQEGHILKMFPSTWYV"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    def default(obj):
        if isinstance(obj, Path):
            return str(obj)
        if hasattr(obj, "tolist"):
            return obj.tolist()
        raise TypeError(type(obj).__name__)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=default) + "\n")
    tmp.replace(path)


def reference():
    """Absolute file import: never put the implementation checkout on sys.path."""
    name = "_scientific_workflow_exact_comparator"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, REFERENCE)
        if spec is None or spec.loader is None:
            raise ImportError(str(REFERENCE))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def freeze(value):
    """Detach evidence without drawing RNG or importing another source tree."""
    import numpy as np
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, np.generic):
        return value.copy()
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.dtype == torch.bfloat16:
            return {"torch_dtype": str(tensor.dtype), "shape": list(tensor.shape),
                    "storage_uint16": tensor.view(torch.uint16).numpy().copy()}
        return tensor.numpy().copy()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): freeze(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(freeze(v) for v in value)
    if isinstance(value, list):
        return [freeze(v) for v in value]
    if value is None or isinstance(value, (str, bytes, bool, int, float)):
        return value
    if hasattr(value, "__dict__"):
        return freeze(vars(value))
    raise TypeError(f"Cannot retain complete evidence: {type(value).__module__}.{type(value).__name__}")


def rng_state(torch=None):
    import numpy as np
    state = {"python": random.getstate(), "numpy": freeze(np.random.get_state())}
    if torch is not None:
        state["torch_cpu"] = torch.get_rng_state().numpy().copy()
        state["torch_seed"] = torch.initial_seed()
        # Never call is_available/device_count/init just to collect evidence.
        state["cuda_initialized"] = bool(torch.cuda.is_initialized())
        if state["cuda_initialized"]:
            # Only already initialized generators. get_rng_state_all() can
            # create generators on other visible devices and is not safe here.
            device = torch.cuda.current_device()
            generators = tuple(getattr(torch.cuda, "default_generators", ()))
            state["cuda"] = {str(device): generators[device].get_state().cpu().numpy().copy()}
            state["cuda_scope"] = "current initialized device only; no other visible device touched"
    return state


def usage():
    result = {}
    for name, who in (("SELF", resource.RUSAGE_SELF), ("CHILDREN", resource.RUSAGE_CHILDREN)):
        u = resource.getrusage(who)
        result[name] = {"user_cpu_s": u.ru_utime, "system_cpu_s": u.ru_stime,
                        "cpu_core_s": u.ru_utime + u.ru_stime,
                        "max_rss_bytes": u.ru_maxrss * (1024 if sys.platform != "darwin" else 1)}
    try:
        lines = Path("/proc/self/status").read_text().splitlines()
        result["self_rss_bytes"] = next(int(x.split()[1]) * 1024 for x in lines if x.startswith("VmRSS:"))
    except (OSError, StopIteration):
        result["self_rss_bytes"] = None
    return result


def usage_delta(before, after):
    return {name: {"cpu_core_s": after[name]["cpu_core_s"] - before[name]["cpu_core_s"],
                   "max_rss_bytes": after[name]["max_rss_bytes"]}
            for name in ("SELF", "CHILDREN")}


def gpu_facts():
    facts = {"pid": os.getpid(), "source": "nvidia-smi (no Torch CUDA initialization)"}
    for name, fields in (("gpus", "index,uuid,name,driver_version,memory.total"),
                         ("processes", "gpu_uuid,pid,process_name,used_gpu_memory")):
        query = "--query-gpu=" if name == "gpus" else "--query-compute-apps="
        try:
            p = subprocess.run(["nvidia-smi", query + fields, "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=8, check=False)
            facts[name] = {"returncode": p.returncode, "columns": fields.split(","),
                           "rows": list(csv.reader(p.stdout.splitlines())), "stderr": p.stderr}
        except (OSError, subprocess.TimeoutExpired) as e:
            facts[name] = {"unavailable": str(e)}
    return facts


def source_identity(source: Path, config: Path):
    files = set((source / "astrolabe").rglob("*.py")) | {source / "run.py"}
    files |= {p for p in (source / "config").rglob("*") if p.is_file()}
    hashes = {str(p.relative_to(source)): sha256(p) for p in sorted(files)}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"root": str(source), "files": hashes, "sha256": digest,
            "original_config": {"path": str(config), "sha256": sha256(config)},
            "harness_sha256": sha256(Path(__file__).resolve()),
            "comparator_sha256": sha256(REFERENCE)}


def select_source(source: Path, *, allow_existing_source=False):
    if any(n == "astrolabe" or n.startswith("astrolabe.") for n in sys.modules):
        if not allow_existing_source or "astrolabe" not in sys.modules:
            raise RuntimeError("Worker must start in a fresh interpreter without imported Astrolabe")
        module_origins(source)
        package = sys.modules["astrolabe"]
        if Path(package.__file__).resolve() != source / "astrolabe/__init__.py":
            raise RuntimeError("Persistent worker cannot change Astrolabe source")
        return package
    if not (source / "astrolabe/__init__.py").is_file() or not (source / "run.py").is_file():
        raise FileNotFoundError(f"Not an Astrolabe source tree: {source}")
    sys.path.insert(0, str(source))
    os.chdir(source)
    import astrolabe
    if Path(astrolabe.__file__).resolve() != source / "astrolabe/__init__.py":
        raise RuntimeError("Astrolabe imported from the wrong checkout")
    return astrolabe


def module_origins(source):
    origins = {}
    for name, module in list(sys.modules.items()):
        if name == "astrolabe" or name.startswith("astrolabe.") or name == "astrolabe_run_module":
            file = getattr(module, "__file__", None)
            if file:
                resolved = Path(file).resolve()
                if not resolved.is_relative_to(source):
                    raise RuntimeError(f"Mixed source import: {name}: {resolved}")
                origins[name] = str(resolved)
    return origins


def canonicalize_initializer_designer(cfg):
    """Relocate an inline initializer factory unchanged to an explicit alias.

    Both frozen sources require sequence_designer to name an existing mutator.
    This compatibility adapter does not add that alias to any mutation stage.
    Raw original_config is retained independently by compose_config.
    """
    from omegaconf import OmegaConf, open_dict
    initializer = cfg.get("initializer")
    if not OmegaConf.is_dict(initializer):
        return []
    designer = initializer.get("sequence_designer")
    if not OmegaConf.is_dict(designer):
        return []
    if initializer.get("mode") != "structure":
        raise ValueError("Inline sequence_designer compatibility is limited to the structure fixture")
    mapping = OmegaConf.to_container(designer, resolve=False)
    resolved = OmegaConf.to_container(designer, resolve=True)
    if not mapping.get("_type"):
        raise ValueError("Inline initializer sequence_designer must preserve an explicit _type")
    terms = cfg.mutator.mutator_terms
    if INITIALIZER_ALIAS in terms:
        raise ValueError(f"Reserved initializer alias already exists: {INITIALIZER_ALIAS}")
    for stage in cfg.get("stages", {}).values():
        selected = stage.get("mutators", []) if stage is not None else []
        if selected == INITIALIZER_ALIAS or (selected is not None and INITIALIZER_ALIAS in selected):
            raise ValueError("Reserved initializer alias must not be selected by a mutation stage")
    with open_dict(cfg):
        terms[INITIALIZER_ALIAS] = OmegaConf.create(copy.deepcopy(mapping))
        initializer.sequence_designer = INITIALIZER_ALIAS
    if OmegaConf.to_container(terms[INITIALIZER_ALIAS], resolve=True) != resolved:
        raise ValueError("Relocating the initializer mapping changes interpolation semantics")
    return [{"kind": "initializer_sequence_designer_mapping_to_alias",
             "alias": INITIALIZER_ALIAS,
             "reason": "Both target argparsers and structure initializers require a mutator alias",
             "before": {"initializer.sequence_designer": mapping},
             "after": {"initializer.sequence_designer": INITIALIZER_ALIAS,
                       "mutator.mutator_terms." + INITIALIZER_ALIAS: copy.deepcopy(mapping)},
             "resolved_mapping": resolved, "mapping_preserved_exactly": True}]


def compose_config(source, path, output, lane, *, canonicalizations=None):
    from omegaconf import OmegaConf, open_dict
    from astrolabe.cli import _compose_hydra_config
    original = OmegaConf.load(path)
    if "defaults" in original:
        raise ValueError("This worker requires the original standalone baseline YAML")
    base = _compose_hydra_config(config_dir=source / "config", config_name="base",
                                 overrides=["env=env_container"])
    with open_dict(base):
        cfg = OmegaConf.merge(base, original)
    changes = canonicalize_initializer_designer(cfg)
    if canonicalizations is not None:
        canonicalizations.extend(changes)
    # Beyond the recorded schema relocation, only namespace/operations change.
    with open_dict(cfg):
        cfg.out_dir = str(output / "trajectory")
        cfg.task_name = "scientific_workflow"
        if lane != "direct":
            cfg.execution.mode = lane
        elif "execution" in cfg:
            cfg.execution.mode = "legacy"
    plain = OmegaConf.to_container(original, resolve=True)
    if (plain.get("seed"), plain.get("steps"), plain.get("repeats")) != (17, 3, 1):
        raise ValueError("Expected original seed=17, steps=3, repeats=1")
    seq = plain.get("seqs", {}).get("A", "")
    if not seq.startswith("+TTCCP ") or len(seq.replace("+", "").replace(" ", "")) != 46:
        raise ValueError("Expected original 46 aa input with fixed +TTCCP segment")
    OmegaConf.save(cfg, output / "composed.yaml", resolve=False)
    return cfg, plain, OmegaConf.to_container(cfg, resolve=False)


@contextmanager
def patch(obj, name, wrapper):
    original = getattr(obj, name)
    setattr(obj, name, wrapper(original))
    try:
        yield
    finally:
        setattr(obj, name, original)


class Capture:
    def __init__(self, torch, lane, envelope, *, init_bindings=None, runtime_shapes=None):
        self.torch, self.lane, self.envelope = torch, lane, envelope
        self.evaluations, self.loads, self.model_loads, self.clients, self.requests = [], [], [], [], []
        self.mpnn_builds, self.initializer_calls = [], []
        self.evidence_seconds = 0.0
        self.profiles = envelope.get("profiles_by_backend", {})
        self.backend_by_client = {}
        self.runtime_shapes = {} if runtime_shapes is None else runtime_shapes
        self.init_bindings = {} if init_bindings is None else init_bindings
        self.model_acquisitions = []

    def collect(self, fn):
        start = time.perf_counter()
        try:
            return fn()
        finally:
            self.evidence_seconds += time.perf_counter() - start

    def loader(self, original):
        def wrapped(*args, **kwargs):
            path = args[0] if args else kwargs.get("f")
            name = path if isinstance(path, (str, bytes, os.PathLike)) else getattr(path, "name", None)
            requested = os.fsdecode(name) if isinstance(name, (str, bytes, os.PathLike)) else None
            row = {"requested": requested, "resolved": str(Path(requested).resolve()) if requested else None,
                   "pid": os.getpid(), "start": time.perf_counter()}
            self.loads.append(row)
            try:
                result = original(*args, **kwargs)
                row["ok"] = True
                return result
            finally:
                row["end"] = time.perf_counter()
                row["seconds"] = row["end"] - row["start"]
        return wrapped

    def ensure_model(self, original):
        def wrapped(evaluator, *args, **kwargs):
            if evaluator._model is not None:
                return original(evaluator, *args, **kwargs)
            row = {"backend": "esmfold", "class": type(evaluator).__qualname__,
                   "sourcefile": inspect.getsourcefile(type(evaluator)),
                   "rng_before": self.collect(lambda: rng_state(self.torch))}
            self.model_loads.append(row)
            row["start"] = time.perf_counter()
            try:
                return original(evaluator, *args, **kwargs)
            finally:
                row["end"] = time.perf_counter()
                row["seconds"] = row["end"] - row["start"]
                row["rng_after"] = self.collect(lambda: rng_state(self.torch))
                model = evaluator._model
                row["actual_model_class"] = type(model).__module__ + "." + type(model).__qualname__
                row["actual_model_sourcefile"] = inspect.getsourcefile(type(model)) if model is not None else None
                row["device"] = str(getattr(evaluator, "_device", None))
        return wrapped

    def evaluate(self, original):
        def wrapped(evaluator, input_feat, *args, **kwargs):
            from astrolabe.core import global_config
            def before():
                return {"step": int(global_config.step), "input": freeze(input_feat),
                        "rng_before": rng_state(self.torch), "usage_before": usage(),
                        "class": type(evaluator).__module__ + "." + type(evaluator).__qualname__,
                        "sourcefile": inspect.getsourcefile(type(evaluator))}
            row = self.collect(before)
            self.evaluations.append(row)
            nested_before = self.evidence_seconds
            row["start"] = time.perf_counter()
            try:
                scores = original(evaluator, input_feat, *args, **kwargs)
            except BaseException:
                row["end"] = time.perf_counter()
                row["rng_after"] = self.collect(lambda: rng_state(self.torch))
                row["failed"] = True
                raise
            row["end"] = time.perf_counter()
            row["seconds"] = row["end"] - row["start"]
            row["nested_evidence_s"] = self.evidence_seconds - nested_before
            row["call_seconds_excluding_nested_capture"] = row["seconds"] - row["nested_evidence_s"]
            def after():
                row["rng_after"] = rng_state(self.torch)
                row["usage_after"] = usage()
                # __dict__ retains *all* fields, including newly added fields.
                row["scores"] = freeze(vars(scores))
                row["scores_class"] = type(scores).__module__ + "." + type(scores).__qualname__
            self.collect(after)
            return scores
        return wrapped

    def client_init(self, original):
        signature = inspect.signature(original)
        def wrapped(client, *args, **kwargs):
            bound = signature.bind(client, *args, **kwargs)
            backend = bound.arguments["backend"]
            if backend not in self.profiles:
                raise ValueError(f"Missing explicit trial profile for backend={backend}")
            profile = copy.deepcopy(self.profiles[backend])
            if profile.get("qualification") is not True:
                raise ValueError("Only explicit qualification:true trial profiles are supported")
            # Parent has already bound these identities outside timing. The
            # runtime qualification constructor otherwise rehashes gigabytes.
            profile.pop("asset_sha256", None)
            bound.arguments["resource_profile"] = profile
            self.backend_by_client[id(client)] = backend
            row = {"backend": backend, "profile": copy.deepcopy(profile), "pid": os.getpid()}
            self.clients.append(row)
            acquisition = None
            if backend == "esmfold":
                acquisition = {"backend": backend, "pid": os.getpid(),
                               "rng_before": self.collect(lambda: rng_state(self.torch))}
                self.model_acquisitions.append(acquisition)
            row["start"] = time.perf_counter()
            if acquisition is not None:
                acquisition["start"] = row["start"]
            try:
                result = original(*bound.args, **bound.kwargs)
                if acquisition is not None:
                    acquisition["ok"] = True
                return result
            finally:
                row["end"] = time.perf_counter()
                row["seconds"] = row["end"] - row["start"]
                if acquisition is not None:
                    acquisition.update(end=row["end"], seconds=row["seconds"],
                                       rng_after=self.collect(lambda: rng_state(self.torch)))
                if hasattr(client, "metadata"):
                    row["metadata"] = self.collect(lambda: freeze(client.metadata))
                    row["owner_pid"] = client.pid
                    if acquisition is not None:
                        metadata = row["metadata"]
                        lifecycle = metadata.get("runtime_lifecycle", {})
                        owner = getattr(client, "owner", None)
                        entry = getattr(owner, "_entry", None)
                        reused = bool(entry is not None and entry.hits > 0)
                        acquisition.update(owner_pid=client.pid, model_id=lifecycle.get("model_id"),
                            identity=lifecycle.get("identity"), pool_attached=bool(entry is not None),
                            pool_reused=reused, actual_runtime_lifecycle=copy.deepcopy(lifecycle),
                            initialization_transition=copy.deepcopy(metadata.get("initialization_transition")))
                        # A private fallback can share model identity with the
                        # retained owner but has a different live profile list.
                        # Keep each owner's original list across trajectories.
                        model_id, identity = lifecycle.get("model_id"), lifecycle.get("identity")
                        if not reused and model_id and identity in self.runtime_shapes:
                            self.runtime_shapes[model_id] = self.runtime_shapes[identity]
                        if reused and acquisition["model_id"] in self.init_bindings:
                            acquisition["initialization_binding"] = copy.deepcopy(
                                self.init_bindings[acquisition["model_id"]])
        return wrapped

    def client_request(self, original):
        def wrapped(client, payload, *args, **kwargs):
            from astrolabe.model_execution import resource_shape
            backend = self.backend_by_client[id(client)]
            shape = self.collect(lambda: resource_shape(payload))
            # Settings are untouched. Bind the explicit trial envelope only to
            # this actual request shape, including the fixed 46 residue input.
            if backend == "esmfold" and payload.get("operation") == "infer":
                seqs = payload.get("sequences", [])
                if len(seqs) != 1 or len(seqs[0]) != 46 or not seqs[0].startswith("TTCCP"):
                    raise ValueError("ESMFold request is outside the fixed baseline scope")
                if payload.get("return_residue_plddt") is not True:
                    raise ValueError("ESMFold trajectory requires owner-side residue confidence reduction")
            profile = client._resource_profile
            allowed = self.profiles[backend].get("shapes", [])
            if backend == "esmfold" and payload.get("operation") == "infer" and allowed and shape not in allowed:
                raise ValueError(f"Request outside parent profile shapes: {backend}: {shape}")
            # A trajectory has one main-thread owner. kernel_handler makes a
            # shallow dict copy of its operational profile, retaining this list.
            # Bind that list to the actual request only; never alter payload or
            # scientific settings. CPU output_to_pdb and the one initializer
            # get their own exact scope, rather than an unclassified profile.
            lifecycle = client.metadata.get("runtime_lifecycle", {})
            identity = lifecycle.get("identity")
            scopes = self.runtime_shapes.get(lifecycle.get("model_id"), self.runtime_shapes.get(identity))
            if scopes is None:
                raise RuntimeError("Cannot bind the trajectory-owned runtime trial profile")
            scopes[:] = [shape]
            row = {"backend": backend, "operation": payload.get("operation"), "shape": shape,
                   "profile": copy.deepcopy(profile), "runtime_scopes": list(scopes)}
            if backend == "esmfold" and payload.get("operation") == "infer":
                row["payload"] = self.collect(lambda: freeze(payload))
            self.requests.append(row)
            row["start"] = time.perf_counter()
            try:
                return original(client, payload, *args, **kwargs)
            finally:
                row["end"] = time.perf_counter()
                row["seconds"] = row["end"] - row["start"]
                row["metadata"] = self.collect(lambda: freeze(client.metadata))
                row["receipt"] = self.collect(lambda: freeze(getattr(client, "last_runtime_receipt", None)))
        return wrapped

    def kernel(self, original):
        def wrapped(settings, *args, **kwargs):
            profile = settings.get("_runtime_profile")
            if not profile or profile.get("qualification") is not True:
                raise ValueError("Native kernel did not receive the explicit trial profile")
            profile["shapes"] = list(profile.get("shapes", []))
            self.runtime_shapes[profile["identity"]] = profile["shapes"]
            return original(settings, *args, **kwargs)
        return wrapped

    def lifecycle(self, phase, records):
        def decorate(original):
            def wrapped(*args, **kwargs):
                row = {"backend": "ligandmpnn", "phase": phase,
                       "sourcefile": inspect.getsourcefile(original),
                       "rng_before": self.collect(lambda: rng_state(self.torch))}
                records.append(row)
                row["start"] = time.perf_counter()
                try:
                    result = original(*args, **kwargs)
                    row["ok"] = True
                    if phase == "initial_sequence_design":
                        row["result"] = self.collect(lambda: freeze(result))
                    return result
                finally:
                    row["end"] = time.perf_counter()
                    row["seconds"] = row["end"] - row["start"]
                    row["rng_after"] = self.collect(lambda: rng_state(self.torch))
            return wrapped
        return decorate

    @contextmanager
    def install(self):
        from astrolabe.evaluators.esmfold import ESMFoldEvaluator
        with ExitStack() as stack:
            stack.enter_context(patch(self.torch, "load", self.loader))
            stack.enter_context(patch(ESMFoldEvaluator, "_ensure_model", self.ensure_model))
            stack.enter_context(patch(ESMFoldEvaluator, "run_function", self.evaluate))
            from astrolabe.mutators import base as mutator_base
            from astrolabe.mutators.ligandmpnn import LigandMPNNMutator
            if self.lane == "direct":
                stack.enter_context(patch(mutator_base, "_build_mpnn_mutator",
                    self.lifecycle("build_mutator_including_model_load", self.mpnn_builds)))
            stack.enter_context(patch(LigandMPNNMutator, "design_initial_sequences",
                self.lifecycle("initial_sequence_design", self.initializer_calls)))
            if self.lane != "direct":
                from astrolabe import model_execution
                NativeModelClient = model_execution.NativeModelClient
                stack.enter_context(patch(model_execution, "kernel_handler", self.kernel))
                stack.enter_context(patch(NativeModelClient, "__init__", self.client_init))
                stack.enter_context(patch(NativeModelClient, "request", self.client_request))
            yield


def parse_pdb(path):
    import numpy as np
    atoms, coords, residues, conect = [], [], {}, []
    for line in path.read_text().splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            # PDB topology and all non-coordinate atom attributes, independent
            # of header creation dates. Preserve occupancy/B-factor exactly.
            atoms.append((line[:6].strip(), line[6:11].strip(), line[12:16], line[16:17],
                          line[17:20], line[21:22], line[22:26], line[26:27],
                          line[54:60].strip(), line[60:66].strip(), line[72:80]))
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            key = (line[21:22], line[22:26], line[26:27])
            residues.setdefault(key, line[17:20].strip())
        elif line.startswith(("CONECT", "TER", "MODEL", "ENDMDL", "SSBOND", "LINK")):
            conect.append(line.rstrip())
    sequences = {}
    for (chain, _, _), residue in residues.items():
        sequences[chain] = sequences.get(chain, "") + AA.get(residue, "X")
    return {"atoms": atoms, "coordinates": np.asarray(coords, dtype=np.float64).reshape((-1, 3)),
            "topology_records": conect, "sequences": sequences}


def artifacts(output):
    tables, pdbs, manifests = [], [], []
    for p in sorted((output / "trajectory").rglob("scores.csv")):
        with p.open(newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if reader.fieldnames is None or len(set(reader.fieldnames)) != len(reader.fieldnames):
                raise ValueError(f"Missing or duplicate CSV columns: {p}")
            if any(None in row or None in row.values() for row in rows):
                raise ValueError(f"Malformed scores CSV: {p}")
        tables.append({"path": str(p), "relative_path": str(p.relative_to(output)),
                       "sha256": sha256(p), "columns": reader.fieldnames, "rows": rows})
    for p in sorted((output / "trajectory").rglob("*.pdb")):
        match = re.search(r"step_(\d+)", p.name)
        if match:
            pdbs.append({"path": str(p), "relative_path": str(p.relative_to(output)), "sha256": sha256(p),
                         "name": p.name, "step": int(match.group(1)),
                         "raw_text": p.read_text(), "structure": parse_pdb(p)})
    for p in sorted((output / "trajectory").rglob("initializer_manifest.json")):
        manifests.append({"path": str(p), "relative_path": str(p.relative_to(output)),
                          "sha256": sha256(p), "contents": json.loads(p.read_text())})
    return {"tables": tables, "pdbs": pdbs, "initializer_manifests": manifests}


def is_path_field(key):
    return key in PATH_FIELDS or key.endswith(("_path", "_paths", "_file", "_dir"))


def science_tree(value, namespace):
    """Ignore path *values*, keep mapping keys and all scientific values.

    Explicit schema path fields retain their structure using a marker. Other
    absolute paths have worker/source namespaces normalized, never basename-only
    matching, so unrelated file identities cannot collapse accidentally.
    """
    if isinstance(value, Mapping):
        return {str(k): path_shape(v) if is_path_field(str(k)) else science_tree(v, namespace)
                for k, v in value.items() if str(k) not in TIMESTAMP_COLUMNS}
    if isinstance(value, tuple):
        return tuple(science_tree(v, namespace) for v in value)
    if isinstance(value, list):
        return [science_tree(v, namespace) for v in value]
    if isinstance(value, str):
        for root, marker in namespace:
            value = value.replace(root, marker)
        return value
    return value


def path_shape(value):
    if isinstance(value, Mapping):
        return {str(k): path_shape(v) for k, v in value.items()}
    if isinstance(value, list):
        return [path_shape(v) for v in value]
    if isinstance(value, tuple):
        return tuple(path_shape(v) for v in value)
    return None if value is None else "<PATH>"


def science_view(evidence, report):
    namespace = sorted([(str(report["output"]), "<OUTPUT>"),
                        (str(report["source"]), "<SOURCE>")], key=lambda x: -len(x[0]))
    cfg = {k: v for k, v in evidence["original_config"].items()
           if k not in {"task_name", "out_dir", "execution"}}
    evaluations = []
    for i, row in enumerate(evidence["evaluations"]):
        item = {k: row[k] for k in ("step", "input", "scores", "rng_after")}
        if i:
            item["rng_before"] = row["rng_before"]
        evaluations.append(science_tree(item, namespace))
    tables = []
    for table in evidence["artifacts"]["tables"]:
        cols = [c for c in table["columns"] if c not in TIMESTAMP_COLUMNS and not is_path_field(c)]
        tables.append({"columns": cols, "rows": [science_tree({c: row[c] for c in cols}, namespace)
                                                for row in table["rows"]]})
    return {"config": science_tree(cfg, namespace), "evaluations": evaluations, "tables": tables,
            "pdbs": [{k: p[k] for k in ("name", "step", "structure")}
                     for p in evidence["artifacts"]["pdbs"]],
            "initializer_manifests": [science_tree(m["contents"], namespace)
                                      for m in evidence["artifacts"]["initializer_manifests"]],
            "final_rng": evidence["final_rng"]}


def initialization_boundaries(evidence):
    """Logical acquisition is the native boundary, never an invented load."""
    return evidence["model_acquisitions"] if "model_acquisitions" in evidence else evidence.get("model_loads", [])


def path_backed_loads(evidence):
    return [r for r in evidence.get("torch_loads", [])
            if r.get("ok") is True and isinstance(r.get("resolved"), str)
            and Path(r["resolved"]).is_absolute()]


def acquisition_complete(row):
    lifecycle = row.get("actual_runtime_lifecycle", {})
    transition = row.get("initialization_transition")
    return (row.get("backend") == "esmfold" and row.get("ok") is True
            and isinstance(lifecycle, Mapping)
            and all(row.get(k) for k in ("model_id", "identity", "owner_pid", "pid"))
            and type(row.get("pool_reused")) is bool and type(row.get("pool_attached")) is bool
            and (not row["pool_reused"] or row["pool_attached"])
            and all(k in row for k in ("rng_before", "rng_after", "start", "end", "seconds"))
            and row["end"] >= row["start"] and row["seconds"] >= 0
            and lifecycle.get("model_id") == row["model_id"]
            and lifecycle.get("identity") == row["identity"]
            and lifecycle.get("pid") == row["owner_pid"]
            and type(lifecycle.get("loader_count")) is int and lifecycle["loader_count"] == 1
            and isinstance(transition, Mapping) and all(
                isinstance(transition.get(k), Mapping) and all(
                    field in transition[k] for field in ("python", "numpy", "torch", "torch_seed"))
                for k in ("before", "cuda_before_place", "after")))


def validate_initialization_binding(row, report):
    """Bind a warm borrow to immutable, completed, real cold-load evidence."""
    try:
        binding = row["initialization_binding"]
        path = Path(binding["worker_path"])
        if (not path.is_absolute() or path.name != "worker.json"
                or str(path.resolve()) == str((Path(report["output"]) / "worker.json").resolve())
                or sha256(path) != binding["sha256"]):
            return False
        cold, receipt = load_worker(path.parent)
        boundaries = cold.get("model_acquisitions", [])
        if len(boundaries) != 1:
            return False
        first = boundaries[0]
        if (not acquisition_complete(first) or first["pool_reused"]
                or not first["pool_attached"] or not any(
                    r.get("pid") == first["owner_pid"] for r in path_backed_loads(cold))
                or receipt.get("status") != "completed"
                or receipt.get("persistent") is not True or report.get("persistent") is not True):
            return False
        for key in ("model_id", "owner_pid", "identity"):
            if binding.get(key) != first[key] or row.get(key) != first[key]:
                return False
        if (receipt.get("pid") != first["owner_pid"] or report.get("pid") != row["owner_pid"]
                or receipt.get("pid") != report.get("pid")
                or receipt.get("source") != report.get("source")
                or not receipt.get("source_identity", {}).get("sha256")
                or receipt.get("source_identity", {}).get("sha256") !=
                   report.get("source_identity", {}).get("sha256")):
            return False
        if not reference().compare_outputs(first["initialization_transition"],
                                           row.get("initialization_transition"))["equal"]:
            return False
        # Validate the original scientific/checkpoint/RNG gates as well as its
        # digests. The first record cannot recurse: it must be a cold attach.
        return all(validate_evidence(cold, receipt).values())
    except Exception:
        return False


def validate_evidence(evidence, report):
    ev = evidence.get("evaluations", [])
    boundaries = initialization_boundaries(evidence)
    art = evidence.get("artifacts", {})
    tables, pdbs = art.get("tables", []), art.get("pdbs", [])
    checks = {"completed": report.get("status") == "completed", "schema": report.get("schema") == SCHEMA,
              "three_captures": len(ev) == 3 and [r.get("step") for r in ev] == EXPECTED_STEPS,
              "complete_captures": all(all(k in r for k in ("input", "scores", "rng_before", "rng_after"))
                                       and not r.get("failed") for r in ev),
              "three_score_rows": len(tables) == 1 and len(tables[0].get("rows", [])) == 3,
              "three_step_pdbs": len(pdbs) == 3 and sorted(p.get("step") for p in pdbs) == EXPECTED_STEPS,
              "final_rng": bool(evidence.get("final_rng")),
              "loader_rng": len(boundaries) == 1 and
                  all(k in boundaries[0] for k in ("rng_before", "rng_after")),
              "source_consistent": report.get("source_unchanged") is True and bool(report.get("module_origins")),
              "checkpoint_loads_recorded": bool(evidence.get("torch_loads"))}
    if "model_acquisitions" in evidence:
        checks["model_acquisition_complete"] = len(boundaries) == 1 and all(
            acquisition_complete(r) and r["pid"] == report.get("pid") and
            (not r["pool_attached"] or (report.get("persistent") is True and r["owner_pid"] == report.get("pid")))
            for r in boundaries)
        warm = [r for r in boundaries if r.get("pool_reused")]
        checks["warm_initialization_bound"] = all(validate_initialization_binding(r, report) for r in warm)
        checks["checkpoint_loads_recorded"] = (checks["warm_initialization_bound"] if warm
                                                 else bool(path_backed_loads(evidence)))
    rows = [r for t in tables for r in t.get("rows", [])]
    # DecisionMaker writes runtime protomer names (pA), while standalone
    # configs identify this protein as A. Keep the actual full CSV schema in
    # scientific comparison; use this explicit alias map only for validation.
    sequence_columns = [c for c in ("pA_sequence", "A_sequence")
                        if rows and all(c in r for r in rows)]
    checks["protein_sequence_column"] = bool(sequence_columns)
    checks["score_steps"] = [r.get("step") for r in rows] == [str(s) for s in EXPECTED_STEPS]
    checks["scores_and_acceptance_present"] = bool(rows) and all(
        all(r.get(k) not in (None, "") for k in ("weighted_score", "accepted") + tuple(sequence_columns)) for r in rows)
    checks["accepted_flags_valid"] = bool(rows) and all(r.get("accepted", "").lower() in
                                                       {"true", "false", "1", "0"} for r in rows)
    sequences = [r.get(c, "") for r in rows for c in sequence_columns]
    sequences += [seq for r in ev for seq in r.get("input", {}).values()]
    sequences += [seq for p in pdbs for seq in p.get("structure", {}).get("sequences", {}).values()]
    checks["fixed_prefix_and_length"] = bool(sequences) and all(
        isinstance(s, str) and s.startswith("TTCCP") and len(s) == 46 for s in sequences)
    checks["pdb_atoms_present"] = bool(pdbs) and all(len(p["structure"]["atoms"]) > 0 for p in pdbs)
    checks["trajectory_sequence_alignment"] = bool(sequence_columns) and len(rows) == len(ev) == len(pdbs) == 3 and all(
        list(ev[i].get("input", {}).values()) == [rows[i].get(sequence_columns[0])] and
        list(pdbs[i].get("structure", {}).get("sequences", {}).values()) == [rows[i].get(sequence_columns[0])] and
        all(rows[i].get(c) == rows[i].get(sequence_columns[0]) for c in sequence_columns)
        for i in range(3))
    mode = evidence.get("original_config", {}).get("initializer", {}).get("mode")
    manifests = art.get("initializer_manifests", [])
    checks["initializer_manifest"] = (len(manifests) == 1 if mode == "structure" else not manifests)
    if mode == "structure" and len(manifests) == 1:
        manifest = manifests[0].get("contents", {})
        checks["initializer_manifest"] = all(k in manifest for k in (
            "mode", "sequence_designer", "chain_map", "resolved_chain_map", "selected_structure",
            "structure", "sequence_design", "designed_sequences_by_chain", "initialized_sequences"))
        checks["initializer_sequences"] = bool(manifest.get("initialized_sequences")) and all(
            s.startswith("TTCCP") and len(s) == 46 for s in manifest.get("initialized_sequences", {}).values())
    checks["rng_fields_present"] = bool(ev) and all(
        all(k in state for k in ("python", "numpy", "torch_cpu", "torch_seed", "cuda_initialized"))
        for state in [evidence.get("final_rng", {})] +
        [r.get(boundary, {}) for r in ev + boundaries
         for boundary in ("rng_before", "rng_after")])
    if report.get("lane") != "direct":
        requests = evidence.get("native_requests", [])
        infer_requests = [r for r in requests
                          if r.get("backend") == "esmfold" and r.get("operation") == "infer"]
        checks["native_receipts"] = bool(requests) and all(r.get("metadata") and r.get("receipt") for r in requests)
        checks["native_scoped_profiles_matched"] = bool(requests) and all(
            r.get("receipt", {}).get("profile_matched") is True for r in requests)
        checks["three_native_infer_requests"] = len(infer_requests) == 3
        checks["owner_residue_confidence_requested"] = bool(infer_requests) and all(
            r.get("payload", {}).get("return_residue_plddt") is True and
            r.get("payload", {}).get("operation") == "infer" for r in infer_requests)
        checks["native_infer_inputs_match"] = len(infer_requests) == len(ev) == 3 and all(
            r.get("payload", {}).get("sequences") == list(e.get("input", {}).values())
            for r, e in zip(infer_requests, ev))
        checks["explicit_profiles"] = bool(evidence.get("native_clients")) and all(
            r.get("profile", {}).get("qualification") is True for r in evidence["native_clients"])
    return checks


def esmfold_infer_payloads(evidence, lane):
    if lane == "direct":
        # The baseline has no RPC: describe its original infer(sequences) call.
        # The parent adds the owner reduction flag when constructing profiles.
        return [{"operation": "infer", "sequences": list(r["input"].values())}
                for r in evidence["evaluations"]]
    # Preserve the actual wire payload, including the owner reduction flag;
    # reconstructing from evaluations would silently describe the old protocol.
    return [r["payload"] for r in evidence["native_requests"]
            if r.get("backend") == "esmfold" and r.get("operation") == "infer" and "payload" in r]


def load_worker(root):
    root = Path(root).resolve()
    report = json.loads((root / "worker.json").read_text())
    expected = report.get("evidence_sha256")
    if not expected or sha256(root / "evidence.pkl") != expected:
        raise ValueError(f"Worker evidence digest missing or mismatched: {root}")
    # Locally trusted evidence only. The digest is integrity, not authentication.
    with (root / "evidence.pkl").open("rb") as f:
        evidence = pickle.load(f)
    if not report.get("arrays_sha256") or sha256(root / "arrays.npz") != report["arrays_sha256"]:
        raise ValueError(f"Worker array digest missing or mismatched: {root}")
    for group in ("tables", "pdbs", "initializer_manifests"):
        for row in evidence.get("artifacts", {}).get(group, []):
            relative = Path(row["relative_path"])
            file = (root / relative).resolve()
            if relative.is_absolute() or not file.is_relative_to(root) or sha256(file) != row["sha256"]:
                raise ValueError(f"Trajectory artifact missing or changed: {relative}")
    return evidence, report


def compare_workers(expected, actual) -> dict:
    """Compare trusted worker roots, failing closed for missing/incomplete data."""
    result = {"equal": False, "checks": {}, "expected": str(expected), "actual": str(actual),
              "path_policy": "explicit path fields excluded; remaining source/output namespaces normalized",
              "excluded_timestamp_columns": sorted(TIMESTAMP_COLUMNS), "tolerance": None}
    try:
        left, lr = load_worker(expected)
        right, rr = load_worker(actual)
        lc, rc = validate_evidence(left, lr), validate_evidence(right, rr)
        result["validation"] = {"expected": lc, "actual": rc}
        result["checks"]["expected_complete"] = all(lc.values())
        result["checks"]["actual_complete"] = all(rc.values())
        if not all(lc.values()) or not all(rc.values()):
            return result
        compare = reference().compare_outputs
        lv, rv = science_view(left, lr), science_view(right, rr)
        science = compare(lv, rv)
        result["comparison"] = science
        result["checks"]["scientific_equal"] = science["equal"]
        for field in lv:
            result["checks"][field + "_equal"] = not any(
                d["path"] == "$" or d["path"] == "$." + field or d["path"].startswith(("$." + field + ".", "$." + field + "["))
                for d in science["differences"])
        first_before = compare(left["evaluations"][0]["rng_before"], right["evaluations"][0]["rng_before"])
        left_boundaries, right_boundaries = initialization_boundaries(left), initialization_boundaries(right)
        raw_loader_before = compare([r["rng_before"] for r in left_boundaries],
                                    [r["rng_before"] for r in right_boundaries])
        # These are different observation positions relative to CUDA context
        # setup. Retain the original complete states above and in evidence.pkl;
        # compare all CPU/seed fields without manufacturing missing CUDA data.
        aligned_left, aligned_right = [], []
        lifecycle_keys = {"cuda_initialized", "cuda", "cuda_scope"}
        for lrow, rrow in zip(left_boundaries, right_boundaries):
            lb, rb = lrow["rng_before"], rrow["rng_before"]
            la = {k: v for k, v in lb.items() if k not in lifecycle_keys}
            ra = {k: v for k, v in rb.items() if k not in lifecycle_keys}
            if "cuda" in lb and "cuda" in rb:
                la["cuda"], ra["cuda"] = lb["cuda"], rb["cuda"]
            aligned_left.append(la)
            aligned_right.append(ra)
        loader_before = compare(aligned_left, aligned_right)
        loader_after = compare([r["rng_after"] for r in left_boundaries],
                               [r["rng_after"] for r in right_boundaries])
        result["checks"]["loader_before_rng_equal"] = loader_before["equal"]
        result["checks"]["loader_after_full_rng_equal"] = loader_after["equal"]
        result["rng_comparisons"] = {"loader_before_aligned": loader_before, "loader_after_full": loader_after}
        result["diagnostics"] = {"first_before_rng": first_before, "loader_before_full_rng": raw_loader_before}
        result["requires_review"] = not first_before["equal"] or not raw_loader_before["equal"]
        result["rng_boundary_policy"] = {
            "first_eval_before": "raw diagnostic; requires_review does not gate equal",
            "loader_before": "all CPU/Python/NumPy/seed exact; CUDA exact if available on both sides",
            "loader_before_cuda_initialization_and_availability": "raw lifecycle diagnostic",
            "loader_after_eval_after_warm_before_final": "full exact RNG equality",
        }
        result["equal"] = all(result["checks"].values())
    except Exception as e:
        result["error"] = {"type": type(e).__name__, "message": str(e)}
    return result


def save_npz(path, evidence):
    import numpy as np
    arrays = {}
    def visit(value, key):
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise TypeError("Object arrays cannot be safely retained in NPZ")
            arrays[key] = value
        elif isinstance(value, Mapping):
            for k, v in value.items():
                visit(v, key + "." + str(k))
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                visit(v, key + f"[{i}]")
    visit(evidence, "evidence")
    np.savez(path, **arrays)


def worker(args, *, persistent=False, allow_existing_source=False, init_bindings=None, runtime_shapes=None):
    if allow_existing_source and not persistent:
        raise ValueError("Existing Astrolabe imports are allowed only in persistent serve mode")
    if persistent and (args.lane == "direct" or threading.current_thread() is not threading.main_thread()):
        raise ValueError("Persistent scientific workers require a native lane on the main thread")
    source, output, config = args.source.resolve(), args.output.resolve(), args.config.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Worker output must be absent or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    report = {"schema": SCHEMA, "lane": args.lane, "source": str(source), "output": str(output),
              "config": str(config), "pid": os.getpid(), "status": "running", "persistent": persistent,
              "request_id": getattr(args, "request_id", None),
              "cpu_affinity": sorted(os.sched_getaffinity(0)),
              "clock_anchor": {"perf_counter": time.perf_counter(), "unix_time": time.time()},
              "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
              "composition": "target base + env=env_container + original standalone YAML",
              "environment": {k: v for k, v in os.environ.items() if k.startswith(("ASTROLABE_", "CUDA_", "SLURM_"))},
              "timing_contract": {"parent": ("persistent request within one bounded service; parent times real service exit"
                                               if persistent else "launch to real interpreter exit, including instrumentation"),
                  "public_run": "Astrolabe(cfg).run instrumented wall time; no evidence hashes",
                  "calls": "blocking original call boundaries; includes native execute host materialization; excludes adjacent evidence capture",
                  "cold": "model initialization outside first eval + first high-level eval",
                  "steady": "high-level eval requests 2 and 3",
                  "checkpoint_hashing": "none; parent binds checkpoint content identity"}}
    evidence = {"evaluations": [], "model_loads": [], "torch_loads": [], "native_clients": [],
                "native_requests": [], "artifacts": {"tables": [], "pdbs": [], "initializer_manifests": []}}
    failure, cap = None, None
    evidence_post_seconds = 0.0
    write_json(output / "worker.json", report)
    try:
        expected_mode = "in_process" if args.lane == "direct" else "daemon"
        if os.environ.get("ASTROLABE_MODEL_EXECUTION") != expected_mode:
            raise ValueError(f"Parent must set ASTROLABE_MODEL_EXECUTION={expected_mode}")
        if os.environ.get("ASTROLABE_MODEL_OWNER") != "trajectory":
            raise ValueError("Parent must set ASTROLABE_MODEL_OWNER=trajectory")
        if args.lane == "direct" and source != Path("/tmp/as5m7-baseline-src").resolve():
            raise ValueError("Direct lane must use the actual /tmp/as5m7-baseline-src checkout")
        started = time.perf_counter()
        report["source_identity"] = source_identity(source, config)
        report["source_hash_before_s"] = time.perf_counter() - started
        envelope = json.loads(args.envelope.read_text()) if args.envelope else {}
        report["envelope"] = envelope
        package = (select_source(source, allow_existing_source=True) if allow_existing_source
                   else select_source(source))
        report["public_entry"] = {"class": package.Astrolabe.__module__ + "." + package.Astrolabe.__qualname__,
                                  "sourcefile": inspect.getsourcefile(package.Astrolabe)}
        import torch
        import numpy as np
        report["versions"] = {"astrolabe": package.__version__, "torch": torch.__version__,
                              "numpy": np.__version__, "torch_cuda_build": torch.version.cuda,
                              "cudnn_build": torch.backends.cudnn.version()}
        canonicalizations = []
        cfg, original, composed = compose_config(source, config, output, args.lane,
                                                  canonicalizations=canonicalizations)
        report["canonicalizations"] = canonicalizations
        evidence.update(original_config=original, composed_config=composed, canonicalizations=canonicalizations)
        write_json(output / "worker.json", report)
        report["gpu_before"] = gpu_facts()
        cap = Capture(torch, args.lane, envelope, init_bindings=init_bindings, runtime_shapes=runtime_shapes)
        evidence.update(evaluations=cap.evaluations, model_loads=cap.model_loads, torch_loads=cap.loads,
                        native_clients=cap.clients, native_requests=cap.requests,
                        direct_mpnn_builds=cap.mpnn_builds, initializer_calls=cap.initializer_calls)
        if args.lane != "direct" and persistent:
            evidence["model_acquisitions"] = cap.model_acquisitions
        with cap.install():
            report["usage_before_run"] = usage()
            begin = time.perf_counter()
            try:
                package.Astrolabe(cfg).run()
            finally:
                report["public_run_wall_instrumented_s"] = time.perf_counter() - begin
                report["usage_after_run"] = usage()
                start = time.perf_counter()
                evidence["final_rng"] = rng_state(torch)
                evidence_post_seconds += time.perf_counter() - start
        report["status"] = "completed"
    except BaseException as e:
        failure = e
        report["status"] = "failed"
        report["failure"] = {"type": type(e).__name__, "message": str(e), "traceback": traceback.format_exc()}
    finally:
        start = time.perf_counter()
        try:
            if cap and cap.model_acquisitions:
                evidence["model_acquisitions"] = cap.model_acquisitions
            evidence["artifacts"] = artifacts(output)
            report["module_origins"] = module_origins(source)
            if "source_identity" in report:
                before = time.perf_counter()
                report["source_identity_after"] = source_identity(source, config)
                report["source_hash_after_s"] = time.perf_counter() - before
                report["source_unchanged"] = report["source_identity_after"] == report["source_identity"]
            report["gpu_after"] = gpu_facts()
            report["esmfold_infer_payloads"] = esmfold_infer_payloads(evidence, args.lane)
            report["esmfold_infer_payloads_origin"] = (
                "reconstructed_baseline_infer_calls" if args.lane == "direct" else "captured_native_requests")
            report["score_sequence_columns"] = [c for t in evidence["artifacts"]["tables"] for c in t["columns"]
                                                 if c in ("pA_sequence", "A_sequence")]
            report["torch_loads"] = evidence["torch_loads"]
            report["torch_load_count"] = len(evidence["torch_loads"])
            report["torch_load_counts_by_resolved_path"] = dict(Counter(str(r["resolved"]) for r in evidence["torch_loads"]))
            report["model_loads"] = [{k: v for k, v in r.items() if k not in ("rng_before", "rng_after")}
                                     for r in evidence["model_loads"]]
            report["model_acquisitions"] = evidence.get("model_acquisitions", [])
            actual_initializations = {}
            for row in cap.clients if cap else []:
                lifecycle = row.get("metadata", {}).get("runtime_lifecycle", {})
                if lifecycle.get("model_id") and lifecycle.get("pid"):
                    actual_initializations[(lifecycle["pid"], lifecycle["model_id"])] = lifecycle
            report["actual_runtime_initializations"] = list(actual_initializations.values())
            warm_ids = {r.get("model_id") for r in evidence.get("model_acquisitions", []) if r.get("pool_reused")}
            report["actual_initialization_count"] = sum(
                r.get("loader_count", 0) for r in actual_initializations.values() if r["model_id"] not in warm_ids)
            report["mpnn_lifecycle"] = [{k: v for k, v in r.items() if k not in ("rng_before", "rng_after", "result")}
                                        for r in evidence.get("direct_mpnn_builds", []) + evidence.get("initializer_calls", [])]
            report["native_clients"] = evidence["native_clients"]
            report["native_requests"] = evidence["native_requests"]
            report["evaluations"] = [{k: r[k] for k in ("step", "start", "end", "seconds", "class", "sourcefile") if k in r}
                                      for r in evidence["evaluations"]]
            if cap and len(cap.evaluations) == 3:
                first = cap.evaluations[0]
                loads = cap.clients if args.lane != "direct" else cap.model_loads + cap.mpnn_builds
                report["cold_model_load_plus_first_eval_s"] = first["seconds"] + sum(
                    max(0.0, min(r["end"], first["start"]) - r["start"]) for r in loads)
                report["steady_request_s"] = [r["seconds"] for r in cap.evaluations[1:]]
            if "usage_before_run" in report:
                report["run_rusage"] = usage_delta(report["usage_before_run"], report["usage_after_run"])
            report["usage_at_evidence"] = usage()
            with (output / "evidence.pkl").open("xb") as f:
                pickle.dump(evidence, f, protocol=5)
            save_npz(output / "arrays.npz", evidence)
            before = time.perf_counter()
            report["evidence_sha256"] = sha256(output / "evidence.pkl")
            report["arrays_sha256"] = sha256(output / "arrays.npz")
            report["evidence_hash_s"] = time.perf_counter() - before
            report["checks"] = validate_evidence(evidence, report)
            if not all(report["checks"].values()):
                report["status"] = "failed"
                failure = failure or RuntimeError("Trajectory evidence failed completeness checks")
        except BaseException as e:
            report["status"] = "failed"
            report["evidence_failure"] = traceback.format_exc()
            failure = failure or e
        evidence_post_seconds += time.perf_counter() - start
        report["evidence_generation_s"] = {"within_run_host_capture": cap.evidence_seconds if cap else 0,
                                            "post_run_including_source_hash_and_gpu_facts": evidence_post_seconds}
        write_json(output / "worker.json", report)
    return 1 if failure else 0


def serve_worker(args):
    """New bounded sequential protocol; never claims fresh-process isolation.

    Replies are files; stdout remains application logging. The coordinator owns
    process-group timeouts. Unknown outcomes are never replayed.
    """
    if threading.current_thread() is not threading.main_thread():
        raise ValueError("Persistent worker service requires the main thread")
    source, root = args.source.resolve(), args.output.resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    for name in ("accepted", "results", "rejections"):
        (root / name).mkdir(mode=0o700)
    final = {"schema": SCHEMA, "pid": os.getpid(), "source": str(source),
             "protocol": "persistent-native-sequential/v1", "status": "starting",
             "accepted": 0, "rejected": 0, "completed": 0, "exitcode": 1}
    bindings, runtime_shapes, seen_ids, outputs = {}, {}, set(), []
    pool = None
    try:
        select_source(source)
        from astrolabe.runtime.model_pool import local_model_pool
        source_snapshot = source_identity(source, source / "run.py")
        with local_model_pool(max_models=1, backends={"esmfold"}) as pool:
            final["status"] = "ready"
            write_json(root / "ready.json", {"schema": SCHEMA, "protocol": final["protocol"],
                "status": "ready", "pid": os.getpid(), "source": str(source),
                "source_identity": source_snapshot, "pool": pool.metadata})
            for line in sys.stdin:
                request = None
                try:
                    request = json.loads(line)
                    if isinstance(request, dict) and set(request) == {"stop"} and request["stop"] is True:
                        final["stop_reason"] = "explicit_stop"
                        break
                    if not isinstance(request, dict) or set(request) != {
                            "request_id", "lane", "config", "source", "envelope", "output"}:
                        raise ValueError("Request requires request_id/lane/config/source/envelope/output")
                    request_id = request["request_id"]
                    if not isinstance(request_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", request_id):
                        raise ValueError("Invalid request_id")
                    if request_id in seen_ids:
                        raise ValueError("Duplicate request_id; replay is forbidden")
                    if request["lane"] not in {"resident_serial", "adaptive"}:
                        raise ValueError("Service accepts only native lanes")
                    paths = {}
                    for key in ("source", "config", "envelope", "output"):
                        if not isinstance(request[key], str) or not Path(request[key]).is_absolute():
                            raise ValueError(f"{key} must be an absolute path")
                        paths[key] = Path(request[key]).resolve()
                    if paths["source"] != source:
                        raise ValueError("Persistent worker cannot change source")
                    output = paths["output"]
                    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in [root] + outputs):
                        raise ValueError("Duplicate or overlapping output path")
                    if (source.is_relative_to(output) or any(output.is_relative_to(source / p)
                            for p in ("astrolabe", "config"))
                            or any(p.is_relative_to(output) for p in (paths["config"], paths["envelope"]))):
                        raise ValueError("Output overlaps source or request inputs")
                    if output.exists() and (not output.is_dir() or any(output.iterdir())):
                        raise ValueError("Evidence output must be absent or empty")
                    if not paths["config"].is_file() or not paths["envelope"].is_file():
                        raise ValueError("Config and envelope must be existing files")
                    if source_identity(source, source / "run.py") != source_snapshot:
                        raise ValueError("Source contents changed since service startup")
                    # Preserve the parent's certified profiles. Capture narrows
                    # the live shape list at each original scientific request.
                    envelope = json.loads(paths["envelope"].read_text())
                    if (not isinstance(envelope, dict) or
                            not isinstance(envelope.get("profiles_by_backend"), dict) or
                            not isinstance(envelope["profiles_by_backend"].get("esmfold"), dict) or
                            envelope["profiles_by_backend"]["esmfold"].get("qualification") is not True):
                        raise ValueError("Persistent worker requires an explicit ESMFold profile")
                except (ValueError, TypeError, OSError) as error:
                    final["rejected"] += 1
                    rejection = {"status": "rejected", "request": request, "pid": os.getpid(),
                                 "error": str(error), "exitcode": None}
                    write_json(root / "rejections" / f'{final["rejected"]:06d}.json', rejection)
                    if (isinstance(request, dict) and isinstance(request.get("request_id"), str)
                            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", request["request_id"])
                            and not (root / "results" / (request["request_id"] + ".json")).exists()):
                        write_json(root / "results" / (request["request_id"] + ".json"), rejection)
                        seen_ids.add(request["request_id"])
                    continue
                seen_ids.add(request_id)
                outputs.append(output)
                accepted = dict(request, status="accepted", pid=os.getpid(), accepted_at=time.time())
                write_json(root / "accepted" / (request_id + ".json"), accepted)
                final["accepted"] += 1
                os.environ.update(ASTROLABE_MODEL_EXECUTION="daemon", ASTROLABE_MODEL_OWNER="trajectory",
                                  ASTROLABE_REQUEST_CONCURRENCY="0" if request["lane"] == "resident_serial" else "1")
                workerpath = output / "worker.json"
                result = {"request_id": request_id, "pid": os.getpid(), "workerpath": str(workerpath)}
                try:
                    code = worker(SimpleNamespace(lane=request["lane"], request_id=request_id, **paths), persistent=True,
                                  allow_existing_source=True, init_bindings=bindings, runtime_shapes=runtime_shapes)
                    result.update(exitcode=code, status="completed" if code == 0 else "failed")
                    if code == 0:
                        evidence, receipt = load_worker(output)
                        if receipt.get("status") != "completed" or not all(validate_evidence(evidence, receipt).values()):
                            raise ValueError("Successful worker lacks valid completed evidence")
                        for row in evidence.get("model_acquisitions", []):
                            if (row.get("pool_attached") and not row.get("pool_reused")
                                    and acquisition_complete(row) and path_backed_loads(evidence)):
                                bindings.setdefault(row["model_id"], {
                                    "worker_path": str(workerpath), "sha256": sha256(workerpath),
                                    "model_id": row["model_id"], "owner_pid": row["owner_pid"],
                                    "identity": row["identity"]})
                        final["completed"] += 1
                except BaseException as error:
                    result.update(exitcode=1, status="failed", error={"type": type(error).__name__, "message": str(error)})
                write_json(root / "results" / (request_id + ".json"), result)
                if result["exitcode"] != 0:
                    final.update(status="failed", stop_reason="scientific_failure")
                    break
            else:
                final["stop_reason"] = "stdin_eof"
        if final["status"] != "failed":
            final.update(status="completed", exitcode=0)
    except BaseException as error:
        final.update(status="failed", error={"type": type(error).__name__, "message": str(error),
                                             "traceback": traceback.format_exc()})
    finally:
        if pool is not None:
            final["pool"] = pool.metadata
        final["init_bindings"] = bindings
        write_json(root / "final.json", final)
    return final["exitcode"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--worker", action="store_true")
    modes.add_argument("--serve-worker", action="store_true")
    modes.add_argument("--compare", action="store_true")
    parser.add_argument("--lane", choices=("direct", "resident_serial", "adaptive"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--envelope", type=Path)
    parser.add_argument("--roots", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    paths = [args.output] + ([args.config, args.source] if args.worker else
                            [args.source] if args.serve_worker else list(args.roots or []))
    if args.envelope:
        paths.append(args.envelope)
    if any(p is None or not p.is_absolute() for p in paths):
        parser.error("All config/source/output/envelope/root paths must be absolute")
    if args.worker:
        if args.lane is None:
            parser.error("--worker requires --lane, --config and --source")
        return worker(args)
    if args.serve_worker:
        return serve_worker(args)
    if not args.roots or len(args.roots) < 2:
        parser.error("--compare requires at least two --roots (first is the reference)")
    results = [compare_workers(args.roots[0], p) for p in args.roots[1:]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error("Comparison output must not already exist")
    write_json(args.output, {"equal": all(r["equal"] for r in results), "comparisons": results})
    return 0 if all(r["equal"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
