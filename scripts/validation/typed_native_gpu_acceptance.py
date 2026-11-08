#!/usr/bin/env python3
"""Typed AF3 + MCMC acceptance with receipt-safe crash recovery.

The CPU-testable harness owns orchestration and evidence collection.  The
default profile executes the registered AlphaFold3 evaluator and MCMC mutator
through the existing native adapters, real ``LossEngine``, ``Runtime``,
``RefinementTrial``, generic search policy and ``DesignRun`` front door.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields, is_dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE = REPO_ROOT / "config" / "container_af3_smoke.yaml"
SEED = 17
MAX_EVALUATIONS = 2
MODEL_DIR = "/models/alphafold3"
EVALUATOR_ALIAS = "AF3"
MUTATOR_ALIAS = "MCMC"
SHARED_SEQUENCE = "GHIK"
FIXED_A = "ACD"
FIXED_B = "MNP"
CRASH_POINTS = ("root", "proposal", "prediction", "prediction-root", "outcome")
SCHEMA = "astrolabe.typed-native-gpu-acceptance/v1"
NATIVE_REQUEST_BUDGET_SCHEMA = "astrolabe.typed-native-request-budget/v1"
DEFAULT_NATIVE_REQUEST_LIMIT = 4
NATIVE_REQUEST_LIMIT_ENV = "ASTROLABE_TYPED_NATIVE_REQUEST_LIMIT"
GLOBAL_NATIVE_REQUEST_BUDGET_ENV = "ASTROLABE_TYPED_GLOBAL_REQUEST_BUDGET"
GLOBAL_NATIVE_REQUEST_LIMIT_ENV = "ASTROLABE_TYPED_GLOBAL_REQUEST_LIMIT"


class AcceptanceError(RuntimeError):
    """The acceptance path cannot produce trustworthy evidence."""


class ControlledCrash(BaseException):
    """Deliberate non-Exception crash after a durable receipt."""

    def __init__(self, point: str) -> None:
        self.point = point
        super().__init__(f"controlled crash after {point}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_value(value: Any) -> Any:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise AcceptanceError("non-finite float in acceptance evidence")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise AcceptanceError("object dtype is unsupported in acceptance JSON")
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if is_dataclass(value):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in fields(value)
        }
    raise AcceptanceError(f"unsupported evidence type {type(value).__name__}")


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(_json_value(value), sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _positive_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise AcceptanceError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise AcceptanceError(f"{name} must be a positive integer")
    return value


def _budget_record(path: Path, *, limit: int) -> dict[str, int]:
    if not path.is_file():
        raise AcceptanceError(f"native request budget file is absent: {path}")
    record = _read_json(path)
    if (
        not isinstance(record, dict)
        or record.get("schema") != NATIVE_REQUEST_BUDGET_SCHEMA
        or type(record.get("limit")) is not int
        or record.get("limit") != limit
        or type(record.get("used")) is not int
        or record["used"] < 0
        or record["used"] > limit
    ):
        raise AcceptanceError(f"invalid native request budget record: {path}")
    return {"limit": limit, "used": record["used"]}


def _reserve_native_request(out_root: Path) -> dict[str, Any]:
    """Reserve a native callback slot before model execution can begin."""

    per_root_path = out_root / "native_request_budget.json"
    per_root_limit = _positive_env_int(
        NATIVE_REQUEST_LIMIT_ENV,
        DEFAULT_NATIVE_REQUEST_LIMIT,
    )
    global_path_text = os.environ.get(GLOBAL_NATIVE_REQUEST_BUDGET_ENV)
    global_path = (
        Path(global_path_text).expanduser().resolve()
        if global_path_text
        else None
    )
    global_limit = (
        _positive_env_int(GLOBAL_NATIVE_REQUEST_LIMIT_ENV, 20)
        if global_path is not None
        else None
    )

    budgets = [(per_root_path, per_root_limit)]
    if global_path is not None:
        budgets.append((global_path, global_limit))

    handles = []
    try:
        for path, limit in sorted(budgets, key=lambda item: str(item[0])):
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.with_suffix(path.suffix + ".lock").open("a+b")
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handles.append(handle)

        records = [
            _budget_record(path, limit=limit)
            for path, limit in budgets
        ]
        exhausted = [
            str(path)
            for (path, _limit), record in zip(budgets, records)
            if record["used"] >= record["limit"]
        ]
        if exhausted:
            raise AcceptanceError(
                "native request budget exhausted before model callback: "
                + ", ".join(exhausted)
            )

        for (path, limit), record in zip(budgets, records):
            _atomic_json(
                path,
                {
                    "schema": NATIVE_REQUEST_BUDGET_SCHEMA,
                    "limit": limit,
                    "used": record["used"] + 1,
                },
            )
        return {
            "per_root_used": records[0]["used"] + 1,
            "per_root_limit": per_root_limit,
            "global_used": (
                records[1]["used"] + 1
                if len(records) > 1
                else None
            ),
            "global_limit": global_limit,
        }
    finally:
        for handle in reversed(handles):
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()


def _process_identity(pid: Any) -> dict[str, Any] | None:
    try:
        numeric_pid = int(pid)
    except (TypeError, ValueError):
        return None
    if numeric_pid <= 0:
        return None
    stat_path = Path("/proc") / str(numeric_pid) / "stat"
    try:
        text = stat_path.read_text(encoding="ascii")
    except OSError:
        return None
    _, separator, suffix = text.rpartition(")")
    fields_after_comm = suffix.strip().split()
    if not separator or len(fields_after_comm) < 20:
        return None
    return {
        "pid": numeric_pid,
        "state": fields_after_comm[0],
        "parent_pid": int(fields_after_comm[1]),
        "start_time_ticks": int(fields_after_comm[19]),
    }


def acceptance_config(
    out_root: Path | str,
    *,
    model_dir: str | Path = MODEL_DIR,
) -> Any:
    """Compose the bounded, one-root/one-child 24 GB AF3 acceptance profile."""

    from omegaconf import open_dict

    from astrolabe.configuration import load_composed_config

    root = Path(out_root).expanduser().resolve()
    cfg = load_composed_config(DEFAULT_PROFILE)
    model_path = str(Path(model_dir).expanduser())
    with open_dict(cfg):
        cfg.task_name = "typed_native_gpu_acceptance"
        cfg.out_dir = str(root)
        cfg.raw_model_results_dir = str(root / "raw_model_results")
        cfg.seed = SEED
        cfg.repeats = 1
        cfg.steps = MAX_EVALUATIONS
        cfg.mutation_num = 1
        cfg.enable_pyrosetta = False
        cfg.skip_complex_poses = True
        cfg.use_metropolis_acceptance = False
        cfg.types = {"A": "p", "B": "p"}
        cfg.seqs = {
            "A": f"{SHARED_SEQUENCE}@shared +{FIXED_A}",
            "B": f"shared +{FIXED_B}",
        }
        cfg.homologies = {}

        cfg.evaluator.evaluator_terms = {
            "AF3": {
                "_type": "Alphafold3",
                "num_seeds": 1,
                "num_recycles": 1,
                "num_samples": 1,
                "run_data_pipeline": False,
                "run_inference": True,
                "flash_attention_implementation": "cudnn",
            }
        }
        cfg.evaluator.enabled_aliases = ["AF3"]
        cfg.evaluator.alphafold3.model_dir = model_path
        cfg.evaluator.alphafold3.num_seeds = 1
        cfg.evaluator.alphafold3.num_recycles = 1
        cfg.evaluator.alphafold3.num_samples = 1
        cfg.evaluator.alphafold3.run_data_pipeline = False
        cfg.evaluator.alphafold3.run_inference = True
        cfg.evaluator.alphafold3.flash_attention_implementation = "cudnn"
        cfg.evaluator.alphafold3.bucket_pad_tolerance = 0
        cfg.evaluator.alphafold3.save_all_samples = False
        cfg.evaluator.alphafold3.jax_compilation_cache_dir = str(root / "jax_cache")

        cfg.mutator.mutator_terms = {
            "MCMC": {
                "_type": "MCMC",
                "avoid_current_residue": True,
            }
        }
        cfg.mutator.enabled_aliases = ["MCMC"]
        cfg.mutator.MCMC.avoid_current_residue = True

        cfg.loss.loss_terms = {}
        cfg.stages.default.oligo = ["AB"]
        cfg.stages.default.evaluator = "AF3"
        cfg.stages.default.mutators = ["MCMC"]
        cfg.stages.default.mutation_num = 1
        cfg.stages.default.weighted_score_expression = "plddt"
        cfg.stages.default.acceptance_threshold = None
        cfg.stages.default.early_stopping = None
        cfg.stages.default.acceptance_gate = None
        cfg.stages.default.prefix_early_stopping = None
        cfg.weight_path.alphafold3 = model_path
    return cfg


def _plain_config(cfg: Any) -> dict[str, Any]:
    from astrolabe.design.configured_refinement import _resolved_active_config

    plain = _resolved_active_config(cfg, apply_run_lifecycle=False)
    plain.pop("defaults", None)
    return _json_value(plain)


def _config_from_plain(value: Mapping[str, Any]) -> Any:
    from omegaconf import OmegaConf

    return OmegaConf.create(dict(value))


def _initialize_root(out_root: Path, cfg: Any) -> dict[str, Any]:
    contract_path = out_root / "contract.json"
    config_path = out_root / "config.json"
    plain = _plain_config(cfg)
    encoded = json.dumps(plain, sort_keys=True, separators=(",", ":")).encode("utf-8")
    config_digest = hashlib.sha256(encoded).hexdigest()
    expected = {
        "schema": SCHEMA,
        "script_sha256": _sha256(Path(__file__).resolve()),
        "config_sha256": config_digest,
        "seed": SEED,
        "max_evaluations": MAX_EVALUATIONS,
        "assemblies": ["A", "B"],
        "fixture": {
            "shared_sequence": SHARED_SEQUENCE,
            "fixed_A": FIXED_A,
            "fixed_B": FIXED_B,
        },
        "af3": {
            "num_seeds": 1,
            "num_recycles": 1,
            "num_samples": 1,
            "empty_msa": True,
            "pyrosetta": False,
            "gpu_profile": "24GB",
        },
    }
    if contract_path.exists():
        current = _read_json(contract_path)
        if current != expected:
            raise AcceptanceError(
                "acceptance root contract changed; use a fresh --out-root"
            )
        saved = _read_json(config_path)
        if saved != plain:
            raise AcceptanceError("stored acceptance configuration changed")
        _budget_record(
            out_root / "native_request_budget.json",
            limit=_positive_env_int(
                NATIVE_REQUEST_LIMIT_ENV,
                DEFAULT_NATIVE_REQUEST_LIMIT,
            ),
        )
        return current

    if out_root.exists() and any(out_root.iterdir()):
        raise AcceptanceError(
            f"--out-root must be absent or empty for a new run: {out_root}"
        )
    out_root.mkdir(parents=True, exist_ok=True)
    try:
        out_root.chmod(0o700)
    except OSError:
        pass
    _atomic_json(config_path, plain)
    _atomic_json(contract_path, expected)
    request_limit = _positive_env_int(
        NATIVE_REQUEST_LIMIT_ENV,
        DEFAULT_NATIVE_REQUEST_LIMIT,
    )
    _atomic_json(
        out_root / "native_request_budget.json",
        {
            "schema": NATIVE_REQUEST_BUDGET_SCHEMA,
            "limit": request_limit,
            "used": 0,
        },
    )
    return expected


def load_acceptance_config(out_root: Path | str) -> Any:
    root = Path(out_root).expanduser().resolve()
    path = root / "config.json"
    if not path.is_file():
        raise AcceptanceError(f"acceptance configuration is absent: {path}")
    return _config_from_plain(_read_json(path))


def _parse_arguments(cfg: Any, out_root: Path) -> Any:
    from astrolabe.argparser import parse_arguments

    root = Path(out_root)
    with tempfile.TemporaryDirectory(
        prefix=".typed-native-parse-",
        dir=root,
    ) as scratch:
        cfg.out_dir = scratch
        args = parse_arguments(cfg)
    args.out_dir = root / "artifacts"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.raw_model_results_dir = root / "raw_model_results"
    args.raw_model_results_dir.mkdir(parents=True, exist_ok=True)
    args._raw_model_results_dir_auto_managed = False
    args._raw_model_results_dir_marker = None
    return args


def _initialize_legacy(args: Any) -> Any:
    from astrolabe.argparser import set_random_seeds
    from astrolabe.core import global_config

    import run as run_module

    run_module.initialize_global_config(args)
    set_random_seeds(SEED)
    run_module.setup_protomers(args)
    run_module.apply_cyclization(args)
    global_config.set_variable("step", 0)
    global_config.set_variable(
        "mutation_num",
        int(getattr(args.stages.default, "mutation_num", 1)),
    )
    global_config.init_const("num_tokens", {})
    return run_module


class LossAssessmentAdapter:
    """Typed Assessment around the existing LossEngine computation."""

    def __init__(self, bindings: Any, loss_engine: Any, loss_identity: Any) -> None:
        self.bindings = bindings
        self.loss_engine = loss_engine
        self.loss_identity = loss_identity

    def __call__(self, request: Any) -> Any:
        from astrolabe.design.native_support import numeric_step
        from astrolabe.design.types import (
            Assessment,
            Dependency,
            manifest,
        )

        if not request.evidence.artifacts:
            raise AcceptanceError("typed assessment requires native score evidence")
        samples = {
            artifact.sample
            for artifact in request.evidence.artifacts
            if artifact.sample is not None
        }
        if len(samples) != 1:
            raise AcceptanceError(
                "typed assessment requires one sample across native artifacts"
            )
        sample = next(iter(samples))
        step = numeric_step(sample)
        scores = self.bindings.scores_for(
            request.evidence,
            destination=Path(self.bindings.work_root) / "assessment" / str(step),
        )
        weighted_score, _terms = self.loss_engine.compute_loss(
            scores,
            {"step": step},
        )
        value = float(weighted_score)
        if not math.isfinite(value):
            raise AcceptanceError(f"LossEngine returned non-finite score {value!r}")
        inputs = manifest(
            (
                Dependency("evidence", request.evidence.identity),
                Dependency("history", request.history.revision),
                Dependency("loss", self.loss_identity),
            )
        )
        return Assessment(
            evidence=request.evidence.identity,
            objective_revision=request.objective_revision,
            history_revision=request.history.revision,
            values=(value,),
            gates=(),
            inputs=inputs,
        )


@dataclass
class AcceptanceComposition:
    args: Any
    data: Any
    translation: Any
    bindings: Any
    evaluator: Any
    mutator: Any
    assess: Any
    loss: Any
    trial: Any
    policy: Any


def _fixed_proposal_seed(_index: int) -> int:
    return SEED


def _configure_af3_step(
    _evaluator: Any,
    request: Any,
) -> dict[str, Any]:
    from astrolabe.core import global_config
    from astrolabe.design.native_support import numeric_step

    step = numeric_step(request.samples[0])
    global_config.set_variable("step", step)
    return {"step": step}


def _configure_native_mutator(
    mutator: Any,
    runtime_config: dict[str, Any],
    *,
    request: Any,
    sample: Any,
    step: int,
    work_root: Path,
) -> dict[str, Any]:
    from astrolabe.core import global_config

    del request, work_root
    if sample.seed != SEED:
        raise AcceptanceError(
            f"native proposal seed must remain {SEED}, got {sample.seed}"
        )
    mutator.mutation_weight = 1.0
    global_config.set_variable("step", step)
    updated = dict(runtime_config)
    updated["step"] = step
    updated["job_name"] = sample.replicate
    updated["proposal_seed"] = sample.seed
    return updated


def build_acceptance_composition(
    out_root: Path | str,
    cfg: Any | None = None,
) -> AcceptanceComposition:
    """Build the real typed composition without invoking a model."""

    from astrolabe.core import global_config
    from astrolabe.design.legacy import LegacyContext
    from astrolabe.design.native_registry import (
        NativeComponentContext,
        NativeOperationHooks,
    )
    from astrolabe.design.native_support import NativeBindings
    from astrolabe.design.native_trial import RefinementTrial
    from astrolabe.design.operations import operation
    from astrolabe.design.phased_search import PhasedBeamSearchPolicy
    from astrolabe.design.store import DataStore, JSONCodec
    from astrolabe.design.types import (
        AssessRequest,
        Assessment,
        Sample,
        SamplingPolicy,
        key_for,
    )
    from astrolabe.evaluators.registry import EvaluatorRegistry
    from astrolabe.losses.base import LossEngine
    from astrolabe.mutators.registry import MutatorRegistry

    root = Path(out_root).expanduser().resolve()
    global_config.reset()
    cfg = acceptance_config(root) if cfg is None else cfg
    args = _parse_arguments(cfg, root)
    _initialize_legacy(args)

    evaluator_term = args.evaluator.evaluator_terms[EVALUATOR_ALIAS]
    mutator_term = args.mutator.mutator_terms[MUTATOR_ALIAS]
    evaluator_type = str(evaluator_term.get("_type", ""))
    mutator_type = str(mutator_term.get("_type", ""))
    if not evaluator_type or not mutator_type:
        raise AcceptanceError("selected native component type is missing")

    data = DataStore(root / "data")
    context = LegacyContext()
    translation = context.capture_candidate(assemblies=("A", "B"), data=data)
    bindings = NativeBindings(
        context=context,
        translation=translation,
        data=data,
        work_root=root / "native-work",
    )
    sampling_options = {
        "seed": SEED,
        "mutation_num": 1,
        "avoid_current_residue": True,
    }
    sampling = SamplingPolicy(
        key_for(("typed-native-gpu-acceptance/sampling/v1", sampling_options)),
        data.put(sampling_options, JSONCodec()),
    )
    native_context = NativeComponentContext(
        bindings=bindings,
        validated_evaluator_conf=args.evaluator,
        validated_mutator_conf=args.mutator,
        raw_model_results_dir=Path(args.raw_model_results_dir),
        sampling_policy=sampling,
        options_schemas={
            MUTATOR_ALIAS.casefold(): {
                "type": "object",
                "properties": {
                    "protomer_scope": {
                        "type": "string",
                        "enum": ["mutable", "all"],
                    },
                    "mutation_weight": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                    },
                },
                "required": ["protomer_scope", "mutation_weight"],
                "additionalProperties": False,
            }
        },
        operation_hooks={
            EVALUATOR_ALIAS: NativeOperationHooks(
                configure=_configure_af3_step,
                identity="typed-native-gpu-acceptance/af3-step/v1",
            ),
            MUTATOR_ALIAS: NativeOperationHooks(
                configure=_configure_native_mutator,
                identity="typed-native-gpu-acceptance/mcmc/v1",
            ),
        },
    )
    evaluator = EvaluatorRegistry.build_operation(
        evaluator_type,
        alias=EVALUATOR_ALIAS,
        options={},
        context=native_context,
    )
    mutator = MutatorRegistry.build_operation(
        mutator_type,
        alias=MUTATOR_ALIAS,
        options={
            "protomer_scope": "mutable",
            "mutation_weight": 1.0,
        },
        context=native_context,
    )

    stage = args.stages.default
    loss = LossEngine(
        args.loss,
        runtime_context=global_config.build_runtime_context(),
    )
    loss.setup(
        stage.weighted_score_expression,
        stage.get("early_stopping"),
        stage.get("acceptance_threshold"),
        stage.get("acceptance_gate"),
        prefix_early_stopping=stage.get("prefix_early_stopping"),
    )
    loss_identity = key_for(
        (
            "typed-native-gpu-acceptance/loss/v1",
            _plain_config(args.loss),
            stage.weighted_score_expression,
            loss.__class__.__module__,
            loss.__class__.__qualname__,
        )
    )
    assessor = LossAssessmentAdapter(bindings, loss, loss_identity)
    assess = operation(
        "assess",
        AssessRequest,
        Assessment,
        assessor,
        version="typed-native-gpu-acceptance/assessment/v1",
        model_identity=loss_identity,
        options={"metric": "plddt"},
        schema={
            "type": "object",
            "properties": {"metric": {"const": "plddt"}},
            "required": ["metric"],
            "additionalProperties": False,
        },
    )
    root_sample = Sample(
        protocol=key_for(("typed-native-gpu-acceptance/root/v1", SEED)),
        replicate="root",
        seed=SEED,
        index="0",
    )
    trial = RefinementTrial(
        translation.system,
        predict=evaluator,
        propose=mutator,
        assess=assess,
        sampling=sampling,
        assemblies=("A", "B"),
        root_sample=root_sample,
        sources=(),
        initialization_order=(mutator, evaluator),
    )
    policy = PhasedBeamSearchPolicy(
        width=1,
        proposals_per_parent=1,
        max_evaluations=MAX_EVALUATIONS,
        proposal_cap=1,
        seed=SEED,
        evaluation_seed=SEED,
        proposal_seed=_fixed_proposal_seed,
        seed_identity="typed-native-gpu-acceptance/fixed-proposal-seed/v1",
        direction="max",
        expansion="each",
    )
    return AcceptanceComposition(
        args=args,
        data=data,
        translation=translation,
        bindings=bindings,
        evaluator=evaluator,
        mutator=mutator,
        assess=assess,
        loss=loss,
        trial=trial,
        policy=policy,
    )


def _loaded_evaluator(evaluator: Any) -> Any:
    owner = getattr(evaluator, "owner", None)
    loaded = getattr(owner, "_value", None)
    get = getattr(owner, "get", None)
    if loaded is None and callable(get):
        return get()
    return loaded


def _native_owner_status(evaluator: Any) -> dict[str, Any] | None:
    loaded = _loaded_evaluator(evaluator)
    runner = getattr(loaded, "model_runner", None)
    client = getattr(runner, "_client", None)
    owner = getattr(client, "owner", None)
    status = getattr(owner, "status", None)
    if not callable(status):
        return None
    payload = status()
    if not isinstance(payload, Mapping):
        raise AcceptanceError("NativeModelClient owner status must be a mapping")
    pid = payload.get("pid")
    return {
        "pid": pid,
        "process": _process_identity(pid),
        "backend": payload.get("backend"),
        "state": payload.get("state"),
        "request_count": int(payload.get("request_count", 0)),
        "completed_count": int(payload.get("completed_count", 0)),
        "failed_count": int(payload.get("failed_count", 0)),
    }


def _install_model_runner_observer(
    evaluator: Any,
    path: Path,
    *,
    out_root: Path,
) -> None:
    loaded = _loaded_evaluator(evaluator)
    runner = getattr(loaded, "model_runner", None)
    observe = getattr(runner, "run_inference", None)
    if not callable(observe):
        return
    if getattr(runner, "_typed_acceptance_observed", False):
        return

    def observed_run_inference(*args, **kwargs):
        reservation = _reserve_native_request(out_root)
        _append_jsonl(
            path,
            {
                "event": "native_inference_request",
                "pid": os.getpid(),
                "time_ns": time.time_ns(),
                "runner": type(runner).__module__ + "." + type(runner).__qualname__,
                "budget": reservation,
            },
        )
        try:
            result = observe(*args, **kwargs)
        except BaseException as error:
            _append_jsonl(
                path,
                {
                    "event": "native_inference_request_failed",
                    "pid": os.getpid(),
                    "time_ns": time.time_ns(),
                    "error_type": type(error).__name__,
                },
            )
            raise
        return result

    runner.run_inference = observed_run_inference
    runner._typed_acceptance_observed = True


class PredictionInstrumentation:
    def __init__(
        self,
        path: Path,
        native_path: Path,
        alias: str,
        evaluator: Any,
    ) -> None:
        self.path = path
        self.native_path = native_path
        self.alias = alias
        self.evaluator = evaluator

    def wrap(self, invoke):
        def observed(prepared):
            request = prepared.request
            sample = request.samples[0]
            _install_model_runner_observer(
                self.evaluator,
                self.native_path,
                out_root=self.native_path.parent,
            )
            before = _native_owner_status(self.evaluator)
            _append_jsonl(
                self.path,
                {
                    "event": "prediction_invocation_started",
                    "alias": self.alias,
                    "pid": os.getpid(),
                    "time_ns": time.time_ns(),
                    "sample": {
                        "replicate": sample.replicate,
                        "seed": sample.seed,
                        "index": sample.index,
                    },
                    "candidate": request.candidate.identity.digest,
                    "folding_inputs": [
                        dependency.identity.digest
                        for dependency in prepared.inputs.dependencies
                        if dependency.role == "folding-input"
                    ],
                    "native_status_before": before,
                },
            )
            try:
                result = invoke(prepared)
            except BaseException as error:
                _append_jsonl(
                    self.path,
                    {
                        "event": "prediction_invocation_failed",
                        "pid": os.getpid(),
                        "time_ns": time.time_ns(),
                        "error_type": type(error).__name__,
                    },
                )
                raise
            after = _native_owner_status(self.evaluator)
            native_delta = None
            if before is not None and after is not None:
                native_delta = {
                    "requests": after["request_count"] - before["request_count"],
                    "completed": (
                        after["completed_count"] - before["completed_count"]
                    ),
                    "failed": after["failed_count"] - before["failed_count"],
                }
            _append_jsonl(
                self.path,
                {
                    "event": "prediction_invocation_completed",
                    "pid": os.getpid(),
                    "time_ns": time.time_ns(),
                    "candidate": request.candidate.identity.digest,
                    "native_status_after": after,
                    "native_status_delta": native_delta,
                },
            )
            return result

        return observed


def _install_crash_injection(
    runtime: Any,
    point: str,
    marker: Path,
    *,
    hard_exit: bool,
) -> None:
    from astrolabe.design.store import design_codec
    from astrolabe.design.types import Blob, Key

    original_write = runtime._write
    original_completion = runtime._save_completion
    fired = False

    def fire() -> None:
        nonlocal fired
        if fired:
            return
        fired = True
        _atomic_json(
            marker,
            {
                "schema": SCHEMA,
                "point": point,
                "pid": os.getpid(),
                "time_ns": time.time_ns(),
                "durable_before_exit": True,
                "exit_mode": (
                    "os_exit_75" if hard_exit else "controlled_exception"
                ),
            },
        )
        if hard_exit:
            os._exit(75)
        raise ControlledCrash(point)

    def write(kind, name, value):
        original_write(kind, name, value)
        if (
            point in ("prediction", "prediction-root")
            and kind == "attempt"
            and value.get("status") == "completed"
            and value.get("component") == EVALUATOR_ALIAS
        ):
            record = value.get("key")
            if not record:
                return
            key = runtime.data.get(
                Blob(Key(record["identity"]), record["codec"]),
                design_codec,
            )
            work_id = key.work_id
            if point == "prediction-root" and work_id == "root":
                fire()
            if point == "prediction" and work_id != "root":
                work = runtime.work_entry(work_id)
                if work.stage == "evaluate":
                    fire()

    def save_completion(name, value, codec=design_codec):
        original_completion(name, value, codec)
        if point == "root" and name == "run/root":
            fire()
        if point in ("proposal", "outcome") and name.startswith("outcome/"):
            work_id = name.split("/", 1)[1]
            work = runtime.work_entry(work_id)
            if point == "proposal" and work.stage == "propose":
                fire()
            if point == "outcome" and work.stage == "evaluate":
                fire()

    runtime._write = write
    runtime._save_completion = save_completion


def _candidate_payload(candidate: Any, translation: Any) -> dict[str, Any]:
    from astrolabe.design.types import Chemical, Polymer

    values = {}
    for assignment in candidate.assignments:
        value = assignment.value
        if isinstance(value, Polymer):
            values[assignment.variable] = {
                "kind": value.kind,
                "sequence": "".join(value.tokens),
                "tokens": list(value.tokens),
                "modifications": list(value.modifications),
            }
        elif isinstance(value, Chemical):
            values[assignment.variable] = {
                "kind": "chemical",
                "graph": value.graph.identity.digest,
            }
        else:
            raise AcceptanceError(
                f"unsupported assignment type {type(value).__name__}"
            )
    return {
        "identity": candidate.identity.digest,
        "system": candidate.system.digest,
        "assignments": values,
        "assemblies": [item.identity for item in translation.assemblies],
    }


def _collect_arrays(value: Any, prefix: str = "scores") -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    if isinstance(value, np.ndarray):
        result[prefix] = np.array(value, copy=True)
        return result
    if isinstance(value, np.generic):
        result[prefix] = np.asarray(value)
        return result
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = str(key).replace("/", "_").replace(" ", "_")
            result.update(_collect_arrays(item, f"{prefix}.{name}"))
        return result
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            result.update(_collect_arrays(item, f"{prefix}.{index}"))
        return result
    if is_dataclass(value):
        for field in fields(value):
            result.update(
                _collect_arrays(
                    getattr(value, field.name),
                    f"{prefix}.{field.name}",
                )
            )
    return result


def _save_arrays(scores: Any, path: Path) -> str:
    arrays = _collect_arrays(scores)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _blob_record(value: Mapping[str, Any]) -> Any:
    from astrolabe.design.types import Blob, Key

    return Blob(Key(value["identity"]), value["codec"])


def _attempt_payload(runtime: Any, attempt: Any) -> dict[str, Any]:
    from astrolabe.design.native_prediction import FOLDING_INPUT_CODEC
    from astrolabe.design.store import design_codec

    row = runtime._read("attempt", attempt.identity)
    inputs = runtime.data.get(_blob_record(row["inputs"]), design_codec)
    references = [
        _blob_record(reference)
        for reference in row.get("references", ())
        if reference is not None
    ]
    folding = []
    for reference in references:
        if reference.codec != FOLDING_INPUT_CODEC.identity:
            continue
        value = runtime.data.get(reference, FOLDING_INPUT_CODEC)
        for assembly, item in value.items():
            folding.append(
                {
                    "blob": reference.identity.digest,
                    "assembly": assembly,
                    "seed": list(item.rng_seeds),
                    "chains": [
                        {
                            "id": chain.id,
                            "sequence": chain.sequence,
                            "unpaired_msa": chain.unpaired_msa,
                            "paired_msa": chain.paired_msa,
                        }
                        for chain in item.chains
                    ],
                }
            )
    return {
        "identity": attempt.identity,
        "work": attempt.key.work_id,
        "round": attempt.key.named_round,
        "status": attempt.status,
        "manifest_roles": [item.role for item in inputs.dependencies],
        "inputs": inputs.identity.digest,
        "references": [
            {
                "codec": reference.codec,
                "identity": reference.identity.digest,
            }
            for reference in references
        ],
        "folding": folding,
    }


def _score_payload(
    runtime: Any,
    bindings: Any,
    scored: Any,
    label: str,
    array_path: Path,
) -> dict[str, Any]:
    scores = bindings.scores_for(
        scored.evidence,
        destination=array_path.parent / f"{label}-restored",
    )
    digest = _save_arrays(scores, array_path)
    structures = []
    for assembly, path in sorted(scores.complex_paths.items()):
        path = Path(path)
        if not path.is_file():
            raise AcceptanceError(
                f"AF3 structure artifact is unavailable: {assembly}={path}"
            )
        structures.append(
            {
                "assembly": assembly,
                "path": str(path),
                "sha256": _sha256(path),
            }
        )
    return {
        "label": label,
        "candidate": _candidate_payload(scored.candidate, bindings.translation),
        "evidence": scored.evidence.identity.digest,
        "assessment": {
            "values": list(scored.assessment.values),
            "gates": dict(scored.assessment.gates),
            "inputs": scored.assessment.inputs.identity.digest,
        },
        "arrays": {
            "path": str(array_path.relative_to(runtime.root.parent)),
            "sha256": digest,
        },
        "structures": structures,
    }


def _result_payload(
    runtime: Any,
    result: Any,
    composition: AcceptanceComposition,
    out_root: Path,
) -> dict[str, Any]:
    from astrolabe.design.store import design_codec

    root_score = runtime.peek_completion("run/root")
    if root_score is None:
        raise AcceptanceError("run/root completion is absent")
    evaluations = [root_score]
    evaluations.extend(
        child
        for outcome in runtime.history.records
        for child in outcome.children
    )
    if len(evaluations) != MAX_EVALUATIONS:
        raise AcceptanceError(
            f"expected {MAX_EVALUATIONS} evaluations, got {len(evaluations)}"
        )

    attempts = sorted(
        runtime.attempts(),
        key=lambda item: (item.key.work_id, item.key.named_round, item.identity),
    )
    prediction_attempts = [
        item for item in attempts if item.key.named_round.endswith("/predict")
    ]
    mutation_attempts = [
        item for item in attempts if item.key.named_round == "propose"
    ]
    if len(prediction_attempts) != MAX_EVALUATIONS:
        raise AcceptanceError(
            f"expected {MAX_EVALUATIONS} prediction attempts, "
            f"got {len(prediction_attempts)}"
        )
    if any(item.status != "completed" for item in prediction_attempts):
        raise AcceptanceError("prediction receipt is not completed")

    invocation_records = _read_jsonl(out_root / "prediction_invocations.jsonl")
    prediction_invocations = [
        item
        for item in invocation_records
        if item.get("event") == "prediction_invocation_started"
    ]
    completed_invocations = [
        item
        for item in invocation_records
        if item.get("event") == "prediction_invocation_completed"
    ]
    if (
        len(prediction_invocations) != len(prediction_attempts)
        or len(completed_invocations) != len(prediction_attempts)
    ):
        raise AcceptanceError(
            "prediction invocation count differs from completed prediction receipts"
        )

    native_request_records = [
        item
        for item in _read_jsonl(out_root / "native_inference_requests.jsonl")
        if item.get("event") == "native_inference_request"
    ]
    status_deltas = [
        item["native_status_delta"]
        for item in completed_invocations
        if item.get("native_status_delta") is not None
    ]
    assembly_count = len(composition.translation.assemblies)
    evaluator_term = composition.args.evaluator.evaluator_terms[EVALUATOR_ALIAS]
    num_seeds = int(evaluator_term.get("num_seeds", 1))
    num_samples = int(evaluator_term.get("num_samples", 1))
    expected_gpu_requests = (
        MAX_EVALUATIONS * assembly_count * num_seeds
    )
    callback_count = len(native_request_records)
    status_request_count = (
        sum(item["requests"] for item in status_deltas)
        if len(status_deltas) == len(prediction_invocations)
        else None
    )
    status_completed_count = (
        sum(item["completed"] for item in status_deltas)
        if len(status_deltas) == len(prediction_invocations)
        else None
    )
    if status_request_count is None or status_completed_count is None:
        raise AcceptanceError(
            "NativeModelClient owner status deltas are required for every "
            "prediction invocation"
        )
    observed_gpu_requests = status_request_count
    if callback_count != expected_gpu_requests:
        raise AcceptanceError(
            f"expected {expected_gpu_requests} run_inference callbacks, "
            f"observed {callback_count}"
        )
    if observed_gpu_requests != expected_gpu_requests:
        raise AcceptanceError(
            f"expected {expected_gpu_requests} native GPU requests, "
            f"observed {observed_gpu_requests}"
        )
    if status_request_count != status_completed_count:
        raise AcceptanceError(
            "NativeModelClient request/completed deltas disagree"
        )

    checkpoint = runtime.load("run/state")
    if checkpoint is None:
        raise AcceptanceError("run/state checkpoint is absent")
    policy_state = runtime.data.get(checkpoint["state"], composition.policy.codec)
    selected = []
    for scored in result.selected:
        selected.append(
            {
                "candidate": scored.candidate.identity.digest,
                "evidence": scored.evidence.identity.digest,
                "values": list(scored.assessment.values),
            }
        )

    records = [
        _score_payload(
            runtime,
            composition.bindings,
            scored,
            "root" if index == 0 else f"evaluation-{index}",
            out_root / "arrays" / ("root.npz" if index == 0 else f"evaluation-{index}.npz"),
        )
        for index, scored in enumerate(evaluations)
    ]
    structure_samples = [
        structure
        for record in records
        for structure in record["structures"]
    ]
    expected_structure_samples = expected_gpu_requests * num_samples
    if len(structure_samples) != expected_structure_samples:
        raise AcceptanceError(
            f"expected {expected_structure_samples} AF3 structure samples, "
            f"got {len(structure_samples)}"
        )
    occurrence_payload = [
        {
            "assembly": item.assembly,
            "chain": item.chain,
            "component": item.component,
            "variable": item.variable,
        }
        for item in composition.translation.system.occurrences
    ]
    domains = [
        {
            "variable": item.variable,
            "fixed_tokens": list(item.fixed_tokens),
            "fixed_atoms": list(item.fixed_atoms),
        }
        for item in composition.translation.system.domains
    ]
    plan_acceptance = {
        "root_evaluations": policy_state.root_evaluations,
        "evaluations": policy_state.evaluations,
        "proposals": policy_state.proposals,
        "duplicates": policy_state.duplicates,
        "generation": policy_state.generation,
        "finished": policy_state.finished,
    }
    if plan_acceptance != {
        "root_evaluations": 1,
        "evaluations": 2,
        "proposals": 1,
        "duplicates": 0,
        "generation": 1,
        "finished": True,
    }:
        raise AcceptanceError(f"unexpected policy budget state: {plan_acceptance}")

    return {
        "schema": SCHEMA,
        "profile": {
            "name": "af3-mcmc-24gb",
            "seed": SEED,
            "num_samples": num_samples,
            "num_recycles": 1,
            "num_seeds": num_seeds,
            "empty_msa": True,
            "enable_pyrosetta": False,
            "skip_complex_poses": True,
            "max_evaluations": MAX_EVALUATIONS,
        },
        "route": {
            "frontdoor": "Astrolabe(DesignRun)",
            "trial": type(composition.trial).__name__,
            "policy": type(composition.policy).__name__,
            "runtime": type(runtime).__name__,
            "loss": type(composition.loss).__name__,
        },
        "operations": {
            "evaluator": {
                "alias": composition.evaluator.descriptor.alias,
                "identity": composition.evaluator.descriptor.identity.digest,
            },
            "mutator": {
                "alias": composition.mutator.descriptor.alias,
                "identity": composition.mutator.descriptor.identity.digest,
            },
            "assessment": {
                "alias": composition.assess.descriptor.alias,
                "identity": composition.assess.descriptor.identity.digest,
            },
        },
        "fixture": {
            "assemblies": [item.identity for item in composition.translation.assemblies],
            "occurrences": occurrence_payload,
            "domains": domains,
        },
        "evaluations": records,
        "selected": selected,
        "policy": plan_acceptance,
        "attempts": [
            _attempt_payload(runtime, item)
            for item in (*mutation_attempts, *prediction_attempts)
        ],
        "prediction_invocations": {
            "count": len(prediction_invocations),
            "processes": sorted({item["pid"] for item in prediction_invocations}),
            "records_sha256": hashlib.sha256(
                (out_root / "prediction_invocations.jsonl").read_bytes()
            ).hexdigest(),
        },
        "observed_gpu_requests": {
            "expected": expected_gpu_requests,
            "observed": observed_gpu_requests,
            "source": "NativeModelClient owner request/completed deltas",
            "native_status_request_delta": status_request_count,
            "native_status_completed_delta": status_completed_count,
            "native_status_entries": len(status_deltas),
            "run_inference_callbacks": callback_count,
            "callback_record_sha256": hashlib.sha256(
                (out_root / "native_inference_requests.jsonl").read_bytes()
            ).hexdigest(),
        },
        "structure_samples": {
            "expected": expected_structure_samples,
            "count": len(structure_samples),
            "items": structure_samples,
        },
        "rng": {
            "final": checkpoint["rng"].identity.digest,
            "receipts": [
                {
                    "attempt": item.identity,
                    "round": item.key.named_round,
                    "before": item.rng.before.identity.digest,
                    "after": item.rng.after.identity.digest,
                }
                for item in prediction_attempts
            ],
        },
        "journal": {
            "path": str(runtime.root / "journal.sqlite"),
            "selected_sha256": _sha256(runtime.root / "selected.json"),
        },
    }


def run_typed_acceptance(
    out_root: Path | str,
    *,
    config: Any | None = None,
    crash_after: str | None = None,
    hard_exit: bool = False,
) -> dict[str, Any]:
    """Run or resume one typed acceptance root."""

    from astrolabe.core import global_config
    from astrolabe.app import Astrolabe
    from astrolabe.design.composition import DesignRun
    from astrolabe.design.runtime import Runtime
    from astrolabe.design.types import key_for

    root = Path(out_root).expanduser().resolve()
    cfg = acceptance_config(root) if config is None else config
    contract = _initialize_root(root, cfg)
    execution_cfg = load_acceptance_config(root)
    if crash_after is not None and crash_after not in CRASH_POINTS:
        raise AcceptanceError(f"unknown crash point {crash_after!r}")

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    composition = None
    runtime = None
    succeeded = False
    try:
        composition = build_acceptance_composition(root, execution_cfg)
        composition.evaluator.invoke = PredictionInstrumentation(
            root / "prediction_invocations.jsonl",
            root / "native_inference_requests.jsonl",
            composition.evaluator.descriptor.alias,
            composition.evaluator,
        ).wrap(composition.evaluator.invoke)

        operations = tuple(composition.trial.parts)
        runtime = Runtime(
            operations,
            root=root / "runtime",
            data=composition.data,
            budgets={"prediction": MAX_EVALUATIONS, "mutate": 1},
            scientific_contract=key_for(
                (
                    "typed-native-gpu-acceptance/runtime/v1",
                    contract["config_sha256"],
                    composition.trial.identity,
                    composition.policy.identity,
                )
            ),
        )
        if crash_after is not None:
            _install_crash_injection(
                runtime,
                crash_after,
                root / "controlled_crash.json",
                hard_exit=hard_exit,
            )

        result = Astrolabe(
            DesignRun(
                composition.translation.system,
                composition.translation.candidate,
                composition.trial,
                composition.policy,
                runtime,
                capacity=1,
            )
        ).run()
        payload = _result_payload(runtime, result, composition, root)
        _atomic_json(root / "acceptance_result.json", payload)
        succeeded = True
        return payload
    finally:
        if runtime is not None:
            runtime.close()
        if composition is not None:
            for component in (composition.mutator, composition.evaluator):
                try:
                    component.shutdown()
                except Exception:
                    if succeeded:
                        raise
        global_config.reset()


def prepare_typed_acceptance(
    out_root: Path | str,
    *,
    config: Any | None = None,
    model_dir: str | Path = MODEL_DIR,
) -> dict[str, Any]:
    """Build the registered operations and durable contract without loading."""

    root = Path(out_root).expanduser().resolve()
    cfg = (
        acceptance_config(root, model_dir=model_dir)
        if config is None
        else config
    )
    contract = _initialize_root(root, cfg)
    execution_cfg = load_acceptance_config(root)
    composition = build_acceptance_composition(root, execution_cfg)
    try:
        payload = {
            "status": "prepared",
            "out_root": str(root),
            "config_sha256": contract["config_sha256"],
            "operations": {
                item.descriptor.alias: item.descriptor.identity.digest
                for item in composition.trial.parts
            },
            "trial_identity": composition.trial.identity.digest,
            "policy_identity": composition.policy.identity.digest,
            "model_owner_loaded": {
                "evaluator": getattr(
                    composition.evaluator.owner,
                    "_value",
                    None,
                )
                is not None,
                "mutator": getattr(composition.mutator.owner, "_value", None)
                is not None,
            },
            "prediction_invocations_allowed": MAX_EVALUATIONS,
            "assemblies": ["A", "B"],
            "num_seeds": 1,
            "num_samples": 1,
            "expected_native_requests_per_complete_trajectory": 4,
        }
        _atomic_json(root / "prepare.json", payload)
        return payload
    finally:
        for component in (composition.mutator, composition.evaluator):
            component.shutdown()
        from astrolabe.core import global_config

        global_config.reset()


def _array_evidence(result: Mapping[str, Any], root: Path) -> dict[str, np.ndarray]:
    output = {}
    for item in result["evaluations"]:
        relative = item["arrays"]["path"]
        path = root / relative
        with np.load(path, allow_pickle=False) as archive:
            output[item["label"]] = {
                key: np.array(archive[key], copy=True)
                for key in archive.files
            }
    return output


def compare_acceptance(
    uninterrupted_root: Path | str,
    crashed_root: Path | str,
) -> dict[str, Any]:
    left_root = Path(uninterrupted_root).expanduser().resolve()
    right_root = Path(crashed_root).expanduser().resolve()
    left = _read_json(left_root / "acceptance_result.json")
    right = _read_json(right_root / "acceptance_result.json")
    left_arrays = _array_evidence(left, left_root)
    right_arrays = _array_evidence(right, right_root)

    checks = {
        "prediction_invocation_count": (
            left["prediction_invocations"]["count"] == MAX_EVALUATIONS
            and right["prediction_invocations"]["count"] == MAX_EVALUATIONS
        ),
        "native_gpu_request_count": (
            left["observed_gpu_requests"]["observed"]
            == left["observed_gpu_requests"]["expected"]
            and right["observed_gpu_requests"]["observed"]
            == right["observed_gpu_requests"]["expected"]
        ),
        "structure_sample_count": (
            left["structure_samples"]["count"]
            == left["structure_samples"]["expected"]
            and right["structure_samples"]["count"]
            == right["structure_samples"]["expected"]
        ),
        "structure_sample_content": [
            (item["assembly"], item["sha256"])
            for item in left["structure_samples"]["items"]
        ]
        == [
            (item["assembly"], item["sha256"])
            for item in right["structure_samples"]["items"]
        ],
        "candidate_identities": [
            item["candidate"]["identity"] for item in left["evaluations"]
        ]
        == [item["candidate"]["identity"] for item in right["evaluations"]],
        "candidate_assignments": [
            item["candidate"]["assignments"] for item in left["evaluations"]
        ]
        == [item["candidate"]["assignments"] for item in right["evaluations"]],
        "assessment_values": [
            item["assessment"]["values"] for item in left["evaluations"]
        ]
        == [item["assessment"]["values"] for item in right["evaluations"]],
        "selected": [
            (item["candidate"], item["values"]) for item in left["selected"]
        ]
        == [
            (item["candidate"], item["values"]) for item in right["selected"]
        ],
        "rng": left["rng"]["final"] == right["rng"]["final"]
        and sorted(
            (item["round"], item["before"], item["after"])
            for item in left["rng"]["receipts"]
        )
        == sorted(
            (item["round"], item["before"], item["after"])
            for item in right["rng"]["receipts"]
        ),
        "attempt_receipts": [
            (item["round"], item["status"])
            for item in left["attempts"]
        ]
        == [
            (item["round"], item["status"])
            for item in right["attempts"]
        ],
        "numpy_arrays": set(left_arrays) == set(right_arrays)
        and all(
            set(left_arrays[label]) == set(right_arrays[label])
            and all(
                left_arrays[label][key].dtype == right_arrays[label][key].dtype
                and left_arrays[label][key].shape == right_arrays[label][key].shape
                and np.array_equal(
                    left_arrays[label][key],
                    right_arrays[label][key],
                    equal_nan=True,
                )
                for key in left_arrays[label]
            )
            for label in left_arrays
        ),
    }
    payload = {
        "schema": SCHEMA,
        "uninterrupted": str(left_root),
        "crashed_resumed": str(right_root),
        "checks": checks,
        "matched": all(checks.values()),
        "counts": {
            "uninterrupted_prediction_invocations": left[
                "prediction_invocations"
            ]["count"],
            "crash_resume_prediction_invocations": right[
                "prediction_invocations"
            ]["count"],
            "uninterrupted_observed_gpu_requests": left[
                "observed_gpu_requests"
            ]["observed"],
            "crash_resume_observed_gpu_requests": right[
                "observed_gpu_requests"
            ]["observed"],
            "uninterrupted_structure_samples": left[
                "structure_samples"
            ]["count"],
            "crash_resume_structure_samples": right[
                "structure_samples"
            ]["count"],
            "uninterrupted_prediction_receipts": sum(
                item["round"].endswith("/predict") for item in left["attempts"]
            ),
            "crash_resume_prediction_receipts": sum(
                item["round"].endswith("/predict") for item in right["attempts"]
            ),
        },
    }
    _atomic_json(right_root / "comparison.json", payload)
    if not payload["matched"]:
        raise AcceptanceError(f"acceptance mismatch: {checks}")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "crash"):
        command = subparsers.add_parser(name)
        command.add_argument("--out-root", required=True, type=Path)
        command.add_argument("--model-dir", default=MODEL_DIR)
        if name == "crash":
            command.add_argument("--after", required=True, choices=CRASH_POINTS)
    resume = subparsers.add_parser("resume")
    resume.add_argument("--out-root", required=True, type=Path)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--out-root", required=True, type=Path)
    prepare.add_argument("--model-dir", default=MODEL_DIR)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--uninterrupted", required=True, type=Path)
    compare.add_argument("--crashed", required=True, type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command in ("run", "crash"):
            cfg = acceptance_config(args.out_root, model_dir=args.model_dir)
            crash_after = args.after if args.command == "crash" else None
            payload = run_typed_acceptance(
                args.out_root,
                config=cfg,
                crash_after=crash_after,
                hard_exit=args.command == "crash",
            )
        elif args.command == "resume":
            cfg = load_acceptance_config(args.out_root)
            payload = run_typed_acceptance(args.out_root, config=cfg)
        elif args.command == "prepare":
            payload = prepare_typed_acceptance(
                args.out_root,
                model_dir=args.model_dir,
            )
        else:
            payload = compare_acceptance(args.uninterrupted, args.crashed)
    except ControlledCrash as error:
        print(json.dumps({"status": "controlled_crash", "point": error.point}))
        return 75
    except Exception as error:
        print(json.dumps({"status": "failed", "error": repr(error)}))
        return 1
    print(json.dumps({"status": "ok", "result": payload}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
