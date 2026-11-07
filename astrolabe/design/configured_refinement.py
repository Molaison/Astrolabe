"""Task-neutral native refinement assembled from an existing run configuration.

The builder reuses Astrolabe's normal argument parsing, component construction,
mutator pipeline, evaluator pipeline, and :class:`LossEngine`.  Search control
comes from the generic branching owner.  Configuration errors for legacy
trajectory features that this search cannot preserve are rejected before any
model prediction runs.

Proposal seeds use the stable formula
``sha256("astrolabe.configured-refinement.proposal/v1:<run seed>:<step>")``.
Prediction calls always receive the repeat seed unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from astrolabe.design.repeats import RepeatRun
from astrolabe.design.owner import SCIENTIFIC_OWNER
from astrolabe.design.store import DataclassCodec, design_codec
from astrolabe.design.types import Blob, Candidate, Chemical, Key, Polymer, key_for
from astrolabe.runtime_requirements import task_requirements


_SEARCH_KEYS = frozenset(
    {
        "algorithm",
        "owner",
        "width",
        "proposals_per_parent",
        "max_evaluations",
        "proposal_cap",
        "measurement",
        "resume",
        "summary_format",
        "proposal_batch_size",
        "execution",
        "prediction_rng",
    }
)
_WEIGHTED_SCORE_MEASUREMENTS = frozenset({None, "", "score", "loss", "weighted_score"})
_DURABLE_UNSAFE_INITIALIZERS = frozenset({"dplm", "structure", "rfd3"})
_EXCLUDED_SCIENTIFIC_KEYS = frozenset(
    {
        "debug",
        "logging",
        "out_dir",
        "raw_model_results_dir",
        "repeat_index",
        "repeat_plan",
        "repeats",
        "resume",
        "scheduling",
        "summary_format",
        "task_name",
    }
)
_ASSET_MANIFEST_SCHEMA = "astrolabe.configured-refinement.assets/v1"
_WEIGHTLESS_COMPONENT_TYPES = frozenset(
    {
        "dummy",
        "fake",
        "mcmc",
        "motifbreaker",
        "pore_blocker_mcmc",
        "xmix",
    }
)
_ASSET_DIRECTORY_KEYS = frozenset(
    {
        "cache",
        "checkpoint_dir",
        "data_dir",
        "model_dir",
        "opendde_root_dir",
        "weights_dir",
    }
)
_ASSET_FILE_KEYS = frozenset(
    {
        "checkpoint",
        "checkpoint_path",
        "ckpt",
        "ckpt_path",
        "load_checkpoint_path",
        "model_ckpt",
        "model_features",
        "model_config",
        "model_path",
        "weights",
        "weights_path",
    }
)
_UNSUPPORTED_STAGE_FIELDS = (
    "_entrance",
    "acceptance_gate",
    "acceptance_threshold",
    "early_stopping",
    "prefix_early_stopping",
    "transition_consistency_gate",
)


@dataclass(frozen=True)
class _SearchSettings:
    width: int
    proposals_per_parent: int
    max_evaluations: int
    proposal_cap: int
    measurement: str
    summary_format: str | None
    resume: bool
    proposal_batch_size: int
    execution_mode: str
    workers: int
    evaluation_width: int
    prediction_rng: str
    model_sharing: str


@dataclass(frozen=True)
class _AssetFile:
    role: str
    path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class _AssetComponent:
    role: str
    type: str
    state: str
    detail: str | None = None


@dataclass(frozen=True)
class _SelectedAssetManifest:
    components: tuple[_AssetComponent, ...]
    files: tuple[_AssetFile, ...]

    @property
    def complete(self) -> bool:
        return all(item.state != "unsupported" for item in self.components)

    def record(self) -> dict[str, Any]:
        return _asset_manifest_record(self)


@dataclass(frozen=True)
class _OwnedComponent:
    """Keep component shutdown with the run lifecycle that constructed it."""

    value: Any

    def shutdown(self) -> None:
        return None


@dataclass(frozen=True)
class _PipelineSession:
    loss_computer: Any
    stage_manager: Any
    inputs: Any


@dataclass(frozen=True)
class _PreparedCheckpoint:
    """Prepared state under the scientific and selected-asset identities."""

    protocol: Key
    candidate: Blob
    rng_before: Blob
    rng_after: Blob


_PREPARED_CHECKPOINT_CODEC = DataclassCodec(
    _PreparedCheckpoint,
    "astrolabe.configured-refinement.prepared/v1",
    nested_types=(Blob, Key),
)


def _positive_int(search: Mapping[str, Any], name: str) -> int:
    value = search.get(name)
    if type(value) is not int or value < 1:
        raise ValueError(f"cfg.search.{name} must be a positive integer")
    return value


def _search_settings(cfg: Any) -> _SearchSettings:
    if cfg is None or not hasattr(cfg, "get"):
        raise TypeError("cfg must be a composed configuration")
    search = cfg.get("search")
    if not isinstance(search, Mapping):
        raise TypeError("cfg.search must be a mapping")
    unknown = set(str(key) for key in search) - _SEARCH_KEYS
    if unknown:
        raise ValueError(f"cfg.search has unsupported fields: {sorted(unknown)!r}")
    if str(search.get("algorithm", "")).strip() != "branching":
        raise ValueError("configured refinement requires search.algorithm='branching'")
    if str(search.get("owner", "")).strip() != "native_refinement":
        raise ValueError(
            "configured refinement requires search.owner='native_refinement'"
        )

    raw_measurement = search.get("measurement")
    measurement = (
        None
        if raw_measurement is None
        else str(raw_measurement).strip().lower()
    )
    if measurement not in _WEIGHTED_SCORE_MEASUREMENTS:
        raise ValueError(
            "configured refinement measurement must use the existing "
            "LossEngine weighted score; got "
            f"{raw_measurement!r}"
        )

    raw_summary = search.get("summary_format")
    summary_format = None if raw_summary is None else str(raw_summary).strip().lower()
    if summary_format not in (None, "json"):
        raise ValueError(
            "search.summary_format must be null or 'json', "
            f"got {raw_summary!r}"
        )
    resume = search.get("resume", False)
    if type(resume) is not bool:
        raise ValueError("cfg.search.resume must be boolean")
    max_evaluations = _positive_int(
        {"max_evaluations": search.get("max_evaluations", cfg.get("steps"))},
        "max_evaluations",
    )
    raw_proposal_cap = search.get("proposal_cap")
    proposal_cap = (
        max(1, max_evaluations - 1)
        if raw_proposal_cap is None
        else raw_proposal_cap
    )
    if type(proposal_cap) is not int or proposal_cap < 1:
        raise ValueError("cfg.search.proposal_cap must be a positive integer")
    execution = search.get("execution", {})
    if not isinstance(execution, Mapping):
        raise ValueError("search.execution must be a mapping")
    unknown_execution = set(execution) - {"mode", "workers", "in_flight", "model_sharing"}
    if unknown_execution:
        raise ValueError(f"Unsupported search.execution fields: {sorted(unknown_execution)}")
    mode = str(execution.get("mode", "serial"))
    if mode not in {"serial", "isolated"}:
        raise ValueError("search.execution.mode must be 'serial' or 'isolated'")
    available_cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    workers = execution.get("workers", min(2, available_cpus) if mode == "isolated" else 1)
    in_flight = execution.get("in_flight", workers)
    if type(workers) is not int or workers < 1:
        raise ValueError("search.execution.workers must be a positive integer")
    if type(in_flight) is not int or not 1 <= in_flight <= workers:
        raise ValueError("search.execution.in_flight must be positive and <= workers")
    if mode == "serial" and (workers != 1 or in_flight != 1):
        raise ValueError("Multiple evaluator workers require search.execution.mode='isolated'")
    sharing = str(execution.get("model_sharing", "shared" if mode == "isolated" else "replicated"))
    if sharing not in {"replicated", "shared"}:
        raise ValueError("search.execution.model_sharing must be replicated or shared")
    if mode == "serial" and sharing != "replicated":
        raise ValueError("Model sharing applies to isolated evaluator workers")
    width = _positive_int({"width": search.get("width", 2)}, "width")
    per_parent = _positive_int(
        {"proposals_per_parent": search.get("proposals_per_parent", 2)},
        "proposals_per_parent",
    )
    batch = search.get(
        "proposal_batch_size", width * per_parent if mode == "isolated" else 1,
    )
    if type(batch) is not int or batch < 1:
        raise ValueError("search.proposal_batch_size must be a positive integer")
    prediction_rng = str(search.get(
        "prediction_rng", "fixed_request" if mode == "isolated" else "continuation",
    ))
    if prediction_rng not in {"fixed_request", "continuation"}:
        raise ValueError("search.prediction_rng must be fixed_request or continuation")
    if mode == "isolated" and prediction_rng != "fixed_request":
        raise ValueError("Isolated prediction requires search.prediction_rng='fixed_request'")
    return _SearchSettings(
        width=width,
        proposals_per_parent=per_parent,
        max_evaluations=max_evaluations,
        proposal_cap=proposal_cap,
        measurement="weighted_score",
        summary_format=summary_format,
        resume=resume,
        proposal_batch_size=batch,
        execution_mode=mode,
        workers=workers,
        evaluation_width=in_flight,
        prediction_rng=prediction_rng,
        model_sharing=sharing,
    )


def _plain_mapping(value: Any) -> dict[str, Any]:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            converted = OmegaConf.to_container(value, resolve=True)
            return dict(converted) if isinstance(converted, Mapping) else {}
    except Exception:
        pass
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _resolved_mapping(value: Any) -> dict[str, Any]:
    from omegaconf import OmegaConf

    if value is None:
        return {}
    if OmegaConf.is_config(value):
        resolved = OmegaConf.to_container(
            value,
            resolve=True,
            throw_on_missing=True,
        )
    else:
        resolved = value
    if not isinstance(resolved, Mapping):
        raise TypeError("selected component configuration must be a mapping")
    return dict(resolved)


def _case_value(mapping: Mapping[str, Any], name: str) -> Any:
    selected = name.lower()
    for key, value in mapping.items():
        if str(key).lower() == selected:
            return value
    return None


def _merge_component_mapping(
    base: Mapping[str, Any] | None,
    override: Mapping[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for mapping in (base, override):
        if not isinstance(mapping, Mapping):
            continue
        for key, value in mapping.items():
            for existing in tuple(result):
                if existing.lower() == str(key).lower():
                    del result[existing]
                    break
            result[str(key)] = value
    return result


def _matching_component_base(
    container: Mapping[str, Any],
    component_type: str,
) -> Mapping[str, Any] | None:
    selected = _component_base_name(component_type)
    for key, value in container.items():
        if str(key).lower() == selected and isinstance(value, Mapping):
            return value
    return None


def _component_base_name(component_type: str) -> str:
    selected = component_type.lower()
    return {
        "proteinmpnn": "ligandmpnn",
        "multistate_proteinmpnn": "ligandmpnn",
        "far_apart_tied_multitarget_ligandmpnn": "ligandmpnn",
        "grouped_ligandmpnn": "ligandmpnn",
    }.get(selected, selected)


def _config_path(value: Any) -> Path | None:
    if isinstance(value, Path):
        path = value
    elif isinstance(value, str):
        text = os.path.expandvars(value.strip())
        if not text or text.lower() in {"none", "null"}:
            return None
        path = Path(os.path.expanduser(text))
    else:
        return None
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _walk_asset_values(
    value: Any,
    *,
    path: str,
) -> tuple[tuple[str, str, Any], ...]:
    if not isinstance(value, Mapping):
        return ()
    values: list[tuple[str, str, Any]] = []
    for raw_key, item in value.items():
        key = str(raw_key)
        item_path = f"{path}.{key}" if path else key
        if isinstance(item, Mapping):
            values.extend(_walk_asset_values(item, path=item_path))
        else:
            values.append((key.lower(), item_path, item))
    return tuple(values)


def _hash_selected_file(role: str, path: Path) -> _AssetFile:
    from astrolabe.asset_download import sha256_file

    if not path.is_file():
        raise FileNotFoundError(
            f"Selected model asset for {role} is unavailable: {path}"
        )
    before = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    before_identity = (
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
        before.st_dev,
        before.st_ino,
    )
    after_identity = (
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
        after.st_dev,
        after.st_ino,
    )
    if before_identity != after_identity:
        raise RuntimeError(
            f"Selected model asset changed while hashing: {path}"
        )
    return _AssetFile(
        role=role,
        path=str(path),
        sha256=digest,
        size_bytes=after.st_size,
    )


def _files_below(role: str, path: Path) -> tuple[_AssetFile, ...]:
    if not path.is_dir():
        raise FileNotFoundError(
            f"Selected model asset directory for {role} is unavailable: {path}"
        )
    before = tuple(sorted(item.resolve() for item in path.rglob("*") if item.is_file()))
    files = tuple(
        _hash_selected_file(role, item)
        for item in before
    )
    after = tuple(sorted(item.resolve() for item in path.rglob("*") if item.is_file()))
    if before != after:
        raise RuntimeError(
            f"Selected model asset directory changed while hashing: {path}"
        )
    if not files:
        raise FileNotFoundError(
            f"Selected model asset directory for {role} contains no files: {path}"
        )
    return files


def _generic_selected_files(
    role: str,
    mapping: Mapping[str, Any],
) -> tuple[_AssetFile, ...]:
    values = _walk_asset_values(mapping, path=role)
    directories: list[Path] = []
    candidates: list[Path] = []
    for key, _item_path, value in values:
        if key in _ASSET_DIRECTORY_KEYS:
            path = _config_path(value)
            if path is not None:
                directories.append(path)
        elif key in _ASSET_FILE_KEYS or key.startswith("checkpoint_"):
            path = _config_path(value)
            if path is not None:
                candidates.append(path)
        elif key == "model_name":
            path = _config_path(value)
            if path is not None and path.exists():
                candidates.append(path)
        elif key == "experiment_path":
            path = _config_path(value)
            if path is not None:
                directories.append(path)

    files: list[_AssetFile] = []
    for directory in directories:
        files.extend(_files_below(role, directory))
    for candidate in candidates:
        if candidate.exists():
            if candidate.is_dir():
                files.extend(_files_below(role, candidate))
            else:
                files.append(_hash_selected_file(role, candidate))
            continue
        matches = [
            directory / candidate
            for directory in directories
            if (directory / candidate).is_file()
        ]
        if matches:
            files.extend(_hash_selected_file(role, item) for item in matches)
            continue
        raise FileNotFoundError(
            f"Configured model asset for {role} is unavailable: {candidate}"
        )
    return tuple(files)


def _model_component_assets(
    role: str,
    component_type: str,
    mapping: Mapping[str, Any],
) -> tuple[_AssetComponent, tuple[_AssetFile, ...]]:
    selected = component_type.lower()
    if selected == "rosetta_crystal":
        states = _case_value(mapping, "states")
        if not isinstance(states, Mapping) or not states:
            raise ValueError(f"{role} requires crystal states for asset verification")
        files: list[_AssetFile] = []
        for state, config in states.items():
            if not isinstance(config, Mapping):
                raise ValueError(f"{role} crystal state {state!r} must be a mapping")
            params = _config_path(config.get("params"))
            references = config.get("references") or {}
            if not isinstance(references, Mapping):
                raise ValueError(f"{role} crystal references must be a mapping")
            reference_values = list(references.values())
            if config.get("reference") is not None:
                reference_values.append(config["reference"])
            if params is None or not reference_values:
                raise ValueError(f"{role} crystal state {state!r} lacks params or reference")
            for value in (params, *reference_values):
                path = _config_path(value)
                if path is None:
                    raise ValueError(f"{role} crystal state {state!r} has an invalid asset path")
                files.append(_hash_selected_file(role, path))
        return _AssetComponent(role, selected, "hashed"), _deduplicate_asset_files(files)
    if selected in _WEIGHTLESS_COMPONENT_TYPES:
        return _AssetComponent(role, selected, "weight_free"), ()
    if selected == "alphafold3":
        raw_model_dir = _case_value(mapping, "model_dir")
        model_dir = _config_path(raw_model_dir)
        if model_dir is None:
            raise ValueError(
                f"{role} alphafold3 model_dir is not resolvable for durable resume"
            )
        files = _files_below(role, model_dir)
        return _AssetComponent(role, selected, "hashed"), files
    if selected == "alphafold2":
        raw_data_dir = _case_value(mapping, "data_dir")
        data_dir = _config_path(raw_data_dir)
        if data_dir is None:
            raise ValueError(
                f"{role} alphafold2 data_dir is not resolvable for durable resume"
            )
        model_id = _case_value(mapping, "model_id")
        suffix = _case_value(mapping, "suffix")
        model_name = f"model_{3 if model_id is None else model_id}_{'multimer_v3' if suffix is None else suffix}"
        files = (_hash_selected_file(role, data_dir / "params" / f"params_{model_name}.npz"),)
        return _AssetComponent(role, selected, "hashed"), files

    files = _generic_selected_files(role, mapping)
    if files:
        return _AssetComponent(role, selected, "hashed"), files
    return (
        _AssetComponent(
            role,
            selected,
            "unsupported",
            "no selected weight/config file path can be content-verified",
        ),
        (),
    )


def _deduplicate_asset_files(
    files: list[_AssetFile],
) -> tuple[_AssetFile, ...]:
    roles: dict[str, set[str]] = {}
    digests: dict[str, tuple[str, int]] = {}
    for item in files:
        roles.setdefault(item.path, set()).add(item.role)
        digests[item.path] = (item.sha256, item.size_bytes)
    return tuple(
        _AssetFile(
            role=",".join(sorted(roles[path])),
            path=path,
            sha256=digests[path][0],
            size_bytes=digests[path][1],
        )
        for path in sorted(roles)
    )


def _selected_asset_manifest(cfg: Any) -> _SelectedAssetManifest:
    config = _resolved_active_config(cfg, apply_run_lifecycle=False)
    stages = _resolved_mapping(config.get("stages"))
    stage = _resolved_mapping(stages.get("default"))
    mutator_config = _resolved_mapping(config.get("mutator"))
    evaluator_config = _resolved_mapping(config.get("evaluator"))
    mutator_terms = _resolved_mapping(mutator_config.get("mutator_terms"))
    evaluator_terms = _resolved_mapping(
        evaluator_config.get("evaluator_terms")
    )

    components: list[_AssetComponent] = []
    files: list[_AssetFile] = []

    mutator_aliases = stage.get("mutators") or ()
    if isinstance(mutator_aliases, str):
        mutator_aliases = (mutator_aliases,)
    for raw_alias in mutator_aliases:
        alias = str(raw_alias)
        term = _resolved_mapping(mutator_terms.get(alias))
        component_type = str(term.get("_type") or "").strip()
        base = _matching_component_base(mutator_config, component_type)
        effective = _merge_component_mapping(base, term)
        component, selected_files = _model_component_assets(
            f"mutator:{alias}",
            component_type,
            effective,
        )
        components.append(component)
        files.extend(selected_files)

    evaluator_alias = str(stage.get("evaluator") or "").strip()
    evaluator_term = _resolved_mapping(
        evaluator_terms.get(evaluator_alias)
    )
    evaluator_type = str(evaluator_term.get("_type") or "").strip()
    evaluator_base = _matching_component_base(
        evaluator_config,
        evaluator_type,
    )
    evaluator_effective = _merge_component_mapping(
        evaluator_base,
        evaluator_term,
    )
    component, selected_files = _model_component_assets(
        f"evaluator:{evaluator_alias}",
        evaluator_type,
        evaluator_effective,
    )
    components.append(component)
    files.extend(selected_files)

    initializer = _resolved_mapping(config.get("initializer"))
    initializer_mode = str(initializer.get("mode") or "random").strip().lower()
    components.append(
        _AssetComponent(
            f"initializer:{initializer_mode}",
            initializer_mode,
            "config_only",
        )
    )

    loss = _resolved_mapping(config.get("loss"))
    expression_terms = _resolved_mapping(loss.get("expression_terms"))
    loss_terms = _resolved_mapping(loss.get("loss_terms"))
    try:
        from astrolabe.losses.base import extract_loss_terms

        objective_names = set(
            extract_loss_terms(
                (
                    stage.get("weighted_score_expression"),
                    stage.get("acceptance_gate"),
                    stage.get("acceptance_threshold"),
                    stage.get("early_stopping"),
                    stage.get("prefix_early_stopping"),
                    *expression_terms.values(),
                )
            )
        )
    except (ImportError, TypeError, ValueError):
        objective_names = set()
    for name in sorted(objective_names):
        configured = loss_terms.get(name, expression_terms.get(name))
        if configured is None:
            continue
        mapping = _resolved_mapping(configured)
        objective_type = str(mapping.get("_type") or name)
        components.append(
            _AssetComponent(
                f"objective:{name}",
                objective_type,
                "config_only",
            )
        )

    return _SelectedAssetManifest(
        components=tuple(components),
        files=_deduplicate_asset_files(files),
    )


def _asset_manifest_record(manifest: _SelectedAssetManifest) -> dict[str, Any]:
    return {
        "schema": _ASSET_MANIFEST_SCHEMA,
        "complete": manifest.complete,
        "components": [
            {
                "role": item.role,
                "type": item.type,
                "state": item.state,
                "detail": item.detail,
            }
            for item in manifest.components
        ],
        "files": [
            {
                "role": item.role,
                "path": item.path,
                "sha256": item.sha256,
                "size_bytes": item.size_bytes,
            }
            for item in manifest.files
        ],
    }


def _asset_manifest_identity(manifest: _SelectedAssetManifest) -> Key:
    return key_for(
        (
            _ASSET_MANIFEST_SCHEMA,
            _asset_manifest_record(manifest),
        )
    )


def _read_asset_manifest(path: Path) -> _SelectedAssetManifest:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {
        "schema",
        "complete",
        "components",
        "files",
    }:
        raise ValueError(f"Invalid selected-asset manifest: {path}")
    if raw["schema"] != _ASSET_MANIFEST_SCHEMA or type(raw["complete"]) is not bool:
        raise ValueError(f"Unsupported selected-asset manifest: {path}")
    components = []
    for item in raw["components"]:
        if (
            not isinstance(item, dict)
            or set(item) != {"role", "type", "state", "detail"}
            or not all(
                isinstance(item[name], str)
                for name in ("role", "type", "state")
            )
            or (
                item["detail"] is not None
                and not isinstance(item["detail"], str)
            )
            or item["state"]
            not in {"config_only", "hashed", "unsupported", "weight_free"}
        ):
            raise ValueError(f"Invalid selected-asset component in {path}")
        components.append(_AssetComponent(**item))
    files = []
    for item in raw["files"]:
        if (
            not isinstance(item, dict)
            or set(item) != {"role", "path", "sha256", "size_bytes"}
            or not all(
                isinstance(item[name], str)
                for name in ("role", "path", "sha256")
            )
            or type(item["size_bytes"]) is not int
            or item["size_bytes"] < 0
            or len(item["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in item["sha256"])
        ):
            raise ValueError(f"Invalid selected-asset file in {path}")
        files.append(_AssetFile(**item))
    manifest = _SelectedAssetManifest(tuple(components), tuple(files))
    if manifest.complete != raw["complete"]:
        raise ValueError(f"Inconsistent selected-asset completeness in {path}")
    return manifest


def _validate_resume_assets(
    cfg: Any,
    search: _SearchSettings,
    current: _SelectedAssetManifest,
) -> None:
    if not search.resume:
        return
    manifest_path = (
        Path(str(cfg.get("out_dir"))).expanduser().resolve()
        / ".framework"
        / "assets.json"
    )
    saved = _read_asset_manifest(manifest_path)
    if not saved.complete:
        unsupported = [
            item.role
            for item in saved.components
            if item.state == "unsupported"
        ]
        raise ValueError(
            "Cannot resume configured refinement because selected assets "
            f"cannot be content-verified: {unsupported!r}"
        )
    if saved != current:
        raise ValueError(
            "Selected model asset content changed on resume: "
            f"{manifest_path}"
        )


def _contains_enabled_hook(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    for hook in value.values():
        if isinstance(hook, Mapping) and bool(hook.get("enabled", False)):
            return True
    return False


def _validate_configured_losses(run_module: Any, cfg: Any) -> None:
    """Inspect registered loss functions without constructing a LossEngine."""

    try:
        import astrolabe.losses  # noqa: F401
    except Exception as exc:
        raise ValueError(
            "configured refinement cannot preflight registered losses"
        ) from exc
    from astrolabe.losses.base import (
        extract_best_namespace_terms,
        extract_loss_terms,
        loss_registry,
    )

    loss_cfg = _plain_mapping(cfg.get("loss"))
    stages = _plain_mapping(cfg.get("stages"))
    stage = _plain_mapping(stages.get("default"))
    expression_terms = _plain_mapping(loss_cfg.get("expression_terms"))
    defined_losses = _plain_mapping(loss_cfg.get("loss_terms"))
    expressions = [
        stage.get("weighted_score_expression"),
        stage.get("acceptance_gate"),
        stage.get("acceptance_threshold"),
        stage.get("early_stopping"),
        stage.get("prefix_early_stopping"),
        *expression_terms.values(),
    ]
    referenced = set(extract_loss_terms(expressions))
    referenced.update(extract_best_namespace_terms(stage.get("prefix_early_stopping")))
    isolated = _plain_mapping(
        _plain_mapping(cfg.get("search")).get("execution")
    ).get("mode", "serial") == "isolated"

    # Defined terms are inspected conservatively even when this stage does not
    # reference them.  That keeps the gate independent of expression pruning.
    for name in sorted(referenced | set(defined_losses)):
        configured = defined_losses.get(name)
        if configured is None:
            if name not in loss_registry.valid_losses:
                continue
            spec_name = name
        else:
            configured = _plain_mapping(configured)
            spec_name = str(configured.get("_type", "")).strip()
            if not spec_name:
                continue
        try:
            function = loss_registry.get_function(spec_name)
        except (KeyError, ValueError) as exc:
            if name in referenced:
                raise ValueError(
                    f"configured refinement cannot resolve active loss {name!r}"
                ) from exc
            continue
        code = getattr(function, "__code__", None)
        needs_pose = spec_name in run_module.PYROSETTA_LOSS_TYPES or (
            code is not None and "_require_pyrosetta" in code.co_names
        )
        if spec_name in run_module.POST_EVAL_HOOK_LOSS_TYPES or (needs_pose and not isolated):
            raise ValueError(
                "configured refinement loss requires live complex poses or a "
                f"post-eval hook ({name!r}); an exact pose codec is required"
            )


def _validate_static_preflight(run_module: Any, args: Any) -> None:
    """Reject trajectory-only controls before constructing any model owner."""

    stages = _plain_mapping(getattr(args, "stages", None))
    active_stages = {
        str(name): value for name, value in stages.items() if value is not None
    }
    if set(active_stages) != {"default"}:
        raise ValueError(
            "configured refinement supports exactly one static stage named "
            f"'default'; got {sorted(active_stages)!r}"
        )
    stage = _plain_mapping(active_stages["default"])
    search_execution = _plain_mapping(
        _plain_mapping(getattr(args, "search", None)).get("execution")
    )
    search_execution.setdefault("mode", "serial")
    if search_execution.get("mode") == "isolated":
        predockers = _plain_mapping(
            _plain_mapping(getattr(args, "predocker", None)).get("predocker_terms")
        )
        loss_specs = _plain_mapping(
            _plain_mapping(getattr(args, "loss", None)).get("loss_terms")
        )
        expressions = [
            str(stage.get("weighted_score_expression") or ""),
            *[str(item) for item in _plain_mapping(
                _plain_mapping(getattr(args, "loss", None)).get("expression_terms")
            ).values()],
        ]
        if (
            any(value is not None for value in predockers.values())
            or any(
                str(_plain_mapping(value).get("_type", alias)).lower() == "autodock_penalty"
                for alias, value in loss_specs.items() if value is not None
            )
            or any("autodock_penalty" in expression for expression in expressions)
        ):
            raise ValueError(
                "isolated refinement cannot preserve predocker/out_dir measurement "
                "dependencies; use serial refinement for this configuration"
            )
    unsupported = [
        name
        for name in _UNSUPPORTED_STAGE_FIELDS
        if stage.get(name) not in (None, "", [], {})
    ]
    if unsupported:
        raise ValueError(
            "configured refinement cannot preserve stage controls "
            f"{unsupported!r}; use the trajectory pipeline"
        )

    mutator = getattr(args, "mutator", None)
    if bool(getattr(mutator, "stack_pre_update", False)):
        raise ValueError(
            "configured refinement does not support mutator.stack_pre_update"
        )
    if getattr(mutator, "stack_pre_update_max_entropy", None) not in (
        None,
        "",
        "null",
        "None",
    ):
        raise ValueError(
            "configured refinement does not support "
            "mutator.stack_pre_update_max_entropy without stack_pre_update"
        )
    if getattr(args, "structure_start", None) is not None:
        raise ValueError("configured refinement does not support structure_start")
    if bool(getattr(args, "enable_pyrosetta", False)) and search_execution.get("mode") != "isolated":
        raise ValueError(
            "configured refinement cannot persist live PyRosetta poses"
        )
    if bool(getattr(args, "use_metropolis_acceptance", False)):
        raise ValueError(
            "configured refinement uses BranchingPolicy selection, not "
            "Metropolis acceptance"
        )
    initializer_mode = run_module._resolve_initializer_mode(args)
    if initializer_mode in _DURABLE_UNSAFE_INITIALIZERS:
        raise ValueError(
            "configured refinement cannot resume model-backed initialization "
            f"({initializer_mode!r}) before the paid model receipt"
        )
    if _contains_enabled_hook(getattr(args, "post_eval_hooks", None)):
        raise ValueError(
            "configured refinement does not support post-eval hooks that "
            "create or rewrite live pose state"
        )

    from astrolabe.per_residue_scores import per_residue_scores_requires_complex_poses

    if per_residue_scores_requires_complex_poses(
        getattr(args, "per_residue_scores", None)
    ):
        raise ValueError(
            "configured refinement cannot persist live per-residue pose scores"
        )
    _validate_configured_losses(run_module, args)


def _generic_proposal_seed(run_seed: int, proposal_step: int) -> int:
    if type(run_seed) is not int or type(proposal_step) is not int:
        raise TypeError("proposal seed inputs must be integers")
    payload = (
        f"astrolabe.configured-refinement.proposal/v1:"
        f"{run_seed}:{proposal_step}"
    ).encode("ascii")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
    return value or 1


def _candidate_state(candidate: Candidate) -> dict[str, Any]:
    assignments: dict[str, Any] = {}
    for assignment in candidate.assignments:
        value = assignment.value
        if isinstance(value, Polymer):
            assignments[assignment.variable] = {
                "kind": value.kind,
                "sequence": "".join(value.tokens),
                "modifications": [
                    {"index": index, "name": name}
                    for index, name in value.modifications
                ],
            }
        elif isinstance(value, Chemical):
            assignments[assignment.variable] = {
                "kind": "chemical",
                "graph": value.graph.identity.digest,
            }
        else:  # pragma: no cover - validated by the design system
            raise TypeError(
                f"unsupported candidate assignment {type(value).__name__}"
            )
    return {
        "identity": candidate.identity.digest,
        "system": candidate.system.digest,
        "assignments": assignments,
    }


def _candidate_label(candidate: Candidate) -> str:
    return key_for(
        ("astrolabe.configured-refinement.candidate/v1", candidate.assignments)
    ).digest


def _structure_paths(scores: Any, assemblies: tuple[str, ...]) -> dict[str, str]:
    raw = getattr(scores, "complex_paths", None)
    if not isinstance(raw, Mapping):
        return {}
    return {
        assembly: str(raw[assembly])
        for assembly in assemblies
        if assembly in raw and raw[assembly] is not None
    }


def _numeric_metric(value: Any) -> float | None:
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    if type(value) not in (int, float):
        return None
    output = float(value)
    return output if math.isfinite(output) else None


def _loss_context(
    loss_computer: Any,
    terms: Mapping[str, Any],
    assemblies: tuple[str, ...],
) -> dict[str, Any]:
    loss_names = tuple(
        str(name) for name in getattr(loss_computer, "loss_names", ()) or ()
    )
    means: dict[str, float] = {}
    by_assembly: dict[str, dict[str, float]] = {}
    for name in loss_names:
        numeric = _numeric_metric(terms.get(name))
        if numeric is not None:
            means[name] = numeric
    for assembly in assemblies:
        values = terms.get(assembly)
        if not isinstance(values, Mapping):
            continue
        row = {}
        for name in loss_names:
            numeric = _numeric_metric(values.get(name))
            if numeric is not None:
                row[name] = numeric
        if row:
            by_assembly[assembly] = row
    return {
        "mean": means,
        "by_assembly": by_assembly,
    }


def measure_configured_scores(loss_computer, scores, runtime_config, assemblies) -> NativeMeasurement:
    """Keep registry order, reductions and the configured expression unchanged."""
    from .native_refinement import NativeMeasurement

    weighted_score, terms = loss_computer.compute_loss(scores, runtime_config)
    score = float(weighted_score)
    if not math.isfinite(score):
        raise ValueError(f"configured refinement score is not finite: {score!r}")
    score_paths = _structure_paths(scores, assemblies)
    return NativeMeasurement(
        score=score,
        metrics={
            "structure_paths": score_paths,
            "loss": _loss_context(loss_computer, terms, assemblies),
        },
        structure_path=next(iter(score_paths.values()), ""),
    )


def snapshot_score_context(runtime_config, data) -> Blob:
    """Reuse the exact native value codec, including structured contributions."""
    from omegaconf import OmegaConf
    from astrolabe.evaluators.utils import StandardScores
    from .native_scores import snapshot_scores

    def plain(value):
        if OmegaConf.is_config(value):
            value = OmegaConf.to_container(value, resolve=True)
        if type(value) is dict:
            return {key: plain(item) for key, item in value.items()}
        if type(value) is list:
            return [plain(item) for item in value]
        if type(value) is tuple:
            return tuple(plain(item) for item in value)
        return value

    envelope = StandardScores(names=[], plddt={}, complex_paths={})
    envelope.configured_runtime_config = plain(runtime_config)
    reference = snapshot_scores(envelope, data)
    data.persist((reference,))
    return reference


def _canonical_config(cfg: Any, repeat: RepeatRun, search: _SearchSettings) -> Any:
    from omegaconf import OmegaConf, open_dict

    if repeat.output is None:
        raise ValueError("RepeatRun.output is required for configured refinement")
    value = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    value._set_flag("allow_objects", True)
    with open_dict(value):
        if value.get("repeat_plan") is not None:
            del value.repeat_plan
        value.seed = repeat.seed
        value.repeats = 1
        value.repeat_index = repeat.index
        value.out_dir = str(Path(repeat.output).expanduser())
        value.search = {
            "algorithm": "branching",
            "owner": "native_refinement",
            "width": search.width,
            "proposals_per_parent": search.proposals_per_parent,
            "max_evaluations": search.max_evaluations,
            "proposal_cap": search.proposal_cap,
            "measurement": search.measurement,
            "summary_format": search.summary_format,
            "resume": search.resume,
            "proposal_batch_size": search.proposal_batch_size,
            "prediction_rng": search.prediction_rng,
            "execution": {
                "mode": search.execution_mode,
                "workers": search.workers,
                "in_flight": search.evaluation_width,
                "model_sharing": search.model_sharing,
            },
        }
    return value


def _resolved_active_config(
    args: Any,
    *,
    apply_run_lifecycle: bool = True,
) -> dict[str, Any]:
    """Resolve selected science inputs without optional inactive roots."""

    from astrolabe import run_config_lifecycle
    from omegaconf import OmegaConf
    from omegaconf.errors import InterpolationKeyError, MissingMandatoryValue

    persisted = (
        run_config_lifecycle.build_persisted_run_config(args)
        if apply_run_lifecycle
        else args
    )
    unresolved = OmegaConf.to_container(persisted, resolve=False)
    if not isinstance(unresolved, Mapping):
        raise ValueError("scientific configuration must resolve to a mapping")

    # A pruned run config can still contain implementation roots retained by an
    # older lifecycle serializer. Resolve selected component subtrees while
    # dropping inactive roots whose optional assets are deliberately unset.
    active_component_names: set[str] = set()
    known_component_names: dict[str, set[str]] = {}
    sections: dict[str, Mapping[str, Any]] = {}
    for section_name, terms_name, selector_name in (
        ("evaluator", "evaluator_terms", "evaluator"),
        ("mutator", "mutator_terms", "mutators"),
    ):
        section = unresolved.get(section_name)
        if not isinstance(section, Mapping):
            continue
        sections[section_name] = section
        terms = section.get(terms_name)
        if not isinstance(terms, Mapping):
            continue
        section_components: set[str] = set()
        for key, value in section.items():
            if key not in {
                terms_name,
                "enabled_aliases",
                "pre_eval_hooks",
                "scheduling",
                "selectors",
            } and isinstance(value, Mapping):
                section_components.add(str(key).casefold())
        known_component_names[section_name] = section_components
        stages = unresolved.get("stages")
        default_stage = (
            stages.get("default", {})
            if isinstance(stages, Mapping)
            else {}
        )
        selected = default_stage.get(selector_name) if isinstance(
            default_stage, Mapping
        ) else None
        aliases = (
            [selected]
            if isinstance(selected, str)
            else list(selected or ())
        )
        for raw_alias in aliases:
            term = terms.get(str(raw_alias))
            if not isinstance(term, Mapping):
                continue
            component_type = str(term.get("_type") or "").strip()
            if component_type:
                active_component_names.add(component_type.casefold())
                active_component_names.add(
                    _component_base_name(component_type)
                )

    unresolved_cfg = OmegaConf.create(unresolved)
    filtered = dict(unresolved)
    for section_name, section in sections.items():
        selected_section = dict(section)
        for key in tuple(selected_section):
            folded = str(key).casefold()
            if (
                folded in known_component_names[section_name]
                and folded not in active_component_names
            ):
                del selected_section[key]
        filtered[section_name] = selected_section

    for key in tuple(filtered):
        folded = str(key).casefold()
        if folded in active_component_names:
            continue
        value = filtered[key]
        if not isinstance(value, Mapping):
            continue
        if any(
            folded in names
            for names in known_component_names.values()
        ):
            del filtered[key]
            continue

    initializer = filtered.get("initializer")
    if isinstance(initializer, Mapping):
        raw_mode = str(initializer.get("mode") or "random").strip().lower()
        mode = {
            "rfd-3": "rfd3",
            "start-from-structure": "structure",
            "start_from_structure": "structure",
            "existing-structure": "structure",
            "existing_structure": "structure",
        }.get(raw_mode, raw_mode)
        selected_initializer = dict(initializer)
        if mode != "rfd3":
            selected_initializer.pop("rfd3", None)
        if mode != "structure":
            selected_initializer.pop("structure", None)
        filtered["initializer"] = selected_initializer

    # Recompute environment requirements after inactive component sections were
    # removed, so unrelated replay-only paths do not enter scientific identity.
    run_config_lifecycle._prune_env_sections(filtered)
    required_env_keys = {
        str(key)
        for key in (
            task_requirements(filtered).get("required_env_keys") or ()
        )
    }
    filtered_cfg = OmegaConf.create(filtered)
    missing = object()

    def is_required(path: str, key: str) -> bool:
        top_level = path.split(".", 1)[0].casefold()
        selected_component_path = (
            top_level in active_component_names
            or path.startswith("evaluator.")
            or path.startswith("mutator.")
        )
        return (
            path.startswith("loss.")
            or path.startswith("stages.")
            or (
                selected_component_path
                and key.casefold()
                in _ASSET_FILE_KEYS | _ASSET_DIRECTORY_KEYS
            )
            or any(
                path == required_key
                or path.startswith(f"{required_key}.")
                or required_key.startswith(f"{path}.")
                for required_key in required_env_keys
            )
            or (
                (path.startswith("evaluator.") or path.startswith("mutator."))
                and key.casefold()
                in {"python_bin", "repo_path", "repo_paths"}
            )
        )

    def missing_value(
        error: Exception,
        *,
        path: str,
        key: str,
    ) -> Any:
        if is_required(path, key):
            raise error
        return missing

    def resolve_node(node: Any, path: str, key: str) -> Any:
        if OmegaConf.is_dict(node):
            result = {}
            for raw_child_name in node.keys():
                child_name = str(raw_child_name)
                child_path = f"{path}.{child_name}" if path else child_name
                try:
                    child = node[raw_child_name]
                except (InterpolationKeyError, MissingMandatoryValue) as exc:
                    resolved = missing_value(
                        exc,
                        path=child_path,
                        key=child_name,
                    )
                    if resolved is not missing:
                        result[child_name] = resolved
                    continue
                if child == "???":
                    resolved = missing_value(
                        MissingMandatoryValue(
                            f"Missing mandatory value at {child_path}"
                        ),
                        path=child_path,
                        key=child_name,
                    )
                    if resolved is not missing:
                        result[child_name] = resolved
                    continue
                resolved = resolve_node(child, child_path, child_name)
                if resolved is not missing:
                    result[child_name] = resolved
            return result
        if OmegaConf.is_list(node):
            result = []
            for index in range(len(node)):
                try:
                    child = node[index]
                except (InterpolationKeyError, MissingMandatoryValue) as exc:
                    resolved = missing_value(
                        exc,
                        path=f"{path}[{index}]",
                        key=key,
                    )
                    if resolved is not missing:
                        result.append(resolved)
                    continue
                resolved = resolve_node(
                    child,
                    f"{path}[{index}]",
                    key,
                )
                if resolved is not missing:
                    result.append(resolved)
            return result
        if not OmegaConf.is_config(node):
            return node
        try:
            if bool(getattr(node, "_is_missing", lambda: False)()):
                raise MissingMandatoryValue(f"Missing value at {path}")
            return OmegaConf.to_container(
                node,
                resolve=True,
                throw_on_missing=True,
            )
        except (InterpolationKeyError, MissingMandatoryValue) as exc:
            return missing_value(exc, path=path, key=key)

    raw = resolve_node(filtered_cfg, "", "")
    if not isinstance(raw, Mapping):
        raise ValueError("scientific configuration must resolve to a mapping")
    result = {str(name): value for name, value in raw.items()}
    for name in tuple(result):
        original = unresolved_cfg[name]
        if (
            result[name] == {}
            and OmegaConf.is_dict(original)
            and len(original) > 0
        ):
            del result[name]
    return result


def _scientific_config_payload(args: Any) -> dict[str, Any]:
    """Return the complete model-affecting config with lifecycle fields removed."""

    raw = _resolved_active_config(args)

    def normalize(value: Any, *, path: str) -> Any:
        if value is None or type(value) in (bool, int, str):
            return value
        if type(value) is float:
            if not math.isfinite(value):
                raise ValueError(f"scientific configuration has non-finite value at {path}")
            return value
        if isinstance(value, Path):
            return value.as_posix()
        if isinstance(value, Mapping):
            result = {}
            for name, item in value.items():
                key = str(name)
                if key in _EXCLUDED_SCIENTIFIC_KEYS:
                    continue
                result[key] = normalize(item, path=f"{path}.{key}")
            return result
        if isinstance(value, (list, tuple)):
            return [
                normalize(item, path=f"{path}[{index}]")
                for index, item in enumerate(value)
            ]
        if isinstance(value, (set, frozenset)):
            return sorted(
                (normalize(item, path=f"{path}{{set}}") for item in value),
                key=lambda item: key_for(item).digest,
            )
        value = getattr(value, "value", value)
        if value is None or type(value) in (bool, int, str):
            return value
        raise ValueError(
            f"scientific configuration value at {path} has no stable codec: "
            f"{type(value).__module__}.{type(value).__qualname__}"
        )

    payload = normalize(raw, path="cfg")
    if not isinstance(payload, dict):
        raise TypeError("scientific configuration payload must be a mapping")
    # Fail before any paid model initialization if a value has no design codec.
    key_for(("astrolabe.configured-refinement.scientific/v1", payload))
    return payload


def _validate_resume_root(cfg: Any, search: _SearchSettings, repeat: RepeatRun) -> None:
    if not search.resume:
        return
    if repeat.total != 1 or repeat.index != 0:
        raise ValueError("search.resume=true requires exactly one repeat")
    configured = Path(str(cfg.get("out_dir"))).expanduser().resolve()
    if repeat.output is None or Path(repeat.output).expanduser().resolve() != configured:
        raise ValueError("search.resume=true requires cfg.out_dir to be the original run root")
    if not configured.is_dir():
        raise FileNotFoundError(f"Resume run directory does not exist: {configured}")
    required = (
        configured / ".framework" / "journal.sqlite",
        configured / ".framework" / "assets.json",
        configured / "config.yaml",
        configured / "provenance.json",
        configured / "01_evaluations.jsonl",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Cannot resume configured refinement without durable run files: "
            + ", ".join(str(path) for path in missing)
        )


def _validate_persisted_scientific_config(args: Any) -> dict[str, Any]:
    from omegaconf import OmegaConf

    current = _scientific_config_payload(args)
    saved_path = Path(args.out_dir) / "config.yaml"
    saved = _scientific_config_payload(OmegaConf.load(saved_path))
    if current != saved:
        raise ValueError(
            f"Scientific configuration changed on resume: {saved_path}"
        )
    return current


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        _write_json(temporary, value)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_configured_refinement(cfg: Any, repeat: RepeatRun) -> dict[str, Any]:
    """Run one refinement repeat under the shared scientific state owner."""

    if not isinstance(repeat, RepeatRun):
        raise TypeError("repeat must be a design.repeats.RepeatRun")
    search = _search_settings(cfg)
    run_cfg = _canonical_config(cfg, repeat, search)
    _validate_resume_root(run_cfg, search, repeat)
    selected_assets = _selected_asset_manifest(run_cfg)
    _validate_resume_assets(run_cfg, search, selected_assets)
    from astrolabe.runtime.session import resource_session

    with resource_session(_plain_mapping(run_cfg.get("execution"))) as resources:
        if search.execution_mode == "isolated" and resources is None:
            terms = _plain_mapping(_plain_mapping(run_cfg.get("evaluator")).get("evaluator_terms"))
            if any(
                str(_plain_mapping(term).get("_type", "")).lower() != "dummy"
                for term in terms.values() if term is not None
            ):
                raise ValueError(
                    "Isolated evaluator workers require a shared resource owner: "
                    "set execution.mode=resident_serial/adaptive, or inherit "
                    "ASTROLABE_RESOURCE_SOCKET"
                )
        with SCIENTIFIC_OWNER:
            return _run_configured_refinement_locked(
                run_cfg, repeat, search, selected_assets,
            )


def _run_configured_refinement_locked(
    run_cfg: Any,
    repeat: RepeatRun,
    search: _SearchSettings,
    selected_assets: _SelectedAssetManifest,
) -> dict[str, Any]:
    """Run one generic refinement repeat over the selected single stage.

    The configured components are constructed once by the normal pipeline
    initializer.  The returned summary contains only nested search metrics,
    complete candidate assignments, and evaluation artifact paths.
    """

    from astrolabe.app import _load_run_module
    from astrolabe.argparser import set_random_seeds
    from astrolabe.core import global_config
    from astrolabe.design.branching import BranchingPolicy
    from astrolabe.design.legacy import LegacyContext
    from astrolabe.design.native_refinement import (
        NativeMeasurement,
        NativeProposal,
        NativeRefinementSpec,
        _PROPOSAL_CODEC,
        run_native_refinement,
    )
    from astrolabe.design.runtime import ContractConflict, Runtime
    from astrolabe.design.store import DataStore

    args = None
    runtime = None
    succeeded = False
    completed_existing = False
    global_config.reset()
    run_module = _load_run_module()

    _validate_static_preflight(run_module, run_cfg)
    try:
        existing_run_dir = (
            Path(str(run_cfg.out_dir)).expanduser().resolve()
            if search.resume
            else None
        )
        args = run_module.parse_arguments(
            run_cfg,
            existing_run_dir=existing_run_dir,
            persist_config=not search.resume,
        )
        run_module.setup_logging(args.out_dir, getattr(args, "logging", None))
        if search.resume:
            provenance = json.loads(
                (Path(args.out_dir) / "provenance.json").read_text(encoding="utf-8")
            )
            completed_existing = provenance.get("state") == "completed"

        scientific_config = (
            _validate_persisted_scientific_config(args)
            if search.resume
            else _scientific_config_payload(args)
        )
        scientific_identity = key_for(
            (
                "astrolabe.configured-refinement.scientific/v1",
                scientific_config,
            )
        )
        asset_identity = _asset_manifest_identity(selected_assets)
        protocol = key_for(
            (
                "astrolabe.configured-refinement.protocol/v3",
                scientific_identity,
                asset_identity,
            )
        )

        output = Path(args.out_dir)
        runtime_root = output / ".framework"
        if not search.resume:
            runtime_root.mkdir(parents=True, exist_ok=True)
            _write_json_atomic(
                runtime_root / "assets.json",
                _asset_manifest_record(selected_assets),
            )
        data = DataStore(runtime_root / "data")
        runtime = Runtime(
            root=runtime_root,
            data=data,
            scientific_contract=scientific_identity,
        )
        runtime.data.register(_PREPARED_CHECKPOINT_CODEC)

        stage = _plain_mapping(_plain_mapping(args.stages).get("default"))
        mutator_aliases = tuple(
            str(alias) for alias in (stage.get("mutators") or ())
        )
        if not mutator_aliases:
            raise ValueError("configured refinement stage selects no mutators")
        evaluator_alias = str(stage.get("evaluator") or "")
        if not evaluator_alias:
            raise ValueError("configured refinement stage selects no evaluator")
        assemblies = tuple(str(name) for name in (stage.get("oligo") or ()))
        if not assemblies:
            raise ValueError("configured refinement stage selects no assemblies")

        prepared_checkpoint = runtime.load(
            "configured_refinement/prepare",
            _PREPARED_CHECKPOINT_CODEC,
        )
        if search.resume and prepared_checkpoint is None:
            raise ContractConflict(
                "Configured refinement resume has no prepared initial state"
            )
        if prepared_checkpoint is not None:
            if prepared_checkpoint.protocol != protocol:
                raise ContractConflict(
                    "Configured refinement scientific configuration changed on resume"
                )
            runtime._restore_rng(prepared_checkpoint.rng_before)
            prepared = run_module.prepare_pipeline_state(
                args,
                apply_initialization=False,
            )
        else:
            rng_before = runtime._capture_rng()
            prepared = run_module.prepare_pipeline_state(args)

        context = LegacyContext()
        translation = context.capture_candidate(assemblies=assemblies, data=data)
        if prepared_checkpoint is None:
            initial_candidate = translation.candidate
            rng_after = runtime._capture_rng()
            candidate_ref = data.put(initial_candidate, design_codec)
            runtime.save(
                "configured_refinement/prepare",
                _PreparedCheckpoint(
                    protocol,
                    candidate_ref,
                    rng_before,
                    rng_after,
                ),
                _PREPARED_CHECKPOINT_CODEC,
            )
        else:
            initial_candidate = data.get(
                prepared_checkpoint.candidate,
                design_codec,
            )
            if initial_candidate.system != translation.system.identity:
                raise ContractConflict(
                    "Prepared initial candidate belongs to another molecular System"
                )
            context.apply_candidate(
                initial_candidate,
                translation=translation,
                data=data,
            )
            runtime._restore_rng(prepared_checkpoint.rng_after)

        runtime_config = {
            "mutator_stack_pre_update": False,
            "state_conflict_diagnostics": getattr(
                getattr(args, "mutator", None),
                "state_conflict_diagnostics",
                None,
            ),
        }
        session_holder: dict[str, _PipelineSession] = {}
        translated_assembly_names = tuple(
            str(item.identity) for item in translation.assemblies
        )
        if translated_assembly_names != assemblies:
            raise ValueError("translated assemblies differ from selected stage assemblies")
        expected_assemblies = assemblies

        def score_context(step):
            name = f"configured-refinement/score-context/{step}"
            reference = runtime.load(name)
            if reference is None:
                if step != 0:
                    raise ContractConflict("Proposal has no frozen scoring context")
                reference = snapshot_score_context(runtime_config, data)
                runtime.save(name, reference)
            return reference

        def load_mutator() -> _OwnedComponent:
            overrides = {}
            if search.execution_mode == "isolated" and search.model_sharing == "replicated":
                from .parallel_refinement import DeferredEvaluator

                def deferred_factory(*_args, **_kwargs):
                    return {evaluator_alias: DeferredEvaluator(evaluator_alias)}

                overrides["evaluator_factory"] = deferred_factory
            session = _PipelineSession(
                *run_module.load_pipeline_components(args, prepared, **overrides)
            )
            loaded_mutators = tuple(
                str(alias)
                for alias in (
                    getattr(session.stage_manager.current_setting, "mutators", None)
                    or ()
                )
            )
            if loaded_mutators != mutator_aliases:
                raise ValueError(
                    "loaded stage mutators differ from the prepared configuration"
                )
            if tuple(session.stage_manager.current_oligo or ()) != assemblies:
                raise ValueError(
                    "loaded stage assemblies differ from the prepared configuration"
                )
            if str(
                session.stage_manager.resolve_stage_evaluator(
                    session.stage_manager.execution_setting
                )
                or ""
            ) != evaluator_alias:
                raise ValueError(
                    "loaded stage evaluator differs from the prepared configuration"
                )
            session_holder["session"] = session
            return _OwnedComponent(session.stage_manager.current_mutators)

        def load_evaluator() -> _OwnedComponent:
            try:
                session = session_holder["session"]
            except KeyError:
                raise RuntimeError(
                    "pipeline session must be loaded before its evaluator"
                ) from None
            return _OwnedComponent(session.stage_manager.current_evaluator)

        def propose(
            mutators: _OwnedComponent,
            parent: Candidate,
            scores: Any,
            *,
            seed: int,
            step: int,
            generation: int,
            parent_id: int,
            proposal_index: int,
            output: Path,
        ) -> NativeProposal:
            del parent, generation, parent_id, proposal_index, output
            set_random_seeds(seed)
            global_config.set_variable("step", int(step))
            try:
                session = session_holder["session"]
            except KeyError:
                raise RuntimeError("pipeline session is not loaded") from None
            if tuple(mutators.value) != tuple(session.stage_manager.current_mutators):
                raise ValueError("mutation callback received another pipeline session")
            # Spawned Score workers do not initialize Rosetta in this process,
            # and their poses are deliberately absent from the score receipt.
            # Rebuild the parent's private selector view without rescoring.
            run_module._ensure_evaluation_complex_poses(
                args, scores, session.loss_computer,
            )
            run_module.execute_mutation_step(
                session.stage_manager.current_mutators,
                scores,
                runtime_config,
                int(step),
            )
            # Mutators queue whole-step changes.  Apply them before translating
            # the child Candidate; pre-update stacks remain explicitly rejected.
            global_config.segment_manager.apply_mutations()
            if search.execution_mode == "isolated":
                runtime.save(
                    f"configured-refinement/score-context/{step}",
                    snapshot_score_context(runtime_config, data),
                )
            child = context.capture_candidate(
                assemblies=assemblies,
                state=translation.state,
                data=data,
            ).candidate
            return NativeProposal(child, _candidate_label(child))

        def evaluate(
            evaluator: _OwnedComponent,
            candidate: Candidate,
            *,
            seed: int,
            step: int,
            assemblies: tuple[str, ...],
            output: Path,
            proposal_id: int,
        ) -> Any:
            del output
            if seed != repeat.seed:
                raise ValueError("native prediction seed differs from the repeat seed")
            if tuple(assemblies) != expected_assemblies:
                raise ValueError("native prediction assemblies changed during the run")
            if search.prediction_rng == "fixed_request":
                set_random_seeds(seed)
            global_config.set_variable("step", int(step))
            del candidate, proposal_id
            try:
                session = session_holder["session"]
            except KeyError:
                raise RuntimeError("pipeline session is not loaded") from None
            if evaluator.value is not session.stage_manager.current_evaluator:
                raise ValueError("evaluation callback received another pipeline session")
            return run_module.perform_evaluation(
                args,
                evaluator.value,
                runtime_config,
                int(step),
                list(assemblies),
                loss_computer=session.loss_computer,
            )

        def measure(
            candidate: Candidate,
            scores: Any,
            job: Any,
        ) -> NativeMeasurement:
            try:
                session = session_holder["session"]
            except KeyError:
                raise RuntimeError("pipeline session is not loaded") from None
            active_assemblies = tuple(session.stage_manager.current_oligo or ())
            if active_assemblies != assemblies:
                raise ValueError("measurement callback stage assemblies changed")
            return measure_configured_scores(
                session.loss_computer, scores, runtime_config, active_assemblies,
            )

        policy = BranchingPolicy(
            width=search.width,
            proposals_per_parent=search.proposals_per_parent,
            max_evaluations=search.max_evaluations,
            proposal_cap=search.proposal_cap,
            seed=repeat.seed,
            evaluation_seed=repeat.seed,
            proposal_seed=_generic_proposal_seed,
            proposal_seed_identity=(
                "astrolabe.configured-refinement.proposal-seed/v1"
            ),
            quota_mode="attempts",
            expansion="round_robin",
            proposal_batch_size=search.proposal_batch_size,
            evaluation_width=search.evaluation_width,
            pipeline=search.execution_mode == "isolated",
            immediate_evaluation=(
                search.execution_mode == "serial"
                and search.proposal_batch_size == 1
            ),
            cap_error=True,
        )

        journal_path = output / "01_evaluations.jsonl"
        if search.resume and not journal_path.is_file():
            raise FileNotFoundError(journal_path)
        prior_events = ()
        if search.resume:
            prior_events = tuple(
                json.loads(line)["event"]
                for line in journal_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        with journal_path.open(
            "a" if search.resume else "x",
            encoding="utf-8",
        ) as journal:
            def event(kind: str, fields: Mapping[str, Any] | None = None) -> None:
                journal.write(
                    json.dumps(
                        {"event": kind, **dict(fields or {})},
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n"
                )
                journal.flush()

            try:
                if not search.resume:
                    event(
                        "search_started",
                        {
                            "algorithm": "branching",
                            "owner": "native_refinement",
                            "seed": repeat.seed,
                            "width": search.width,
                            "proposals_per_parent": search.proposals_per_parent,
                            "max_evaluations": search.max_evaluations,
                            "proposal_cap": search.proposal_cap,
                            "assemblies": assemblies,
                            "proposal_batch_size": search.proposal_batch_size,
                            "execution": {
                                "mode": search.execution_mode,
                                "workers": search.workers,
                                "in_flight": search.evaluation_width,
                                "model_sharing": search.model_sharing,
                                "prediction_rng": (
                                    "fixed-request-seed/v1"
                                    if search.prediction_rng == "fixed_request"
                                    else "legacy-continuation"
                                ),
                            },
                        },
                    )
                spec = NativeRefinementSpec(
                    output=output,
                    seed=repeat.seed,
                    width=search.width,
                    proposals_per_parent=search.proposals_per_parent,
                    max_evaluations=search.max_evaluations,
                    evaluator_alias=evaluator_alias,
                    mutator_alias="stack:" + ",".join(mutator_aliases),
                    assemblies=assemblies,
                    initial_candidate=initial_candidate,
                    context=context,
                    translation=translation,
                    data=data,
                    load_mutator=load_mutator,
                    load_evaluator=load_evaluator,
                    propose=propose,
                    evaluate=evaluate,
                    measure=measure,
                    label=_candidate_label,
                    proposal_seed=_generic_proposal_seed,
                    on_event=event,
                )
                if search.execution_mode == "isolated":
                    from omegaconf import OmegaConf
                    from .parallel_refinement import run_parallel_native_refinement

                    result = run_parallel_native_refinement(
                        spec, policy=policy, protocol=protocol, runtime=runtime,
                        config_yaml=OmegaConf.to_yaml(run_cfg, resolve=True),
                        workers=search.workers,
                        worker_output=output / ".prediction_workers",
                        model_sharing=search.model_sharing,
                        loss_order=lambda: tuple(session_holder["session"].loss_computer.loss_specs),
                        score_context=score_context,
                    )
                else:
                    result = run_native_refinement(
                        spec, policy=policy, protocol=protocol, runtime=runtime,
                    )
                if not result.selected:
                    raise RuntimeError("configured refinement selected no candidate")

                selected_ids = {member.id for member in result.selected}
                evaluations = []
                for evaluation_id, member in result.member_items():
                    metrics = result.metrics(member)
                    evaluations.append(
                        {
                            "evaluation_id": evaluation_id,
                            "generation": member.generation,
                            "parent_id": member.parent_id,
                            "score": member.score,
                            "candidate_identity": member.candidate.identity.digest,
                            "system_identity": member.candidate.system.digest,
                            "structure_paths": metrics["structure_paths"],
                            "loss": metrics["loss"],
                            "selected": evaluation_id in selected_ids,
                        }
                    )
                state = result.state
                beam = []
                for member, record in zip(
                    result.selected,
                    result.members(),
                    strict=True,
                ):
                    metrics = result.metrics(record)
                    beam.append(
                        {
                            "evaluation_id": member.id,
                            "score": member.score,
                            "candidate": _candidate_state(record.candidate),
                            "structure_paths": metrics["structure_paths"],
                            "loss": metrics["loss"],
                        }
                    )
                summary = {
                    "out_dir": str(output),
                    "routing": {
                        "algorithm": "branching",
                        "owner": "native_refinement",
                    },
                    "seed": repeat.seed,
                    "repeats": 1,
                    "search": {
                        "width": search.width,
                        "proposals_per_parent": search.proposals_per_parent,
                        "max_evaluations": search.max_evaluations,
                        "proposal_cap": search.proposal_cap,
                        "proposal_batch_size": search.proposal_batch_size,
                        "execution": {
                            "mode": search.execution_mode,
                            "workers": search.workers,
                            "in_flight": search.evaluation_width,
                            "prediction_rng": search.prediction_rng,
                            "model_sharing": search.model_sharing,
                        },
                        "evaluations": result.evaluations,
                        "proposals": result.proposals,
                        "duplicates": int(getattr(state, "duplicates", 0)),
                        "stop_reason": getattr(state, "stop_reason", None),
                        "budget_complete": result.evaluations >= search.max_evaluations,
                    },
                    "components": {
                        "mutators": list(mutator_aliases),
                        "evaluator": evaluator_alias,
                        "load_order": ["mutator", "evaluator"],
                    },
                    "beam": beam,
                    "evaluations": evaluations,
                }
                _write_json(output / "03_final_beam.json", summary)
                if "search_completed" not in prior_events:
                    event(
                        "search_completed",
                        {
                            "evaluations": result.evaluations,
                            "proposals": result.proposals,
                            "duplicates": summary["search"]["duplicates"],
                        },
                    )
            except BaseException as error:
                event("search_failed", {"error": repr(error)})
                raise

        succeeded = True
        if not completed_existing:
            run_module._finish_run_provenance(args, "completed")
        return summary
    except BaseException as error:
        if not completed_existing:
            run_module._finish_run_provenance(args, "failed", error=error)
        raise
    finally:
        try:
            run_module._shutdown_pipeline_resources()
        finally:
            try:
                run_module._cleanup_auto_managed_raw_dir(
                    args=args,
                    succeeded=succeeded,
                )
            finally:
                try:
                    if runtime is not None:
                        runtime.close()
                finally:
                    global_config.reset()


__all__ = ["run_configured_refinement"]
