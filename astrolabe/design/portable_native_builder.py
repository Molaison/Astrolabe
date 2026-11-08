"""Production builder for the durable, static ``PortableNativeFlow`` profile.

This module owns configuration preflight, model-free pipeline preparation,
explicit initializer recovery, source/asset identity, cold native operations,
and construction of the frozen portable flow.  The scientific decision math is
injected only after the contract is fixed because the current ``Decisions``
implementation spans ``run._initialize_optimization_state`` and the mutation,
evaluation, and stage-entry portions of ``run._advance_optimization_state``.
No formula from that implementation is copied here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
from typing import Any, Protocol

from omegaconf import OmegaConf, open_dict

from .legacy import LegacyContext
from .native_mutation_state import (
    MutationContextPolicy,
    native_mutation_segments,
)
from .native_registry import NativeComponentContext, NativeOperationHooks
from .native_scores import NATIVE_SCORES_ENVELOPE_CODEC
from .native_support import NativeBindings
from .portable_native_flow import (
    PortableNativeFlow,
    PortableNativeMath,
    PortableNativeSupport,
    run_portable_native,
    validate_static_nonpose_support,
)
from .portable_native_state import (
    DECISION_REQUEST_CODEC,
    DECISION_RESULT_CODEC,
    NATIVE_RUNTIME_CONTEXT_CODEC,
    NATIVE_STEP_STATE_CODEC,
    NUMERIC_TREE_CODEC,
    SCIENTIFIC_CONTRACT_SCHEMA,
    SCIENTIFIC_RUN_CONTRACT_CODEC,
    NativeRuntimeContext,
    ScientificRunContract,
    store_numeric_tree,
)
from .runtime import ContractConflict, Runtime
from .store import DataclassCodec, DataStore, JSONCodec, design_codec
from .types import Blob, Candidate, Key, Polymer, SamplingPolicy, key_for


PROFILE_SCHEMA = "astrolabe.portable-native.profile/v1"
PREPARED_SCHEMA = "astrolabe.portable-native.prepared/v1"
PREPARED_MANIFEST_SCHEMA = "astrolabe.portable-native.prepared-manifest/v1"
AF3_MCMC_TEMPLATE_SCHEMA = "astrolabe.portable-native.af3-mcmc-template/v1"
SCIENTIFIC_CONFIG_SCHEMA = "astrolabe.portable-native.scientific-config/v3"

_RECOVERY_KEYS = frozenset({"mode", "resume"})
_MODEL_INITIALIZERS = frozenset({"dplm", "structure", "rfd3"})
_UNSUPPORTED_STAGE_FIELDS = (
    "_entrance",
    "transition_consistency_gate",
)
_SCIENTIFIC_CONTEXT_FIELDS = (
    "binding_site",
    "chai_restraints",
    "channel",
    "covalent_bonds",
    "cyclization",
    "dynamic_binding_site_enabled",
    "ligand_resnames",
    "named_files",
    "nonstandard_residues",
    "protenix_constraint",
    "user_ccd",
)
_JSON = JSONCodec()
_MISSING = object()


class UnsupportedPortableNativeProfile(ValueError):
    """The selected configuration cannot be represented by the portable flow."""


class PortableNativeMathUnavailable(RuntimeError):
    """The caller did not provide the shared current-``Decisions`` adapter."""


@dataclass(frozen=True)
class PortableRecoveryPolicy:
    mode: str
    resume: bool

    def __post_init__(self) -> None:
        if self.mode != "durable":
            raise UnsupportedPortableNativeProfile(
                "portable native requires recovery.mode='durable'"
            )
        if type(self.resume) is not bool:
            raise UnsupportedPortableNativeProfile(
                "portable native recovery.resume must be boolean"
            )


@dataclass(frozen=True)
class ContentRecord:
    """One exact source or scientific asset selected by the profile."""

    role: str
    path: str
    kind: str
    sha256: str
    size: int | None
    entries: tuple[tuple[str, str, int], ...] = ()


@dataclass(frozen=True)
class PortableNativeMathContext:
    """All immutable inputs needed to construct the shared Decisions adapter."""

    args: Any
    prepared: Any
    loss_computer: Any
    stage_manager: Any
    bindings: NativeBindings
    contract: ScientificRunContract
    initial_candidate: Candidate
    runtime_context: Any
    root_runtime_config: Mapping[str, Any]
    sample_protocol: Key
    run_module: Any
    mutation_context: MutationContextPolicy


class PortableNativeMathFactory(Protocol):
    def __call__(self, context: PortableNativeMathContext) -> PortableNativeMath:
        ...


@dataclass(frozen=True)
class _PreparedReceipt:
    schema: str
    config_identity: Key
    source_identity: Key
    resource_manifest: Key
    initializer_mode: str
    initial_candidate: Candidate
    rng_before: Blob
    rng_after: Blob
    initializer_result: Blob | None


_INITIALIZER_RESULT_CODEC = DataclassCodec(
    importlib.import_module("astrolabe.initializers.base").InitializerResult,
    "astrolabe.portable-native.initializer-result/v1",
)
_PREPARED_RECEIPT_CODEC = DataclassCodec(
    _PreparedReceipt,
    PREPARED_SCHEMA,
    nested_types=(Blob, Candidate, Key),
)


def _mapping(
    value: Any,
    *,
    name: str,
    resolve: bool = False,
) -> dict[str, Any]:
    if value is None:
        return {}
    try:
        if OmegaConf.is_config(value):
            value = OmegaConf.to_container(
                value,
                resolve=resolve,
                throw_on_missing=resolve,
            )
    except Exception as error:
        raise UnsupportedPortableNativeProfile(
            f"{name} cannot be resolved exactly"
        ) from error
    if not isinstance(value, Mapping):
        raise UnsupportedPortableNativeProfile(f"{name} must be a mapping")
    return dict(value)


def _sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(value)
    return (value,)


def _empty(value: Any) -> bool:
    return value in (None, "", (), [], {})


def _enabled_hooks(value: Any) -> bool:
    hooks = _mapping(value, name="hooks")
    return any(
        isinstance(hook, Mapping) and bool(hook.get("enabled", False))
        for hook in hooks.values()
    )


def _plain_config(
    value: Any,
    *,
    path: str = "$",
    resolve: bool = False,
) -> Any:
    try:
        if OmegaConf.is_config(value):
            value = OmegaConf.to_container(
                value,
                resolve=resolve,
                throw_on_missing=resolve,
            )
    except Exception as error:
        raise UnsupportedPortableNativeProfile(
            f"configuration at {path} cannot be resolved exactly"
        ) from error
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise UnsupportedPortableNativeProfile(
                f"configuration has a non-finite float at {path}"
            )
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        result = {}
        for name, item in value.items():
            key = str(name)
            result[key] = _plain_config(
                item,
                path=f"{path}.{key}",
                resolve=resolve,
            )
        return result
    if isinstance(value, (tuple, list)):
        return [
            _plain_config(
                item,
                path=f"{path}[{index}]",
                resolve=resolve,
            )
            for index, item in enumerate(value)
        ]
    if isinstance(value, (set, frozenset)):
        return sorted(
            (
                _plain_config(
                    item,
                    path=f"{path}{{set}}",
                    resolve=resolve,
                )
                for item in value
            ),
            key=lambda item: key_for(item).digest,
        )
    enum_value = getattr(value, "value", None)
    enum_name = type(value).__name__
    if enum_name and enum_value is not None and type(enum_value) in {
        str,
        bool,
        int,
    }:
        return {
            "$enum": f"{type(value).__module__}.{type(value).__qualname__}",
            "value": enum_value,
        }
    raise UnsupportedPortableNativeProfile(
        f"configuration value at {path} has no stable representation: "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _resolved_node(value: Any, *, path: str) -> Any:
    return _plain_config(value, path=path, resolve=True)


def _resolved_child(value: Any, *, name: str, path: str) -> Any:
    if value is None:
        return None
    if OmegaConf.is_config(value):
        if name not in value:
            return None
        return _resolved_node(value[name], path=f"{path}.{name}")
    if isinstance(value, Mapping):
        if name not in value:
            return None
        return _resolved_node(value[name], path=f"{path}.{name}")
    return None


def _selected_initializer_payload(args: Any) -> dict[str, Any]:
    initializer = getattr(args, "initializer", None)
    mode = _resolved_child(initializer, name="mode", path="initializer")
    if not isinstance(mode, str) or not mode:
        raise UnsupportedPortableNativeProfile(
            "selected initializer.mode must resolve to a non-empty string"
        )
    mode = mode.strip().lower()
    field_names = {
        "random": (),
        "all_a": (),
        "aa_set": ("aa_set",),
        "mask_mix": ("mask_fraction", "aa_set"),
        "pssm": ("pssm",),
        "x_mix": ("x_fraction", "aa_set"),
        "dplm": (),
        "structure": (),
        "rfd3": (),
    }
    if mode not in field_names:
        raise UnsupportedPortableNativeProfile(
            f"portable native cannot resolve initializer mode {mode!r}"
        )
    payload = {"mode": mode}
    for name in field_names[mode]:
        value = _resolved_child(initializer, name=name, path="initializer")
        if value is not None:
            payload[name] = value
    return payload


def _selected_loss_payload(args: Any, stage: Mapping[str, Any]) -> dict[str, Any]:
    from astrolabe.losses.base import (
        extract_loss_terms,
        extract_variables,
    )

    loss = getattr(args, "loss", None)
    raw_expression_terms = _mapping(
        getattr(loss, "expression_terms", None),
        name="loss.expression_terms",
    )
    raw_loss_terms = _mapping(
        getattr(loss, "loss_terms", None),
        name="loss.loss_terms",
    )
    expressions = [
        stage.get("weighted_score_expression"),
        stage.get("acceptance_gate"),
        stage.get("acceptance_threshold"),
        stage.get("early_stopping"),
        stage.get("prefix_early_stopping"),
    ]
    needed = set()
    for expression in expressions:
        needed.update(
            extract_variables(expression).intersection(raw_expression_terms)
        )
    pending = list(sorted(needed))
    while pending:
        name = pending.pop()
        expression = raw_expression_terms.get(name)
        for dependency in extract_variables(expression).intersection(
            raw_expression_terms
        ):
            if dependency not in needed:
                needed.add(dependency)
                pending.append(dependency)
    expression_payload = {
        name: _resolved_child(
            loss.expression_terms,
            name=name,
            path="loss.expression_terms",
        )
        for name in sorted(needed)
    }
    objective_names = set(
        extract_loss_terms(
            (
                *expressions,
                *expression_payload.values(),
            ),
            excluded_terms=tuple(expression_payload),
        )
    )
    defined_losses = {
        name: _resolved_child(
            loss.loss_terms,
            name=name,
            path="loss.loss_terms",
        )
        for name in raw_loss_terms
    }
    defined_losses = {
        name: value
        for name, value in defined_losses.items()
        if value is not None
    }
    required_loss_names = objective_names | set(defined_losses)
    loss_payload = {
        name: defined_losses[name]
        for name in sorted(required_loss_names)
        if name in defined_losses
    }
    return {
        "expression_terms": expression_payload,
        "loss_terms": loss_payload,
    }


def _selected_component_payload(
    args: Any,
    *,
    family: str,
    alias: str,
) -> dict[str, Any]:
    from .configured_refinement import _component_base_name

    container = getattr(args, family, None)
    terms = getattr(container, f"{family}_terms", None)
    term = _resolved_child(
        terms,
        name=alias,
        path=f"{family}.{family}_terms",
    )
    if not isinstance(term, Mapping):
        raise UnsupportedPortableNativeProfile(
            f"selected {family} term must resolve to a mapping"
        )
    type_name = term.get("_type")
    if not isinstance(type_name, str) or not type_name.strip():
        raise UnsupportedPortableNativeProfile(
            f"selected {family} term is missing _type"
        )
    raw_container = _mapping(container, name=family)
    base_name = _component_base_name(type_name)
    base_key = next(
        (
            name
            for name in raw_container
            if str(name).casefold() == base_name
        ),
        None,
    )
    base = None if base_key is None else container[base_key]
    defaults = {}
    if base is not None:
        resolved = _resolved_node(
            base,
            path=f"{family}.{type_name}",
        )
        if not isinstance(resolved, Mapping):
            raise UnsupportedPortableNativeProfile(
                f"selected {family} defaults must resolve to a mapping"
            )
        defaults = dict(resolved)
    payload = {
        "alias": alias,
        "type": type_name,
        "base_key": base_key,
        "defaults": defaults,
        "term": dict(term),
    }
    if family == "mutator":
        context = {}
        for name in (
            "selector_expression",
            "selectors",
        ):
            value = _resolved_child(container, name=name, path="mutator")
            if value not in (None, "", {}, []):
                context[name] = value
        diagnostics = _resolved_child(
            container,
            name="state_conflict_diagnostics",
            path="mutator",
        )
        if (
            isinstance(diagnostics, Mapping)
            and diagnostics.get("enabled") is True
        ):
            context["state_conflict_diagnostics"] = diagnostics
        payload["context"] = context
    return payload


def _selected_component_config(
    component: Mapping[str, Any],
    *,
    family: str,
) -> Mapping[str, Any]:
    """Compile the one selected component view used by identity and factories."""

    alias = component.get("alias")
    type_name = component.get("type")
    base_key = component.get("base_key")
    defaults = component.get("defaults")
    term = component.get("term")
    context = component.get("context", {})
    if not isinstance(alias, str) or not alias:
        raise UnsupportedPortableNativeProfile(
            f"selected {family} alias must be a non-empty string"
        )
    if not isinstance(type_name, str) or not type_name:
        raise UnsupportedPortableNativeProfile(
            f"selected {family} type must be a non-empty string"
        )
    if base_key is not None and (
        not isinstance(base_key, str) or not base_key
    ):
        raise UnsupportedPortableNativeProfile(
            f"selected {family} base_key must be a non-empty string or None"
        )
    if not isinstance(defaults, Mapping):
        raise UnsupportedPortableNativeProfile(
            f"selected {family} defaults must be a mapping"
        )
    if not isinstance(term, Mapping):
        raise UnsupportedPortableNativeProfile(
            f"selected {family} term must be a mapping"
        )
    if not isinstance(context, Mapping):
        raise UnsupportedPortableNativeProfile(
            f"selected {family} context must be a mapping"
        )
    terms_field = f"{family}_terms"
    defaults_root = base_key or type_name
    selected = {
        defaults_root: dict(defaults),
        terms_field: {alias: dict(term)},
        "enabled_aliases": [alias],
        **dict(context),
    }
    return OmegaConf.create(selected)


def _selected_scientific_payload(args: Any) -> dict[str, Any]:
    stages = getattr(args, "stages", None)
    stage_node = (
        stages.get("default")
        if isinstance(stages, Mapping)
        else getattr(stages, "default", _MISSING)
    )
    if stage_node is _MISSING or stage_node is None:
        raise UnsupportedPortableNativeProfile(
            "portable native requires stages.default"
        )
    stage = _resolved_node(stage_node, path="stages.default")
    if not isinstance(stage, Mapping):
        raise UnsupportedPortableNativeProfile(
            "selected stages.default must resolve to a mapping"
        )
    stage = dict(stage)
    evaluator_alias = stage.get("evaluator")
    if not isinstance(evaluator_alias, str) or not evaluator_alias:
        raise UnsupportedPortableNativeProfile(
            "selected stage evaluator alias is missing"
        )
    mutator_aliases = _sequence(stage.get("mutators"))
    if len(mutator_aliases) != 1 or not isinstance(mutator_aliases[0], str):
        raise UnsupportedPortableNativeProfile(
            "selected stage must contain one mutator alias"
        )
    mutator_alias = mutator_aliases[0]
    context_payload = {}
    for name in _SCIENTIFIC_CONTEXT_FIELDS:
        value = _resolved_child(args, name=name, path="cfg")
        if value not in (None, "", {}, []):
            context_payload[name] = value
    return {
        "stage": stage,
        "initializer": _selected_initializer_payload(args),
        "evaluator": _selected_component_payload(
            args,
            family="evaluator",
            alias=evaluator_alias,
        ),
        "mutator": _selected_component_payload(
            args,
            family="mutator",
            alias=mutator_alias,
        ),
        "loss": _selected_loss_payload(args, stage),
        "context": context_payload,
    }


def _canonical_optional(args: Any, name: str) -> Any:
    value = _resolved_child(args, name=name, path="cfg")
    return None if value in (None, "", "null", "None") else value


def _acceptance_initial_temperature(args: Any) -> float | None:
    if not bool(getattr(args, "use_metropolis_acceptance", False)):
        return None
    value = _resolved_child(args, name="initial_temperature", path="cfg")
    if value is None:
        raise UnsupportedPortableNativeProfile(
            "Metropolis acceptance requires initial_temperature"
        )
    return float(value)


def _fill_inactive_runtime_defaults(args: Any) -> None:
    """Supply process-only defaults for pruned, unselected runtime fields."""

    if not OmegaConf.is_config(args):
        return
    if "obabel_path" not in args:
        with open_dict(args):
            args.obabel_path = None

    evaluator = getattr(args, "evaluator", None)
    terms = getattr(evaluator, "evaluator_terms", None)
    evaluator_alias = _stage_selection(args)[2]
    term = terms.get(evaluator_alias) if isinstance(terms, Mapping) else None
    if not isinstance(term, Mapping):
        return
    evaluator_type = _resolved_child(
        term,
        name="_type",
        path=f"evaluator.evaluator_terms.{evaluator_alias}",
    )
    if not isinstance(evaluator_type, str) or evaluator_type.casefold() != "alphafold3":
        return

    run_data_pipeline = _resolved_child(
        term,
        name="run_data_pipeline",
        path=f"evaluator.evaluator_terms.{evaluator_alias}",
    )
    if run_data_pipeline is not False:
        return

    from .configured_refinement import _matching_component_base

    base = _matching_component_base(evaluator, evaluator_type)
    for value in (term, base):
        if isinstance(value, Mapping) and "datapipeline" in value:
            with open_dict(value):
                del value["datapipeline"]


def _selected_run_payload(
    args: Any,
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    use_metropolis = bool(getattr(args, "use_metropolis_acceptance", False))
    per_residue = _resolved_child(
        args,
        name="per_residue_scores",
        path="cfg",
    )
    if not isinstance(per_residue, Mapping) or not bool(
        per_residue.get("enabled", False)
    ):
        normalized_per_residue = {"enabled": False}
    else:
        normalized_per_residue = dict(per_residue)
    return {
        "seed": _resolved_child(args, name="seed", path="cfg"),
        "steps": _resolved_child(args, name="steps", path="cfg"),
        "exclude_AA": (
            _resolved_child(args, name="exclude_AA", path="cfg") or []
        ),
        "mutation_num_fallback": (
            None
            if "mutation_num" in stage
            else _canonical_optional(args, "mutation_num")
        ),
        "use_metropolis_acceptance": use_metropolis,
        "initial_temperature": (
            _canonical_optional(args, "initial_temperature")
            if use_metropolis
            else None
        ),
        "enable_pyrosetta": bool(
            getattr(args, "enable_pyrosetta", False)
        ),
        "skip_complex_poses": bool(
            getattr(args, "skip_complex_poses", False)
        ),
        "allow_mmcif_pdb_fallback": bool(
            getattr(args, "allow_mmcif_pdb_fallback", False)
        ),
        "per_residue_scores": normalized_per_residue,
    }


def _selected_system_payload(args: Any) -> dict[str, Any]:
    return {
        "seqs": _resolved_child(args, name="seqs", path="cfg") or {},
        "types": _resolved_child(args, name="types", path="cfg") or {},
        "homologies": (
            _resolved_child(args, name="homologies", path="cfg") or {}
        ),
    }


def scientific_config_payload(args: Any) -> dict[str, Any]:
    """Return the closed, parsed scientific contract for this portable run.

    Unselected configuration, persisted-run bookkeeping, output paths, and
    interpolation markers are outside this projection.  Callers pass the
    parsed arguments used by the current construction attempt; mixing an
    initial raw Config with a resume YAML creates another contract source.
    """

    selected = _selected_scientific_payload(args)
    payload = {
        "schema": SCIENTIFIC_CONFIG_SCHEMA,
        "system": _selected_system_payload(args),
        "run": _selected_run_payload(args, selected["stage"]),
        "selected": selected,
    }
    key_for(("astrolabe.portable-native.config/v3", payload))
    return payload


def read_recovery_policy(cfg: Any) -> PortableRecoveryPolicy:
    """Read the required explicit portable recovery contract before execution."""

    raw = _mapping(cfg, name="cfg")
    recovery = _mapping(raw.get("recovery"), name="cfg.recovery")
    if not recovery:
        raise UnsupportedPortableNativeProfile(
            "portable native requires an explicit cfg.recovery mapping"
        )
    unknown = set(recovery) - _RECOVERY_KEYS
    missing = _RECOVERY_KEYS - set(recovery)
    if unknown or missing:
        raise UnsupportedPortableNativeProfile(
            "portable native cfg.recovery must contain exactly mode and resume; "
            f"missing={sorted(missing)!r}, unknown={sorted(unknown)!r}"
        )
    return PortableRecoveryPolicy(
        mode=str(recovery["mode"]),
        resume=recovery["resume"],
    )


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_record(role: str, path: Path) -> ContentRecord:
    selected = path.expanduser().resolve()
    if not selected.exists():
        raise FileNotFoundError(f"selected portable native resource is missing: {selected}")
    if selected.is_file():
        return ContentRecord(
            role=role,
            path=selected.as_posix(),
            kind="file",
            sha256=_file_digest(selected),
            size=selected.stat().st_size,
        )
    if not selected.is_dir():
        raise ValueError(f"unsupported selected resource kind: {selected}")
    entries = []
    for item in sorted(selected.rglob("*")):
        if not item.is_file():
            continue
        relative = item.relative_to(selected).as_posix()
        entries.append((relative, _file_digest(item), item.stat().st_size))
    if not entries:
        raise ValueError(f"selected resource directory contains no files: {selected}")
    aggregate = hashlib.sha256()
    for relative, digest, size in entries:
        aggregate.update(f"{relative}\0{digest}\0{size}\n".encode("utf-8"))
    return ContentRecord(
        role=role,
        path=selected.as_posix(),
        kind="directory",
        sha256=aggregate.hexdigest(),
        size=None,
        entries=tuple(entries),
    )


def _normalize_paths(
    value: Mapping[str, Any] | Sequence[Any] | None,
    *,
    prefix: str,
) -> tuple[tuple[str, Path], ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        items = tuple((str(role), Path(path)) for role, path in value.items())
    else:
        items = tuple(
            (f"{prefix}/{index}", Path(path))
            for index, path in enumerate(value)
        )
    if any(not role for role, _ in items):
        raise ValueError("resource roles must be non-empty")
    roles = [role for role, _ in items]
    if len(set(roles)) != len(roles):
        raise ValueError("resource roles must be unique")
    return items


def content_manifest(
    paths: Mapping[str, Any] | Sequence[Any] | None,
    *,
    prefix: str,
) -> tuple[ContentRecord, ...]:
    return tuple(
        _content_record(role, path)
        for role, path in _normalize_paths(paths, prefix=prefix)
    )


def _add_asset_path(
    selected: dict[str, Path],
    role: str,
    value: Any,
    *,
    base_dir: Path | None = None,
) -> None:
    if value is None:
        return
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() if base_dir is None else base_dir) / path
    path = path.resolve()
    if not path.exists():
        raise UnsupportedPortableNativeProfile(
            f"selected input dependency is missing: {role}={path}"
        )
    previous = selected.get(role)
    if previous is not None and previous != path:
        raise UnsupportedPortableNativeProfile(
            f"selected input dependency role {role!r} has multiple paths"
        )
    selected[role] = path


def _msa_path(value: Any) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.startswith(">"):
        return None
    file_text, *suffix = text.split(":", 1)
    lower = file_text.casefold()
    if suffix:
        if not lower.endswith(".json"):
            raise UnsupportedPortableNativeProfile(
                f"unsupported selected homology file syntax: {text!r}"
            )
        return Path(file_text)
    if lower.endswith((".a3m", ".json")):
        return Path(file_text)
    raise UnsupportedPortableNativeProfile(
        f"unsupported selected homology file format: {text!r}"
    )


def _framework_source_assets(
    value: Any,
    *,
    role: str,
    selected: dict[str, Path],
    base_dir: Path | None = None,
) -> None:
    if value is None:
        return
    if isinstance(value, (list, tuple)):
        return
    if isinstance(value, Mapping):
        for name in ("unpairedMsa", "pairedMsa"):
            if name not in value:
                continue
            path = _msa_path(value[name])
            if path is not None:
                _add_asset_path(
                    selected,
                    f"{role}/{name}",
                    path,
                    base_dir=base_dir,
                )
        return
    path = _msa_path(value)
    if path is None:
        return
    resolved = path.expanduser()
    if not resolved.is_absolute():
        resolved = (Path.cwd() if base_dir is None else base_dir) / resolved
    resolved = resolved.resolve()
    _add_asset_path(selected, role, resolved)
    if resolved.suffix.casefold() in {".yaml", ".yml", ".json"}:
        try:
            loaded = OmegaConf.load(resolved)
        except Exception as error:
            raise UnsupportedPortableNativeProfile(
                f"selected framework homology source cannot be parsed: {resolved}"
            ) from error
        _framework_source_assets(
            loaded,
            role=role,
            selected=selected,
            base_dir=resolved.parent,
        )


def _homology_assets(
    value: Any,
    *,
    role: str,
    selected: dict[str, Path],
) -> None:
    if value is None:
        return
    if isinstance(value, str):
        path = _msa_path(value)
        if path is not None:
            _add_asset_path(selected, role, path)
        return
    if isinstance(value, (list, tuple)):
        return
    if not isinstance(value, Mapping):
        raise UnsupportedPortableNativeProfile(
            f"selected homology value has no supported representation: {role}"
        )
    if str(value.get("_type", "")).strip() == "antibody_framework_homology":
        _framework_source_assets(
            value.get("source"),
            role=f"{role}/framework-source",
            selected=selected,
        )
        return
    for name in ("unpairedMsa", "pairedMsa"):
        if name not in value:
            continue
        path = _msa_path(value[name])
        if path is not None:
            _add_asset_path(selected, f"{role}/{name}", path)
    # Embedded template mmCIF text is already part of the selected config.


def _selected_evaluator_assets(
    selected_config: Mapping[str, Any],
) -> Mapping[str, Any]:
    assets: dict[str, Any] = {}

    def visit(value: Any, *, path: str, parent: str | None = None) -> None:
        if isinstance(value, Mapping):
            normalized_parent = "" if parent is None else parent.casefold()
            for raw_name, child in value.items():
                name = str(raw_name)
                normalized = name.casefold()
                child_path = f"{path}.{name}"
                if normalized in {
                    "constraint_path",
                    "msa_directory",
                    "msa_path",
                    "msa_paths",
                    "msas_dir",
                    "restraint_path",
                    "template_files",
                    "template_path",
                } or (
                    normalized == "path"
                    and (
                        "initial_guess" in child_path.casefold()
                        or normalized_parent == "references"
                    )
                ):
                    assets[child_path] = child
                else:
                    visit(
                        child,
                        path=child_path,
                        parent=name,
                    )
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(
                    child,
                    path=f"{path}[{index}]",
                    parent=parent,
                )

    visit(selected_config, path="evaluator")
    return assets


def _loss_file_field(name: str) -> bool:
    normalized = name.casefold()
    if normalized in {
        "binder_unbound_structure",
        "bindcraft_binder_alone_pdb",
        "bindcraft_starting_pdb",
        "bindcraft_trajectory_pdb",
        "hole_radius_file",
        "reference_complex",
        "reference_structure",
        "template_files",
        "tmscore_bin",
        "tmscore_cpp",
    }:
        return True
    return normalized.endswith(
        (
            "_bin",
            "_complex",
            "_cpp",
            "_dir",
            "_file",
            "_files",
            "_path",
            "_pdb",
            "_structure",
        )
    )


_LOSS_TOOL_KEYS = frozenset(
    {
        "usalign_path",
        "dssp_path",
        "ananas_path",
    }
)
_LOSS_TOOL_ENV = {
    "usalign_path": "GH_USALIGN",
    "dssp_path": "GH_DSSP",
    "ananas_path": "GH_ANANAS",
}
_LOSS_TOOL_DEFAULTS = {
    "usalign_path": ("USalign", "usalign"),
    "dssp_path": ("dssp", "mkdssp"),
    "ananas_path": ("AnAnaS", "ananas", "ananas-linux"),
}


def _active_loss_tool_keys(args: Any) -> tuple[str, ...]:
    """Infer executable requirements with the shared runtime-requirements logic."""

    from astrolabe.losses.base import (
        extract_best_namespace_terms,
        extract_loss_terms,
    )
    from astrolabe.runtime_requirements import task_requirements

    selected = _selected_scientific_payload(args)
    stage = selected["stage"]
    selected_loss = selected["loss"]
    expressions = [
        stage.get("weighted_score_expression"),
        stage.get("acceptance_gate"),
        stage.get("acceptance_threshold"),
        stage.get("early_stopping"),
        stage.get("prefix_early_stopping"),
        *selected_loss["expression_terms"].values(),
    ]
    referenced = set(extract_loss_terms(expressions))
    referenced.update(
        extract_best_namespace_terms(stage.get("prefix_early_stopping"))
    )
    requirement_terms = dict(selected_loss["loss_terms"])
    for name in referenced:
        requirement_terms.setdefault(name, {"_type": name})
    requirements = task_requirements(
        {
            "enable_pyrosetta": False,
            "post_eval_hooks": {},
            "initializer": {"mode": "random"},
            "evaluator": {"evaluator_terms": {}},
            "mutator": {"mutator_terms": {}},
            "predocker": {"predocker_terms": {}},
            "loss": {
                "loss_terms": requirement_terms,
                "expression_terms": selected_loss["expression_terms"],
            },
            "stages": {"default": stage},
        }
    )
    return tuple(
        sorted(
            set(requirements["required_env_keys"]).intersection(
                _LOSS_TOOL_KEYS
            )
        )
    )


def _resolve_loss_tool(args: Any, key: str) -> Path:
    raw = _resolved_child(args, name=key, path="cfg")
    if raw in (None, "", "None", "null"):
        raw = os.environ.get(_LOSS_TOOL_ENV[key])
    elif not isinstance(raw, str):
        raw = str(raw)

    candidates: list[Path] = []
    if raw not in (None, "", "None", "null"):
        text = os.path.expandvars(str(raw).strip())
        expanded = Path(os.path.expanduser(text))
        if expanded.is_absolute():
            candidates.append(expanded)
        else:
            candidates.append(Path.cwd() / expanded)
            candidates.append(Path(__file__).resolve().parents[2] / expanded)
        found = shutil.which(text)
        if found is not None:
            candidates.append(Path(found))
    else:
        for name in _LOSS_TOOL_DEFAULTS[key]:
            found = shutil.which(name)
            if found is not None:
                candidates.append(Path(found))

    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_file() and os.access(resolved, os.X_OK):
            return resolved
    raise UnsupportedPortableNativeProfile(
        f"active portable loss requires unbound executable {key}; "
        f"set {key} or {_LOSS_TOOL_ENV[key]}, or place "
        f"{' or '.join(_LOSS_TOOL_DEFAULTS[key])} on PATH"
    )


def _asset_value_paths(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        return tuple(value.values())
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


def _selected_loss_assets(args: Any) -> Mapping[str, Any]:
    from astrolabe.losses.base import loss_registry

    selected = _selected_scientific_payload(args)
    assets: dict[str, Any] = {}
    for loss_name, term in selected["loss"]["loss_terms"].items():
        if not isinstance(term, Mapping):
            continue
        type_name = str(term.get("_type", loss_name) or loss_name)
        function = loss_registry.get_function(type_name)
        metadata = loss_registry.get_metadata(type_name)
        allowed = set(metadata.required_params or ()) | set(
            metadata.optional_params or ()
        )
        for parameter, value in term.items():
            name = str(parameter)
            if name not in allowed or not _loss_file_field(name):
                continue
            for index, item in enumerate(_asset_value_paths(value)):
                role = f"loss/{loss_name}/{name}"
                if isinstance(value, (list, tuple, Mapping)):
                    role = f"{role}/{index}"
                assets[role] = item
        # Fail early if a declared file-bearing parameter cannot resolve to
        # the registered loss implementation.
        if not callable(function):
            raise UnsupportedPortableNativeProfile(
                f"selected loss implementation is not callable: {type_name}"
            )
    for key in _active_loss_tool_keys(args):
        assets[f"loss-tools/{key}"] = _resolve_loss_tool(args, key)
    return assets


def _supplemental_asset_paths(
    args: Any,
    initializer_result: Any,
) -> Mapping[str, Path]:
    selected: dict[str, Path] = {}
    named_files = getattr(args, "named_files", None)
    if isinstance(named_files, Mapping):
        for role, value in named_files.items():
            path = Path(str(value)).expanduser()
            if path.is_file():
                selected[f"named-files/{role}"] = path

    structure_path = getattr(initializer_result, "structure_path", None)
    if structure_path is not None:
        path = Path(str(structure_path)).expanduser()
        if path.is_file():
            selected["initializer/structure"] = path
    manifest_path = getattr(initializer_result, "manifest_path", None)
    if manifest_path is not None:
        path = Path(str(manifest_path)).expanduser()
        if path.is_file():
            selected["initializer/manifest"] = path

    pssm_node = getattr(getattr(args, "initializer", None), "pssm", None)
    pssm = _mapping(
        pssm_node,
        name="pssm",
        resolve=True,
    )
    for role, value in pssm.items():
        if role not in {"path", "file"} or value in (None, ""):
            continue
        _add_asset_path(
            selected,
            f"initializer/pssm/{role}",
            value,
        )
    user_ccd = getattr(args, "user_ccd", None)
    if isinstance(user_ccd, str):
        value = user_ccd.strip()
        if (
            "\n" not in value
            and not value.startswith("data_")
            and Path(value).expanduser().is_file()
        ):
            _add_asset_path(
                selected,
                "context/user-ccd",
                value,
            )
    homologies = _resolved_child(args, name="homologies", path="cfg")
    if isinstance(homologies, Mapping):
        for protomer, value in homologies.items():
            _homology_assets(
                value,
                role=f"homologies/{protomer}",
                selected=selected,
            )
    selected_scientific = _selected_scientific_payload(args)
    for role, value in _selected_evaluator_assets(
        selected_scientific["evaluator"]
    ).items():
        for index, item in enumerate(_asset_value_paths(value)):
            item_role = role if not isinstance(value, (list, tuple, Mapping)) else f"{role}/{index}"
            _add_asset_path(selected, item_role, item)
    for role, value in _selected_loss_assets(args).items():
        _add_asset_path(selected, role, value)
    return selected


def _asset_projection(
    value: Any,
    *,
    path: str,
    directory_keys: frozenset[str],
    file_keys: frozenset[str],
) -> dict[str, Any]:
    """Resolve only fields consumed by configured-refinement's asset helper."""

    result: dict[str, Any] = {}
    if not isinstance(value, Mapping):
        return result
    entries = (
        value.items_ex(resolve=False)
        if OmegaConf.is_config(value)
        else value.items()
    )
    for raw_name, child in entries:
        name = str(raw_name)
        normalized = name.casefold()
        child_path = f"{path}.{name}"
        if (
            normalized in directory_keys
            or normalized in file_keys
            or normalized.startswith("checkpoint_")
            or normalized in {"model_name", "experiment_path"}
        ):
            resolved = _resolved_node(child, path=child_path)
            if resolved not in (None, "", "null", "None"):
                result[normalized] = resolved
        elif isinstance(child, Mapping):
            result.update(
                _asset_projection(
                    child,
                    path=child_path,
                    directory_keys=directory_keys,
                    file_keys=file_keys,
                )
            )
    return result


def _selected_portable_assets(args: Any):
    """Reuse configured refinement's selected-asset hash implementation."""

    from .configured_refinement import (
        _ASSET_DIRECTORY_KEYS,
        _ASSET_FILE_KEYS,
        _asset_manifest_identity,
        _asset_manifest_record,
        _matching_component_base,
        _selected_asset_manifest,
    )

    selected = _selected_scientific_payload(args)
    evaluator_alias = selected["evaluator"]["alias"]
    mutator_alias = selected["mutator"]["alias"]
    evaluator_config = getattr(args, "evaluator", None)
    mutator_config = getattr(args, "mutator", None)
    evaluator_term = evaluator_config.evaluator_terms[evaluator_alias]
    mutator_term = mutator_config.mutator_terms[mutator_alias]
    evaluator_type = _resolved_child(
        evaluator_term,
        name="_type",
        path=f"evaluator.evaluator_terms.{evaluator_alias}",
    )
    mutator_type = _resolved_child(
        mutator_term,
        name="_type",
        path=f"mutator.mutator_terms.{mutator_alias}",
    )
    if not isinstance(evaluator_type, str) or not evaluator_type:
        raise UnsupportedPortableNativeProfile(
            "selected evaluator _type is missing"
        )
    if not isinstance(mutator_type, str) or not mutator_type:
        raise UnsupportedPortableNativeProfile(
            "selected mutator _type is missing"
        )
    evaluator_fields = _asset_projection(
        _matching_component_base(evaluator_config, evaluator_type),
        path=f"evaluator.{evaluator_type}",
        directory_keys=_ASSET_DIRECTORY_KEYS,
        file_keys=_ASSET_FILE_KEYS,
    )
    evaluator_fields.update(
        _asset_projection(
            evaluator_term,
            path=f"evaluator.evaluator_terms.{evaluator_alias}",
            directory_keys=_ASSET_DIRECTORY_KEYS,
            file_keys=_ASSET_FILE_KEYS,
        )
    )
    mutator_fields = _asset_projection(
        _matching_component_base(mutator_config, mutator_type),
        path=f"mutator.{mutator_type}",
        directory_keys=_ASSET_DIRECTORY_KEYS,
        file_keys=_ASSET_FILE_KEYS,
    )
    mutator_fields.update(
        _asset_projection(
            mutator_term,
            path=f"mutator.mutator_terms.{mutator_alias}",
            directory_keys=_ASSET_DIRECTORY_KEYS,
            file_keys=_ASSET_FILE_KEYS,
        )
    )
    asset_view = OmegaConf.create(
        {
            "stages": {"default": selected["stage"]},
            "evaluator": {
                "enabled_aliases": [evaluator_alias],
                "evaluator_terms": {
                    evaluator_alias: {"_type": evaluator_type}
                },
                evaluator_type: {
                    "_type": evaluator_type,
                    **evaluator_fields,
                },
            },
            "mutator": {
                "enabled_aliases": [mutator_alias],
                "mutator_terms": {
                    mutator_alias: {"_type": mutator_type}
                },
                mutator_type: {
                    "_type": mutator_type,
                    **mutator_fields,
                },
            },
            "initializer": selected["initializer"],
            "loss": selected["loss"],
        }
    )
    manifest = _selected_asset_manifest(asset_view)
    if not manifest.complete:
        missing = [
            item.role
            for item in manifest.components
            if item.state == "unsupported"
        ]
        raise UnsupportedPortableNativeProfile(
            "selected native components have no content-verifiable assets: "
            + ", ".join(missing)
        )
    return (
        manifest,
        _asset_manifest_record(manifest),
        _asset_manifest_identity(manifest),
    )


def _resume_paths(root: Path) -> tuple[Path, ...]:
    return (
        root / "config.yaml",
        root / "provenance.json",
        root / ".framework" / "journal.sqlite",
        root / ".framework" / "portable-native" / "prepared.json",
    )


def _validate_resume_root(root: Path) -> None:
    if not root.is_dir():
        raise FileNotFoundError(f"portable native resume root does not exist: {root}")
    missing = [path for path in _resume_paths(root) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "portable native resume requires prior durable run files: "
            + ", ".join(str(path) for path in missing)
        )


def _initializer_mode(args: Any, run_module: Any) -> str:
    resolver = getattr(run_module, "_resolve_initializer_mode", None)
    if callable(resolver):
        mode = resolver(args)
    else:
        initializer = getattr(args, "initializer", None)
        mode = getattr(initializer, "mode", None)
        if mode is None and isinstance(initializer, Mapping):
            mode = initializer.get("mode")
    if not isinstance(mode, str) or not mode:
        raise UnsupportedPortableNativeProfile(
            "portable native requires an explicit initializer.mode"
        )
    return mode.strip().lower()


def _stage_selection(args: Any) -> tuple[str, tuple[str, ...], str, tuple[str, ...]]:
    stages = getattr(args, "stages", None)
    active = {
        str(name): value
        for name, value in _mapping(stages, name="stages").items()
        if value is not None
    }
    if set(active) != {"default"}:
        raise UnsupportedPortableNativeProfile(
            "portable native supports exactly one static stage named 'default'; "
            f"got {sorted(active)!r}"
        )
    stage_node = (
        stages["default"]
        if OmegaConf.is_config(stages)
        else active["default"]
    )
    stage = _mapping(
        _resolved_node(stage_node, path="stages.default"),
        name="stages.default",
    )
    unsupported = [
        name
        for name in _UNSUPPORTED_STAGE_FIELDS
        if not _empty(stage.get(name))
    ]
    if unsupported:
        raise UnsupportedPortableNativeProfile(
            "portable native cannot preserve stage transition controls "
            f"{unsupported!r}"
        )
    mutators = tuple(str(item) for item in _sequence(stage.get("mutators")))
    if len(mutators) != 1 or not mutators[0]:
        raise UnsupportedPortableNativeProfile(
            "portable native requires exactly one native mutator alias"
        )
    evaluator = stage.get("evaluator")
    if not isinstance(evaluator, str) or not evaluator:
        raise UnsupportedPortableNativeProfile(
            "portable native requires exactly one explicit stage evaluator alias"
        )
    assemblies = tuple(str(item) for item in _sequence(stage.get("oligo")))
    if not assemblies or len(set(assemblies)) != len(assemblies):
        raise UnsupportedPortableNativeProfile(
            "portable native stage oligo must contain unique assembly names"
        )
    return "default", mutators, evaluator, assemblies


def _validate_loss_outputs(args: Any, run_module: Any) -> None:
    try:
        import astrolabe.losses  # noqa: F401 - registers configured losses
        from astrolabe.losses.base import (
            extract_best_namespace_terms,
            extract_loss_terms,
            loss_registry,
        )
    except Exception as error:
        raise UnsupportedPortableNativeProfile(
            "portable native cannot preflight configured loss outputs"
        ) from error

    selected = _selected_scientific_payload(args)
    stage = selected["stage"]
    selected_loss = selected["loss"]
    expressions = [
        stage.get("weighted_score_expression"),
        stage.get("acceptance_gate"),
        stage.get("acceptance_threshold"),
        stage.get("early_stopping"),
        stage.get("prefix_early_stopping"),
        *selected_loss["expression_terms"].values(),
    ]
    referenced = set(extract_loss_terms(expressions))
    referenced.update(
        extract_best_namespace_terms(stage.get("prefix_early_stopping"))
    )
    defined = selected_loss["loss_terms"]
    pose_types = set(getattr(run_module, "PYROSETTA_LOSS_TYPES", ()) or ())
    hook_types = set(getattr(run_module, "POST_EVAL_HOOK_LOSS_TYPES", ()) or ())
    for name in sorted(referenced | set(defined)):
        configured = _mapping(defined.get(name), name=f"loss.loss_terms.{name}")
        type_name = str(configured.get("_type", name) or name).strip()
        if type_name not in loss_registry.valid_losses:
            if name in referenced:
                raise UnsupportedPortableNativeProfile(
                    f"portable native cannot resolve active loss {name!r}"
                )
            continue
        function = loss_registry.get_function(type_name)
        code = getattr(function, "__code__", None)
        names = set(getattr(code, "co_names", ()))
        if (
            type_name in pose_types
            or type_name in hook_types
            or "_require_pyrosetta" in names
            or "complex_poses" in names
        ):
            raise UnsupportedPortableNativeProfile(
                f"portable native cannot persist live Pose loss output {name!r}"
            )


def validate_portable_native_profile(
    args: Any,
    run_module: Any | None = None,
) -> None:
    """Reject unsupported science before model/component construction."""

    if run_module is None:
        run_module = importlib.import_module("run")
    _stage_selection(args)

    repeats = getattr(args, "repeats", 1)
    if type(repeats) is not int or repeats != 1:
        raise UnsupportedPortableNativeProfile(
            "portable native requires repeats=1; materialize repeat seeds before build"
        )
    repeat_plan = _mapping(getattr(args, "repeat_plan", None), name="repeat_plan")
    repeat_count = repeat_plan.get("count", 1)
    if type(repeat_count) is not int or repeat_count != 1:
        raise UnsupportedPortableNativeProfile(
            "portable native requires repeat_plan.count=1"
        )
    search = getattr(args, "search", None)
    if search not in (None, {}) and _mapping(search, name="search"):
        raise UnsupportedPortableNativeProfile(
            "portable native owns its coordinator route and rejects other search controls"
        )

    if not bool(getattr(args, "skip_complex_poses", False)):
        raise UnsupportedPortableNativeProfile(
            "portable native requires skip_complex_poses=true; live Pose output "
            "has no durable codec"
        )
    if bool(getattr(args, "enable_pyrosetta", False)):
        raise UnsupportedPortableNativeProfile(
            "portable native does not support PyRosetta Pose state"
        )
    if getattr(args, "structure_start", None) is not None:
        raise UnsupportedPortableNativeProfile(
            "portable native does not support structure_start/reference transactions"
        )

    mutator = getattr(args, "mutator", None)
    if bool(getattr(mutator, "stack_pre_update", False)):
        raise UnsupportedPortableNativeProfile(
            "portable native does not support mutator.stack_pre_update"
        )
    if getattr(mutator, "stack_pre_update_max_entropy", None) not in (
        None,
        "",
        "null",
        "None",
    ):
        raise UnsupportedPortableNativeProfile(
            "portable native does not support stack_pre_update_max_entropy"
        )
    if _enabled_hooks(getattr(args, "pre_eval_hooks", None)):
        raise UnsupportedPortableNativeProfile(
            "portable native rejects enabled pre-evaluation hooks"
        )
    if _enabled_hooks(getattr(args, "post_eval_hooks", None)):
        raise UnsupportedPortableNativeProfile(
            "portable native rejects enabled post-evaluation hooks"
        )

    mode = _initializer_mode(args, run_module)
    if mode in _MODEL_INITIALIZERS:
        raise UnsupportedPortableNativeProfile(
            f"portable native has no durable paid-initializer receipt for mode={mode!r}"
        )
    try:
        from astrolabe.initializers.registry import InitializerRegistry

        registration = InitializerRegistry.get(mode)
    except Exception as error:
        raise UnsupportedPortableNativeProfile(
            f"portable native cannot validate initializer.mode={mode!r}"
        ) from error
    if registration.requires_mutators:
        raise UnsupportedPortableNativeProfile(
            f"portable native initializer.mode={mode!r} requires model-owned mutators"
        )
    if bool(getattr(args, "reinitialize_random_segments_with_dplm", False)):
        raise UnsupportedPortableNativeProfile(
            "portable native rejects legacy DPLM reinitialization"
        )

    from astrolabe.per_residue_scores import (
        per_residue_scores_requires_complex_poses,
    )

    if per_residue_scores_requires_complex_poses(
        getattr(args, "per_residue_scores", None)
    ):
        raise UnsupportedPortableNativeProfile(
            "portable native cannot persist live per-residue Pose scores"
        )
    _validate_loss_outputs(args, run_module)


def _python_tree_record(role: str, root: Path) -> ContentRecord:
    """Content-identify one repository's Python source tree."""

    selected = root.expanduser().resolve()
    if not selected.is_dir():
        raise FileNotFoundError(f"selected Python source tree is missing: {selected}")
    entries = []
    for item in sorted(selected.rglob("*.py")):
        if not item.is_file():
            continue
        relative = item.relative_to(selected).as_posix()
        entries.append((relative, _file_digest(item), item.stat().st_size))
    if not entries:
        raise ValueError(f"selected Python source tree contains no files: {selected}")
    aggregate = hashlib.sha256()
    for relative, digest, size in entries:
        aggregate.update(f"{relative}\0{digest}\0{size}\n".encode("utf-8"))
    return ContentRecord(
        role=role,
        path=selected.as_posix(),
        kind="python-tree",
        sha256=aggregate.hexdigest(),
        size=None,
        entries=tuple(entries),
    )


def _runtime_package_records(
    role: str,
    module: str,
) -> tuple[ContentRecord, ...]:
    """Content-identify the package selected by the active runtime interpreter."""

    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ModuleNotFoundError, ValueError) as error:
        raise UnsupportedPortableNativeProfile(
            f"runtime {module} source package cannot be resolved"
        ) from error
    locations = (
        tuple(
            Path(item).expanduser().resolve()
            for item in (spec.submodule_search_locations or ())
        )
        if spec is not None
        else ()
    )
    if not locations:
        raise UnsupportedPortableNativeProfile(
            f"runtime {module} source package has no directory origin"
        )

    records = []
    for index, location in enumerate(locations):
        if not location.is_dir():
            raise UnsupportedPortableNativeProfile(
                f"runtime {module} source directory is unavailable: {location}"
            )
        entries = tuple(
            (
                item.relative_to(location).as_posix(),
                _file_digest(item),
                item.stat().st_size,
            )
            for item in sorted(location.rglob("*"))
            if item.is_file()
            and "__pycache__" not in item.parts
            and item.suffix not in {".pyc", ".pyo"}
        )
        if not entries:
            raise UnsupportedPortableNativeProfile(
                f"runtime {module} source package contains no files: {location}"
            )
        aggregate = hashlib.sha256()
        for relative, digest, size in entries:
            aggregate.update(f"{relative}\0{digest}\0{size}\n".encode("utf-8"))
        item_role = role if len(locations) == 1 else f"{role}/{index}"
        records.append(
            ContentRecord(
                role=item_role,
                path=location.as_posix(),
                kind="python-package",
                sha256=aggregate.hexdigest(),
                size=None,
                entries=entries,
            )
        )
    return tuple(records)


def _source_values(value: Any, *, path: str):
    if not isinstance(value, Mapping):
        return
    for raw_name, child in value.items():
        name = str(raw_name)
        child_path = f"{path}.{name}"
        normalized = name.casefold()
        if normalized in {"repo_path", "source_path", "source_root", "package_path"}:
            yield child_path, child
        elif normalized in {"repo_paths"}:
            if isinstance(child, Mapping):
                for item_name, item in child.items():
                    yield f"{child_path}.{item_name}", item
            else:
                yield child_path, child
        elif isinstance(child, Mapping):
            yield from _source_values(child, path=child_path)


def _selected_third_party_sources(args: Any) -> Mapping[str, Any]:
    selected = _selected_scientific_payload(args)
    sources: dict[str, Any] = {}
    for family in ("evaluator", "mutator"):
        component = selected[family]
        for name, value in _source_values(
            component,
            path=f"selected.{family}",
        ):
            sources[f"selected/{family}/{name}"] = value
    return sources


def source_manifest(
    run_module: Any,
    extra_paths: Mapping[str, Any] | Sequence[Any] | None = None,
    *,
    args: Any | None = None,
) -> tuple[ContentRecord, ...]:
    run_path = inspect.getsourcefile(run_module) or inspect.getfile(run_module)
    if not run_path:
        raise ValueError("run module source is unavailable")
    run_path = Path(run_path).resolve()
    selected = [
        _content_record("application/run", run_path),
        _python_tree_record(
            "application/astrolabe",
            run_path.parent / "astrolabe",
        ),
    ]
    if args is not None:
        selected.extend(
            content_manifest(
                _selected_third_party_sources(args),
                prefix="selected-third-party",
            )
        )
        evaluator_type = _selected_scientific_payload(args)["evaluator"]["type"]
        if evaluator_type.casefold() == "alphafold3":
            selected.extend(
                _runtime_package_records(
                    "runtime/alphafold3",
                    "alphafold3",
                )
            )
    selected.extend(content_manifest(extra_paths, prefix="source"))
    roles = [record.role for record in selected]
    if len(set(roles)) != len(roles):
        raise ValueError("source manifest roles must be unique")
    return tuple(selected)


def _prepare_manifest_path(root: Path) -> Path:
    return root / ".framework" / "portable-native" / "prepared.json"


def _blob_record(blob):
    return {"identity": blob.identity.digest, "codec": blob.codec}


def _blob_from_record(record):
    from .types import Blob

    return Blob(Key(record["identity"]), record["codec"])


def _write_prepared_manifest(
    root: Path,
    receipt_blob,
) -> None:
    path = _prepare_manifest_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": PREPARED_MANIFEST_SCHEMA,
        "receipt": _blob_record(receipt_blob),
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _ensure_journal_exists(root: Path) -> None:
    journal = root / ".framework" / "journal.sqlite"
    journal.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(journal))
    connection.close()


def _read_prepared_receipt(data: DataStore, root: Path) -> _PreparedReceipt:
    path = _prepare_manifest_path(root)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractConflict(
            f"portable native prepared receipt is unreadable: {path}"
        ) from error
    if payload.get("schema") != PREPARED_MANIFEST_SCHEMA:
        raise ContractConflict("portable native prepared receipt has another schema")
    record = payload.get("receipt")
    if not isinstance(record, dict) or set(record) != {"identity", "codec"}:
        raise ContractConflict("portable native prepared receipt pointer is invalid")
    blob = _blob_from_record(record)
    if blob.codec != _PREPARED_RECEIPT_CODEC.identity:
        raise ContractConflict("portable native prepared receipt codec changed")
    receipt = data.get(blob, _PREPARED_RECEIPT_CODEC)
    if not isinstance(receipt, _PreparedReceipt):
        raise ContractConflict("portable native prepared receipt decoded incorrectly")
    return receipt


def _config_aliases(args: Any) -> tuple[str, str]:
    _, mutators, evaluator, _ = _stage_selection(args)
    return evaluator, mutators[0]


def _component_type(args: Any, *, family: str, alias: str) -> str:
    if family == "evaluator":
        terms = getattr(args.evaluator, "evaluator_terms", None)
    else:
        terms = getattr(args.mutator, "mutator_terms", None)
    if terms is None or alias not in terms:
        raise UnsupportedPortableNativeProfile(
            f"selected {family} alias {alias!r} has no mapping"
        )
    type_name = _resolved_child(
        terms[alias],
        name="_type",
        path=f"{family}_terms.{alias}",
    )
    if not isinstance(type_name, str) or not type_name:
        raise UnsupportedPortableNativeProfile(
            f"selected {family} alias {alias!r} has no _type"
        )
    return type_name


def _validate_hooks(
    hooks: Mapping[str, NativeOperationHooks],
    *,
    evaluator_alias: str,
    mutator_alias: str,
) -> dict[str, NativeOperationHooks]:
    if not isinstance(hooks, Mapping):
        raise TypeError("operation_hooks must be a mapping")
    selected = {evaluator_alias.casefold(), mutator_alias.casefold()}
    normalized = {}
    for alias, hook in hooks.items():
        key = str(alias).casefold()
        if key not in selected:
            raise UnsupportedPortableNativeProfile(
                f"operation hook alias {alias!r} is outside the selected static stage"
            )
        if not isinstance(hook, NativeOperationHooks):
            raise TypeError(
                f"operation hook for {alias!r} must be NativeOperationHooks"
            )
        normalized[key] = hook
    return normalized


def _sampling_policy(args: Any, data: DataStore, *, stage: Mapping[str, Any]) -> SamplingPolicy:
    mutation_num = stage.get("mutation_num", getattr(args, "mutation_num", None))
    if type(mutation_num) is not int or mutation_num < 0:
        raise UnsupportedPortableNativeProfile(
            "portable native stage.mutation_num must be a non-negative integer"
        )
    options = {
        "mutation_num": mutation_num,
        "seed": int(args.seed),
    }
    selector = _resolved_child(
        getattr(args, "mutator", None),
        name="selector_expression",
        path="mutator",
    )
    if selector not in (None, ""):
        options["selector_expression"] = str(selector)
    return SamplingPolicy(
        identity=key_for(("astrolabe.portable-native.sampling/v1", options)),
        options=data.put(options, _JSON),
    )


def _runtime_context(
    args: Any,
    data: DataStore,
) -> tuple[Any, dict[str, Any]]:
    mutator = getattr(args, "mutator", None)
    selector_payload = {
        "selector_expression": _resolved_child(
            mutator,
            name="selector_expression",
            path="mutator",
        ),
        "selectors": _resolved_child(
            mutator,
            name="selectors",
            path="mutator",
        ),
        "state_conflict_diagnostics": _resolved_child(
            mutator,
            name="state_conflict_diagnostics",
            path="mutator",
        ),
    }
    decision_payload = {
        "initial_temperature": _acceptance_initial_temperature(args),
        "use_metropolis_acceptance": bool(args.use_metropolis_acceptance),
    }
    context = NativeRuntimeContext(
        schema="astrolabe.portable-native.runtime-context/v1",
        selector_and_contribution_context=store_numeric_tree(data, selector_payload),
        decision_context=store_numeric_tree(data, decision_payload),
    )
    blob = data.put(context, NATIVE_RUNTIME_CONTEXT_CODEC)
    runtime_config = {
        "mutator_stack_pre_update": False,
        "state_conflict_diagnostics": getattr(
            mutator, "state_conflict_diagnostics", None
        ),
    }
    return blob, runtime_config


def _check_loaded_stage(
    stage_manager: Any,
    *,
    evaluator_alias: str,
    mutator_alias: str,
    assemblies: tuple[str, ...],
) -> None:
    loaded_mutators = tuple(
        str(alias)
        for alias in (
            getattr(stage_manager.current_setting, "mutators", None) or ()
        )
    )
    if loaded_mutators != (mutator_alias,):
        raise UnsupportedPortableNativeProfile(
            "loaded pipeline mutators differ from the static stage"
        )
    if tuple(stage_manager.current_oligo or ()) != assemblies:
        raise UnsupportedPortableNativeProfile(
            "loaded pipeline assemblies differ from the static stage"
        )
    selected_evaluator = stage_manager.resolve_stage_evaluator(
        stage_manager.execution_setting
    )
    if str(selected_evaluator or "") != evaluator_alias:
        raise UnsupportedPortableNativeProfile(
            "loaded pipeline evaluator differs from the static stage"
        )


def _rng_snapshot() -> Any:
    from astrolabe.model_execution import capture_random_state

    return capture_random_state()


def _restore_rng(snapshot: Any) -> None:
    from astrolabe.model_execution import restore_random_state

    restore_random_state(snapshot)


def _rng_equal(left: Any, right: Any) -> bool:
    return key_for(left) == key_for(right)


@dataclass(frozen=True)
class PortableNativeRecipe:
    """A fully cold, rooted portable-native composition."""

    schema: str
    recovery: PortableRecoveryPolicy
    args: Any
    prepared: Any
    runtime: Runtime
    flow: PortableNativeFlow
    contract: ScientificRunContract
    source_manifest: tuple[ContentRecord, ...]
    asset_manifest: Mapping[str, Any]
    supplemental_asset_manifest: tuple[ContentRecord, ...]
    selected_evaluator_alias: str
    selected_mutator_alias: str
    assemblies: tuple[str, ...]
    rounds: int

    def run(self):
        """Run through the single common coordinator and accepted-successor policy."""

        return run_portable_native(
            self.flow,
            rounds=self.rounds,
            runtime=self.runtime,
        )

    def close(self) -> None:
        self.runtime.close()


def build_portable_native_recipe(
    cfg: Any,
    *,
    math_factory: PortableNativeMathFactory | None = None,
    operation_hooks: Mapping[str, NativeOperationHooks] | None = None,
    asset_paths: Mapping[str, Any] | Sequence[Any] | None = None,
    source_paths: Mapping[str, Any] | Sequence[Any] | None = None,
    options_schemas: Mapping[str, Mapping[str, Any]] | None = None,
    run_module: Any | None = None,
) -> PortableNativeRecipe:
    """Prepare and cold-build a portable composition without model inference.

    ``math_factory`` defaults to the adapter that delegates to the current
    ``Decisions`` helpers in ``run.py``.  The builder intentionally has no
    second/local Metropolis or gate implementation.
    """

    if run_module is None:
        run_module = importlib.import_module("run")
    if math_factory is None:
        from .portable_native_math import build_portable_native_math

        math_factory = build_portable_native_math
    if not callable(math_factory):
        raise TypeError("portable native math_factory must be callable")

    recovery = read_recovery_policy(cfg)
    out_dir = _resolved_child(cfg, name="out_dir", path="cfg")
    if out_dir in (None, "", "null", "None"):
        raise UnsupportedPortableNativeProfile(
            "portable native requires cfg.out_dir"
        )
    resume_root = Path(str(out_dir)).expanduser().resolve()
    if recovery.resume:
        _validate_resume_root(resume_root)
        existing_run_dir = resume_root
        persist_config = False
    else:
        existing_run_dir = None
        persist_config = True

    args = run_module.parse_arguments(
        cfg,
        existing_run_dir=existing_run_dir,
        persist_config=persist_config,
    )
    _fill_inactive_runtime_defaults(args)
    validate_portable_native_profile(args, run_module)
    # Validate selected input files before state preparation.  This is
    # model-free and ensures supported file fields never degrade to path-only
    # identity when the referenced content disappears.
    _supplemental_asset_paths(args, None)

    root = Path(args.out_dir).expanduser().resolve()
    runtime_root = root / ".framework"
    data = DataStore(runtime_root / "data")
    for codec in (
        design_codec,
        _JSON,
        _PREPARED_RECEIPT_CODEC,
        _INITIALIZER_RESULT_CODEC,
    ):
        data.register(codec)

    context = LegacyContext()
    config_identity = key_for(
        (
            "astrolabe.portable-native.config/v3",
            scientific_config_payload(args),
        )
    )
    sources = source_manifest(run_module, source_paths, args=args)
    source_identity = key_for(
        (
            "astrolabe.portable-native.source/v1",
            sources,
        )
    )

    if recovery.resume:
        receipt = _read_prepared_receipt(data, root)
        if receipt.config_identity != config_identity:
            raise ContractConflict(
                "portable native scientific configuration changed on resume"
            )
        if receipt.source_identity != source_identity:
            raise ContractConflict(
                "portable native selected source changed on resume"
            )
        prepared = run_module.prepare_pipeline_state(
            args,
            apply_initialization=False,
        )
        baseline_translation = context.capture_candidate(
            assemblies=_stage_selection(args)[3],
            data=data,
        )
        if receipt.initial_candidate.system != baseline_translation.system.identity:
            raise ContractConflict(
                "persisted initializer candidate belongs to another molecular System"
            )
        context.apply_candidate(
            receipt.initial_candidate,
            translation=baseline_translation,
            data=data,
        )
        _restore_rng(data.get(receipt.rng_after, design_codec))
        initializer_result = (
            None
            if receipt.initializer_result is None
            else data.get(receipt.initializer_result, _INITIALIZER_RESULT_CODEC)
        )
    else:
        rng_before = _rng_snapshot()
        prepared = run_module.prepare_pipeline_state(args)
        rng_after = _rng_snapshot()
        initializer_result = getattr(prepared, "initializer_result", None)

    (
        _selected_manifest,
        selected_asset_record,
        selected_asset_identity,
    ) = _selected_portable_assets(args)
    supplemental_asset_values: dict[str, Any] = {
        **dict(_normalize_paths(asset_paths, prefix="asset")),
        **_supplemental_asset_paths(args, initializer_result),
    }
    supplemental_assets = content_manifest(
        supplemental_asset_values,
        prefix="asset",
    )
    resource_identity = key_for(
        (
            "astrolabe.portable-native.resources/v2",
            selected_asset_identity,
            tuple(supplemental_assets),
        )
    )

    if recovery.resume:
        if receipt.resource_manifest != resource_identity:
            raise ContractConflict(
                "portable native selected asset content changed on resume"
            )
    else:
        translation = context.capture_candidate(
            assemblies=_stage_selection(args)[3],
            data=data,
        )
        initializer_result_blob = (
            None
            if initializer_result is None
            else data.put(initializer_result, _INITIALIZER_RESULT_CODEC)
        )
        rng_before_blob = data.put(rng_before, design_codec)
        rng_after_blob = data.put(rng_after, design_codec)
        receipt = _PreparedReceipt(
            schema=PREPARED_SCHEMA,
            config_identity=config_identity,
            source_identity=source_identity,
            resource_manifest=resource_identity,
            initializer_mode=_initializer_mode(args, run_module),
            initial_candidate=translation.candidate,
            rng_before=rng_before_blob,
            rng_after=rng_after_blob,
            initializer_result=initializer_result_blob,
        )
        receipt_blob = data.put(receipt, _PREPARED_RECEIPT_CODEC)
        data.persist(
            tuple(
                item
                for item in (
                    receipt_blob,
                    initializer_result_blob,
                    rng_before_blob,
                    rng_after_blob,
                )
                if item is not None
            )
        )
        _ensure_journal_exists(root)
        _write_prepared_manifest(root, receipt_blob)

    prepare_pipeline_math = getattr(run_module, "prepare_pipeline_math", None)
    if not callable(prepare_pipeline_math):
        raise UnsupportedPortableNativeProfile(
            "portable native requires run.prepare_pipeline_math; model-loading "
            "load_pipeline_components is not a cold fallback"
        )
    before_math = _rng_snapshot()
    math_components = prepare_pipeline_math(args, prepared)
    if not isinstance(math_components, tuple) or len(math_components) < 3:
        raise TypeError(
            "run.prepare_pipeline_math must return loss, stage manager, and inputs"
        )
    loss_computer, stage_manager, _inputs = math_components[:3]
    after_math = _rng_snapshot()
    if not _rng_equal(before_math, after_math):
        _restore_rng(before_math)
        raise UnsupportedPortableNativeProfile(
            "model-free pipeline math and tool-context construction consumed "
            "scientific RNG"
        )

    selected_scientific = _selected_scientific_payload(args)
    stage_dict = dict(selected_scientific["stage"])
    selected_evaluator_config = _selected_component_config(
        selected_scientific["evaluator"],
        family="evaluator",
    )
    selected_mutator_config = _selected_component_config(
        selected_scientific["mutator"],
        family="mutator",
    )
    _, selected_mutators, evaluator_alias, assemblies = _stage_selection(args)
    mutator_alias = selected_mutators[0]
    _check_loaded_stage(
        stage_manager,
        evaluator_alias=evaluator_alias,
        mutator_alias=mutator_alias,
        assemblies=assemblies,
    )

    translation = context.capture_candidate(assemblies=assemblies, data=data)
    initial_candidate = (
        receipt.initial_candidate if recovery.resume else translation.candidate
    )
    if translation.candidate != initial_candidate:
        raise ContractConflict(
            "legacy initializer state does not equal the durable prepared candidate"
        )
    native_segments = native_mutation_segments()
    native_segment_names = tuple(name for name, _segment in native_segments)
    assignments = {
        assignment.variable: assignment.value
        for assignment in translation.candidate.assignments
    }
    polymer_occurrence_names = tuple(
        dict.fromkeys(
            occurrence.variable
            for occurrence in translation.system.occurrences
            if isinstance(assignments.get(occurrence.variable), Polymer)
        )
    )
    if (
        len(native_segment_names) != len(set(native_segment_names))
        or not polymer_occurrence_names
        or set(native_segment_names) != set(polymer_occurrence_names)
    ):
        raise UnsupportedPortableNativeProfile(
            "portable native mutation topology requires one native polymer "
            "context and one physical polymer occurrence for every managed "
            "segment: "
            f"segments={native_segment_names!r}, "
            f"polymer_occurrences={polymer_occurrence_names!r}"
        )
    bindings = NativeBindings(
        context=context,
        translation=translation,
        data=data,
        work_root=root / "native-work",
    )
    sampling_policy = _sampling_policy(args, data, stage=stage_dict)
    explicit_hooks = _validate_hooks(
        operation_hooks or {},
        evaluator_alias=evaluator_alias,
        mutator_alias=mutator_alias,
    )
    from .portable_native_math import (
        PortableNativeHookContext,
        build_portable_native_operation_hooks,
    )

    default_hooks = {
        alias.casefold(): hook
        for alias, hook in build_portable_native_operation_hooks(
            PortableNativeHookContext(
                args=args,
                run_module=run_module,
                loss_computer=loss_computer,
                stage_manager=stage_manager,
                evaluator_alias=evaluator_alias,
                mutator_alias=mutator_alias,
                config_identity=config_identity,
                source_identity=source_identity,
            )
        ).items()
    }
    hooks = {**default_hooks, **explicit_hooks}
    selected_mutator_hook = hooks.get(mutator_alias.casefold())
    if (
        selected_mutator_hook is None
        or not isinstance(
            selected_mutator_hook.mutation_context,
            MutationContextPolicy,
        )
    ):
        raise UnsupportedPortableNativeProfile(
            "portable native requires the selected mutator hook to declare an "
            "explicit MutationContextPolicy for durable pending state"
        )
    component_context = NativeComponentContext(
        bindings=bindings,
        validated_evaluator_conf=selected_evaluator_config,
        validated_mutator_conf=selected_mutator_config,
        raw_model_results_dir=Path(args.raw_model_results_dir),
        sampling_policy=sampling_policy,
        options_schemas=options_schemas or {},
        operation_hooks=hooks,
    )

    from astrolabe.evaluators.registry import EvaluatorRegistry
    from astrolabe.mutators.registry import MutatorRegistry

    evaluator_type = _component_type(
        args,
        family="evaluator",
        alias=evaluator_alias,
    )
    mutator_type = _component_type(
        args,
        family="mutator",
        alias=mutator_alias,
    )
    evaluator = EvaluatorRegistry.build_operation(
        evaluator_type,
        alias=evaluator_alias,
        options={},
        context=component_context,
    )
    mutator = MutatorRegistry.build_operation(
        mutator_type,
        alias=mutator_alias,
        options={},
        context=component_context,
    )

    support = PortableNativeSupport((str(stage_manager.current_stage),))
    static_stage = validate_static_nonpose_support(support)
    runtime_context, root_runtime_config = _runtime_context(args, data)
    sample_protocol = key_for(
        ("astrolabe.portable-native.sample/v1", int(args.seed))
    )
    acceptance_protocol = key_for(
        (
            "astrolabe.portable-native.acceptance/v1",
            bool(args.use_metropolis_acceptance),
            _acceptance_initial_temperature(args),
            stage_dict.get("weighted_score_expression"),
            stage_dict.get("acceptance_gate"),
            stage_dict.get("acceptance_threshold"),
            stage_dict.get("early_stopping"),
            stage_dict.get("prefix_early_stopping"),
        )
    )
    objective = key_for(
        (
            "astrolabe.portable-native.objective/v1",
            stage_dict.get("weighted_score_expression"),
            selected_scientific["loss"],
        )
    )
    contract = ScientificRunContract(
        schema=SCIENTIFIC_CONTRACT_SCHEMA,
        config_identity=config_identity,
        source_identity=source_identity,
        system_identity=translation.system.identity,
        resource_manifest=resource_identity,
        static_stage=static_stage,
        objective=objective,
        component_identities=(
            evaluator.descriptor.identity,
            mutator.descriptor.identity,
        ),
        seed=int(args.seed),
        acceptance_protocol=acceptance_protocol,
        codec_identities=(
            design_codec.identity,
            _JSON.identity,
            SCIENTIFIC_RUN_CONTRACT_CODEC.identity,
            NATIVE_STEP_STATE_CODEC.identity,
            NUMERIC_TREE_CODEC.identity,
            NATIVE_RUNTIME_CONTEXT_CODEC.identity,
            NATIVE_SCORES_ENVELOPE_CODEC.identity,
            DECISION_REQUEST_CODEC.identity,
            DECISION_RESULT_CODEC.identity,
        ),
        recovery_capabilities=(
            "initializer_receipt",
            "model_ready",
            "raw_predict",
            "decision",
        ),
    )
    math_context = PortableNativeMathContext(
        args=args,
        prepared=prepared,
        loss_computer=loss_computer,
        stage_manager=stage_manager,
        bindings=bindings,
        contract=contract,
        initial_candidate=initial_candidate,
        runtime_context=runtime_context,
        root_runtime_config=root_runtime_config,
        sample_protocol=sample_protocol,
        run_module=run_module,
        mutation_context=selected_mutator_hook.mutation_context,
    )
    before_math = _rng_snapshot()
    math_adapter = math_factory(math_context)
    if not isinstance(math_adapter, PortableNativeMath):
        raise TypeError(
            "portable native math_factory must return the shared PortableNativeMath adapter"
        )
    if not _rng_equal(before_math, _rng_snapshot()):
        _restore_rng(before_math)
        raise ValueError("portable native math construction consumed scientific RNG")

    flow = PortableNativeFlow(
        contract=contract,
        support=support,
        system=translation.system,
        initial_candidate=initial_candidate,
        evaluator=evaluator,
        mutator=mutator,
        bindings=bindings,
        sampling_policy=sampling_policy,
        assemblies=assemblies,
        math=math_adapter,
        runtime_context=runtime_context,
        sample_protocol=sample_protocol,
        initialization_order=("mutator", "evaluator"),
    )
    runtime = Runtime(
        flow.operations,
        root=runtime_root,
        data=data,
        scientific_contract=contract.config_identity,
    )
    if type(args.steps) is not int:
        raise UnsupportedPortableNativeProfile(
            "portable native steps must be an integer"
        )
    rounds = args.steps - 1
    if rounds < 0:
        raise UnsupportedPortableNativeProfile(
            "portable native steps must include the root evaluation"
        )
    return PortableNativeRecipe(
        schema=PROFILE_SCHEMA,
        recovery=recovery,
        args=args,
        prepared=prepared,
        runtime=runtime,
        flow=flow,
        contract=contract,
        source_manifest=sources,
        asset_manifest=selected_asset_record,
        supplemental_asset_manifest=supplemental_assets,
        selected_evaluator_alias=evaluator_alias,
        selected_mutator_alias=mutator_alias,
        assemblies=assemblies,
        rounds=rounds,
    )


def af3_mcmc_single_child_template() -> dict[str, Any]:
    """Return a low-cost MCMC+AF3 acceptance template, not a GPU claim."""

    return {
        "schema": AF3_MCMC_TEMPLATE_SCHEMA,
        "task_name": "portable_af3_mcmc",
        "out_dir": "outputs/portable_af3_mcmc",
        "seed": 1729,
        "repeats": 1,
        "steps": 2,
        "types": {"A": "p"},
        "seqs": {"A": "ACDEFGHIKLMN"},
        "initializer": {"mode": "random"},
        "evaluator": {
            "evaluator_terms": {
                "af3": {
                    "_type": "Alphafold3",
                    "num_seeds": 1,
                    "num_recycles": 1,
                    "num_samples": 1,
                    "run_data_pipeline": False,
                    "run_inference": True,
                }
            }
        },
        "mutator": {
            "mutator_terms": {
                "MCMC": {"_type": "MCMC"},
            }
        },
        "loss": {"loss_terms": {}},
        "stages": {
            "default": {
                "evaluator": "af3",
                "oligo": ["A"],
                "mutators": ["MCMC"],
                "mutation_num": 1,
                "weighted_score_expression": "plddt",
                "acceptance_threshold": None,
                "early_stopping": None,
            }
        },
        "enable_pyrosetta": False,
        "skip_complex_poses": True,
        "recovery": {"mode": "durable", "resume": False},
    }


__all__ = [
    "AF3_MCMC_TEMPLATE_SCHEMA",
    "ContentRecord",
    "PortableNativeMathContext",
    "PortableNativeMathFactory",
    "PortableNativeMathUnavailable",
    "PortableNativeRecipe",
    "PortableRecoveryPolicy",
    "SCIENTIFIC_CONFIG_SCHEMA",
    "UnsupportedPortableNativeProfile",
    "af3_mcmc_single_child_template",
    "build_portable_native_recipe",
    "content_manifest",
    "read_recovery_policy",
    "scientific_config_payload",
    "source_manifest",
    "validate_portable_native_profile",
]
