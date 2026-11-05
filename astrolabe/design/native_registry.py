"""Cold typed-operation adapters for the existing native family registries.

Family registries remain the source of native factory names. This owner only
bridges their validated configurations into the existing lazy evaluator and
mutator operation makers; no native component is constructed while an
operation is built or preflighted.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import inspect
from pathlib import Path
from typing import Any

from .native_mutation import make_mutator_operation
from .native_mutation_state import MutationContextPolicy
from .native_prediction import make_evaluator_operation
from .native_support import NativeBindings
from .types import Key, SamplingPolicy, key_for


NATIVE_REGISTRY_CODE_VERSION = "astrolabe-native-registry/v1"
_COMPONENT_CONFIG_ALIASES = {
    "proteinmpnn": "ligandmpnn",
    "multistate_proteinmpnn": "ligandmpnn",
    "far_apart_tied_multitarget_ligandmpnn": "ligandmpnn",
    "grouped_ligandmpnn": "ligandmpnn",
}


@dataclass(frozen=True)
class NativeOperationHooks:
    """Frozen alias-specific configuration hooks for a native operation."""

    configure: Callable[..., Mapping[str, Any] | None]
    identity: str
    mutation_context: MutationContextPolicy | None = None
    invoke: Callable[..., Any] | None = None
    invoke_identity: str | None = None

    def __post_init__(self) -> None:
        if not callable(self.configure):
            raise TypeError("native operation hook configure must be callable")
        if not isinstance(self.identity, str) or not self.identity.strip():
            raise ValueError("native operation hook identity must be non-empty")
        if self.mutation_context is not None and not isinstance(
            self.mutation_context, MutationContextPolicy
        ):
            raise TypeError(
                "native operation hook mutation_context must be "
                "MutationContextPolicy or None"
            )
        if (self.invoke is None) != (self.invoke_identity is None):
            raise ValueError(
                "native operation hook invoke and invoke_identity "
                "must be supplied together"
            )
        if self.invoke is not None and not callable(self.invoke):
            raise TypeError("native operation hook invoke must be callable")
        if self.invoke_identity is not None and (
            not isinstance(self.invoke_identity, str)
            or not self.invoke_identity.strip()
        ):
            raise ValueError(
                "native operation hook invoke_identity must be non-empty"
            )


@dataclass(frozen=True)
class NativeComponentContext:
    """Inputs required to bind configured native families to typed operations."""

    bindings: NativeBindings
    validated_evaluator_conf: Any | None
    validated_mutator_conf: Any | None
    raw_model_results_dir: Path
    sampling_policy: SamplingPolicy
    options_schemas: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    operation_hooks: Mapping[str, NativeOperationHooks] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.bindings, NativeBindings):
            raise TypeError("bindings must be NativeBindings")
        if not isinstance(self.sampling_policy, SamplingPolicy):
            raise TypeError("sampling_policy must be SamplingPolicy")
        if not isinstance(self.raw_model_results_dir, (str, Path)):
            raise TypeError("raw_model_results_dir must be path-like")
        object.__setattr__(
            self,
            "raw_model_results_dir",
            Path(self.raw_model_results_dir),
        )
        if not isinstance(self.options_schemas, Mapping):
            raise TypeError("options_schemas must be a mapping")
        schemas: dict[str, Mapping[str, Any]] = {}
        for name, schema in self.options_schemas.items():
            if not isinstance(name, str) or not name:
                raise ValueError("options_schemas keys must be non-empty strings")
            if not isinstance(schema, Mapping):
                raise TypeError(f"options schema for {name!r} must be a mapping")
            schemas[name.casefold()] = deepcopy(dict(schema))
        object.__setattr__(self, "options_schemas", schemas)
        if not isinstance(self.operation_hooks, Mapping):
            raise TypeError("operation_hooks must be a mapping")
        hooks: dict[str, NativeOperationHooks] = {}
        for name, hook in self.operation_hooks.items():
            if not isinstance(name, str) or not name:
                raise ValueError("operation_hooks keys must be non-empty strings")
            if not isinstance(hook, NativeOperationHooks):
                raise TypeError(
                    f"operation hook for {name!r} must be NativeOperationHooks"
                )
            key = name.casefold()
            if key in hooks:
                raise ValueError(
                    f"duplicate operation hook alias after case folding: {name!r}"
                )
            hooks[key] = hook
        object.__setattr__(self, "operation_hooks", hooks)


def _explicit_override(owner, name, *, alias, options, context):
    factories = getattr(owner, "_operation_factories", None) or {}
    factory = factories.get(str(name).strip().lower())
    if factory is None:
        return None
    return factory(alias=alias, options=dict(options), context=context)


def _field(config: Any, name: str) -> Any:
    if isinstance(config, Mapping):
        return config.get(name)
    return getattr(config, name, None)


def _set_field(config: Any, name: str, value: Any) -> None:
    if isinstance(config, Mapping):
        try:
            config[name] = value
            return
        except (TypeError, KeyError):
            pass
    try:
        setattr(config, name, value)
        return
    except (AttributeError, TypeError):
        pass
    try:
        from omegaconf import OmegaConf

        OmegaConf.update(config, name, value, merge=False, force_add=True)
    except Exception as exc:
        raise TypeError(f"cannot replace family configuration field {name!r}") from exc


def _selected_config(
    config: Any,
    *,
    owner: Any,
    terms_field: str,
    alias: str,
    type_name: str,
) -> Any:
    if config is None:
        raise ValueError("native family configuration is missing")
    terms = _field(config, terms_field)
    if not isinstance(terms, Mapping) or alias not in terms:
        raise ValueError(
            f"selected operation alias {alias!r} is absent from {terms_field}"
        )
    term = terms[alias]
    if not isinstance(term, Mapping):
        raise ValueError(f"{terms_field}.{alias} must be a mapping")
    configured_type = term.get("_type")
    if not isinstance(configured_type, str) or (
        configured_type.casefold() != type_name.casefold()
    ):
        raise ValueError(
            f"{terms_field}.{alias} selects type {configured_type!r}, "
            f"not requested type {type_name!r}"
        )
    from omegaconf import OmegaConf

    def resolve(value: Any, *, path: str) -> Any:
        if OmegaConf.is_config(value):
            try:
                return OmegaConf.to_container(value, resolve=True)
            except Exception as exc:
                raise ValueError(
                    f"Failed to resolve selected native configuration at {path}: {exc}"
                ) from exc
        return deepcopy(value)

    selected_base = _COMPONENT_CONFIG_ALIASES.get(
        type_name.casefold(),
        type_name.casefold(),
    )
    component_keys = {
        str(name).casefold() for name in owner.list_available()
    }
    component_keys.update(_COMPONENT_CONFIG_ALIASES.values())
    component_keys.add(type_name.casefold())

    selected: dict[str, Any] = {}
    items = config.items() if isinstance(config, Mapping) else ()
    for raw_name, value in items:
        name = str(raw_name)
        folded = name.casefold()
        if name == terms_field:
            continue
        if name == "enabled_aliases":
            continue
        if folded in component_keys and folded != selected_base:
            continue
        selected[name] = resolve(
            value,
            path=f"{terms_field}.__context__.{name}",
        )
    selected[terms_field] = {
        alias: resolve(term, path=f"{terms_field}.{alias}")
    }
    selected["enabled_aliases"] = [alias]
    return OmegaConf.create(selected)


def _plain_config(value: Any) -> Any:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
    except ImportError:
        pass
    if isinstance(value, Mapping):
        return {
            str(name): _plain_config(item)
            for name, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_plain_config(item) for item in value)
    if isinstance(value, list):
        return [_plain_config(item) for item in value]
    if value is None or type(value) in {str, bool, int, float}:
        return value
    raise TypeError(
        "validated native configuration contains unsupported value "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _source_record(source: Any) -> dict[str, Any]:
    try:
        path_text = inspect.getsourcefile(source) or inspect.getfile(source)
    except (OSError, TypeError) as exc:
        raise ValueError(
            "native operation source is unavailable; "
            f"got {getattr(source, '__module__', type(source).__module__)}."
            f"{getattr(source, '__qualname__', type(source).__qualname__)}"
        ) from exc
    if not path_text:
        raise ValueError("native operation source is unavailable")
    path = Path(path_text).resolve()
    if not path.is_file():
        raise ValueError(f"factory source file is unavailable: {path}")
    return {
        "module": getattr(source, "__module__", ""),
        "qualname": getattr(source, "__qualname__", ""),
        "path": path.as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _adapter_sources() -> tuple[dict[str, Any], ...]:
    sources = []
    for module_name in (
        "astrolabe.design.native_prediction",
        "astrolabe.design.native_mutation",
        "astrolabe.design.native_support",
        "astrolabe.design.native_registry",
    ):
        module = __import__(module_name, fromlist=["*"])
        path = Path(inspect.getsourcefile(module) or inspect.getfile(module)).resolve()
        sources.append(
            {
                "module": module_name,
                "path": path.as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return tuple(sources)


def _model_identity(
    *,
    family: str,
    type_name: str,
    alias: str,
    terms_field: str,
    effective_config: Any,
    factory: Any,
    asset_record: Mapping[str, Any],
) -> Key:
    return key_for(
        (
            NATIVE_REGISTRY_CODE_VERSION,
            family,
            type_name.casefold(),
            alias,
            _plain_config(effective_config),
            asset_record,
            _source_record(factory),
            _adapter_sources(),
        )
    )


def _asset_identity_record(
    *,
    family: str,
    type_name: str,
    alias: str,
    terms_field: str,
    effective_config: Any,
) -> dict[str, Any]:
    """Content-identify the assets reachable from the selected component."""

    from .configured_refinement import (
        _AssetComponent,
        _SelectedAssetManifest,
        _asset_manifest_record,
        _matching_component_base,
        _merge_component_mapping,
        _model_component_assets,
    )

    plain = _plain_config(effective_config)
    terms = plain.get(terms_field)
    if not isinstance(terms, Mapping) or not isinstance(terms.get(alias), Mapping):
        raise ValueError(
            f"selected operation alias {alias!r} is absent from {terms_field}"
        )
    selected_mapping = _merge_component_mapping(
        _matching_component_base(plain, type_name),
        terms[alias],
    )
    role = f"{family}:{type_name}"
    try:
        component, files = _model_component_assets(
            role,
            type_name,
            selected_mapping,
        )
    except ValueError as exc:
        component = _AssetComponent(
            role,
            type_name.casefold(),
            "unsupported",
            str(exc),
        )
        files = ()
    return _asset_manifest_record(_SelectedAssetManifest((component,), files))


def _options_schema(
    context: NativeComponentContext,
    *,
    alias: str,
    type_name: str,
    options: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    candidates = []
    for key in (alias.casefold(), type_name.casefold()):
        if key in context.options_schemas:
            schema = context.options_schemas[key]
            if all(schema != existing for existing in candidates):
                candidates.append(schema)
    if len(candidates) > 1:
        raise ValueError(
            f"conflicting options schemas for alias {alias!r} and type {type_name!r}"
        )
    schema = candidates[0] if candidates else None
    if options and schema is None:
        raise ValueError(
            f"non-empty options for native type {type_name!r} require an explicit "
            "options schema in NativeComponentContext.options_schemas"
        )
    return schema


def _operation_hook(
    context: NativeComponentContext,
    *,
    alias: str,
) -> (
    tuple[
        NativeOperationHooks,
        Callable[..., Mapping[str, Any] | None],
        str,
        Callable[..., Any] | None,
        str | None,
    ]
    | None
):
    hook = context.operation_hooks.get(alias.casefold())
    if hook is None:
        return None
    source = _source_record(hook.configure)
    configure_identity = key_for((hook.identity, source)).digest
    invoke_identity = (
        None
        if hook.invoke is None
        else key_for(
            (
                "native-operation-invoker/v1",
                hook.invoke_identity,
                _source_record(hook.invoke),
            )
        ).digest
    )
    return (
        hook,
        hook.configure,
        configure_identity,
        hook.invoke,
        invoke_identity,
    )


def _registered_factory(owner, name: str):
    try:
        return owner.get(name)
    except KeyError as exc:
        available = ", ".join(owner.list_available()) or "<none>"
        raise ValueError(
            f"Unknown {owner.__name__} native type {name!r}; "
            f"available types: {available}"
        ) from exc


def build_evaluator_operation(
    owner,
    name: str,
    *,
    alias: str,
    options: Mapping[str, Any] | None,
    context: NativeComponentContext,
):
    """Build an evaluator operation from the live evaluator registry."""

    options = dict(options or {})
    override = _explicit_override(
        owner, name, alias=alias, options=options, context=context
    )
    if override is not None:
        return override
    if not isinstance(context, NativeComponentContext):
        raise TypeError("native evaluator build requires NativeComponentContext")
    import astrolabe.evaluators.utils  # noqa: F401 - registers native factories

    factory = _registered_factory(owner, name)
    effective_config = _selected_config(
        context.validated_evaluator_conf,
        owner=owner,
        terms_field="evaluator_terms",
        alias=alias,
        type_name=name,
    )
    schema = _options_schema(
        context,
        alias=alias,
        type_name=name,
        options=options,
    )
    asset_record = _asset_identity_record(
        family="evaluator", type_name=name, alias=alias,
        terms_field="evaluator_terms", effective_config=effective_config,
    )
    model_identity = _model_identity(
        family="evaluator",
        type_name=name,
        alias=alias,
        terms_field="evaluator_terms",
        effective_config=effective_config,
        factory=factory,
        asset_record=asset_record,
    )
    operation_hook = _operation_hook(context, alias=alias)
    _, configure, configure_identity, invoke, invoke_identity = (
        operation_hook
        if operation_hook is not None
        else (None, None, None, None, None)
    )

    def lazy_factory():
        from astrolabe.evaluators.utils import construct_evaluators

        oligo_names = tuple(
            assembly.identity for assembly in context.bindings.translation.assemblies
        )
        config = deepcopy(effective_config)
        evaluators = construct_evaluators(
            config,
            context.raw_model_results_dir,
            oligo_names=oligo_names,
            enabled_aliases=[alias],
        )
        try:
            return evaluators[alias]
        except KeyError:
            raise ValueError(
                f"construct_evaluators did not build selected alias {alias!r}"
            ) from None

    operation = make_evaluator_operation(
        alias,
        lazy_factory,
        context.bindings,
        version=NATIVE_REGISTRY_CODE_VERSION,
        model_identity=model_identity,
        options=options,
        schema=schema,
        configure=configure,
        configure_identity=configure_identity,
        invoke=invoke,
        invoke_identity=invoke_identity,
    )
    operation.resume_assets_verified = asset_record["complete"]
    return operation


def build_mutator_operation(
    owner,
    name: str,
    *,
    alias: str,
    options: Mapping[str, Any] | None,
    context: NativeComponentContext,
):
    """Build a mutator operation from the live mutator registry."""

    options = dict(options or {})
    override = _explicit_override(
        owner, name, alias=alias, options=options, context=context
    )
    if override is not None:
        return override
    if not isinstance(context, NativeComponentContext):
        raise TypeError("native mutator build requires NativeComponentContext")
    import astrolabe.mutators.base  # noqa: F401 - registers native factories

    factory = _registered_factory(owner, name)
    effective_config = _selected_config(
        context.validated_mutator_conf,
        owner=owner,
        terms_field="mutator_terms",
        alias=alias,
        type_name=name,
    )
    schema = _options_schema(
        context,
        alias=alias,
        type_name=name,
        options=options,
    )
    asset_record = _asset_identity_record(
        family="mutator", type_name=name, alias=alias,
        terms_field="mutator_terms", effective_config=effective_config,
    )
    model_identity = _model_identity(
        family="mutator",
        type_name=name,
        alias=alias,
        terms_field="mutator_terms",
        effective_config=effective_config,
        factory=factory,
        asset_record=asset_record,
    )
    operation_hook = _operation_hook(context, alias=alias)
    hook, configure, configure_identity, invoke, invoke_identity = (
        operation_hook
        if operation_hook is not None
        else (None, None, None, None, None)
    )
    mutation_context = (
        hook.mutation_context
        if hook is not None
        else None
    )

    def lazy_factory():
        from astrolabe.mutators.base import build_mutators

        config = deepcopy(effective_config)
        mutators = build_mutators(config, context.raw_model_results_dir)
        try:
            return mutators[alias]
        except KeyError:
            raise ValueError(
                f"build_mutators did not build selected alias {alias!r}"
            ) from None

    operation = make_mutator_operation(
        alias,
        lazy_factory,
        context.bindings,
        sampling_policy=context.sampling_policy,
        version=NATIVE_REGISTRY_CODE_VERSION,
        model_identity=model_identity,
        options=options,
        schema=schema,
        configure=configure,
        configure_identity=configure_identity,
        invoke=invoke,
        invoke_identity=invoke_identity,
        mutation_context=mutation_context,
    )
    operation.resume_assets_verified = asset_record["complete"]
    return operation


__all__ = [
    "NATIVE_REGISTRY_CODE_VERSION",
    "NativeComponentContext",
    "NativeOperationHooks",
    "build_evaluator_operation",
    "build_mutator_operation",
]
