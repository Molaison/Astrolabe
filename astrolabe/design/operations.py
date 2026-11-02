"""Typed scientific operations. Execution and durable binding belong to Runtime.

Factories live at their scientific family owner. This module contains no model
names, model loading, scheduling, or fallback implementation.
"""
from __future__ import annotations

import inspect
import json
from copy import deepcopy
from collections.abc import Callable
from typing import Generic, TypeVar

from .types import (
    Capability, ComponentDescriptor, Dependency, InputManifest, Key, Requirement,
    SamplingPolicy, SelectedOptionsSchema, key_for, manifest, _Prepared,
)

I = TypeVar("I")
O = TypeVar("O")
INVOKE = Capability("astrolabe.operation", "invoke", "1")
Implementation = Callable[..., object] | type


class FunctionOperation(Generic[I, O]):
    """Adapter for a typed owner function, with optional input projection.

    A projector must enumerate the *actual* scientific inputs. The default is
    deliberately conservative: the entire encoded request is a dependency.
    File contents and model/provider identities must be bound by the owner,
    never represented by a mutable pathname alone.
    """

    def __init__(self, descriptor: ComponentDescriptor[I, O],
                 invoke: Callable[[I], O], *,
                 prepare: Callable[[I], _Prepared[I]] | None = None,
                 paid_boundary: str = "",
                 terminal_errors: tuple[type[Exception], ...] = (),
                 reference_codecs: tuple = (), reconciler: Callable | None = None,
                 on_result: Callable[[O], None] | None = None):
        self.descriptor = descriptor
        self._invoke = invoke
        self._prepare = prepare
        self.paid_boundary = paid_boundary
        self.terminal_errors = terminal_errors
        self.reference_codecs = reference_codecs
        self.reconcile = reconciler
        self.on_result = on_result

    def failure_is_terminal(self, error: Exception, *, paid_boundary: str) -> bool:
        # The adapter's explicit error contract is required for paid/external
        # operations. A disconnect by itself is never terminal evidence.
        return not paid_boundary or isinstance(error, self.terminal_errors)

    def prepare(self, request: I) -> _Prepared[I]:
        if not isinstance(request, self.descriptor.request_type):
            raise TypeError(f"{self.descriptor.alias}: expected {self.descriptor.request_type.__name__}")
        # The request codec validates and freezes supported values before issue.
        codec = self.descriptor.request_codec
        request = codec.decode(codec.encode(request))
        if self._prepare is not None:
            prepared = self._prepare(request)
            if not isinstance(prepared, _Prepared):
                raise TypeError("Operation prepare must return _Prepared")
            if not isinstance(prepared.request, self.descriptor.request_type):
                raise TypeError("Prepared request does not match the declared request type")
            if manifest(prepared.inputs.dependencies) != prepared.inputs:
                raise ValueError("Prepared input manifest does not match its dependencies")
            if prepared.paid_boundary != self.paid_boundary:
                raise ValueError("Prepared accounting boundary differs from its operation declaration")
            request = codec.decode(codec.encode(prepared.request))
            dependencies = prepared.inputs.dependencies
            requirements = prepared.requirements
            references = prepared.references
        else:
            dependencies, requirements, references = (), (), ()
        if any(item.role == "prepared-request" for item in dependencies):
            raise ValueError("prepared-request dependency is reserved")
        digest = key_for((codec.identity, codec.encode(request).hex()))
        inputs = manifest((*dependencies, Dependency("prepared-request", digest)))
        return _Prepared(request, inputs, requirements, self.paid_boundary, references)

    def invoke(self, prepared: _Prepared[I]) -> O:
        codec = self.descriptor.request_codec
        digest = key_for((codec.identity, codec.encode(prepared.request).hex()))
        if manifest(prepared.inputs.dependencies) != prepared.inputs or (
            Dependency("prepared-request", digest) not in prepared.inputs.dependencies
        ):
            raise ValueError("prepared request changed after its input identity was recorded")
        result = self._invoke(prepared.request)
        if not isinstance(result, self.descriptor.result_type):
            raise TypeError(f"{self.descriptor.alias}: invalid result type {type(result).__name__}")
        return result


def _source_identity(value: Implementation | None, explicit_identity: Key | None = None):
    if value is None:
        return None
    opaque = not (inspect.isfunction(value) or inspect.isclass(value)) or (
        inspect.isfunction(value)
        and bool(value.__closure__ or value.__defaults__ or value.__kwdefaults__)
    )
    if opaque and explicit_identity is None:
        raise ValueError("stateful callable requires an explicit model_identity")
    try:
        source = inspect.getsource(value)
    except (OSError, TypeError):
        if explicit_identity is None:
            raise ValueError("callable source is unavailable; supply model_identity") from None
        module = getattr(value, "__module__", type(value).__module__)
        qualname = getattr(value, "__qualname__", type(value).__qualname__)
        source = f"{module}.{qualname}"
    return (source, explicit_identity) if opaque else source


def _describe_operation(
    alias: str,
    request_type: type[I],
    result_type: type[O],
    *,
    implementation: tuple[Implementation, ...],
    version: str,
    request_codec=None,
    result_codec=None,
    options: dict | None = None,
    schema: dict | None = None,
    capabilities: tuple[Capability, ...] = (),
    model_identity: Key | None = None,
    prepare: Callable[..., object] | None = None,
    paid_boundary: str = "",
    terminal_errors: tuple[type[Exception], ...] = (),
    reference_codecs: tuple = (),
    reconciler: Callable[..., object] | None = None,
    on_result: Callable[..., object] | None = None,
) -> ComponentDescriptor[I, O]:
    """Describe an operation from the sources that actually provide it."""

    from .store import DataclassCodec
    if not alias or not version:
        raise ValueError("Operation alias and implementation version are required")
    if model_identity is not None and not isinstance(model_identity, Key):
        raise TypeError("model_identity must be a content Key")
    if not isinstance(implementation, tuple) or not implementation:
        raise ValueError("implementation must be a non-empty tuple of callables or classes")
    if any(
        not isinstance(value, type) and not callable(value)
        for value in implementation
    ):
        raise TypeError("implementation entries must be callables or classes")
    if any(not isinstance(error, type) or not issubclass(error, Exception) for error in terminal_errors):
        raise TypeError("terminal_errors must contain explicit Exception classes")
    request_codec = request_codec or DataclassCodec(request_type, f"{request_type.__module__}.{request_type.__qualname__}/v1")
    result_codec = result_codec or DataclassCodec(result_type, f"{result_type.__module__}.{result_type.__qualname__}/v1")
    options = dict(options or {})
    schema = schema or {"type": "object", "properties": {}, "additionalProperties": False}
    # Validate selected options before registration/model construction.
    import jsonschema
    jsonschema.validate(options, schema)
    selected = SelectedOptionsSchema(alias, version,
        json.dumps(schema, sort_keys=True, separators=(",", ":"), allow_nan=False),
        json.dumps(options, sort_keys=True, separators=(",", ":"), allow_nan=False))
    sources = tuple(_source_identity(value, model_identity) for value in implementation)
    identity = key_for((alias, version, sources, selected, model_identity,
                        request_codec.identity, result_codec.identity, capabilities,
                        _source_identity(prepare, model_identity), paid_boundary,
                        _source_identity(reconciler, model_identity),
                        _source_identity(on_result, model_identity),
                        tuple(codec.identity for codec in reference_codecs),
                        tuple((e.__module__, e.__qualname__) for e in terminal_errors)))
    return ComponentDescriptor(identity, alias, request_type, result_type,
        selected, request_codec, result_codec, tuple(dict.fromkeys((INVOKE,) + capabilities)))


def operation(alias: str, request_type: type[I], result_type: type[O],
              invoke: Callable[[I], O], *, version: str,
              request_codec=None, result_codec=None,
              options: dict | None = None, schema: dict | None = None,
              capabilities: tuple[Capability, ...] = (),
              model_identity: Key | None = None,
              prepare: Callable[[I], _Prepared[I]] | None = None,
              paid_boundary: str = "",
              terminal_errors: tuple[type[Exception], ...] = (),
              reference_codecs: tuple = (),
              reconciler: Callable | None = None,
              on_result: Callable[[O], None] | None = None) -> FunctionOperation[I, O]:
    """Build a function-backed operation without invoking or loading it.

    Version is mandatory; changing options, implementation source, callbacks,
    codecs, or model identity changes the descriptor and conflicts with an
    existing call slot. Stateful callables require a model_identity that also
    identifies the effective captured state.
    """
    descriptor = _describe_operation(
        alias,
        request_type,
        result_type,
        implementation=(invoke,),
        version=version,
        request_codec=request_codec,
        result_codec=result_codec,
        options=options,
        schema=schema,
        capabilities=capabilities,
        model_identity=model_identity,
        prepare=prepare,
        paid_boundary=paid_boundary,
        terminal_errors=terminal_errors,
        reference_codecs=reference_codecs,
        reconciler=reconciler,
        on_result=on_result,
    )
    return FunctionOperation(descriptor, invoke, prepare=prepare, paid_boundary=paid_boundary,
                             terminal_errors=terminal_errors, reference_codecs=reference_codecs,
                             reconciler=reconciler, on_result=on_result)


class OperationOwner:
    """Mixin for an existing family registry; no duplicate backend inventory."""
    _operation_factories = None

    @classmethod
    def register_operation(cls, name: str, factory: Callable):
        if "_operation_factories" not in cls.__dict__ or cls._operation_factories is None:
            cls._operation_factories = {}
        cls._operation_factories[str(name).lower()] = factory

    @classmethod
    def build_operation(cls, name: str, *, alias: str, options: dict, context):
        factories = cls._operation_factories or {}
        try:
            factory = factories[str(name).lower()]
        except KeyError:
            raise ValueError(f"{cls.__name__}: no typed adapter registered for {name!r}") from None
        return factory(alias=alias, options=options, context=context)


class Components:
    """Selected operation bindings; family registries remain their owners."""
    def __init__(self, bindings, *, sampling=None):
        self._bindings = dict(bindings)
        self._public_aliases = tuple(self._bindings)
        for component in tuple(self._bindings.values()):
            initialization = getattr(component, "initialization", None)
            if initialization is not None:
                alias = initialization.descriptor.alias
                if alias in self._bindings and self._bindings[alias] is not initialization:
                    raise ValueError(f"Conflicting initialization owner {alias}")
                self._bindings[alias] = initialization
        self._sampling = dict(sampling or {})
        if any(not isinstance(value, SamplingPolicy) for value in self._sampling.values()):
            raise TypeError("Sampling bindings must be typed SamplingPolicy values")
        for alias, component in self._bindings.items():
            if alias != component.descriptor.alias:
                raise ValueError("Component alias does not match its descriptor")

    @classmethod
    def from_config(cls, config, *, owners=None, context=None):
        from pathlib import Path
        if isinstance(config, (str, Path)):
            import yaml
            config = yaml.safe_load(Path(config).read_text())
        owners = owners or {}
        selected = config.get("components", {})
        bindings = {}
        for alias, specification in selected.items():
            extra = set(specification) - {"family", "type", "options"}
            if extra:
                raise ValueError(f"{alias}: unknown component fields {sorted(extra)}")
            family = specification["family"]
            if family not in owners:
                raise ValueError(f"No owning registry for component family {family!r}")
            bindings[alias] = owners[family].build_operation(specification["type"],
                alias=alias, options=deepcopy(specification.get("options", {})), context=context)
        return cls(bindings)

    def operation(self, alias: str, request: type[I], result: type[O]):
        component = self._bindings[alias]
        descriptor = component.descriptor
        if descriptor.request_type is not request or descriptor.result_type is not result:
            raise TypeError(f"{alias}: requested operation signature differs from its owner")
        return component

    def requirements(self, aliases: tuple[str, ...]) -> tuple[Requirement, ...]:
        for alias in aliases:
            if alias not in self._bindings:
                raise KeyError(alias)
        required = [Requirement(alias, INVOKE, "required") for alias in aliases]
        for alias in aliases:
            initialization = getattr(self._bindings[alias], "initialization", None)
            if initialization is not None:
                item = Requirement(initialization.descriptor.alias, INVOKE, "required")
                if item not in required:
                    required.append(item)
        return tuple(required)

    def selected_options(self):
        return {alias: json.loads(op.descriptor.options_schema.effective_options_json)
                for alias, op in self._bindings.items() if alias in self._public_aliases}

    def sampling(self, alias: str) -> SamplingPolicy:
        return self._sampling[alias]

    def __iter__(self):
        return iter(self._bindings.values())
