"""Typed mutation operations backed by existing native mutator owners.

The native mutator remains responsible for preparing scientific features,
sampling residues, and retaining its model.  This adapter supplies the
canonical boundary: request validation, serialized native inputs, restoration
of existing ``StandardScores``, temporary legacy-state activation, and a
complete assignment proposal.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import inspect
from pathlib import Path
import threading
from typing import Any, Callable

from astrolabe.core import global_config
from astrolabe.mutators.rfd4_types import GenerationResult

from .generated_structure import (
    GENERATED_STRUCTURE_REFERENCE_CODECS,
    bind_generated_structure,
)

from .legacy import (
    LEGACY_REFERENCE_CODECS,
    capture_legacy_candidate,
    finalize_pending_mutations,
)
from .native_mutation_state import (
    NATIVE_MUTATION_REFERENCE_CODECS,
    MutationContextPolicy,
    NativeMutationContextDraft,
    NativeMutationResultState,
    NativePendingMutation,
    NativeScoreContribution,
    UnsupportedMutationContext,
    bind_native_mutation_context,
    capture_native_mutation_context,
    capture_native_mutation_contexts,
    native_mutation_segments,
    restore_native_mutation_context,
)
from .native_scores import NATIVE_SCORES_REFERENCE_CODECS
from .native_support import LazyNativeOwner, numeric_step
from .operations import FunctionOperation, _describe_operation
from .store import DataStore, JSONCodec, design_codec
from .types import (
    Blob,
    Candidate,
    Dependency,
    EvidenceUse,
    Key,
    Proposal,
    Proposals,
    ProposeRequest,
    SamplingPolicy,
    System,
    _Prepared,
    key_for,
    manifest,
)


_RUNTIME_CONFIG_CODEC = JSONCodec()
_SCOPES = frozenset({"mutable", "all"})


class UnsupportedNativeMutation(ValueError):
    """The request cannot be handed to an existing native mutator safely."""


def _unique_codecs(*groups: Sequence[Any]) -> tuple[Any, ...]:
    result = []
    seen: set[str] = set()
    for group in groups:
        for codec in group:
            if codec.identity not in seen:
                seen.add(codec.identity)
                result.append(codec)
    return tuple(result)


def _callable_identity(value: Callable[..., Any] | None) -> Any:
    if value is None:
        return None
    try:
        return inspect.getsource(value)
    except (OSError, TypeError):
        return f"{value.__module__}.{value.__qualname__}"


def _deduplicate_blobs(blobs: Sequence[Blob]) -> tuple[Blob, ...]:
    result = []
    seen: set[tuple[str, str]] = set()
    for blob in blobs:
        if not isinstance(blob, Blob):
            raise TypeError("operation references must be Blob values")
        identity = (blob.codec, blob.identity.digest)
        if identity not in seen:
            seen.add(identity)
            result.append(blob)
    return tuple(result)


def _prepared(
    request: ProposeRequest,
    inputs,
    references: tuple[Blob, ...],
) -> _Prepared[ProposeRequest]:
    return _Prepared(
        request=request,
        inputs=inputs,
        requirements=(),
        paid_boundary="mutate",
        references=references,
    )


def _role_blob(
    prepared: _Prepared[ProposeRequest],
    role: str,
    codec: str,
) -> Blob:
    matches = tuple(
        dependency
        for dependency in prepared.inputs.dependencies
        if dependency.role == role
    )
    if len(matches) != 1:
        raise ValueError(
            f"prepared native mutation requires exactly one {role!r} dependency"
        )
    return Blob(matches[0].identity, codec)


class NativeMutationOperation(FunctionOperation[ProposeRequest, Proposals]):
    """A cold ``ProposeRequest`` operation around a native mutator owner."""

    def __init__(
        self,
        descriptor,
        *,
        factory: Callable[[], Any],
        bindings: Any,
        sampling_policy: SamplingPolicy,
        options: Mapping[str, Any],
        configure: Callable[..., Mapping[str, Any] | None] | None,
        runtime_config_factory: Callable[[ProposeRequest], Mapping[str, Any] | None]
        | None,
        owner_identity: Key,
        native_invoke: Callable[..., Any] | None = None,
        native_invoke_identity: str | None = None,
        mutation_context: MutationContextPolicy | None = None,
        prepare_native: bool = False,
    ) -> None:
        self.factory = factory
        self.bindings = bindings
        self.sampling_policy = sampling_policy
        self.options = dict(options)
        self.configure = configure
        self.runtime_config_factory = runtime_config_factory
        self.native_invoke = native_invoke
        self.native_invoke_identity = native_invoke_identity
        self.mutation_context = mutation_context
        self.prepare_native = prepare_native
        self._execution_lock = threading.RLock()
        self.owner = LazyNativeOwner(
            factory,
            bindings.data,
            owner_identity,
            alias=descriptor.alias,
        )
        super().__init__(
            descriptor,
            lambda request: None,
            prepare=self._prepare,
            paid_boundary="mutate",
            reference_codecs=_unique_codecs(
                LEGACY_REFERENCE_CODECS,
                NATIVE_SCORES_REFERENCE_CODECS,
                GENERATED_STRUCTURE_REFERENCE_CODECS,
                (
                    NATIVE_MUTATION_REFERENCE_CODECS
                    if self.mutation_context is not None
                    else ()
                ),
            ),
        )
        self.initialization = self.owner.initialization
        self.initialization_request = self.owner.initialization_request

    @property
    def protomer_scope(self) -> str:
        return str(self.options["protomer_scope"])

    def _validate_request(self, request: ProposeRequest) -> None:
        if not isinstance(request, ProposeRequest):
            raise TypeError("native mutation requires a ProposeRequest")
        if request.system is None or not isinstance(request.system, System):
            raise TypeError("ProposeRequest.system must be a System")
        if request.sampling != self.sampling_policy:
            raise UnsupportedNativeMutation(
                "request sampling identity/options do not match the declared native policy"
            )
        if request.selection:
            raise UnsupportedNativeMutation(
                "native mutation does not yet support typed residue Selection"
            )
        if request.packed is not None:
            raise UnsupportedNativeMutation(
                "native mutation does not support packed multistate scores"
            )
        if len(request.evidence) != 1:
            raise UnsupportedNativeMutation(
                "native mutation requires exactly one EvidenceUse; one Evidence "
                "may cover multiple assemblies"
            )
        use = request.evidence[0]
        if not isinstance(use, EvidenceUse):
            raise TypeError("request.evidence must contain EvidenceUse values")
        if not request.parents:
            raise UnsupportedNativeMutation(
                "native mutation requires at least one parent Candidate"
            )
        numeric_step(request.sample)

        bindings = self.bindings
        if not callable(getattr(bindings, "validate", None)):
            raise TypeError("bindings must provide validate(system, candidate)")
        for candidate in request.parents:
            bindings.validate(request.system, candidate)
        bindings.validate(request.system, request.working)

    def _validate_context_representability(self) -> None:
        if self.mutation_context is None:
            return
        try:
            segments = native_mutation_segments()
            if any(getattr(segment, "length_bounds", None) is not None
                   or getattr(segment, "pending_sequence", None) is not None
                   for _, segment in segments):
                raise UnsupportedMutationContext(
                    "complete variable-length proposals cannot be reconstructed by the substitution-only mutation context codec"
                )
            manager = global_config.segment_manager
            if getattr(manager, "_pre_update_snapshot", None) is not None:
                raise UnsupportedMutationContext(
                    "typed mutation context requires disabled pre-update stack"
                )
        except UnsupportedMutationContext as error:
            raise UnsupportedNativeMutation(
                f"native mutation context is not representable: {error}"
            ) from error
        if not segments:
            raise UnsupportedNativeMutation(
                "native mutation context requires at least one polymer segment"
            )

    def _runtime_config(self, request: ProposeRequest) -> Mapping[str, Any]:
        if self.runtime_config_factory is None:
            config = {}
        else:
            config = self.runtime_config_factory(request)
            if config is None:
                config = {}
        if not isinstance(config, Mapping):
            raise TypeError(
                "runtime_config_factory must return a mapping or None"
            )
        config = dict(config)
        # Sample.index is a job identity. A scientific schedule can declare a
        # different step in the frozen owner runtime configuration.
        config.setdefault("step", numeric_step(request.sample))
        if type(config["step"]) is not int or config["step"] < 0:
            raise ValueError("Native mutator runtime step must be a non-negative integer")
        return config

    @staticmethod
    def _references_for(
        request: ProposeRequest,
        runtime_blob: Blob,
    ) -> tuple[Blob, ...]:
        references: list[Blob] = [
            request.system.topology,
            request.sampling.options,
            runtime_blob,
        ]
        references.extend(
            occurrence.index_map for occurrence in request.system.occurrences
        )
        for candidate in (request.working, *request.parents):
            for assignment in candidate.assignments:
                graph = getattr(assignment.value, "graph", None)
                if isinstance(graph, Blob):
                    references.append(graph)
        for use in request.evidence:
            evidence = use.evidence
            references.extend(
                artifact.content for artifact in evidence.artifacts
            )
            for measurement in evidence.measurements:
                if measurement.values is not None:
                    references.append(measurement.values)
                if measurement.valid_mask is not None:
                    references.append(measurement.valid_mask)
                references.extend(axis.index_map for axis in measurement.axes)
        return _deduplicate_blobs(references)

    def _prepare(self, request: ProposeRequest) -> _Prepared[ProposeRequest]:
        self._validate_request(request)
        self._validate_context_representability()
        if not self.owner.ready:
            raise UnsupportedNativeMutation(
                "native mutator initialization must complete before proposal preparation"
            )
        data = self.bindings.data
        if not isinstance(data, DataStore):
            raise TypeError("bindings.data must be a design DataStore")

        runtime_config = self._runtime_config(request)
        runtime_blob = data.put(runtime_config, _RUNTIME_CONFIG_CODEC)
        request_encoded = design_codec.encode(request)
        dependencies = (
            Dependency(
                "request",
                key_for((design_codec.identity, request_encoded.hex())),
            ),
            Dependency("system", request.system.identity),
            Dependency("bindings", self.bindings.translation.system.identity),
            Dependency("parents", key_for(tuple(item.identity for item in request.parents))),
            Dependency("working", request.working.identity),
            Dependency(
                "evidence",
                key_for(tuple(use.evidence.identity for use in request.evidence)),
            ),
            Dependency("sample", key_for(request.sample)),
            Dependency(
                "sampling-policy",
                key_for(
                    (
                        "native-mutation-sampling/v1",
                        request.sampling.identity,
                        request.sampling.options.identity,
                    )
                ),
            ),
            Dependency("runtime-config", runtime_blob.identity),
            *(
                (
                    Dependency(
                        "invoke",
                        key_for(
                            (
                                "native-mutation-invoke/v1",
                                self.native_invoke_identity,
                            )
                        ),
                    ),
                )
                if self.native_invoke is not None
                else ()
            ),
            Dependency(
                "model",
                key_for(
                    (
                        "native-mutation-model-input/v1",
                        self.owner.identity,
                        self.options,
                    )
                ),
            ),
        )
        references = self._references_for(request, runtime_blob)
        owner = self.owner.get() if self.prepare_native else None
        prepare_native = getattr(owner, "prepare_native_mutation", None)
        if self.prepare_native and not callable(prepare_native):
            raise TypeError("prepared native owner must provide prepare_native_mutation()")
        if callable(prepare_native):
            runtime_config.setdefault("seed", request.sample.seed)
            if self.native_invoke is not None:
                raise UnsupportedNativeMutation(
                    "prepared native mutation cannot use an independent custom invoker"
                )
            with self._execution_lock, self.bindings.activate(request.working):
                scores = self.bindings.scores_for(request.evidence[0].evidence)
                if self.configure is not None:
                    configured = self.configure(
                        owner, runtime_config, request=request, sample=request.sample,
                        step=runtime_config["step"], work_root=Path(self.bindings.work_root),
                    )
                    if configured is not None:
                        if not isinstance(configured, Mapping):
                            raise TypeError("configure callback must return a mapping or None")
                        runtime_config = dict(configured)
                prepared_native = prepare_native(self._scope_mapping(), scores, runtime_config)
            owner_blob = data.put(prepared_native, _RUNTIME_CONFIG_CODEC)
            runtime_blob = data.put(runtime_config, _RUNTIME_CONFIG_CODEC)
            dependencies = tuple(
                Dependency("runtime-config", runtime_blob.identity)
                if dependency.role == "runtime-config" else dependency
                for dependency in dependencies
            ) + (Dependency("native-prepared", owner_blob.identity),)
            references = self._references_for(request, runtime_blob) + (owner_blob,)
        return _prepared(
            request,
            manifest(dependencies),
            references,
        )

    def _scope_mapping(self) -> Mapping[str, Any]:
        if self.protomer_scope == "all":
            protomers = global_config.protomers
        else:
            protomers = global_config.mutable_protomers
        if not isinstance(protomers, Mapping) or not protomers:
            raise UnsupportedNativeMutation(
                f"legacy {self.protomer_scope} protomer scope is unavailable or empty"
            )
        return protomers

    @staticmethod
    def _replacements(
        working: Candidate,
        captured: Candidate,
    ) -> tuple[Any, ...]:
        from .types import Assignment

        if captured.system != working.system:
            raise UnsupportedNativeMutation(
                "native mutation changed the molecular System topology"
            )
        working_values = {
            assignment.variable: assignment.value
            for assignment in working.assignments
        }
        replacements = tuple(
            Assignment(assignment.variable, assignment.value)
            for assignment in captured.assignments
            if working_values.get(assignment.variable) != assignment.value
        )
        return replacements

    @staticmethod
    def _validate_nonpolymer_assignments_unchanged(
        base: Candidate,
        proposed: Candidate,
    ) -> None:
        from .types import Polymer

        def nonpolymer(candidate: Candidate) -> dict[str, Any]:
            return {
                assignment.variable: assignment.value
                for assignment in candidate.assignments
                if not isinstance(assignment.value, Polymer)
            }

        if nonpolymer(base) != nonpolymer(proposed):
            raise UnsupportedNativeMutation(
                "durable mutation context cannot preserve changed "
                "non-polymer assignments"
            )

    def _run_native(
        self,
        request: ProposeRequest,
        prepared: _Prepared[ProposeRequest],
    ) -> tuple[Candidate, tuple[NativeMutationContextDraft, ...], GenerationResult | None]:
        data = self.bindings.data
        runtime_blob = _role_blob(
            prepared,
            "runtime-config",
            _RUNTIME_CONFIG_CODEC.identity,
        )
        runtime_config = data.get(runtime_blob, _RUNTIME_CONFIG_CODEC)
        if not isinstance(runtime_config, dict):
            raise TypeError("persisted native runtime config must decode to a mapping")

        use = request.evidence[0]
        scores = self.bindings.scores_for(use.evidence)
        mutator = self.owner.get()
        if not self.prepare_native and callable(getattr(mutator, "prepare_native_mutation", None)):
            raise UnsupportedNativeMutation(
                "this native owner requires prepare_native=True to bind assets before Runtime lookup"
            )
        if self.native_invoke is None and not callable(
            getattr(mutator, "mutate", None)
        ):
            raise TypeError("native mutator factory must return an object with mutate()")

        step = runtime_config["step"]
        has_native_prepared = any(
            dependency.role == "native-prepared" for dependency in prepared.inputs.dependencies
        )
        if self.configure is not None and not has_native_prepared:
            configured = self.configure(
                mutator,
                runtime_config,
                request=request,
                sample=request.sample,
                step=step,
                work_root=Path(getattr(self.bindings, "work_root")),
            )
            if configured is not None:
                if not isinstance(configured, Mapping):
                    raise TypeError(
                        "configure callback must mutate its config or return a mapping"
                    )
                runtime_config = dict(configured)

        selected_protomers = self._scope_mapping()
        response = None
        if has_native_prepared:
            owner_blob = _role_blob(prepared, "native-prepared", _RUNTIME_CONFIG_CODEC.identity)
            response = mutator.mutate_prepared(
                data.get(owner_blob, _RUNTIME_CONFIG_CODEC),
                selected_protomers, scores, runtime_config,
            )
        elif self.native_invoke is None:
            response = mutator.mutate(
                selected_protomers,
                scores,
                runtime_config,
            )
        else:
            response = self.native_invoke(
                mutator,
                selected_protomers,
                scores,
                runtime_config,
            )
            if response is not None and not isinstance(response, GenerationResult):
                raise TypeError("native mutator invoker must return None")
        drafts: tuple[NativeMutationContextDraft, ...] = ()
        if self.mutation_context is not None:
            try:
                drafts = capture_native_mutation_contexts(
                    data,
                    policy=self.mutation_context,
                    runtime_config=runtime_config,
                )
            except UnsupportedMutationContext as error:
                raise UnsupportedNativeMutation(
                    f"native mutation context capture failed: {error}"
                ) from error
        # Every segment must be captured successfully before any pending
        # mutation is finalized into the working legacy sequence.
        finalize_pending_mutations()

        captured = capture_legacy_candidate(
            assemblies=tuple(
                assembly.identity
                for assembly in self.bindings.translation.assemblies
            ),
            state=self.bindings.translation.state,
            data=data,
        )
        if self.mutation_context is not None:
            self._validate_nonpolymer_assignments_unchanged(
                request.working,
                captured.candidate,
            )
        replacements = self._replacements(
            request.working,
            captured.candidate,
        )
        candidate = request.system.edit(
            request.working,
            replacements,
            data=data,
        )
        return candidate, drafts, response if isinstance(response, GenerationResult) else None

    def invoke(self, prepared: _Prepared[ProposeRequest]) -> Proposals:
        if not isinstance(prepared, _Prepared):
            raise TypeError("native mutation invoke requires _Prepared input")
        request = prepared.request
        self._validate_request(request)
        data = self.bindings.data
        if not isinstance(data, DataStore):
            raise TypeError("bindings.data must be a design DataStore")

        with self._execution_lock:
            activation = self.bindings.activate(request.working)
            if not hasattr(activation, "__enter__") or not hasattr(
                activation, "__exit__"
            ):
                raise TypeError(
                    "bindings.activate(candidate) must return a context manager"
                )
            with activation:
                candidate, drafts, generation = self._run_native(request, prepared)

        parents = tuple(item.identity for item in request.parents)
        used_evidence = tuple(
            use.evidence.identity for use in request.evidence
        )
        artifacts = tuple(
            bind_native_mutation_context(
                draft,
                data=data,
                system=request.system,
                base=request.working,
                proposed=candidate,
                sample=request.sample,
            )[1]
            for draft in drafts
        )
        if generation is not None:
            artifacts += (bind_generated_structure(
                generation, data=data, system=request.system, base=request.working,
                proposed=candidate, sample=request.sample, inputs=prepared.inputs,
            ),)
        identity_payload = (
            "proposal/native-mutation/v1",
            self.descriptor.identity,
            prepared.inputs,
            parents,
            request.working.identity,
            candidate.identity,
            used_evidence,
            request.sampling.identity,
        )
        if artifacts:
            identity_payload += (artifacts,)
        proposal = Proposal(
            identity=key_for(identity_payload),
            parents=parents,
            base=request.working.identity,
            candidate=candidate,
            used_evidence=used_evidence,
            protocol=request.sampling.identity,
            artifacts=artifacts,
        )
        return Proposals((proposal,))

    def shutdown(self) -> None:
        with self._execution_lock:
            self.owner.shutdown()


def _mutation_schema(
    schema: Mapping[str, Any] | None,
) -> dict[str, Any]:
    result = (
        {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }
        if schema is None
        else dict(schema)
    )
    if result.get("type") != "object":
        raise ValueError("native mutation options schema must describe an object")
    properties = result.get("properties")
    if properties is None:
        properties = {}
    elif not isinstance(properties, Mapping):
        raise ValueError("native mutation schema properties must be a mapping")
    else:
        properties = dict(properties)
    properties["protomer_scope"] = {
        "type": "string",
        "enum": ["mutable", "all"],
    }
    result["properties"] = properties
    return result


def make_mutator_operation(
    alias: str,
    factory: Callable[[], Any],
    bindings: Any,
    *,
    sampling_policy: SamplingPolicy,
    version: str,
    model_identity: Key,
    options: Mapping[str, Any] | None = None,
    schema: Mapping[str, Any] | None = None,
    configure: Callable[..., Mapping[str, Any] | None] | None = None,
    configure_identity: str | None = None,
    runtime_config_factory: Callable[[ProposeRequest], Mapping[str, Any] | None]
    | None = None,
    invoke: Callable[..., Any] | None = None,
    invoke_identity: str | None = None,
    mutation_context: MutationContextPolicy | None = None,
    prepare_native: bool = False,
) -> FunctionOperation[ProposeRequest, Proposals]:
    """Build a cold native mutation operation.

    ``factory`` is never called while constructing the operation. The shared
    ``LazyNativeOwner`` constructs it in the explicit initialization operation
    and can reconstruct it after restart before a new inference.
    ``configure(mutator, runtime_config, *, request, sample, step, work_root)``
    runs after model construction and immediately before ``mutate``.
    ``runtime_config_factory(request)`` is evaluated during preparation so its
    exact JSON value is journaled and available to a cold rebuild.
    ``prepare_native=True`` opts an owner into pre-cache preparation through
    ``prepare_native_mutation`` and invocation through ``mutate_prepared``.
    Ordinary native owners remain lazy during journal replay.
    """

    if not isinstance(alias, str) or not alias.strip():
        raise ValueError("alias must be a non-empty string")
    if not callable(factory):
        raise TypeError("factory must be callable")
    if not isinstance(sampling_policy, SamplingPolicy):
        raise TypeError("sampling_policy must be a SamplingPolicy")
    if not isinstance(version, str) or not version:
        raise ValueError("version must be a non-empty string")
    if not isinstance(model_identity, Key):
        raise TypeError("model_identity must be a content Key")
    if (configure is None) != (configure_identity is None):
        raise ValueError(
            "configure and configure_identity must be supplied together"
        )
    if configure_identity is not None and (
        not isinstance(configure_identity, str) or not configure_identity.strip()
    ):
        raise ValueError("configure_identity must be a non-empty string")
    if runtime_config_factory is not None and not callable(runtime_config_factory):
        raise TypeError("runtime_config_factory must be callable")
    if (invoke is None) != (invoke_identity is None):
        raise ValueError("invoke and invoke_identity must be supplied together")
    if invoke is not None and not callable(invoke):
        raise TypeError("invoke must be callable")
    if invoke_identity is not None and (
        not isinstance(invoke_identity, str) or not invoke_identity.strip()
    ):
        raise ValueError("invoke_identity must be a non-empty string")
    if mutation_context is not None and not isinstance(
        mutation_context, MutationContextPolicy
    ):
        raise TypeError("mutation_context must be a MutationContextPolicy or None")
    if type(prepare_native) is not bool:
        raise TypeError("prepare_native must be boolean")

    selected_options = {"protomer_scope": "mutable"}
    selected_options.update(dict(options or {}))
    scope = selected_options.get("protomer_scope")
    if scope not in _SCOPES:
        raise ValueError("protomer_scope must be 'mutable' or 'all'")

    effective_schema = _mutation_schema(schema)
    owner_identity = key_for(
        (
            "native-mutator-owner/v1",
            *(("prepared-owner/v1",) if prepare_native else ()),
            model_identity,
            selected_options,
            sampling_policy,
            configure_identity,
            _callable_identity(factory),
            _callable_identity(configure),
            _callable_identity(runtime_config_factory),
            invoke_identity,
            _callable_identity(invoke),
            effective_schema,
            bindings.translation.system.identity,
            bindings.translation.candidate.identity,
            (
                None
                if mutation_context is None
                else (
                    mutation_context.descriptor(),
                )
            ),
        )
    )

    implementations = [
        NativeMutationOperation,
        LazyNativeOwner,
        factory,
    ]
    if configure is not None:
        implementations.append(configure)
    if runtime_config_factory is not None:
        implementations.append(runtime_config_factory)
    if invoke is not None:
        implementations.append(invoke)
    if mutation_context is not None:
        implementations.extend(
            (
                mutation_context.__class__,
                NativeMutationContextDraft,
                NativeMutationResultState,
                NativePendingMutation,
                NativeScoreContribution,
                capture_native_mutation_contexts,
                capture_native_mutation_context,
                native_mutation_segments,
                bind_native_mutation_context,
                restore_native_mutation_context,
            )
        )

    descriptor = _describe_operation(
        alias,
        ProposeRequest,
        Proposals,
        implementation=tuple(implementations),
        version=version,
        request_codec=design_codec,
        result_codec=design_codec,
        options=selected_options,
        schema=effective_schema,
        model_identity=owner_identity,
        prepare=NativeMutationOperation._prepare,
        paid_boundary="mutate",
        reference_codecs=_unique_codecs(
            LEGACY_REFERENCE_CODECS,
            NATIVE_SCORES_REFERENCE_CODECS,
            GENERATED_STRUCTURE_REFERENCE_CODECS,
            (
                NATIVE_MUTATION_REFERENCE_CODECS
                if mutation_context is not None
                else ()
            ),
        ),
    )
    return NativeMutationOperation(
        descriptor,
        factory=factory,
        bindings=bindings,
        sampling_policy=sampling_policy,
        options=selected_options,
        configure=configure,
        runtime_config_factory=runtime_config_factory,
        owner_identity=owner_identity,
        native_invoke=invoke,
        native_invoke_identity=invoke_identity,
        mutation_context=mutation_context,
        prepare_native=prepare_native,
    )


__all__ = [
    "NativeMutationOperation",
    "UnsupportedNativeMutation",
    "make_mutator_operation",
]
