"""Typed adapter for existing evaluator/model owners.

The operation owns only the live evaluator object and an explicit ``shutdown``
method.  Recovery-critical inputs are persisted as immutable ``DataStore``
blobs and are addressed through the prepared manifest, never through a private
live request cache.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
import inspect
import json
import math
from pathlib import Path
import threading
from typing import Any, Callable

from astrolabe.core.cyclization import CyclicRelativeOffset, CyclicRelativePatch
from astrolabe.core.folding_input import DnaChain, Input, Ligand, ProteinChain
from astrolabe.core.folding_input import RnaChain, Template

from .legacy import LEGACY_REFERENCE_CODECS, standard_scores_to_evidence
from .native_scores import (
    NATIVE_SCORES_ENVELOPE_CODEC,
    NATIVE_SCORES_REFERENCE_CODECS,
    snapshot_scores,
)
from .operations import FunctionOperation
from .owner import SCIENTIFIC_OWNER
from .store import DataStore, JSONCodec
from .types import (
    Artifact,
    Blob,
    Candidate,
    Dependency,
    Evidence,
    Key,
    PredictRequest,
    System,
    _Prepared,
    key_for,
    manifest,
)


class UnsupportedNativePrediction(ValueError):
    """The selected native evaluator cannot honor a typed request safely."""


_FOLDING_TYPES = {
    "astrolabe.core.folding_input.Input": Input,
    "astrolabe.core.folding_input.ProteinChain": ProteinChain,
    "astrolabe.core.folding_input.RnaChain": RnaChain,
    "astrolabe.core.folding_input.DnaChain": DnaChain,
    "astrolabe.core.folding_input.Ligand": Ligand,
    "astrolabe.core.folding_input.Template": Template,
    "astrolabe.core.cyclization.CyclicRelativeOffset": CyclicRelativeOffset,
    "astrolabe.core.cyclization.CyclicRelativePatch": CyclicRelativePatch,
}


class FoldingInputCodec:
    """Safe canonical codec for inline folding inputs and topology metadata."""

    identity = "astrolabe.design.folding-input/v1"

    def encode(self, value: Mapping[str, Input]) -> bytes:
        if not isinstance(value, Mapping):
            raise TypeError("folding inputs must be an assembly -> Input mapping")
        if any(not isinstance(name, str) or not name for name in value):
            raise ValueError("folding assembly names must be non-empty strings")
        if any(not isinstance(item, Input) for item in value.values()):
            raise TypeError("folding input values must be Input instances")
        return json.dumps(
            self._encode_value(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")

    def decode(self, data: bytes) -> dict[str, Input]:
        if not isinstance(data, bytes):
            raise TypeError("encoded folding inputs must be bytes")
        value = self._decode_value(
            json.loads(
                data.decode("utf-8"),
                parse_constant=lambda item: (_ for _ in ()).throw(
                    ValueError(f"invalid JSON constant {item!r}")
                ),
            )
        )
        if not isinstance(value, dict) or any(
            not isinstance(name, str) or not isinstance(item, Input)
            for name, item in value.items()
        ):
            raise ValueError("decoded folding inputs are not assembly -> Input")
        return value

    def references(self, value: Mapping[str, Input]) -> tuple[Blob, ...]:
        if not isinstance(value, Mapping):
            raise TypeError("folding inputs must be a mapping")
        return ()

    def compatibility_key(self) -> tuple[str, str]:
        return (type(self).__name__, self.identity)

    def _encode_value(self, value: Any) -> Any:
        if value is None or type(value) in {str, bool, int}:
            return value
        if type(value) is float:
            if not math.isfinite(value):
                raise UnsupportedNativePrediction(
                    "non-finite floating-point folding metadata is unsupported"
                )
            return value
        if isinstance(value, bytes):
            return {"$bytes": base64.b64encode(value).decode("ascii")}
        if isinstance(value, tuple):
            return {"$tuple": [self._encode_value(item) for item in value]}
        if isinstance(value, list):
            return {"$list": [self._encode_value(item) for item in value]}
        if isinstance(value, Mapping):
            entries = []
            for key, item in value.items():
                entries.append([self._encode_value(key), self._encode_value(item)])
            return {"$mapping": entries}
        if is_dataclass(value):
            tag = f"{type(value).__module__}.{type(value).__qualname__}"
            if tag not in _FOLDING_TYPES:
                raise UnsupportedNativePrediction(
                    f"unsupported folding-input dataclass {tag!r}"
                )
            return {
                "$dataclass": tag,
                "fields": {
                    field.name: self._encode_value(getattr(value, field.name))
                    for field in fields(value)
                    if not (isinstance(value, Input) and field.name == "constraint_ir"
                            and value.constraint_ir is None)
                },
            }
        raise UnsupportedNativePrediction(
            "unsupported folding-input value: "
            f"{type(value).__module__}.{type(value).__qualname__}"
        )

    def _decode_value(self, value: Any) -> Any:
        if value is None or type(value) in {str, bool, int, float}:
            return value
        if not isinstance(value, dict):
            raise ValueError("invalid tagged folding-input value")
        if set(value) == {"$bytes"}:
            return base64.b64decode(value["$bytes"].encode("ascii"), validate=True)
        if set(value) == {"$tuple"}:
            return tuple(self._decode_value(item) for item in value["$tuple"])
        if set(value) == {"$list"}:
            return [self._decode_value(item) for item in value["$list"]]
        if set(value) == {"$mapping"}:
            result = {}
            for entry in value["$mapping"]:
                if not isinstance(entry, list) or len(entry) != 2:
                    raise ValueError("invalid folding-input mapping entry")
                key = self._decode_value(entry[0])
                try:
                    hash(key)
                except TypeError as error:
                    raise ValueError("folding-input mapping keys must be hashable") from error
                if key in result:
                    raise ValueError("duplicate folding-input mapping key")
                result[key] = self._decode_value(entry[1])
            return result
        if set(value) == {"$dataclass", "fields"}:
            tag = value["$dataclass"]
            value_type = _FOLDING_TYPES.get(tag)
            if value_type is None:
                raise UnsupportedNativePrediction(
                    f"unknown folding-input dataclass tag {tag!r}"
                )
            expected = {field.name for field in fields(value_type)}
            encoded_fields = value["fields"]
            legacy_input_fields = expected - {"constraint_ir"} if value_type is Input else expected
            if set(encoded_fields) not in (expected, legacy_input_fields):
                raise ValueError(f"folding-input fields changed for {tag!r}")
            return value_type(
                **{
                    name: self._decode_value(item)
                    for name, item in encoded_fields.items()
                }
            )
        raise ValueError("unknown tagged folding-input value")


FOLDING_INPUT_CODEC = FoldingInputCodec()


def _unique_codecs(*groups: Sequence[Any]) -> tuple[Any, ...]:
    result = []
    seen = set()
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


def _make_prepared(
    request: PredictRequest,
    inputs_manifest,
    references: tuple[Blob, ...],
    paid_boundary: str,
) -> _Prepared[PredictRequest]:
    values = {
        "request": request,
        "inputs": inputs_manifest,
        "requirements": (),
        "paid_boundary": paid_boundary,
    }
    parameters = inspect.signature(_Prepared).parameters
    if "references" in parameters:
        values["references"] = references
    return _Prepared(**values)


def _binding_system(bindings: Any) -> System:
    system = getattr(bindings, "system", None)
    if system is not None:
        return system
    translation = getattr(bindings, "translation", None)
    system = getattr(translation, "system", None)
    if system is None:
        raise TypeError("bindings must expose system or translation.system")
    return system


class NativePredictionOperation(FunctionOperation[PredictRequest, Evidence]):
    """FunctionOperation with lazy native owner and manifest-addressed inputs."""

    def __init__(
        self,
        descriptor,
        *,
        factory: Callable[[], Any],
        bindings: Any,
        options: Mapping[str, Any],
        configure: Callable[..., Mapping[str, Any] | None] | None,
        configure_identity: str | None,
        native_invoke: Callable[..., Any] | None,
        native_invoke_identity: str | None,
        supplemental_artifacts: Callable[[Any, PredictRequest], tuple[Artifact, ...]] | None,
        supplemental_identity: Key | None,
        model_identity: Key,
        owner: Any,
    ) -> None:
        self.factory = factory
        self.bindings = bindings
        self.options = dict(options)
        self.configure = configure
        self.configure_identity = configure_identity
        self.native_invoke = native_invoke
        self.native_invoke_identity = native_invoke_identity
        self.supplemental_artifacts = supplemental_artifacts
        self.supplemental_identity = supplemental_identity
        self.model_identity = model_identity
        self.owner = owner
        self._execution_lock = threading.RLock()
        super().__init__(
            descriptor,
            self.invoke,
            prepare=self._prepare,
            paid_boundary="predict",
            reference_codecs=_unique_codecs(
                LEGACY_REFERENCE_CODECS,
                NATIVE_SCORES_REFERENCE_CODECS,
                (FOLDING_INPUT_CODEC,),
            ),
        )
        self.initialization = owner.initialization
        self.initialization_request = owner.initialization_request
        self.shutdown = owner.shutdown

    def _validate_request(self, request: PredictRequest) -> Any:
        if len(request.samples) != 1:
            raise UnsupportedNativePrediction(
                "native evaluator operations require exactly one requested Sample"
            )
        if request.conditioning:
            raise UnsupportedNativePrediction(
                "native evaluator adapter has no owner handler for conditioning"
            )
        assemblies = tuple(str(name) for name in request.assemblies)
        if not assemblies or any(not name for name in assemblies):
            raise UnsupportedNativePrediction(
                "native evaluator operation requires selected assemblies"
            )
        if len(set(assemblies)) != len(assemblies):
            raise UnsupportedNativePrediction("selected assemblies must be unique")
        known = {occurrence.assembly for occurrence in request.system.occurrences}
        unknown = set(assemblies) - known
        if unknown:
            raise UnsupportedNativePrediction(
                f"assemblies are absent from the System: {sorted(unknown)!r}"
            )
        bindings = self.bindings
        if not callable(getattr(bindings, "validate", None)):
            raise TypeError("bindings must provide validate(system, candidate)")
        bindings.validate(request.system, request.candidate)
        return assemblies

    def _prepare(self, request: PredictRequest) -> _Prepared[PredictRequest]:
        assemblies = self._validate_request(request)
        if not self.owner.ready:
            raise UnsupportedNativePrediction(
                "native model initialization must complete before inference"
            )
        sample = request.samples[0]
        numeric_step = getattr(self.bindings, "numeric_step", None)
        if callable(numeric_step):
            step = numeric_step(sample)
        else:
            from .native_support import numeric_step as binding_numeric_step

            try:
                step = binding_numeric_step(sample)
            except ValueError as error:
                raise UnsupportedNativePrediction(str(error)) from error
        activation = self.bindings.activate(request.candidate)
        if not hasattr(activation, "__enter__") or not hasattr(activation, "__exit__"):
            raise TypeError("bindings.activate(candidate) must return a context manager")
        with activation:
            from astrolabe.utils.base import construct_standard_input

            folding_inputs = construct_standard_input(assemblies, int(sample.seed))
            runtime_config = {"step": step}
            data = getattr(self.bindings, "data", None)
            if not isinstance(data, DataStore):
                raise TypeError("bindings.data must be a design DataStore")
            folding_blob = data.put(folding_inputs, FOLDING_INPUT_CODEC)
            runtime_config_blob = data.put(runtime_config, JSONCodec())

        dependencies = (
            Dependency("system", request.system.identity),
            Dependency("candidate", request.candidate.identity),
            Dependency("sample", key_for(sample)),
            Dependency("folding-input", folding_blob.identity),
            Dependency("runtime-config", runtime_config_blob.identity),
            Dependency("assemblies", key_for(assemblies)),
            Dependency(
                "configure",
                key_for(("native-prediction-configure/v1", self.configure_identity)),
            ),
            Dependency(
                "supplemental-artifacts",
                key_for(
                    (
                        "native-prediction-supplemental/v1",
                        self.supplemental_identity,
                    )
                ),
            ),
            *(
                (
                    Dependency(
                        "invoke",
                        key_for(
                            (
                                "native-prediction-invoke/v1",
                                self.native_invoke_identity,
                            )
                        ),
                    ),
                )
                if self.native_invoke is not None
                else ()
            ),
            Dependency("model", self.model_identity),
        )
        inputs_manifest = manifest(dependencies)
        return _make_prepared(
            request,
            inputs_manifest,
            (folding_blob, runtime_config_blob),
            self.paid_boundary,
        )

    @staticmethod
    def _blob_for_role(
        prepared: _Prepared[PredictRequest],
        role: str,
        codec: Any,
    ) -> Blob:
        dependencies = [
            dependency
            for dependency in prepared.inputs.dependencies
            if dependency.role == role
        ]
        if len(dependencies) != 1:
            raise ValueError(f"prepared input requires exactly one {role!r} dependency")
        return Blob(dependencies[0].identity, codec.identity)

    def invoke(self, prepared: _Prepared[PredictRequest]) -> Evidence:
        request = prepared.request
        data = getattr(self.bindings, "data", None)
        if not isinstance(data, DataStore):
            raise TypeError("bindings.data must be a design DataStore")
        folding_blob = self._blob_for_role(prepared, "folding-input", FOLDING_INPUT_CODEC)
        runtime_blob = self._blob_for_role(prepared, "runtime-config", JSONCodec())
        folding_inputs = data.get(folding_blob, FOLDING_INPUT_CODEC)
        runtime_config = data.get(runtime_blob, JSONCodec())
        sample = request.samples[0]

        activation = self.bindings.activate(request.candidate)
        if not hasattr(activation, "__enter__") or not hasattr(activation, "__exit__"):
            raise TypeError("bindings.activate(candidate) must return a context manager")
        with activation, self._execution_lock, SCIENTIFIC_OWNER:
            if not self.owner.ready:
                raise UnsupportedNativePrediction(
                    "native model initialization must complete before inference"
                )
            evaluator = self.owner.get()
            if self.native_invoke is None and not callable(
                getattr(evaluator, "evaluate", None)
            ):
                raise TypeError("native owner must return an evaluator with evaluate()")
            if self.configure is not None:
                configured = self.configure(evaluator, request)
                if configured is not None:
                    if not isinstance(configured, Mapping):
                        raise TypeError(
                            "configure callback must return a mapping or None"
                        )
                    runtime_config = dict(configured)
            if self.native_invoke is None:
                scores = evaluator.evaluate(folding_inputs, runtime_config)
            else:
                scores = self.native_invoke(
                    evaluator,
                    folding_inputs,
                    runtime_config,
                )
            from astrolabe.evaluators.utils import StandardScores

            if type(scores) is not StandardScores:
                raise TypeError("native evaluator must return StandardScores unchanged")

            supplemental = ()
            if self.supplemental_artifacts is not None:
                supplemental = self.supplemental_artifacts(scores, request)
                if not isinstance(supplemental, tuple) or any(
                    not isinstance(artifact, Artifact)
                    for artifact in supplemental
                ):
                    raise TypeError(
                        "supplemental_artifacts must return tuple[Artifact, ...]"
                    )
                for artifact in supplemental:
                    if artifact.candidate != request.candidate.identity:
                        raise ValueError(
                            "supplemental artifact belongs to another candidate"
                        )
                    if artifact.sample != sample:
                        raise ValueError(
                            "supplemental artifact belongs to another result sample"
                        )

            score_blob = snapshot_scores(scores, data)
            selected_occurrences = tuple(
                occurrence
                for occurrence in request.system.occurrences
                if occurrence.assembly in set(request.assemblies)
            )
            score_artifact = Artifact(
                identity=key_for(
                    (
                        "artifact/native-scores/v1",
                        request.candidate.identity,
                        selected_occurrences,
                        sample,
                        score_blob,
                        prepared.inputs,
                    )
                ),
                candidate=request.candidate.identity,
                occurrences=selected_occurrences,
                sample=sample,
                content=score_blob,
                derived_from=tuple(source.identity for source in request.sources),
                inputs=prepared.inputs,
            )
            evidence = standard_scores_to_evidence(
                scores,
                request.system,
                request.candidate,
                data,
                samples=sample,
                state=self.descriptor.alias,
                producer_label=self.descriptor.alias,
            )
        combined_inputs = manifest(
            tuple(evidence.inputs.dependencies) + tuple(prepared.inputs.dependencies)
        )
        combined = Evidence(
            identity=key_for(("evidence-pending/v1",)),
            evaluated_candidate=request.candidate.identity,
            artifacts=tuple(evidence.artifacts) + tuple(supplemental) + (score_artifact,),
            measurements=(),
            inputs=combined_inputs,
        )
        return combined.with_measurements(tuple(evidence.measurements))

def make_evaluator_operation(
    alias: str,
    factory: Callable[[], Any],
    bindings: Any,
    *,
    version: str,
    model_identity: Key,
    options: Mapping[str, Any] | None = None,
    schema: Mapping[str, Any] | None = None,
    configure: Callable[..., Mapping[str, Any] | None] | None = None,
    configure_identity: str | None = None,
    invoke: Callable[..., Any] | None = None,
    invoke_identity: str | None = None,
    supplemental_artifacts: Callable[[Any, PredictRequest], tuple[Artifact, ...]] | None = None,
    supplemental_identity: Key | None = None,
) -> FunctionOperation[PredictRequest, Evidence]:
    """Build a cold native evaluator operation without loading the model."""

    if not callable(factory):
        raise TypeError("factory must be callable")
    if not isinstance(model_identity, Key):
        raise TypeError("model_identity must be a content Key")
    if (configure is None) != (configure_identity is None):
        raise ValueError(
            "configure and configure_identity must be supplied together"
        )
    if (invoke is None) != (invoke_identity is None):
        raise ValueError("invoke and invoke_identity must be supplied together")
    if invoke is not None and not callable(invoke):
        raise TypeError("invoke must be callable")
    if (supplemental_artifacts is None) != (supplemental_identity is None):
        raise ValueError(
            "supplemental_artifacts and supplemental_identity must be supplied together"
        )
    if configure_identity is not None and (
        not isinstance(configure_identity, str) or not configure_identity.strip()
    ):
        raise ValueError("configure_identity must be a non-empty string")
    if invoke_identity is not None and (
        not isinstance(invoke_identity, str) or not invoke_identity.strip()
    ):
        raise ValueError("invoke_identity must be a non-empty string")
    if supplemental_identity is not None and not isinstance(
        supplemental_identity, Key
    ):
        raise TypeError("supplemental_identity must be a content Key")
    selected_options = dict(options or {})
    effective_model_identity = key_for(
        (
            "native-prediction-model-binding/v1",
            model_identity,
            configure_identity,
            invoke_identity,
            supplemental_identity,
            selected_options,
            _callable_identity(factory),
            _callable_identity(configure),
            _callable_identity(invoke),
            _callable_identity(supplemental_artifacts),
        )
    )

    from .operations import _describe_operation
    from .native_support import LazyNativeOwner

    implementations = [
        NativePredictionOperation,
        FoldingInputCodec,
        LazyNativeOwner,
        factory,
    ]
    if configure is not None:
        implementations.append(configure)
    if invoke is not None:
        implementations.append(invoke)
    if supplemental_artifacts is not None:
        implementations.append(supplemental_artifacts)
    descriptor = _describe_operation(
        alias,
        PredictRequest,
        Evidence,
        implementation=tuple(implementations),
        version=version,
        options=selected_options,
        schema=dict(schema or {"type": "object", "properties": {}, "additionalProperties": False}),
        model_identity=effective_model_identity,
        paid_boundary="predict",
        reference_codecs=_unique_codecs(
            LEGACY_REFERENCE_CODECS,
            NATIVE_SCORES_REFERENCE_CODECS,
            (FOLDING_INPUT_CODEC,),
        ),
    )

    owner = LazyNativeOwner(
        factory,
        bindings.data,
        effective_model_identity,
        alias=alias,
    )
    return NativePredictionOperation(
        descriptor,
        factory=factory,
        bindings=bindings,
        options=selected_options,
        configure=configure,
        configure_identity=configure_identity,
        native_invoke=invoke,
        native_invoke_identity=invoke_identity,
        supplemental_artifacts=supplemental_artifacts,
        supplemental_identity=supplemental_identity,
        model_identity=effective_model_identity,
        owner=owner,
    )


__all__ = [
    "FOLDING_INPUT_CODEC",
    "FoldingInputCodec",
    "NativePredictionOperation",
    "UnsupportedNativePrediction",
    "make_evaluator_operation",
]
