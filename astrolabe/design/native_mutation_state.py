"""Exact, restart-safe context for native mutation decisions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
import operator
from types import MappingProxyType
from typing import Any

from astrolabe.core import global_config
from astrolabe.score_contributions import (
    ScoreContribution,
    normalize_score_contributions,
)

from .store import ArrayCodec, DataStore, DataclassCodec, JSONCodec
from .types import (
    Artifact,
    Blob,
    Candidate,
    Dependency,
    Key,
    Polymer,
    Sample,
    System,
    key_for,
    manifest,
)


NATIVE_MUTATION_CONTEXT_SCHEMA = {
    "type": "object",
    "properties": {
        "schema": {"type": "string"},
        "base_candidate": {"type": "string"},
        "proposed_candidate": {"type": "string"},
        "variable": {"type": "string"},
        "sequence_before_pending": {"type": "string"},
        "working_sequence": {"type": "string"},
        "pending_mutations": {"type": "array"},
        "selector_contexts": {"type": "object"},
        "score_contributions": {"type": "array"},
        "pre_update_snapshot": {"type": "null"},
    },
    "required": [
        "schema",
        "base_candidate",
        "proposed_candidate",
        "variable",
        "sequence_before_pending",
        "working_sequence",
        "pending_mutations",
        "selector_contexts",
        "score_contributions",
        "pre_update_snapshot",
    ],
    "additionalProperties": False,
}


class UnsupportedMutationContext(ValueError):
    """Mutation context cannot be represented without losing scientific state."""


def _canonical_json(value: Any, *, path: str = "$") -> Any:
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise UnsupportedMutationContext(
                f"non-finite mutation context value at {path}"
            )
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise UnsupportedMutationContext(
                    f"mutation context object key at {path} is not a string"
                )
            result[key] = _canonical_json(item, path=f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            _canonical_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return _canonical_json(value.item(), path=path)
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise UnsupportedMutationContext(
                    f"object-dtype mutation context array at {path}"
                )
            return _canonical_json(value.tolist(), path=path)
    except ImportError:
        pass
    raise UnsupportedMutationContext(
        f"unsupported mutation context value at {path}: "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _pending_position(value: Any) -> int:
    try:
        import numpy as np
    except ImportError:
        np = None
    if isinstance(value, bool) or (
        np is not None and isinstance(value, np.bool_)
    ):
        raise UnsupportedMutationContext("pending mutation position is invalid")
    try:
        position = operator.index(value)
    except TypeError as error:
        raise UnsupportedMutationContext(
            "pending mutation position is invalid"
        ) from error
    if position < 0:
        raise UnsupportedMutationContext("pending mutation position is invalid")
    return position


def _segment_type_name(segment: Any) -> str:
    type_name = getattr(getattr(segment, "type", None), "name", None)
    if not isinstance(type_name, str) or not type_name:
        raise UnsupportedMutationContext(
            "mutation segment does not expose a known segment type"
        )
    return type_name


def _has_pending_mutations(segment: Any) -> bool:
    pending = getattr(segment, "mutations", None)
    try:
        return pending is not None and len(pending) > 0
    except TypeError as error:
        raise UnsupportedMutationContext(
            "segment pending mutations are not a finite sequence"
        ) from error


def native_mutation_segments(
    manager: Any | None = None,
) -> tuple[tuple[str, Any], ...]:
    """Return unique polymer segments in stable manager insertion order.

    Shared aliases collapse to one segment because ``unique_segments`` is
    keyed by object identity. Immutable chemicals with no pending mutation are
    explicitly skipped; mutable or pending chemicals and unknown segment types
    fail closed before a native mutator can run.
    """

    selected_manager = (
        global_config.segment_manager if manager is None else manager
    )
    try:
        unique = selected_manager.unique_segments
        segments = selected_manager.segments
        primary_names = selected_manager.primary_names
    except AttributeError as error:
        raise UnsupportedMutationContext(
            "legacy segment manager lacks unique segment identity maps"
        ) from error
    if not isinstance(unique, Mapping) or not unique:
        raise UnsupportedMutationContext(
            "mutation context requires at least one unique legacy segment"
        )
    if not isinstance(segments, Mapping) or not isinstance(
        primary_names,
        Mapping,
    ):
        raise UnsupportedMutationContext(
            "legacy segment manager maps are malformed"
        )
    if set(primary_names) != set(unique):
        raise UnsupportedMutationContext(
            "legacy segment manager primary-name map is incomplete or stale"
        )

    known_ids = set(unique)
    dangling = sorted(
        str(name)
        for name, segment in segments.items()
        if id(segment) not in known_ids
    )
    if dangling:
        raise UnsupportedMutationContext(
            "legacy segment aliases reference segments outside manager order: "
            f"{dangling}"
        )

    result: list[tuple[str, Any]] = []
    seen_primary: set[str] = set()
    for segment_id, segment in unique.items():
        if type(segment_id) is not int or segment_id != id(segment):
            raise UnsupportedMutationContext(
                "legacy unique segment key differs from object identity"
            )
        primary = primary_names.get(segment_id)
        if not isinstance(primary, str) or not primary:
            raise UnsupportedMutationContext(
                "legacy segment manager has an invalid primary name"
            )
        if primary in seen_primary:
            raise UnsupportedMutationContext(
                f"duplicate primary segment name {primary!r}"
            )
        seen_primary.add(primary)
        if segments.get(primary) is not segment:
            raise UnsupportedMutationContext(
                f"primary segment name {primary!r} does not identify its segment"
            )
        aliases = tuple(
            name
            for name, candidate in segments.items()
            if candidate is segment
        )
        if not all(isinstance(name, str) and name for name in aliases):
            raise UnsupportedMutationContext(
                f"segment {primary!r} has an invalid exact alias"
            )

        kind = _segment_type_name(segment)
        if kind in {"PROTEIN", "RNA", "DNA"}:
            result.append((primary, segment))
            continue
        if kind in {"SMILES", "CCD"}:
            if getattr(segment, "is_fixed", False) is not True:
                raise UnsupportedMutationContext(
                    f"mutable chemical segment {primary!r} is unsupported"
                )
            if _has_pending_mutations(segment):
                raise UnsupportedMutationContext(
                    f"pending mutations on chemical segment {primary!r} "
                    "are unsupported"
                )
            continue
        raise UnsupportedMutationContext(
            f"unsupported mutation segment kind {kind!r} for {primary!r}"
        )

    if not result:
        raise UnsupportedMutationContext(
            "mutation context requires at least one unique polymer segment"
        )
    return tuple(result)


@dataclass(frozen=True)
class MutationContextPolicy:
    """Immutable declaration that a mutator opts into exact context capture."""

    identity: str
    schema: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.identity, str) or not self.identity.strip():
            raise ValueError("mutation context identity must be non-empty")
        if not isinstance(self.schema, Mapping):
            raise TypeError("mutation context schema must be a mapping")
        normalized = _canonical_json(self.schema, path="policy.schema")
        if not isinstance(normalized, dict):
            raise TypeError("mutation context schema must normalize to an object")
        object.__setattr__(self, "schema", _freeze_json(normalized))

    def descriptor(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "schema": _canonical_json(self.schema, path="policy.schema"),
        }


@dataclass(frozen=True)
class NativePendingMutation:
    position: int
    from_token: str
    to_token: str
    entropy: float
    score: float


@dataclass(frozen=True)
class NativeScoreContribution:
    source: str
    metric_name: str
    value: Blob
    scope: str
    entity: str | None
    shape: str
    aggregation: str
    flatten_as: str | None
    provenance: str | None
    enabled: bool


@dataclass(frozen=True)
class NativeMutationResultState:
    schema: str
    base_candidate: Key
    proposed_candidate: Key
    variable: str
    sequence_before_pending: str
    working_sequence: str
    pending_mutations: tuple[NativePendingMutation, ...]
    selector_contexts: Blob
    score_contributions: tuple[NativeScoreContribution, ...]
    pre_update_snapshot: None = None


@dataclass(frozen=True)
class NativeMutationContextDraft:
    policy: MutationContextPolicy
    variable: str
    sequence_before_pending: str
    working_sequence: str
    pending_mutations: tuple[NativePendingMutation, ...]
    selector_contexts: Blob
    score_contributions: tuple[NativeScoreContribution, ...]


_SELECTOR_CONTEXT_CODEC = JSONCodec(
    "astrolabe.design.native-mutation.selector-contexts/v1"
)
_CONTRIBUTION_JSON_CODEC = JSONCodec(
    "astrolabe.design.native-mutation.score-contribution-json/v1"
)
_ARRAYS = ArrayCodec()
NATIVE_MUTATION_CONTEXT_CODEC = DataclassCodec(
    NativeMutationResultState,
    "astrolabe.design.native-mutation.context/v1",
    nested_types=(
        NativePendingMutation,
        NativeScoreContribution,
        Blob,
        Key,
    ),
)
NATIVE_MUTATION_REFERENCE_CODECS = (
    NATIVE_MUTATION_CONTEXT_CODEC,
    _SELECTOR_CONTEXT_CODEC,
    _CONTRIBUTION_JSON_CODEC,
    _ARRAYS,
)


def _store_contribution_value(data: DataStore, value: Any) -> Blob:
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            array = np.asarray(value)
            if array.dtype.hasobject:
                raise UnsupportedMutationContext(
                    "object-dtype score contribution arrays are unsupported"
                )
            return data.put(array, _ARRAYS)
        if isinstance(value, np.generic):
            value = value.item()
        if (
            isinstance(value, (list, tuple))
            and value
            and all(isinstance(item, (int, float, bool, np.number)) for item in value)
        ):
            return data.put(np.asarray(value), _ARRAYS)
    except ImportError:
        pass
    return data.put(
        _canonical_json(value, path="score_contribution.value"),
        _CONTRIBUTION_JSON_CODEC,
    )


def _contribution_records(
    data: DataStore,
    raw: Any,
) -> tuple[NativeScoreContribution, ...]:
    records = []
    for contribution in normalize_score_contributions(raw):
        if not isinstance(contribution, ScoreContribution):
            raise TypeError("normalized score contributions must be ScoreContribution")
        records.append(
            NativeScoreContribution(
                source=contribution.source,
                metric_name=contribution.metric_name,
                value=_store_contribution_value(data, contribution.value),
                scope=contribution.scope,
                entity=contribution.entity,
                shape=contribution.shape,
                aggregation=contribution.aggregation,
                flatten_as=contribution.flatten_as,
                provenance=contribution.provenance,
                enabled=bool(contribution.enabled),
            )
        )
    return tuple(records)


def _pending_mutations(
    segment: Any,
) -> tuple[NativePendingMutation, ...]:
    pending = []
    seen_positions: set[int] = set()
    for raw in tuple(getattr(segment, "mutations", None) or ()):
        if not isinstance(raw, (list, tuple)) or len(raw) != 5:
            raise UnsupportedMutationContext(
                "pending mutations must be complete five-tuples"
            )
        position, from_token, to_token, entropy, score = raw
        position = _pending_position(position)
        if position in seen_positions:
            raise UnsupportedMutationContext(
                f"duplicate pending mutation position {position}"
            )
        seen_positions.add(position)
        if not isinstance(from_token, str) or not isinstance(to_token, str):
            raise UnsupportedMutationContext(
                "pending mutation tokens must be strings"
            )
        try:
            entropy_value = float(entropy)
            score_value = float(score)
        except (TypeError, ValueError) as error:
            raise UnsupportedMutationContext(
                "pending mutation entropy and score must be finite numbers"
            ) from error
        if not math.isfinite(entropy_value) or not math.isfinite(score_value):
            raise UnsupportedMutationContext(
                "pending mutation entropy and score must be finite"
            )
        pending.append(
            NativePendingMutation(
                position=position,
                from_token=from_token,
                to_token=to_token,
                entropy=entropy_value,
                score=score_value,
            )
        )
    return tuple(pending)


def _capture_segment_context(
    *,
    policy: MutationContextPolicy,
    segment_name: str,
    segment: Any,
    selector_contexts: Blob,
    score_contributions: tuple[NativeScoreContribution, ...],
) -> NativeMutationContextDraft:
    sequence = getattr(segment, "sequence", None)
    if not isinstance(sequence, str):
        raise UnsupportedMutationContext(
            f"segment {segment_name!r} base sequence must be a string"
        )
    pending = _pending_mutations(segment)
    for mutation in pending:
        if mutation.position >= len(sequence):
            raise UnsupportedMutationContext(
                f"pending mutation position {mutation.position} is outside "
                f"segment {segment_name!r}"
            )
        if sequence[mutation.position] != mutation.from_token:
            raise UnsupportedMutationContext(
                f"pending mutation source token differs for segment "
                f"{segment_name!r}"
            )
    working_sequence = getattr(segment, "mutated_sequence", None)
    if not isinstance(working_sequence, str):
        raise UnsupportedMutationContext(
            f"segment {segment_name!r} working sequence must be a string"
        )
    if len(sequence) != len(working_sequence):
        raise UnsupportedMutationContext(
            f"segment {segment_name!r} changed sequence length"
        )
    for mutation in pending:
        if working_sequence[mutation.position] != mutation.to_token:
            raise UnsupportedMutationContext(
                f"pending mutation target token differs for segment "
                f"{segment_name!r}"
            )
    return NativeMutationContextDraft(
        policy=policy,
        variable=segment_name,
        sequence_before_pending=sequence,
        working_sequence=working_sequence,
        pending_mutations=pending,
        selector_contexts=selector_contexts,
        score_contributions=score_contributions,
    )


def capture_native_mutation_contexts(
    data: DataStore,
    *,
    policy: MutationContextPolicy,
    runtime_config: Mapping[str, Any],
) -> tuple[NativeMutationContextDraft, ...]:
    """Capture every unique polymer segment after mutate and before apply."""

    if not isinstance(policy, MutationContextPolicy):
        raise TypeError("policy must be MutationContextPolicy")
    if not isinstance(runtime_config, Mapping):
        raise TypeError("runtime_config must be a mapping")
    manager = global_config.segment_manager
    if getattr(manager, "_pre_update_snapshot", None) is not None:
        raise UnsupportedMutationContext(
            "typed mutation context requires disabled pre-update stack; "
            "manager snapshot is not None"
        )
    segments = native_mutation_segments(manager)

    selector_contexts = _canonical_json(
        runtime_config.get("mutator_selector_score_contexts") or (),
        path="runtime_config.mutator_selector_score_contexts",
    )
    selector_blob = data.put(selector_contexts, _SELECTOR_CONTEXT_CODEC)
    contributions = _contribution_records(
        data,
        runtime_config.get("score_contributions"),
    )
    drafts = tuple(
        _capture_segment_context(
            policy=policy,
            segment_name=segment_name,
            segment=segment,
            selector_contexts=selector_blob,
            score_contributions=contributions,
        )
        for segment_name, segment in segments
    )
    if not drafts:
        raise UnsupportedMutationContext(
            "mutation context captured no polymer segments"
        )
    return drafts


def capture_native_mutation_context(
    data: DataStore,
    *,
    policy: MutationContextPolicy,
    runtime_config: Mapping[str, Any],
) -> NativeMutationContextDraft:
    """Capture one polymer segment as a strict plural-capture convenience."""

    drafts = capture_native_mutation_contexts(
        data,
        policy=policy,
        runtime_config=runtime_config,
    )
    if len(drafts) != 1:
        raise UnsupportedMutationContext(
            "singular mutation context requires exactly one unique polymer "
            f"segment; found {len(drafts)}"
        )
    return drafts[0]


def _assignment_sequence(candidate: Candidate, variable: str) -> str:
    matches = [
        assignment.value
        for assignment in candidate.assignments
        if assignment.variable == variable
    ]
    if len(matches) != 1 or not isinstance(matches[0], Polymer):
        raise UnsupportedMutationContext(
            f"candidate does not contain one polymer variable {variable!r}"
        )
    return "".join(matches[0].tokens)


def _validate_context_state(state: NativeMutationResultState) -> None:
    if not isinstance(state, NativeMutationResultState):
        raise TypeError("native mutation context has the wrong state type")
    if not isinstance(state.schema, str) or not state.schema.strip():
        raise UnsupportedMutationContext("native mutation context schema is empty")
    if not isinstance(state.variable, str) or not state.variable:
        raise UnsupportedMutationContext("native mutation context variable is empty")
    if not isinstance(state.sequence_before_pending, str) or not isinstance(
        state.working_sequence, str
    ):
        raise UnsupportedMutationContext(
            "native mutation context sequences must be strings"
        )
    if len(state.sequence_before_pending) != len(state.working_sequence):
        raise UnsupportedMutationContext(
            "native mutation context changed sequence length"
        )
    seen_positions: set[int] = set()
    for mutation in state.pending_mutations:
        if not isinstance(mutation, NativePendingMutation):
            raise UnsupportedMutationContext(
                "native mutation context contains an invalid pending mutation"
            )
        if (
            type(mutation.position) is not int
            or mutation.position < 0
            or mutation.position >= len(state.sequence_before_pending)
        ):
            raise UnsupportedMutationContext(
                "native mutation context pending position is outside the sequence"
            )
        if mutation.position in seen_positions:
            raise UnsupportedMutationContext(
                f"duplicate pending mutation position {mutation.position}"
            )
        seen_positions.add(mutation.position)
        if state.sequence_before_pending[mutation.position] != mutation.from_token:
            raise UnsupportedMutationContext(
                "pending mutation source token differs from the base sequence"
            )
        if state.working_sequence[mutation.position] != mutation.to_token:
            raise UnsupportedMutationContext(
                "pending mutation target token differs from the working sequence"
            )
    if state.pre_update_snapshot is not None:
        raise UnsupportedMutationContext(
            "native mutation context cannot omit a non-None pre-update snapshot"
        )


def bind_native_mutation_context(
    draft: NativeMutationContextDraft,
    *,
    data: DataStore,
    system: System,
    base: Candidate,
    proposed: Candidate,
    sample: Sample,
) -> tuple[Blob, Artifact]:
    """Bind a pre-apply capture to strict base and proposed candidates."""

    if base.system != system.identity or proposed.system != system.identity:
        raise UnsupportedMutationContext(
            "mutation context candidates belong to another System"
        )
    if _assignment_sequence(base, draft.variable) != draft.sequence_before_pending:
        raise UnsupportedMutationContext(
            "base candidate sequence differs from the captured pre-pending sequence"
        )
    if _assignment_sequence(proposed, draft.variable) != draft.working_sequence:
        raise UnsupportedMutationContext(
            "proposed candidate sequence differs from the captured working sequence"
        )
    for mutation in draft.pending_mutations:
        if mutation.position >= len(draft.sequence_before_pending):
            raise UnsupportedMutationContext(
                "pending mutation position is outside the captured sequence"
            )
        if draft.sequence_before_pending[mutation.position] != mutation.from_token:
            raise UnsupportedMutationContext(
                "pending mutation source token differs from the captured sequence"
            )
        if draft.working_sequence[mutation.position] != mutation.to_token:
            raise UnsupportedMutationContext(
                "pending mutation target token differs from the working sequence"
            )

    state = NativeMutationResultState(
        schema=draft.policy.identity,
        base_candidate=base.identity,
        proposed_candidate=proposed.identity,
        variable=draft.variable,
        sequence_before_pending=draft.sequence_before_pending,
        working_sequence=draft.working_sequence,
        pending_mutations=draft.pending_mutations,
        selector_contexts=draft.selector_contexts,
        score_contributions=draft.score_contributions,
        pre_update_snapshot=None,
    )
    _validate_context_state(state)
    content = data.put(state, NATIVE_MUTATION_CONTEXT_CODEC)
    occurrences = tuple(
        occurrence
        for occurrence in system.occurrences
        if occurrence.variable == draft.variable
    )
    if not occurrences:
        raise UnsupportedMutationContext(
            f"no physical occurrences bind mutation context variable {draft.variable!r}"
        )
    policy_identity = key_for(
        (
            "native-mutation-context-policy/v1",
            draft.policy.identity,
            draft.policy.descriptor()["schema"],
        )
    )
    dependencies = [
        Dependency("system", system.identity),
        Dependency("base-candidate", base.identity),
        Dependency("proposed-candidate", proposed.identity),
        Dependency("sample", key_for(sample)),
        Dependency("context-content", content.identity),
        Dependency("selector-contexts", draft.selector_contexts.identity),
        Dependency("context-policy", policy_identity),
    ]
    dependencies.extend(
        Dependency("score-contribution-value", item.value.identity)
        for item in draft.score_contributions
    )
    inputs = manifest(tuple(dependencies))
    artifact = Artifact(
        identity=key_for(
            (
                "artifact/native-mutation-context/v1",
                base.identity,
                proposed.identity,
                occurrences,
                sample,
                content,
                inputs,
            )
        ),
        candidate=proposed.identity,
        occurrences=occurrences,
        sample=sample,
        content=content,
        derived_from=(base.identity, proposed.identity),
        inputs=inputs,
    )
    return content, artifact


def restore_native_mutation_context(
    artifact: Artifact,
    data: DataStore,
) -> NativeMutationResultState:
    if not isinstance(artifact, Artifact):
        raise TypeError("artifact must be Artifact")
    if artifact.content.codec != NATIVE_MUTATION_CONTEXT_CODEC.identity:
        raise UnsupportedMutationContext(
            "artifact does not contain a native mutation context"
        )
    state = data.get(artifact.content, NATIVE_MUTATION_CONTEXT_CODEC)
    _validate_context_state(state)
    if state.proposed_candidate != artifact.candidate:
        raise UnsupportedMutationContext(
            "mutation context proposed candidate differs from its Artifact"
        )
    if set(artifact.derived_from) != {
        state.base_candidate,
        state.proposed_candidate,
    }:
        raise UnsupportedMutationContext(
            "mutation context base/proposed candidates differ from its Artifact"
        )
    if not artifact.occurrences or any(
        occurrence.variable != state.variable
        for occurrence in artifact.occurrences
    ):
        raise UnsupportedMutationContext(
            "mutation context variable differs from its physical occurrences"
        )
    dependencies = {
        (dependency.role, dependency.identity)
        for dependency in artifact.inputs.dependencies
    }
    required = {
        ("base-candidate", state.base_candidate),
        ("proposed-candidate", state.proposed_candidate),
        ("context-content", artifact.content.identity),
    }
    if not required.issubset(dependencies):
        raise UnsupportedMutationContext(
            "mutation context Artifact inputs do not bind its exact state"
        )
    return state


__all__ = [
    "MutationContextPolicy",
    "NATIVE_MUTATION_CONTEXT_CODEC",
    "NATIVE_MUTATION_CONTEXT_SCHEMA",
    "NATIVE_MUTATION_REFERENCE_CODECS",
    "NativeMutationContextDraft",
    "NativeMutationResultState",
    "NativePendingMutation",
    "NativeScoreContribution",
    "UnsupportedMutationContext",
    "bind_native_mutation_context",
    "capture_native_mutation_context",
    "capture_native_mutation_contexts",
    "native_mutation_segments",
    "restore_native_mutation_context",
]
