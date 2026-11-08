"""Durable, static-stage native state for the portable default profile.

The records in this module contain scientific data and content references only.
They never contain model objects, Python functions, live poses, or process
resources.  Live ``Decisions`` and ``StandardScores`` values are reconstructed
from these immutable records at an explicit owner boundary.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
import math
from pathlib import Path
from typing import Any

import numpy as np

from .native_scores import (
    NATIVE_SCORES_ENVELOPE_CODEC,
    NATIVE_SCORES_REFERENCE_CODECS,
    restore_scores,
    snapshot_scores,
)
from .store import DataclassCodec, DataStore
from .types import (
    Artifact,
    Assessment,
    Blob,
    Candidate,
    Evidence,
    Key,
    Proposal,
    Scored,
)


DECISION_STATE_SCHEMA = "astrolabe.portable-native.decision-state/v1"
NATIVE_STEP_STATE_SCHEMA = "astrolabe.portable-native.step-state/v1"
SCIENTIFIC_CONTRACT_SCHEMA = "astrolabe.portable-native.run-contract/v1"
NUMERIC_TREE_CODEC_ID = "astrolabe.portable-native.numeric-tree/v1"


class UnsupportedPortableNativeState(ValueError):
    """The selected native state cannot be represented without loss."""


class NumericTreeCodec:
    """Exact typed codec for supported score-detail and row-history trees."""

    identity = NUMERIC_TREE_CODEC_ID

    def encode(self, value: Any) -> bytes:
        import json

        return json.dumps(
            self._encode(value, path="$"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")

    def decode(self, data: bytes) -> Any:
        import json

        if not isinstance(data, bytes):
            raise TypeError("numeric-tree payload must be bytes")
        try:
            encoded = json.loads(
                data.decode("utf-8"),
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"invalid JSON constant {value!r}")
                ),
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("invalid numeric-tree JSON") from error
        return self._decode(encoded, path="$")

    def references(self, value: Any) -> tuple[Blob, ...]:
        self._encode(value, path="$")
        return ()

    def compatibility_key(self) -> tuple[str, str]:
        return (type(self).__name__, self.identity)

    def _encode(self, value: Any, *, path: str) -> Any:
        if value is None or type(value) in {str, bool, int}:
            return value
        if type(value) is float:
            if math.isnan(value):
                return {"$float": "nan"}
            if math.isinf(value):
                return {"$float": "inf" if value > 0 else "-inf"}
            return value
        if isinstance(value, Path):
            return {"$path": value.as_posix()}
        if isinstance(value, bytes):
            return {"$bytes": base64.b64encode(value).decode("ascii")}
        if isinstance(value, tuple):
            return {
                "$tuple": [
                    self._encode(item, path=f"{path}[{index}]")
                    for index, item in enumerate(value)
                ]
            }
        if isinstance(value, list):
            return {
                "$list": [
                    self._encode(item, path=f"{path}[{index}]")
                    for index, item in enumerate(value)
                ]
            }
        if isinstance(value, Mapping):
            if any(not isinstance(key, str) for key in value):
                raise UnsupportedPortableNativeState(
                    f"numeric-tree mapping keys at {path} must be strings"
                )
            return {
                "$dict": {
                    key: self._encode(item, path=f"{path}.{key}")
                    for key, item in value.items()
                }
            }
        if isinstance(value, np.generic):
            array = np.asarray(value)
            self._validate_array(array, path=path)
            return {
                "$numpy_scalar": {
                    "dtype": array.dtype.str,
                    "data": base64.b64encode(
                        np.ascontiguousarray(array).tobytes()
                    ).decode("ascii"),
                }
            }
        if isinstance(value, np.ndarray):
            self._validate_array(value, path=path)
            return {
                "$ndarray": {
                    "dtype": value.dtype.str,
                    "shape": list(value.shape),
                    "data": base64.b64encode(
                        np.ascontiguousarray(value).tobytes()
                    ).decode("ascii"),
                }
            }
        raise UnsupportedPortableNativeState(
            f"unsupported numeric-tree value at {path}: "
            f"{type(value).__module__}.{type(value).__qualname__}"
        )

    @staticmethod
    def _validate_array(value: np.ndarray, *, path: str) -> None:
        if value.dtype.hasobject:
            raise UnsupportedPortableNativeState(
                f"object-dtype array is unsupported at {path}"
            )
        if value.dtype.fields is not None:
            raise UnsupportedPortableNativeState(
                f"structured dtype is unsupported at {path}"
            )

    def _decode(self, value: Any, *, path: str) -> Any:
        if value is None or type(value) in {str, bool, int, float}:
            return value
        if not isinstance(value, dict):
            raise ValueError(f"invalid numeric-tree value at {path}")
        if set(value) == {"$float"}:
            token = value["$float"]
            if token == "nan":
                return float("nan")
            if token == "inf":
                return float("inf")
            if token == "-inf":
                return float("-inf")
            raise ValueError(f"invalid floating-point token at {path}")
        if set(value) == {"$path"}:
            return Path(value["$path"])
        if set(value) == {"$bytes"}:
            return base64.b64decode(value["$bytes"].encode("ascii"), validate=True)
        if set(value) == {"$tuple"}:
            return tuple(
                self._decode(item, path=f"{path}[{index}]")
                for index, item in enumerate(value["$tuple"])
            )
        if set(value) == {"$list"}:
            return [
                self._decode(item, path=f"{path}[{index}]")
                for index, item in enumerate(value["$list"])
            ]
        if set(value) == {"$dict"}:
            entries = value["$dict"]
            if not isinstance(entries, dict) or any(
                not isinstance(key, str) for key in entries
            ):
                raise ValueError(f"invalid dictionary at {path}")
            return {
                key: self._decode(item, path=f"{path}.{key}")
                for key, item in entries.items()
            }
        if set(value) == {"$ndarray"}:
            record = value["$ndarray"]
            return self._decode_array(record, path=path, scalar=False)
        if set(value) == {"$numpy_scalar"}:
            record = value["$numpy_scalar"]
            return self._decode_array(record, path=path, scalar=True)
        raise ValueError(f"unknown numeric-tree envelope at {path}")

    def _decode_array(self, record: Any, *, path: str, scalar: bool) -> Any:
        if not isinstance(record, dict):
            raise ValueError(f"invalid array record at {path}")
        expected = {"dtype", "data"} if scalar else {"dtype", "shape", "data"}
        if set(record) != expected:
            raise ValueError(f"invalid array fields at {path}")
        try:
            dtype = np.dtype(record["dtype"])
            data = base64.b64decode(record["data"].encode("ascii"), validate=True)
            if dtype.hasobject or dtype.fields is not None:
                raise ValueError("unsafe dtype")
            if scalar:
                array = np.frombuffer(data, dtype=dtype).copy()
                if array.shape != (1,):
                    raise ValueError("scalar payload has the wrong length")
                result = array.reshape(())[()]
            else:
                shape = record["shape"]
                if not isinstance(shape, list) or any(
                    type(item) is not int or item < 0 for item in shape
                ):
                    raise ValueError("invalid array shape")
                array = np.frombuffer(data, dtype=dtype).copy().reshape(tuple(shape))
                array.setflags(write=False)
                result = array
        except (TypeError, ValueError, AttributeError) as error:
            raise ValueError(f"invalid array record at {path}") from error
        if np.asarray(result).dtype.str != dtype.str:
            raise ValueError(f"decoded dtype changed at {path}")
        return result


NUMERIC_TREE_CODEC = NumericTreeCodec()


@dataclass(frozen=True)
class ScientificRunContract:
    schema: str
    config_identity: Key
    source_identity: Key
    system_identity: Key
    resource_manifest: Key
    static_stage: Key
    objective: Key
    component_identities: tuple[Key, ...]
    seed: int
    acceptance_protocol: Key
    codec_identities: tuple[str, ...]
    recovery_capabilities: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema != SCIENTIFIC_CONTRACT_SCHEMA:
            raise ValueError(f"unsupported scientific contract schema {self.schema!r}")
        for name in (
            "config_identity",
            "source_identity",
            "system_identity",
            "resource_manifest",
            "static_stage",
            "objective",
            "acceptance_protocol",
        ):
            if not isinstance(getattr(self, name), Key):
                raise TypeError(f"{name} must be a content Key")
        if type(self.seed) is not int:
            raise TypeError("seed must be an integer")
        if (
            not self.component_identities
            or any(not isinstance(item, Key) for item in self.component_identities)
        ):
            raise ValueError("component_identities must be non-empty content Keys")
        if len(set(self.component_identities)) != len(self.component_identities):
            raise ValueError("component identities must be unique")
        if not self.codec_identities or len(set(self.codec_identities)) != len(
            self.codec_identities
        ):
            raise ValueError("codec identities must be non-empty and unique")
        if (
            not self.recovery_capabilities
            or len(set(self.recovery_capabilities))
            != len(self.recovery_capabilities)
        ):
            raise ValueError("recovery capabilities must be non-empty and unique")


@dataclass(frozen=True)
class SequenceAtStep:
    step: int
    assignments: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if type(self.step) is not int or self.step < 0:
            raise ValueError("sequence step must be nonnegative")
        if not self.assignments:
            raise ValueError("sequence step requires at least one assignment")
        if any(
            not isinstance(name, str)
            or not name
            or not isinstance(value, str)
            for name, value in self.assignments
        ):
            raise TypeError("sequence assignments must be non-empty string pairs")
        if len({name for name, _ in self.assignments}) != len(self.assignments):
            raise ValueError("sequence assignment names must be unique")


@dataclass(frozen=True)
class ScoreAtStep:
    step: int
    value: float

    def __post_init__(self) -> None:
        if type(self.step) is not int or self.step < 0:
            raise ValueError("score step must be nonnegative")
        if not math.isfinite(self.value):
            raise ValueError("portable native weighted scores must be finite")


@dataclass(frozen=True)
class BlobAtStep:
    step: int
    value: Blob

    def __post_init__(self) -> None:
        if type(self.step) is not int or self.step < 0:
            raise ValueError("blob step must be nonnegative")
        if not isinstance(self.value, Blob):
            raise TypeError("step value must be a Blob")


@dataclass(frozen=True)
class DecisionAssessmentState:
    details: Blob
    evaluation_scores: Blob | None

    def __post_init__(self) -> None:
        if not isinstance(self.details, Blob) or (
            self.details.codec != NUMERIC_TREE_CODEC.identity
        ):
            raise TypeError("assessment details must be a numeric-tree Blob")
        if self.evaluation_scores is not None and (
            not isinstance(self.evaluation_scores, Blob)
            or self.evaluation_scores.codec != NATIVE_SCORES_ENVELOPE_CODEC.identity
        ):
            raise TypeError("assessment evaluation must be a native-score Blob or None")


@dataclass(frozen=True)
class DecisionState:
    schema: str
    sequences: tuple[SequenceAtStep, ...]
    scores: tuple[ScoreAtStep, ...]
    scores_detail: tuple[BlobAtStep, ...]
    accepted_steps: tuple[int, ...]
    last_accepted: int
    stage_history: tuple[tuple[int, str], ...]
    row_history: Blob
    evaluation_scores: tuple[BlobAtStep, ...]
    assessments: tuple[DecisionAssessmentState, ...] | None = field(
        default=None, metadata={"omit_none": True}
    )
    observation_state: Blob | None = field(
        default=None, metadata={"omit_none": True}
    )

    def __post_init__(self) -> None:
        if self.schema != DECISION_STATE_SCHEMA:
            raise ValueError(f"unsupported decision state schema {self.schema!r}")
        if (self.assessments is None) != (self.observation_state is None):
            raise ValueError("assessment history and observation state must occur together")
        if self.assessments is not None and any(
            not isinstance(item, DecisionAssessmentState) for item in self.assessments
        ):
            raise TypeError("assessment history must contain DecisionAssessmentState values")
        if self.observation_state is not None and (
            not isinstance(self.observation_state, Blob)
            or self.observation_state.codec != NUMERIC_TREE_CODEC.identity
        ):
            raise TypeError("observation state must be a numeric-tree Blob or None")
        for name in ("sequences", "scores", "scores_detail", "evaluation_scores"):
            values = getattr(self, name)
            if not values:
                raise ValueError(f"{name} cannot be empty")
        score_steps = tuple(item.step for item in self.scores)
        if score_steps != tuple(sorted(score_steps)) or len(set(score_steps)) != len(
            score_steps
        ):
            raise ValueError("scores must have unique ascending steps")
        if tuple(item.step for item in self.sequences) != score_steps:
            raise ValueError("sequences and scores must have the same steps")
        if tuple(item.step for item in self.scores_detail) != score_steps:
            raise ValueError("score details and scores must have the same steps")
        if tuple(item.step for item in self.evaluation_scores) != score_steps:
            raise ValueError("evaluation scores and scores must have the same steps")
        if tuple(step for step, _ in self.stage_history) != score_steps:
            raise ValueError("stage history and scores must have the same steps")
        if any(not stage for _, stage in self.stage_history):
            raise ValueError("stage history names must be non-empty")
        if (
            not self.accepted_steps
            or tuple(sorted(set(self.accepted_steps))) != self.accepted_steps
        ):
            raise ValueError("accepted steps must be a non-empty ascending tuple")
        if any(step not in score_steps for step in self.accepted_steps):
            raise ValueError("accepted step is absent from score history")
        if self.last_accepted not in self.accepted_steps:
            raise ValueError("last_accepted must be in accepted_steps")
        if not isinstance(self.row_history, Blob) or (
            self.row_history.codec != NUMERIC_TREE_CODEC.identity
        ):
            raise TypeError("row_history must be a numeric-tree Blob")
        if any(
            item.value.codec != NUMERIC_TREE_CODEC.identity
            for item in self.scores_detail
        ):
            raise TypeError("score details must be numeric-tree Blobs")
        if any(
            item.value.codec != NATIVE_SCORES_ENVELOPE_CODEC.identity
            for item in self.evaluation_scores
        ):
            raise TypeError("evaluation scores must be native-score envelope Blobs")


@dataclass(frozen=True)
class NativeRuntimeContext:
    schema: str
    selector_and_contribution_context: Blob
    decision_context: Blob
    conflict_diagnostics: Blob | None = None

    def __post_init__(self) -> None:
        if self.schema != "astrolabe.portable-native.runtime-context/v1":
            raise ValueError(f"unsupported native runtime context {self.schema!r}")
        for name in (
            "selector_and_contribution_context",
            "decision_context",
        ):
            blob = getattr(self, name)
            if (
                not isinstance(blob, Blob)
                or blob.codec != NUMERIC_TREE_CODEC.identity
            ):
                raise TypeError(f"{name} must be a numeric-tree Blob")
        if self.conflict_diagnostics is not None and (
            not isinstance(self.conflict_diagnostics, Blob)
            or self.conflict_diagnostics.codec != NUMERIC_TREE_CODEC.identity
        ):
            raise TypeError("conflict_diagnostics must be a numeric-tree Blob or None")


@dataclass(frozen=True)
class PendingEdit:
    variable: str
    positions: tuple[int, ...]
    before: tuple[str, ...]
    after: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.variable:
            raise ValueError("pending edit variable must be non-empty")
        if len(self.positions) != len(self.before) or len(self.positions) != len(
            self.after
        ):
            raise ValueError("pending edit arrays must have equal lengths")
        if any(type(item) is not int or item < 0 for item in self.positions):
            raise ValueError("pending edit positions must be nonnegative integers")
        if any(not item for item in (*self.before, *self.after)):
            raise ValueError("pending edit residues must be non-empty")


@dataclass(frozen=True)
class ProposalSnapshot:
    candidate: Candidate
    pending_edits: tuple[PendingEdit, ...]


@dataclass(frozen=True)
class ProposalStackState:
    accepted_candidate: Candidate
    working_candidate: Candidate
    pending_edits: tuple[PendingEdit, ...]
    pre_update_base: ProposalSnapshot | None
    base_evidence: tuple[Blob, ...]
    selector_and_contribution_context: Blob
    next_mutator_index: int

    def __post_init__(self) -> None:
        if self.accepted_candidate.system != self.working_candidate.system:
            raise ValueError("proposal stack candidates belong to different Systems")
        if self.next_mutator_index < 0:
            raise ValueError("next_mutator_index must be nonnegative")
        if any(not isinstance(item, Blob) for item in self.base_evidence):
            raise TypeError("base_evidence must contain Blobs")
        if (
            not isinstance(self.selector_and_contribution_context, Blob)
            or self.selector_and_contribution_context.codec
            != NUMERIC_TREE_CODEC.identity
        ):
            raise TypeError("proposal selector context must be a numeric-tree Blob")


@dataclass(frozen=True)
class NativeStepState:
    schema: str
    candidate: Candidate
    current_scores: Blob
    decision_state: DecisionState
    static_stage: Key
    runtime_context: Blob
    next_step: int

    def __post_init__(self) -> None:
        if self.schema != NATIVE_STEP_STATE_SCHEMA:
            raise ValueError(f"unsupported native step schema {self.schema!r}")
        if not isinstance(self.candidate, Candidate):
            raise TypeError("candidate must be a Candidate")
        if (
            not isinstance(self.current_scores, Blob)
            or self.current_scores.codec != NATIVE_SCORES_ENVELOPE_CODEC.identity
        ):
            raise TypeError("current_scores must be a native-score envelope Blob")
        if not isinstance(self.decision_state, DecisionState):
            raise TypeError("decision_state must be DecisionState")
        if not isinstance(self.static_stage, Key):
            raise TypeError("static_stage must be a content Key")
        if (
            not isinstance(self.runtime_context, Blob)
            or self.runtime_context.codec
            != "astrolabe.portable-native.runtime-context-codec/v1"
        ):
            raise TypeError("runtime_context must be a NativeRuntimeContext Blob")
        if type(self.next_step) is not int or self.next_step < 1:
            raise ValueError("next_step must be a positive integer")
        if self.decision_state.scores[-1].step != self.next_step - 1:
            raise ValueError("next_step must follow the latest recorded score step")


@dataclass(frozen=True)
class RootBootstrapRequest:
    candidate: Candidate
    evidence: Evidence
    raw_scores: Blob
    runtime_context: Blob

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, Candidate):
            raise TypeError("root candidate must be a Candidate")
        if not isinstance(self.evidence, Evidence):
            raise TypeError("root evidence must be Evidence")
        if (
            self.evidence.evaluated_candidate != self.candidate.identity
        ):
            raise ValueError("root evidence belongs to another candidate")
        if (
            not isinstance(self.raw_scores, Blob)
            or self.raw_scores.codec != NATIVE_SCORES_ENVELOPE_CODEC.identity
        ):
            raise TypeError("raw_scores must be a native-score envelope")
        if (
            not isinstance(self.runtime_context, Blob)
            or self.runtime_context.codec
            != "astrolabe.portable-native.runtime-context-codec/v1"
        ):
            raise TypeError("runtime_context must be a NativeRuntimeContext Blob")


@dataclass(frozen=True)
class DecisionRequest:
    before: NativeStepState
    proposal: Proposal
    proposed_scores: Blob
    proposal_context: Blob
    step: int

    def __post_init__(self) -> None:
        if self.step != self.before.next_step:
            raise ValueError("decision step differs from NativeStepState.next_step")
        if self.proposal.base != self.before.candidate.identity:
            raise ValueError("proposal base differs from the accepted candidate")
        if self.before.candidate.identity not in self.proposal.parents:
            raise ValueError("proposal parents omit the accepted candidate")
        if (
            not isinstance(self.proposed_scores, Blob)
            or self.proposed_scores.codec != NATIVE_SCORES_ENVELOPE_CODEC.identity
        ):
            raise TypeError("proposed_scores must be a native-score envelope")
        if (
            not isinstance(self.proposal_context, Blob)
            or self.proposal_context.codec
            != "astrolabe.portable-native.runtime-context-codec/v1"
        ):
            raise TypeError("proposal_context must be a NativeRuntimeContext Blob")


@dataclass(frozen=True)
class DecisionResult:
    after: NativeStepState
    accepted: bool
    stop_reason: str | None
    decision_details: Blob

    def __post_init__(self) -> None:
        if not isinstance(self.after, NativeStepState):
            raise TypeError("after must be NativeStepState")
        if type(self.accepted) is not bool:
            raise TypeError("accepted must be boolean")
        if self.stop_reason is not None and (
            not isinstance(self.stop_reason, str) or not self.stop_reason
        ):
            raise ValueError("stop_reason must be a non-empty string or None")
        if (
            not isinstance(self.decision_details, Blob)
            or self.decision_details.codec != NUMERIC_TREE_CODEC.identity
        ):
            raise TypeError("decision_details must be a numeric-tree Blob")


@dataclass(frozen=True)
class RootResult:
    state: NativeStepState
    scored: Scored

    def __post_init__(self) -> None:
        if not isinstance(self.state, NativeStepState):
            raise TypeError("root state must be NativeStepState")
        if not isinstance(self.scored, Scored):
            raise TypeError("root scored must be Scored")
        if self.scored.candidate != self.state.candidate:
            raise ValueError("root Scored and NativeStepState candidates differ")


@dataclass(frozen=True)
class EvaluationResult:
    candidate: Key
    raw_scores: Blob
    finalized_scores: Blob
    evidence: Evidence
    measurement_context: Blob

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, Key):
            raise TypeError("evaluation candidate must be a content Key")
        for name in ("raw_scores", "finalized_scores"):
            blob = getattr(self, name)
            if (
                not isinstance(blob, Blob)
                or blob.codec != NATIVE_SCORES_ENVELOPE_CODEC.identity
            ):
                raise TypeError(f"{name} must be a native-score envelope")
        if not isinstance(self.evidence, Evidence):
            raise TypeError("evaluation evidence must be Evidence")
        if self.evidence.evaluated_candidate != self.candidate:
            raise ValueError("evaluation evidence belongs to another candidate")
        if (
            not isinstance(self.measurement_context, Blob)
            or self.measurement_context.codec != NUMERIC_TREE_CODEC.identity
        ):
            raise TypeError("measurement_context must be a numeric-tree Blob")


def _plain_config(value: Any) -> Any:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            return _plain_config(OmegaConf.to_container(value, resolve=True))
    except ImportError:
        pass
    if value is None or type(value) in {str, bool, int, float}:
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {
            str(name): _plain_config(item)
            for name, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_plain_config(item) for item in value]
    return value


def store_numeric_tree(data: DataStore, value: Any, *, codec=NUMERIC_TREE_CODEC) -> Blob:
    if not isinstance(data, DataStore):
        raise TypeError("data must be a design DataStore")
    if codec is not NUMERIC_TREE_CODEC:
        raise TypeError("portable native state requires the registered numeric-tree codec")
    return data.put(_plain_config(value), codec)


def read_numeric_tree(data: DataStore, blob: Blob) -> Any:
    if not isinstance(blob, Blob) or blob.codec != NUMERIC_TREE_CODEC.identity:
        raise TypeError("blob is not a portable numeric-tree reference")
    return data.get(blob, NUMERIC_TREE_CODEC)


def snapshot_decision_state(decisions: Any, data: DataStore) -> DecisionState:
    """Freeze every scientific field currently held by legacy ``Decisions``."""
    from astrolabe.decision_maker import DecisionAssessment, Decisions
    from astrolabe.evaluators.utils import StandardScores

    if type(decisions) is not Decisions:
        raise TypeError("decision state snapshot requires the legacy Decisions owner")
    expected_fields = tuple(field.name for field in fields(Decisions))
    # ``vars`` exposes extensions without invoking properties or __post_init__.
    extension_names = sorted(set(vars(decisions)) - set(expected_fields))
    if extension_names:
        raise UnsupportedPortableNativeState(
            "Decisions has unrepresented instance fields: "
            + ", ".join(extension_names)
        )

    score_steps = tuple(sorted(int(step) for step in decisions.scores))
    evaluation_steps = tuple(sorted(int(step) for step in decisions.evaluation_results))
    if score_steps != evaluation_steps:
        raise UnsupportedPortableNativeState(
            "Decisions score and evaluation step histories differ"
        )
    if any(
        type(decisions.evaluation_results[step]) is not StandardScores
        for step in score_steps
    ):
        raise UnsupportedPortableNativeState(
            "Decisions contains a non-StandardScores evaluation value"
        )
    sequences = []
    for step in score_steps:
        values = decisions.sequences.get(step)
        if not isinstance(values, dict) or not values:
            raise UnsupportedPortableNativeState(
                f"Decisions has no sequence mapping for step {step}"
            )
        sequences.append(
            SequenceAtStep(
                step,
                tuple(
                    sorted(
                        (str(name), str(value))
                        for name, value in values.items()
                    )
                ),
            )
        )
    scores = []
    for step in score_steps:
        value = float(decisions.scores[step])
        if not math.isfinite(value):
            raise UnsupportedPortableNativeState(
                "portable native weighted scores must be finite (for step "
                f"{step} got {value!r})"
            )
        scores.append(ScoreAtStep(step, value))
    details = tuple(
        BlobAtStep(
            step,
            store_numeric_tree(
                data,
                decisions.scores_detail.get(step, {}),
            ),
        )
        for step in score_steps
    )
    evaluations = tuple(
        BlobAtStep(
            step,
            snapshot_scores(decisions.evaluation_results[step], data),
        )
        for step in score_steps
    )
    stages = tuple(
        (step, str(decisions.stage_history[step]))
        for step in score_steps
    )
    row_history = store_numeric_tree(data, list(decisions.row_history))
    accepted = tuple(sorted(int(step) for step in decisions.accepted_steps))
    if not accepted:
        raise UnsupportedPortableNativeState("Decisions accepted_steps cannot be empty")
    if int(decisions.last_accepted) != accepted[-1]:
        raise UnsupportedPortableNativeState(
            "Decisions last_accepted is not the latest accepted step"
        )
    assessments = []
    for assessment in decisions.assessments:
        if type(assessment) is not DecisionAssessment:
            raise UnsupportedPortableNativeState(
                "Decisions contains a non-DecisionAssessment observation"
            )
        assessments.append(
            DecisionAssessmentState(
                # Do not normalize tuples or Paths through _plain_config: these
                # are already frozen observations, not configuration values.
                details=data.put(
                    {
                        item.name: getattr(assessment, item.name)
                        for item in fields(DecisionAssessment)
                        if item.name != "evaluation_result"
                    },
                    NUMERIC_TREE_CODEC,
                ),
                evaluation_scores=(
                    None if assessment.evaluation_result is None
                    else snapshot_scores(assessment.evaluation_result, data)
                ),
            )
        )
    observation_state = data.put(
        {
            "observation_errors": decisions.observation_errors,
            "_attempt_count": decisions._attempt_count,
            "_accepted_attempt": decisions._accepted_attempt,
            "_revision_known": decisions._revision_known,
        },
        NUMERIC_TREE_CODEC,
    )
    return DecisionState(
        schema=DECISION_STATE_SCHEMA,
        sequences=tuple(sequences),
        scores=tuple(scores),
        scores_detail=details,
        accepted_steps=accepted,
        last_accepted=int(decisions.last_accepted),
        stage_history=stages,
        row_history=row_history,
        evaluation_scores=evaluations,
        assessments=tuple(assessments),
        observation_state=observation_state,
    )


def restore_decision_state(
    state: DecisionState,
    data: DataStore,
    destination: Path | str,
) -> Any:
    """Reconstruct the legacy owner without replaying its scientific methods."""
    from astrolabe.decision_maker import DecisionAssessment, Decisions

    if not isinstance(state, DecisionState):
        raise TypeError("state must be DecisionState")
    if not isinstance(data, DataStore):
        raise TypeError("data must be a design DataStore")
    destination = Path(destination)
    details = {
        item.step: read_numeric_tree(data, item.value)
        for item in state.scores_detail
    }
    if any(not isinstance(value, dict) for value in details.values()):
        raise UnsupportedPortableNativeState("score details must decode to mappings")
    evaluations = {
        item.step: restore_scores(
            item.value,
            data,
            destination / "scores" / f"step_{item.step:08d}",
        )
        for item in state.evaluation_scores
    }
    rows = read_numeric_tree(data, state.row_history)
    if not isinstance(rows, list):
        raise UnsupportedPortableNativeState("row history must decode to a list")
    # Construction observes a new "initial" assessment. Restore storage only;
    # neither a scientific decision nor an observation may be replayed here.
    decisions = Decisions.__new__(Decisions)
    decisions.sequences = {
        item.step: dict(item.assignments) for item in state.sequences
    }
    decisions.scores = {item.step: item.value for item in state.scores}
    decisions.evaluation_results = evaluations
    decisions.scores_detail = details
    decisions.accepted_steps = list(state.accepted_steps)
    decisions.last_accepted = state.last_accepted
    decisions.stage_history = dict(state.stage_history)
    decisions.row_history = list(rows)
    decisions.assessments = []
    # Older records contain no attempt identities or assessment evidence.
    # Leave that history absent, and do not infer a revision from accepted_steps.
    decisions.observation_errors = []
    decisions._attempt_count = 0
    decisions._accepted_attempt = None
    decisions._revision_known = False
    if state.observation_state is not None:
        observation = read_numeric_tree(data, state.observation_state)
        decisions.observation_errors = observation["observation_errors"]
        decisions._attempt_count = observation["_attempt_count"]
        decisions._accepted_attempt = observation["_accepted_attempt"]
        decisions._revision_known = observation["_revision_known"]
        for index, assessment in enumerate(state.assessments):
            values = read_numeric_tree(data, assessment.details)
            result = (
                None if assessment.evaluation_scores is None
                else restore_scores(
                    assessment.evaluation_scores,
                    data,
                    destination / "assessments" / f"assessment_{index:08d}",
                )
            )
            decisions.assessments.append(
                DecisionAssessment(**values, evaluation_result=result)
            )
    return decisions


def _nested_types() -> tuple[type[Any], ...]:
    return (
        ScientificRunContract,
        SequenceAtStep,
        ScoreAtStep,
        BlobAtStep,
        DecisionAssessmentState,
        DecisionState,
        NativeRuntimeContext,
        PendingEdit,
        ProposalSnapshot,
        ProposalStackState,
        NativeStepState,
        RootBootstrapRequest,
        DecisionRequest,
        DecisionResult,
        RootResult,
        EvaluationResult,
        Candidate,
        Blob,
        Key,
        Evidence,
        Artifact,
        Proposal,
        Scored,
        Assessment,
    )


SCIENTIFIC_RUN_CONTRACT_CODEC = DataclassCodec(
    ScientificRunContract,
    "astrolabe.portable-native.scientific-contract/v1",
    nested_types=_nested_types(),
)
DECISION_STATE_CODEC = DataclassCodec(
    DecisionState,
    "astrolabe.portable-native.decision-state-codec/v1",
    nested_types=_nested_types(),
)
NATIVE_RUNTIME_CONTEXT_CODEC = DataclassCodec(
    NativeRuntimeContext,
    "astrolabe.portable-native.runtime-context-codec/v1",
    nested_types=_nested_types(),
)
PROPOSAL_STACK_CODEC = DataclassCodec(
    ProposalStackState,
    "astrolabe.portable-native.proposal-stack/v1",
    nested_types=_nested_types(),
)
NATIVE_STEP_STATE_CODEC = DataclassCodec(
    NativeStepState,
    "astrolabe.portable-native.step-state-codec/v1",
    nested_types=_nested_types(),
)
ROOT_BOOTSTRAP_REQUEST_CODEC = DataclassCodec(
    RootBootstrapRequest,
    "astrolabe.portable-native.root-bootstrap-request/v1",
    nested_types=_nested_types(),
)
ROOT_RESULT_CODEC = DataclassCodec(
    RootResult,
    "astrolabe.portable-native.root-result/v1",
    nested_types=_nested_types(),
)
DECISION_REQUEST_CODEC = DataclassCodec(
    DecisionRequest,
    "astrolabe.portable-native.decision-request/v1",
    nested_types=_nested_types(),
)
DECISION_RESULT_CODEC = DataclassCodec(
    DecisionResult,
    "astrolabe.portable-native.decision-result/v1",
    nested_types=_nested_types(),
)
EVALUATION_RESULT_CODEC = DataclassCodec(
    EvaluationResult,
    "astrolabe.portable-native.evaluation-result/v1",
    nested_types=_nested_types(),
)

PORTABLE_NATIVE_REFERENCE_CODECS = (
    NATIVE_SCORES_REFERENCE_CODECS
    + (
        NUMERIC_TREE_CODEC,
        DECISION_STATE_CODEC,
        NATIVE_RUNTIME_CONTEXT_CODEC,
        NATIVE_STEP_STATE_CODEC,
    )
)


__all__ = [
    "BlobAtStep",
    "DECISION_REQUEST_CODEC",
    "DECISION_RESULT_CODEC",
    "DECISION_STATE_CODEC",
    "DECISION_STATE_SCHEMA",
    "DecisionRequest",
    "DecisionAssessmentState",
    "DecisionResult",
    "DecisionState",
    "EVALUATION_RESULT_CODEC",
    "EvaluationResult",
    "NATIVE_RUNTIME_CONTEXT_CODEC",
    "NATIVE_STEP_STATE_CODEC",
    "NATIVE_STEP_STATE_SCHEMA",
    "NUMERIC_TREE_CODEC",
    "NUMERIC_TREE_CODEC_ID",
    "NativeRuntimeContext",
    "NativeStepState",
    "NumericTreeCodec",
    "PORTABLE_NATIVE_REFERENCE_CODECS",
    "PROPOSAL_STACK_CODEC",
    "PendingEdit",
    "ProposalSnapshot",
    "ProposalStackState",
    "ROOT_BOOTSTRAP_REQUEST_CODEC",
    "ROOT_RESULT_CODEC",
    "RootBootstrapRequest",
    "RootResult",
    "ScoreAtStep",
    "SequenceAtStep",
    "SCIENTIFIC_CONTRACT_SCHEMA",
    "SCIENTIFIC_RUN_CONTRACT_CODEC",
    "ScientificRunContract",
    "UnsupportedPortableNativeState",
    "read_numeric_tree",
    "restore_decision_state",
    "snapshot_decision_state",
    "store_numeric_tree",
]
