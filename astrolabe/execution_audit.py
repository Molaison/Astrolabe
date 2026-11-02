"""Observational run evidence; never a second search-state or resume engine."""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from contextlib import contextmanager
import json
import logging
import math
from pathlib import Path
from typing import Any

from astrolabe.runtime_support import atomic_text_writer

logger = logging.getLogger(__name__)


@dataclass
class EvaluationInvocation:
    invocation_id: str
    step: int
    committed_stage: str | None
    execution_stage: str | None
    backend_alias: str
    inputs: tuple[Any, ...]
    status: str = "running"
    records: tuple[Any, ...] = ()
    error_type: str | None = None
    observation_errors: list[dict[str, str]] = field(default_factory=list)


@dataclass
class EvaluationJournal:
    """Single-producer observations, independent of model and search RNGs."""

    run_id: str
    invocations: list[EvaluationInvocation] = field(default_factory=list)
    transitions: list[dict[str, Any]] = field(default_factory=list)
    cache_reuses: list[dict[str, Any]] = field(default_factory=list)
    observation_errors: list[dict[str, str]] = field(default_factory=list)
    _execution_stages: list[str | None] = field(default_factory=list, repr=False)

    @contextmanager
    def execution_stage(self, stage: str | None):
        """Keep observation-only context out of backend runtime dictionaries."""
        self._execution_stages.append(stage)
        try:
            yield
        finally:
            self._execution_stages.pop()

    @property
    def has_execution_stage(self) -> bool:
        return bool(self._execution_stages)

    @property
    def current_execution_stage(self) -> str | None:
        return self._execution_stages[-1] if self._execution_stages else None

    def begin(self, *, step, committed_stage, execution_stage, backend_alias, inputs):
        invocation = EvaluationInvocation(
            invocation_id=f"{self.run_id}::evaluation-{len(self.invocations) + 1:06d}",
            step=int(step), committed_stage=committed_stage,
            execution_stage=execution_stage, backend_alias=backend_alias,
            inputs=tuple(inputs),
        )
        self.invocations.append(invocation)
        return invocation

    def record_error(self, phase: str, error: Exception, invocation=None):
        record = {"phase": phase, "error_type": type(error).__name__}
        self.observation_errors.append(record)
        if invocation is not None:
            invocation.observation_errors.append(record)

    def finish(self, invocation, *, result=None, error: BaseException | None = None):
        # Backend completion is independent of optional metadata collection.
        invocation.status = "failed" if error is not None else "completed"
        invocation.error_type = type(error).__name__ if error is not None else None
        try:
            invocation.records = tuple(getattr(result, "evaluation_records", None) or ())
        except Exception as exc:
            self.record_error("finish_evaluation_observation", exc, invocation)


def _json_value(value: Any) -> Any:
    """Keep compact scalar evidence; large native arrays remain in artifacts."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"nonfinite": str(value)}
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _json_value(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    # NumPy-like values do not require importing a backend for audit output.
    if hasattr(value, "shape") and hasattr(value, "tolist"):
        if int(getattr(value, "size", 0)) > 256:
            return {"array_shape": list(value.shape), "dtype": str(value.dtype),
                    "values": "omitted; inspect source artifacts"}
        return _json_value(value.tolist())
    return {"unserialized_type": f"{type(value).__module__}.{type(value).__qualname__}"}


def build_execution_audit(*, run_id: str, status: str, decisions=None, journal=None) -> dict:
    assessments = []
    for record in getattr(decisions, "assessments", ()) or ():
        snapshot = getattr(record, "evidence_snapshot", None)
        # Compatibility with externally supplied historical records. New records
        # always carry an observation-time snapshot, never a live result view.
        if snapshot is None:
            result = record.evaluation_result
            snapshot = {name: getattr(result, name, None) for name in (
                "evaluation_records", "complex_paths", "derived_structures", "candidate_groups"
            )}
        assessments.append({
            "assessment_id": record.assessment_id,
            "attempt_id": record.attempt_id,
            "accepted_revision": record.accepted_revision,
            "kind": record.kind, "step": record.step,
            "accepted": record.accepted, "stage": record.stage,
            "weighted_score": record.weighted_score,
            "score_details": record.score_details,
            "sequences": record.sequences,
            "evaluation_records": snapshot.get("evaluation_records") or (),
            "complex_paths": snapshot.get("complex_paths") or {},
            "derived_structures": snapshot.get("derived_structures") or {},
            "candidate_groups": snapshot.get("candidate_groups") or (),
        })
    return _json_value({
        "schema_version": 1, "run_id": run_id, "terminal_status": status,
        "recovery": {"exact_resume": False, "purpose": "observational evidence only"},
        "identity_scope": "assessment and attempt IDs are local to this run",
        "assessment_coverage": "completed decision methods; exceptions may have partial effects",
        "sampling": "input seeds are observed, backend RNG/sample identity is not inferred",
        "invocations": getattr(journal, "invocations", ()) or (),
        "stage_entry_observations": getattr(journal, "transitions", ()) or (),
        "cache_reuses": getattr(journal, "cache_reuses", ()) or (),
        "observation_errors": (
            list(getattr(journal, "observation_errors", ()) or ())
            + list(getattr(decisions, "observation_errors", ()) or ())
        ),
        "assessments": assessments,
    })


def write_execution_audit(out_dir, *, status: str, decisions=None, journal=None) -> Path | None:
    """Best effort: an optional evidence file must not change a search outcome."""
    path = Path(out_dir) / "execution_audit.json"
    try:
        payload = build_execution_audit(
            run_id=str(Path(out_dir).resolve()), status=status,
            decisions=decisions, journal=journal,
        )
        with atomic_text_writer(path) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except Exception:
        logger.exception("Could not write observational execution audit: %s", path)
        return None
    return path
