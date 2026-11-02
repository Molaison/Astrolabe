"""Observational run evidence; never a second search-state or resume engine."""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
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


@dataclass
class EvaluationJournal:
    """Single-producer observations, independent of model and search RNGs."""

    run_id: str
    invocations: list[EvaluationInvocation] = field(default_factory=list)
    transitions: list[dict[str, Any]] = field(default_factory=list)

    def begin(self, *, step, committed_stage, execution_stage, backend_alias, inputs):
        invocation = EvaluationInvocation(
            invocation_id=f"{self.run_id}::evaluation-{len(self.invocations) + 1:06d}",
            step=int(step), committed_stage=committed_stage,
            execution_stage=execution_stage, backend_alias=backend_alias,
            inputs=tuple(inputs),
        )
        self.invocations.append(invocation)
        return invocation

    def finish(self, invocation, *, result=None, error: BaseException | None = None):
        invocation.status = "failed" if error is not None else "completed"
        invocation.error_type = type(error).__name__ if error is not None else None
        invocation.records = tuple(getattr(result, "evaluation_records", None) or ())


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
        result = record.evaluation_result
        assessments.append({
            "assessment_id": record.assessment_id,
            "attempt_id": record.attempt_id,
            "accepted_revision": record.accepted_revision,
            "kind": record.kind, "step": record.step,
            "accepted": record.accepted, "stage": record.stage,
            "weighted_score": record.weighted_score,
            "score_details": record.score_details,
            "sequences": record.sequences,
            "evaluation_records": getattr(result, "evaluation_records", None) or (),
            "complex_paths": getattr(result, "complex_paths", None) or {},
            "derived_structures": getattr(result, "derived_structures", None) or {},
            "candidate_groups": getattr(result, "candidate_groups", None) or (),
        })
    return _json_value({
        "schema_version": 1, "run_id": run_id, "terminal_status": status,
        "recovery": {"exact_resume": False, "purpose": "observational evidence only"},
        "identity_scope": "assessment and attempt IDs are local to this run",
        "assessment_coverage": "completed decision methods; exceptions may have partial effects",
        "sampling": "input seeds are observed, backend RNG/sample identity is not inferred",
        "invocations": getattr(journal, "invocations", ()) or (),
        "stage_entry_observations": getattr(journal, "transitions", ()) or (),
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
