"""Read-only projection of phased-search checkpoints into legacy beam outputs.

This module reconstructs the legacy evaluation journal from immutable Runtime
history. It does not execute work, choose a frontier, inspect model-specific
artifacts, or import a scientific family.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
import math
from typing import Any

from .design.phased_search import PhasedBeamSearchPolicy, PhasedBeamState
from .design.types import (
    Attempt,
    Candidate,
    History,
    Runtime,
    Scored,
    TrialOutcome,
    Work,
)


@dataclass(frozen=True)
class BeamCandidateView:
    """Legacy serialized beam record.

    The fields intentionally match the prior ``BeamCandidate`` JSON shape even
    though the unified framework uses typed identities, samples and evidence.
    """

    evaluation_id: int
    sequence: str
    score: float
    structure_path: str
    generation: int
    parent_id: int | None
    metrics: dict[str, Any]


@dataclass
class BeamReport:
    """Rebuilt legacy evaluation rows, selected beam and append-only events."""

    records: list[dict[str, Any]]
    beam: list[BeamCandidateView]
    events: list[dict[str, Any]]
    proposals: int
    evaluations: int
    duplicates: int
    generation: int
    budget_complete: bool


_ROW_FIELDS = frozenset(
    {
        "evaluation_id",
        "proposal_step",
        "sequence",
        "generation",
        "parent_id",
        "status",
        "score",
        "structure_path",
        "entered_beam",
        "in_final_beam",
        "metrics",
    }
)


def _scalar_score(scored: Scored, *, subject: str) -> float:
    values = scored.assessment.values
    if len(values) != 1:
        raise ValueError(
            f"{subject} legacy beam score requires exactly one assessment value"
        )
    score = float(values[0])
    if not math.isfinite(score):
        raise ValueError(f"{subject} legacy beam score must be finite")
    return score


def _details(
    scored: Scored,
    details_of: Callable[[Scored], tuple[str, dict]],
) -> tuple[str, dict[str, Any]]:
    structure_path, raw_metrics = details_of(scored)
    if not isinstance(structure_path, str):
        raise TypeError("details_of must return structure_path as str")
    if not isinstance(raw_metrics, dict):
        raise TypeError("details_of must return metrics as dict")
    metrics = dict(raw_metrics)
    collisions = _ROW_FIELDS.intersection(metrics)
    if collisions:
        raise ValueError(
            f"details_of metric names collide with legacy fields: {sorted(collisions)}"
        )
    return structure_path, metrics


def _record(
    *,
    evaluation_id: int,
    sequence: str,
    generation: int,
    parent_id: int | None,
    proposal_step: int,
    status: str,
    score: float | None,
    structure_path: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "evaluation_id": evaluation_id,
        "proposal_step": proposal_step,
        "sequence": sequence,
        "generation": generation,
        "parent_id": parent_id,
        "status": status,
        "score": score,
        "structure_path": structure_path,
        "entered_beam": None,
        "in_final_beam": False,
    }
    row.update(metrics)
    row["metrics"] = dict(metrics)
    return row


def _event_fields(
    row: dict[str, Any],
    *,
    status: str | None = None,
    started: bool = False,
) -> dict[str, Any]:
    event = {
        key: row[key]
        for key, value in row.items()
        if key in _ROW_FIELDS and key != "metrics"
    }
    event["metrics"] = {} if started else dict(row["metrics"])
    if status is not None:
        event["status"] = status
    if started:
        event.update(
            score=None,
            structure_path="",
            entered_beam=None,
            in_final_beam=False,
        )
    return event


def _view(row: dict[str, Any]) -> BeamCandidateView:
    score = row["score"]
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        raise ValueError("legacy beam view requires a numeric score")
    return BeamCandidateView(
        evaluation_id=row["evaluation_id"],
        sequence=row["sequence"],
        score=float(score),
        structure_path=row["structure_path"],
        generation=row["generation"],
        parent_id=row["parent_id"],
        metrics=dict(row["metrics"]),
    )


def _component_identity(runtime: Runtime, prediction_alias: str):
    operation = runtime.operations.get(prediction_alias)
    if operation is None:
        raise KeyError(
            f"prediction alias {prediction_alias!r} is not registered in Runtime"
        )
    return operation.descriptor.identity


def _prediction_attempt(
    runtime: Runtime,
    work_id: str,
    prediction_alias: str,
) -> Attempt | None:
    component = _component_identity(runtime, prediction_alias)
    consumed = ("completed", "failed", "unknown")
    matches = tuple(
        attempt
        for attempt in runtime.attempts(work_id)
        if attempt.key.component == component and attempt.status in consumed
    )
    return matches[-1] if matches else None


def _fallback_status(attempt: Attempt) -> str:
    # A completed prediction whose measurement did not complete is still a
    # failed legacy evaluation row; transport/terminal failures retain their
    # explicit status.
    return "failed" if attempt.status == "completed" else attempt.status


def _parent_id(work: Work, candidate_ids: dict[str, int]) -> int | None:
    if not work.parents:
        return None
    if len(work.parents) != 1:
        raise ValueError("legacy beam projection requires one parent per Work")
    digest = work.parents[0].candidate.identity.digest
    if digest not in candidate_ids:
        raise ValueError("Work parent has no chronological evaluation identity")
    return candidate_ids[digest]


def _proposal_index(work: Work) -> int:
    replicate = work.sample.replicate
    marker = "_proposal_"
    position = replicate.rfind(marker)
    if position < 0:
        raise ValueError("proposal Work sample has no proposal ordinal")
    encoded = replicate[position + len(marker):].split("_", 1)[0]
    try:
        return int(encoded)
    except ValueError:
        raise ValueError("proposal Work sample has an invalid proposal ordinal") from None


def _checkpoint_state(
    runtime: Runtime,
    policy: PhasedBeamSearchPolicy,
    checkpoint: dict[str, Any],
) -> PhasedBeamState:
    reference = checkpoint.get("state")
    if reference is None:
        raise ValueError("run/state checkpoint does not contain a policy state")
    state = runtime.data.get(reference, policy.codec)
    if not isinstance(state, PhasedBeamState):
        raise TypeError("run/state does not contain a PhasedBeamState")
    return state


def _events_for_row(
    events: list[dict[str, Any]],
    row: dict[str, Any],
    *,
    outcome: str,
    seed: int | None,
    error: str | None = None,
) -> None:
    started = _event_fields(row, status="running", started=True)
    started["seed"] = seed
    events.append({"event": "evaluation_started", **started})
    if outcome == "completed":
        events.append(
            {
                "event": "evaluation_completed",
                **_event_fields(row),
                "seed": seed,
            }
        )
    else:
        failed = _event_fields(row, status=outcome)
        failed["seed"] = seed
        if error is not None:
            failed["error"] = error
        events.append({"event": "evaluation_failed", **failed})


def build_beam_report(
    runtime: Runtime,
    policy: PhasedBeamSearchPolicy,
    *,
    initial: Candidate,
    sequence_of: Callable[[Candidate], str],
    details_of: Callable[[Scored], tuple[str, dict]],
    prediction_alias: str,
) -> BeamReport:
    """Rebuild the legacy beam journal without executing or selecting work.

    ``history`` is replayed through the same policy to recover generation
    boundaries and admitted children. Evaluation IDs are assigned from durable
    history order: root is ``0`` and each evaluation attempt consumes the next
    ID, including failed/unknown attempts. Duplicate proposals consume no
    evaluation ID.

    ``details_of`` returns ``(structure_path, raw_metrics)``. Sequence and
    artifact interpretation remain owned by the scientific caller.
    """

    if not isinstance(initial, Candidate):
        raise TypeError("initial must be a Candidate")
    checkpoint = runtime.load("run/state")
    root = runtime.peek_completion("run/root")
    records: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    candidate_ids: dict[str, int] = {}
    candidate_rows: dict[str, dict[str, Any]] = {}
    row_by_evaluation: dict[int, dict[str, Any]] = {}

    if root is None:
        attempt = _prediction_attempt(runtime, "root", prediction_alias)
        if attempt is None:
            raise ValueError("root has neither a scored completion nor a consumed prediction attempt")
        row = _record(
            evaluation_id=0,
            sequence=sequence_of(initial),
            generation=0,
            parent_id=None,
            proposal_step=0,
            status=_fallback_status(attempt),
            score=None,
            structure_path="",
            metrics={},
        )
        records.append(row)
        row_by_evaluation[0] = row
        candidate_ids[initial.identity.digest] = 0
        candidate_rows[initial.identity.digest] = row
        _events_for_row(
            events,
            row,
            outcome=_fallback_status(attempt),
            seed=getattr(policy, "evaluation_seed", None),
            error=f"root prediction attempt is {attempt.status}",
        )
        return BeamReport(
            records=records,
            beam=[],
            events=events,
            proposals=0,
            evaluations=len(records),
            duplicates=0,
            generation=0,
            budget_complete=False,
        )

    if not isinstance(root, Scored):
        raise TypeError("run/root completion is not a Scored value")
    if root.candidate.identity != initial.identity:
        raise ValueError("run/root completion belongs to another initial Candidate")
    root_sequence = sequence_of(root.candidate)
    root_path, root_metrics = _details(root, details_of)
    root_row = _record(
        evaluation_id=0,
        sequence=root_sequence,
        generation=0,
        parent_id=None,
        proposal_step=0,
        status="completed",
        score=_scalar_score(root, subject="root"),
        structure_path=root_path,
        metrics=root_metrics,
    )
    records.append(root_row)
    row_by_evaluation[0] = root_row
    candidate_ids[root.candidate.identity.digest] = 0
    candidate_rows[root.candidate.identity.digest] = root_row
    _events_for_row(
        events,
        root_row,
        outcome="completed",
        seed=getattr(policy, "evaluation_seed", None),
    )

    state = policy.start((root,))
    root_entered = any(
        selected.candidate.identity == root.candidate.identity
        for selected in state.frontier
    )
    root_row["entered_beam"] = root_entered
    root_views = [_view(candidate_rows[value.candidate.identity.digest]) for value in state.frontier]
    events.append(
        {
            "event": "beam_selected",
            "generation": 0,
            "candidates": [0],
            "beam": [asdict(view) for view in root_views],
        }
    )

    if checkpoint is None:
        raise ValueError("run/state checkpoint is missing after a completed root")
    history = checkpoint.get("history")
    if not isinstance(history, History):
        raise TypeError("run/state checkpoint does not contain typed History")
    pending = tuple(checkpoint.get("pending", ()))
    current_generation_children: list[int] = []
    next_evaluation_id = 1

    def emit_generation_boundary(
        *,
        before: PhasedBeamState,
        after: PhasedBeamState,
    ) -> None:
        selected_digests = {
            scored.candidate.identity.digest for scored in after.frontier
        }
        for evaluation_id in current_generation_children:
            row = row_by_evaluation[evaluation_id]
            digest = next(
                candidate_digest
                for candidate_digest, mapped_id in candidate_ids.items()
                if mapped_id == evaluation_id
            )
            row["entered_beam"] = digest in selected_digests
        views = [
            _view(candidate_rows[value.candidate.identity.digest])
            for value in after.frontier
        ]
        events.append(
            {
                "event": "beam_selected",
                "generation": before.generation,
                "candidates": list(current_generation_children),
                "beam": [asdict(view) for view in views],
            }
        )
        current_generation_children.clear()

    for outcome in history.records:
        if not isinstance(outcome, TrialOutcome):
            raise TypeError("search history contains a non-TrialOutcome record")
        work = runtime.work_entry(outcome.work)
        before = state
        if work.stage == "propose":
            after = policy.tell(before, (outcome,))
            if after.duplicates > before.duplicates:
                if len(outcome.proposals) != 1:
                    raise ValueError("duplicate proposal outcome lacks its Proposal")
                proposal = outcome.proposals[0]
                events.append(
                    {
                        "event": "duplicate_proposal",
                        "generation": before.generation,
                        "parent_id": _parent_id(work, candidate_ids),
                        "proposal_index": _proposal_index(work),
                        "sequence": sequence_of(proposal.candidate),
                        "seed": work.sample.seed,
                    }
                )
            state = after
        elif work.stage == "evaluate":
            if work.candidate is None:
                raise ValueError("evaluation Work lacks its accepted Candidate")
            evaluation_id = next_evaluation_id
            next_evaluation_id += 1
            parent_id = _parent_id(work, candidate_ids)
            completion = outcome.status == "completed"
            if completion:
                if len(outcome.children) != 1:
                    raise ValueError(
                        "completed evaluation outcome must contain one Scored child"
                    )
                child = outcome.children[0]
                child_sequence = sequence_of(child.candidate)
                structure_path, metrics = _details(child, details_of)
                row = _record(
                    evaluation_id=evaluation_id,
                    proposal_step=int(work.sample.index),
                    sequence=child_sequence,
                    generation=before.generation,
                    parent_id=parent_id,
                    status="completed",
                    score=_scalar_score(child, subject="completed evaluation"),
                    structure_path=structure_path,
                    metrics=metrics,
                )
                current_generation_children.append(evaluation_id)
            else:
                row = _record(
                    evaluation_id=evaluation_id,
                    proposal_step=int(work.sample.index),
                    sequence=sequence_of(work.candidate),
                    generation=before.generation,
                    parent_id=parent_id,
                    status=outcome.status,
                    score=None,
                    structure_path="",
                    metrics={},
                )
            records.append(row)
            row_by_evaluation[evaluation_id] = row
            candidate_ids[work.candidate.identity.digest] = evaluation_id
            candidate_rows[work.candidate.identity.digest] = row
            _events_for_row(
                events,
                row,
                outcome=outcome.status,
                seed=work.sample.seed,
                error=None,
            )
            state = policy.tell(before, (outcome,))
        else:
            raise ValueError(f"unsupported Work stage {work.stage!r}")

        generation_changed = state.generation != before.generation
        if generation_changed or (state.finished and not before.finished):
            emit_generation_boundary(before=before, after=state)

    for work in pending:
        if not isinstance(work, Work):
            raise TypeError("run/state pending entries must be typed Work")
        if work.stage != "evaluate":
            continue
        if work.candidate is None:
            raise ValueError("pending evaluation Work lacks its accepted Candidate")
        attempt = _prediction_attempt(runtime, work.identity, prediction_alias)
        if attempt is None:
            continue
        evaluation_id = next_evaluation_id
        next_evaluation_id += 1
        status = _fallback_status(attempt)
        row = _record(
            evaluation_id=evaluation_id,
            proposal_step=int(work.sample.index),
            sequence=sequence_of(work.candidate),
            generation=state.generation,
            parent_id=_parent_id(work, candidate_ids),
            status=status,
            score=None,
            structure_path="",
            metrics={},
        )
        records.append(row)
        row_by_evaluation[evaluation_id] = row
        candidate_ids[work.candidate.identity.digest] = evaluation_id
        candidate_rows[work.candidate.identity.digest] = row
        _events_for_row(
            events,
            row,
            outcome=status,
            seed=work.sample.seed,
            error=f"prediction attempt is {attempt.status}",
        )

    expected = _checkpoint_state(runtime, policy, checkpoint)
    if state != expected:
        raise ValueError("durable history does not reproduce the committed policy state")

    final_digests = {
        scored.candidate.identity.digest for scored in expected.frontier
    }
    for row in records:
        row_digest = next(
            (
                digest
                for digest, mapped_id in candidate_ids.items()
                if mapped_id == row["evaluation_id"]
            ),
            None,
        )
        row["in_final_beam"] = row_digest in final_digests
    beam = [
        _view(candidate_rows[value.candidate.identity.digest])
        for value in expected.frontier
    ]
    return BeamReport(
        records=records,
        beam=beam,
        events=events,
        proposals=expected.proposals,
        evaluations=len(records),
        duplicates=expected.duplicates,
        generation=expected.generation,
        budget_complete=expected.evaluations >= expected.max_evaluations,
    )


__all__ = ["BeamCandidateView", "BeamReport", "build_beam_report"]
