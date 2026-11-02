"""Immutable observations of evaluator dispatch and derived artifacts.

These records do not own candidate state or assert an unknown sample identity.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from astrolabe.core.folding_input import Input
    from astrolabe.evaluators.utils import StandardScores


@dataclass(frozen=True)
class DerivedStructure:
    source_path: Path
    output_path: Path
    operation: str


@dataclass(frozen=True)
class InputObservation:
    oligomer: str
    # Physical chain order and IDs are retained; ligands have no sequence.
    chains: tuple[tuple[str, str, str | None], ...]
    rng_seeds: tuple[int, ...]


@dataclass(frozen=True)
class EvaluationRecord:
    invocation_id: int | str
    step: int
    committed_stage: str | None
    execution_stage: str | None
    backend_alias: str
    input: InputObservation
    # Canonical evaluator output for this oligomer, before any post-eval hook.
    original_complex_path: Path | None
    # rng_seeds are the dispatched Input values, not verified backend usage.
    seed_semantics: str = "dispatched_input_rng_seeds"
    sample_identity: None = None


@dataclass(frozen=True)
class CandidateGroup:
    # Ordered path snapshots within one result; no cross-group correspondence.
    candidate_paths: tuple[tuple[tuple[str, Path], ...], ...]
    invocation_ids: tuple[int | str, ...]
    alignment: str = "unknown"


def observe_evaluation_inputs(inputs: Mapping[str, Input]) -> tuple[InputObservation, ...]:
    """Capture dispatch inputs before the evaluator can mutate them."""
    return tuple(
        InputObservation(
            oligomer=oligo,
            chains=tuple((chain.id, type(chain).__name__, getattr(chain, "sequence", None))
                         for chain in value.chains),
            rng_seeds=tuple(value.rng_seeds),
        )
        for oligo, value in inputs.items()
    )


def attach_evaluation_records(
    result: StandardScores,
    observations: tuple[InputObservation, ...],
    *,
    invocation_id: int | str,
    step: int,
    committed_stage: str | None,
    execution_stage: str | None,
    backend_alias: str,
) -> StandardScores:
    """Attach invocation context, including to candidates, before post-eval work.

    Candidate context refers to the invocation's canonical output path; it does
    not claim that path identifies the candidate or any particular sample.
    """
    paths = result.complex_paths or {}
    records = tuple(
        EvaluationRecord(
            invocation_id=invocation_id, step=step,
            committed_stage=committed_stage, execution_stage=execution_stage,
            backend_alias=backend_alias, input=observation,
            original_complex_path=(Path(paths[observation.oligomer])
                                   if paths.get(observation.oligomer) is not None else None),
        )
        for observation in observations
    )
    result.evaluation_records = (result.evaluation_records or ()) + records
    for candidate in result.candidate_scores or ():
        allowed = set(candidate.names)
        context = tuple(record for record in records if record.input.oligomer in allowed)
        candidate.evaluation_records = (candidate.evaluation_records or ()) + context
    return result


def _filter_group(group: CandidateGroup, allowed: set[str]) -> CandidateGroup | None:
    paths = tuple(tuple((key, path) for key, path in candidate if key in allowed)
                  for candidate in group.candidate_paths)
    if not any(paths):
        return None
    return CandidateGroup(paths, group.invocation_ids, group.alignment)


def filter_candidate_groups(groups: tuple[CandidateGroup, ...], allowed: set[str]) -> tuple[CandidateGroup, ...]:
    return tuple(filtered for group in groups if (filtered := _filter_group(group, allowed)) is not None)


def merge_evidence_metadata(
    merged: StandardScores, results: list[StandardScores], *, strict: bool = False
) -> StandardScores:
    """Attach evidence to an existing score merge, preserving its candidate policy.

    Records and candidate groups follow result/request order. Candidate groups
    are observations only and are never promoted to canonical candidate_scores.
    """
    if strict and sum(bool(r.candidate_scores or r.candidate_groups) for r in results) > 1:
        raise ValueError("Cannot align multiple candidate groups without candidate identity")
    reports = {}
    derived = {}
    records = []
    groups = []
    for result in results:
        reports.update(result.post_eval_hook_reports or {})
        for oligo, artifacts in (result.derived_structures or {}).items():
            derived.setdefault(oligo, []).extend(artifacts)
        records.extend(result.evaluation_records or ())
        if result.candidate_groups:
            groups.extend(result.candidate_groups)
        elif result.candidate_scores:
            ids = tuple(dict.fromkeys(record.invocation_id for record in result.evaluation_records or ()))
            groups.append(CandidateGroup(
                candidate_paths=tuple(tuple((name, Path(path)) for name, path in
                                            candidate.complex_paths.items())
                                      for candidate in result.candidate_scores),
                invocation_ids=ids,
            ))
    # Leave merged.candidate_scores untouched: the caller owns canonical selection.
    merged.post_eval_hook_reports = reports or None
    merged.derived_structures = derived or None
    merged.evaluation_records = tuple(records) or None
    merged.candidate_groups = tuple(groups) or None
    return merged
