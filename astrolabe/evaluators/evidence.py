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
    # Identifies the dispatch that created this group, never a sample.
    origin_id: int | str | None = None


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
    candidates = tuple(result.candidate_scores or ())
    group = None
    if candidates:
        # Construct the complete snapshot and every pending assignment before
        # publishing any metadata. A malformed later candidate must not leave
        # earlier ones or the canonical result partially annotated.
        group = CandidateGroup(
            candidate_paths=tuple(
                tuple((name, Path(path)) for name, path in candidate.complex_paths.items())
                for candidate in candidates
            ),
            invocation_ids=(invocation_id,),
            origin_id=invocation_id,
        )
    result_groups = result.candidate_groups
    if group is not None:
        result_groups = _merge_candidate_groups((*(result_groups or ()), group))
    updates = [(result, (result.evaluation_records or ()) + records, result_groups)]
    seen = {id(result)}
    for candidate in candidates:
        if id(candidate) in seen:
            continue
        seen.add(id(candidate))
        allowed = set(candidate.names)
        context = tuple(record for record in records if record.input.oligomer in allowed)
        candidate_groups = _merge_candidate_groups((*(candidate.candidate_groups or ()), group))
        updates.append((candidate, (candidate.evaluation_records or ()) + context, candidate_groups))

    # StandardScores fields are ordinary attributes. Keep prior values for a
    # rollback if a custom subclass rejects a metadata assignment.
    old = [(obj, obj.evaluation_records, obj.candidate_groups) for obj, _, _ in updates]
    try:
        for obj, new_records, new_groups in updates:
            obj.evaluation_records = new_records
            obj.candidate_groups = new_groups
    except BaseException:
        for obj, old_records, old_groups in old:
            object.__setattr__(obj, "evaluation_records", old_records)
            object.__setattr__(obj, "candidate_groups", old_groups)
        raise
    return result


def _filter_group(group: CandidateGroup, allowed: set[str]) -> CandidateGroup | None:
    paths = tuple(tuple((key, path) for key, path in candidate if key in allowed)
                  for candidate in group.candidate_paths)
    if not any(paths):
        return None
    return CandidateGroup(paths, group.invocation_ids, group.alignment, group.origin_id)


def filter_candidate_groups(groups: tuple[CandidateGroup, ...], allowed: set[str]) -> tuple[CandidateGroup, ...]:
    return tuple(filtered for group in groups if (filtered := _filter_group(group, allowed)) is not None)


def _merge_candidate_groups(groups: tuple[CandidateGroup, ...]) -> tuple[CandidateGroup, ...]:
    """Coalesce filtered views of one known origin, keeping request order."""
    combined: list[CandidateGroup] = []
    positions: dict[int | str, int] = {}
    for group in groups:
        origin = group.origin_id
        if origin is None or origin not in positions:
            if origin is not None:
                positions[origin] = len(combined)
            combined.append(group)
            continue
        index = positions[origin]
        previous = combined[index]
        if (len(previous.candidate_paths) != len(group.candidate_paths)
                or previous.invocation_ids != group.invocation_ids
                or previous.alignment != group.alignment):
            raise ValueError("Inconsistent candidate group for the same origin")
        merged_paths = []
        for before, after in zip(previous.candidate_paths, group.candidate_paths):
            paths = dict(before)
            for name, path in after:
                if name in paths and paths[name] != path:
                    raise ValueError("Conflicting candidate path for the same origin")
                paths[name] = path
            merged_paths.append(tuple(paths.items()))
        combined[index] = CandidateGroup(tuple(merged_paths), previous.invocation_ids,
                                         previous.alignment, origin)
    return tuple(combined)


def merge_evidence_metadata(
    merged: StandardScores, results: list[StandardScores], *, strict: bool = False
) -> StandardScores:
    """Attach evidence to an existing score merge, preserving its candidate policy.

    Records and candidate groups follow result/request order. Candidate groups
    are observations only and are never promoted to canonical candidate_scores.
    """
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
    groups = _merge_candidate_groups(tuple(groups))
    if strict and len(groups) > 1:
        raise ValueError("Cannot align multiple candidate groups without candidate identity")
    # Leave merged.candidate_scores untouched: the caller owns canonical selection.
    merged.post_eval_hook_reports = reports or None
    merged.derived_structures = derived or None
    merged.evaluation_records = tuple(records) or None
    merged.candidate_groups = groups or None
    return merged
