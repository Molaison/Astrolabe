"""Declared provenance for evaluator outputs; no inference of sample identity."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from astrolabe.evaluators.utils import StandardScores


@dataclass(frozen=True)
class DerivedStructure:
    source_path: Path
    output_path: Path
    operation: str


def merge_evidence_metadata(
    merged: StandardScores, results: list[StandardScores]
) -> StandardScores:
    """Attach metadata to an existing score merge without pairing candidates.

    Multiple candidate lists cannot be merged safely without an explicit identity
    shared across results. Refuse them before mutating the output.
    """
    candidate_lists = [r.candidate_scores for r in results if r.candidate_scores]
    if len(candidate_lists) > 1:
        raise ValueError("Cannot merge multiple candidate lists without candidate identity")
    reports = {}
    derived = {}
    for result in results:
        reports.update(result.post_eval_hook_reports or {})
        for oligo, records in (result.derived_structures or {}).items():
            derived.setdefault(oligo, []).extend(records)
    merged.candidate_scores = list(candidate_lists[0]) if candidate_lists else None
    merged.post_eval_hook_reports = reports or None
    merged.derived_structures = derived or None
    return merged
