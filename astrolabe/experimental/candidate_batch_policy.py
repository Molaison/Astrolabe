"""Candidate-named compatibility facade over generic collection primitives.

The implementation lives in ``astrolabe.design.collection``. This module keeps
the experimental import surface stable while the process-specific worker lives
in ``candidate_batch_runner``.
"""

from __future__ import annotations

from astrolabe.design.collection import (
    CollectionConfig,
    CollectionExecutionError,
    CollectionTrigger,
    CompletionPool,
    FailurePolicy,
    Job,
    JobResult,
    JobStatus,
    ResultCollection,
    ResultSelector,
    payload_to_summary,
    select_highest_score,
)


CandidatePoolConfig = CollectionConfig
CandidateTask = Job
CandidateResult = JobResult
CandidateCollection = ResultCollection
CandidatePoolScheduler = CompletionPool
CandidateExecutionError = CollectionExecutionError
CandidateSelector = ResultSelector
FailPolicy = FailurePolicy
CandidateStatus = JobStatus
select_highest_score_candidate = select_highest_score


__all__ = [
    "CandidateCollection",
    "CandidateExecutionError",
    "CandidatePoolConfig",
    "CandidatePoolScheduler",
    "CandidateResult",
    "CandidateSelector",
    "CandidateStatus",
    "CandidateTask",
    "CollectionTrigger",
    "FailPolicy",
    "payload_to_summary",
    "select_highest_score_candidate",
]
