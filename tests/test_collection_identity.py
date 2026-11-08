"""Identity guards at the shared pool boundary, independent of worker receipts."""

from dataclasses import replace

import pytest

from astrolabe.design.collection import (
    CollectionConfig,
    CollectionExecutionError,
    CompletionPool,
    InlineExecutor,
    Job,
    JobResult,
)
from astrolabe.design.runtime import OutcomeUnknown


def _pool(worker, **kwargs):
    return CompletionPool(
        CollectionConfig(num_jobs=1, min_completed=1, max_wait_s=1),
        worker,
        executor_factory=InlineExecutor,
        **kwargs,
    )


@pytest.mark.parametrize("bad_suffix", ["generation", "duplicate", "iterator"])
def test_invalid_generation_has_no_admission_or_worker_side_effects(bad_suffix):
    admitted, executed = [], []
    first = Job("first", generation=1)

    def jobs():
        yield first
        if bad_suffix == "iterator":
            raise ValueError("input iterator failed")
        yield first if bad_suffix == "duplicate" else Job("last", generation=2)

    with _pool(lambda job: executed.append(job) or 1.0, admission=admitted.append) as pool:
        with pytest.raises((ValueError, CollectionExecutionError)):
            pool.submit_generation(1, jobs())
        assert admitted == executed == []
        # A rejected batch has not reserved its otherwise valid prefix.
        pool.submit_generation(1, [first])
        assert pool.collect_until_ready().selected.job is first
    assert admitted == executed == [first]


@pytest.mark.parametrize("state", ["pending", "completed", "failed", "admitted"])
def test_duplicate_job_is_rejected_before_new_prefix_dispatch(state):
    job = Job("original", generation=1)
    executed, admitted = [], []

    def worker(request):
        executed.append(request)
        if state == "failed":
            raise ValueError("worker failed")
        return 0.5

    def admission(request):
        admitted.append(request)
        if state == "admitted":
            return JobResult(job=request, score=0.5)
        return None

    with _pool(worker, admission=admission) as pool:
        pool.submit_generation(1, [job])
        if state in {"completed", "failed"}:
            pool.collect_until_ready()
        before = (list(admitted), list(executed))
        with pytest.raises(CollectionExecutionError, match="already submitted"):
            pool.submit_generation(1, [Job("new", generation=1), job])
        assert (admitted, executed) == before
        if state in {"pending", "admitted"}:
            pool.collect_until_ready()
        with pytest.raises(CollectionExecutionError, match="already submitted"):
            pool.submit_generation(1, [job])


def test_job_ids_remain_pool_wide_across_generations():
    calls = []
    with _pool(lambda job: calls.append(job) or 0.5) as pool:
        original = Job("same", generation=1)
        pool.submit_generation(1, [original])
        pool.collect_until_ready()
        with pytest.raises(CollectionExecutionError):
            pool.submit_generation(2, [Job("fresh", 2), Job("same", 2)])
        assert calls == [original]


def test_rehydrated_changed_content_is_rejected_before_prefix_dispatch():
    original = Job("restored", 1, {"sequence": "AAA"})
    calls = []
    with _pool(lambda job: calls.append(job) or 0.5, admission=lambda job: None) as pool:
        pool.rehydrate(ready_results=[JobResult(original, score=0.5)])
        with pytest.raises(CollectionExecutionError, match="changed its admitted content"):
            pool.submit_generation(
                1, [Job("new", 1), replace(original, payload={"sequence": "BBB"})]
            )
        assert calls == []


@pytest.mark.parametrize("restored", ["ready", "issued"])
def test_rehydrated_identity_requires_terminal_proof_to_resubmit(restored):
    job = Job("restored", 1)
    calls = []
    with _pool(lambda request: calls.append(request) or 0.5) as pool:
        pool.rehydrate(**(
            {"ready_results": [JobResult(job, score=0.5)]}
            if restored == "ready" else {"issued_jobs": [job]}
        ))
        with pytest.raises(OutcomeUnknown, match="terminal proof"):
            pool.submit_generation(1, [job])
        assert calls == []


@pytest.mark.parametrize("restored", ["ready", "issued"])
def test_rehydrated_terminal_admission_is_reconciled_once_without_execution(restored):
    job = Job("restored", 1)
    saved = JobResult(job, score=0.5)
    calls = []
    with _pool(
        lambda request: calls.append(request) or 0.5,
        admission=lambda request: saved,
    ) as pool:
        pool.rehydrate(collection_index=3, **(
            {"ready_results": [saved]} if restored == "ready" else {"issued_jobs": [job]}
        ))
        pool.submit_generation(1, [job])
        collection, = pool.drain()
        assert collection.results == (saved,)
        assert collection.collection_index == 4
        with pytest.raises(CollectionExecutionError, match="already submitted"):
            pool.submit_generation(1, [job])
    assert calls == []


def test_rehydrated_issued_job_keeps_owner_unknown_fence():
    job = Job("issued", 1)

    def admission(request):
        raise OutcomeUnknown("owner has no terminal proof")

    with _pool(lambda request: pytest.fail("must not replay"), admission=admission) as pool:
        pool.rehydrate(issued_jobs=[job])
        with pytest.raises(OutcomeUnknown, match="owner has no terminal proof"):
            pool.submit_generation(1, [job])
        with pytest.raises(OutcomeUnknown, match="no owner terminal proof"):
            pool.drain()


@pytest.mark.parametrize("status", ["issued", "unknown"])
def test_nonterminal_admission_cannot_clear_rehydrated_issued_fence(status):
    job = Job("issued", 1)
    with _pool(
        lambda request: pytest.fail("must not replay"),
        admission=lambda request: JobResult(job, status=status),
    ) as pool:
        pool.rehydrate(issued_jobs=[job])
        with pytest.raises(CollectionExecutionError, match="non-terminal"):
            pool.submit_generation(1, [job])
        with pytest.raises(OutcomeUnknown, match="no owner terminal proof"):
            pool.drain()


@pytest.mark.parametrize("foreign", [Job("other", 1), Job("submitted", 2)])
@pytest.mark.parametrize("status", ["completed", "failed"])
@pytest.mark.parametrize("collect", ["collect_until_ready", "drain"])
def test_foreign_worker_result_is_failed_under_submitted_identity(foreign, status, collect):
    job = Job("submitted", 1)
    observed = []
    with _pool(
        lambda request: JobResult(foreign, status=status, score=100, payload="foreign"),
        terminal_observed=lambda request, result: observed.append((request, result)),
    ) as pool:
        pool.submit_generation(1, [job])
        result = getattr(pool, collect)()
        collection = result[0] if collect == "drain" else result
        assert collection.selected is None
        failed, = collection.failed
        assert failed.job is job
        assert failed.score is None and failed.payload is None
        assert "identity" in failed.error
        assert observed == [(job, failed)]
        assert pool.collect_until_ready(target_generation=2) is None


@pytest.mark.parametrize("collect", ["collect_until_ready", "drain"])
def test_foreign_worker_result_honors_fail_fast(collect):
    job = Job("submitted", 1)
    observed = []
    with CompletionPool(
        CollectionConfig(num_jobs=1, fail_policy="fail-fast"),
        lambda request: JobResult(Job("foreign", 1), score=100),
        terminal_observed=lambda request, result: observed.append((request, result)),
        executor_factory=InlineExecutor,
    ) as pool:
        pool.submit_generation(1, [job])
        with pytest.raises(CollectionExecutionError, match="Job submitted failed:.*identity"):
            getattr(pool, collect)()
    assert observed[0][0] is observed[0][1].job is job


def test_valid_identity_uses_original_job_and_preserves_result_payload():
    # A transport can reconstruct Job without preserving opaque input objects.
    class Opaque:
        def __eq__(self, other):
            raise AssertionError("identity validation must not compare opaque payloads")

    job = Job("same", 1, Opaque())
    output = object()
    returned = JobResult(Job("same", 1, Opaque()), score=0.7, payload=output, duration_s=2)
    with _pool(lambda request: returned) as pool:
        pool.submit_generation(1, [job])
        result = pool.collect_until_ready().selected
    assert result.job is job
    assert result.payload is output
    assert result.score == 0.7 and result.duration_s == 2


def test_valid_finite_iterator_and_float_results_keep_selection_protocol():
    with _pool(lambda request: float(request.payload)) as pool:
        pool.submit_generation(1, (Job(name, 1, score) for name, score in [
            ("a", 0.4), ("b", 0.8),
        ]))
        collections = pool.drain()
    assert len(collections) == 1
    assert collections[0].selected.job_id == "b"
    assert {result.job_id for result in collections[0].results} == {"a", "b"}
