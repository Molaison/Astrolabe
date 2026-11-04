from __future__ import annotations

import random
import threading

import pytest

from astrolabe.experimental.candidate_batch_policy import (
    CandidateExecutionError,
    CandidatePoolConfig,
    CandidatePoolScheduler,
    CandidateResult,
    CandidateTask,
)


WRONG_IDENTITIES = [
    pytest.param("foreign", 1, id="wrong-candidate"),
    pytest.param("submitted", 0, id="past-generation"),
    pytest.param("submitted", 2, id="future-generation"),
    pytest.param("foreign", 2, id="wrong-candidate-and-generation"),
]


@pytest.mark.parametrize("returned_id,returned_generation", WRONG_IDENTITIES)
def test_foreign_result_is_failed_under_submitted_identity(
    returned_id: str, returned_generation: int
) -> None:
    submitted = CandidateTask("submitted", 1, payload={"sequence": "AAAA"})
    foreign_result = CandidateResult(
        task=CandidateTask(returned_id, returned_generation),
        status="completed",
        score=999.0,
        payload={"sequence": "GGGG"},
    )
    config = CandidatePoolConfig(num_candidates=1, min_completed=1, max_wait_s=1.0)
    with CandidatePoolScheduler(config, lambda _task: foreign_result) as scheduler:
        scheduler.submit_generation(1, [submitted])
        collection = scheduler.collect_until_ready(target_generation=1)
        assert collection is not None
        assert collection.selected is None
        assert collection.completed == ()
        assert len(collection.failed) == 1
        failure = collection.failed[0]
        assert failure.task is submitted
        assert failure.score is None
        assert failure.payload is None
        assert "candidate result identity mismatch" in failure.error
        assert scheduler.collect_until_ready(target_generation=0) is None
        assert scheduler.collect_until_ready(target_generation=2) is None
        assert scheduler.drain() == []


@pytest.mark.parametrize("returned_id,returned_generation", WRONG_IDENTITIES)
@pytest.mark.parametrize("collection_method", ["collect_until_ready", "drain"])
def test_foreign_result_obeys_fail_fast(
    returned_id: str, returned_generation: int, collection_method: str
) -> None:
    submitted = CandidateTask("submitted", 1)
    foreign_result = CandidateResult(
        task=CandidateTask(returned_id, returned_generation),
        status="completed",
        score=999.0,
    )
    config = CandidatePoolConfig(
        num_candidates=1,
        min_completed=1,
        max_wait_s=1.0,
        fail_policy="fail-fast",
    )
    with CandidatePoolScheduler(config, lambda _task: foreign_result) as scheduler:
        scheduler.submit_generation(1, [submitted])
        with pytest.raises(
            CandidateExecutionError,
            match="Candidate submitted failed: candidate result identity mismatch",
        ):
            getattr(scheduler, collection_method)()


@pytest.mark.parametrize("return_scalar", [False, True], ids=["result", "scalar"])
def test_valid_result_preserves_scientific_data_and_worker_rng(return_scalar: bool) -> None:
    rng = random.Random(17)
    expected_rng = random.Random(17)
    expected_score = expected_rng.random()
    payload = {"sequence": ["AAAA"]}
    submitted = CandidateTask("submitted", 1, payload={"parent": "CCCC"})
    returned_results = []

    def worker(task: CandidateTask) -> CandidateResult | float:
        score = rng.random()
        if return_scalar:
            return score
        # A transport may reconstruct the task; identity is not object equality.
        result = CandidateResult(
            task=CandidateTask(task.candidate_id, task.generation),
            status="completed",
            score=score,
            payload=payload,
            duration_s=0.125,
        )
        returned_results.append(result)
        return result

    config = CandidatePoolConfig(num_candidates=1, min_completed=1, max_wait_s=1.0)
    with CandidatePoolScheduler(config, worker) as scheduler:
        scheduler.submit_generation(1, [submitted])
        collection = scheduler.collect_until_ready(target_generation=1)

    assert collection is not None
    assert len(collection.completed) == 1
    assert collection.failed == ()
    assert collection.selected is not None
    assert collection.selected.score == expected_score
    assert collection.selected.task.candidate_id == submitted.candidate_id
    assert collection.selected.task.generation == submitted.generation
    assert rng.getstate() == expected_rng.getstate()
    assert submitted.payload == {"parent": "CCCC"}
    if return_scalar:
        assert collection.selected.task is submitted
    else:
        assert collection.selected is returned_results[0]
        assert collection.selected.payload is payload
        assert collection.selected.duration_s == 0.125


@pytest.mark.parametrize("wrong_generation", [0, 2], ids=["past", "future"])
@pytest.mark.parametrize("invalid_position", [0, 1], ids=["first", "last"])
def test_mismatched_submission_generation_rejects_entire_batch(
    wrong_generation: int, invalid_position: int
) -> None:
    calls = []

    def worker(task: CandidateTask) -> float:
        calls.append(task.candidate_id)
        return 0.5

    valid = [CandidateTask("a", 1), CandidateTask("b", 1)]
    invalid = list(valid)
    invalid[invalid_position] = CandidateTask(
        valid[invalid_position].candidate_id, wrong_generation
    )
    config = CandidatePoolConfig(num_candidates=2, min_completed=2, max_wait_s=1.0)
    with CandidatePoolScheduler(config, worker) as scheduler:
        with pytest.raises(ValueError, match="candidate task generation mismatch"):
            scheduler.submit_generation(1, invalid)
        assert scheduler.drain() == []
        assert calls == []
        # Rejection must not reserve either identity or poison a valid retry.
        scheduler.submit_generation(1, valid)
        collections = scheduler.drain()

    assert sorted(calls) == ["a", "b"]
    assert len(collections) == 1
    assert len(collections[0].completed) == 2


def test_failing_input_iterator_does_not_partially_submit() -> None:
    calls = []

    def worker(task: CandidateTask) -> float:
        calls.append(task.candidate_id)
        return 0.5

    def incomplete_batch():
        yield CandidateTask("prefix", 1)
        raise RuntimeError("producer failed while building generation")

    config = CandidatePoolConfig(num_candidates=2, min_completed=1, max_wait_s=1.0)
    with CandidatePoolScheduler(config, worker) as scheduler:
        with pytest.raises(RuntimeError, match="producer failed"):
            scheduler.submit_generation(1, incomplete_batch())
        assert scheduler.drain() == []
    assert calls == []


def test_fixed_parent_generator_keeps_task_count_and_best_score() -> None:
    calls = []
    scores = {"a": 0.25, "b": 0.75, "c": 0.5}

    def worker(task: CandidateTask) -> float:
        calls.append(task.candidate_id)
        return scores[task.candidate_id]

    config = CandidatePoolConfig(num_candidates=3, min_completed=3, max_wait_s=1.0)
    with CandidatePoolScheduler(config, worker) as scheduler:
        scheduler.submit_generation(1, (CandidateTask(name, 1) for name in scores))
        collection = scheduler.collect_until_ready(target_generation=1)
    assert sorted(calls) == ["a", "b", "c"]
    assert collection is not None
    assert {result.candidate_id: result.score for result in collection.completed} == scores
    assert collection.selected is not None
    assert collection.selected.candidate_id == "b"


@pytest.mark.parametrize("reuse_object", [False, True], ids=["same-id", "same-task"])
def test_duplicate_submission_identity_rejects_entire_batch(reuse_object: bool) -> None:
    calls = []

    def worker(task: CandidateTask) -> float:
        calls.append(task.candidate_id)
        return float(task.payload)

    prefix = CandidateTask("prefix", 1, payload=0.25)
    candidate = CandidateTask("duplicate", 1, payload=0.5)
    duplicate = candidate if reuse_object else CandidateTask("duplicate", 1, payload=0.9)
    config = CandidatePoolConfig(num_candidates=3, min_completed=1, max_wait_s=1.0)
    with CandidatePoolScheduler(config, worker) as scheduler:
        with pytest.raises(ValueError, match="duplicate candidate task identity"):
            scheduler.submit_generation(1, [prefix, candidate, duplicate])
        assert scheduler.drain() == []
        assert calls == []
        scheduler.submit_generation(1, [prefix, candidate])
        collections = scheduler.drain()

    assert sorted(calls) == ["duplicate", "prefix"]
    assert len(collections) == 1
    assert collections[0].selected is not None
    assert collections[0].selected.score == 0.5


@pytest.mark.parametrize("phase", ["running", "collected", "failed"])
def test_submitted_identity_cannot_be_replayed(phase: str) -> None:
    calls = []
    entered = threading.Event()
    release = threading.Event()

    def worker(task: CandidateTask) -> float:
        calls.append(task.candidate_id)
        if task.candidate_id == "original":
            entered.set()
            if phase == "running" and not release.wait(2.0):
                raise RuntimeError("test controller did not release worker")
            if phase == "failed":
                raise RuntimeError("worker failure does not authorize replay")
        return 0.5

    candidate = CandidateTask("original", 1)
    prefix = CandidateTask("new-prefix", 1)
    config = CandidatePoolConfig(num_candidates=2, min_completed=1, max_wait_s=1.0)
    with CandidatePoolScheduler(config, worker) as scheduler:
        try:
            scheduler.submit_generation(1, [candidate])
            assert entered.wait(1.0)
            if phase != "running":
                scheduler.drain()
            with pytest.raises(ValueError, match="duplicate candidate task identity"):
                scheduler.submit_generation(1, [prefix, candidate])
        finally:
            release.set()
        scheduler.drain()
        assert calls == ["original"]
        # A rejected batch cannot consume the fresh prefix's identity.
        scheduler.submit_generation(1, [prefix])
        scheduler.drain()
    assert calls == ["original", "new-prefix"]


def test_same_candidate_id_in_distinct_generations_is_allowed() -> None:
    calls = []

    def worker(task: CandidateTask) -> float:
        calls.append((task.candidate_id, task.generation))
        return float(task.payload)

    config = CandidatePoolConfig(num_candidates=2, min_completed=1, max_wait_s=1.0)
    with CandidatePoolScheduler(config, worker) as scheduler:
        scheduler.submit_generation(1, [CandidateTask("candidate", 1, payload=0.25)])
        scheduler.submit_generation(2, [CandidateTask("candidate", 2, payload=0.75)])
        first = scheduler.collect_until_ready(target_generation=1)
        second = scheduler.collect_until_ready(target_generation=2)
    assert sorted(calls) == [("candidate", 1), ("candidate", 2)]
    assert first is not None and first.selected is not None
    assert second is not None and second.selected is not None
    assert first.selected.score == 0.25
    assert second.selected.score == 0.75
