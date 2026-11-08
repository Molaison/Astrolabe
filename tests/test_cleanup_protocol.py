"""Failure-injected adapter contracts; no native model or GPU qualification."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from astrolabe import model_execution as execution
from astrolabe.runtime_support import CleanupRecord, CloseReport, ResourceOwner


class IdempotentOwner:
    """Fixture with an explicitly retry-safe shutdown operation."""

    attached = False

    def __init__(self, failure=None, action=None):
        self.failure = failure
        self.action = action
        self.calls = 0

    def shutdown(self):
        self.calls += 1
        if self.action is not None:
            self.action()
        if self.failure is not None:
            raise self.failure


def test_failure_receipt_is_retained_until_explicit_retry():
    failure = OSError("adapter could not confirm shutdown")
    resource = IdempotentOwner(failure)
    owner = ResourceOwner()
    owner.add(resource, "primary")
    owner.add(resource, "alias")
    first = owner.close()
    assert isinstance(first, CloseReport)
    assert isinstance(first.failed[0], CleanupRecord)
    assert not first.complete
    assert first.failed[0].resource is resource
    assert first.failed[0].error is failure
    assert first.failed[0].operation == "shutdown"
    assert first.failed[0].attempts == 1
    assert first.failed[0].label == "primary"
    resource.failure = None
    assert owner.close().failed[0].error is failure
    owner.add(resource, "another alias")
    assert resource.calls == 1
    retried = owner.close(retry_failed=True)
    assert retried.complete
    assert retried.records[0].state == "closed"
    assert retried.records[0].attempts == 2
    assert retried.records[0].error is None
    assert retried.records[0].errors == (failure,)
    assert first.failed[0].error is failure  # historical snapshots do not mutate
    owner.close(retry_failed=True)
    assert resource.calls == 2


def test_successful_run_inside_except_still_surfaces_cleanup_failure():
    resource = IdempotentOwner(OSError("cleanup failed"))
    try:
        raise ValueError("unrelated caller exception")
    except ValueError:
        with pytest.raises(RuntimeError, match="failed to stop"):
            with execution.model_run_scope():
                execution.register_model_owner(resource)


def test_explicit_retry_failure_is_attempted_only_once_per_call():
    resource = IdempotentOwner(OSError("still live"))
    owner = ResourceOwner()
    owner.add(resource, "retry-safe")
    owner.close()
    report = owner.close(retry_failed=True)
    assert resource.calls == 2
    assert report.failed[0].attempts == 2
    assert len(report.failed[0].errors) == 2
    assert not report.complete


@pytest.mark.parametrize("failure", [OSError("ordinary"), KeyboardInterrupt("cleanup")])
def test_cleanup_failure_does_not_skip_remaining_resources(failure):
    first, second = IdempotentOwner(failure), IdempotentOwner()
    owner = ResourceOwner()
    owner.add(first, "first")
    owner.add(second, "second")
    report = owner.close()
    assert first.calls == second.calls == 1
    assert [record.state for record in report.records] == ["failed", "closed"]
    assert report.failed[0].error is failure


def test_missing_cleanup_operation_is_not_claimed_as_successful_shutdown():
    owner = ResourceOwner()
    owner.add(object(), "no adapter")
    report = owner.close()
    assert report.complete
    assert report.records[0].state == "skipped"
    assert report.records[0].operation is None


def test_concurrent_and_reentrant_close_are_nonblocking_and_do_not_double_shutdown():
    owner = ResourceOwner()
    entered, release = threading.Event(), threading.Event()
    reentrant = []

    def shutdown():
        reentrant.append(owner.close())
        entered.set()
        assert release.wait(5)

    resource = IdempotentOwner(action=shutdown)
    owner.add(resource, "blocked")
    with ThreadPoolExecutor(max_workers=2) as executor:
        draining = executor.submit(owner.close)
        try:
            assert entered.wait(5)
            concurrent = executor.submit(owner.close).result(timeout=5)
            for report in [*reentrant, concurrent, owner.report]:
                assert report.draining
                assert not report.complete
                assert report.records[0].state == "closing"
                assert report.records[0].attempts == 1
        finally:
            release.set()
        assert draining.result(timeout=5).complete
    assert owner.close().complete
    assert resource.calls == 1


def test_shutdown_can_join_a_thread_that_calls_close_without_deadlock():
    owner = ResourceOwner()
    reports = []

    def shutdown():
        child = threading.Thread(target=lambda: reports.append(owner.close()), daemon=True)
        child.start()
        child.join(timeout=5)
        assert not child.is_alive()

    resource = IdempotentOwner(action=shutdown)
    owner.add(resource, "joins callback")
    assert owner.close().complete
    assert reports[0].draining
    assert resource.calls == 1


def test_late_registration_during_and_after_drain_is_cleaned_and_retained():
    owner = ResourceOwner()
    during = IdempotentOwner()
    first = IdempotentOwner(action=lambda: owner.add(during, "during"))
    owner.add(first, "first")
    assert owner.close().complete
    after = IdempotentOwner(OSError("late failure"))
    owner.add(after, "after")
    assert first.calls == during.calls == after.calls == 1
    assert owner.report.failed[0].resource is after
    owner.close()
    assert after.calls == 1


def test_startup_failure_stays_observable_and_stopped_requires_success():
    scope = execution.StartupScope()
    bad, good = IdempotentOwner(OSError("failed stop")), IdempotentOwner()
    scope.add(bad)
    scope.add(bad)
    scope.add(good)
    assert len(scope.cancel()) == 1
    assert scope.stopped == {id(good)}
    assert scope.owners == [bad, good]
    assert scope.cleanup_report.failed[0].resource is bad
    bad.failure = None
    assert len(scope.cancel()) == 1
    assert bad.calls == good.calls == 1
    assert scope.cancel(retry_failed=True) == []
    assert scope.stopped == {id(bad), id(good)}
    assert bad.calls == 2


def test_late_cancelled_startup_registration_retains_failed_owner_and_error():
    scope = execution.StartupScope()
    scope.cancel()
    resource = IdempotentOwner(OSError("late owner still live"))
    with pytest.raises(RuntimeError, match="cancelled; owner cleanup failed") as caught:
        scope.add(resource)
    assert scope.owners == [resource]
    assert not scope.stopped
    assert caught.value.cleanup_report.failed[0].resource is resource
    assert caught.value.cleanup_scopes == (scope,)
    assert len(scope.cancel()) == 1
    assert resource.calls == 1
    resource.failure = None
    assert scope.cancel(retry_failed=True) == []
    assert scope.stopped == {id(resource)}


@pytest.mark.parametrize("late", [False, True])
def test_startup_shutdown_runs_without_lifecycle_lock_and_guards_inflight(late):
    scope = execution.StartupScope()
    entered, release = threading.Event(), threading.Event()
    reentrant = []

    def shutdown():
        reentrant.append(scope.cancel())
        entered.set()
        assert release.wait(5)

    resource = IdempotentOwner(action=shutdown)
    if late:
        scope.cancel()
    else:
        scope.add(resource)
    with ThreadPoolExecutor(max_workers=2) as executor:
        draining = executor.submit(scope.add, resource) if late else executor.submit(scope.cancel)
        try:
            assert entered.wait(5)
            assert scope.lock.acquire(timeout=5), "shutdown held the lifecycle lock"
            scope.lock.release()
            assert executor.submit(scope.cancel).result(timeout=5) == []
            assert not scope.stopped
            assert not scope.cleanup_report.complete
            assert scope.cleanup_report.records[0].state == "closing"
        finally:
            release.set()
        if late:
            with pytest.raises(RuntimeError, match="cancelled"):
                draining.result(timeout=5)
        else:
            assert draining.result(timeout=5) == []
    assert scope.cleanup_report.complete
    assert scope.stopped == {id(resource)}
    assert resource.calls == 1


def test_startup_cancel_cleans_a_reentrantly_created_late_owner():
    scope = execution.StartupScope()
    late = IdempotentOwner()

    def shutdown():
        with pytest.raises(RuntimeError, match="cancelled"):
            scope.add(late)

    first = IdempotentOwner(action=shutdown)
    scope.add(first)
    assert scope.cancel() == []
    assert scope.stopped == {id(first), id(late)}
    assert first.calls == late.calls == 1


def test_run_cleanup_preserves_scientific_error_and_retains_failed_owner():
    original = ValueError("scientific failure")
    bad, good = IdempotentOwner(KeyboardInterrupt("cleanup interrupt")), IdempotentOwner()
    with pytest.raises(ValueError) as caught:
        with execution.model_run_scope() as scope:
            execution.register_model_owner(good)
            execution.register_model_owner(bad)
            raise original
    assert caught.value is original
    assert bad.calls == good.calls == 1
    assert original.cleanup_report.failed[0].resource is bad
    assert original.cleanup_scopes == (scope,)
    assert "cleanup interrupt" in " ".join(original.__notes__)


def test_run_cleanup_surfaces_error_when_no_scientific_error_and_allows_explicit_retry():
    resource = IdempotentOwner(OSError("stop failed"))
    with pytest.raises(RuntimeError, match="failed to stop") as caught:
        with execution.model_run_scope() as scope:
            execution.register_model_owner(resource)
    assert caught.value.__cause__ is resource.failure
    assert caught.value.cleanup_scopes == (scope,)
    assert caught.value.cleanup_report.failed[0].resource is resource
    resource.failure = None
    scope.cancel()
    assert resource.calls == 1
    assert scope.cancel(retry_failed=True) == []
    assert resource.calls == 2


@pytest.mark.parametrize("failure", [None, OSError("failed shutdown")])
def test_construction_cancel_and_run_unwind_share_receipts_without_implicit_retry(failure):
    resource = IdempotentOwner(failure)
    original = ValueError("bad checkpoint")
    with execution.model_run_scope() as run:
        with execution.startup_scope() as startup:
            with execution.model_construction_scope():
                execution.register_model_owner(resource)
            with pytest.raises(ValueError) as caught:
                with execution.model_construction_scope():
                    execution.register_model_owner(resource)
                    raise original
            assert caught.value is original
            startup.cancel()
            assert resource.calls == 1
        if failure is not None:
            resource.failure = None
            assert startup.cancel(retry_failed=True) == []
    assert run.cleanup_report.complete
    assert resource.calls == (1 if failure is None else 2)


def test_failed_partial_constructor_retains_owner_without_startup_scope():
    resource = IdempotentOwner(OSError("partial stop failed"))
    original = ValueError("constructor failed before returning")
    with pytest.raises(ValueError) as caught:
        with execution.model_construction_scope() as scope:
            execution.register_model_owner(resource)
            raise original
    assert caught.value is original
    assert caught.value.cleanup_report.failed[0].resource is resource
    assert caught.value.cleanup_scopes == (scope,)
    assert resource.calls == 1


def test_failed_startup_receipt_is_not_retried_on_run_exit():
    resource = IdempotentOwner(OSError("unknown stop"))
    original = ValueError("startup failure")
    with pytest.raises(ValueError) as caught:
        with execution.model_run_scope():
            with execution.startup_scope() as startup:
                with execution.model_construction_scope():
                    execution.register_model_owner(resource)
                startup.cancel()
                raise original
    assert caught.value is original
    assert original.cleanup_report.failed[0].resource is resource
    assert resource.calls == 1


def test_parent_drain_during_construction_transfer_cleans_all_local_owners():
    first, second = IdempotentOwner(), IdempotentOwner(OSError("late stop failed"))
    with execution.model_run_scope() as parent:
        with pytest.raises(RuntimeError, match="cancelled") as caught:
            with execution.model_construction_scope():
                execution.register_model_owner(first)
                execution.register_model_owner(second)
                parent.cancel()
        assert first.calls == second.calls == 1
        assert caught.value.cleanup_report.failed[0].resource is second
