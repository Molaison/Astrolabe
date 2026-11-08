from __future__ import annotations

from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path

import pytest

from astrolabe.design.branch_owner import NativeBranchTrial, NativeJobReceipt
from astrolabe.design.branch_types import (
    BranchAction,
    BranchEvent,
    BranchJob,
    BranchMember,
    BranchProposal,
    BranchTerminal,
    BranchingState,
    branch_codec,
)
from astrolabe.design.branching import BranchingPolicy
from astrolabe.design.composition import ControlRun
from astrolabe.design.runtime import ContractConflict, OutcomeUnknown, Runtime
from astrolabe.design.store import DataclassCodec, JSONCodec
from astrolabe.design.types import key_for
from astrolabe.model_daemon import (
    DaemonChildExited,
    DaemonDisconnected,
    DaemonRequestFailed,
    RequestOutcomeUnknown,
)


JSON = JSONCodec()
RECEIPT = DataclassCodec(
    NativeJobReceipt,
    "astrolabe.native-job-receipt/v1",
)


class _CompletedPool:
    def __init__(self, worker) -> None:
        self.worker = worker
        self.calls: list[dict] = []

    def submit(self, _worker, task):
        self.calls.append(task)
        future = Future()
        try:
            future.set_result(self.worker(task))
        except BaseException as error:
            future.set_exception(error)
        return future


class _RejectingPool:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def submit(self, _worker, task):
        self.calls.append(task)
        raise AssertionError("native submit must not occur")


def _policy(*, proposal_batch_size: int = 1, max_evaluations: int = 12):
    return BranchingPolicy(
        width=1,
        proposals_per_parent=max(1, proposal_batch_size),
        max_evaluations=max_evaluations,
        proposal_cap=20,
        seed=7,
        evaluation_seed=13,
        proposal_batch_size=proposal_batch_size,
        evaluation_width=1,
    )


def _runtime(tmp_path: Path, name: str = "runtime") -> Runtime:
    return Runtime(
        root=tmp_path / name,
        scientific_contract=key_for(("test.design-branch-owner/v1",)),
    )


def _ready_proposal_request(
    runtime: Runtime,
    *,
    proposal_batch_size: int = 1,
    prior_proposals: int = 0,
):
    policy = _policy(proposal_batch_size=proposal_batch_size)
    root_job = BranchJob(
        phase="evaluate",
        id=0,
        generation=0,
        seed=policy.evaluation_seed,
    )
    root = BranchMember(
        id=0,
        key="root",
        value=runtime.data.put({"sequence": "ROOT"}, JSON),
        score=0.0,
    )
    state = policy.start(root_job)
    root_action = policy.ask(state)
    assert root_action.jobs == (root_job,)
    state = policy.admit(state, root_action)
    state = policy.tell(
        state,
        BranchEvent(
            (
                BranchTerminal(
                    job=root_job,
                    status="completed",
                    value=root,
                ),
            )
        ),
    )
    advance = policy.ask(state)
    assert advance.jobs == ()
    state = policy.admit(state, advance)
    assert policy.tell(state, BranchEvent(())) == state
    state = replace(state, proposals=prior_proposals)

    proposals = policy.ask(state)
    assert all(job.phase == "propose" for job in proposals.jobs)
    admitted = policy.admit(state, proposals)
    assert admitted.batch == proposals.jobs
    return policy, BranchAction(
        state=admitted,
        jobs=proposals.jobs,
        wait="all",
    )


def _task_factory(job: BranchJob) -> dict:
    return {
        "id": job.id,
        "kind": job.phase,
        "generation": job.generation,
        "parent_id": job.parent.id if job.parent is not None else None,
        "phase": job.phase,
        "seed": job.seed,
        "sequence": "ACDE",
        "attempt": 1,
        "charge_token": f"charge-{job.identity}-1",
    }


def _raw_result(task: dict, **changes) -> dict:
    raw = {
        "id": task["id"],
        "kind": task["kind"],
        "generation": task["generation"],
        "parent_id": task["parent_id"],
        "_request_hash": task["_request_hash"],
        "seed": task["seed"],
        "sequence": task["sequence"],
        "attempt": task["attempt"],
        "native_payload": {"source": "fake-owner"},
    }
    raw.update(changes)
    return raw


def _proposal_result(runtime: Runtime, job: BranchJob, raw: dict):
    return BranchProposal(
        id=job.id,
        parent_id=job.parent.id,
        key=f"child-{job.id}",
        value=runtime.data.put(raw, JSON),
    )


def _evaluation_result(runtime: Runtime, job: BranchJob, raw: dict):
    return BranchMember(
        id=job.id,
        key=raw.get("key", "root"),
        value=runtime.data.put(raw, JSON),
        score=float(raw.get("score", 1.0)),
    )


def _trial(
    runtime: Runtime,
    pool,
    *,
    worker,
    result_factory,
    reserve=None,
    retry=None,
    identity_name: str = "trial",
) -> NativeBranchTrial:
    return NativeBranchTrial(
        pool,
        runtime,
        identity=key_for(("test.design-branch-owner/trial/v1", identity_name)),
        worker=worker,
        task_factory=_task_factory,
        result_factory=result_factory,
        probe=lambda _job, _task: None,
        reserve=reserve,
        retry=retry,
    )


def _receipt(runtime: Runtime, name: str):
    return runtime.load("native-jobs/" + name, RECEIPT)


def _install_unknown_receipt(
    runtime: Runtime,
    request: BranchAction,
) -> None:
    inner = RequestOutcomeUnknown("native owner outcome is uncertain")

    def fail(_task):
        wrapped = RuntimeError("wrapped native transport failure")
        wrapped.__cause__ = inner
        raise wrapped

    trial = _trial(
        runtime,
        _CompletedPool(fail),
        worker=fail,
        result_factory=lambda job, raw: _proposal_result(runtime, job, raw),
        identity_name="unknown-seed",
    )
    with pytest.raises(RuntimeError, match="wrapped native transport"):
        trial.perform(request)
    assert _receipt(runtime, request.jobs[0].identity).status == "unknown"


def test_verified_proposal_batch_recovers_without_duplicate_native_submit(
    tmp_path: Path,
) -> None:
    with _runtime(tmp_path) as runtime:
        _, request = _ready_proposal_request(runtime, proposal_batch_size=2)
        first_pool = _CompletedPool(lambda task: _raw_result(task))
        first_trial = _trial(
            runtime,
            first_pool,
            worker=lambda task: _raw_result(task),
            result_factory=lambda job, raw: _proposal_result(runtime, job, raw),
            identity_name="first",
        )

        first = first_trial.perform(request)

        assert tuple(terminal.job for terminal in first.terminals) == request.jobs
        assert len(first_pool.calls) == len(request.jobs)
        for job in request.jobs:
            receipt = _receipt(runtime, job.identity)
            assert receipt.status == "completed"
            assert receipt.result is not None
            assert runtime.data.get(receipt.result, JSON) == _raw_result(
                first_trial.task_record(job)
            )

        recovery_pool = _RejectingPool()
        recovered_trial = _trial(
            runtime,
            recovery_pool,
            worker=lambda task: _raw_result(task),
            result_factory=lambda job, raw: _proposal_result(runtime, job, raw),
            identity_name="recovery",
        )

        assert recovered_trial.recover(request) is None
        assert recovered_trial.recover(request) is None
        recovered = recovered_trial.perform(request)

        assert tuple(terminal.job for terminal in recovered.terminals) == request.jobs
        assert tuple(terminal.status for terminal in recovered.terminals) == (
            "completed",
            "completed",
        )
        assert recovery_pool.calls == []


def test_verified_probe_future_is_restored_without_resubmission(tmp_path: Path) -> None:
    with _runtime(tmp_path) as runtime:
        _, request = _ready_proposal_request(runtime)
        _install_unknown_receipt(runtime, request)
        pool = _RejectingPool()
        trial = _trial(runtime, pool, worker=lambda task: _raw_result(task),
                       result_factory=lambda job, raw: _proposal_result(runtime, job, raw))
        future = Future()
        calls = []

        def probe(job, task):
            calls.append(job.identity)
            return future

        trial.probe = probe
        trial.recover(request)
        trial.recover(request)
        assert tuple(trial.executor.one_pending) == (future,)
        assert calls == [request.jobs[0].identity]
        future.set_result(_raw_result(trial.task_record(request.jobs[0])))
        result = trial.perform(request)
        assert result.terminals[0].status == "completed"
        assert pool.calls == []


def test_issued_lazy_proposal_without_terminal_proof_cannot_be_reissued(tmp_path):
    with _runtime(tmp_path) as runtime:
        _, request = _ready_proposal_request(runtime)
        future = Future()
        submitted = []

        class LazyPool:
            def submit(self, worker, task):
                submitted.append(task)
                return future

        trial = _trial(runtime, LazyPool(), worker=lambda task: task,
                       result_factory=lambda job, raw: _proposal_result(runtime, job, raw))
        assert trial.perform(replace(request, wait="none")) == BranchEvent(())
        assert not future.done()
        assert _receipt(runtime, request.jobs[0].identity).status == "issued"
        rejecting = _RejectingPool()
        recovered = _trial(runtime, rejecting, worker=lambda task: task,
                           result_factory=lambda job, raw: _proposal_result(runtime, job, raw))
        with pytest.raises(OutcomeUnknown):
            recovered.recover(request)
        assert len(submitted) == 1
        assert rejecting.calls == []


def test_mixed_ready_group_preserves_success_before_peer_failure(
    tmp_path: Path,
) -> None:
    with _runtime(tmp_path) as runtime:
        _, request = _ready_proposal_request(
            runtime,
            proposal_batch_size=2,
            prior_proposals=2,
        )
        successful, failed = request.jobs
        assert successful.identity == "propose/3"
        assert failed.identity == "propose/4"

        def mixed_worker(task):
            if task["id"] == failed.id:
                raise RuntimeError("peer CPU transport failed")
            return _raw_result(task)

        first_pool = _CompletedPool(mixed_worker)
        first_trial = _trial(
            runtime,
            first_pool,
            worker=mixed_worker,
            result_factory=lambda job, raw: _proposal_result(runtime, job, raw),
            identity_name="mixed-ready-group",
        )

        with pytest.raises(RuntimeError, match="peer CPU transport failed"):
            first_trial.perform(request)

        success_receipt = _receipt(runtime, successful.identity)
        failure_receipt = _receipt(runtime, failed.identity)
        cached_raw = runtime.data.get(success_receipt.result, JSON)
        assert success_receipt.status == "completed"
        assert cached_raw["id"] == successful.id
        assert failure_receipt.status == "failed"
        assert failure_receipt.result is None
        assert len(first_pool.calls) == 2

        # Reconcile the known successful proposal without touching the known
        # terminal peer failure. This is the admitted proposal subset, not a new
        # policy decision or a replay of the complete ready group.
        successful_request = replace(
            request,
            state=replace(request.state, batch=(successful,)),
            jobs=(successful,),
        )
        recovery_pool = _RejectingPool()
        recovered_trial = _trial(
            runtime,
            recovery_pool,
            worker=mixed_worker,
            result_factory=lambda job, raw: _proposal_result(runtime, job, raw),
            identity_name="mixed-ready-group-recovery",
        )

        recovered_trial.recover(successful_request)
        event = recovered_trial.perform(successful_request)

        assert event.terminals[0].job == successful
        assert event.terminals[0].status == "completed"
        assert event.terminals[0].value.value == runtime.data.put(cached_raw, JSON)
        assert recovery_pool.calls == []


def test_wrapped_request_outcome_unknown_is_unknown_and_is_not_retried(
    tmp_path: Path,
) -> None:
    with _runtime(tmp_path) as runtime:
        _, request = _ready_proposal_request(runtime)
        retry_calls: list[str] = []

        def fail(_task):
            inner = RequestOutcomeUnknown("native owner lost the request")
            wrapped = RuntimeError("outer transport failure")
            wrapped.__cause__ = inner
            raise wrapped

        pool = _CompletedPool(fail)
        trial = _trial(
            runtime,
            pool,
            worker=fail,
            result_factory=lambda job, raw: _proposal_result(runtime, job, raw),
            retry=lambda job, task, receipt: retry_calls.append(job.identity),
        )

        with pytest.raises(RuntimeError, match="outer transport"):
            trial.perform(request)

        receipt = _receipt(runtime, request.jobs[0].identity)
        assert receipt.status == "unknown"
        assert receipt.error == "RuntimeError('outer transport failure')"
        assert len(pool.calls) == 1
        assert retry_calls == []


@pytest.mark.parametrize(
    ("error_type", "chain"),
    [
        (DaemonDisconnected, "cause"),
        (DaemonChildExited, "context"),
        (EOFError, "cause"),
    ],
)
def test_daemon_disconnect_or_child_exit_chain_is_unknown_without_resend(
    tmp_path: Path,
    error_type,
    chain: str,
) -> None:
    with _runtime(tmp_path) as runtime:
        _, request = _ready_proposal_request(runtime)
        job = request.jobs[0]

        def fail(_task):
            inner = error_type("connection closed after request send")
            if chain == "cause":
                raise RuntimeError("outer daemon failure") from inner
            try:
                raise inner
            except error_type:
                raise RuntimeError("outer daemon failure")

        trial = _trial(
            runtime,
            _CompletedPool(fail),
            worker=fail,
            result_factory=lambda current, raw: _proposal_result(
                runtime, current, raw
            ),
        )
        with pytest.raises(RuntimeError, match="outer daemon failure"):
            trial.perform(request)

        receipt = _receipt(runtime, job.identity)
        task = runtime.data.get(receipt.task, JSON)
        assert receipt.status == "unknown"
        assert task["charge_token"] == f"charge-{job.identity}-1"

        recovery_pool = _RejectingPool()
        recovered = _trial(
            runtime,
            recovery_pool,
            worker=fail,
            result_factory=lambda current, raw: _proposal_result(
                runtime, current, raw
            ),
            retry=lambda _job, task, _receipt: {
                **task,
                "attempt": 2,
                "charge_token": "must-not-be-charged",
            },
            identity_name="daemon-unknown-recovery",
        )
        with pytest.raises(OutcomeUnknown):
            recovered.recover(request)
        assert recovery_pool.calls == []


def test_known_daemon_request_failure_remains_terminal_for_linked_retry(
    tmp_path: Path,
) -> None:
    with _runtime(tmp_path) as runtime:
        _, request = _ready_proposal_request(runtime)
        job = request.jobs[0]

        def fail(_task):
            raise DaemonRequestFailed(
                "adapter rejected request",
                remote_type="ValueError",
            )

        first_pool = _CompletedPool(fail)
        first = _trial(
            runtime,
            first_pool,
            worker=fail,
            result_factory=lambda current, raw: _proposal_result(
                runtime, current, raw
            ),
            identity_name="daemon-terminal-first",
        )
        with pytest.raises(DaemonRequestFailed, match="adapter rejected"):
            first.perform(request)
        receipt = _receipt(runtime, job.identity)
        assert receipt.status == "failed"

        retries = []

        def retry(_job, task, _receipt):
            retries.append(task["attempt"])
            return {
                **task,
                "attempt": 2,
                "charge_token": "known-terminal-attempt-2",
            }

        recovery_pool = _CompletedPool(lambda task: _raw_result(task))
        recovered = _trial(
            runtime,
            recovery_pool,
            worker=lambda task: _raw_result(task),
            result_factory=lambda current, raw: _proposal_result(
                runtime, current, raw
            ),
            retry=retry,
            identity_name="daemon-terminal-recovery",
        )
        recovered.recover(request)
        event = recovered.perform(request)
        assert retries == [1]
        assert event.terminals[0].status == "completed"
        assert len(recovery_pool.calls) == 1


def test_unknown_receipt_cannot_be_relabelled_as_a_linked_retry(
    tmp_path: Path,
) -> None:
    with _runtime(tmp_path) as runtime:
        _, request = _ready_proposal_request(runtime)
        job = request.jobs[0]
        _install_unknown_receipt(runtime, request)
        old = _receipt(runtime, job.identity)
        old_task = runtime.data.get(old.task, JSON)
        retry_calls: list[int] = []

        def retry(_job, task, _receipt):
            retry_calls.append(task["attempt"])
            return {
                **task,
                "attempt": 2,
                "charge_token": "must-not-be-charged",
            }

        pool = _RejectingPool()
        trial = _trial(
            runtime,
            pool,
            worker=lambda task: _raw_result(task),
            result_factory=lambda current, raw: _proposal_result(
                runtime, current, raw
            ),
            retry=retry,
            identity_name="unknown-linked-retry",
        )

        with pytest.raises(OutcomeUnknown):
            trial.recover(request)

        assert retry_calls == []
        assert pool.calls == []
        assert _receipt(runtime, job.identity + "/attempt/1") is None
        assert _receipt(runtime, job.identity) == old
        assert old_task["attempt"] == 1


def test_known_terminal_receipt_can_use_a_linked_retry(
    tmp_path: Path,
) -> None:
    with _runtime(tmp_path) as runtime:
        _, request = _ready_proposal_request(runtime)
        job = request.jobs[0]
        first_pool = _CompletedPool(lambda _task: (_ for _ in ()).throw(
            DaemonRequestFailed(
                "adapter rejected request",
                remote_type="ValueError",
            )
        ))
        first = _trial(
            runtime,
            first_pool,
            worker=lambda task: _raw_result(task),
            result_factory=lambda current, raw: _proposal_result(
                runtime, current, raw
            ),
            identity_name="terminal-seed",
        )
        with pytest.raises(DaemonRequestFailed, match="adapter rejected"):
            first.perform(request)
        old = _receipt(runtime, job.identity)
        assert old.status == "failed"
        retry_calls: list[int] = []

        def retry(_job, task, _receipt):
            retry_calls.append(task["attempt"])
            return {
                **task,
                "attempt": 2,
                "charge_token": "charge-propose-1-2",
            }

        pool = _CompletedPool(lambda task: _raw_result(task))
        trial = _trial(
            runtime,
            pool,
            worker=lambda task: _raw_result(task),
            result_factory=lambda current, raw: _proposal_result(
                runtime, current, raw
            ),
            retry=retry,
            identity_name="linked-retry",
        )

        trial.recover(request)
        event = trial.perform(request)

        archived = _receipt(runtime, job.identity + "/attempt/1")
        current = _receipt(runtime, job.identity)
        current_task = runtime.data.get(current.task, JSON)
        assert retry_calls == [1]
        assert len(pool.calls) == 1
        assert archived == old
        assert archived.status == "failed"
        assert current.status == "completed"
        assert current_task["attempt"] == 2
        assert current_task["charge_token"] == "charge-propose-1-2"
        assert event.terminals == (
            BranchTerminal(
                job=job,
                status="completed",
                value=_proposal_result(
                    runtime,
                    job,
                    runtime.data.get(current.result, JSON),
                ),
            ),
        )


def test_completed_raw_output_survives_cpu_conversion_failure(
    tmp_path: Path,
) -> None:
    with _runtime(tmp_path) as runtime:
        _, request = _ready_proposal_request(runtime)
        job = request.jobs[0]
        first_pool = _CompletedPool(lambda task: _raw_result(task))

        def fail_cpu_scoring(_job, _raw):
            raise RuntimeError("CPU scoring failed after native completion")

        first_trial = _trial(
            runtime,
            first_pool,
            worker=lambda task: _raw_result(task),
            result_factory=fail_cpu_scoring,
            identity_name="cpu-failure",
        )

        with pytest.raises(RuntimeError, match="CPU scoring failed"):
            first_trial.perform(request)

        completed = _receipt(runtime, job.identity)
        raw = runtime.data.get(completed.result, JSON)
        assert completed.status == "completed"
        assert raw["native_payload"] == {"source": "fake-owner"}
        assert len(first_pool.calls) == 1

        scored_raw: list[dict] = []
        recovery_pool = _RejectingPool()

        def recover_scoring(current, value):
            scored_raw.append(value)
            return _proposal_result(runtime, current, value)

        recovered_trial = _trial(
            runtime,
            recovery_pool,
            worker=lambda task: _raw_result(task),
            result_factory=recover_scoring,
            identity_name="cpu-recovery",
        )

        recovered_trial.recover(request)
        event = recovered_trial.perform(request)

        assert event.terminals[0].status == "completed"
        assert scored_raw == [raw]
        assert recovery_pool.calls == []


def test_reservation_refusal_never_reaches_native_pool(tmp_path: Path) -> None:
    with _runtime(tmp_path) as runtime:
        _, request = _ready_proposal_request(runtime)
        job = request.jobs[0]
        pool = _RejectingPool()
        trial = _trial(
            runtime,
            pool,
            worker=lambda task: _raw_result(task),
            result_factory=lambda current, raw: _proposal_result(
                runtime, current, raw
            ),
            reserve=lambda _job, _task: None,
        )

        event = trial.perform(request)

        assert event.terminals == (BranchTerminal(job=job, status="refused"),)
        assert _receipt(runtime, job.identity).status == "refused"
        assert pool.calls == []


def test_request_hash_mismatch_is_unknown_and_blocks_automatic_retry(
    tmp_path: Path,
) -> None:
    with _runtime(tmp_path) as runtime:
        _, request = _ready_proposal_request(runtime)
        job = request.jobs[0]

        def mismatched(task):
            return _raw_result(task, _request_hash="another-native-request")

        pool = _CompletedPool(mismatched)
        trial = _trial(
            runtime,
            pool,
            worker=mismatched,
            result_factory=lambda current, raw: _proposal_result(
                runtime, current, raw
            ),
        )

        with pytest.raises(
            ContractConflict,
            match="another request or scientific context",
        ):
            trial.perform(request)

        receipt = _receipt(runtime, job.identity)
        assert receipt.status == "unknown"
        assert receipt.result is None
        assert trial.executor.observed == {}
        assert len(pool.calls) == 1


def test_candidate_identity_mismatch_is_unknown_and_blocks_automatic_retry(
    tmp_path: Path,
) -> None:
    with _runtime(tmp_path) as runtime:
        root = BranchMember(
            id=0,
            key="root",
            value=runtime.data.put({"sequence": "ROOT"}, JSON),
            score=0.0,
        )
        proposal = BranchProposal(
            id=1,
            parent_id=0,
            key="child-1",
            value=runtime.data.put({"sequence": "ACDE"}, JSON),
        )
        job = BranchJob(
            phase="evaluate",
            id=1,
            generation=1,
            seed=13,
            parent=root,
            proposal=proposal,
            slot=0,
            source=proposal.value,
        )
        request = BranchAction(
            state=BranchingState(
                generation=1,
                next_evaluation=2,
                parents=(root,),
                records=(root,),
                pending=(job,),
            ),
            jobs=(job,),
            wait="all",
        )

        def wrong_candidate(task):
            return _raw_result(task, sequence="WRONG")

        pool = _CompletedPool(wrong_candidate)
        trial = _trial(
            runtime,
            pool,
            worker=wrong_candidate,
            result_factory=lambda current, raw: _evaluation_result(
                runtime, current, raw
            ),
        )

        with pytest.raises(
            ContractConflict,
            match="another candidate assignment",
        ):
            trial.perform(request)

        receipt = _receipt(runtime, job.identity)
        assert receipt.status == "unknown"
        assert receipt.result is None
        assert trial.executor.observed == {}
        assert len(pool.calls) == 1


def test_native_owner_composes_with_atomic_control_run(tmp_path: Path) -> None:
    policy = _policy(max_evaluations=1)
    root_job = BranchJob(
        phase="evaluate",
        id=0,
        generation=0,
        seed=policy.evaluation_seed,
    )
    pool = _CompletedPool(
        lambda task: _raw_result(task, key="root", score=1.0)
    )

    with _runtime(tmp_path, "control") as runtime:
        trial = _trial(
            runtime,
            pool,
            worker=lambda task: _raw_result(task, key="root", score=1.0),
            result_factory=lambda job, raw: _evaluation_result(
                runtime, job, raw
            ),
        )
        result = ControlRun(
            policy=policy,
            trial=trial,
            initial_state=policy.start(root_job),
            runtime=runtime,
            state_codec=branch_codec(BranchingState),
            request_codec=branch_codec(BranchAction),
            result_codec=branch_codec(BranchEvent),
        ).run()

        assert isinstance(result, BranchingState)
        assert result.finished is True
        assert tuple(member.key for member in result.frontier) == ("root",)
        assert len(pool.calls) == 1
        assert pool.calls[0]["phase"] == "evaluate"
