"""Single-writer bridge between branching control and isolated native owners.

The callbacks own native task/result schemas and scientific validation. The
bridge owns durable per-job receipts and verified reconstruction of Futures;
it has no model, molecular task, score or candidate-layout branches.
"""
from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass, replace
from typing import Callable

from .branch_execution import BranchExecutor
from .branch_types import BranchAction, BranchEvent, BranchJob, BranchTerminal
from .runtime import ContractConflict, OutcomeUnknown, outcome_is_unknown
from .store import DataclassCodec, JSONCodec
from .types import Blob, key_for


@dataclass(frozen=True)
class NativeJobReceipt:
    identity: str
    task: Blob
    status: str
    result: Blob | None = None
    error: str | None = None


_RECEIPT = DataclassCodec(NativeJobReceipt, "astrolabe.native-job-receipt/v1")
_JSON = JSONCodec()


@dataclass(frozen=True)
class _Refused:
    identity: str


class NativeBranchTrial:
    """Bounded native issue/collection with one authoritative control journal."""

    requirements = ()

    def __init__(self, pool, runtime, *, identity, worker, task_factory,
                 result_factory, probe, reserve=None, retry=None, report=None,
                 terminal_proposal_errors=True, charge_identity=None,
                 before_wait=None, ordered_results=False):
        self.runtime, self.data = runtime, runtime.data
        self.identity = identity
        self.worker, self.task_factory = worker, task_factory
        self.result_factory, self.probe = result_factory, probe
        self.reserve, self.retry, self.report = reserve, retry, report
        self.terminal_proposal_errors = terminal_proposal_errors
        self.charge_identity = charge_identity or (lambda task: task.get("charge_token"))
        self.data.register(_RECEIPT)
        self.data.register(_JSON)
        self._restored = set()
        self._state = self._request = None
        self.executor = BranchExecutor(pool, submit=self._submit,
            resolve=self._resolve, on_error=self._failure, on_observed=self._observe,
            before_wait=before_wait, ordered_results=ordered_results)

    def _name(self, job):
        return "native-jobs/" + job.identity

    def _receipt(self, job):
        return self.runtime.load(self._name(job), _RECEIPT)

    def task_record(self, job):
        """Read the immutable admitted native task for a compatibility view."""
        receipt = self._receipt(job)
        return None if receipt is None else self.data.get(receipt.task, _JSON)

    def _save(self, job, task, status, *, result=None, error=None):
        receipt = NativeJobReceipt(job.identity, self.data.put(task, _JSON), status,
            None if result is None else self.data.put(result, _JSON), error)
        self.runtime.save(self._name(job), receipt, _RECEIPT)
        return receipt

    def _submit_task(self, pool, job, task):
        if self.reserve is not None:
            task = self.reserve(job, dict(task))
        if task is None:
            original = self.task_factory(job)
            self._save(job, original, "refused")
            future = Future()
            future.set_result(_Refused(job.identity))
            return future
        task = dict(task)
        task.pop("_request_hash", None)
        task["_request_hash"] = key_for((self.identity, job, task)).digest
        self._save(job, task, "reserved")
        # Native reservation and this issued identity precede pool submission.
        # A crash in this window remains uncertain; it is never assumed free.
        self._save(job, task, "issued")
        if self.report is not None and self._state is not None:
            self.report(self._state, request=self._request)
        return pool.submit(self.worker, task)

    def _submit(self, pool, job):
        if self._receipt(job) is not None:
            raise ContractConflict("A native job requires reconciliation before another submission")
        return self._submit_task(pool, job, self.task_factory(job))

    def _observe(self, job, raw):
        if isinstance(raw, _Refused):
            if raw.identity != job.identity:
                raise ValueError("Native refusal belongs to another job")
            return
        if not isinstance(raw, dict):
            raise TypeError("Native owner must return its declared JSON record")
        receipt = self._receipt(job)
        if receipt is None:
            raise ContractConflict("Native completion has no admitted request")
        task = self.data.get(receipt.task, _JSON)
        if raw.get("id") != task.get("id"):
            raise ValueError("Native result identity mismatch")
        for field in ("kind", "generation", "parent_id", "seed"):
            if raw.get(field) != task.get(field):
                raise ContractConflict(f"Native result changed its admitted {field}")
        if raw.get("attempt", 1) != task.get("attempt", 1):
            raise ContractConflict("Native result changed its admitted attempt")
        if job.phase == "evaluate" and raw.get("sequence") != task.get("sequence"):
            raise ContractConflict("Native evaluation returned another candidate assignment")
        expected = task.get("_request_hash")
        if expected is not None and raw.get("_request_hash") != expected:
            raise ContractConflict("Native result belongs to another request or scientific context")
        # Preserve the native result before CPU conversion/measurement. A failed
        # scoring step can reuse verified native output without another model call.
        self._save(job, task, "completed", result=raw)

    def _resolve(self, job, raw):
        if isinstance(raw, _Refused):
            return BranchTerminal(job, "refused")
        member = self.result_factory(job, raw)
        return BranchTerminal(job, "completed", member)

    def _failure(self, job, error):
        receipt = self._receipt(job)
        if receipt is None:
            return
        task = self.data.get(receipt.task, _JSON)
        uncertain = outcome_is_unknown(error)
        terminal = job.phase == "propose" and self.terminal_proposal_errors and not uncertain
        self._save(job, task, "failed" if terminal else "unknown", error=repr(error))

    def _restore(self, job):
        if job.identity in self._restored:
            return
        receipt = self._receipt(job)
        if receipt is None:
            # The state admission was durable but no per-job issue was made.
            future = self._submit_task(self.executor.pool, job, self.task_factory(job))
        else:
            task = self.data.get(receipt.task, _JSON)
            raw = None if receipt.result is None else self.data.get(receipt.result, _JSON)
            if receipt.status == "refused":
                future = Future()
                future.set_result(_Refused(job.identity))
            else:
                if raw is None:
                    raw = self.probe(job, task)
                if raw is not None:
                    if isinstance(raw, Future):
                        future = raw
                    else:
                        future = Future()
                        future.set_result(raw)
                elif receipt.status == "reserved":
                    future = self._submit_task(self.executor.pool, job, task)
                elif receipt.status in ("issued", "unknown"):
                    # The native request may have been delivered. Only owner
                    # terminal proof may resolve it; a retry callback must not
                    # create another charge or submission.
                    raise OutcomeUnknown(job.identity)
                else:
                    retry_task = (None if self.retry is None else
                                  self.retry(job, dict(task), receipt))
                    if retry_task is None:
                        raise OutcomeUnknown(job.identity)
                    previous_attempt = int(task.get("attempt", 1))
                    next_attempt = retry_task.get("attempt")
                    if type(next_attempt) is not int or next_attempt <= previous_attempt:
                        raise ContractConflict("A native retry requires a new linked attempt")
                    old_token, next_token = self.charge_identity(task), self.charge_identity(retry_task)
                    if old_token is not None and old_token == next_token:
                        raise ContractConflict("A native retry cannot reuse an issued charge token")
                    self.runtime.save(self._name(job) + f"/attempt/{previous_attempt}", receipt, _RECEIPT)
                    future = self._submit_task(self.executor.pool, job, retry_task)
        self.executor.restore(job, future)
        self._restored.add(job.identity)

    def recover(self, request):
        self._state, self._request = request.state, request
        for job in self._outstanding(request):
            self._restore(job)
        return None

    def _outstanding(self, request):
        delivered = {terminal.job.identity for terminal in request.state.batch_results}
        return tuple({job.identity: job for job in request.state.pending + request.state.batch
                      if job.identity not in delivered}.values())

    def perform(self, request):
        self._state, self._request = request.state, request
        known = {job.identity for job in self.executor.one_pending.values()}
        new = {job.identity for job in request.jobs}
        for job in self._outstanding(request):
            if job.identity not in known and job.identity not in new:
                self._restore(job)
        jobs = tuple(job for job in request.jobs if job.identity not in self._restored)
        result = self.executor.perform(replace(request, jobs=jobs))
        for terminal in result.terminals:
            self._restored.discard(terminal.job.identity)
        return result

    def checkpoint(self, state, *, request):
        self._state, self._request = state, request
        if self.report is not None:
            self.report(state, request=request)
