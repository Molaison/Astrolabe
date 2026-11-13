"""Isolated, resident prediction and CPU Score for generic refinement.

Legacy mutation and member commits remain on the single scientific writer.
NativeBranchTrial records every issue
before dispatch and refuses to replay a request without terminal evidence.
"""
from __future__ import annotations

from concurrent.futures import (ALL_COMPLETED, FIRST_COMPLETED, Future,
                                ProcessPoolExecutor, ThreadPoolExecutor, wait)
from collections import deque
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import copy_context
from dataclasses import asdict, replace
import hashlib
import inspect
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import threading

from astrolabe.profiling import event as profile_event, span

from .branch_owner import NativeBranchTrial
from .branch_types import BranchAction, BranchEvent, BranchProposal, branch_codec
from .native_refinement import NativeRefinementResult
from .native_refinement_runtime import (
    DurableMeasurementRequest,
    DurableNativeRefinementOwner,
    RawPredictionReceipt,
    _RAW_PREDICTION_CODEC,
    _SCORE_RECEIPT_CODEC,
    _source_identity,
)
from .runtime import ContractConflict
from .store import design_codec
from .types import Blob, History, Key, key_for


def _blob(record):
    return Blob(Key(record["identity"]["digest"]), record["codec"])


class DeferredEvaluator:
    """Parent-side marker; prediction is owned by isolated workers."""

    def __init__(self, alias):
        self.alias = alias

    def evaluate(self, *args, **kwargs):
        raise RuntimeError("Isolated refinement prediction must run in its worker")

    def shutdown(self):
        pass


@contextmanager
def _shared_models(owner, enabled):
    if not enabled:
        yield
        return
    from astrolabe.model_execution import NativeModelClient
    from astrolabe.evaluators.dummy import DummyEvaluator

    wrapped = owner.models.get().evaluator
    evaluator = getattr(wrapped, "value", wrapped)
    seen, clients, lifecycle = set(), {}, {}

    def collect(value, depth=0):
        if id(value) in seen or depth > 4:
            return
        seen.add(id(value))
        if isinstance(value, NativeModelClient):
            daemon = value.owner
            socket_path = getattr(daemon, "socket_path", None)
            if socket_path is None:
                raise ValueError("Shared evaluator models require daemon execution")
            identity = daemon.identity
            clients[identity] = {
                "backend": daemon.spec.backend, "socket": str(socket_path),
            }
            lifecycle[identity] = value.metadata.get("runtime_lifecycle", {})
            return
        if isinstance(value, Mapping):
            children = value.values()
        elif isinstance(value, (list, tuple)):
            children = value
        elif value is evaluator:
            children = vars(value).values()
        else:
            prepare = getattr(value, "prepare_shared_model", None)
            prepared = prepare() if callable(prepare) else None
            client = prepared if isinstance(prepared, NativeModelClient) else getattr(value, "_client", None)
            children = () if client is None else (client,)
        for child in children:
            collect(child, depth + 1)

    collect(evaluator)
    if not clients and not isinstance(evaluator, DummyEvaluator):
        raise ValueError(
            "Selected evaluator has no shareable native model client; use a "
            "daemon adapter or explicit replicated workers"
        )
    root = owner.worker_output
    root.mkdir(parents=True, exist_ok=True)
    registry = root / "resident_models.json"
    # Endpoint paths are per-process. A resumed run publishes its new owner;
    # paid job receipts remain bound to immutable scientific request identities.
    temporary = registry.with_suffix(".tmp")
    temporary.write_text(json.dumps({
        "schema": "astrolabe.resident-model-registry/v1",
        "broker_socket": os.environ.get("ASTROLABE_RESOURCE_SOCKET"),
        "models": clients,
    }, sort_keys=True), encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(registry)
    previous = os.environ.get("ASTROLABE_RESIDENT_MODEL_REGISTRY")
    os.environ["ASTROLABE_RESIDENT_MODEL_REGISTRY"] = str(registry)
    owner.spec.event(
        "resident_models_shared", model_count=len(clients),
        models=list(clients), lifecycle=lifecycle,
    )
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("ASTROLABE_RESIDENT_MODEL_REGISTRY", None)
        else:
            os.environ["ASTROLABE_RESIDENT_MODEL_REGISTRY"] = previous


class _PredictionPool:
    def __init__(self, owner, pool, *, asynchronous=False):
        self.owner, self.pool = owner, pool
        self.asynchronous = asynchronous
        self._proposals = deque()
        self._live = None
        self._action = None
        self._transport = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mutator-wait")

    def submit(self, worker, task):
        if task["kind"] == "evaluate":
            return self.pool.submit(worker, task)
        future = Future()
        future.set_running_or_notify_cancel()
        runtime = self.owner.runtime
        self._proposals.append((future, task, runtime._work, runtime.history))
        return future

    def advance(self, action):
        self._action = action
        if action.wait == "none":
            return
        limit = len(self._proposals) + int(self._live is not None) if action.wait == "all" else 1
        for _ in range(limit):
            if not self._advance_one():
                break

    def _advance_one(self):
        from .continuation import TransportWait
        from .native_refinement_runtime import DurableProposalRequest
        runtime = self.owner.runtime
        if self._live is None:
            if not self._proposals:
                return False
            future, task, work, history = self._proposals.popleft()
            job = self.owner.data.get(_blob(task["job"]), self.owner.job_codec)
            self._live = dict(future=future, task=task, work=work, history=history,
                              job=job, iterator=None, waiting=None, rng=None)
        live = self._live
        if live["waiting"] is not None and not live["waiting"].done():
            return False
        previous = runtime._work, runtime.history
        try:
            if live["work"] is None or runtime._work is None:
                raise RuntimeError("Queued proposals require a bound Runtime work entry")
            runtime._work, runtime.history = live["work"], live["history"]
            if not self.asynchronous:
                proposal = self.owner._terminal(live["job"]).value
            else:
                if live["iterator"] is None:
                    job = live["job"]
                    live["iterator"] = runtime.call_continuation(
                        f"native-refinement/propose/{job.id}/proposal",
                        self.owner.proposal_operation,
                        DurableProposalRequest(job, self.owner.protocol, self.owner.spec.output.as_posix()),
                        work_id=self.owner._job_work_id(job),
                    )
                value, error = None, None
                if live["waiting"] is not None:
                    # Intermediate continuation RNG is private to this proposal;
                    # Score/control publication must never replace it.
                    runtime._restore_rng(live["rng"])
                    try:
                        value = live["waiting"].result()
                    except BaseException as failure:
                        error = failure
                try:
                    pending = (live["iterator"].throw(error) if error is not None
                               else live["iterator"].send(value))
                except StopIteration as done:
                    proposal = done.value
                else:
                    if not isinstance(pending, TransportWait):
                        raise TypeError("Proposal yielded work other than frozen transport")
                    if self.owner.spec.context._active:
                        raise RuntimeError("Remote wait cannot retain LegacyContext activation")
                    live["rng"] = runtime._capture_rng()
                    context = copy_context()
                    live["waiting"] = self._transport.submit(context.run, pending.transport)
                    return False
            reference = self.owner.data.put(proposal, self.owner.proposal_codec)
            rng = runtime._capture_rng()
            self.owner.data.persist((reference, rng))
            live["future"].set_result({
                **live["task"], "proposal": asdict(reference), "after_rng": asdict(rng),
            })
            self._live = None
            return True
        except BaseException as error:
            # Closing a suspended call records unknown through the exact same
            # Runtime error path; never prepare/reenter the operation again.
            if live["iterator"] is not None:
                try:
                    live["iterator"].close()
                except BaseException as close_error:
                    error.add_note(f"Continuation close also failed: {close_error!r}")
            live["future"].set_exception(error)
            self._live = None
            return True
        finally:
            runtime._work, runtime.history = previous

    def progress(self, action):
        if isinstance(self.pool, _ScoreHandoff):
            return self.pool.progress(action)
        return BranchEvent(())

    def wait(self, futures, *, return_when):
        while True:
            if return_when == ALL_COMPLETED:
                if all(future.done() for future in futures):
                    return set(futures), set()
            elif any(future.done() for future in futures):
                return {f for f in futures if f.done()}, {f for f in futures if not f.done()}
            extra = []
            if isinstance(self.pool, _ScoreHandoff) and self.pool.capacity:
                # Snapshot signal before checking progress to avoid a lost wake.
                extra.append(self.pool.wakeup())
                progress = self.progress(self._action)
                if (progress.prediction_done or progress.score_admitted or progress.score_done
                        or progress.credit_epoch is not None):
                    return set(), set(futures)
            if self._live is not None and self._live["waiting"] is not None:
                extra.append(self._live["waiting"])
            ready, _ = wait([*futures, *extra], return_when=FIRST_COMPLETED)
            if self._live is not None and self._live["waiting"] in ready:
                self.advance(self._action)
            if self._live is None and self._proposals and return_when == ALL_COMPLETED:
                self.advance(self._action)

    def close(self):
        # Own the transport lifetime even when control failed or was cancelled.
        self._transport.shutdown(wait=True)
        if self._live is not None:
            live = self._live
            runtime = self.owner.runtime
            previous = runtime._work, runtime.history
            try:
                runtime._work, runtime.history = live["work"], live["history"]
                if live["rng"] is not None:
                    runtime._restore_rng(live["rng"])
                if live["iterator"] is not None:
                    live["iterator"].close()
            finally:
                runtime._work, runtime.history = previous
                self._live = None


class _ScoreHandoff:
    """Bounded producer/raw-holding lane and independent waiting+running Score.

    Callbacks never wait for credits: a full Score lane leaves verified raw in
    the already reserved producer slot. Only immutable receipts cross threads.
    """

    def __init__(self, prediction_pool, score_pool, score_worker, *,
                 prediction_width=None, capacity=0, verify=None):
        self.prediction_pool, self.score_pool = prediction_pool, score_pool
        self.score_worker = score_worker
        self.prediction_width, self.capacity = prediction_width, capacity
        self.verify = verify
        self._lock = threading.RLock()
        self._entries = {}
        self._raw = deque()
        self._scores = 0
        self._wake = Future()
        self._recovery_epoch = None
        self.finals, self.scores = set(), set()

    def _changed(self, transition, task):
        profile_event("pipeline.credit", transition=transition, evaluation_id=task["id"],
                      predicting=sum(e["stage"] == "predicting" for e in self._entries.values()),
                      raw_ready_held=len(self._raw), score_pending_running=self._scores,
                      producer_capacity=self.prediction_width, score_capacity=self.capacity)
        wake, self._wake = self._wake, Future()
        wake.set_result(None)

    def wakeup(self):
        with self._lock:
            return self._wake

    def begin_recovery(self, state):
        with self._lock:
            if self.capacity and self._recovery_epoch is None:
                self._recovery_epoch = state.credit_epoch + 1

    def progress(self, action):
        if not self.capacity:
            return BranchEvent(())
        with self._lock:
            ids = {job.id for job in action.state.pending}
            # Completed observations carry no retained model result/future.
            for identity in tuple(self._entries):
                if identity not in ids and self._entries[identity]["stage"] == "done":
                    del self._entries[identity]
            reset = (self._recovery_epoch if self._recovery_epoch is not None
                     and self._recovery_epoch > action.state.credit_epoch else None)
            def delta(name, field):
                return tuple(sorted(identity for identity in ids
                                    if identity in self._entries and self._entries[identity][field]
                                    and (reset is not None or identity not in getattr(action.state, name))))
            return BranchEvent((), delta("prediction_done", "raw_verified"),
                               delta("score_admitted", "score_admitted"),
                               delta("score_done", "score_done"), reset)

    def _entry(self, task, stage):
        if task["id"] in self._entries:
            raise ContractConflict("Evaluation already owns pipeline credits")
        if self.capacity:
            producers = sum(e["stage"] in {"predicting", "raw"} for e in self._entries.values())
            if producers >= self.prediction_width:
                raise RuntimeError("Prediction/raw-holding producer capacity exceeded")
        final = Future()
        final.set_running_or_notify_cancel()
        entry = dict(task=task, final=final, stage=stage,
                     raw_verified=False, score_admitted=False, score_done=False)
        self._entries[task["id"]] = entry
        self.finals.add(final)
        self._changed("producer_reserved", task)
        return entry

    def submit_score(self, task):
        # Recovery can resume Score only from freshly verified raw evidence.
        if self.verify is not None:
            self.verify(task, None)
        with self._lock:
            entry = self._entry(task, "raw")
            entry["raw_verified"] = True
            self._raw.append(entry)
            self._changed("raw_verified_recovery", task)
            self._pump()
            return entry["final"]

    def submit(self, worker, task):
        with self._lock:
            entry = self._entry(task, "predicting")
        try:
            prediction = self.prediction_pool.submit(worker, task)
        except BaseException as error:
            with self._lock:
                self._failed(entry, error)
            return entry["final"]

        def predicted(done):
            try:
                raw = done.result()
                if self.verify is not None:
                    self.verify(task, raw)
                with self._lock:
                    entry.update(stage="raw", raw_verified=True)
                    self._raw.append(entry)
                    self._changed("prediction_released_raw_verified", task)
                    self._pump()
            except BaseException as error:
                with self._lock:
                    self._failed(entry, error)
        context = copy_context()
        prediction.add_done_callback(lambda done: context.run(predicted, done))
        return entry["final"]

    def _failed(self, entry, error):
        entry["stage"] = "done"
        self._changed("failure_release", entry["task"])
        entry["final"].set_exception(error)
        self.finals.discard(entry["final"])
        if not self.capacity:
            self._entries.pop(entry["task"]["id"], None)

    def _pump(self):
        # The manager callback must never block on a credit that only another
        # callback can release. Submission is nonblocking; raw remains bounded.
        while self._raw and (not self.capacity or self._scores < self.capacity):
            entry = self._raw.popleft()
            entry.update(stage="scoring", score_admitted=True)
            self._scores += 1
            self._changed("raw_to_score_admitted", entry["task"])
            try:
                future = self.score_pool.submit(self.score_worker, entry["task"])
            except BaseException as error:
                self._scores -= 1
                self._failed(entry, error)
                continue
            self.scores.add(future)

            def scored(done, entry=entry):
                try:
                    value, error = done.result(), None
                except BaseException as failure:
                    value, error = None, failure
                with self._lock:
                    self._scores -= 1
                    self.scores.discard(done)
                    entry.update(stage="done", score_done=error is None)
                    self._changed("score_released", entry["task"])
                    if error is None:
                        entry["final"].set_result(value)
                    else:
                        entry["final"].set_exception(error)
                    self.finals.discard(entry["final"])
                    if not self.capacity:
                        self._entries.pop(entry["task"]["id"], None)
                    self._pump()
            context = copy_context()
            future.add_done_callback(lambda done, context=context, scored=scored: context.run(scored, done))

    def drain(self):
        while True:
            with self._lock:
                pending = tuple(self.finals | self.scores)
            if not pending:
                return
            wait(pending)


class ParallelRefinementOwner(DurableNativeRefinementOwner):
    """Single-writer proposal/measurement with bounded resident predictors."""

    def __init__(self, spec, runtime, *, protocol, pool, worker, worker_output,
                 loss_order=None, score_context=None, async_mutator=False):
        super().__init__(spec, runtime, protocol=protocol)
        from .branch_types import BranchJob
        from .refinement_worker import execute, initialize
        from . import continuation

        self.job_codec = branch_codec(BranchJob)
        self.proposal_codec = branch_codec(BranchProposal)
        self.data.register(self.job_codec)
        self.data.register(self.proposal_codec)
        self.worker_output = Path(worker_output)
        self.pool, self.worker = pool, worker
        self.loss_order = loss_order
        self.score_context = score_context
        from .configured_refinement import measure_configured_scores
        from astrolabe.losses.base import LossEngine, loss_registry
        self.identity = key_for((
            "astrolabe.parallel-refinement/v1", self.identity,
            inspect.getsource(sys.modules[__name__]),
            inspect.getsource(sys.modules[initialize.__module__]),
            inspect.getsource(NativeBranchTrial),
            inspect.getsource(continuation),
            inspect.getsource(type(runtime)),
            inspect.getsource(sys.modules[measure_configured_scores.__module__]),
            inspect.getsource(sys.modules[LossEngine.__module__]),
            tuple((name, _source_identity(loss_registry.get_function(name)))
                  for name in sorted(loss_registry.valid_losses)),
            loss_order,
        ))
        prediction_pool = _PredictionPool(self, pool, asynchronous=async_mutator)
        self.proposal_pool = prediction_pool
        self.trial = NativeBranchTrial(
            prediction_pool, runtime, identity=self.identity,
            worker=worker, task_factory=self._task, result_factory=self._result,
            probe=self._probe, terminal_proposal_errors=False,
            report=self._report,
            before_wait=prediction_pool.advance, ordered_results=True,
            waiter=prediction_pool.wait, progress=prediction_pool.progress,
        )

    def _task(self, job):
        reference = self.data.put(job, self.job_codec)
        self.data.persist((reference,))
        task = dict(
            id=job.id, kind=job.phase, generation=job.generation,
            parent_id=None if job.parent is None else job.parent.id,
            seed=job.seed, job=asdict(reference),
        )
        if job.phase == "evaluate":
            proposal = self._proposal_record(job)
            candidate = self.data.put(proposal.candidate, design_codec)
            self.data.persist((candidate,))
            task.update(
                candidate_blob=asdict(candidate),
                candidate_identity=proposal.candidate.identity.digest,
                system_identity=proposal.candidate.system.digest,
                sequence=proposal.label, step=proposal.proposal_step,
                scoring_identity=self.protocol.digest,
                loss_order=None if self.loss_order is None else list(self.loss_order),
                score_context=(None if self.score_context is None else
                               asdict(self.score_context(proposal.proposal_step))),
                parent_id=None if proposal.parent_id < 0 else proposal.parent_id,
            )
            self.spec.event(
                "evaluation_started", evaluation_id=job.id,
                generation=proposal.generation,
                parent_id=None if proposal.parent_id < 0 else proposal.parent_id,
                proposal_index=proposal.proposal_index,
                proposal_step=proposal.proposal_step, sequence=proposal.label,
                seed=job.seed,
            )
        return task

    def _probe(self, job, task):
        if job.phase == "propose":
            # Runtime.call below can recover a completed proposal and refuses
            # issued/unknown work. The native bridge itself never resubmits it.
            rounds = dict(self._rounds_for(job))
            name, _component = next(iter(rounds.items()))
            attempts = [item for item in self.runtime.attempts()
                        if item.key.named_round == name]
            if not attempts or attempts[-1].status != "completed":
                return None
            attempt = attempts[-1]
            proposal = self.data.get(attempt.result, self.proposal_codec)
            if not isinstance(proposal, BranchProposal):
                raise TypeError("Recovered proposal has another type")
            receipt = attempt.rng
            reference = self.data.put(proposal, self.proposal_codec)
            self.data.persist((reference, receipt.after))
            return {
                **task, "proposal": asdict(reference),
                "after_rng": asdict(receipt.after),
            }
        path = self.worker_output / "receipts" / (task["_request_hash"] + ".score.json")
        if not path.is_file():
            raw_path = path.with_name(task["_request_hash"] + ".json")
            if not raw_path.is_file():
                return None
            # execute verifies immutable raw evidence before doing Score only.
            # It never falls through to inference when raw evidence is damaged.
            if isinstance(self.pool, _ScoreHandoff):
                return self.pool.submit_score(task)
            return self.pool.submit(self.worker, task)
        raw = json.loads(path.read_text(encoding="utf-8"))
        for name, value in task.items():
            if raw.get(name) != value:
                raise ContractConflict(f"Prediction receipt changed {name}")
        return raw

    def _result(self, job, raw):
        if job.phase == "propose":
            self.runtime._restore_rng(_blob(raw["after_rng"]))
            return self.data.get(_blob(raw["proposal"]), self.proposal_codec)
        task = self.trial.task_record(job)
        if task is None or any(raw.get(name) != value for name, value in task.items()):
            raise ContractConflict("Score result differs from its admitted task")
        proposal = self._proposal_record(job)
        scores = _blob(raw["raw_scores"])
        # Decode and validate before a paid prediction is treated as complete.
        from .native_scores import restore_scores
        restore_scores(scores, self.data, self._score_destination(job, "worker-check"))
        prediction = RawPredictionReceipt(
            job, self._proposal_blob(job), proposal.candidate, scores,
        )
        reference = self.data.put(prediction, _RAW_PREDICTION_CODEC)
        self.data.persist((reference,))
        self.spec.event(
            "prediction_received", evaluation_id=job.id,
            worker_pid=raw["worker_pid"],
            worker_constructor_count=raw["constructor_count"],
            prediction_wall_s=raw["prediction_wall_s"],
        )
        raw_dir = Path(raw["job_raw_dir"]).resolve()
        worker_root = self.worker_output.resolve()
        if not raw_dir.is_relative_to(worker_root):
            raise ContractConflict("Prediction raw directory escapes its worker output")
        for item in raw["sidecars"]:
            path = Path(item["path"]).resolve()
            if not path.is_relative_to(raw_dir):
                raise ContractConflict("Prediction sidecar escapes its request output")
            content = path.read_bytes()
            if (len(content) != item["size_bytes"]
                    or hashlib.sha256(content).hexdigest() != item["sha256"]):
                raise ContractConflict("Prediction sidecar changed after completion")
        from .refinement_worker import _sidecars
        if _sidecars(raw_dir) != raw["sidecars"]:
            raise ContractConflict("Prediction sidecar set changed after completion")
        score_reference = _blob(raw["score_receipt"])
        score = self.data.get(score_reference, _SCORE_RECEIPT_CODEC)
        if (score.request_identity != Key(raw["_request_hash"])
                or score.scoring_identity != self.protocol
                or score.raw_scores != scores):
            raise ContractConflict("Score receipt differs from its admitted request")
        score_raw = Path(raw["score_raw_dir"]).resolve()
        if not score_raw.is_relative_to(worker_root):
            raise ContractConflict("Score directory escapes its worker output")
        if _sidecars(score_raw) != raw["score_sidecars"]:
            raise ContractConflict("Score sidecars changed after completion")
        restore_scores(score.measured_scores, self.data, self._score_destination(job, "worker-score-check"))
        self.data.get(score.after_rng, design_codec)
        # Worker receipts identify the admitted transport request; the durable
        # member operation identifies job + prediction + protocol + output.
        # Translate only after validating the complete worker evidence above.
        # Both immutable receipts remain durable, without repeating Score.
        measurement_request = DurableMeasurementRequest(
            job, reference, self.protocol, self.spec.output.as_posix(),
        )
        durable_score = replace(
            score, request_identity=self._scoring_identity(measurement_request),
        )
        score_reference = self.data.put(durable_score, _SCORE_RECEIPT_CODEC)
        self.data.persist((score_reference,))
        return self.runtime.call(
            f"native-refinement/evaluate/{job.id}/measurement",
            self.measurement_operation,
            replace(measurement_request, score=score_reference),
            work_id=self._job_work_id(job),
        )

    def _report(self, state, *, request):
        super().checkpoint(state, request=request)

    def checkpoint(self, state, *, request, report=True):
        super().checkpoint(state, request=request, report=report)
        self.trial._state, self.trial._request = state, request

    def perform(self, request):
        if isinstance(self.pool, _ScoreHandoff):
            known = {job.identity for job in self.trial.executor.one_pending.values()}
            fresh = {job.identity for job in request.jobs}
            if any(job.identity not in known | fresh for job in request.state.pending):
                self.pool.begin_recovery(request.state)
        return self.trial.perform(request)

    def shutdown(self):
        self.proposal_pool.close()
        super().shutdown()

    def recover(self, request):
        # Control recovery precedes the outer action's bind. A checkpoint may
        # have admitted, but not issued, a proposal; its queued submission
        # must therefore carry an explicit valid Runtime work scope.
        action = self._action_identity(request)
        if isinstance(self.pool, _ScoreHandoff):
            self.pool.begin_recovery(request.state)
        with self.runtime.bind(
            "parallel-refinement/recovery/" + action.digest,
            History(key_for(()), ()),
        ):
            return self.trial.recover(request)


def run_parallel_native_refinement(
    spec, *, policy, protocol, runtime, config_yaml, workers, worker_output,
    model_sharing="replicated", score_queue_size=0, prediction_width=None,
    async_mutator=False,
    loss_order=None,
    score_context=None,
):
    """Run the same branching policy with spawn-isolated resident evaluators."""
    from astrolabe.app import Astrolabe
    from .composition import ControlRun
    from .control_journal import _CHECKPOINT
    from .refinement_worker import execute, initialize, ready

    if runtime.root is None or spec.data.root is None:
        raise ValueError("Isolated refinement requires rooted runtime and data")
    initargs = (config_yaml, str(spec.data.root), spec.assemblies,
                spec.translation.state, str(worker_output))
    context = mp.get_context("spawn")
    score_pool = None
    handoff = None
    owner = None
    try:
        if model_sharing == "shared":
            score_pool = ProcessPoolExecutor(
                max_workers=workers, mp_context=context, initializer=initialize,
                initargs=(*initargs, "score"),
            )
            # Spawn every CPU resident before bootstrap/model load, but do not
            # wait for readiness on the GPU critical path.
            previous = {name: os.environ.get(name) for name in
                        ("CUDA_VISIBLE_DEVICES", "JAX_PLATFORMS", "GH_CUDA_PREINITIALIZE")}
            try:
                os.environ.update(CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu",
                                  GH_CUDA_PREINITIALIZE="0")
                warm = [score_pool.submit(ready) for _ in range(workers)]
            finally:
                for name, value in previous.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=context, initializer=initialize,
            initargs=(*initargs, "prediction" if score_pool is not None else "combined"),
        ) as prediction_pool:
            pool = prediction_pool
            if score_pool is not None:
                from .refinement_worker import read_raw_receipt
                def verify(task, returned):
                    with span("pipeline.raw_verified", evaluation_id=task["id"]):
                        raw = read_raw_receipt(spec.data.root, worker_output, task)
                        if returned is not None and raw != returned:
                            raise ContractConflict("Returned prediction differs from durable raw receipt")
                        return raw
                handoff = _ScoreHandoff(
                    prediction_pool, score_pool, execute,
                    prediction_width=prediction_width or workers,
                    capacity=score_queue_size, verify=verify,
                )
                pool = handoff
            owner = ParallelRefinementOwner(
                spec, runtime, protocol=protocol, pool=pool, worker=execute,
                worker_output=worker_output,
                score_context=score_context, async_mutator=async_mutator,
            )
            owner.bootstrap()
            if loss_order is not None:
                # A replayed bootstrap receipt does not rebuild process-local
                # pipeline math. Rehydrate its owner before reading the plan.
                owner.models.get()
                current_order = tuple(loss_order())
                saved_order = runtime.load("parallel-refinement/loss-order")
                if saved_order is None:
                    saved_order = current_order
                    runtime.save("parallel-refinement/loss-order", saved_order)
                if set(saved_order) != set(current_order):
                    raise ContractConflict("Persisted Score loss plan differs from the configured terms")
                owner.loss_order = tuple(saved_order)
            initial = policy.start(owner.root_job(protocol=protocol))
            codec = branch_codec(type(initial))
            runtime.data.register(codec)
            persisted = runtime.load("control/state", _CHECKPOINT)
            if persisted is not None:
                owner.checkpoint(
                    runtime.data.get(persisted.state, codec), request=None, report=False,
                )
            with _shared_models(owner, model_sharing == "shared"):
                final = Astrolabe(ControlRun(
                    policy, owner, initial, runtime, codec,
                    branch_codec(BranchAction), branch_codec(BranchEvent),
                )).run()
            owner.checkpoint(final, request=None, report=False)
            return NativeRefinementResult(final.frontier, final, owner)
    finally:
        if owner is not None:
            owner.proposal_pool.close()
        if handoff is not None:
            handoff.drain()
        if score_pool is not None:
            score_pool.shutdown(wait=True)
        if owner is not None:
            owner.shutdown()
