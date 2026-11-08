"""Isolated, resident prediction and CPU Score for generic refinement.

Legacy mutation and member commits remain on the single scientific writer.
NativeBranchTrial records every issue
before dispatch and refuses to replay a request without terminal evidence.
"""
from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor, wait
from collections import deque
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict, replace
import hashlib
import inspect
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys

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
    def __init__(self, owner, pool):
        self.owner, self.pool = owner, pool
        self._proposals = deque()

    def submit(self, worker, task):
        if task["kind"] == "evaluate":
            return self.pool.submit(worker, task)
        # Proposals own legacy state and the parent's RNG. Never submit them
        # to an executor thread, even when several are admitted in one round.
        future = Future()
        runtime = self.owner.runtime
        self._proposals.append((future, task, runtime._work, runtime.history))
        return future

    def advance(self, action):
        # Dispatch-only turns never do proposal work. An any turn yields after
        # one proposal so control can dispatch its committed prefix to Score.
        limit = len(self._proposals) if action.wait == "all" else int(action.wait == "any")
        for _ in range(min(limit, len(self._proposals))):
            self._advance_one()

    def _advance_one(self):
        future, task, work, history = self._proposals.popleft()
        runtime = self.owner.runtime
        previous = runtime._work, runtime.history
        try:
            if work is None or runtime._work is None:
                raise RuntimeError("Queued proposals require a bound Runtime work entry")
            # _terminal's explicit job work_id fixes receipt/replay identity;
            # the admitted job fixes the proposal seed. This metadata swap is
            # only for the promised original admission scope of deferred work,
            # not needed for those identities. Keep the active parent's RNG:
            # bind() cannot nest, and reentering an old binding would restore
            # its entry RNG across intervening proposals/Score. No RNG restore
            # or new receipt is introduced here; issued work still needs proof.
            runtime._work, runtime.history = work, history
            job = self.owner.data.get(_blob(task["job"]), self.owner.job_codec)
            terminal = self.owner._terminal(job)
            reference = self.owner.data.put(terminal.value, self.owner.proposal_codec)
            rng = self.owner.runtime._capture_rng()
            self.owner.data.persist((reference, rng))
            future.set_result({
                **task, "proposal": asdict(reference), "after_rng": asdict(rng),
            })
        except BaseException as error:
            future.set_exception(error)
        finally:
            runtime._work, runtime.history = previous


class _ScoreHandoff:
    """Join resident prediction and Score without touching parent science state."""

    def __init__(self, prediction_pool, score_pool, score_worker):
        self.prediction_pool = prediction_pool
        self.score_pool = score_pool
        self.score_worker = score_worker
        self.finals = []
        self.scores = []

    def submit_score(self, task):
        future = self.score_pool.submit(self.score_worker, task)
        self.scores.append(future)
        return future

    def submit(self, worker, task):
        final = Future()
        prediction = self.prediction_pool.submit(worker, task)
        final.set_running_or_notify_cancel()
        self.finals.append(final)

        def scored(done):
            try:
                final.set_result(done.result())
            except BaseException as error:
                final.set_exception(error)

        def predicted(done):
            try:
                done.result()  # Raw receipt is durable; Score reads and verifies it.
                self.submit_score(task).add_done_callback(scored)
            except BaseException as error:
                final.set_exception(error)

        prediction.add_done_callback(predicted)
        return final

    def drain(self):
        # A final becomes done only after its prediction and Score callback.
        wait(self.finals)
        wait(self.scores)


class ParallelRefinementOwner(DurableNativeRefinementOwner):
    """Single-writer proposal/measurement with bounded resident predictors."""

    def __init__(self, spec, runtime, *, protocol, pool, worker, worker_output,
                 loss_order=None, score_context=None):
        super().__init__(spec, runtime, protocol=protocol)
        from .branch_types import BranchJob
        from .refinement_worker import execute, initialize

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
            inspect.getsource(sys.modules[measure_configured_scores.__module__]),
            inspect.getsource(sys.modules[LossEngine.__module__]),
            tuple((name, _source_identity(loss_registry.get_function(name)))
                  for name in sorted(loss_registry.valid_losses)),
            loss_order,
        ))
        prediction_pool = _PredictionPool(self, pool)
        self.trial = NativeBranchTrial(
            prediction_pool, runtime, identity=self.identity,
            worker=worker, task_factory=self._task, result_factory=self._result,
            probe=self._probe, terminal_proposal_errors=False,
            report=self._report,
            before_wait=prediction_pool.advance, ordered_results=True,
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
        return self.trial.perform(request)

    def recover(self, request):
        # Control recovery precedes the outer action's bind. A checkpoint may
        # have admitted, but not issued, a proposal; its queued submission
        # must therefore carry an explicit valid Runtime work scope.
        action = self._action_identity(request)
        with self.runtime.bind(
            "parallel-refinement/recovery/" + action.digest,
            History(key_for(()), ()),
        ):
            return self.trial.recover(request)


def run_parallel_native_refinement(
    spec, *, policy, protocol, runtime, config_yaml, workers, worker_output,
    model_sharing="replicated",
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
                handoff = _ScoreHandoff(prediction_pool, score_pool, execute)
                pool = handoff
            owner = ParallelRefinementOwner(
                spec, runtime, protocol=protocol, pool=pool, worker=execute,
                worker_output=worker_output,
                score_context=score_context,
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
        if handoff is not None:
            handoff.drain()
        if score_pool is not None:
            score_pool.shutdown(wait=True)
        if owner is not None:
            owner.shutdown()
