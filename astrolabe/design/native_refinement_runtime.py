"""Durable native refinement through per-stage ``Runtime.call`` receipts.

The legacy ``NativeRefinementOwner`` remains available for process-lifetime
callers. This adapter is used only when the caller supplies a rooted Runtime.
It separates model loading, proposal generation, raw prediction, and CPU
measurement so recovery never treats a whole BranchAction as one paid call.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path
import re
from typing import Any

from astrolabe.evaluators.utils import StandardScores
from astrolabe.model_execution import register_model_owner

from .branch_types import (
    BranchAction,
    BranchEvent,
    BranchJob,
    BranchMember,
    BranchProposal,
    BranchTerminal,
    branch_codec,
)
from .legacy import LEGACY_REFERENCE_CODECS
from .native_refinement import (
    NativeMeasurement,
    NativeProposal,
    NativeRefinementResult,
    NativeRefinementSpec,
    RefinementMemberRecord,
    RefinementProposalRecord,
    _MEMBER_CODEC,
    _METRICS_CODEC,
    _PROPOSAL_CODEC,
    _finite_score,
    _json_metric,
)
from .native_scores import (
    NATIVE_SCORES_REFERENCE_CODECS,
    restore_scores,
    snapshot_scores,
)
from .native_support import LazyNativeOwner
from .operations import INVOKE, operation
from .runtime import ContractConflict, OutcomeUnknown, Runtime
from .store import DataclassCodec, JSONCodec, design_codec
from .types import (
    Assignment,
    Blob,
    Candidate,
    Chemical,
    History,
    Key,
    Polymer,
    Requirement,
    System,
    key_for,
)


_CODE_VERSION = "astrolabe.native-refinement-recovery/v2"
_BOOTSTRAP_WORK = "native-refinement/bootstrap/v1"
_PROPOSAL_ALIAS = "native-refinement/propose"
_PREDICTION_ALIAS = "native-refinement/predict"
_SCORING_ALIAS = "native-refinement/score"
_MEASUREMENT_ALIAS = "native-refinement/measure"


@dataclass(frozen=True)
class DurableProposalRequest:
    job: BranchJob
    protocol: Key
    output: str


@dataclass(frozen=True)
class DurablePredictionRequest:
    job: BranchJob
    protocol: Key
    output: str


@dataclass(frozen=True)
class RawPredictionReceipt:
    """The exact native result committed before any CPU measurement runs."""

    job: BranchJob
    proposal: Blob
    candidate: Candidate
    scores: Blob


@dataclass(frozen=True)
class ScoreReceipt:
    request_identity: Key
    raw_scores: Blob
    scoring_identity: Key
    measured_scores: Blob
    measurement: NativeMeasurement
    after_rng: Blob


@dataclass(frozen=True)
class DurableMeasurementRequest:
    job: BranchJob
    prediction: Blob
    protocol: Key
    output: str
    score: Blob | None = None


@dataclass(frozen=True)
class BufferedTerminal:
    """A completed inner terminal awaiting collection by a later action."""

    origin_action: Key
    position: int
    terminal: BranchTerminal
    after_rng: Blob


@dataclass(frozen=True)
class TerminalBuffer:
    """Ordered terminal delivery buffer persisted before its outer action."""

    entries: tuple[BufferedTerminal, ...] = ()
    delivery_action: Key | None = None


_BRANCH_MEMBER_RESULT_CODEC = branch_codec(BranchMember)
_BRANCH_PROPOSAL_RESULT_CODEC = branch_codec(BranchProposal)
_TERMINAL_BUFFER_CODEC = DataclassCodec(
    TerminalBuffer,
    "astrolabe.design.native-refinement.terminal-buffer/v1",
    nested_types=(
        BufferedTerminal,
        BranchTerminal,
        BranchJob,
        BranchMember,
        BranchProposal,
        Blob,
        Key,
    ),
)
_PROPOSAL_REQUEST_CODEC = DataclassCodec(
    DurableProposalRequest,
    "astrolabe.design.native-refinement.proposal-request/v1",
    nested_types=(BranchJob, BranchMember, BranchProposal, Blob, Key),
)
_PREDICTION_REQUEST_CODEC = DataclassCodec(
    DurablePredictionRequest,
    "astrolabe.design.native-refinement.prediction-request/v1",
    nested_types=(BranchJob, BranchMember, BranchProposal, Blob, Key),
)
_RAW_PREDICTION_CODEC = DataclassCodec(
    RawPredictionReceipt,
    "astrolabe.design.native-refinement.raw-prediction/v1",
    nested_types=(
        BranchJob,
        BranchMember,
        BranchProposal,
        Blob,
        Candidate,
        Assignment,
        Polymer,
        Chemical,
        System,
        Key,
    ),
)
_MEASUREMENT_REQUEST_CODEC = DataclassCodec(
    DurableMeasurementRequest,
    "astrolabe.design.native-refinement.measurement-request/v1",
    nested_types=(BranchJob, BranchMember, BranchProposal, Blob, Key),
)
_SCORE_RECEIPT_CODEC = DataclassCodec(
    ScoreReceipt,
    "astrolabe.design.native-refinement.score-receipt/v1",
    nested_types=(Blob, Key, NativeMeasurement),
)
_JSON = JSONCodec()
_TERMINAL_BUFFER_NAME = "native-refinement/terminal-buffer"


def _unique_codecs(*groups):
    result = []
    seen = set()
    for group in groups:
        for codec in group:
            if codec.identity not in seen:
                seen.add(codec.identity)
                result.append(codec)
    return tuple(result)


_REFERENCE_CODECS = _unique_codecs(
    LEGACY_REFERENCE_CODECS,
    NATIVE_SCORES_REFERENCE_CODECS,
    (_MEMBER_CODEC, _PROPOSAL_CODEC, _METRICS_CODEC, _JSON),
    (
        _PROPOSAL_REQUEST_CODEC,
        _PREDICTION_REQUEST_CODEC,
        _RAW_PREDICTION_CODEC,
        _MEASUREMENT_REQUEST_CODEC,
        _SCORE_RECEIPT_CODEC,
        _BRANCH_MEMBER_RESULT_CODEC,
        _BRANCH_PROPOSAL_RESULT_CODEC,
        _TERMINAL_BUFFER_CODEC,
    ),
)


def _source_identity(value: Any) -> str:
    try:
        return inspect.getsource(value)
    except (OSError, TypeError):
        module = getattr(value, "__module__", type(value).__module__)
        qualname = getattr(value, "__qualname__", type(value).__qualname__)
        return f"{module}.{qualname}"


class _LoadedModels:
    def __init__(self, mutator: Any, evaluator: Any) -> None:
        self.mutator = mutator
        self.evaluator = evaluator
        self._shutdown = False

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        first_error = None
        for owner in (self.evaluator, self.mutator):
            close = getattr(owner, "shutdown", None)
            if close is None:
                continue
            try:
                close()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error


class DurableNativeRefinementOwner:
    """A restartable serial owner with independent native-stage receipts."""

    request_type = BranchAction
    result_type = BranchEvent

    def __init__(
        self,
        spec: NativeRefinementSpec,
        runtime: Runtime,
        *,
        protocol: Key,
    ) -> None:
        if not isinstance(spec, NativeRefinementSpec):
            raise TypeError("spec must be a NativeRefinementSpec")
        if not isinstance(runtime, Runtime):
            raise TypeError("runtime must be an astrolabe.design.Runtime")
        if runtime.root is None:
            raise ValueError("durable native refinement requires a rooted Runtime")
        if runtime.data is not spec.data:
            raise ValueError("Runtime.data must be the exact DataStore owned by spec.data")
        if not isinstance(protocol, Key):
            raise TypeError("protocol must be a content Key")

        self.spec = spec
        self.runtime = runtime
        self.data = spec.data
        self.protocol = protocol
        self.controller_state = None
        self.member_blobs: dict[int, Blob] = {}
        self._observed_proposals: set[int] = set()
        self._state = None
        self._request = None
        self._shutdown = False

        model_sources = tuple(
            _source_identity(callback)
            for callback in (
                spec.load_mutator,
                spec.load_evaluator,
                spec.propose,
                spec.evaluate,
                spec.measure,
            )
        )
        self.model_identity = key_for(
            (
                _CODE_VERSION,
                protocol,
                spec.mutator_alias,
                spec.evaluator_alias,
                spec.output.as_posix(),
                spec.translation.system.identity,
                spec.initial_candidate.identity,
                model_sources,
            )
        )
        self.models = LazyNativeOwner(
            self._load_models,
            self.data,
            self.model_identity,
            alias="native-refinement/models",
        )
        self.initialization = self.models.initialization
        self.initialization_request = self.models.initialization_request

        common = dict(
            version=_CODE_VERSION,
            model_identity=self.model_identity,
            reference_codecs=_REFERENCE_CODECS,
        )
        self.proposal_operation = operation(
            _PROPOSAL_ALIAS,
            DurableProposalRequest,
            BranchProposal,
            self._propose,
            request_codec=_PROPOSAL_REQUEST_CODEC,
            result_codec=_BRANCH_PROPOSAL_RESULT_CODEC,
            paid_boundary="propose",
            **common,
        )
        self.prediction_operation = operation(
            _PREDICTION_ALIAS,
            DurablePredictionRequest,
            RawPredictionReceipt,
            self._predict,
            request_codec=_PREDICTION_REQUEST_CODEC,
            result_codec=_RAW_PREDICTION_CODEC,
            paid_boundary="predict",
            **common,
        )
        self.measurement_operation = operation(
            _MEASUREMENT_ALIAS,
            DurableMeasurementRequest,
            BranchMember,
            self._measure,
            request_codec=_MEASUREMENT_REQUEST_CODEC,
            result_codec=_BRANCH_MEMBER_RESULT_CODEC,
            paid_boundary="measure",
            **common,
        )
        self.scoring_operation = operation(
            _SCORING_ALIAS,
            DurableMeasurementRequest,
            ScoreReceipt,
            self._score,
            request_codec=_MEASUREMENT_REQUEST_CODEC,
            result_codec=_SCORE_RECEIPT_CODEC,
            paid_boundary="score",
            **common,
        )
        self.operations = (
            self.initialization,
            self.proposal_operation,
            self.prediction_operation,
            self.scoring_operation,
            self.measurement_operation,
        )
        self.identity = key_for(
            (
                _CODE_VERSION,
                inspect.getsource(type(self)),
                inspect.getsource(_LoadedModels),
                inspect.getsource(LazyNativeOwner),
                inspect.getsource(DurableProposalRequest),
                inspect.getsource(DurablePredictionRequest),
                inspect.getsource(RawPredictionReceipt),
                inspect.getsource(DurableMeasurementRequest),
                self.model_identity,
                self.spec.seed,
                self.spec.width,
                self.spec.proposals_per_parent,
                self.spec.max_evaluations,
                tuple(item.descriptor.identity for item in self.operations),
            )
        )
        self.requirements = tuple(
            Requirement(item.descriptor.alias, INVOKE, "required")
            for item in self.operations
        )
        self.attach()

    def _load_models(self):
        mutator = self.spec.load_mutator()
        try:
            evaluator = self.spec.load_evaluator()
        except BaseException:
            close = getattr(mutator, "shutdown", None)
            if close is not None:
                close()
            raise
        return _LoadedModels(mutator, evaluator)

    def attach(self) -> None:
        """Attach this owner's typed operations to the caller's Runtime."""

        for component in self.operations:
            existing = self.runtime.operations.get(component.descriptor.alias)
            if existing is not None and (
                existing.descriptor.identity != component.descriptor.identity
            ):
                raise ContractConflict(
                    f"Runtime operation {component.descriptor.alias!r} belongs "
                    "to another scientific implementation"
                )
            self.runtime.operations[component.descriptor.alias] = component
            self.runtime.data.register(component.descriptor.request_codec)
            self.runtime.data.register(component.descriptor.result_codec)
            for codec in component.reference_codecs:
                self.runtime.data.register(codec)
        self.data.register(_JSON)

    def bootstrap(self) -> None:
        """Load models once under a work ID independent of branch transitions."""

        self.runtime.preflight(self.requirements)
        history = History(key_for(()), ())
        with self.runtime.bind(_BOOTSTRAP_WORK, history):
            self.runtime.call(
                "models/initialize",
                self.initialization,
                self.initialization_request,
            )
        register_model_owner(self)

    @property
    def ready(self) -> bool:
        return self.models.ready

    def root_job(self, *, protocol: Key) -> BranchJob:
        if protocol != self.protocol:
            raise ValueError("root protocol differs from the durable owner contract")
        proposal = RefinementProposalRecord(
            candidate=self.spec.initial_candidate,
            label=self.spec.label(self.spec.initial_candidate),
            generation=0,
            parent_id=-1,
            proposal_index=0,
            proposal_step=0,
            proposal_seed=self.spec.seed,
            protocol=protocol,
        )
        value = self.data.put(proposal, _PROPOSAL_CODEC)
        return BranchJob(
            phase="evaluate",
            id=0,
            generation=0,
            seed=self.spec.seed,
            proposal=None,
            slot=0,
            source=value,
        )

    @staticmethod
    def _proposal_blob(job: BranchJob) -> Blob:
        value = job.proposal.value if job.proposal is not None else job.source
        if value is None or value.codec != _PROPOSAL_CODEC.identity:
            raise ValueError("branch job does not carry a refinement proposal record")
        return value

    def _proposal_record(self, job: BranchJob) -> RefinementProposalRecord:
        proposal = self.data.get(self._proposal_blob(job), _PROPOSAL_CODEC)
        if not isinstance(proposal, RefinementProposalRecord):
            raise TypeError("proposal blob did not decode to a refinement proposal")
        return proposal

    def _member_record(self, job: BranchJob) -> RefinementMemberRecord:
        if job.parent is None or job.parent.value.codec != _MEMBER_CODEC.identity:
            raise ValueError("branch job parent is not a refinement member")
        member = self.data.get(job.parent.value, _MEMBER_CODEC)
        if not isinstance(member, RefinementMemberRecord):
            raise TypeError("member blob did not decode to a refinement member")
        if member.score != job.parent.score or member.label != job.parent.key:
            raise ValueError("parent BranchMember does not match its owner record")
        return member

    def _activate(self, candidate: Candidate):
        return self.spec.context.activate(
            candidate,
            translation=self.spec.translation,
            data=self.data,
        )

    def _score_destination(self, job: BranchJob, stage: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{job.phase}_{job.id}")
        destination = self.spec.output / ".native_refinement" / f"{stage}_{safe}"
        destination.mkdir(parents=True, exist_ok=True)
        return destination

    def _propose(self, request: DurableProposalRequest) -> BranchProposal:
        if request.protocol != self.protocol:
            raise ValueError("proposal request protocol differs from the owner contract")
        job = request.job
        if job.phase != "propose" or job.parent is None:
            raise ValueError("proposal operation requires a proposal job with a parent")
        parent = self._member_record(job)
        proposal_seed = self.spec.proposal_seed(self.spec.seed, job.id)
        if type(proposal_seed) is not int or not 0 <= proposal_seed <= 2**32 - 1:
            raise ValueError("proposal seed callback must return a valid uint32 seed")
        if job.seed != proposal_seed:
            raise ValueError("proposal job seed differs from the declared owner schedule")

        self.spec.event(
            "proposal_started",
            generation=job.generation,
            parent_id=job.parent.id,
            proposal_index=job.slot,
            proposal_step=job.id,
            seed=proposal_seed,
        )
        scores = restore_scores(
            parent.scores,
            self.data,
            self._score_destination(job, "proposal-input"),
        )
        loaded = self.models.get()
        try:
            with self._activate(parent.candidate):
                proposed = self.spec.propose(
                    loaded.mutator,
                    parent.candidate,
                    scores,
                    seed=proposal_seed,
                    step=job.id,
                    generation=job.generation,
                    parent_id=job.parent.id,
                    proposal_index=job.slot,
                    output=Path(request.output),
                )
        except BaseException as error:
            self.spec.event(
                "proposal_failed",
                generation=job.generation,
                parent_id=job.parent.id,
                proposal_index=job.slot,
                proposal_step=job.id,
                seed=proposal_seed,
                error=repr(error),
            )
            raise
        if not isinstance(proposed, NativeProposal):
            raise TypeError("propose callback must return NativeProposal")
        if proposed.candidate.system != self.spec.translation.system.identity:
            raise ValueError("propose callback returned a candidate from another System")
        record = RefinementProposalRecord(
            candidate=proposed.candidate,
            label=proposed.label,
            generation=job.generation,
            parent_id=job.parent.id,
            proposal_index=job.slot,
            proposal_step=job.id,
            proposal_seed=proposal_seed,
            protocol=self.protocol,
        )
        value = self.data.put(record, _PROPOSAL_CODEC)
        self.spec.event(
            "proposal_completed",
            generation=job.generation,
            parent_id=job.parent.id,
            proposal_index=job.slot,
            proposal_step=job.id,
            sequence=proposed.label,
            seed=proposal_seed,
        )
        return BranchProposal(
            id=job.id,
            parent_id=job.parent.id,
            key=proposed.label,
            value=value,
            valid=True,
        )

    def _predict(self, request: DurablePredictionRequest) -> RawPredictionReceipt:
        if request.protocol != self.protocol:
            raise ValueError("prediction request protocol differs from the owner contract")
        job = request.job
        if job.phase != "evaluate" or job.seed != self.spec.seed:
            raise ValueError("prediction operation requires the fixed evaluation seed")
        proposal_blob = self._proposal_blob(job)
        proposal = self._proposal_record(job)
        self.spec.event(
            "evaluation_started",
            evaluation_id=job.id,
            generation=proposal.generation,
            parent_id=None if proposal.parent_id < 0 else proposal.parent_id,
            proposal_index=proposal.proposal_index,
            proposal_step=proposal.proposal_step,
            sequence=proposal.label,
            seed=self.spec.seed,
        )
        loaded = self.models.get()
        try:
            with self._activate(proposal.candidate):
                scores = self.spec.evaluate(
                    loaded.evaluator,
                    proposal.candidate,
                    seed=self.spec.seed,
                    step=proposal.proposal_step,
                    assemblies=self.spec.assemblies,
                    output=Path(request.output),
                    proposal_id=job.id,
                )
                if type(scores) is not StandardScores:
                    raise TypeError("native evaluator must return StandardScores unchanged")
                scores_blob = snapshot_scores(scores, self.data)
                self.data.persist((scores_blob,))
        except BaseException as error:
            self.spec.event(
                "evaluation_failed",
                evaluation_id=job.id,
                generation=proposal.generation,
                parent_id=None if proposal.parent_id < 0 else proposal.parent_id,
                proposal_index=proposal.proposal_index,
                proposal_step=proposal.proposal_step,
                sequence=proposal.label,
                seed=self.spec.seed,
                stage="prediction",
                error=repr(error),
            )
            raise
        return RawPredictionReceipt(
            job=job,
            proposal=proposal_blob,
            candidate=proposal.candidate,
            scores=scores_blob,
        )

    def _scoring_identity(self, request: DurableMeasurementRequest) -> Key:
        return key_for((request.job, request.prediction, request.protocol, request.output))

    def _score(self, request: DurableMeasurementRequest) -> ScoreReceipt:
        """Measure a private restored raw result, without committing a member."""
        if request.protocol != self.protocol or request.score is not None:
            raise ContractConflict("Score request differs from its prediction contract")
        prediction = self.data.get(request.prediction, _RAW_PREDICTION_CODEC)
        if prediction.job != request.job:
            raise ContractConflict("Score request belongs to another prediction job")
        proposal = self.data.get(prediction.proposal, _PROPOSAL_CODEC)
        scores = restore_scores(
            prediction.scores, self.data,
            self._score_destination(request.job, "measurement-input"),
        )
        self.models.get()
        try:
            with self._activate(prediction.candidate):
                measurement = self.spec.measure(prediction.candidate, scores, request.job)
            if not isinstance(measurement, NativeMeasurement):
                raise TypeError("measure callback must return NativeMeasurement")
            measurement = NativeMeasurement(
                score=_finite_score(measurement.score),
                metrics=_json_metric(dict(measurement.metrics)),
                structure_path=str(measurement.structure_path),
            )
            measured_scores = snapshot_scores(scores, self.data)
            after_rng = self.runtime._capture_rng()
            self.data.persist((measured_scores, after_rng))
            return ScoreReceipt(
                self._scoring_identity(request), prediction.scores, self.protocol,
                measured_scores, measurement, after_rng,
            )
        except BaseException as error:
            self.spec.event(
                "evaluation_failed", evaluation_id=request.job.id,
                generation=proposal.generation,
                parent_id=None if proposal.parent_id < 0 else proposal.parent_id,
                proposal_index=proposal.proposal_index,
                proposal_step=proposal.proposal_step, sequence=proposal.label,
                seed=self.spec.seed, stage="measurement", error=repr(error),
            )
            raise

    def _measure(self, request: DurableMeasurementRequest) -> BranchMember:
        if request.protocol != self.protocol:
            raise ValueError("measurement request protocol differs from the owner contract")
        prediction = self.data.get(request.prediction, _RAW_PREDICTION_CODEC)
        if not isinstance(prediction, RawPredictionReceipt):
            raise TypeError("prediction blob did not decode to a raw prediction receipt")
        if prediction.job != request.job:
            raise ValueError("measurement request belongs to another prediction job")
        proposal = self.data.get(prediction.proposal, _PROPOSAL_CODEC)
        if not isinstance(proposal, RefinementProposalRecord):
            raise TypeError("raw prediction proposal did not decode")
        try:
            if request.score is not None:
                receipt = self.data.get(request.score, _SCORE_RECEIPT_CODEC)
                if (receipt.scoring_identity != self.protocol
                        or receipt.raw_scores != prediction.scores
                        or receipt.request_identity != self._scoring_identity(request)):
                    raise ContractConflict("Score receipt differs from its prediction contract")
                measurement = receipt.measurement
                measured_scores_blob = receipt.measured_scores
                self.runtime._restore_rng(receipt.after_rng)
            else:
                receipt = self._score(request)
                measurement, measured_scores_blob = receipt.measurement, receipt.measured_scores
            if not isinstance(measurement, NativeMeasurement):
                raise TypeError("measure callback must return NativeMeasurement")
        except BaseException as error:
            self.spec.event(
                "evaluation_failed",
                evaluation_id=request.job.id,
                generation=proposal.generation,
                parent_id=None if proposal.parent_id < 0 else proposal.parent_id,
                proposal_index=proposal.proposal_index,
                proposal_step=proposal.proposal_step,
                sequence=proposal.label,
                seed=self.spec.seed,
                stage="measurement",
                error=repr(error),
            )
            raise

        metrics_blob = self.data.put(
            _json_metric(dict(measurement.metrics)),
            _METRICS_CODEC,
        )
        self.data.persist((measured_scores_blob,))
        record = RefinementMemberRecord(
            candidate=prediction.candidate,
            scores=measured_scores_blob,
            score=_finite_score(measurement.score),
            structure_path=str(measurement.structure_path),
            metrics=metrics_blob,
            label=proposal.label,
            generation=proposal.generation,
            parent_id=None if proposal.parent_id < 0 else proposal.parent_id,
            proposal_step=proposal.proposal_step,
        )
        value = self.data.put(record, _MEMBER_CODEC)
        member = BranchMember(
            id=request.job.id,
            key=proposal.label,
            value=value,
            score=record.score,
            eligible=True,
        )
        self.spec.event(
            "evaluation_completed",
            evaluation_id=request.job.id,
            generation=proposal.generation,
            parent_id=record.parent_id,
            proposal_index=proposal.proposal_index,
            proposal_step=proposal.proposal_step,
            sequence=proposal.label,
            seed=self.spec.seed,
            score=member.score,
            structure_path=record.structure_path,
        )
        return member

    @staticmethod
    def _job_key(job: BranchJob) -> tuple[str, int]:
        return job.phase, job.id

    @staticmethod
    def _job_work_id(job: BranchJob) -> str:
        return f"native-refinement/job/{job.identity}"

    def _outstanding_jobs(self, request: BranchAction) -> tuple[BranchJob, ...]:
        values = {}
        for job in (
            *request.jobs,
            *request.state.batch,
            *request.state.pending,
        ):
            values[job.identity] = job
        return tuple(values.values())

    def _action_identity(self, request: BranchAction) -> Key:
        return key_for(
            (
                "astrolabe.native-refinement.action/v1",
                self.protocol,
                request,
            )
        )

    def _load_terminal_buffer(self) -> TerminalBuffer:
        value = self.runtime.load(
            _TERMINAL_BUFFER_NAME,
            _TERMINAL_BUFFER_CODEC,
        )
        if value is None:
            return TerminalBuffer()
        if not isinstance(value, TerminalBuffer):
            raise TypeError("native refinement terminal buffer has the wrong type")
        return value

    def _save_terminal_buffer(
        self,
        entries: tuple[BufferedTerminal, ...],
        *,
        delivery_action: Key | None = None,
    ) -> None:
        self.runtime.save(
            _TERMINAL_BUFFER_NAME,
            TerminalBuffer(entries, delivery_action),
            _TERMINAL_BUFFER_CODEC,
        )

    def _restore_entry_rng(
        self,
        entries: tuple[BufferedTerminal, ...],
    ) -> None:
        for entry in entries:
            self.runtime._restore_rng(entry.after_rng)

    def _restore_same_origin_rng(
        self,
        action: Key,
        entries: tuple[BufferedTerminal, ...],
    ) -> tuple[BufferedTerminal, ...]:
        selected = tuple(entry for entry in entries if entry.origin_action == action)
        if not selected:
            return ()
        if entries[-len(selected) :] != selected:
            raise ContractConflict(
                "A reentered action's buffered terminals are not a suffix"
            )
        previous = -1
        for entry in selected:
            if entry.position <= previous:
                raise ContractConflict(
                    "A reentered action's terminals are not in execution order"
                )
            previous = entry.position
        self._restore_entry_rng(selected)
        return selected

    def _rounds_for(self, job: BranchJob):
        base = f"native-refinement/{job.phase}/{job.id}"
        if job.phase == "propose":
            return ((base + "/proposal", self.proposal_operation),)
        return (
            (base + "/prediction", self.prediction_operation),
            (base + "/score", self.scoring_operation),
            (base + "/measurement", self.measurement_operation),
        )

    def _terminal(self, job: BranchJob) -> BranchTerminal:
        base = f"native-refinement/{job.phase}/{job.id}"
        if job.phase == "propose":
            proposal = self.runtime.call(
                base + "/proposal",
                self.proposal_operation,
                DurableProposalRequest(job, self.protocol, self.spec.output.as_posix()),
                work_id=self._job_work_id(job),
            )
            return BranchTerminal(job, "completed", proposal)
        prediction = self.runtime.call(
            base + "/prediction",
            self.prediction_operation,
            DurablePredictionRequest(job, self.protocol, self.spec.output.as_posix()),
            work_id=self._job_work_id(job),
        )
        prediction_ref = self.data.put(prediction, _RAW_PREDICTION_CODEC)
        score = self.runtime.call(
            base + "/score",
            self.scoring_operation,
            DurableMeasurementRequest(
                job, prediction_ref, self.protocol, self.spec.output.as_posix(),
            ),
            work_id=self._job_work_id(job),
        )
        score_ref = self.data.put(score, _SCORE_RECEIPT_CODEC)
        member = self.runtime.call(
            base + "/measurement",
            self.measurement_operation,
            DurableMeasurementRequest(
                job,
                prediction_ref,
                self.protocol,
                self.spec.output.as_posix(),
                score_ref,
            ),
            work_id=self._job_work_id(job),
        )
        return BranchTerminal(job, "completed", member)

    def perform(self, request: BranchAction) -> BranchEvent:
        if type(request) is not BranchAction:
            raise TypeError("durable native refinement expects BranchAction")
        if request.wait not in ("any", "all", "none"):
            raise ValueError(f"unsupported wait mode {request.wait!r}")
        register_model_owner(self)
        self._state, self._request = request.state, request
        action = self._action_identity(request)
        buffer = self._load_terminal_buffer()
        if buffer.delivery_action is not None:
            if buffer.delivery_action == action:
                self._restore_entry_rng(buffer.entries)
                return BranchEvent(
                    tuple(entry.terminal for entry in buffer.entries)
                )
            # The delivering action completed before a different action started.
            # Its RNG is already represented by the caller's later checkpoint.
            self._save_terminal_buffer(())

        buffered = () if buffer.delivery_action is not None else buffer.entries
        self._restore_same_origin_rng(action, buffered)

        jobs = tuple(sorted(self._outstanding_jobs(request), key=self._job_key))
        positions = {job.identity: index for index, job in enumerate(jobs)}
        entries = list(buffered)
        unknown_buffered = {
            entry.terminal.job.identity for entry in entries
        } - set(positions)
        if unknown_buffered:
            raise ContractConflict(
                "Buffered native terminals are no longer outstanding: "
                f"{sorted(unknown_buffered)!r}"
            )
        completed = {entry.terminal.job.identity for entry in entries}
        for job in jobs:
            if job.identity in completed:
                continue
            if job.identity not in positions:
                raise ContractConflict(
                    f"Buffered native job {job.identity!r} is not outstanding"
                )
            entries.append(
                BufferedTerminal(
                    origin_action=action,
                    position=positions[job.identity],
                    terminal=self._terminal(job),
                    after_rng=self.runtime._capture_rng(),
                )
            )
            completed.add(job.identity)
            self._save_terminal_buffer(tuple(entries))

        if request.wait == "none" or not entries:
            if request.wait == "none":
                self._save_terminal_buffer(tuple(entries))
            return BranchEvent(())
        terminals = tuple(entry.terminal for entry in entries)
        self._save_terminal_buffer(entries, delivery_action=action)
        return BranchEvent(terminals)

    def recover(self, request: BranchAction):
        """Permit reentry only when every inner paid call is definitely free."""

        if type(request) is not BranchAction:
            raise TypeError("durable native refinement recovery expects BranchAction")
        expected = {}
        for job in self._outstanding_jobs(request):
            for named_round, component in self._rounds_for(job):
                expected[named_round] = component
        for attempt in self.runtime.attempts():
            component = expected.get(attempt.key.named_round)
            if component is None:
                continue
            if attempt.status in ("issued", "unknown"):
                raise OutcomeUnknown(attempt.identity)
            if attempt.status == "completed":
                if attempt.result is None or attempt.rng is None:
                    raise ContractConflict(
                        f"Completed native refinement receipt {attempt.identity} is incomplete"
                    )
                self.data.get(attempt.result, component.descriptor.result_codec)
                if attempt.rng.protocol != self.runtime.scientific_contract:
                    raise ContractConflict(
                        f"Native refinement receipt {attempt.identity} has another RNG contract"
                    )
                self.data.get(attempt.rng.after, design_codec)
            elif attempt.status == "failed":
                # Reentry raises Runtime's recorded failure without invoking the
                # failed callback again.
                continue
            elif attempt.status not in ("reserved",):
                raise ContractConflict(
                    f"Unknown native refinement receipt state {attempt.status!r}"
                )
        self._state, self._request = request.state, request
        return None

    def checkpoint(self, state: Any, *, request: Any, report: bool = True) -> None:
        self.controller_state = state
        self._state, self._request = state, request
        if not report:
            self._rehydrate_observed_proposals(state)
            return
        admitted = {
            job.proposal.id
            for job in (
                *tuple(getattr(state, "queued", ())),
                *tuple(getattr(state, "pending", ())),
            )
            if job.proposal is not None
        }
        for terminal in tuple(getattr(state, "batch_results", ())):
            if terminal.job.phase != "propose":
                continue
            identifier = terminal.job.id
            if identifier in self._observed_proposals:
                continue
            self._observed_proposals.add(identifier)
            if terminal.status != "completed":
                self._report_proposal_terminal(terminal, "refused")
                continue
            proposal = terminal.value
            if not proposal.valid:
                self._report_proposal_terminal(terminal, "rejected")
            elif (
                proposal.id not in admitted
                and proposal.key in tuple(getattr(state, "seen", ()))
            ):
                self._report_proposal_terminal(terminal, "duplicate")

    def _rehydrate_observed_proposals(self, state: Any) -> None:
        """Mark checkpointed proposal outcomes as already reported."""

        self._observed_proposals.update(
            terminal.job.id
            for terminal in tuple(getattr(state, "batch_results", ()))
            if terminal.job.phase == "propose"
        )

    def _report_proposal_terminal(
        self,
        terminal: BranchTerminal,
        outcome: str,
    ) -> None:
        proposal = terminal.value
        if not isinstance(proposal, BranchProposal):
            raise TypeError("proposal terminal must carry BranchProposal")
        record = self.proposal(proposal)
        self.spec.event(
            f"{outcome}_proposal",
            generation=record.generation,
            parent_id=record.parent_id,
            proposal_index=record.proposal_index,
            proposal_step=record.proposal_step,
            sequence=record.label,
            seed=record.proposal_seed,
        )

    def member(self, value: BranchMember) -> RefinementMemberRecord:
        if value.value.codec != _MEMBER_CODEC.identity:
            raise ValueError("member is not a native refinement member")
        record = self.data.get(value.value, _MEMBER_CODEC)
        if not isinstance(record, RefinementMemberRecord):
            raise TypeError("member did not decode to a refinement record")
        return record

    def proposal(self, value: BranchProposal) -> RefinementProposalRecord:
        if value.value.codec != _PROPOSAL_CODEC.identity:
            raise ValueError("proposal is not a native refinement proposal")
        record = self.data.get(value.value, _PROPOSAL_CODEC)
        if not isinstance(record, RefinementProposalRecord):
            raise TypeError("proposal did not decode to a refinement record")
        return record

    def metrics(self, member: RefinementMemberRecord) -> dict[str, Any]:
        result = self.data.get(member.metrics, _METRICS_CODEC)
        if not isinstance(result, dict):
            raise TypeError("refinement metrics did not decode to a mapping")
        return result

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        self.models.shutdown()


def run_durable_native_refinement(
    spec: NativeRefinementSpec,
    *,
    policy: Any,
    protocol: Key,
    runtime: Runtime,
) -> NativeRefinementResult:
    from astrolabe.app import Astrolabe

    from .composition import ControlRun

    owner = DurableNativeRefinementOwner(spec, runtime, protocol=protocol)
    try:
        owner.bootstrap()
        initial = policy.start(owner.root_job(protocol=protocol))
        state_codec = branch_codec(type(initial))
        from .control_journal import _CHECKPOINT

        runtime.data.register(state_codec)
        persisted = runtime.load("control/state", _CHECKPOINT)
        if persisted is not None:
            owner.checkpoint(
                runtime.data.get(persisted.state, state_codec),
                request=None,
                report=False,
            )
        final = Astrolabe(
            ControlRun(
                policy,
                owner,
                initial,
                runtime,
                state_codec,
                branch_codec(BranchAction),
                branch_codec(BranchEvent),
            )
        ).run()
        selected = getattr(final, "frontier", None)
        if not isinstance(selected, tuple):
            raise TypeError("BranchingPolicy.selected must return branching state")
        if any(not isinstance(member, BranchMember) for member in selected):
            raise TypeError("selected branch members must be BranchMember values")
        owner.checkpoint(final, request=None, report=False)
        return NativeRefinementResult(selected, final, owner)
    except BaseException:
        owner.shutdown()
        raise


__all__ = [
    "DurableMeasurementRequest",
    "DurableNativeRefinementOwner",
    "DurablePredictionRequest",
    "DurableProposalRequest",
    "RawPredictionReceipt",
    "run_durable_native_refinement",
]
