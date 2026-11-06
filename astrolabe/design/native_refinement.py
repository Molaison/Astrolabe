"""Generic serial native refinement over the shared branching control contracts.

This module owns mutable native model objects and the legacy globals activated
for each call.  Search decisions, budget accounting, duplicate handling and
frontier selection remain in ``BranchingPolicy``.  Scientific measurement is
injected by the caller; this module has no RNA, protein, backend or task-name
branch.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import Future
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

from .branch_types import (
    BranchAction,
    BranchEvent,
    BranchJob,
    BranchMember,
    BranchProposal,
    BranchTerminal,
)
from .branch_execution import BranchExecutor
from .legacy import LegacyContext, LegacyTranslation
from .native_scores import (
    NATIVE_SCORES_REFERENCE_CODECS,
    restore_scores,
    snapshot_scores,
)
from .store import DataStore, DataclassCodec, JSONCodec
from .types import (
    Assignment,
    Blob,
    Candidate,
    Chemical,
    Key,
    Polymer,
    System,
)


@dataclass(frozen=True)
class RefinementMemberRecord:
    """Restart-safe member payload referenced by ``BranchMember.value``."""

    candidate: Candidate
    scores: Blob
    score: float
    structure_path: str
    metrics: Blob
    label: str
    generation: int
    parent_id: int | None
    proposal_step: int


@dataclass(frozen=True)
class RefinementProposalRecord:
    """Proposal payload. It contains candidate state, never a live model value."""

    candidate: Candidate
    label: str
    generation: int
    parent_id: int
    proposal_index: int
    proposal_step: int
    proposal_seed: int
    protocol: Key


@dataclass(frozen=True)
class NativeMeasurement:
    """Caller-owned scientific interpretation of one native score result."""

    score: float
    metrics: Mapping[str, Any]
    structure_path: str


@dataclass(frozen=True)
class NativeProposal:
    """Caller-owned complete proposal result."""

    candidate: Candidate
    label: str


_MEMBER_CODEC = DataclassCodec(
    RefinementMemberRecord,
    "astrolabe.design.native-refinement-member/v1",
    nested_types=(
        RefinementProposalRecord,
        NativeMeasurement,
        NativeProposal,
        Candidate,
        Assignment,
        Polymer,
        Chemical,
        System,
        Blob,
        Key,
    ),
)
_PROPOSAL_CODEC = DataclassCodec(
    RefinementProposalRecord,
    "astrolabe.design.native-refinement-proposal/v1",
    nested_types=(
        RefinementMemberRecord,
        NativeMeasurement,
        NativeProposal,
        Candidate,
        Assignment,
        Polymer,
        Chemical,
        System,
        Blob,
        Key,
    ),
)
_METRICS_CODEC = JSONCodec("astrolabe.design.native-refinement-metrics/v1")
REFINEMENT_REFERENCE_CODECS = NATIVE_SCORES_REFERENCE_CODECS + (
    _MEMBER_CODEC,
    _PROPOSAL_CODEC,
    _METRICS_CODEC,
)


def _finite_score(value: float) -> float:
    score = float(value)
    if not math.isfinite(score):
        raise ValueError(f"native refinement score must be finite, got {score!r}")
    return score


def _json_metric(value: Any) -> Any:
    """Convert numeric/path metric leaves without importing scientific stacks."""

    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("native refinement metrics must be finite")
        return value
    if isinstance(value, Path):
        return value.as_posix()
    item = getattr(value, "item", None)
    if callable(item):
        scalar = item()
        if scalar is not value:
            return _json_metric(scalar)
    if isinstance(value, Mapping):
        result = {}
        for name, item in value.items():
            if not isinstance(name, str):
                raise TypeError("native refinement metric names must be strings")
            result[name] = _json_metric(item)
        return result
    if isinstance(value, (tuple, list)):
        return [_json_metric(item) for item in value]
    raise TypeError(
        "native refinement metrics must contain JSON scalar/list/mapping values; "
        f"got {type(value).__module__}.{type(value).__qualname__}"
    )


@dataclass(frozen=True)
class NativeRefinementSpec:
    """Model-neutral native owner configuration.

    ``load_mutator`` and ``load_evaluator`` are called once in that order.
    ``propose`` runs with the parent Candidate and restored StandardScores.
    ``evaluate`` runs with the proposed Candidate and must return the existing
    native StandardScores unchanged. ``measure`` is the only scientific scoring
    policy and is supplied by the owning workflow.
    """

    output: Path
    seed: int
    width: int
    proposals_per_parent: int
    max_evaluations: int
    evaluator_alias: str
    mutator_alias: str
    assemblies: tuple[str, ...]
    initial_candidate: Candidate
    context: LegacyContext
    translation: LegacyTranslation
    data: DataStore
    load_mutator: Callable[[], Any]
    load_evaluator: Callable[[], Any]
    propose: Callable[..., NativeProposal]
    evaluate: Callable[..., Any]
    measure: Callable[[Candidate, Any, BranchJob], NativeMeasurement]
    label: Callable[[Candidate], str]
    proposal_seed: Callable[[int, int], int]
    on_event: Callable[[str, Mapping[str, Any]], None] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "output", Path(self.output))
        object.__setattr__(
            self,
            "assemblies",
            tuple(str(name) for name in self.assemblies),
        )
        if type(self.seed) is not int or not 1 <= self.seed <= 2**32 - 1:
            raise ValueError("seed must be an integer in 1..2**32-1")
        for name, value in (
            ("width", self.width),
            ("proposals_per_parent", self.proposals_per_parent),
            ("max_evaluations", self.max_evaluations),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not self.evaluator_alias or not self.mutator_alias:
            raise ValueError("evaluator_alias and mutator_alias must be non-empty")
        if self.evaluator_alias == self.mutator_alias:
            raise ValueError("evaluator and mutator aliases must be distinct")
        if not self.assemblies or len(set(self.assemblies)) != len(self.assemblies):
            raise ValueError("assemblies must be non-empty and unique")
        known = {occurrence.assembly for occurrence in self.translation.system.occurrences}
        unknown = set(self.assemblies) - known
        if unknown:
            raise ValueError(f"unknown assemblies: {sorted(unknown)!r}")
        if self.initial_candidate.system != self.translation.system.identity:
            raise ValueError("initial candidate belongs to another translated System")
        if not isinstance(self.data, DataStore):
            raise TypeError("data must be a design DataStore")
        from .types import make_candidate

        if make_candidate(
            self.translation.system,
            self.initial_candidate.assignments,
        ) != self.initial_candidate:
            raise ValueError("initial candidate identity does not match its state")
        for codec in REFINEMENT_REFERENCE_CODECS:
            self.data.register(codec)

    def event(self, name: str, **fields: Any) -> None:
        if self.on_event is not None:
            self.on_event(name, fields)


class _SerialFuturePool:
    """Concrete Future adapter that intentionally provides no parallelism."""

    def submit(self, worker: Callable[[BranchJob], BranchTerminal], job: BranchJob):
        future = Future()
        try:
            future.set_result(worker(job))
        except BaseException as error:
            future.set_exception(error)
        return future


class NativeRefinementOwner:
    """Serial mutable-model owner used by one generic branching control run."""

    attached = False

    def __init__(self, spec: NativeRefinementSpec) -> None:
        if not isinstance(spec, NativeRefinementSpec):
            raise TypeError("spec must be a NativeRefinementSpec")
        self.spec = spec
        self._mutator = spec.load_mutator()
        try:
            self._evaluator = spec.load_evaluator()
        except BaseException:
            close = getattr(self._mutator, "shutdown", None)
            if close is not None:
                close()
            raise
        self._shutdown = False
        self.member_blobs: dict[int, Blob] = {}
        self.proposal_count = 0
        self.duplicate_count = 0
        self.evaluations = 0
        self.controller_state = None
        self._observed_proposals: set[int] = set()
        self._protocol: Key | None = None
        self._branch_executor = BranchExecutor(
            _SerialFuturePool(),
            worker=self.execute,
        )
        from astrolabe.model_execution import register_model_owner

        register_model_owner(self)

    def _activate(self, candidate: Candidate):
        return self.spec.context.activate(
            candidate,
            translation=self.spec.translation,
            data=self.spec.data,
        )

    def _activate_score_destination(self, identifier: int | str) -> Path:
        destination = (
            self.spec.output
            / ".native_refinement"
            / f"evaluation_{identifier}"
        )
        destination.mkdir(parents=True, exist_ok=True)
        return destination

    @staticmethod
    def _member_blob(job: BranchJob) -> Blob:
        if job.parent is None or job.parent.value.codec != _MEMBER_CODEC.identity:
            raise ValueError("branch job parent is not a native refinement member")
        return job.parent.value

    @staticmethod
    def _proposal_blob(job: BranchJob) -> Blob:
        if (
            job.proposal is None
            or job.proposal.value.codec != _PROPOSAL_CODEC.identity
        ):
            raise ValueError("branch job does not carry a native refinement proposal")
        return job.proposal.value

    def _put_member(
        self,
        *,
        identifier: int,
        candidate: Candidate,
        scores: Any,
        label: str,
        generation: int,
        parent_id: int | None,
        proposal_step: int,
        measurement: NativeMeasurement,
    ) -> BranchMember:
        score_blob = snapshot_scores(scores, self.spec.data)
        metrics_blob = self.spec.data.put(
            _json_metric(dict(measurement.metrics)),
            _METRICS_CODEC,
        )
        record = RefinementMemberRecord(
            candidate=candidate,
            scores=score_blob,
            score=_finite_score(measurement.score),
            structure_path=str(measurement.structure_path),
            metrics=metrics_blob,
            label=label,
            generation=generation,
            parent_id=parent_id,
            proposal_step=proposal_step,
        )
        value = self.spec.data.put(record, _MEMBER_CODEC)
        member = BranchMember(
            id=identifier,
            key=label,
            value=value,
            score=record.score,
            eligible=True,
        )
        self.member_blobs[identifier] = value
        self.evaluations += 1
        return member

    def root_job(self, *, protocol: Key) -> BranchJob:
        """Build the initial evaluation job consumed by ``policy.start``."""

        if not isinstance(protocol, Key):
            raise TypeError("protocol must be a content Key")
        self._protocol = protocol
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
        value = self.spec.data.put(proposal, _PROPOSAL_CODEC)
        return BranchJob(
            phase="evaluate",
            id=0,
            generation=0,
            seed=self.spec.seed,
            proposal=None,
            slot=0,
            source=value,
        )

    def _propose(self, job: BranchJob) -> BranchTerminal:
        if job.parent is None:
            raise ValueError("proposal job requires a parent member")
        parent_blob = self._member_blob(job)
        parent = self.spec.data.get(parent_blob, _MEMBER_CODEC)
        if not isinstance(parent, RefinementMemberRecord):
            raise TypeError("parent member did not decode to a refinement record")
        if parent.score != job.parent.score or parent.label != job.parent.key:
            raise ValueError("parent BranchMember does not match its owner record")
        destination = self._activate_score_destination(f"proposal_{job.id:08d}")
        scores = restore_scores(parent.scores, self.spec.data, destination)
        proposal_seed = self.spec.proposal_seed(self.spec.seed, job.id)
        if type(proposal_seed) is not int or not 0 <= proposal_seed <= 2**32 - 1:
            raise ValueError("proposal seed callback must return a valid uint32 seed")
        if job.seed != proposal_seed:
            raise ValueError("proposal job seed differs from the declared owner schedule")
        if self._protocol is None:
            raise RuntimeError("root_job(protocol=...) must be built before proposals")
        self.spec.event(
            "proposal_started",
            generation=job.generation,
            parent_id=job.parent.id,
            proposal_index=job.slot,
            proposal_step=job.id,
            seed=proposal_seed,
        )
        with self._activate(parent.candidate):
            try:
                proposed = self.spec.propose(
                    self._mutator,
                    parent.candidate,
                    scores,
                    seed=proposal_seed,
                    step=job.id,
                    generation=job.generation,
                    parent_id=job.parent.id,
                    proposal_index=job.slot,
                    output=self.spec.output,
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
            protocol=self._protocol,
        )
        value = self.spec.data.put(record, _PROPOSAL_CODEC)
        self.proposal_count += 1
        self.spec.event(
            "proposal_completed",
            generation=job.generation,
            parent_id=job.parent.id,
            proposal_index=job.slot,
            proposal_step=job.id,
            sequence=proposed.label,
            seed=proposal_seed,
        )
        result = BranchProposal(
            id=job.id,
            parent_id=job.parent.id,
            key=proposed.label,
            value=value,
            valid=True,
        )
        return BranchTerminal(job=job, status="completed", value=result)

    def _evaluate(self, job: BranchJob) -> BranchTerminal:
        if job.seed != self.spec.seed:
            raise ValueError("evaluation job seed differs from the fixed prediction seed")
        proposal_blob = (
            self._proposal_blob(job)
            if job.proposal is not None
            else job.source
        )
        if (
            proposal_blob is None
            or proposal_blob.codec != _PROPOSAL_CODEC.identity
        ):
            raise ValueError("evaluation job does not carry a native proposal")
        proposal = self.spec.data.get(proposal_blob, _PROPOSAL_CODEC)
        if not isinstance(proposal, RefinementProposalRecord):
            raise TypeError("proposal did not decode to a refinement record")
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
        with self._activate(proposal.candidate):
            try:
                scores = self.spec.evaluate(
                    self._evaluator,
                    proposal.candidate,
                    seed=self.spec.seed,
                    step=proposal.proposal_step,
                    assemblies=self.spec.assemblies,
                    output=self.spec.output,
                    proposal_id=job.id,
                )
                measurement = self.spec.measure(proposal.candidate, scores, job)
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
                    error=repr(error),
                )
                raise
        if not isinstance(measurement, NativeMeasurement):
            raise TypeError("measure callback must return NativeMeasurement")
        member = self._put_member(
            identifier=job.id,
            candidate=proposal.candidate,
            scores=scores,
            label=proposal.label,
            generation=proposal.generation,
            parent_id=None if proposal.parent_id < 0 else proposal.parent_id,
            proposal_step=proposal.proposal_step,
            measurement=measurement,
        )
        self.spec.event(
            "evaluation_completed",
            evaluation_id=job.id,
            generation=proposal.generation,
            parent_id=None if proposal.parent_id < 0 else proposal.parent_id,
            proposal_index=proposal.proposal_index,
            proposal_step=proposal.proposal_step,
            sequence=proposal.label,
            seed=self.spec.seed,
            score=member.score,
            structure_path=measurement.structure_path,
        )
        return BranchTerminal(job=job, status="completed", value=member)

    def execute(self, job: BranchJob) -> BranchTerminal:
        """Execute one typed branch job; inference is serialized by construction."""

        if self._shutdown:
            raise RuntimeError("native refinement owner is shut down")
        if not isinstance(job, BranchJob):
            raise TypeError("job must be a BranchJob")
        if job.phase == "propose":
            return self._propose(job)
        if job.phase == "evaluate":
            return self._evaluate(job)
        raise ValueError(f"unsupported branch phase {job.phase!r}")

    def perform(self, action: BranchAction) -> BranchEvent:
        """Serial BranchAction trial with the shared coordinator contract."""

        if not isinstance(action, BranchAction):
            raise TypeError("native refinement trial expects BranchAction")
        return self._branch_executor.perform(action)

    def checkpoint(self, state: Any, *, request: Any) -> None:
        """Observe the generic policy checkpoint without owning its transitions."""

        self.controller_state = state
        duplicates = int(getattr(state, "duplicates", self.duplicate_count))
        self.duplicate_count = duplicates
        admitted = set()
        for job in tuple(getattr(state, "queued", ())) + tuple(
            getattr(state, "pending", ())
        ):
            if job.proposal is not None:
                admitted.add(job.proposal.id)
        for terminal in getattr(state, "batch_results", ()):
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
        result = self.spec.data.get(value.value, _MEMBER_CODEC)
        if not isinstance(result, RefinementMemberRecord):
            raise TypeError("member did not decode to a refinement record")
        return result

    def proposal(self, value: BranchProposal) -> RefinementProposalRecord:
        if value.value.codec != _PROPOSAL_CODEC.identity:
            raise ValueError("proposal is not a native refinement proposal")
        result = self.spec.data.get(value.value, _PROPOSAL_CODEC)
        if not isinstance(result, RefinementProposalRecord):
            raise TypeError("proposal did not decode to a refinement record")
        return result

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        first_error = None
        for owner in (self._evaluator, self._mutator):
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


@dataclass(frozen=True)
class NativeRefinementResult:
    """Policy result plus restart-safe records owned by the native boundary."""

    selected: tuple[BranchMember, ...]
    state: Any
    owner: Any

    @property
    def evaluations(self) -> int:
        if hasattr(self.state, "records"):
            return len(self.state.records)
        return self.owner.evaluations

    @property
    def proposals(self) -> int:
        if hasattr(self.state, "proposals"):
            return self.state.proposals
        return self.owner.proposal_count

    def members(self) -> tuple[RefinementMemberRecord, ...]:
        return tuple(self.owner.member(member) for member in self.selected)

    def member_items(self) -> tuple[tuple[int, RefinementMemberRecord], ...]:
        records = getattr(self.state, "records", None)
        if isinstance(records, tuple):
            return tuple(
                (member.id, self.owner.member(member))
                for member in sorted(records, key=lambda item: item.id)
            )
        return tuple(
            (
                identifier,
                self.owner.member(BranchMember(identifier, "", blob, 0.0)),
            )
            for identifier, blob in sorted(self.owner.member_blobs.items())
        )

    def all_members(self) -> tuple[RefinementMemberRecord, ...]:
        return tuple(record for _, record in self.member_items())

    def metrics(self, member: RefinementMemberRecord) -> dict[str, Any]:
        if not isinstance(member, RefinementMemberRecord):
            raise TypeError("member must be a RefinementMemberRecord")
        result = self.owner.spec.data.get(member.metrics, _METRICS_CODEC)
        if not isinstance(result, dict):
            raise TypeError("refinement metrics did not decode to a mapping")
        return result


def run_native_refinement(
    spec: NativeRefinementSpec,
    *,
    policy: Any,
    protocol: Key,
    runtime: Any = None,
) -> NativeRefinementResult:
    """Run one caller-owned BranchingPolicy through ``ControlRun``.

    The policy remains the sole budget/frontier/duplicate owner.  This function
    only supplies a serial typed trial and the root evaluation job.
    """

    if policy is None:
        raise TypeError("policy is required")
    if not hasattr(policy, "start"):
        raise TypeError("policy must provide start(rootjob)")
    if runtime is not None:
        from .native_refinement_runtime import run_durable_native_refinement

        return run_durable_native_refinement(
            spec,
            policy=policy,
            protocol=protocol,
            runtime=runtime,
        )
    from astrolabe.app import Astrolabe

    from .composition import ControlRun

    owner = NativeRefinementOwner(spec)
    try:
        initial = policy.start(owner.root_job(protocol=protocol))
        final = Astrolabe(ControlRun(policy, owner, initial)).run()
        selected = getattr(final, "frontier", None)
        if not isinstance(selected, tuple):
            raise TypeError("BranchingPolicy.selected must return branching state")
        if any(not isinstance(member, BranchMember) for member in selected):
            raise TypeError("selected branch members must be BranchMember values")
        return NativeRefinementResult(selected, final, owner)
    except BaseException:
        owner.shutdown()
        raise


__all__ = [
    "NativeMeasurement",
    "NativeProposal",
    "NativeRefinementOwner",
    "NativeRefinementResult",
    "NativeRefinementSpec",
    "REFINEMENT_REFERENCE_CODECS",
    "RefinementMemberRecord",
    "RefinementProposalRecord",
    "run_native_refinement",
]
