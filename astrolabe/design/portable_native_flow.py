"""Portable static-stage native composition for the default coordinator.

The flow owns phase order and recovery receipts.  Existing native registry
operations perform prediction and mutation, while an explicitly identified
math owner adapts the current ``Decisions`` implementation without copying its
Metropolis or gate formulas.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Any, Callable

from . import coordinator
from .legacy import LEGACY_REFERENCE_CODECS, standard_scores_to_evidence
from .native_scores import (
    NATIVE_SCORES_ENVELOPE_CODEC,
    NATIVE_SCORES_REFERENCE_CODECS,
    restore_scores,
)
from .native_support import NativeBindings, native_score_artifact, numeric_step
from .operations import Components, operation
from .portable_native_state import (
    DECISION_REQUEST_CODEC,
    DECISION_RESULT_CODEC,
    NATIVE_STEP_STATE_CODEC,
    SCIENTIFIC_RUN_CONTRACT_CODEC,
    DecisionRequest,
    DecisionResult,
)
from .store import DataStore
from .successor_search import AcceptedSuccessorPolicy
from .types import (
    Artifact,
    Assessment,
    Blob,
    Candidate,
    Continuation,
    Dependency,
    Evidence,
    EvidenceUse,
    Key,
    PredictRequest,
    Proposal,
    Proposals,
    ProposeRequest,
    Scored,
    System,
    TrialOutcome,
    Work,
    key_for,
    manifest,
)


class UnsupportedPortableNativeProfile(ValueError):
    """The selected static non-Pose capability set cannot be portable yet."""


def _measured_scores_blob(state):
    matches = tuple(
        item.value
        for item in state.decision_state.evaluation_scores
        if item.step == state.decision_state.last_accepted
    )
    if len(matches) != 1:
        raise ValueError(
            "native decision state requires exactly one measured score snapshot "
            "for its last accepted step"
        )
    return matches[0]


def _deduplicate_artifacts(artifacts):
    by_identity = {}
    for artifact in artifacts:
        if not isinstance(artifact, Artifact):
            raise TypeError("projected Evidence artifacts must be Artifact values")
        previous = by_identity.get(artifact.identity.digest)
        if previous is not None and previous != artifact:
            raise ValueError(
                f"projected Evidence has conflicting artifact {artifact.identity.digest!r}"
            )
        by_identity[artifact.identity.digest] = artifact
    return tuple(by_identity.values())


@dataclass(frozen=True)
class PortableNativeSupport:
    """Config-derived capabilities checked before any model owner is built."""

    stage_keys: tuple[str, ...]
    transition_gates: bool = False
    structure_start: bool = False
    live_pose: bool = False
    model_owned_mutable_state: bool = False
    unsupported_score_extensions: bool = False

    def __post_init__(self) -> None:
        if not self.stage_keys or any(
            not isinstance(key, str) or not key for key in self.stage_keys
        ):
            raise ValueError("stage_keys must contain non-empty strings")
        if len(set(self.stage_keys)) != len(self.stage_keys):
            raise ValueError("stage_keys must be unique")
        for name in (
            "transition_gates",
            "structure_start",
            "live_pose",
            "model_owned_mutable_state",
            "unsupported_score_extensions",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be boolean")


def validate_static_nonpose_support(support: PortableNativeSupport) -> Key:
    """Return the static stage identity or fail with every unsupported reason."""
    if not isinstance(support, PortableNativeSupport):
        raise TypeError("support must be PortableNativeSupport")
    reasons = []
    if len(support.stage_keys) != 1:
        reasons.append("exactly one effective static stage is required")
    if support.transition_gates:
        reasons.append("stage transition gates are not portable yet")
    if support.structure_start:
        reasons.append("structure_start/reference transactions are not portable")
    if support.live_pose:
        reasons.append("live or complex Pose state has no lossless codec")
    if support.model_owned_mutable_state:
        reasons.append("model-owned mutable scientific state has no codec")
    if support.unsupported_score_extensions:
        reasons.append("selected native score extensions are not portable")
    if reasons:
        raise UnsupportedPortableNativeProfile("; ".join(reasons))
    return key_for(("astrolabe.portable-native.static-stage/v1", support.stage_keys[0]))


@dataclass(frozen=True)
class PortableNativeMath:
    """Explicit adapter to the current native measurement/decision math.

    Callbacks receive restored ``StandardScores`` and the active immutable
    before-state. They may create a new decision state, but must not retain a
    model, process resource, or mutable reference outside its returned DTO.
    """

    identity: Key
    finalize_root: Callable[
        [Any, Any, DataStore, Path],
        Any,
    ]
    decide_step: Callable[
        [DecisionRequest, Any, DataStore, Path],
        DecisionResult,
    ]
    requires_mutation_context: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.identity, Key):
            raise TypeError("math identity must be a content Key")
        if not callable(self.finalize_root) or not callable(self.decide_step):
            raise TypeError("portable native math callbacks must be callable")
        if type(self.requires_mutation_context) is not bool:
            raise TypeError("requires_mutation_context must be boolean")


class PortableNativeFlow:
    """One bounded Trial for a single non-Pose static native stage.

    Search remains in ``AcceptedSuccessorPolicy`` and control remains in the
    common coordinator. Every mutually dependent interval that can consume RNG
    or invoke a native owner is a named operation receipt.
    """

    fail_fast = True

    def __init__(
        self,
        *,
        contract: Any,
        support: PortableNativeSupport,
        system: System,
        initial_candidate: Candidate,
        evaluator: Any,
        mutator: Any,
        bindings: NativeBindings,
        sampling_policy: Any,
        assemblies: tuple[str, ...],
        math: PortableNativeMath,
        runtime_context: Any,
        sample_protocol: Key,
        initialization_order: tuple[str, str] = ("mutator", "evaluator"),
    ) -> None:
        from .portable_native_state import (
            NATIVE_RUNTIME_CONTEXT_CODEC,
            SCIENTIFIC_CONTRACT_SCHEMA,
            SCIENTIFIC_RUN_CONTRACT_CODEC,
            ScientificRunContract,
        )

        if not isinstance(contract, ScientificRunContract):
            raise TypeError("contract must be ScientificRunContract")
        if contract.schema != SCIENTIFIC_CONTRACT_SCHEMA:
            raise ValueError("portable native contract has another schema")
        if not isinstance(bindings, NativeBindings):
            raise TypeError("bindings must be NativeBindings")
        if not isinstance(bindings.data, DataStore):
            raise TypeError("bindings.data must be a design DataStore")
        if not isinstance(system, System) or system != bindings.translation.system:
            raise ValueError("portable native flow system differs from native bindings")
        if initial_candidate.system != system.identity:
            raise ValueError("initial candidate belongs to another System")
        if evaluator.descriptor.request_type is not PredictRequest or (
            evaluator.descriptor.result_type is not Evidence
        ):
            raise TypeError("evaluator operation must produce Evidence")
        if mutator.descriptor.request_type is not ProposeRequest or (
            mutator.descriptor.result_type is not Proposals
        ):
            raise TypeError("mutator operation must produce Proposals")
        if (
            bindings.data is not evaluator.bindings.data
            or bindings.data is not mutator.bindings.data
        ):
            raise ValueError("prediction, mutation and flow must share one DataStore")
        if not isinstance(sampling_policy, type(mutator.sampling_policy)):
            raise TypeError("sampling policy has an unexpected type")
        if sampling_policy != mutator.sampling_policy:
            raise ValueError("proposal sampling identity differs from the native mutator")
        normalized_assemblies = tuple(str(name) for name in assemblies)
        if (
            not normalized_assemblies
            or len(set(normalized_assemblies)) != len(normalized_assemblies)
        ):
            raise ValueError("assemblies must be non-empty and unique")
        known = {occurrence.assembly for occurrence in system.occurrences}
        if set(normalized_assemblies) - known:
            raise ValueError(
                f"unknown native assemblies: {sorted(set(normalized_assemblies) - known)!r}"
            )
        if initialization_order not in {
            ("evaluator", "mutator"),
            ("mutator", "evaluator"),
        }:
            raise ValueError("initialization_order must initialize both owners once")
        if evaluator.initialization is None or mutator.initialization is None:
            raise TypeError("evaluator and mutator must expose explicit initialization operations")
        if not isinstance(sample_protocol, Key):
            raise TypeError("sample_protocol must be a content Key")

        static_stage = validate_static_nonpose_support(support)
        if contract.static_stage != static_stage:
            raise ValueError("scientific contract static stage differs from config capabilities")
        if contract.system_identity != system.identity:
            raise ValueError("scientific contract belongs to another molecular System")
        if type(contract.seed) is not int or not 0 <= contract.seed <= 2**32 - 1:
            raise ValueError("contract seed must be an integer in 0..2**32-1")
        missing_components = {
            evaluator.descriptor.identity,
            mutator.descriptor.identity,
        } - set(contract.component_identities)
        if missing_components:
            raise ValueError(
                "scientific contract omits selected native component identities"
            )

        self.contract = contract
        self.support = support
        self.system = system
        self.initial = initial_candidate
        self.evaluator = evaluator
        self.mutator = mutator
        self.bindings = bindings
        self.data = bindings.data
        self.sampling_policy = sampling_policy
        self.assemblies = normalized_assemblies
        self.math = math
        self.runtime_context = runtime_context
        self.sample_protocol = sample_protocol
        self.initialization_order = initialization_order
        if (
            not isinstance(runtime_context, Blob)
            or runtime_context.codec != NATIVE_RUNTIME_CONTEXT_CODEC.identity
        ):
            raise TypeError("runtime_context must use the native runtime-context codec")
        self.data.register(NATIVE_RUNTIME_CONTEXT_CODEC)

        self._normalize = operation(
            "portable/root-finalize",
            self._root_request_type(),
            self._root_result_type(),
            self._root_finalize,
            version="portable-native-root-finalize/v1",
            request_codec=self._root_request_codec(),
            result_codec=self._root_result_codec(),
            model_identity=math.identity,
            reference_codecs=(
                LEGACY_REFERENCE_CODECS
                + NATIVE_SCORES_REFERENCE_CODECS
                + (self._numeric_tree_codec(),)
            ),
        )
        self._decide = operation(
            "portable/step-decide",
            DecisionRequest,
            DecisionResult,
            self._step_decide,
            version="portable-native-step-decide/v1",
            request_codec=DECISION_REQUEST_CODEC,
            result_codec=DECISION_RESULT_CODEC,
            model_identity=math.identity,
            reference_codecs=(
                LEGACY_REFERENCE_CODECS
                + NATIVE_SCORES_REFERENCE_CODECS
                + (self._numeric_tree_codec(),)
            ),
        )
        self._initializers = {
            "evaluator": (evaluator.initialization, evaluator.initialization_request),
            "mutator": (mutator.initialization, mutator.initialization_request),
        }
        operations = {
            evaluator.descriptor.alias: evaluator,
            mutator.descriptor.alias: mutator,
            self._normalize.descriptor.alias: self._normalize,
            self._decide.descriptor.alias: self._decide,
        }
        if len(operations) != 4:
            raise ValueError("portable native operation aliases must be distinct")
        self.components = Components(
            operations,
            sampling={mutator.descriptor.alias: sampling_policy},
        )
        self.operations = tuple(self.components)
        self.requirements = self.components.requirements(tuple(operations))
        self._support_validated = True

    @staticmethod
    def _root_request_type():
        from .portable_native_state import RootBootstrapRequest

        return RootBootstrapRequest

    @staticmethod
    def _root_result_type():
        from .portable_native_state import NativeStepState

        return NativeStepState

    @staticmethod
    def _root_request_codec():
        from .portable_native_state import ROOT_BOOTSTRAP_REQUEST_CODEC

        return ROOT_BOOTSTRAP_REQUEST_CODEC

    @staticmethod
    def _root_result_codec():
        from .portable_native_state import NATIVE_STEP_STATE_CODEC

        return NATIVE_STEP_STATE_CODEC

    @staticmethod
    def _numeric_tree_codec():
        from .portable_native_state import NUMERIC_TREE_CODEC

        return NUMERIC_TREE_CODEC

    def _activate(self, candidate: Candidate):
        return self.bindings.activate(candidate)

    def _state_name(self, candidate: Candidate) -> str:
        return "portable/state/" + candidate.identity.digest

    def _save_state(self, runtime, state: Any):
        if state.candidate.system != self.system.identity:
            raise ValueError("portable native state belongs to another System")
        if state.static_stage != self.contract.static_stage:
            raise ValueError("portable native state changed its static stage")
        return runtime.save(self._state_name(state.candidate), state, NATIVE_STEP_STATE_CODEC)

    @staticmethod
    def _state_blob(scored: Scored):
        dependencies = tuple(
            dependency
            for dependency in scored.assessment.inputs.dependencies
            if dependency.role == "native-state"
        )
        if len(dependencies) != 1:
            raise ValueError(
                "portable native Scored must reference exactly one native-state input"
            )
        from .types import Blob

        return Blob(
            dependencies[0].identity,
            NATIVE_STEP_STATE_CODEC.identity,
        )

    def _load_state(self, parent: Scored, runtime):
        from .runtime import ContractConflict

        if runtime.load("portable/contract", SCIENTIFIC_RUN_CONTRACT_CODEC) != (
            self.contract
        ):
            raise ContractConflict(
                "portable native Runtime contract differs from the active flow"
            )
        state_blob = self._state_blob(parent)
        try:
            state = runtime.data.get(state_blob, NATIVE_STEP_STATE_CODEC)
        except KeyError as error:
            raise ValueError(
                "portable native predecessor references unavailable state bytes"
            ) from error
        if state.candidate != parent.candidate:
            raise ValueError("portable native state belongs to another candidate")
        if state.static_stage != self.contract.static_stage:
            raise ValueError("portable native state belongs to another static stage")
        return state

    def bootstrap(self, runtime) -> None:
        """Validate support, persist the contract, then replay explicit readiness."""
        validate_static_nonpose_support(self.support)
        if runtime.data is not self.data:
            raise ValueError("portable native flow and Runtime must share one DataStore")
        contract_codec = SCIENTIFIC_RUN_CONTRACT_CODEC
        if runtime.load("portable/contract", contract_codec) is None:
            runtime.save("portable/contract", self.contract, contract_codec)
        elif runtime.load("portable/contract", contract_codec) != self.contract:
            from .runtime import ContractConflict

            raise ContractConflict("Portable native scientific contract changed on resume")
        for index, name in enumerate(self.initialization_order):
            component, request = self._initializers[name]
            runtime.call(
                f"bootstrap/{index}/{name}",
                component,
                request,
            )
        evaluator_ready = self.evaluator.owner.ready
        mutator_ready = self.mutator.owner.ready
        if not evaluator_ready or not mutator_ready:
            raise RuntimeError("Portable native model readiness was not rebound after initialization")

    def _root_finalize(self, request):
        scores = self.bindings.scores_for(
            request.evidence,
            self.bindings.work_root / "portable" / "root" / "scores",
        )
        with self._activate(request.candidate):
            state = self.math.finalize_root(
                request,
                scores,
                self.data,
                self.bindings.work_root / "portable" / "root",
            )
        if not isinstance(state, self._root_result_type()):
            raise TypeError("finalize_root must return NativeStepState")
        if state.candidate != request.candidate:
            raise ValueError("finalize_root returned another candidate")
        if state.next_step != 1:
            raise ValueError("root NativeStepState.next_step must be 1")
        return state

    def _step_decide(self, request: DecisionRequest):
        scores = restore_scores(
            request.proposed_scores,
            self.data,
            self.bindings.work_root
            / "portable"
            / f"step_{request.step:08d}"
            / "scores",
        )
        # ``Decisions.make_mutation_decision`` owns apply/clear of pending
        # legacy mutations. Restore the accepted predecessor so its existing
        # accept and reject branches retain their original sequence semantics.
        with self._activate(request.before.candidate):
            result = self.math.decide_step(
                request,
                scores,
                self.data,
                self.bindings.work_root / "portable" / f"step_{request.step:08d}",
            )
        if not isinstance(result, DecisionResult):
            raise TypeError("decide_step must return DecisionResult")
        return result

    def _measured_evidence(self, state, state_blob, source_evidence):
        source_snapshot = native_score_artifact(source_evidence)
        destination = (
            self.bindings.work_root
            / "portable"
            / "measured"
            / (
                f"step_{state.next_step - 1:08d}_"
                f"{state.current_scores.identity.digest[:16]}"
            )
        )
        scores = restore_scores(state.current_scores, self.data, destination)
        translated = standard_scores_to_evidence(
            scores,
            self.system,
            state.candidate,
            self.data,
            samples=source_snapshot.sample,
            state="portable-native/measured",
            producer_label="portable-native.measured",
        )
        inputs = manifest(
            tuple(translated.inputs.dependencies)
            + (
                Dependency("source-evidence", source_evidence.identity),
                Dependency("source-native-scores", source_snapshot.identity),
                Dependency("native-state", state_blob.identity),
                Dependency("current-scores", state.current_scores.identity),
                Dependency("objective", self.contract.objective),
            )
        )
        score_artifact = Artifact(
            identity=key_for(
                (
                    "artifact/native-scores-measured/v1",
                    state.candidate.identity,
                    source_snapshot.occurrences,
                    source_snapshot.sample,
                    state.current_scores,
                    source_evidence.identity,
                    state_blob.identity,
                    inputs,
                )
            ),
            candidate=state.candidate.identity,
            occurrences=source_snapshot.occurrences,
            sample=source_snapshot.sample,
            content=state.current_scores,
            derived_from=tuple(
                dict.fromkeys(
                    (source_snapshot.identity,) + source_snapshot.derived_from
                )
            ),
            inputs=inputs,
        )
        source_artifacts = tuple(
            artifact
            for artifact in source_evidence.artifacts
            if artifact.content.codec != NATIVE_SCORES_ENVELOPE_CODEC.identity
        )
        evidence = Evidence(
            identity=key_for(("evidence-pending/v1",)),
            evaluated_candidate=state.candidate.identity,
            artifacts=_deduplicate_artifacts(
                tuple(translated.artifacts)
                + source_artifacts
                + (score_artifact,)
            ),
            measurements=(),
            inputs=inputs,
        )
        return evidence.with_measurements(translated.measurements)

    def _project(self, state, state_blob, source_evidence, runtime):
        if source_evidence.evaluated_candidate != state.candidate.identity:
            raise ValueError("portable successor evidence belongs to another candidate")
        evidence = self._measured_evidence(state, state_blob, source_evidence)
        scores = {item.step: item.value for item in state.decision_state.scores}
        try:
            value = scores[state.decision_state.last_accepted]
        except KeyError as error:
            raise ValueError("decision state has no last accepted score") from error
        inputs = manifest(
            (
                Dependency("evidence", evidence.identity),
                Dependency("objective", self.contract.objective),
                Dependency("history", runtime.history.revision),
                Dependency("native-state", state_blob.identity),
            )
        )
        return Scored(
            candidate=state.candidate,
            evidence=evidence,
            assessment=Assessment(
                evidence=evidence.identity,
                objective_revision=self.contract.objective,
                history_revision=runtime.history.revision,
                values=(value,),
                gates=(),
                inputs=inputs,
            ),
        )

    def root(self, candidate: Candidate, runtime):
        if candidate != self.initial:
            raise ValueError("PortableNativeFlow.root received another initial candidate")
        sample = self._sample(index="0", replicate="0")
        evidence = runtime.call(
            "root/raw_predict",
            self.evaluator,
            PredictRequest(
                system=self.system,
                candidate=candidate,
                assemblies=self.assemblies,
                conditioning=(),
                sources=(),
                samples=(sample,),
            ),
        )
        score_blob = native_score_artifact(evidence).content
        request = self._root_request_type()(
            candidate=candidate,
            evidence=evidence,
            raw_scores=score_blob,
            runtime_context=self.runtime_context,
        )
        state = runtime.call("root/finalize_state", self._normalize, request)
        state_blob = self._save_state(runtime, state)
        return self._project(state, state_blob, evidence, runtime)

    def _sample(self, *, index: str, replicate: str):
        from .types import Sample

        return Sample(
            protocol=self.sample_protocol,
            replicate=replicate,
            seed=self.contract.seed,
            index=index,
        )

    def perform(self, work: Work, runtime):
        if work.stage != "advance" or len(work.parents) != 1:
            raise ValueError("Portable native work requires one accepted predecessor")
        parent = work.parents[0]
        before = self._load_state(parent, runtime)
        if before.candidate != parent.candidate:
            raise ValueError("portable predecessor state differs from Work")
        step = numeric_step(work.sample)
        if step != before.next_step:
            raise ValueError("native Sample.index differs from NativeStepState.next_step")
        if work.sample.seed != self.contract.seed:
            raise ValueError("native Sample.seed differs from the scientific contract")

        proposal_request = ProposeRequest(
            system=self.system,
            parents=(parent.candidate,),
            working=parent.candidate,
            evidence=(
                EvidenceUse(
                    evidence=parent.evidence,
                    relation="current",
                    working_candidate=parent.candidate.identity,
                ),
            ),
            selection=(),
            packed=None,
            sampling=self.sampling_policy,
            sample=work.sample,
        )
        proposals = runtime.call(
            f"step/{step}/propose",
            self.mutator,
            proposal_request,
        )
        if len(proposals.items) != 1 or not isinstance(proposals.items[0], Proposal):
            raise ValueError("portable static profile requires exactly one native Proposal")
        proposal = proposals.items[0]
        if proposal.candidate.system != self.system.identity:
            raise ValueError("native Proposal returned a candidate from another System")
        if self.math.requires_mutation_context:
            from .portable_native_math import validate_mutation_context_artifacts

            validate_mutation_context_artifacts(
                proposal,
                before=before,
                sample=work.sample,
                step=step,
                policy=None,
                data=self.data,
                system=self.system,
                contract_seed=self.contract.seed,
            )

        evidence = runtime.call(
            f"step/{step}/raw_predict",
            self.evaluator,
            PredictRequest(
                system=self.system,
                candidate=proposal.candidate,
                assemblies=self.assemblies,
                conditioning=(),
                sources=(),
                samples=(work.sample,),
            ),
        )
        score_blob = native_score_artifact(evidence).content
        decision_request = DecisionRequest(
            before=before,
            proposal=proposal,
            proposed_scores=score_blob,
            proposal_context=before.runtime_context,
            step=step,
        )
        result = runtime.call(
            f"step/{step}/decide",
            self._decide,
            decision_request,
        )
        if not isinstance(result, DecisionResult):
            raise TypeError("portable decision operation returned another result type")
        after = result.after
        expected_candidate = (
            proposal.candidate if result.accepted else before.candidate
        )
        if after.candidate != expected_candidate:
            raise ValueError("DecisionResult candidate contradicts its accepted flag")
        if after.next_step != step + 1:
            raise ValueError("DecisionResult did not advance NativeStepState.next_step")
        if after.static_stage != self.contract.static_stage:
            raise ValueError("DecisionResult changed the static stage identity")
        scores = {item.step: item.value for item in after.decision_state.scores}
        expected_score_steps = tuple(
            item.step for item in before.decision_state.scores
        ) + (step,)
        if tuple(item.step for item in after.decision_state.scores) != expected_score_steps:
            raise ValueError("DecisionResult omitted the evaluated step")
        if result.accepted:
            if after.decision_state.last_accepted != step:
                raise ValueError("accepted DecisionResult did not update last_accepted")
            if after.decision_state.accepted_steps != (
                before.decision_state.accepted_steps + (step,)
            ):
                raise ValueError("accepted DecisionResult has invalid accepted history")
            if after.current_scores != _measured_scores_blob(after):
                raise ValueError(
                    "accepted DecisionResult did not retain its measured scores"
                )
            successor_evidence = evidence
        else:
            if after.decision_state.last_accepted != before.decision_state.last_accepted:
                raise ValueError("rejected DecisionResult changed last_accepted")
            if after.decision_state.accepted_steps != before.decision_state.accepted_steps:
                raise ValueError("rejected DecisionResult changed accepted history")
            if after.current_scores != before.current_scores:
                raise ValueError("rejected DecisionResult changed accepted scores")
            successor_evidence = parent.evidence

        state_blob = self._save_state(runtime, after)
        successor = self._project(
            after,
            state_blob,
            successor_evidence,
            runtime,
        )
        return TrialOutcome(
            work=work.identity,
            status="completed",
            proposals=(proposal,),
            children=(),
            attempts=(),
            continuation=Continuation(successor, result.stop_reason),
        )

    @property
    def identity(self) -> Key:
        sources = []
        for callback in (self.math.finalize_root, self.math.decide_step):
            try:
                sources.append(inspect.getsource(callback))
            except (OSError, TypeError):
                sources.append(
                    f"{getattr(callback, '__module__', type(callback).__module__)}."
                    f"{getattr(callback, '__qualname__', type(callback).__qualname__)}"
                )
        return key_for(
            (
                "astrolabe.portable-native-flow/v1",
                self.contract,
                self.math.identity,
                tuple(component.descriptor.identity for component in self.operations),
                self.initialization_order,
                sources,
            )
        )


def run_portable_native(
    flow: PortableNativeFlow,
    *,
    rounds: int,
    runtime=None,
):
    """Run the portable Trial only through the common coordinator."""
    if not isinstance(flow, PortableNativeFlow):
        raise TypeError("flow must be PortableNativeFlow")
    if type(rounds) is not int or rounds < 0:
        raise ValueError("rounds must be a nonnegative integer")
    from .runtime import Runtime

    owns_runtime = runtime is None
    if runtime is None:
        runtime = Runtime(
            flow.operations,
            data=flow.data,
            scientific_contract=key_for(
                ("astrolabe.portable-native-runtime/v1", flow.identity)
            ),
        )
    elif runtime.data is not flow.data:
        raise ValueError("portable native flow and Runtime must share one DataStore")
    try:
        return coordinator.run(
            flow.system,
            flow.initial,
            flow,
            AcceptedSuccessorPolicy(rounds=rounds, seed=flow.contract.seed),
            runtime,
        )
    finally:
        if owns_runtime:
            runtime.close()


__all__ = [
    "PortableNativeFlow",
    "PortableNativeMath",
    "PortableNativeSupport",
    "UnsupportedPortableNativeProfile",
    "run_portable_native",
    "validate_static_nonpose_support",
]
