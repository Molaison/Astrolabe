"""Shared ``Decisions`` adapter for the static portable-native profile.

The adapter delegates root initialization and evaluated-step decisions to the
same ``run.py`` helpers used by the legacy loop.  It restores only explicit
typed mutation artifacts and never reconstructs pending mutations by comparing
candidate sequences.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import inspect
from pathlib import Path
from types import ModuleType
from typing import Any

from astrolabe.core import global_config
from astrolabe.score_contributions import ScoreContribution

from .native_mutation_state import (
    NATIVE_MUTATION_CONTEXT_CODEC,
    NATIVE_MUTATION_CONTEXT_SCHEMA,
    MutationContextPolicy,
    NativeMutationResultState,
    NativePendingMutation,
    NativeScoreContribution,
    native_mutation_segments,
    restore_native_mutation_context,
)
from .native_registry import NativeOperationHooks
from .portable_native_flow import PortableNativeMath
from .portable_native_state import (
    NATIVE_STEP_STATE_SCHEMA,
    DecisionRequest,
    DecisionResult,
    NativeStepState,
    snapshot_decision_state,
    store_numeric_tree,
)
from .runtime import ContractConflict
from .store import ArrayCodec, DataStore, JSONCodec
from .types import Artifact, Candidate, Key, Polymer, key_for


PORTABLE_NATIVE_MATH_SCHEMA = "astrolabe.portable-native.shared-decisions/v1"
PORTABLE_NATIVE_HOOKS_SCHEMA = "astrolabe.portable-native.default-hooks/v1"
PORTABLE_NATIVE_MUTATION_CONTEXT_POLICY = MutationContextPolicy(
    "astrolabe.portable-native.mutation-context/v1",
    NATIVE_MUTATION_CONTEXT_SCHEMA,
)
_ARRAY_CODEC = ArrayCodec()
_SELECTOR_CONTEXT_CODEC = JSONCodec(
    "astrolabe.design.native-mutation.selector-contexts/v1"
)
_CONTRIBUTION_JSON_CODEC = JSONCodec(
    "astrolabe.design.native-mutation.score-contribution-json/v1"
)


class UnsupportedPortableNativeMath(ValueError):
    """The selected native proposal lacks exact durable decision state."""


@dataclass(frozen=True)
class PortableNativeHookContext:
    """Frozen inputs for the shared evaluator and mutator hook callbacks."""

    args: Any
    run_module: Any
    loss_computer: Any
    stage_manager: Any
    evaluator_alias: str
    mutator_alias: str
    config_identity: Key
    source_identity: Key

    def __post_init__(self) -> None:
        if not isinstance(self.run_module, ModuleType):
            raise TypeError("run_module must be a module")
        for name in ("evaluator_alias", "mutator_alias"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("config_identity", "source_identity"):
            value = getattr(self, name)
            if not isinstance(value, Key):
                raise TypeError(f"{name} must be a content Key")


def _callable_source(value: Any) -> str:
    try:
        return inspect.getsource(value)
    except (OSError, TypeError):
        return (
            f"{getattr(value, '__module__', type(value).__module__)}."
            f"{getattr(value, '__qualname__', type(value).__qualname__)}"
        )


def _require_helpers(run_module: Any) -> tuple[Any, Any]:
    initialize = getattr(run_module, "_initialize_optimization_state", None)
    decide = getattr(run_module, "_decide_evaluated_optimization_state", None)
    if not callable(initialize) or not callable(decide):
        raise UnsupportedPortableNativeMath(
            "portable native requires run._initialize_optimization_state and "
            "run._decide_evaluated_optimization_state"
        )
    return initialize, decide


def _candidate_sequence(candidate: Candidate, variable: str) -> str:
    matches = [
        assignment.value
        for assignment in candidate.assignments
        if assignment.variable == variable
    ]
    if len(matches) != 1 or not isinstance(matches[0], Polymer):
        raise ContractConflict(
            f"candidate does not contain one polymer variable {variable!r}"
        )
    return "".join(matches[0].tokens)


def _validate_nonpolymer_assignments_unchanged(
    base_candidate: Candidate,
    proposed_candidate: Candidate,
) -> None:
    """Keep chemicals and every other non-polymer assignment bit-exact."""

    base_values = {
        assignment.variable: assignment.value
        for assignment in base_candidate.assignments
    }
    proposed_values = {
        assignment.variable: assignment.value
        for assignment in proposed_candidate.assignments
    }
    if set(base_values) != set(proposed_values):
        raise ContractConflict(
            "mutation proposal changed non-polymer assignment coverage"
        )
    for variable, value in base_values.items():
        if isinstance(value, Polymer):
            continue
        if proposed_values[variable] != value:
            raise ContractConflict(
                f"mutation proposal changed non-polymer assignment {variable!r}"
            )


def _native_mutation_segments(
    system: Any | None = None,
    candidate: Candidate | None = None,
) -> tuple[tuple[str, Any], ...]:
    """Return the legacy owner's stable, shared-alias-deduplicated polymers."""

    segments = tuple(native_mutation_segments())
    if not segments:
        raise UnsupportedPortableNativeMath(
            "native mutation topology contains no polymer segments"
        )
    names = []
    object_ids = []
    for item in segments:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError(
                "native_mutation_segments must return (primary_name, segment) tuples"
            )
        name, segment = item
        if not isinstance(name, str) or not name:
            raise ContractConflict(
                "native mutation topology contains an invalid primary name"
            )
        names.append(name)
        object_ids.append(id(segment))
    if len(names) != len(set(names)) or len(object_ids) != len(set(object_ids)):
        raise ContractConflict(
            "native mutation topology contains duplicate segment coverage"
        )
    if system is not None:
        all_occurrence_names = tuple(
            dict.fromkeys(
                occurrence.variable for occurrence in system.occurrences
            )
        )
        if candidate is None:
            polymer_occurrence_names = all_occurrence_names
        else:
            assignment_kinds = {
                assignment.variable: assignment.value
                for assignment in candidate.assignments
            }
            polymer_occurrence_names = tuple(
                name
                for name in all_occurrence_names
                if isinstance(assignment_kinds.get(name), Polymer)
            )
        if not polymer_occurrence_names:
            raise UnsupportedPortableNativeMath(
                "selected molecular System has no polymer occurrences"
            )
        segment_names = tuple(name for name, _segment in segments)
        missing_segments = tuple(
            name for name in polymer_occurrence_names if name not in segment_names
        )
        unmatched_segments = tuple(
            name for name in segment_names if name not in polymer_occurrence_names
        )
        if missing_segments:
            raise UnsupportedPortableNativeMath(
                "selected polymer occurrences lack native mutation contexts: "
                f"{missing_segments!r}"
            )
        if unmatched_segments:
            raise UnsupportedPortableNativeMath(
                "native polymer segments lack selected physical polymer "
                f"occurrences: {unmatched_segments!r}"
            )
    return segments


def validate_native_mutation_topology(
    *,
    system: Any | None = None,
    candidate: Candidate | None = None,
) -> tuple[tuple[str, Any], ...]:
    """Model-free preflight for exact legacy polymer-segment coverage."""

    return _native_mutation_segments(system, candidate)


def _mutation_context_artifacts(proposal: Any) -> tuple[Artifact, ...]:
    artifacts = tuple(
        artifact
        for artifact in proposal.artifacts
        if artifact.content.codec == NATIVE_MUTATION_CONTEXT_CODEC.identity
    )
    if not artifacts:
        raise UnsupportedPortableNativeMath(
            "portable native decisions require typed mutation-context Proposal "
            "artifacts; sequence differences are not a recovery source"
        )
    identities = tuple(artifact.identity.digest for artifact in artifacts)
    if len(identities) != len(set(identities)):
        raise ContractConflict(
            "Proposal contains duplicate mutation-context artifacts"
        )
    return artifacts


def _validate_mutation_state(
    state: NativeMutationResultState,
    *,
    artifact: Artifact,
    base_candidate: Candidate,
    proposed_candidate: Candidate,
    sample: Any | None,
    step: int,
    policy: MutationContextPolicy,
    data: DataStore,
    system: Any,
    contract_seed: int,
) -> None:
    if state.schema != policy.identity:
        raise ContractConflict(
            "mutation-context artifact policy differs from the selected mutator"
        )
    if state.pre_update_snapshot is not None:
        raise UnsupportedPortableNativeMath(
            "typed mutation decisions reject mutator stack pre-update snapshots"
        )
    if state.base_candidate != base_candidate.identity:
        raise ContractConflict(
            "mutation-context artifact base differs from the accepted predecessor"
        )
    if state.proposed_candidate != proposed_candidate.identity:
        raise ContractConflict(
            "mutation-context artifact proposal differs from the typed Proposal"
        )
    if artifact.derived_from != (
        base_candidate.identity,
        proposed_candidate.identity,
    ):
        raise ContractConflict(
            "mutation-context artifact derived_from does not bind both candidates"
        )
    if artifact.candidate != proposed_candidate.identity:
        raise ContractConflict(
            "mutation-context artifact belongs to another proposed candidate"
        )
    if sample is not None and artifact.sample != sample:
        raise ContractConflict(
            "mutation-context artifact sample differs from the proposal Work"
        )
    if artifact.sample.seed != contract_seed:
        raise ContractConflict(
            "mutation-context artifact sample seed differs from the contract"
        )
    if (
        not artifact.sample.index.isdecimal()
        or int(artifact.sample.index) != step
    ):
        raise ContractConflict(
            "mutation-context artifact sample index differs from the decision step"
        )
    if len(state.sequence_before_pending) != len(state.working_sequence):
        raise ContractConflict(
            "mutation-context artifact changed its polymer sequence length"
        )
    if _candidate_sequence(base_candidate, state.variable) != (
        state.sequence_before_pending
    ):
        raise ContractConflict(
            "accepted predecessor sequence differs from mutation-context input"
        )
    if _candidate_sequence(proposed_candidate, state.variable) != (
        state.working_sequence
    ):
        raise ContractConflict(
            "proposed candidate sequence differs from mutation-context working sequence"
        )

    if not isinstance(state.pending_mutations, tuple):
        raise ContractConflict(
            "mutation-context pending mutations must be an ordered tuple"
        )
    working = list(state.sequence_before_pending)
    seen: set[int] = set()
    for mutation in state.pending_mutations:
        if not isinstance(mutation, NativePendingMutation):
            raise TypeError("pending mutation has an unexpected type")
        if mutation.position in seen:
            raise ContractConflict(
                f"mutation-context repeats pending position {mutation.position}"
            )
        seen.add(mutation.position)
        if mutation.position >= len(working):
            raise ContractConflict(
                "mutation-context pending position is outside the candidate sequence"
            )
        if working[mutation.position] != mutation.from_token:
            raise ContractConflict(
                "mutation-context pending source token differs from accepted sequence"
            )
        working[mutation.position] = mutation.to_token
    if "".join(working) != state.working_sequence:
        raise ContractConflict(
            "mutation-context pending tuple order does not produce its working sequence"
        )

    expected_occurrences = tuple(
        occurrence
        for occurrence in system.occurrences
        if occurrence.variable == state.variable
    )
    if artifact.occurrences != expected_occurrences:
        raise ContractConflict(
            "mutation-context artifact occurrences differ from the molecular System"
        )

    # Ensure every referenced selector/contribution payload is readable before
    # applying anything to the legacy owner.
    if state.selector_contexts.codec != _SELECTOR_CONTEXT_CODEC.identity:
        raise ContractConflict(
            "mutation-context selector payload uses an unsupported codec"
        )
    data.get(state.selector_contexts, _SELECTOR_CONTEXT_CODEC)
    tuple(_restore_score_contributions(data, state.score_contributions))


def validate_mutation_context_artifacts(
    proposal: Any,
    *,
    before: NativeStepState,
    sample: Any | None,
    step: int,
    policy: MutationContextPolicy | None,
    data: DataStore,
    system: Any,
    contract_seed: int,
) -> tuple[NativeMutationResultState, ...]:
    """Validate complete plural coverage without mutating legacy owner state."""

    _validate_nonpolymer_assignments_unchanged(
        before.candidate,
        proposal.candidate,
    )
    artifacts = _mutation_context_artifacts(proposal)
    expected_segments = _native_mutation_segments(system, before.candidate)
    expected_names = tuple(name for name, _segment in expected_segments)
    states_by_name: dict[str, NativeMutationResultState] = {}
    actual_names = []
    for artifact in artifacts:
        state = restore_native_mutation_context(artifact, data)
        if state.variable in states_by_name:
            raise ContractConflict(
                f"duplicate mutation-context coverage for segment {state.variable!r}"
            )
        selected_policy = (
            policy
            if policy is not None
            else MutationContextPolicy(
                state.schema,
                NATIVE_MUTATION_CONTEXT_SCHEMA,
            )
        )
        _validate_mutation_state(
            state,
            artifact=artifact,
            base_candidate=before.candidate,
            proposed_candidate=proposal.candidate,
            sample=sample,
            step=step,
            policy=selected_policy,
            data=data,
            system=system,
            contract_seed=contract_seed,
        )
        states_by_name[state.variable] = state
        actual_names.append(state.variable)

    if tuple(actual_names) != expected_names:
        raise ContractConflict(
            "mutation-context artifact order/coverage differs from native polymer "
            f"segments: expected {expected_names!r}, got {tuple(actual_names)!r}"
        )

    states = tuple(states_by_name[name] for name in expected_names)
    if len({state.schema for state in states}) != 1:
        raise ContractConflict(
            "mutation-context artifacts use different context policies"
        )
    selector_contexts = states[0].selector_contexts
    score_contributions = states[0].score_contributions
    if any(
        state.selector_contexts != selector_contexts for state in states[1:]
    ):
        raise ContractConflict(
            "mutation-context artifacts disagree on global selector records"
        )
    if any(
        state.score_contributions != score_contributions
        for state in states[1:]
    ):
        raise ContractConflict(
            "mutation-context artifacts disagree on global score contributions"
        )
    return states


def _restore_score_contributions(
    data: DataStore,
    records: tuple[NativeScoreContribution, ...],
) -> list[ScoreContribution]:
    restored = []
    for record in records:
        if not isinstance(record, NativeScoreContribution):
            raise TypeError("score contribution has an unexpected type")
        if record.value.codec == _ARRAY_CODEC.identity:
            value = data.get(record.value, _ARRAY_CODEC)
        elif record.value.codec == _CONTRIBUTION_JSON_CODEC.identity:
            value = data.get(record.value, _CONTRIBUTION_JSON_CODEC)
        else:
            raise ContractConflict(
                "mutation-context score contribution uses an unsupported value codec"
            )
        restored.append(
            ScoreContribution(
                source=record.source,
                metric_name=record.metric_name,
                value=value,
                scope=record.scope,
                entity=record.entity,
                shape=record.shape,
                aggregation=record.aggregation,
                flatten_as=record.flatten_as,
                provenance=record.provenance,
                enabled=record.enabled,
            )
        )
    return restored


def _measured_scores_blob(state: Any) -> Blob:
    """Return the exact score snapshot captured after shared math extensions."""

    matches = tuple(
        item.value
        for item in state.evaluation_scores
        if item.step == state.last_accepted
    )
    if len(matches) != 1:
        raise ContractConflict(
            "portable native decision state does not have exactly one "
            "evaluation snapshot for its last accepted step"
        )
    return matches[0]


def _apply_pending_mutations(
    states: tuple[NativeMutationResultState, ...],
    *,
    runtime_config: dict[str, Any],
    data: DataStore,
    system: Any,
    candidate: Candidate,
) -> None:
    expected_segments = _native_mutation_segments(system, candidate)
    expected_names = tuple(name for name, _segment in expected_segments)
    state_names = tuple(state.variable for state in states)
    if state_names != expected_names:
        raise UnsupportedPortableNativeMath(
            "mutation-context state does not cover native polymer segments in "
            f"stable order: expected {expected_names!r}, got {state_names!r}"
        )

    first = states[0]
    if first.selector_contexts.codec != _SELECTOR_CONTEXT_CODEC.identity:
        raise ContractConflict(
            "mutation-context selector payload uses an unsupported codec"
        )
    selectors = data.get(first.selector_contexts, _SELECTOR_CONTEXT_CODEC)
    contributions = _restore_score_contributions(
        data,
        first.score_contributions,
    )

    # Validate every segment and every pending tuple before touching the
    # legacy owner. A malformed later artifact must not leave earlier segments
    # partially mutated.
    prepared = []
    for state, (name, segment) in zip(states, expected_segments, strict=True):
        if state.variable != name:
            raise ContractConflict(
                f"mutation-context variable {state.variable!r} differs from {name!r}"
            )
        if segment.sequence != state.sequence_before_pending:
            raise ContractConflict(
                f"active legacy segment {name!r} differs from its accepted sequence"
            )
        if segment.mutations not in (None, []):
            raise ContractConflict(
                f"active legacy segment {name!r} contains unrepresented mutations"
            )
        simulated = state.sequence_before_pending
        for mutation in state.pending_mutations:
            if (
                type(mutation.position) is not int
                or mutation.position < 0
                or mutation.position >= len(simulated)
            ):
                raise ContractConflict(
                    f"mutation-context pending position is outside segment {name!r}"
                )
            if simulated[mutation.position] != mutation.from_token:
                raise ContractConflict(
                    f"mutation-context source token differs for segment {name!r}"
                )
            simulated = (
                simulated[: mutation.position]
                + mutation.to_token
                + simulated[mutation.position + 1 :]
            )
        if simulated != state.working_sequence:
            raise ContractConflict(
                f"restored pending mutations do not reproduce segment {name!r}"
            )
        prepared.append((state, segment, getattr(segment, "mutations", None)))

    try:
        for state, segment, _original in prepared:
            for mutation in state.pending_mutations:
                segment.add_mutations(
                    (
                        mutation.position,
                        mutation.from_token,
                        mutation.to_token,
                        mutation.entropy,
                        mutation.score,
                    )
                )
        for state, segment, _original in prepared:
            if segment.mutated_sequence != state.working_sequence:
                raise ContractConflict(
                    f"legacy segment {state.variable!r} did not reproduce its "
                    "mutation-context working sequence"
                )
    except Exception:
        for _state, segment, original in prepared:
            segment.mutations = None if original is None else list(original)
        raise

    runtime_config["mutator_selector_score_contexts"] = selectors
    runtime_config["score_contributions"] = contributions


def _set_decisions(decisions: Any) -> None:
    try:
        global_config.update_const("decisions", decisions)
    except (KeyError, RuntimeError):
        global_config.init_const("decisions", decisions)


def _set_global_step(value: Any) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("portable native step must be a non-negative integer")
    global_config.set_variable("step", value)
    return value


def _apply_single_mutator_stage(stage_manager: Any, mutator: Any) -> None:
    apply_mutation_num = getattr(stage_manager, "_apply_mutation_num", None)
    if not callable(apply_mutation_num):
        raise UnsupportedPortableNativeMath(
            "portable native stage manager cannot resolve mutation_num"
        )
    apply_mutation_num()
    # The portable profile admits exactly one selected mutator, so the legacy
    # StageManager normalization for one alias is necessarily 1.0.
    mutator.mutation_weight = 1.0


def _hook_identity(
    context: PortableNativeHookContext,
    *,
    role: str,
    configure: Any,
    invoke: Any,
) -> tuple[str, str]:
    common = (
        PORTABLE_NATIVE_HOOKS_SCHEMA,
        role,
        getattr(context, f"{role}_alias"),
        context.config_identity,
        context.source_identity,
    )
    identity = key_for(
        common + (_callable_source(configure), _callable_source(invoke))
    ).digest
    invoke_identity = key_for(
        (
            "astrolabe.portable-native.operation-invoker/v1",
            identity,
            _callable_source(invoke),
        )
    ).digest
    return identity, invoke_identity


def build_portable_native_operation_hooks(
    context: PortableNativeHookContext,
) -> Mapping[str, NativeOperationHooks]:
    """Build the frozen production hooks without constructing native models."""

    if not isinstance(context, PortableNativeHookContext):
        raise TypeError("context must be PortableNativeHookContext")
    run_module = context.run_module
    finalize_evaluation = getattr(
        run_module,
        "_finalize_evaluation_result",
        None,
    )
    execute_mutation = getattr(run_module, "execute_mutation_step", None)
    if not callable(finalize_evaluation) or not callable(execute_mutation):
        raise UnsupportedPortableNativeMath(
            "portable native default hooks require the shared run evaluation "
            "finalizer and mutation-step executor"
        )

    def configure_evaluator(evaluator: Any, request: Any):
        del evaluator
        sample = request.samples[0]
        if not sample.index.isdecimal():
            raise UnsupportedPortableNativeMath(
                "portable native evaluator requires a numeric Sample.index"
            )
        _set_global_step(int(sample.index))
        return None

    def invoke_evaluator(
        evaluator: Any,
        inputs: Mapping[str, Any],
        runtime_config: Mapping[str, Any],
    ):
        step = _set_global_step(int(runtime_config.get("step", -1)))
        if step < 0:
            raise UnsupportedPortableNativeMath(
                "portable native evaluator runtime_config lacks a valid step"
            )
        raw_scores = evaluator.evaluate(inputs, runtime_config)
        return finalize_evaluation(
            context.args,
            raw_scores,
            context.loss_computer,
        )

    def configure_mutator(
        mutator: Any,
        runtime_config: Mapping[str, Any],
        *,
        request: Any,
        sample: Any,
        step: int,
        work_root: Path,
    ):
        del request, sample, work_root
        _set_global_step(step)
        _apply_single_mutator_stage(context.stage_manager, mutator)
        if not isinstance(runtime_config, Mapping):
            raise TypeError("native mutator runtime config must be a mapping")
        return None

    def invoke_mutator(
        mutator: Any,
        protomers: Mapping[str, Any],
        scores: Any,
        runtime_config: Mapping[str, Any],
    ) -> None:
        del protomers
        step = _set_global_step(int(runtime_config.get("step", -1)))
        _apply_single_mutator_stage(context.stage_manager, mutator)
        execute_mutation(
            [mutator],
            scores,
            runtime_config,
            step,
        )

    evaluator_identity, evaluator_invoke_identity = _hook_identity(
        context,
        role="evaluator",
        configure=configure_evaluator,
        invoke=invoke_evaluator,
    )
    mutator_identity, mutator_invoke_identity = _hook_identity(
        context,
        role="mutator",
        configure=configure_mutator,
        invoke=invoke_mutator,
    )
    return {
        context.evaluator_alias: NativeOperationHooks(
            configure=configure_evaluator,
            identity=evaluator_identity,
            invoke=invoke_evaluator,
            invoke_identity=evaluator_invoke_identity,
        ),
        context.mutator_alias: NativeOperationHooks(
            configure=configure_mutator,
            identity=mutator_identity,
            mutation_context=PORTABLE_NATIVE_MUTATION_CONTEXT_POLICY,
            invoke=invoke_mutator,
            invoke_identity=mutator_invoke_identity,
        ),
    }


def build_portable_native_math(
    context: Any,
    *,
    run_module: Any | None = None,
) -> PortableNativeMath:
    """Construct the shared root/step decision adapter from frozen context."""

    from .portable_native_builder import PortableNativeMathContext

    if not isinstance(context, PortableNativeMathContext):
        raise TypeError("context must be PortableNativeMathContext")
    if run_module is not None and run_module is not context.run_module:
        raise ContractConflict(
            "portable math run_module must be the same module object selected by "
            "the recipe builder; importing a second run module is unsupported"
        )
    policy = context.mutation_context
    if not isinstance(policy, MutationContextPolicy):
        raise UnsupportedPortableNativeMath(
            "portable native decisions require an explicit MutationContextPolicy"
        )
    run_module = context.run_module if run_module is None else run_module
    initialize, decide = _require_helpers(run_module)
    data = context.bindings.data
    if not isinstance(data, DataStore):
        raise TypeError("portable math context bindings.data must be a DataStore")
    root_runtime_config = dict(context.root_runtime_config)
    if bool(root_runtime_config.get("mutator_stack_pre_update", False)):
        raise UnsupportedPortableNativeMath(
            "portable native does not support mutator stack pre-update"
        )

    def finalize_root(request, scores, callback_data, destination):
        if callback_data is not data:
            raise ContractConflict("portable root math received another DataStore")
        runtime_config = dict(root_runtime_config)
        runtime_config["step"] = 0
        decisions, accepted_scores = initialize(
            context.args,
            context.loss_computer,
            scores,
            context.stage_manager,
            runtime_config,
        )
        if accepted_scores is not scores:
            raise UnsupportedPortableNativeMath(
                "root stage entry changed the accepted evaluation before portable "
                "state capture"
            )
        state = snapshot_decision_state(decisions, callback_data)
        if state.last_accepted != 0 or tuple(
            score.step for score in state.scores
        ) != (0,):
            raise UnsupportedPortableNativeMath(
                "shared root decision math did not produce one accepted step-zero state"
            )
        return NativeStepState(
            schema=NATIVE_STEP_STATE_SCHEMA,
            candidate=request.candidate,
            current_scores=_measured_scores_blob(state),
            decision_state=state,
            static_stage=context.contract.static_stage,
            runtime_context=request.runtime_context,
            next_step=1,
        )

    def decide_step(
        request: DecisionRequest,
        scores,
        callback_data: DataStore,
        destination: Path,
    ) -> DecisionResult:
        if callback_data is not data:
            raise ContractConflict("portable step math received another DataStore")
        if request.before.runtime_context != context.runtime_context:
            raise ContractConflict(
                "portable decision predecessor references another runtime context"
            )
        if request.proposal_context != request.before.runtime_context:
            raise ContractConflict(
                "portable Proposal context differs from the accepted predecessor"
            )
        if request.before.static_stage != context.contract.static_stage:
            raise ContractConflict("portable decision predecessor changed static stage")

        mutation_states = validate_mutation_context_artifacts(
            request.proposal,
            before=request.before,
            sample=None,
            step=request.step,
            policy=policy,
            data=callback_data,
            system=context.bindings.translation.system,
            contract_seed=context.contract.seed,
        )

        from .portable_native_state import restore_decision_state

        decisions = restore_decision_state(
            request.before.decision_state,
            callback_data,
            Path(destination) / "restored-decisions",
        )
        _set_decisions(decisions)
        runtime_config = dict(root_runtime_config)
        runtime_config["step"] = request.step
        _apply_pending_mutations(
            mutation_states,
            runtime_config=runtime_config,
            data=callback_data,
            system=context.bindings.translation.system,
            candidate=request.before.candidate,
        )

        accepted_scores, accepted, early_stop = decide(
            context.args,
            context.loss_computer,
            scores,
            context.stage_manager,
            runtime_config,
            decisions,
            request.step,
        )
        if accepted_scores is not decisions.evaluation_results[
            decisions.last_accepted
        ]:
            raise ContractConflict(
                "shared decision math returned another accepted evaluation"
            )
        decision_state = snapshot_decision_state(decisions, callback_data)
        after = NativeStepState(
            schema=NATIVE_STEP_STATE_SCHEMA,
            candidate=(
                request.proposal.candidate if accepted else request.before.candidate
            ),
            current_scores=(
                _measured_scores_blob(decision_state)
                if accepted
                else request.before.current_scores
            ),
            decision_state=decision_state,
            static_stage=context.contract.static_stage,
            runtime_context=request.before.runtime_context,
            next_step=request.step + 1,
        )
        details = store_numeric_tree(
            callback_data,
            {
                "accepted": bool(accepted),
                "early_stop": bool(early_stop),
                "last_accepted": int(decisions.last_accepted),
                "step": int(request.step),
            },
        )
        return DecisionResult(
            after=after,
            accepted=bool(accepted),
            stop_reason="scientific_early_stop" if early_stop else None,
            decision_details=details,
        )

    helper_source = (
        _callable_source(initialize),
        _callable_source(decide),
    )
    return PortableNativeMath(
        identity=key_for(
            (
                PORTABLE_NATIVE_MATH_SCHEMA,
                context.contract.config_identity,
                context.contract.source_identity,
                policy.descriptor(),
                helper_source,
            )
        ),
        finalize_root=finalize_root,
        decide_step=decide_step,
        requires_mutation_context=True,
    )


__all__ = [
    "PORTABLE_NATIVE_HOOKS_SCHEMA",
    "PORTABLE_NATIVE_MATH_SCHEMA",
    "PORTABLE_NATIVE_MUTATION_CONTEXT_POLICY",
    "PortableNativeHookContext",
    "UnsupportedPortableNativeMath",
    "build_portable_native_math",
    "build_portable_native_operation_hooks",
    "validate_mutation_context_artifacts",
    "validate_native_mutation_topology",
]
