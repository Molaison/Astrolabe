from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest

import run as run_module
import astrolabe.decision_maker as decision_module
from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import Protomer, SegmentManager
from astrolabe.decision_maker import Decisions
from astrolabe.design.legacy import LegacyContext, capture_legacy_candidate
from astrolabe.design.native_mutation import make_mutator_operation
from astrolabe.design.native_mutation_state import (
    NATIVE_MUTATION_CONTEXT_SCHEMA,
    MutationContextPolicy,
    restore_native_mutation_context,
)
from astrolabe.design.native_prediction import make_evaluator_operation
from astrolabe.design.native_scores import NATIVE_SCORES_ENVELOPE_CODEC
from astrolabe.design.native_support import NativeBindings, native_score_artifact
from astrolabe.design.portable_native_builder import PortableNativeMathContext
from astrolabe.design.portable_native_flow import (
    PortableNativeFlow,
    PortableNativeMath,
    PortableNativeSupport,
    run_portable_native,
    validate_static_nonpose_support,
)
from astrolabe.design.portable_native_math import build_portable_native_math
from astrolabe.design.portable_native_state import (
    DECISION_STATE_SCHEMA,
    NATIVE_RUNTIME_CONTEXT_CODEC,
    NATIVE_STEP_STATE_CODEC,
    NATIVE_STEP_STATE_SCHEMA,
    NUMERIC_TREE_CODEC,
    SCIENTIFIC_CONTRACT_SCHEMA,
    DecisionResult,
    NativeRuntimeContext,
    NativeStepState,
    ScientificRunContract,
    UnsupportedPortableNativeState,
    restore_decision_state,
    snapshot_decision_state,
    store_numeric_tree,
)
from astrolabe.design.runtime import Runtime
from astrolabe.design.store import DataStore
from astrolabe.design.successor_search import AcceptedSuccessorStateCodec
from astrolabe.design.types import (
    History,
    Sample,
    SamplingPolicy,
    Work,
    key_for,
)
from astrolabe.evaluators.utils import StandardScores
from astrolabe.losses.base import LossEngine
from astrolabe.score_contributions import ScoreContribution
from astrolabe.stage_manager import StageManager


@dataclass(frozen=True)
class _ScriptedStep:
    sequence: str
    weighted: float
    early_stop: bool = False
    soft_accepted: bool = False
    gate_passed: bool = True


@pytest.fixture(autouse=True)
def _clean_global_config():
    global_config.reset()
    yield
    global_config.reset()


def _scores(tmp_path: Path, step: int, scripted: _ScriptedStep) -> StandardScores:
    path = tmp_path / f"portable_step_{step:02d}.cif"
    path.write_text(
        f"data_portable_step_{step}\n# sequence={scripted.sequence}\n",
        encoding="utf-8",
    )
    scores = StandardScores(
        names=["A"],
        plddt={"A": [np.asarray([1.0, 0.5], dtype=np.float32)]},
        complex_paths={"A": path},
    )
    scores.weighted = float(scripted.weighted)
    scores.early_stop = bool(scripted.early_stop)
    scores.soft_accepted = bool(scripted.soft_accepted)
    scores.gate_passed = bool(scripted.gate_passed)
    scores.sequence = scripted.sequence
    return scores


class _ScriptedLoss(LossEngine):
    def __init__(self) -> None:
        super().__init__(
            {"loss_terms": {}},
            runtime_context=global_config.build_runtime_context(),
            validate_config=False,
        )
        self.runtime_configs = []

    def compute_loss(self, eval_result, runtime_config):
        self.runtime_configs.append(dict(runtime_config))
        return (
            float(eval_result.weighted),
            {
                "weighted": float(eval_result.weighted),
                "early_stop": 1.0 if eval_result.early_stop else 0.0,
                "soft_accepted": 1.0 if eval_result.soft_accepted else 0.0,
                "gate_passed": 1.0 if eval_result.gate_passed else 0.0,
            },
        )

    def is_early_stopping(self, scores_detail):
        return bool(scores_detail.get("early_stop"))

    def is_soft_accepted(self, scores_detail):
        return bool(scores_detail.get("soft_accepted"))

    def passes_acceptance_gate(self, scores_detail):
        return bool(scores_detail.get("gate_passed", 1.0))


class _ScriptedEvaluator:
    def __init__(self, tmp_path: Path, script: dict[int, _ScriptedStep], calls: list):
        self.tmp_path = tmp_path
        self.script = script
        self.calls = calls

    def evaluate(self, inputs, runtime_config):
        step = int(runtime_config["step"])
        sequence = inputs["A"].chains[0].sequence
        expected = self.script[step]
        assert sequence == expected.sequence
        self.calls.append((step, sequence))
        return _scores(self.tmp_path, step, expected)

    def shutdown(self):
        pass


class _ScriptedMutator:
    def __init__(
        self,
        script: dict[int, _ScriptedStep],
        calls: list,
        observations: list | None = None,
    ):
        self.script = script
        self.calls = calls
        self.observations = [] if observations is None else observations

    def mutate(self, mutable_protomers, evaluation_result, runtime_config):
        step = int(runtime_config["step"])
        self.observations.append((step, evaluation_result))
        target = self.script[step].sequence
        protomer = mutable_protomers["A"]
        segment = global_config.segments[protomer.components[0]]
        current = protomer.mutated_sequence
        differences = [
            index
            for index, (left, right) in enumerate(zip(current, target, strict=True))
            if left != right
        ]
        assert len(differences) <= 1
        if differences:
            index = differences[0]
            assert not segment.mutations
            segment.add_mutations(
                (index, current[index], target[index], 0.0, 0.0)
            )
        else:
            assert current == target
        runtime_config["mutator_selector_score_contexts"] = [
            {"step": step, "candidate": target}
        ]
        runtime_config["score_contributions"] = [
            ScoreContribution(
                source="test/portable-native-flow",
                metric_name="ordered_pending",
                value=np.asarray(
                    [
                        float(step),
                        float(differences[0] if differences else -1),
                    ],
                    dtype=np.float32,
                ),
            )
        ]
        self.calls.append((step, target))

    def shutdown(self):
        pass


def _configure_legacy(
    tmp_path: Path,
    *,
    script: dict[int, _ScriptedStep],
    metropolis: bool,
    evaluator_calls: list,
    mutator_calls: list,
    evaluator_loads: list,
    mutator_loads: list,
    load_order: list,
    mutation_context: MutationContextPolicy | None,
    initialize_decisions: bool,
    mutator_observations: list | None = None,
):
    global_config.init_const("segments", {})
    global_config.init_const("protomers", {})
    global_config.init_const("mutable_protomers", {})
    global_config.init_const("named_files", {})
    global_config.init_const("segment_manager", SegmentManager())
    protomer = Protomer("A", "AAAA", "p")
    global_config.protomers["A"] = protomer
    global_config.mutable_protomers["A"] = protomer
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("initial_temperature", 1.0)
    global_config.init_const("use_metropolis_acceptance", metropolis)
    if initialize_decisions:
        global_config.init_const("decisions", None)
    global_config.init_const("structure_start", None)

    loss = _ScriptedLoss()
    stages = OmegaConf.create(
        {
            "default": {
                "oligo": ["A"],
                "mutation_num": 1,
                "weighted_score_expression": "weighted",
                "early_stopping": "early_stop > 0.0",
                "acceptance_threshold": "soft_accepted > 0.0",
                "evaluator": "fold",
                "mutators": ["mutate"],
            }
        }
    )
    stage_manager = StageManager(
        stages,
        loss_computer=loss,
        runtime_context=global_config.build_runtime_context(),
    )
    global_config.init_const("stage_manager", stage_manager)
    global_config.init_const("loss_computer", loss)
    global_config.init_const("mutators", {})
    global_config.init_const("evaluators", {})
    data = DataStore(tmp_path / "data")
    translation = capture_legacy_candidate(assemblies=("A",), data=data)
    context = LegacyContext()
    bindings = NativeBindings(context, translation, data, tmp_path / "work")
    sampling = SamplingPolicy(
        identity=key_for(("portable-test-sampling/v1",)),
        options=data.put({"mutation_count": 1}, NUMERIC_TREE_CODEC),
    )

    def evaluator_factory():
        evaluator_loads.append("evaluator")
        load_order.append("evaluator")
        return _ScriptedEvaluator(tmp_path, script, evaluator_calls)

    def mutator_factory():
        mutator_loads.append("mutator")
        load_order.append("mutator")
        return _ScriptedMutator(script, mutator_calls, mutator_observations)

    evaluator = make_evaluator_operation(
        "fold",
        evaluator_factory,
        bindings,
        version="portable-test-evaluator/v1",
        model_identity=key_for(("portable-test-evaluator-model/v1", tuple(script))),
    )
    mutator = make_mutator_operation(
        "mutate",
        mutator_factory,
        bindings,
        sampling_policy=sampling,
        version="portable-test-mutator/v1",
        model_identity=key_for(("portable-test-mutator-model/v1", tuple(script))),
        mutation_context=mutation_context,
    )
    return (
        bindings,
        evaluator,
        mutator,
        sampling,
        loss,
        stage_manager,
        translation,
    )


def _root_finalizer(contract, runtime_context):
    def finalize(request, scores, data, destination):
        decisions = Decisions.create(
            scores,
            {"step": 0},
            global_config.loss_computer.compute_loss,
        )
        global_config.update_const("decisions", decisions)
        decision_state = snapshot_decision_state(decisions, data)
        current_scores = next(
            item.value
            for item in decision_state.evaluation_scores
            if item.step == decision_state.last_accepted
        )
        return NativeStepState(
            schema=NATIVE_STEP_STATE_SCHEMA,
            candidate=request.candidate,
            current_scores=current_scores,
            decision_state=decision_state,
            static_stage=contract.static_stage,
            runtime_context=runtime_context,
            next_step=1,
        )

    return finalize


def _step_decider(contract, runtime_context, script):
    def decide(request, scores, data, destination):
        decisions = restore_decision_state(
            request.before.decision_state,
            data,
            destination / "restored",
        )
        global_config.update_const("decisions", decisions)

        before_tokens = request.before.candidate.assignments[0].value.tokens
        proposed_tokens = request.proposal.candidate.assignments[0].value.tokens
        differences = [
            index
            for index, (left, right) in enumerate(
                zip(before_tokens, proposed_tokens, strict=True)
            )
            if left != right
        ]
        assert len(differences) == 1
        index = differences[0]
        protomer = global_config.mutable_protomers["A"]
        segment = global_config.segments[protomer.components[0]]
        segment.add_mutations(
            (index, before_tokens[index], proposed_tokens[index], 0.0, 0.0)
        )

        early_stop, _accepted_scores, accepted = decisions.make_mutation_decision(
            request.step,
            scores,
            global_config.loss_computer,
            {"step": request.step},
        )
        global_config.update_const("decisions", decisions)
        decision_state = snapshot_decision_state(decisions, data)
        current_scores = next(
            item.value
            for item in decision_state.evaluation_scores
            if item.step == decision_state.last_accepted
        )
        after = NativeStepState(
            schema=NATIVE_STEP_STATE_SCHEMA,
            candidate=request.proposal.candidate if accepted else request.before.candidate,
            current_scores=(
                current_scores if accepted else request.before.current_scores
            ),
            decision_state=decision_state,
            static_stage=contract.static_stage,
            runtime_context=runtime_context,
            next_step=request.step + 1,
        )
        details = store_numeric_tree(
            data,
            {
                "accepted": accepted,
                "early_stop": early_stop,
                "last_row": decisions.row_history[-1],
                "step": request.step,
            },
        )
        return DecisionResult(after, accepted, "scientific_early_stop" if early_stop else None, details)

    return decide


def _flow_fixture(
    tmp_path: Path,
    *,
    script: dict[int, _ScriptedStep],
    metropolis: bool,
    evaluator_calls=None,
    mutator_calls=None,
    evaluator_loads=None,
    mutator_loads=None,
    shared_math: bool = False,
    mutator_observations=None,
):
    evaluator_calls = [] if evaluator_calls is None else evaluator_calls
    mutator_calls = [] if mutator_calls is None else mutator_calls
    evaluator_loads = [] if evaluator_loads is None else evaluator_loads
    mutator_loads = [] if mutator_loads is None else mutator_loads
    mutator_observations = (
        [] if mutator_observations is None else mutator_observations
    )
    load_order: list[str] = []
    (
        bindings,
        evaluator,
        mutator,
        sampling,
        loss,
        stage_manager,
        translation,
    ) = _configure_legacy(
        tmp_path,
        script=script,
        metropolis=metropolis,
        evaluator_calls=evaluator_calls,
        mutator_calls=mutator_calls,
        evaluator_loads=evaluator_loads,
        mutator_loads=mutator_loads,
        load_order=load_order,
        mutation_context=(
            MutationContextPolicy(
                "test/portable-native-flow/mutation-context/v1",
                NATIVE_MUTATION_CONTEXT_SCHEMA,
            )
            if shared_math
            else None
        ),
        initialize_decisions=not shared_math,
        mutator_observations=mutator_observations,
    )
    support = PortableNativeSupport(("default",))
    static_stage = validate_static_nonpose_support(support)
    context = NativeRuntimeContext(
        schema="astrolabe.portable-native.runtime-context/v1",
        selector_and_contribution_context=store_numeric_tree(bindings.data, {}),
        decision_context=store_numeric_tree(
            bindings.data,
            {"use_metropolis_acceptance": metropolis},
        ),
    )
    runtime_context = bindings.data.put(context, NATIVE_RUNTIME_CONTEXT_CODEC)
    contract = ScientificRunContract(
        schema=SCIENTIFIC_CONTRACT_SCHEMA,
        config_identity=key_for(("portable-test-config/v1", tuple(script))),
        source_identity=key_for(("portable-test-source/v1",)),
        system_identity=translation.system.identity,
        resource_manifest=key_for(("portable-test-resources/v1",)),
        static_stage=static_stage,
        objective=key_for(("portable-test-objective/v1", "weighted")),
        component_identities=(
            evaluator.descriptor.identity,
            mutator.descriptor.identity,
        ),
        seed=5,
        acceptance_protocol=sampling.identity,
        codec_identities=(
            NATIVE_STEP_STATE_CODEC.identity,
            NUMERIC_TREE_CODEC.identity,
        ),
        recovery_capabilities=("raw_predict", "decision", "model_ready"),
    )
    sample_protocol = key_for(("portable-test-sample/v1",))
    if shared_math:
        mutation_context = MutationContextPolicy(
            "test/portable-native-flow/mutation-context/v1",
            NATIVE_MUTATION_CONTEXT_SCHEMA,
        )
        math_context = PortableNativeMathContext(
            args=SimpleNamespace(steps=2),
            prepared=None,
            loss_computer=loss,
            stage_manager=stage_manager,
            bindings=bindings,
            contract=contract,
            initial_candidate=translation.candidate,
            runtime_context=runtime_context,
            root_runtime_config={
                "mutator_stack_pre_update": False,
                "state_conflict_diagnostics": None,
            },
            sample_protocol=sample_protocol,
            run_module=run_module,
            mutation_context=mutation_context,
        )
        math = build_portable_native_math(math_context)
    else:
        math = PortableNativeMath(
            identity=key_for(("portable-test-math/v1", tuple(sorted(script.items())))),
            finalize_root=_root_finalizer(contract, runtime_context),
            decide_step=_step_decider(contract, runtime_context, script),
        )
    flow = PortableNativeFlow(
        contract=contract,
        support=support,
        system=translation.system,
        initial_candidate=translation.candidate,
        evaluator=evaluator,
        mutator=mutator,
        bindings=bindings,
        sampling_policy=sampling,
        assemblies=("A",),
        math=math,
        runtime_context=runtime_context,
        sample_protocol=sample_protocol,
        initialization_order=("mutator", "evaluator"),
    )
    runtime = Runtime(
        flow.operations,
        root=tmp_path / "runtime",
        data=bindings.data,
        scientific_contract=key_for(("portable-test-runtime/v1", flow.identity)),
    )
    return SimpleNamespace(
        flow=flow,
        runtime=runtime,
        bindings=bindings,
        evaluator=evaluator,
        mutator=mutator,
        loss=loss,
        stage_manager=stage_manager,
        evaluator_calls=evaluator_calls,
        mutator_calls=mutator_calls,
        evaluator_loads=evaluator_loads,
        mutator_loads=mutator_loads,
        mutator_observations=mutator_observations,
        load_order=load_order,
    )


def test_static_nonpose_support_rejects_all_unportable_features() -> None:
    with pytest.raises(Exception, match="exactly one"):
        validate_static_nonpose_support(PortableNativeSupport(("one", "two")))
    with pytest.raises(Exception) as captured:
        validate_static_nonpose_support(
            PortableNativeSupport(
                ("one",),
                transition_gates=True,
                structure_start=True,
                live_pose=True,
                model_owned_mutable_state=True,
                unsupported_score_extensions=True,
            )
        )
    message = str(captured.value)
    assert "transition gates" in message
    assert "structure_start" in message
    assert "Pose" in message
    assert "model-owned" in message
    assert "score extensions" in message


def test_unsupported_profile_fails_before_any_model_factory_load(tmp_path: Path) -> None:
    case = _flow_fixture(
        tmp_path,
        script={0: _ScriptedStep("AAAA", 1.0)},
        metropolis=False,
    )
    unsupported = replace(
        case.flow.support,
        stage_keys=("step-one", "step-two"),
        live_pose=True,
    )
    with pytest.raises(Exception, match="exactly one|Pose"):
        PortableNativeFlow(
            contract=case.flow.contract,
            support=unsupported,
            system=case.flow.system,
            initial_candidate=case.flow.initial,
            evaluator=case.evaluator,
            mutator=case.mutator,
            bindings=case.bindings,
            sampling_policy=case.flow.sampling_policy,
            assemblies=case.flow.assemblies,
            math=case.flow.math,
            runtime_context=case.flow.runtime_context,
            sample_protocol=case.flow.sample_protocol,
        )
    assert case.evaluator_loads == []
    assert case.mutator_loads == []
    assert case.evaluator_calls == []
    assert case.mutator_calls == []


def test_numeric_tree_codec_preserves_arrays_nonfinite_and_nested_types(tmp_path: Path) -> None:
    value = {
        "array": np.asarray([[1.0, np.nan]], dtype=np.float32),
        "scalar": np.float64(np.inf),
        "tuple": ("A", 17, Path("x.cif")),
        "missing": None,
    }
    data = DataStore(tmp_path / "data")
    blob = data.put(value, NUMERIC_TREE_CODEC)
    restored = data.get(blob, NUMERIC_TREE_CODEC)
    assert restored["array"].dtype == np.float32
    assert restored["array"].shape == (1, 2)
    assert np.array_equal(restored["array"], value["array"], equal_nan=True)
    assert math.isinf(float(restored["scalar"]))
    assert restored["tuple"] == ("A", 17, Path("x.cif"))
    with pytest.raises(UnsupportedPortableNativeState, match="object-dtype"):
        NUMERIC_TREE_CODEC.encode({"bad": np.asarray([object()], dtype=object)})


@pytest.mark.parametrize(
    ("weighted", "metropolis", "seed", "expected_accepted", "expected_score"),
    [
        (0.6, True, 0, True, 0.6),
        (0.9, False, 19, False, 1.0),
    ],
)
def test_portable_flow_root_and_one_child_match_current_decision_math(
    tmp_path: Path,
    weighted: float,
    metropolis: bool,
    seed: int,
    expected_accepted: bool,
    expected_score: float,
) -> None:
    script = {
        0: _ScriptedStep("AAAA", 1.0),
        1: _ScriptedStep("GAAA", weighted),
    }
    case = _flow_fixture(
        tmp_path,
        script=script,
        metropolis=metropolis,
    )
    expected_rng = np.random.RandomState(seed)
    if metropolis:
        expected_rng.uniform(0, 1)
    expected_rng_state = expected_rng.get_state()
    np.random.seed(seed)
    result = run_portable_native(
        case.flow,
        rounds=1,
        runtime=case.runtime,
    )

    assert case.evaluator_loads == ["evaluator"]
    assert case.mutator_loads == ["mutator"]
    assert case.load_order == ["mutator", "evaluator"]
    assert case.evaluator_calls == [(0, "AAAA"), (1, "GAAA")]
    assert case.mutator_calls == [(1, "GAAA")]
    decisions = global_config.decisions
    assert decisions.scores == {0: 1.0, 1: weighted}
    assert decisions.last_accepted == (1 if expected_accepted else 0)
    assert (1 in decisions.accepted_steps) is expected_accepted
    assert result.selected[0].assessment.values == (expected_score,)
    assert result.selected[0].candidate.assignments[0].value.tokens == tuple(
        "GAAA" if expected_accepted else "AAAA"
    )

    checkpoint = case.runtime.load("run/state")
    policy = case.runtime.data.get(checkpoint["state"], AcceptedSuccessorStateCodec())
    assert policy.used == 1
    assert policy.stop_reason is None
    attempts = case.runtime.attempts()
    assert sum(item.key.named_round == "root/raw_predict" for item in attempts) == 1
    assert sum(item.key.named_round == "step/1/raw_predict" for item in attempts) == 1
    assert all(item.status == "completed" for item in attempts)
    actual_rng_state = np.random.get_state()
    assert actual_rng_state[0] == expected_rng_state[0]
    np.testing.assert_array_equal(actual_rng_state[1], expected_rng_state[1])
    assert actual_rng_state[2:] == expected_rng_state[2:]


@pytest.mark.parametrize(
    (
        "second",
        "metropolis",
        "seed",
        "expected_accepted",
        "expected_sequence",
        "expected_score",
    ),
    [
        (_ScriptedStep("AAAA", 1.5), False, 7, True, "AAAA", 1.5),
        (_ScriptedStep("CAAA", 0.5), False, 7, False, "AAAA", 1.0),
        (_ScriptedStep("CAAA", 0.5), True, 0, True, "CAAA", 0.5),
    ],
)
def test_shared_decisions_adapter_preserves_noop_reject_and_metropolis(
    tmp_path: Path,
    second: _ScriptedStep,
    metropolis: bool,
    seed: int,
    expected_accepted: bool,
    expected_sequence: str,
    expected_score: float,
) -> None:
    script = {
        0: _ScriptedStep("AAAA", 1.0),
        1: second,
    }
    case = _flow_fixture(
        tmp_path,
        script=script,
        metropolis=metropolis,
        shared_math=True,
    )
    expected_rng = np.random.RandomState(seed)
    if metropolis:
        expected_rng.uniform(0, 1)
    expected_rng_state = expected_rng.get_state()
    np.random.seed(seed)

    result = run_portable_native(case.flow, rounds=1, runtime=case.runtime)

    decisions = global_config.decisions
    assert decisions.last_accepted == (1 if expected_accepted else 0)
    assert decisions.scores == {0: 1.0, 1: second.weighted}
    assert result.selected[0].assessment.values == (expected_score,)
    assert result.selected[0].candidate.assignments[0].value.tokens == tuple(
        expected_sequence
    )
    assert case.evaluator_calls == [
        (0, "AAAA"),
        (1, second.sequence),
    ]
    assert case.mutator_calls == [(1, second.sequence)]
    observed = case.loss.runtime_configs[-1]
    assert observed["mutator_selector_score_contexts"] == [
        {"step": 1, "candidate": second.sequence}
    ]
    contribution = observed["score_contributions"][0]
    assert contribution.source == "test/portable-native-flow"
    assert contribution.metric_name == "ordered_pending"
    np.testing.assert_array_equal(
        contribution.value,
        np.asarray(
            [
                1.0,
                0.0 if second.sequence != "AAAA" else -1.0,
            ],
            dtype=np.float32,
        ),
    )

    step_outcome = next(
        item
        for item in case.runtime.history.records
        if item.status == "completed"
    )
    artifact = next(
        item
        for item in step_outcome.proposals[0].artifacts
        if item.content.codec
        == "astrolabe.design.native-mutation.context/v1"
    )
    mutation_state = restore_native_mutation_context(artifact, case.bindings.data)
    expected_pending = [] if second.sequence == "AAAA" else [
        (0, "A", second.sequence[0])
    ]
    assert [
        (item.position, item.from_token, item.to_token)
        for item in mutation_state.pending_mutations
    ] == expected_pending

    actual_rng_state = np.random.get_state()
    assert actual_rng_state[0] == expected_rng_state[0]
    np.testing.assert_array_equal(actual_rng_state[1], expected_rng_state[1])
    assert actual_rng_state[2:] == expected_rng_state[2:]


def test_measured_score_extensions_survive_cut_and_rejected_step(
    tmp_path: Path,
) -> None:
    script = {
        0: _ScriptedStep("AAAA", 1.0),
        1: _ScriptedStep("GAAA", 1.5, gate_passed=False),
        2: _ScriptedStep("CAAA", 1.2),
    }
    observations = []
    case = _flow_fixture(
        tmp_path,
        script=script,
        metropolis=False,
        shared_math=True,
        mutator_observations=observations,
    )
    original_compute_loss = case.loss.compute_loss

    def compute_loss_with_extensions(evaluation_result, runtime_config):
        step = int(runtime_config["step"])
        if not getattr(evaluation_result, "candidate_scores", None):
            evaluation_result.score_contributions = [
                ScoreContribution(
                    source="test/portable-native-flow/measured",
                    metric_name="post_raw_snapshot",
                    value=np.asarray(
                        [float(step), step + 0.25],
                        dtype=np.float32,
                    ),
                )
            ]
            evaluation_result.per_proto_score_terms = {
                "A": {
                    "measured_residue": [
                        np.asarray([0.1 + step, 0.2 + step], dtype=np.float32)
                    ]
                }
            }
            candidate = StandardScores(
                names=["A"],
                plddt={
                    "A": [
                        np.asarray(
                            [0.3 + step, 0.4 + step],
                            dtype=np.float32,
                        )
                    ]
                },
                complex_paths={},
            )
            candidate.portable_measurement_step = step
            evaluation_result.candidate_scores = [candidate]
        return original_compute_loss(evaluation_result, runtime_config)

    case.loss.compute_loss = compute_loss_with_extensions
    runtime = case.runtime
    empty = History(key_for(()), ())
    runtime.preflight(case.flow.requirements)
    with runtime.bind(
        "bootstrap",
        empty,
        entry=(case.flow.system, case.flow.initial),
    ):
        case.flow.bootstrap(runtime)
    with runtime.bind(
        "root",
        empty,
        entry=(case.flow.system, case.flow.initial),
    ):
        root = case.flow.root(case.flow.initial, runtime)

    root_attempt = next(
        item
        for item in runtime.attempts()
        if item.key.named_round == "root/raw_predict"
    )
    assert root_attempt.status == "completed"
    raw_evidence = runtime.data.get(
        root_attempt.result,
        case.evaluator.descriptor.result_codec,
    )
    raw_snapshot = native_score_artifact(raw_evidence)
    root_snapshot = native_score_artifact(root.evidence)
    root_native_artifacts = tuple(
        artifact
        for artifact in root.evidence.artifacts
        if artifact.content.codec == NATIVE_SCORES_ENVELOPE_CODEC.identity
    )
    assert root_native_artifacts == (root_snapshot,)
    assert root_snapshot.content != raw_snapshot.content
    assert root_snapshot.derived_from[0] == raw_snapshot.identity

    first = Work(
        identity="measured-step-1",
        parents=(root,),
        stage="advance",
        sample=Sample(
            protocol=case.flow.sample_protocol,
            replicate="0",
            seed=case.flow.contract.seed,
            index="1",
        ),
    )
    with runtime.bind(first.identity, empty, entry=first):
        first_outcome = case.flow.perform(first, runtime)
    first_successor = first_outcome.continuation.successor
    assert first_successor.candidate == root.candidate

    second = Work(
        identity="measured-step-2",
        parents=(first_successor,),
        stage="advance",
        sample=Sample(
            protocol=case.flow.sample_protocol,
            replicate="0",
            seed=case.flow.contract.seed,
            index="2",
        ),
    )
    with runtime.bind(second.identity, empty, entry=second):
        case.flow.perform(second, runtime)

    assert [step for step, _scores in observations] == [1, 2]
    for observed_step, scores in observations:
        del observed_step
        contribution, = scores.score_contributions
        assert contribution.source == "test/portable-native-flow/measured"
        assert contribution.metric_name == "post_raw_snapshot"
        assert contribution.value.dtype == np.float32
        np.testing.assert_array_equal(
            contribution.value,
            np.asarray([0.0, 0.25], dtype=np.float32),
        )
        residue_values = scores.per_proto_score_terms["A"]["measured_residue"]
        assert residue_values[0].dtype == np.float32
        np.testing.assert_array_equal(
            residue_values[0],
            np.asarray([0.1, 0.2], dtype=np.float32),
        )
        candidate, = scores.candidate_scores
        assert candidate.portable_measurement_step == 0
        np.testing.assert_array_equal(
            candidate.plddt["A"][0],
            np.asarray([0.3, 0.4], dtype=np.float32),
        )

    rejected_snapshot = native_score_artifact(first_successor.evidence)
    retained_snapshot = native_score_artifact(second.parents[0].evidence)
    assert first_successor.candidate == second.parents[0].candidate
    assert retained_snapshot.content == rejected_snapshot.content
    assert retained_snapshot.derived_from


def test_rejected_repeated_candidate_loads_exact_historical_state_blob(
    tmp_path: Path,
) -> None:
    script = {
        0: _ScriptedStep("AAAA", 1.0),
        1: _ScriptedStep("CAAA", 0.5, soft_accepted=True),
        2: _ScriptedStep("AAAA", 2.0, gate_passed=False),
    }
    case = _flow_fixture(tmp_path, script=script, metropolis=False)
    runtime = case.runtime
    empty = History(key_for(()), ())
    runtime.preflight(case.flow.requirements)
    with runtime.bind(
        "bootstrap",
        empty,
        entry=(case.flow.system, case.flow.initial),
    ):
        case.flow.bootstrap(runtime)
    with runtime.bind(
        "root",
        empty,
        entry=(case.flow.system, case.flow.initial),
    ):
        root = case.flow.root(case.flow.initial, runtime)

    first = Work(
        identity="manual-step-1",
        parents=(root,),
        stage="advance",
        sample=Sample(
            protocol=case.flow.sample_protocol,
            replicate="0",
            seed=case.flow.contract.seed,
            index="1",
        ),
    )
    with runtime.bind(first.identity, empty, entry=first):
        first_outcome = case.flow.perform(first, runtime)
    first_successor = first_outcome.continuation.successor

    second = Work(
        identity="manual-step-2",
        parents=(first_successor,),
        stage="advance",
        sample=Sample(
            protocol=case.flow.sample_protocol,
            replicate="0",
            seed=case.flow.contract.seed,
            index="2",
        ),
    )
    with runtime.bind(second.identity, empty, entry=second):
        second_outcome = case.flow.perform(second, runtime)
    second_successor = second_outcome.continuation.successor

    assert first_successor.candidate == second_successor.candidate
    first_state = case.flow._load_state(first_successor, runtime)
    second_state = case.flow._load_state(second_successor, runtime)
    assert first_state.candidate == second_state.candidate
    assert first_state.next_step == 2
    assert second_state.next_step == 3
    assert tuple(item.step for item in first_state.decision_state.scores) == (0, 1)
    assert tuple(item.step for item in second_state.decision_state.scores) == (
        0,
        1,
        2,
    )


def test_decision_state_roundtrip_restores_full_decisions_owner(tmp_path: Path) -> None:
    script = {0: _ScriptedStep("AAAA", 1.0)}
    case = _flow_fixture(tmp_path, script=script, metropolis=False)
    scores = _scores(tmp_path, 0, script[0])
    decisions = Decisions.create(scores, {"step": 0}, case.loss.compute_loss)
    state = snapshot_decision_state(decisions, case.bindings.data)
    assert state.schema == DECISION_STATE_SCHEMA
    restored = restore_decision_state(
        state,
        case.bindings.data,
        tmp_path / "restore",
    )
    assert restored.scores == decisions.scores
    assert restored.sequences == decisions.sequences
    assert restored.accepted_steps == decisions.accepted_steps
    assert restored.last_accepted == decisions.last_accepted
    assert restored.stage_history == decisions.stage_history
    assert restored.row_history == decisions.row_history
    expected = restored.evaluation_results[0]
    assert expected.sequence == "AAAA"
    assert expected.weighted == 1.0
    np.testing.assert_array_equal(expected.plddt["A"][0], scores.plddt["A"][0])


def test_portable_early_stop_preserves_current_decision_semantics(tmp_path: Path) -> None:
    script = {
        0: _ScriptedStep("AAAA", 1.0),
        1: _ScriptedStep("GAAA", 0.6, early_stop=True),
    }
    case = _flow_fixture(tmp_path, script=script, metropolis=False)
    np.random.seed(31)
    result = run_portable_native(case.flow, rounds=4, runtime=case.runtime)

    decisions = global_config.decisions
    assert decisions.scores == {0: 1.0, 1: 0.6}
    assert decisions.accepted_steps == [0]
    assert result.selected[0].assessment.values == (1.0,)
    checkpoint = case.runtime.load("run/state")
    policy = case.runtime.data.get(checkpoint["state"], AcceptedSuccessorStateCodec())
    assert policy.used == 1
    assert policy.stop_reason == "scientific_early_stop"
    assert case.evaluator_calls == [(0, "AAAA"), (1, "GAAA")]
    assert case.mutator_calls == [(1, "GAAA")]


def test_completed_portable_run_does_not_reload_or_replay(tmp_path: Path) -> None:
    script = {
        0: _ScriptedStep("AAAA", 1.0),
        1: _ScriptedStep("GAAA", 0.9),
    }
    case = _flow_fixture(tmp_path, script=script, metropolis=False)
    first = run_portable_native(case.flow, rounds=1, runtime=case.runtime)
    calls = (
        list(case.evaluator_calls),
        list(case.mutator_calls),
        list(case.evaluator_loads),
        list(case.mutator_loads),
    )

    second = run_portable_native(case.flow, rounds=1, runtime=case.runtime)

    assert second.selected == first.selected
    assert (
        case.evaluator_calls,
        case.mutator_calls,
        case.evaluator_loads,
        case.mutator_loads,
    ) == calls
    attempts = case.runtime.attempts()
    assert sum(item.key.named_round == "step/1/decide" for item in attempts) == 1
    assert all(item.status == "completed" for item in attempts)


def test_independent_flow_and_runtime_restart_rebinds_without_model_load(
    tmp_path: Path,
) -> None:
    script = {
        0: _ScriptedStep("AAAA", 1.0),
        1: _ScriptedStep("GAAA", 0.6),
    }
    first_case = _flow_fixture(tmp_path, script=script, metropolis=True)
    np.random.seed(0)
    first = run_portable_native(
        first_case.flow,
        rounds=1,
        runtime=first_case.runtime,
    )
    first_case.runtime.close()
    global_config.reset()

    restarted = _flow_fixture(tmp_path, script=script, metropolis=True)
    second = run_portable_native(
        restarted.flow,
        rounds=1,
        runtime=restarted.runtime,
    )

    assert second.selected == first.selected
    assert restarted.evaluator_calls == []
    assert restarted.mutator_calls == []
    assert restarted.evaluator_loads == []
    assert restarted.mutator_loads == []
    restarted.runtime.close()


def test_decision_completion_is_reused_after_cut_before_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = {
        0: _ScriptedStep("AAAA", 1.0),
        1: _ScriptedStep("GAAA", 0.6),
    }
    case = _flow_fixture(tmp_path, script=script, metropolis=True)
    original_project = case.flow._project
    projections = []

    def cut_second_projection(*args, **kwargs):
        projections.append(1)
        if len(projections) == 2:
            raise RuntimeError("simulated cut after decision completion")
        return original_project(*args, **kwargs)

    monkeypatch.setattr(case.flow, "_project", cut_second_projection)
    expected_rng = np.random.RandomState(0)
    expected_rng.uniform(0, 1)
    expected_rng_state = expected_rng.get_state()
    np.random.seed(0)
    with pytest.raises(RuntimeError, match="simulated cut"):
        run_portable_native(case.flow, rounds=1, runtime=case.runtime)

    calls_before_retry = (
        list(case.evaluator_calls),
        list(case.mutator_calls),
        list(case.evaluator_loads),
        list(case.mutator_loads),
    )
    result = run_portable_native(case.flow, rounds=1, runtime=case.runtime)

    assert result.selected[0].assessment.values == (0.6,)
    assert (
        case.evaluator_calls,
        case.mutator_calls,
        case.evaluator_loads,
        case.mutator_loads,
    ) == calls_before_retry
    attempts = case.runtime.attempts()
    assert sum(item.key.named_round == "step/1/decide" for item in attempts) == 1
    assert all(item.status == "completed" for item in attempts)
    actual_rng_state = np.random.get_state()
    assert actual_rng_state[0] == expected_rng_state[0]
    np.testing.assert_array_equal(actual_rng_state[1], expected_rng_state[1])
    assert actual_rng_state[2:] == expected_rng_state[2:]


def test_root_completion_is_reused_after_cut_before_coordinator_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _flow_fixture(
        tmp_path,
        script={0: _ScriptedStep("AAAA", 1.0)},
        metropolis=False,
    )
    original_project = case.flow._project

    def cut_before_root_commit(*args, **kwargs):
        raise RuntimeError("simulated cut after root prediction")

    monkeypatch.setattr(case.flow, "_project", cut_before_root_commit)
    with pytest.raises(RuntimeError, match="simulated cut"):
        run_portable_native(case.flow, rounds=0, runtime=case.runtime)
    assert case.evaluator_calls == [(0, "AAAA")]
    attempts_before = case.runtime.attempts()
    assert sum(item.key.named_round == "root/raw_predict" for item in attempts_before) == 1
    assert sum(item.key.named_round == "root/finalize_state" for item in attempts_before) == 1

    monkeypatch.setattr(case.flow, "_project", original_project)
    result = run_portable_native(case.flow, rounds=0, runtime=case.runtime)

    assert result.selected[0].assessment.values == (1.0,)
    assert case.evaluator_calls == [(0, "AAAA")]
    assert case.evaluator_loads == ["evaluator"]
    assert case.mutator_loads == ["mutator"]
    attempts = case.runtime.attempts()
    assert sum(item.key.named_round == "root/raw_predict" for item in attempts) == 1
    assert sum(item.key.named_round == "root/finalize_state" for item in attempts) == 1
