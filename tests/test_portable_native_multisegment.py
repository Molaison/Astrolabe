from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf, open_dict
import pytest

import run as run_module
from astrolabe.configuration import load_composed_config
from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import Protomer, SegmentManager
from astrolabe.design.legacy import LegacyContext
from astrolabe.design.native_mutation import make_mutator_operation
from astrolabe.design.native_mutation_state import (
    NATIVE_MUTATION_CONTEXT_CODEC,
    NATIVE_MUTATION_CONTEXT_SCHEMA,
    MutationContextPolicy,
    NativePendingMutation,
    bind_native_mutation_context,
    capture_native_mutation_contexts,
    native_mutation_segments,
    restore_native_mutation_context,
)
from astrolabe.design.native_prediction import make_evaluator_operation
from astrolabe.design.native_support import NativeBindings
from astrolabe.design.portable_native_math import (
    PORTABLE_NATIVE_MUTATION_CONTEXT_POLICY,
    _apply_pending_mutations,
    build_portable_native_math,
    validate_mutation_context_artifacts,
    validate_native_mutation_topology,
)
from astrolabe.design.portable_native_builder import (
    PortableNativeMathContext,
    build_portable_native_recipe,
)
from astrolabe.design.portable_native_flow import (
    PortableNativeFlow,
    PortableNativeSupport,
    run_portable_native,
    validate_static_nonpose_support,
)
from astrolabe.design.portable_native_state import (
    NATIVE_RUNTIME_CONTEXT_CODEC,
    NATIVE_STEP_STATE_CODEC,
    NUMERIC_TREE_CODEC,
    SCIENTIFIC_CONTRACT_SCHEMA,
    NativeRuntimeContext,
    ScientificRunContract,
    store_numeric_tree,
)
from astrolabe.design.runtime import ContractConflict
from astrolabe.design.store import DataStore
from astrolabe.design.types import (
    Assignment,
    Blob,
    Chemical,
    Polymer,
    Sample,
    SamplingPolicy,
    key_for,
)
from astrolabe.evaluators.utils import StandardScores
from astrolabe.losses.base import LossEngine
from astrolabe.mutators.random_substitution import RandomSubstitutionMutator
from astrolabe.score_contributions import ScoreContribution
from astrolabe.stage_manager import StageManager


PAPER_SEQUENCES = {
    "A": "+ACDEFGHIKL@shared",
    "B": "shared +MNPQRSTVWY",
}


def _init_globals() -> None:
    global_config.reset()
    global_config.init_const("segments", {})
    global_config.init_const("protomers", {})
    global_config.init_const("mutable_protomers", {})
    global_config.init_const("named_files", {})
    global_config.init_const("segment_manager", SegmentManager())


@pytest.fixture(autouse=True)
def _clean_globals():
    _init_globals()
    yield
    global_config.reset()


def _sample() -> Sample:
    return Sample(
        protocol=key_for(("portable-multisegment/sample/v1",)),
        replicate="0",
        seed=17,
        index="1",
    )


def _paper_bundle(tmp_path: Path):
    protomer_a = Protomer("A", PAPER_SEQUENCES["A"], "p")
    protomer_b = Protomer("B", PAPER_SEQUENCES["B"], "p")
    global_config.protomers.update({"A": protomer_a, "B": protomer_b})
    global_config.mutable_protomers.update(
        {"A": protomer_a, "B": protomer_b}
    )
    data = DataStore(tmp_path / "paper-data")
    translation = LegacyContext().capture_candidate(
        assemblies=("AB",),
        data=data,
    )
    return SimpleNamespace(data=data, translation=translation)


def _two_mutable_bundle(tmp_path: Path):
    protomer_a = Protomer("A", "ACDE@shared", "p")
    protomer_b = Protomer("B", "shared FGHI", "p")
    global_config.protomers.update({"A": protomer_a, "B": protomer_b})
    global_config.mutable_protomers.update(
        {"A": protomer_a, "B": protomer_b}
    )
    data = DataStore(tmp_path / "mutable-data")
    translation = LegacyContext().capture_candidate(
        assemblies=("AB",),
        data=data,
    )
    return SimpleNamespace(data=data, translation=translation)


def _capture(
    bundle,
    *,
    selectors: list[dict[str, object]] | None = None,
    contributions: list[ScoreContribution] | None = None,
):
    policy = MutationContextPolicy(
        "test/portable-native-multisegment/context/v1",
        NATIVE_MUTATION_CONTEXT_SCHEMA,
    )
    drafts = capture_native_mutation_contexts(
        bundle.data,
        policy=policy,
        runtime_config={
            "mutator_selector_score_contexts": (
                [{"global": "selector"}] if selectors is None else selectors
            ),
            "score_contributions": [] if contributions is None else contributions,
        },
    )
    return policy, drafts


def _bind_all(bundle, drafts, *, base, proposed, sample):
    return tuple(
        bind_native_mutation_context(
            draft,
            data=bundle.data,
            system=bundle.translation.system,
            base=base,
            proposed=proposed,
            sample=sample,
        )[1]
        for draft in drafts
    )


def _validate(bundle, artifacts, *, base, proposed, sample, policy):
    proposal = SimpleNamespace(
        candidate=proposed,
        artifacts=tuple(artifacts),
    )
    return validate_mutation_context_artifacts(
        proposal,
        before=SimpleNamespace(candidate=base),
        sample=sample,
        step=1,
        policy=policy,
        data=bundle.data,
        system=bundle.translation.system,
        contract_seed=17,
    )


def _segment_mutation_snapshot():
    return {
        name: tuple(segment.mutations or ())
        for name, segment in native_mutation_segments()
    }


class _PairLoss(LossEngine):
    def __init__(self) -> None:
        super().__init__(
            {"loss_terms": {}},
            runtime_context=global_config.build_runtime_context(),
            validate_config=False,
        )
        self.runtime_configs = []

    def compute_loss(self, eval_result, runtime_config):
        self.runtime_configs.append(dict(runtime_config))
        return float(eval_result.weighted), {
            "weighted": float(eval_result.weighted),
            "early_stop": 0.0,
            "soft_accepted": 0.0,
            "gate_passed": 1.0,
        }

    def is_early_stopping(self, scores_detail):
        return False

    def is_soft_accepted(self, scores_detail):
        return False

    def passes_acceptance_gate(self, scores_detail):
        return True


class _PairEvaluator:
    def __init__(self, tmp_path: Path, calls: list[tuple[int, tuple[str, ...]]]):
        self.tmp_path = tmp_path
        self.calls = calls

    def evaluate(self, inputs, runtime_config):
        step = int(runtime_config["step"])
        assembly = inputs["AB"]
        sequences = tuple(chain.sequence for chain in assembly.chains)
        self.calls.append((step, sequences))
        path = self.tmp_path / f"pair_step_{step}.cif"
        path.write_text(f"data_pair_step_{step}\n", encoding="utf-8")
        length = sum(len(sequence) for sequence in sequences)
        scores = StandardScores(
            names=["AB"],
            plddt={
                "AB": [
                    np.full(length, 0.9 - (0.1 * step), dtype=np.float32)
                ]
            },
            complex_paths={"AB": path},
        )
        scores.weighted = 1.0 - (0.1 * step)
        scores.early_stop = False
        scores.soft_accepted = False
        scores.gate_passed = True
        return scores

    def shutdown(self):
        pass


def _paper_flow_fixture(tmp_path: Path):
    bundle = _paper_bundle(tmp_path)
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("initial_temperature", 1.0)
    global_config.init_const("use_metropolis_acceptance", False)
    global_config.init_const("structure_start", None)
    global_config.init_const("exclude_AA", [])
    global_config.set_variable("ignored_complexes", [])
    global_config.set_variable("negative_design_complexes", [])

    loss = _PairLoss()
    stages = OmegaConf.create(
        {
            "default": {
                "oligo": ["AB"],
                "mutation_num": 1,
                "weighted_score_expression": "weighted",
                "early_stopping": "early_stop > 0.0",
                "acceptance_threshold": "soft_accepted > 0.0",
                "evaluator": "af3",
                "mutators": ["mcmc"],
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

    bindings = NativeBindings(
        LegacyContext(),
        bundle.translation,
        bundle.data,
        tmp_path / "work",
    )
    sampling = SamplingPolicy(
        key_for(("portable-multisegment/sampling/v1",)),
        bundle.data.put(
            {"mutation_num": 1, "seed": 17},
            NUMERIC_TREE_CODEC,
        ),
    )
    evaluator_calls = []
    mutator_calls = []
    mutator_loads = []
    evaluator_loads = []

    def evaluator_factory():
        evaluator_loads.append(1)
        return _PairEvaluator(tmp_path, evaluator_calls)

    def mutator_factory():
        mutator_loads.append(1)
        return RandomSubstitutionMutator(
            mutation_method="frequency_adjusted",
            select_positions="plddt",
            avoid_current_residue=True,
        )

    def configure(mutator, runtime_config, **_kwargs):
        global_config.set_variable("mutation_num", 1)
        runtime_config["mutation_num"] = 1
        mutator.mutation_weight = 1.0
        mutator_calls.append(int(runtime_config["step"]))

    evaluator = make_evaluator_operation(
        "af3",
        evaluator_factory,
        bindings,
        version="portable-multisegment-evaluator/v1",
        model_identity=key_for(("portable-multisegment/evaluator-model/v1",)),
    )
    mutator = make_mutator_operation(
        "mcmc",
        mutator_factory,
        bindings,
        sampling_policy=sampling,
        version="portable-multisegment-mutator/v1",
        model_identity=key_for(("portable-multisegment/mutator-model/v1",)),
        options={"protomer_scope": "mutable"},
        configure=configure,
        configure_identity="portable-multisegment/configure/v1",
        mutation_context=PORTABLE_NATIVE_MUTATION_CONTEXT_POLICY,
    )

    support = PortableNativeSupport(("default",))
    static_stage = validate_static_nonpose_support(support)
    runtime_context = NativeRuntimeContext(
        schema="astrolabe.portable-native.runtime-context/v1",
        selector_and_contribution_context=store_numeric_tree(bundle.data, {}),
        decision_context=store_numeric_tree(
            bundle.data,
            {"use_metropolis_acceptance": False},
        ),
    )
    runtime_context_blob = bundle.data.put(
        runtime_context,
        NATIVE_RUNTIME_CONTEXT_CODEC,
    )
    contract = ScientificRunContract(
        schema=SCIENTIFIC_CONTRACT_SCHEMA,
        config_identity=key_for(("portable-multisegment/config/v1",)),
        source_identity=key_for(("portable-multisegment/source/v1",)),
        system_identity=bundle.translation.system.identity,
        resource_manifest=key_for(("portable-multisegment/resources/v1",)),
        static_stage=static_stage,
        objective=key_for(("portable-multisegment/objective/v1", "weighted")),
        component_identities=(
            evaluator.descriptor.identity,
            mutator.descriptor.identity,
        ),
        seed=17,
        acceptance_protocol=sampling.identity,
        codec_identities=(
            NATIVE_STEP_STATE_CODEC.identity,
            NUMERIC_TREE_CODEC.identity,
        ),
        recovery_capabilities=("raw_predict", "decision", "model_ready"),
    )
    sample_protocol = key_for(("portable-multisegment/sample/v1",))
    math_context = PortableNativeMathContext(
        args=SimpleNamespace(steps=2),
        prepared=None,
        loss_computer=loss,
        stage_manager=stage_manager,
        bindings=bindings,
        contract=contract,
        initial_candidate=bundle.translation.candidate,
        runtime_context=runtime_context_blob,
        root_runtime_config={
            "mutator_stack_pre_update": False,
            "state_conflict_diagnostics": None,
        },
        sample_protocol=sample_protocol,
        run_module=run_module,
        mutation_context=PORTABLE_NATIVE_MUTATION_CONTEXT_POLICY,
    )
    math = build_portable_native_math(math_context)
    flow = PortableNativeFlow(
        contract=contract,
        support=support,
        system=bundle.translation.system,
        initial_candidate=bundle.translation.candidate,
        evaluator=evaluator,
        mutator=mutator,
        bindings=bindings,
        sampling_policy=sampling,
        assemblies=("AB",),
        math=math,
        runtime_context=runtime_context_blob,
        sample_protocol=sample_protocol,
        initialization_order=("mutator", "evaluator"),
    )
    from astrolabe.design.runtime import Runtime

    runtime = Runtime(
        flow.operations,
        root=tmp_path / "runtime",
        data=bundle.data,
        scientific_contract=key_for(("portable-multisegment/runtime/v1", flow.identity)),
    )
    return SimpleNamespace(
        bundle=bundle,
        flow=flow,
        runtime=runtime,
        loss=loss,
        evaluator_calls=evaluator_calls,
        mutator_calls=mutator_calls,
        evaluator_loads=evaluator_loads,
        mutator_loads=mutator_loads,
    )


def test_actual_paper_fixture_captures_two_fixed_segments_with_shared_alias(
    tmp_path: Path,
) -> None:
    bundle = _paper_bundle(tmp_path)
    segments = native_mutation_segments()

    assert [name for name, _segment in segments] == ["A_1", "B_2"]
    assert all(segment.is_fixed for _name, segment in segments)
    assert global_config.segments["shared"] is global_config.segments["A_1"]
    assert global_config.segments["B_1"] is global_config.segments["A_1"]
    occurrence_variables = tuple(
        occurrence.variable for occurrence in bundle.translation.system.occurrences
    )
    assert occurrence_variables == ("A_1", "A_1", "B_2")

    policy, drafts = _capture(bundle)
    sample = _sample()
    artifacts = _bind_all(
        bundle,
        drafts,
        base=bundle.translation.candidate,
        proposed=bundle.translation.candidate,
        sample=sample,
    )
    states = _validate(
        bundle,
        artifacts,
        base=bundle.translation.candidate,
        proposed=bundle.translation.candidate,
        sample=sample,
        policy=policy,
    )

    assert tuple(state.variable for state in states) == ("A_1", "B_2")
    assert all(state.pending_mutations == () for state in states)
    assert states[0].selector_contexts == states[1].selector_contexts
    assert states[0].score_contributions == states[1].score_contributions
    assert all(
        artifact.content.codec == NATIVE_MUTATION_CONTEXT_CODEC.identity
        for artifact in artifacts
    )


def test_actual_paper_config_cold_builder_keeps_both_native_owners_unloaded(
    tmp_path: Path,
) -> None:
    global_config.reset()
    cfg = load_composed_config("config/paper/evaluate_af3.yaml")
    model_dir = tmp_path / "af3"
    model_dir.mkdir()
    (model_dir / "weights.bin").write_bytes(b"cpu-shape-only")
    with open_dict(cfg):
        cfg.task_name = "paper-cold-builder"
        cfg.out_dir = str(tmp_path / "run")
        cfg.raw_model_results_dir = str(tmp_path / "raw")
        cfg.steps = 2
        cfg.repeats = 1
        cfg.seed = 17
        cfg.types = {"A": "p", "B": "p"}
        cfg.seqs = dict(PAPER_SEQUENCES)
        cfg.homologies = {}
        cfg.mutation_num = 1
        cfg.use_metropolis_acceptance = False
        cfg.skip_complex_poses = True
        cfg.enable_pyrosetta = False
        cfg.structure_start = None
        cfg.recovery = {"mode": "durable", "resume": False}
        cfg.initializer.mode = "random"
        cfg.predocker.predocker_terms = {}
        cfg.evaluator.enabled_aliases = ["Alphafold3"]
        cfg.evaluator.evaluator_terms.Alphafold3.model_dir = str(model_dir)
        cfg.evaluator.evaluator_terms.Alphafold3.num_seeds = 1
        cfg.evaluator.evaluator_terms.Alphafold3.num_recycles = 1
        cfg.evaluator.evaluator_terms.Alphafold3.num_samples = 1
        cfg.evaluator.evaluator_terms.Alphafold3.run_data_pipeline = False
        cfg.evaluator.evaluator_terms.Alphafold3.run_inference = True
        cfg.evaluator.alphafold3.model_dir = str(model_dir)
        cfg.evaluator.alphafold3.num_seeds = 1
        cfg.evaluator.alphafold3.num_recycles = 1
        cfg.evaluator.alphafold3.num_samples = 1
        cfg.evaluator.alphafold3.run_data_pipeline = False
        cfg.evaluator.alphafold3.run_inference = True
        cfg.mutator.enabled_aliases = ["MCMC"]
        cfg.mutator.mutator_terms.MCMC._type = "MCMC"
        cfg.stages.default.oligo = ["AB"]
        cfg.stages.default.evaluator = "Alphafold3"
        cfg.stages.default.mutators = ["MCMC"]
        cfg.stages.default.mutation_num = 1
        cfg.stages.default.weighted_score_expression = "plddt"

    recipe = build_portable_native_recipe(
        cfg,
        math_factory=build_portable_native_math,
        run_module=run_module,
    )
    try:
        assert recipe.flow.evaluator.owner.ready is False
        assert recipe.flow.mutator.owner.ready is False
        assert recipe.contract.system_identity == recipe.flow.system.identity
        assert tuple(
            dict.fromkeys(
                occurrence.variable
                for occurrence in recipe.flow.system.occurrences
            )
        ) == ("A_1", "B_2")
        assert recipe.flow.mutator.mutation_context is not None
    finally:
        recipe.close()


def test_topology_preflight_rejects_native_segment_without_physical_occurrence(
    tmp_path: Path,
) -> None:
    bundle = _paper_bundle(tmp_path)
    partial = LegacyContext().capture_candidate(
        assemblies=("A",),
        data=bundle.data,
    )

    with pytest.raises(ValueError, match="lack selected physical polymer"):
        validate_native_mutation_topology(
            system=partial.system,
            candidate=partial.candidate,
        )


def test_plural_context_rejects_changed_chemical_assignment(tmp_path: Path) -> None:
    bundle = _paper_bundle(tmp_path)
    base_graph = Blob(key_for(("chemical/base",)), "test/chemical/v1")
    changed_graph = Blob(key_for(("chemical/changed",)), "test/chemical/v1")
    base_assignments = bundle.translation.candidate.assignments + (
        Assignment("ligand", Chemical(base_graph)),
    )
    changed_assignments = bundle.translation.candidate.assignments + (
        Assignment("ligand", Chemical(changed_graph)),
    )
    base = SimpleNamespace(
        identity=bundle.translation.candidate.identity,
        assignments=base_assignments,
    )
    changed = SimpleNamespace(
        identity=bundle.translation.candidate.identity,
        assignments=changed_assignments,
    )

    with pytest.raises(ContractConflict, match="non-polymer assignment"):
        validate_mutation_context_artifacts(
            SimpleNamespace(candidate=changed, artifacts=()),
            before=SimpleNamespace(candidate=base),
            sample=None,
            step=1,
            policy=None,
            data=bundle.data,
            system=bundle.translation.system,
            contract_seed=17,
        )


@pytest.mark.parametrize(
    ("variant", "message"),
    [
        ("missing", "order/coverage"),
        ("duplicate", "duplicate mutation-context artifacts"),
        ("reordered", "order/coverage"),
    ],
)
def test_plural_context_rejects_partial_duplicate_or_reordered_coverage(
    tmp_path: Path,
    variant: str,
    message: str,
) -> None:
    bundle = _paper_bundle(tmp_path)
    policy, drafts = _capture(bundle)
    sample = _sample()
    artifacts = _bind_all(
        bundle,
        drafts,
        base=bundle.translation.candidate,
        proposed=bundle.translation.candidate,
        sample=sample,
    )
    before = _segment_mutation_snapshot()
    invalid = {
        "missing": artifacts[:1],
        "duplicate": (artifacts[0], artifacts[0]),
        "reordered": tuple(reversed(artifacts)),
    }[variant]

    with pytest.raises((ContractConflict, ValueError), match=message):
        _validate(
            bundle,
            invalid,
            base=bundle.translation.candidate,
            proposed=bundle.translation.candidate,
            sample=sample,
            policy=policy,
        )

    assert _segment_mutation_snapshot() == before


def test_plural_context_rejects_mismatched_global_selector_records(
    tmp_path: Path,
) -> None:
    bundle = _paper_bundle(tmp_path)
    policy, first = _capture(bundle, selectors=[{"global": "first"}])
    _second_policy, second = _capture(
        bundle,
        selectors=[{"global": "second"}],
    )
    sample = _sample()
    artifacts = _bind_all(
        bundle,
        (first[0], second[1]),
        base=bundle.translation.candidate,
        proposed=bundle.translation.candidate,
        sample=sample,
    )
    before = _segment_mutation_snapshot()

    with pytest.raises(ContractConflict, match="global selector"):
        _validate(
            bundle,
            artifacts,
            base=bundle.translation.candidate,
            proposed=bundle.translation.candidate,
            sample=sample,
            policy=policy,
        )

    assert _segment_mutation_snapshot() == before


def test_tampered_second_context_is_rejected_before_any_segment_mutation(
    tmp_path: Path,
) -> None:
    bundle = _paper_bundle(tmp_path)
    policy, drafts = _capture(bundle)
    sample = _sample()
    artifacts = list(
        _bind_all(
            bundle,
            drafts,
            base=bundle.translation.candidate,
            proposed=bundle.translation.candidate,
            sample=sample,
        )
    )
    state = restore_native_mutation_context(artifacts[1], bundle.data)
    damaged_content = bundle.data.put(
        replace(state, working_sequence="MUTATED-LENGTH"),
        NATIVE_MUTATION_CONTEXT_CODEC,
    )
    artifacts[1] = replace(artifacts[1], content=damaged_content)
    before = _segment_mutation_snapshot()

    with pytest.raises((ContractConflict, ValueError)):
        _validate(
            bundle,
            artifacts,
            base=bundle.translation.candidate,
            proposed=bundle.translation.candidate,
            sample=sample,
            policy=policy,
        )

    assert _segment_mutation_snapshot() == before


def test_two_ordered_pending_mutable_contexts_apply_together_and_reuse_global_payload(
    tmp_path: Path,
) -> None:
    bundle = _two_mutable_bundle(tmp_path)
    contribution = ScoreContribution(
        source="test/portable-native-multisegment",
        metric_name="ordered",
        value=np.asarray([17.0, 23.0], dtype=np.float32),
    )
    policy, drafts = _capture(bundle, contributions=[contribution])
    sample = _sample()
    assignments = []
    changed_drafts = []
    for draft, assignment in zip(
        drafts,
        bundle.translation.candidate.assignments,
        strict=True,
    ):
        assert isinstance(assignment.value, Polymer)
        position = 1 if draft.variable == "A_1" else 2
        target = (
            "A"
            if assignment.value.tokens[position] != "A"
            else "C"
        )
        tokens = list(assignment.value.tokens)
        tokens[position] = target
        working = "".join(tokens)
        assignments.append(
            Assignment(
                assignment.variable,
                Polymer(
                    assignment.value.kind,
                    tuple(tokens),
                    assignment.value.modifications,
                ),
            )
        )
        changed_drafts.append(
            replace(
                draft,
                pending_mutations=(
                    NativePendingMutation(
                        position=position,
                        from_token=assignment.value.tokens[position],
                        to_token=target,
                        entropy=0.25,
                        score=0.75,
                    ),
                ),
                working_sequence=working,
            )
        )
    proposed = bundle.translation.system.edit(
        bundle.translation.candidate,
        tuple(assignments),
        data=bundle.data,
    )
    artifacts = _bind_all(
        bundle,
        tuple(changed_drafts),
        base=bundle.translation.candidate,
        proposed=proposed,
        sample=sample,
    )
    states = _validate(
        bundle,
        artifacts,
        base=bundle.translation.candidate,
        proposed=proposed,
        sample=sample,
        policy=policy,
    )

    runtime_config = {}
    _apply_pending_mutations(
        states,
        runtime_config=runtime_config,
        data=bundle.data,
        system=bundle.translation.system,
        candidate=bundle.translation.candidate,
    )

    segments = native_mutation_segments()
    assert [name for name, _segment in segments] == ["A_1", "B_2"]
    assert [
        (mutation[0], mutation[1], mutation[2])
        for _name, segment in segments
        for mutation in segment.mutations
    ] == [
        (1, states[0].sequence_before_pending[1], states[0].working_sequence[1]),
        (2, states[1].sequence_before_pending[2], states[1].working_sequence[2]),
    ]
    assert runtime_config["mutator_selector_score_contexts"] == [
        {"global": "selector"}
    ]
    restored_contribution, = runtime_config["score_contributions"]
    np.testing.assert_array_equal(
        restored_contribution.value,
        np.asarray([17.0, 23.0], dtype=np.float32),
    )


def test_paper_noop_mcmc_makes_child_prediction_and_resumes_completed_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _paper_flow_fixture(tmp_path)
    original_project = first.flow._project
    projections = []

    def cut_after_decision(*args, **kwargs):
        projections.append(1)
        if len(projections) == 2:
            raise RuntimeError("simulated cut after plural decision completion")
        return original_project(*args, **kwargs)

    monkeypatch.setattr(first.flow, "_project", cut_after_decision)
    np.random.seed(23)
    with pytest.raises(RuntimeError, match="plural decision completion"):
        run_portable_native(first.flow, rounds=1, runtime=first.runtime)

    assert [step for step, _sequences in first.evaluator_calls] == [0, 1]
    assert first.evaluator_calls[0][1] == first.evaluator_calls[1][1]
    assert first.mutator_calls == [1]
    assert first.evaluator_loads == [1]
    assert first.mutator_loads == [1]
    attempts = first.runtime.attempts()
    for named_round in (
        "root/raw_predict",
        "step/1/propose",
        "step/1/raw_predict",
        "step/1/decide",
    ):
        matching = [item for item in attempts if item.key.named_round == named_round]
        assert len(matching) == 1
        assert matching[0].status == "completed"
    proposal_attempt = next(
        item for item in attempts if item.key.named_round == "step/1/propose"
    )
    proposals = first.bundle.data.get(
        proposal_attempt.result,
        first.flow.mutator.descriptor.result_codec,
    )
    artifacts = proposals.items[0].artifacts
    assert len(artifacts) == 2
    assert all(
        artifact.content.codec == NATIVE_MUTATION_CONTEXT_CODEC.identity
        for artifact in artifacts
    )
    saved_rng = np.random.get_state()
    first.runtime.close()

    _init_globals()
    resumed = _paper_flow_fixture(tmp_path)
    np.random.seed(91)
    result = run_portable_native(
        resumed.flow,
        rounds=1,
        runtime=resumed.runtime,
    )

    assert result.selected
    assert resumed.evaluator_calls == []
    assert resumed.mutator_calls == []
    assert resumed.evaluator_loads == []
    assert resumed.mutator_loads == []
    attempts = resumed.runtime.attempts()
    for named_round in (
        "root/raw_predict",
        "step/1/propose",
        "step/1/raw_predict",
        "step/1/decide",
    ):
        assert (
            sum(item.key.named_round == named_round for item in attempts) == 1
        )
    actual_rng = np.random.get_state()
    assert saved_rng[0] == actual_rng[0]
    np.testing.assert_array_equal(saved_rng[1], actual_rng[1])
    assert saved_rng[2:] == actual_rng[2:]
