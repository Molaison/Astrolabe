from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import astrolabe.design.native_mutation as native_mutation_module
from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import (
    Protomer,
    ProtomerTypes,
    Segment,
    SegmentManager,
)
from astrolabe.design.legacy import LegacyContext, translate_legacy_state
from astrolabe.design.native_mutation import (
    UnsupportedNativeMutation,
    make_mutator_operation,
)
from astrolabe.design.native_mutation_state import (
    NATIVE_MUTATION_CONTEXT_CODEC,
    NATIVE_MUTATION_CONTEXT_SCHEMA,
    MutationContextPolicy,
    NativePendingMutation,
    UnsupportedMutationContext,
    capture_native_mutation_context,
    capture_native_mutation_contexts,
    native_mutation_segments,
    restore_native_mutation_context,
)
from astrolabe.design.native_scores import snapshot_scores
from astrolabe.design.native_support import NativeBindings, numeric_step
from astrolabe.design.operations import INVOKE
from astrolabe.design.runtime import Runtime
from astrolabe.design.store import ArrayCodec, DataStore, JSONCodec
from astrolabe.design.types import (
    Artifact,
    Assignment,
    Candidate,
    Evidence,
    EvidenceUse,
    History,
    PackedStates,
    Polymer,
    Proposal,
    ProposeRequest,
    Sample,
    SamplingPolicy,
    Selection,
    Subject,
    Requirement,
    key_for,
    manifest,
    Dependency,
)
from astrolabe.evaluators.utils import StandardScores
from astrolabe.mutators.random_substitution import RandomSubstitutionMutator
from astrolabe.score_contributions import ScoreContribution


JSON = JSONCodec()


@pytest.fixture(autouse=True)
def clean_legacy_globals():
    global_config.reset()
    global_config.init_const("segments", {})
    global_config.init_const("protomers", {})
    global_config.init_const("mutable_protomers", {})
    global_config.init_const("named_files", {})
    global_config.init_const("segment_manager", SegmentManager())
    yield
    global_config.reset()


class FakeNativeMutator:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = []
        self.shutdown_calls = 0

    def mutate(self, protomers, scores, runtime_config):
        self.calls.append((tuple(protomers), scores, dict(runtime_config)))
        if self.fail:
            raise RuntimeError("native mutation failed")
        step = int(runtime_config["step"])
        replacement = "T" if step % 2 else "G"
        for protomer in protomers.values():
            segment = global_config.segments[protomer.components[0]]
            segment.mutations = [
                (
                    1,
                    segment.sequence[1],
                    replacement,
                    0.25,
                    0.75,
                )
            ]

    def shutdown(self):
        self.shutdown_calls += 1


def _sample(index: str, *, seed: int = 29) -> Sample:
    return Sample(
        protocol=key_for("protocol/legacy-mutator"),
        replicate="job-from-request",
        seed=seed,
        index=index,
    )


def _sampling(data: DataStore) -> SamplingPolicy:
    return SamplingPolicy(
        key_for("policy/native-mutator"),
        data.put({"temperature": 1.0, "mutation_count": 1}, JSON),
    )


def _state(data: DataStore, *, fixed: bool = False):
    a = Protomer("A", "+GC@shared" if fixed else "AC@shared", "p")
    b = Protomer("B", "shared", "p")
    protomers = {"A": a, "B": b}
    global_config.protomers.update(protomers)
    global_config.mutable_protomers.update({"A": a})
    context = LegacyContext()
    translation = context.capture_candidate(assemblies=("A", "B"), data=data)
    return protomers, context, translation


def _multi_state(
    data: DataStore,
    *,
    fixed_first: bool = False,
    fixed_second: bool = False,
):
    first = ("+" if fixed_first else "") + "AC@shared"
    second = ("+" if fixed_second else "") + "GT@second"
    a = Protomer("A", f"{first} {second}", "p")
    b = Protomer("B", "shared second", "p")
    protomers = {"A": a, "B": b}
    global_config.protomers.update(protomers)
    global_config.mutable_protomers.update(protomers)
    context = LegacyContext()
    translation = context.capture_candidate(
        assemblies=("A", "B"),
        data=data,
    )
    return protomers, context, translation


def _multi_working(
    translation,
    data: DataStore,
    *,
    first: tuple[str, ...] = ("A", "G"),
    second: tuple[str, ...] = ("T", "T"),
):
    replacements = tuple(
        Assignment(
            assignment.variable,
            Polymer(
                assignment.value.kind,
                first if assignment.variable == "A_1" else second,
                (),
            ),
        )
        for assignment in translation.candidate.assignments
    )
    return translation.system.edit(
        translation.candidate,
        replacements,
        data=data,
    )


def _evidence(
    data: DataStore,
    translation,
    *,
    sample: Sample | None = None,
) -> Evidence:
    sample = sample or _sample("0", seed=5)
    scores = StandardScores(
        names=["A", "B"],
        plddt={
            "A": [
                np.asarray([0.2, 0.8], dtype=np.float32),
            ],
            "B": [
                np.asarray([0.4, 0.6], dtype=np.float32),
            ],
        },
        complex_paths={},
        per_proto_scores={
            "A": [np.asarray([0.3, 0.7], dtype=np.float64)],
        },
    )
    content = snapshot_scores(scores, data)
    inputs = manifest(
        (
            Dependency("system", translation.system.identity),
            Dependency("candidate", translation.candidate.identity),
            Dependency("sample", key_for(sample)),
        )
    )
    artifact = Artifact(
        identity=key_for(
            (
                "artifact/native-scores/test",
                translation.candidate.identity,
                translation.system.occurrences,
                sample,
                content,
                inputs,
            )
        ),
        candidate=translation.candidate.identity,
        occurrences=translation.system.occurrences,
        sample=sample,
        content=content,
        derived_from=(),
        inputs=inputs,
    )
    return Evidence(
        identity=key_for(
            (
                "evidence/native-scores/test",
                translation.candidate.identity,
                artifact,
            )
        ),
        evaluated_candidate=translation.candidate.identity,
        artifacts=(artifact,),
        measurements=(),
        inputs=inputs,
    )


def _request(
    translation,
    evidence: Evidence,
    sampling: SamplingPolicy,
    *,
    index: str = "3",
    seed: int = 29,
    parents: tuple[Candidate, ...] | None = None,
    working: Candidate | None = None,
) -> tuple[ProposeRequest, Candidate]:
    if working is None:
        working = translation.system.edit(
            translation.candidate,
            (Assignment("A_1", Polymer("protein", ("A", "G"), ())),),
            data=DataStore(),
        )
    parent_values = parents or (translation.candidate, working)
    request = ProposeRequest(
        system=translation.system,
        parents=parent_values,
        working=working,
        evidence=(
            EvidenceUse(
                evidence=evidence,
                relation="base_guidance",
                working_candidate=working.identity,
            ),
        ),
        selection=(),
        packed=None,
        sampling=sampling,
        sample=_sample(index, seed=seed),
    )
    return request, working


def _operation(
    *,
    factory,
    bindings,
    sampling,
    configure=None,
    configure_identity=None,
    options=None,
    schema=None,
    runtime_config_factory=None,
    mutation_context=None,
    invoke=None,
    invoke_identity=None,
):
    return make_mutator_operation(
        "legacy_mutator",
        factory,
        bindings,
        sampling_policy=sampling,
        version="1",
        model_identity=key_for("model/fake-native-mutator"),
        options=options
        or {
            "protomer_scope": "mutable",
            "checkpoint": "checkpoint-7",
        },
        schema=schema
        or {
            "type": "object",
            "properties": {
                "protomer_scope": {
                    "type": "string",
                    "enum": ["mutable", "all"],
                },
                "checkpoint": {"type": "string"},
            },
            "required": ["protomer_scope", "checkpoint"],
            "additionalProperties": False,
        },
        configure=configure,
        configure_identity=configure_identity,
        runtime_config_factory=runtime_config_factory,
        mutation_context=mutation_context,
        invoke=invoke,
        invoke_identity=invoke_identity,
    )


def _bootstrap(operation) -> None:
    prepared = operation.initialization.prepare(operation.initialization_request)
    ready = operation.initialization.invoke(prepared)
    operation.initialization.on_result(ready)


def test_native_mutation_loads_once_preserves_parents_evidence_and_cached_scores(
    tmp_path: Path,
):
    data = DataStore(tmp_path / "data")
    protomers, context, translation = _state(data)
    evidence = _evidence(data, translation)
    sampling = _sampling(data)
    loaded = []
    configure_events = []
    events = []

    def factory():
        loaded.append("load")
        events.append(("load",))
        return FakeNativeMutator()

    def configure(mutator, runtime_config, *, request, sample, step, work_root):
        events.append(("configure", step))
        configure_events.append(
            ("configure", sample.seed, sample.replicate, step, work_root)
        )
        runtime_config["seed"] = sample.seed
        runtime_config["batch_size"] = 2
        runtime_config["job_name"] = sample.replicate

    operation = _operation(
        factory=factory,
        bindings=NativeBindings(context, translation, data, tmp_path / "work"),
        sampling=sampling,
        configure=configure,
        configure_identity="owner/configure/v1",
        runtime_config_factory=lambda request: {
            "step": numeric_step(request.sample),
            "source": "owner",
        },
    )
    assert operation.initialization is operation.owner.initialization
    assert operation.initialization_request is operation.owner.initialization_request
    _bootstrap(operation)

    first_request, working = _request(translation, evidence, sampling, index="3")
    second_request, _ = _request(
        translation,
        evidence,
        sampling,
        index="4",
        seed=31,
        parents=(translation.candidate,),
    )
    first = operation.invoke(operation.prepare(first_request))
    second = operation.invoke(operation.prepare(second_request))

    assert loaded == ["load"]
    first_proposal = first.items[0]
    second_proposal = second.items[0]
    assert first_proposal.parents == tuple(
        parent.identity for parent in first_request.parents
    )
    assert second_proposal.parents == (translation.candidate.identity,)
    assert first_proposal.base == working.identity
    assert first_proposal.used_evidence == (evidence.identity,)
    assert second_proposal.used_evidence == (evidence.identity,)
    assert first_proposal.protocol == sampling.identity
    assert first_proposal.protocol != first_request.sample.protocol
    assert first_request.system.variables == ("A_1",)
    assert len(first_request.system.occurrences) == 2
    assert len(first_proposal.candidate.assignments) == 1
    assert first_proposal.candidate.assignments[0].value.tokens == ("A", "T")
    assert first_proposal.candidate != second_proposal.candidate
    assert first_proposal.identity != second_proposal.identity

    score_calls = [call[1] for call in operation.owner.get().calls]
    assert all(isinstance(scores, StandardScores) for scores in score_calls)
    assert score_calls[0].plddt["A"][0].dtype == np.float32
    np.testing.assert_array_equal(
        score_calls[0].plddt["A"][0],
        np.asarray([0.2, 0.8], dtype=np.float32),
    )
    runtime_calls = [call[2] for call in operation.owner.get().calls]
    assert [call["step"] for call in runtime_calls] == [3, 4]
    assert [call["seed"] for call in runtime_calls] == [29, 31]
    assert [call["job_name"] for call in runtime_calls] == [
        "job-from-request",
        "job-from-request",
    ]
    assert configure_events == [
        ("configure", 29, "job-from-request", 3, tmp_path / "work"),
        ("configure", 31, "job-from-request", 4, tmp_path / "work"),
    ]
    assert events == [
        ("load",),
        ("configure", 3),
        ("configure", 4),
    ]
    # Activation is isolated and never leaks the working or proposed sequence.
    assert global_config.protomers == protomers
    assert global_config.segments["A_1"].sequence == "AC"
    assert global_config.segments["A_1"].mutations in (None, [])
    loaded_mutator = operation.owner.get()
    operation.shutdown()
    assert loaded_mutator.shutdown_calls == 1


def test_fixed_domain_rejects_pending_mutation_and_restores_globals(tmp_path: Path):
    data = DataStore(tmp_path / "data")
    _, context, translation = _state(data, fixed=True)
    segment = global_config.segments["A_1"]
    previous_pending = [(0, "G", "C", 0.0, 0.0)]
    segment.mutations = list(previous_pending)
    evidence = _evidence(data, translation)
    sampling = _sampling(data)
    factory_calls = []
    operation = _operation(
        factory=lambda: factory_calls.append(1) or FakeNativeMutator(),
        bindings=NativeBindings(context, translation, data, tmp_path / "work"),
        sampling=sampling,
        runtime_config_factory=lambda request: {"step": numeric_step(request.sample)},
    )
    _bootstrap(operation)
    request, _ = _request(
        translation,
        evidence,
        sampling,
        index="3",
        working=translation.candidate,
    )

    with pytest.raises(ValueError, match="fixed token"):
        operation.invoke(operation.prepare(request))

    assert factory_calls == [1]
    assert global_config.segments["A_1"].sequence == "GC"
    assert global_config.segments["A_1"].mutations == previous_pending


def test_native_failure_rolls_back_pending_and_source_sequence(tmp_path: Path):
    data = DataStore(tmp_path / "data")
    _, context, translation = _state(data)
    segment = global_config.segments["A_1"]
    previous_pending = [(0, "A", "G", 0.1, 0.2)]
    segment.mutations = list(previous_pending)
    evidence = _evidence(data, translation)
    sampling = _sampling(data)
    operation = _operation(
        factory=lambda: FakeNativeMutator(fail=True),
        bindings=NativeBindings(context, translation, data, tmp_path / "work"),
        sampling=sampling,
        runtime_config_factory=lambda request: {"step": numeric_step(request.sample)},
    )
    _bootstrap(operation)
    request, _ = _request(translation, evidence, sampling)

    with pytest.raises(RuntimeError, match="native mutation failed"):
        operation.invoke(operation.prepare(request))

    assert global_config.segments["A_1"].sequence == "AC"
    assert global_config.segments["A_1"].mutations == previous_pending


@pytest.mark.parametrize("scope", ["mutable", "all"])
def test_protomer_scope_is_explicit_and_defaults_to_mutable(tmp_path: Path, scope):
    data = DataStore(tmp_path / scope)
    protomers, context, translation = _state(data)
    evidence = _evidence(data, translation)
    sampling = _sampling(data)
    seen = []

    class ScopeRecorder(FakeNativeMutator):
        def mutate(self, selected, scores, runtime_config):
            seen.append(tuple(selected))
            super().mutate(selected, scores, runtime_config)

    options = {"checkpoint": "checkpoint-7"}
    if scope == "all":
        options["protomer_scope"] = "all"
    operation = _operation(
        factory=ScopeRecorder,
        bindings=NativeBindings(context, translation, data, tmp_path / "work"),
        sampling=sampling,
        options=options,
        runtime_config_factory=lambda request: {"step": numeric_step(request.sample)},
    )
    _bootstrap(operation)
    request, _ = _request(translation, evidence, sampling)
    operation.invoke(operation.prepare(request))

    assert seen == [("A",) if scope == "mutable" else ("A", "B")]
    assert operation.options["protomer_scope"] == scope
    if scope == "all":
        assert set(seen[0]) <= set(protomers)


def test_persisted_inputs_rebuild_exact_scores_without_extra_native_factory_call(
    tmp_path: Path,
):
    data = DataStore(tmp_path / "data")
    _, context, translation = _state(data)
    evidence = _evidence(data, translation)
    sampling = _sampling(data)
    loads = []
    operation = _operation(
        factory=lambda: loads.append("load") or FakeNativeMutator(),
        bindings=NativeBindings(context, translation, data, tmp_path / "work"),
        sampling=sampling,
        runtime_config_factory=lambda request: {"step": numeric_step(request.sample)},
    )
    request, _ = _request(translation, evidence, sampling)
    _bootstrap(operation)
    assert loads == ["load"]
    prepared = operation.prepare(request)
    data.persist(prepared.references)

    reopened = DataStore(tmp_path / "data")
    reopened_bindings = NativeBindings(
        context,
        translation,
        reopened,
        tmp_path / "work",
    )
    restored = reopened_bindings.scores_for(
        evidence,
        destination=tmp_path / "restored-inputs",
    )

    assert loads == ["load"]
    np.testing.assert_array_equal(
        restored.plddt["A"][0],
        np.asarray([0.2, 0.8], dtype=np.float32),
    )
    assert restored.plddt["A"][0].dtype == np.float32
    np.testing.assert_array_equal(
        restored.per_proto_scores["A"][0],
        np.asarray([0.3, 0.7], dtype=np.float64),
    )


def test_unsupported_request_shapes_fail_before_native_load(tmp_path: Path):
    data = DataStore(tmp_path / "data")
    _, context, translation = _state(data)
    evidence = _evidence(data, translation)
    sampling = _sampling(data)
    loads = []
    operation = _operation(
        factory=lambda: loads.append("load") or FakeNativeMutator(),
        bindings=NativeBindings(context, translation, data, tmp_path / "work"),
        sampling=sampling,
    )
    request, _ = _request(translation, evidence, sampling)
    use = request.evidence[0]
    subject = Subject(
        candidate=translation.candidate.identity,
        state=None,
        assembly="A",
        occurrences=tuple(
            occurrence.identity for occurrence in translation.system.occurrences
        ),
        artifact=None,
        sample=None,
    )
    selection = Selection(
        subject=subject,
        axis_name="residue",
        index_map=data.put({"entries": []}, JSON),
        mask=data.put({"mask": []}, JSON),
    )
    packed = PackedStates(
        artifact=evidence.artifacts[0],
        mapping=translation.system.topology,
        ties=translation.system.topology,
        state_weights=(1.0,),
    )
    invalid_requests = (
        replace(request, selection=(selection,)),
        replace(request, packed=packed),
        replace(request, evidence=(use, use)),
        replace(
            request,
            sampling=SamplingPolicy(
                key_for("policy/wrong"),
                sampling.options,
            ),
        ),
    )

    for invalid in invalid_requests:
        with pytest.raises(UnsupportedNativeMutation):
            operation.prepare(invalid)
    assert loads == []


def test_valid_request_requires_initialization_and_does_not_load_in_prepare(
    tmp_path: Path,
):
    data = DataStore(tmp_path / "data")
    _, context, translation = _state(data)
    evidence = _evidence(data, translation)
    sampling = _sampling(data)
    loads = []
    operation = _operation(
        factory=lambda: loads.append("load") or FakeNativeMutator(),
        bindings=NativeBindings(context, translation, data, tmp_path / "work"),
        sampling=sampling,
    )
    request, _ = _request(translation, evidence, sampling)

    with pytest.raises(
        UnsupportedNativeMutation,
        match="initialization must complete",
    ):
        operation.prepare(request)
    assert loads == []


def test_descriptor_hashes_shared_owner_and_real_factory_callables(
    tmp_path: Path,
    monkeypatch,
):
    data = DataStore(tmp_path / "data")
    _, context, translation = _state(data)
    sampling = _sampling(data)
    seen = {}

    def factory():
        return FakeNativeMutator()

    def configure(mutator, runtime_config, **kwargs):
        return None

    def runtime_config_factory(request):
        return {}

    original = native_mutation_module._describe_operation

    def capture(*args, **kwargs):
        seen["implementation"] = kwargs["implementation"]
        return original(*args, **kwargs)

    monkeypatch.setattr(native_mutation_module, "_describe_operation", capture)
    _operation(
        factory=factory,
        bindings=NativeBindings(context, translation, data, tmp_path / "work"),
        sampling=sampling,
        configure=configure,
        configure_identity="configure/test/v1",
        runtime_config_factory=runtime_config_factory,
    )

    assert seen["implementation"] == (
        native_mutation_module.NativeMutationOperation,
        native_mutation_module.LazyNativeOwner,
        factory,
        configure,
        runtime_config_factory,
    )


def test_runtime_cached_initialization_rebinds_ready_receipt_without_reload(
    tmp_path: Path,
):
    root = tmp_path / "runtime"
    data = DataStore(root / "data")
    _, context, translation = _state(data)
    evidence = _evidence(data, translation)
    sampling = _sampling(data)
    loads = []

    def factory():
        loads.append("load")
        return FakeNativeMutator()

    first_bindings = NativeBindings(
        context,
        translation,
        data,
        tmp_path / "work",
    )
    first = _operation(
        factory=factory,
        bindings=first_bindings,
        sampling=sampling,
    )
    requirements = (
        Requirement(first.initialization.descriptor.alias, INVOKE, "required"),
    )
    with Runtime(
        (first.initialization, first),
        root=root,
        data=first_bindings.data,
    ) as runtime:
        runtime.preflight(requirements)
        with runtime.bind("bootstrap", History(key_for(()), ())):
            ready = runtime.call(
                "initialize",
                first.initialization,
                first.initialization_request,
            )
        assert ready.model == first.owner.identity
        assert first.owner.ready
    assert loads == ["load"]
    first.shutdown()

    reopened = DataStore(root / "data")
    second_bindings = NativeBindings(
        context,
        translation,
        reopened,
        tmp_path / "work",
    )
    second = _operation(
        factory=factory,
        bindings=second_bindings,
        sampling=sampling,
    )
    with Runtime(
        (second.initialization, second),
        root=root,
        data=second_bindings.data,
    ) as runtime:
        runtime.preflight(requirements)
        with runtime.bind("bootstrap", History(key_for(()), ())):
            restored_ready = runtime.call(
                "initialize",
                second.initialization,
                second.initialization_request,
            )
        assert restored_ready == ready
        assert second.owner.ready
    assert loads == ["load"]


def test_sampling_and_configure_identity_change_operation_identity(tmp_path: Path):
    data = DataStore(tmp_path / "data")
    _, context, translation = _state(data)
    sampling = _sampling(data)
    other_sampling = SamplingPolicy(
        key_for("policy/native-mutator-other"),
        sampling.options,
    )
    bindings = NativeBindings(context, translation, data, tmp_path / "work")

    def configure_one(mutator, runtime_config, **kwargs):
        runtime_config["variant"] = "one"

    def configure_two(mutator, runtime_config, **kwargs):
        runtime_config["variant"] = "two"

    first = _operation(
        factory=FakeNativeMutator,
        bindings=bindings,
        sampling=sampling,
        configure=configure_one,
        configure_identity="configure/one",
    )
    second = _operation(
        factory=FakeNativeMutator,
        bindings=bindings,
        sampling=sampling,
        configure=configure_two,
        configure_identity="configure/two",
    )
    changed_sampling = _operation(
        factory=FakeNativeMutator,
        bindings=bindings,
        sampling=other_sampling,
        configure=configure_one,
        configure_identity="configure/one",
    )

    assert len(
        {
            first.descriptor.identity,
            second.descriptor.identity,
            changed_sampling.descriptor.identity,
        }
    ) == 3


def test_frozen_runtime_step_overrides_numeric_sample_index_for_native_cadence(
    tmp_path: Path,
):
    data = DataStore(tmp_path / "data")
    _, context, translation = _state(data)
    evidence = _evidence(data, translation)
    sampling = _sampling(data)
    configure_calls = []

    def configure(mutator, runtime_config, *, sample, step, **kwargs):
        configure_calls.append((sample.index, step, dict(runtime_config)))
        runtime_config["job_name"] = sample.index

    operation = _operation(
        factory=FakeNativeMutator,
        bindings=NativeBindings(context, translation, data, tmp_path / "work"),
        sampling=sampling,
        configure=configure,
        configure_identity="runtime-step-regression/v1",
        runtime_config_factory=lambda request: {"step": 7},
    )
    _bootstrap(operation)
    request, _ = _request(translation, evidence, sampling, index="91")
    prepared = operation.prepare(request)

    first = operation.invoke(prepared)
    second = operation.invoke(operation.prepare(request))

    assert first == second
    assert configure_calls == [
        ("91", 7, {"step": 7}),
        ("91", 7, {"step": 7}),
    ]
    native_calls = operation.owner.get().calls
    assert [call[2]["step"] for call in native_calls] == [7, 7]
    assert [call[2]["job_name"] for call in native_calls] == ["91", "91"]
    native_inputs = tmp_path / "work" / "native-inputs" / evidence.identity.digest
    assert native_inputs.is_dir()
    assert first.items[0].identity == second.items[0].identity


def test_negative_frozen_runtime_step_is_rejected_before_native_mutation(
    tmp_path: Path,
):
    data = DataStore(tmp_path / "data")
    _, context, translation = _state(data)
    evidence = _evidence(data, translation)
    sampling = _sampling(data)
    operation = _operation(
        factory=FakeNativeMutator,
        bindings=NativeBindings(context, translation, data, tmp_path / "work"),
        sampling=sampling,
        runtime_config_factory=lambda request: {"step": -1},
    )
    _bootstrap(operation)
    request, _ = _request(translation, evidence, sampling, index="91")

    with pytest.raises(ValueError, match="non-negative integer"):
        operation.prepare(request)

    assert operation.owner.get().calls == []


def test_default_context_policy_keeps_empty_artifacts_and_legacy_proposal_identity(
    tmp_path: Path,
    monkeypatch,
):
    def unexpected_capture(*args, **kwargs):
        raise AssertionError("disabled mutation context must not perform capture")

    monkeypatch.setattr(
        native_mutation_module,
        "capture_native_mutation_contexts",
        unexpected_capture,
    )
    data = DataStore(tmp_path / "data")
    _, context, translation = _state(data)
    evidence = _evidence(data, translation)
    sampling = _sampling(data)
    operation = _operation(
        factory=FakeNativeMutator,
        bindings=NativeBindings(context, translation, data, tmp_path / "work"),
        sampling=sampling,
    )
    _bootstrap(operation)
    request, _ = _request(translation, evidence, sampling, index="3")
    prepared = operation.prepare(request)

    proposal = operation.invoke(prepared).items[0]

    assert proposal.artifacts == ()
    expected_identity = key_for(
        (
            "proposal/native-mutation/v1",
            operation.descriptor.identity,
            prepared.inputs,
            tuple(parent.identity for parent in request.parents),
            request.working.identity,
            proposal.candidate.identity,
            (evidence.identity,),
            request.sampling.identity,
        )
    )
    assert proposal.identity == expected_identity


def test_enabled_context_captures_preapply_pending_state_and_restores_cold(
    tmp_path: Path,
):
    root = tmp_path / "data"
    data = DataStore(root)
    _, context, translation = _state(data)
    evidence = _evidence(data, translation)
    sampling = _sampling(data)
    contribution_value = np.asarray(
        [[0.125, 0.25], [0.375, 0.5]],
        dtype=np.float32,
    )
    selector_value = [
        {
            "selector": "entropy",
            "entries": [
                {"position": 1, "weight": 0.25},
                {"position": 2, "weight": 0.75},
            ],
        }
    ]

    def configure(mutator, runtime_config, **kwargs):
        del mutator, kwargs
        runtime_config["mutator_selector_score_contexts"] = selector_value
        runtime_config["score_contributions"] = (
            ScoreContribution(
                source="native-mutator",
                metric_name="selection_weight",
                value=contribution_value,
                scope="protomer",
                entity="A",
                shape="residue_pair",
                aggregation="weights",
                flatten_as="mutation_weight",
                provenance="test/runtime",
            ),
        )

    class OrderedContextMutator(FakeNativeMutator):
        def mutate(self, protomers, scores, runtime_config):
            self.calls.append((tuple(protomers), scores, dict(runtime_config)))
            segment = global_config.segments["A_1"]
            segment.mutations = [
                (0, "A", "C", 0.1, 0.2),
                (1, "G", "T", 0.3, 0.4),
            ]

    policy = MutationContextPolicy(
        "test/native-mutation-context/v1",
        NATIVE_MUTATION_CONTEXT_SCHEMA,
    )
    operation = _operation(
        factory=OrderedContextMutator,
        bindings=NativeBindings(context, translation, data, tmp_path / "work"),
        sampling=sampling,
        configure=configure,
        configure_identity="test/native-mutation-context-configure/v1",
        mutation_context=policy,
    )
    _bootstrap(operation)
    request, _ = _request(translation, evidence, sampling, index="3")

    prepared = operation.prepare(request)
    proposal = operation.invoke(prepared).items[0]

    assert len(proposal.artifacts) == 1
    artifact = proposal.artifacts[0]
    assert artifact.candidate == proposal.candidate.identity
    assert artifact.sample == request.sample
    assert set(artifact.derived_from) == {
        request.working.identity,
        proposal.candidate.identity,
    }
    state = restore_native_mutation_context(artifact, data)
    assert state.schema == policy.identity
    assert state.base_candidate == request.working.identity
    assert state.proposed_candidate == proposal.candidate.identity
    assert state.sequence_before_pending == "AG"
    assert state.working_sequence == "CT"
    assert [
        (
            item.position,
            item.from_token,
            item.to_token,
            item.entropy,
            item.score,
        )
        for item in state.pending_mutations
    ] == [
        (0, "A", "C", 0.1, 0.2),
        (1, "G", "T", 0.3, 0.4),
    ]
    assert state.pre_update_snapshot is None
    assert data.get(
        state.selector_contexts,
        JSONCodec("astrolabe.design.native-mutation.selector-contexts/v1"),
    ) == selector_value
    assert len(state.score_contributions) == 1
    contribution = state.score_contributions[0]
    assert (
        contribution.source,
        contribution.metric_name,
        contribution.scope,
        contribution.entity,
        contribution.shape,
        contribution.aggregation,
        contribution.flatten_as,
        contribution.provenance,
        contribution.enabled,
    ) == (
        "native-mutator",
        "selection_weight",
        "protomer",
        "A",
        "residue_pair",
        "weights",
        "mutation_weight",
        "test/runtime",
        True,
    )
    np.testing.assert_array_equal(
        data.get(contribution.value, ArrayCodec()),
        contribution_value,
    )

    data.persist((artifact.content,))
    reopened = DataStore(root)
    restored = restore_native_mutation_context(artifact, reopened)
    assert restored == state
    np.testing.assert_array_equal(
        reopened.get(restored.score_contributions[0].value, ArrayCodec()),
        contribution_value,
    )


def test_plural_context_segments_are_stable_order_and_deduplicate_aliases(
    tmp_path: Path,
):
    data = DataStore(tmp_path / "data")
    _multi_state(data, fixed_second=True)
    segments = native_mutation_segments()
    assert tuple(name for name, _segment in segments) == ("A_1", "A_2")
    assert segments[0][1] is global_config.segments["shared"]
    assert segments[0][1] is global_config.segments["B_1"]
    assert segments[1][1] is global_config.segments["second"]
    assert segments[1][1] is global_config.segments["B_2"]
    assert segments[1][1].is_fixed is True

    policy = MutationContextPolicy(
        "test/native-mutation-context/v1",
        NATIVE_MUTATION_CONTEXT_SCHEMA,
    )
    with pytest.raises(
        UnsupportedMutationContext,
        match="exactly one unique polymer segment",
    ):
        capture_native_mutation_context(
            data,
            policy=policy,
            runtime_config={},
        )


def test_native_mutation_segments_rejects_stale_or_unknown_identity():
    manager = SegmentManager()
    first = Segment("AC", type=ProtomerTypes.PROTEIN)
    second = Segment("GT", type=ProtomerTypes.RNA)
    manager.add_segment("A_1", first)
    manager.add_segment("A_2", second)
    manager.primary_names[id(second)] = "A_1"
    with pytest.raises(UnsupportedMutationContext, match="duplicate primary"):
        native_mutation_segments(manager)

    manager.primary_names[id(second)] = "missing"
    with pytest.raises(
        UnsupportedMutationContext,
        match="does not identify its segment",
    ):
        native_mutation_segments(manager)

    unknown = SimpleNamespace(
        sequence="AC",
        mutations=None,
        is_fixed=True,
        type=None,
    )
    manager.add_segment("unknown", unknown)
    manager.primary_names[id(second)] = "A_2"
    with pytest.raises(
        UnsupportedMutationContext,
        match="known segment type",
    ):
        native_mutation_segments(manager)


def test_native_mutation_emits_one_cold_context_artifact_per_polymer(
    tmp_path: Path,
):
    root = tmp_path / "data"
    data = DataStore(root)
    _, context, translation = _multi_state(data)
    evidence = _evidence(data, translation)
    sampling = _sampling(data)
    working = _multi_working(translation, data)

    class TwoSegmentMutator(FakeNativeMutator):
        def mutate(self, protomers, scores, runtime_config):
            self.calls.append((tuple(protomers), scores, dict(runtime_config)))
            global_config.segments["A_1"].mutations = [
                (0, "A", "C", 0.1, 0.2),
            ]
            global_config.segments["A_2"].mutations = [
                (1, "T", "G", 0.3, 0.4),
            ]

    policy = MutationContextPolicy(
        "test/native-mutation-context/v1",
        NATIVE_MUTATION_CONTEXT_SCHEMA,
    )
    operation = _operation(
        factory=TwoSegmentMutator,
        bindings=NativeBindings(context, translation, data, tmp_path / "work"),
        sampling=sampling,
        mutation_context=policy,
    )
    _bootstrap(operation)
    request, _ = _request(
        translation,
        evidence,
        sampling,
        index="3",
        working=working,
    )

    proposal = operation.invoke(operation.prepare(request)).items[0]

    assert len(proposal.artifacts) == 2
    assert len({artifact.identity for artifact in proposal.artifacts}) == 2
    states = {
        state.variable: state
        for state in (
            restore_native_mutation_context(artifact, data)
            for artifact in proposal.artifacts
        )
    }
    assert tuple(states) == ("A_1", "A_2")
    assert states["A_1"].sequence_before_pending == "AG"
    assert states["A_1"].working_sequence == "CG"
    assert states["A_2"].sequence_before_pending == "TT"
    assert states["A_2"].working_sequence == "TG"
    assert {
        state.variable: state.pending_mutations
        for state in states.values()
    } == {
        "A_1": (
            NativePendingMutation(0, "A", "C", 0.1, 0.2),
        ),
        "A_2": (
            NativePendingMutation(1, "T", "G", 0.3, 0.4),
        ),
    }
    assert {
        assignment.variable: assignment.value.tokens
        for assignment in proposal.candidate.assignments
    } == {
        "A_1": ("C", "G"),
        "A_2": ("T", "G"),
    }
    for artifact in proposal.artifacts:
        assert artifact.candidate == proposal.candidate.identity
        assert set(artifact.derived_from) == {
            request.working.identity,
            proposal.candidate.identity,
        }

    data.persist(tuple(artifact.content for artifact in proposal.artifacts))
    reopened = DataStore(root)
    assert {
        restore_native_mutation_context(artifact, reopened)
        for artifact in proposal.artifacts
    } == set(states.values())


def test_all_fixed_polymer_segments_emit_noop_contexts(tmp_path: Path):
    data = DataStore(tmp_path / "data")
    _, context, translation = _multi_state(
        data,
        fixed_first=True,
        fixed_second=True,
    )
    evidence = _evidence(data, translation)
    sampling = _sampling(data)

    class NoopMutator(FakeNativeMutator):
        def mutate(self, protomers, scores, runtime_config):
            self.calls.append((tuple(protomers), scores, dict(runtime_config)))

    operation = _operation(
        factory=NoopMutator,
        bindings=NativeBindings(context, translation, data, tmp_path / "work"),
        sampling=sampling,
        mutation_context=MutationContextPolicy(
            "test/native-mutation-context/v1",
            NATIVE_MUTATION_CONTEXT_SCHEMA,
        ),
    )
    _bootstrap(operation)
    request, _ = _request(
        translation,
        evidence,
        sampling,
        parents=(translation.candidate,),
        working=translation.candidate,
    )

    proposal = operation.invoke(operation.prepare(request)).items[0]

    assert len(proposal.artifacts) == 2
    states = [
        restore_native_mutation_context(artifact, data)
        for artifact in proposal.artifacts
    ]
    assert [state.variable for state in states] == ["A_1", "A_2"]
    assert all(state.pending_mutations == () for state in states)
    assert all(
        state.sequence_before_pending == state.working_sequence
        for state in states
    )
    assert proposal.candidate == translation.candidate


def test_invalid_second_segment_capture_does_not_finalize(
    tmp_path: Path,
    monkeypatch,
):
    data = DataStore(tmp_path / "data")
    _, context, translation = _multi_state(data)
    evidence = _evidence(data, translation)
    sampling = _sampling(data)
    working = _multi_working(translation, data)
    finalized = []

    class InvalidSecondMutator(FakeNativeMutator):
        def mutate(self, protomers, scores, runtime_config):
            self.calls.append((tuple(protomers), scores, dict(runtime_config)))
            global_config.segments["A_1"].mutations = [
                (0, "A", "C", 0.1, 0.2),
            ]
            global_config.segments["A_2"].mutations = [
                (0.5, "T", "G", 0.3, 0.4),
            ]

    monkeypatch.setattr(
        native_mutation_module,
        "finalize_pending_mutations",
        lambda: finalized.append(True),
    )
    operation = _operation(
        factory=InvalidSecondMutator,
        bindings=NativeBindings(context, translation, data, tmp_path / "work"),
        sampling=sampling,
        mutation_context=MutationContextPolicy(
            "test/native-mutation-context/v1",
            NATIVE_MUTATION_CONTEXT_SCHEMA,
        ),
    )
    _bootstrap(operation)
    request, _ = _request(
        translation,
        evidence,
        sampling,
        working=working,
    )

    with pytest.raises(
        UnsupportedNativeMutation,
        match="context capture failed",
    ):
        operation.invoke(operation.prepare(request))

    assert finalized == []
    assert global_config.segments["A_1"].sequence == "AC"
    assert global_config.segments["A_2"].sequence == "GT"
    assert global_config.segments["A_1"].mutations in (None, [])
    assert global_config.segments["A_2"].mutations in (None, [])


def test_immutable_chemical_is_explicitly_skipped_without_pending(
    tmp_path: Path,
):
    _state(DataStore(tmp_path / "data"))
    chemical = Segment(
        "CCO",
        is_fixed=True,
        type=ProtomerTypes.SMILES,
    )
    global_config.segments["ligand"] = chemical
    global_config.segment_manager.add_segment("ligand", chemical)

    assert native_mutation_segments() == (
        ("A_1", global_config.segments["A_1"]),
    )


@pytest.mark.parametrize(
    ("is_fixed", "pending"),
    [
        (False, None),
        (True, [(0, "C", "N", 0.0, 0.0)]),
    ],
)
def test_chemical_guard_rejects_before_native_invocation(
    tmp_path: Path,
    is_fixed: bool,
    pending,
):
    data = DataStore(tmp_path / "data")
    _, context, translation = _state(data)
    evidence = _evidence(data, translation)
    sampling = _sampling(data)
    chemical = Segment(
        "CCO",
        is_fixed=is_fixed,
        type=ProtomerTypes.SMILES,
        mutations=None if pending is None else list(pending),
    )
    global_config.segments["ligand"] = chemical
    global_config.segment_manager.add_segment("ligand", chemical)
    factory_calls = []
    operation = _operation(
        factory=lambda: factory_calls.append("load") or FakeNativeMutator(),
        bindings=NativeBindings(context, translation, data, tmp_path / "work"),
        sampling=sampling,
        mutation_context=MutationContextPolicy(
            "test/native-mutation-context/v1",
            NATIVE_MUTATION_CONTEXT_SCHEMA,
        ),
    )
    request, _ = _request(translation, evidence, sampling)

    with pytest.raises(
        UnsupportedNativeMutation,
        match="chemical segment",
    ):
        operation.prepare(request)

    assert factory_calls == []
    assert chemical.sequence == "CCO"
    assert chemical.mutations == (
        None if pending is None else pending
    )


def test_direct_fixed_chemical_edit_fails_and_rolls_back(
    tmp_path: Path,
):
    data = DataStore(tmp_path / "data")
    polymer = Protomer("A", "AC", "p")
    chemical = Protomer("L", "CCO", "s")
    global_config.protomers.update({"A": polymer, "L": chemical})
    global_config.mutable_protomers.update({"A": polymer})
    context = LegacyContext()
    translation = context.capture_candidate(
        assemblies=("A", "L"),
        data=data,
    )
    sample = _sample("3", seed=7)
    scores = StandardScores(
        names=["A", "L"],
        plddt={
            "A": [np.asarray([0.2, 0.8], dtype=np.float32)],
            "L": [np.asarray([0.6], dtype=np.float32)],
        },
        complex_paths={},
        per_proto_scores={
            "A": [np.asarray([0.3, 0.7], dtype=np.float64)],
        },
    )
    content = snapshot_scores(scores, data)
    inputs = manifest(
        (
            Dependency("system", translation.system.identity),
            Dependency("candidate", translation.candidate.identity),
            Dependency("sample", key_for(sample)),
        )
    )
    score_artifact = Artifact(
        identity=key_for(
            (
                "artifact/native-scores/test",
                translation.candidate.identity,
                translation.system.occurrences,
                sample,
                content,
                inputs,
            )
        ),
        candidate=translation.candidate.identity,
        occurrences=translation.system.occurrences,
        sample=sample,
        content=content,
        derived_from=(),
        inputs=inputs,
    )
    evidence = Evidence(
        identity=key_for(
            ("evidence/native-scores/test", translation.candidate.identity),
        ),
        evaluated_candidate=translation.candidate.identity,
        artifacts=(score_artifact,),
        measurements=(),
        inputs=inputs,
    )
    sampling = _sampling(data)

    class ChemicalEditMutator(FakeNativeMutator):
        def mutate(self, protomers, scores, runtime_config):
            self.calls.append(
                (tuple(protomers), scores, dict(runtime_config))
            )
            global_config.segments["L_1"].sequence = "CCN"

    mutator = ChemicalEditMutator()
    operation = _operation(
        factory=lambda: mutator,
        bindings=NativeBindings(context, translation, data, tmp_path / "work"),
        sampling=sampling,
        mutation_context=MutationContextPolicy(
            "test/native-mutation-context/v1",
            NATIVE_MUTATION_CONTEXT_SCHEMA,
        ),
    )
    _bootstrap(operation)
    request, _ = _request(
        translation,
        evidence,
        sampling,
        index=sample.index,
        seed=sample.seed,
    )

    with pytest.raises(
        UnsupportedNativeMutation,
        match="non-polymer assignments",
    ):
        operation.invoke(operation.prepare(request))

    assert len(mutator.calls) == 1
    assert global_config.segments["L_1"].sequence == "CCO"
    assert global_config.segments["L_1"].mutations in (None, [])


def test_context_descriptor_closure_includes_plural_implementation(
    tmp_path: Path,
    monkeypatch,
):
    data = DataStore(tmp_path / "data")
    _, context, translation = _multi_state(data)
    sampling = _sampling(data)
    seen = {}

    original = native_mutation_module._describe_operation

    def capture(*args, **kwargs):
        seen["implementation"] = kwargs["implementation"]
        return original(*args, **kwargs)

    monkeypatch.setattr(native_mutation_module, "_describe_operation", capture)
    _operation(
        factory=FakeNativeMutator,
        bindings=NativeBindings(context, translation, data, tmp_path / "work"),
        sampling=sampling,
        mutation_context=MutationContextPolicy(
            "test/native-mutation-context/v1",
            NATIVE_MUTATION_CONTEXT_SCHEMA,
        ),
    )

    assert capture_native_mutation_contexts in seen["implementation"]
    assert native_mutation_segments in seen["implementation"]


def test_real_mcmc_numpy_pending_fields_survive_durable_context_roundtrip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "data"
    data = DataStore(root)
    _, context, translation = _state(data)
    global_config.init_const("exclude_AA", [])
    global_config.set_variable("ignored_complexes", [])
    global_config.set_variable("negative_design_complexes", [])
    evidence = _evidence(data, translation)
    sampling = _sampling(data)
    observed_pending = []
    observed_scores = []
    original_capture = native_mutation_module.capture_native_mutation_contexts

    def record_pending(data, *, policy, runtime_config):
        manager = global_config.segment_manager
        assert len(manager.unique_segments) == 1
        segment = next(iter(manager.unique_segments.values()))
        observed_pending.extend(tuple(segment.mutations or ()))
        return original_capture(
            data,
            policy=policy,
            runtime_config=runtime_config,
        )

    monkeypatch.setattr(
        native_mutation_module,
        "capture_native_mutation_contexts",
        record_pending,
    )

    class RecordingMCMC(RandomSubstitutionMutator):
        def mutate(self, protomers, scores, runtime_config):
            observed_scores.append(scores)
            return super().mutate(protomers, scores, runtime_config)

    def configure(mutator, runtime_config, **kwargs):
        del runtime_config, kwargs
        global_config.set_variable("mutation_num", 1)
        mutator.mutation_weight = 1.0

    policy = MutationContextPolicy(
        "test/native-mutation-context/real-mcmc/v1",
        NATIVE_MUTATION_CONTEXT_SCHEMA,
    )
    operation = _operation(
        factory=lambda: RecordingMCMC(
            mutation_method="frequency_adjusted",
            select_positions="plddt",
            avoid_current_residue=True,
        ),
        bindings=NativeBindings(context, translation, data, tmp_path / "work"),
        sampling=sampling,
        configure=configure,
        configure_identity="test/real-mcmc-configure/v1",
        mutation_context=policy,
    )
    _bootstrap(operation)
    request, _ = _request(translation, evidence, sampling, index="3")
    np.random.seed(19)

    proposal = operation.invoke(operation.prepare(request)).items[0]

    assert len(observed_scores) == 1
    assert type(observed_scores[0]) is StandardScores
    assert observed_scores[0].plddt["A"][0].dtype == np.float32
    np.testing.assert_array_equal(
        observed_scores[0].plddt["A"][0],
        np.asarray([0.2, 0.8], dtype=np.float32),
    )
    assert len(observed_pending) == 1
    raw = observed_pending[0]
    assert type(raw[0]) is np.int64
    assert type(raw[1]) is str
    assert type(raw[2]) is np.str_
    assert type(raw[3]) is np.float64
    assert type(raw[4]) is np.float64

    artifact = proposal.artifacts[0]
    state = restore_native_mutation_context(artifact, data)
    assert len(state.pending_mutations) == 1
    pending = state.pending_mutations[0]
    assert type(pending.position) is int
    assert pending.position == int(raw[0])
    assert pending.from_token == raw[1]
    assert pending.to_token == raw[2]
    assert pending.entropy == float(raw[3])
    assert pending.score == float(raw[4])
    expected_working = list(state.sequence_before_pending)
    expected_working[pending.position] = pending.to_token
    assert state.working_sequence == "".join(expected_working)

    data.persist((artifact.content,))
    reopened = DataStore(root)
    restored = restore_native_mutation_context(artifact, reopened)
    assert restored == state
    assert type(restored.pending_mutations[0].position) is int
    assert type(restored.pending_mutations[0].entropy) is float
    assert type(restored.pending_mutations[0].score) is float
    assert (
        restored.pending_mutations[0].position,
        restored.pending_mutations[0].from_token,
        restored.pending_mutations[0].to_token,
        restored.pending_mutations[0].entropy,
        restored.pending_mutations[0].score,
    ) == (
        int(raw[0]),
        raw[1],
        str(raw[2]),
        float(raw[3]),
        float(raw[4]),
    )


@pytest.mark.parametrize(
    "invalid_position",
    [True, np.bool_(True), 1.0, np.float64(1.0)],
)
def test_pending_position_rejects_bool_and_float_indices(
    tmp_path: Path,
    invalid_position,
):
    data = DataStore(tmp_path / "data")
    _state(data)
    manager = global_config.segment_manager
    segment = next(iter(manager.unique_segments.values()))
    segment.mutations = [
        (invalid_position, "A", "C", np.float64(0.25), np.float64(0.75))
    ]
    policy = MutationContextPolicy(
        "test/native-mutation-context/invalid-position/v1",
        NATIVE_MUTATION_CONTEXT_SCHEMA,
    )

    with pytest.raises(
        UnsupportedMutationContext,
        match="pending mutation position is invalid",
    ):
        capture_native_mutation_context(
            data,
            policy=policy,
            runtime_config={},
        )


def test_context_policy_and_schema_change_descriptor_and_reference_closure(
    tmp_path: Path,
):
    data = DataStore(tmp_path / "data")
    _, context, translation = _state(data)
    sampling = _sampling(data)
    bindings = NativeBindings(context, translation, data, tmp_path / "work")
    base_schema = dict(NATIVE_MUTATION_CONTEXT_SCHEMA)
    changed_schema = dict(base_schema)
    changed_schema["title"] = "changed scientific context schema"

    disabled = _operation(
        factory=FakeNativeMutator,
        bindings=bindings,
        sampling=sampling,
    )
    first = _operation(
        factory=FakeNativeMutator,
        bindings=bindings,
        sampling=sampling,
        mutation_context=MutationContextPolicy(
            "test/native-mutation-context/v1",
            base_schema,
        ),
    )
    changed_identity = _operation(
        factory=FakeNativeMutator,
        bindings=bindings,
        sampling=sampling,
        mutation_context=MutationContextPolicy(
            "test/native-mutation-context/v2",
            base_schema,
        ),
    )
    changed_schema_operation = _operation(
        factory=FakeNativeMutator,
        bindings=bindings,
        sampling=sampling,
        mutation_context=MutationContextPolicy(
            "test/native-mutation-context/v1",
            changed_schema,
        ),
    )

    with pytest.raises(TypeError):
        first.mutation_context.schema["title"] = "post-construction mutation"
    with pytest.raises(TypeError):
        first.mutation_context.schema["properties"]["pending_mutations"][
            "type"
        ] = "null"
    assert len(
        {
            first.descriptor.identity,
            changed_identity.descriptor.identity,
            changed_schema_operation.descriptor.identity,
        }
    ) == 3
    assert disabled.descriptor.identity != first.descriptor.identity
    assert NATIVE_MUTATION_CONTEXT_CODEC.identity not in {
        codec.identity for codec in disabled.reference_codecs
    }
    assert NATIVE_MUTATION_CONTEXT_CODEC.identity in {
        codec.identity for codec in first.reference_codecs
    }


def test_context_rejects_preupdate_stack_and_mismatched_candidate_binding(
    tmp_path: Path,
):
    data = DataStore(tmp_path / "data")
    _, _, translation = _state(data)
    policy = MutationContextPolicy(
        "test/native-mutation-context/v1",
        NATIVE_MUTATION_CONTEXT_SCHEMA,
    )
    manager = global_config.segment_manager
    segment = manager.unique_segments[id(manager.segments["A_1"])]
    segment.mutations = [(1, "C", "T", 0.5, 0.75)]

    manager._pre_update_snapshot = {}
    with pytest.raises(UnsupportedMutationContext, match="pre-update stack"):
        capture_native_mutation_context(
            data,
            policy=policy,
            runtime_config={},
        )
    manager._pre_update_snapshot = None

    draft = capture_native_mutation_context(
        data,
        policy=policy,
        runtime_config={},
    )
    proposed = translation.system.edit(
        translation.candidate,
        (Assignment("A_1", Polymer("protein", ("A", "T"), ())),),
        data=data,
    )
    with pytest.raises(
        UnsupportedMutationContext,
        match="captured pre-pending sequence",
    ):
        from astrolabe.design.native_mutation_state import (
            bind_native_mutation_context,
        )

        bind_native_mutation_context(
            draft,
            data=data,
            system=translation.system,
            base=proposed,
            proposed=proposed,
            sample=_sample("3"),
        )


def test_explicit_mutation_invoker_runs_once_before_context_capture(
    tmp_path: Path,
):
    data = DataStore(tmp_path / "data")
    _, context, translation = _state(data)
    evidence = _evidence(data, translation)
    sampling = _sampling(data)
    events = []

    class NeverMutate(FakeNativeMutator):
        def mutate(self, protomers, scores, runtime_config):
            raise AssertionError("explicit invoker must replace mutate()")

    def invoke(mutator, protomers, scores, runtime_config):
        events.append("invoke")
        assert isinstance(mutator, NeverMutate)
        assert tuple(protomers) == ("A",)
        assert isinstance(scores, StandardScores)
        assert type(runtime_config["step"]) is int
        global_config.segments["A_1"].mutations = [
            (1, "G", "T", 0.4, 0.8),
        ]

    policy = MutationContextPolicy(
        "test/native-mutation-context/v1",
        NATIVE_MUTATION_CONTEXT_SCHEMA,
    )
    operation = _operation(
        factory=NeverMutate,
        bindings=NativeBindings(context, translation, data, tmp_path / "work"),
        sampling=sampling,
        mutation_context=policy,
        invoke=invoke,
        invoke_identity="test/mutation-invoker/v1",
    )
    _bootstrap(operation)
    request, _ = _request(translation, evidence, sampling, index="4")

    prepared = operation.prepare(request)
    proposal = operation.invoke(prepared).items[0]

    assert events == ["invoke"]
    assert len(proposal.artifacts) == 1
    state = restore_native_mutation_context(proposal.artifacts[0], data)
    assert state.pending_mutations == (
        NativePendingMutation(
            position=1,
            from_token="G",
            to_token="T",
            entropy=0.4,
            score=0.8,
        ),
    )
    assert "invoke" in {
        dependency.role for dependency in prepared.inputs.dependencies
    }


def test_default_mutation_path_still_calls_mutate_once(tmp_path: Path):
    data = DataStore(tmp_path / "data")
    _, context, translation = _state(data)
    evidence = _evidence(data, translation)
    sampling = _sampling(data)
    evaluator = FakeNativeMutator()

    operation = _operation(
        factory=lambda: evaluator,
        bindings=NativeBindings(context, translation, data, tmp_path / "work"),
        sampling=sampling,
    )
    assert operation.native_invoke is None
    _bootstrap(operation)
    request, _ = _request(translation, evidence, sampling, index="0")

    operation.invoke(operation.prepare(request))

    assert len(evaluator.calls) == 1


def test_mutation_invoker_identity_and_source_change_descriptor(tmp_path: Path):
    data = DataStore(tmp_path / "data")
    _, context, translation = _state(data)
    sampling = _sampling(data)
    bindings = NativeBindings(context, translation, data, tmp_path / "work")

    def invoke_a(mutator, protomers, scores, runtime_config):
        return mutator.mutate(protomers, scores, runtime_config)

    def invoke_b(mutator, protomers, scores, runtime_config):
        value = mutator.mutate(protomers, scores, runtime_config)
        return value

    first = _operation(
        factory=FakeNativeMutator,
        bindings=bindings,
        sampling=sampling,
        invoke=invoke_a,
        invoke_identity="test/mutation-invoker/v1",
    )
    changed_source = _operation(
        factory=FakeNativeMutator,
        bindings=bindings,
        sampling=sampling,
        invoke=invoke_b,
        invoke_identity="test/mutation-invoker/v1",
    )
    changed_identity = _operation(
        factory=FakeNativeMutator,
        bindings=bindings,
        sampling=sampling,
        invoke=invoke_a,
        invoke_identity="test/mutation-invoker/v2",
    )

    assert len(
        {
            first.descriptor.identity,
            changed_source.descriptor.identity,
            changed_identity.descriptor.identity,
        }
    ) == 3
    assert len(
        {
            first.owner.identity,
            changed_source.owner.identity,
            changed_identity.owner.identity,
        }
    ) == 3
