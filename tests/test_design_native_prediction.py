from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest

from astrolabe.core import global_config
from astrolabe.core.cyclization import CyclicRelativeOffset, CyclicRelativePatch
from astrolabe.core.folding_input import (
    Input,
    Ligand,
    ProteinChain,
    Template,
)
from astrolabe.core.protomer_config_parser import Protomer, SegmentManager
from astrolabe.design.legacy import LegacyContext
from astrolabe.design.native_prediction import (
    FOLDING_INPUT_CODEC,
    UnsupportedNativePrediction,
    make_evaluator_operation,
)
from astrolabe.design.native_scores import NATIVE_SCORES_ENVELOPE_CODEC
from astrolabe.design.native_support import NativeBindings, numeric_step
from astrolabe.design.operations import INVOKE
from astrolabe.design.runtime import ContractConflict, Runtime
from astrolabe.design.store import DataStore, JSONCodec, design_codec
from astrolabe.design.types import (
    Artifact,
    Blob,
    Conditioning,
    Dependency,
    History,
    PredictRequest,
    Requirement,
    Sample,
    key_for,
    manifest,
)
from astrolabe.evaluators.utils import StandardScores


@pytest.fixture(autouse=True)
def clean_legacy_globals(monkeypatch):
    global_config.reset()
    global_config.init_const("segments", {})
    global_config.init_const("protomers", {})
    global_config.init_const("mutable_protomers", {})
    global_config.init_const("named_files", {})
    global_config.init_const("segment_manager", SegmentManager())
    global_config.init_const("covalent_bonds", None)
    global_config.init_const("cyclization", None)
    global_config.init_const("user_ccd", None)
    if importlib.util.find_spec("jsonschema") is None:
        module = ModuleType("jsonschema")

        def validate(instance, schema):
            for name in schema.get("required", ()):
                if name not in instance:
                    raise ValueError(f"missing required option {name}")
            if not schema.get("additionalProperties", True):
                extra = set(instance) - set(schema.get("properties", {}))
                if extra:
                    raise ValueError(f"unknown options {sorted(extra)}")
            for name, value in instance.items():
                expected = schema.get("properties", {}).get(name, {}).get("type")
                if expected == "integer" and type(value) is not int:
                    raise TypeError(f"{name} must be integer")

        module.validate = validate
        monkeypatch.setitem(sys.modules, "jsonschema", module)
    yield
    global_config.reset()


def _sample(index: str) -> Sample:
    return Sample(
        protocol=key_for("prediction-protocol"),
        replicate="replicate",
        seed=41,
        index=index,
    )


def _setup(tmp_path):
    protomer = Protomer("A", "AC", "p")
    global_config.protomers["A"] = protomer
    segment = global_config.segments["A_1"]
    segment.ptm_dict = {1: "PHOS"}
    protomer._homology = {
        "unpairedMsa": ">query\nAC\n",
        "pairedMsa": ">pair\nAC\n",
    }
    protomer._mask_mutable_msa = False
    protomer.add_locked_positions([0])
    segment.mutations = [(0, "A", "G", 0.1, 0.2)]
    segment.probs = [np.asarray([[0.9, 0.1], [0.2, 0.8]])]
    segment.scores = [[0.8, 0.6]]
    data = DataStore(tmp_path / "data")
    context = LegacyContext()
    translation = context.capture_candidate(assemblies=("A",), data=data)
    bindings = NativeBindings(
        context=context,
        translation=translation,
        data=data,
        work_root=tmp_path / "work",
    )
    return protomer, segment, bindings


class FakeEvaluator:
    def __init__(self, output: Path, events: list[str]):
        self.output = output
        self.events = events
        self.evaluate_calls: list[tuple[dict, dict]] = []
        self.shutdown_calls = 0

    def evaluate(self, inputs, runtime_config=None):
        self.events.append("evaluate")
        self.evaluate_calls.append((inputs, dict(runtime_config or {})))
        self.output.write_bytes(b"synthetic selected structure\n")
        return StandardScores(
            names=list(inputs),
            plddt={"A": [np.asarray([0.4, 0.6], dtype=np.float32)]},
            complex_paths={"A": self.output},
            per_proto_scores={
                "A": [np.asarray([0.2, 0.3], dtype=np.float64)]
            },
            pae={"A": [np.asarray([[1.0, 2.0], [3.0, 4.0]])]},
        )

    def shutdown(self):
        self.shutdown_calls += 1


def _operation(bindings, evaluator_factory, events, **kwargs):
    return make_evaluator_operation(
        "fold",
        evaluator_factory,
        bindings,
        version="test/v1",
        model_identity=key_for("test-model"),
        options={"recycles": 1},
        schema={
            "type": "object",
            "properties": {"recycles": {"type": "integer"}},
            "required": ["recycles"],
            "additionalProperties": False,
        },
        **kwargs,
    )


def _bootstrap(operation):
    prepared = operation.initialization.prepare(operation.initialization_request)
    ready = operation.initialization.invoke(prepared)
    operation.initialization.on_result(ready)


def test_prepare_binds_actual_folding_bytes_and_restores_legacy_state(tmp_path):
    protomer, segment, bindings = _setup(tmp_path)
    events = []
    factory_calls = []
    output = tmp_path / "prediction.cif"

    def factory():
        factory_calls.append("factory")
        return FakeEvaluator(output, events)

    configure_calls = []

    def configure(model, request):
        events.append("configure")
        configure_calls.append((model, request))
        return {
            "step": numeric_step(request.samples[0]),
            "structure_path": str(output),
        }

    op = _operation(
        bindings,
        factory,
        events,
        configure=configure,
        configure_identity="configure/v1",
    )
    assert op.initialization is op.owner.initialization
    assert op.initialization_request is op.owner.initialization_request
    request = PredictRequest(
        system=bindings.translation.system,
        candidate=bindings.translation.candidate,
        assemblies=("A",),
        conditioning=(),
        sources=(),
        samples=(_sample("3"),),
    )
    with pytest.raises(
        UnsupportedNativePrediction,
        match="initialization must complete",
    ):
        op.prepare(request)
    assert factory_calls == []
    _bootstrap(op)
    prepared = op.prepare(request)

    assert factory_calls == ["factory"]
    assert len(prepared.references) == 2
    roles = {dependency.role for dependency in prepared.inputs.dependencies}
    assert {"folding-input", "runtime-config", "model"} <= roles
    folding_blob = Blob(
        next(
            dependency.identity
            for dependency in prepared.inputs.dependencies
            if dependency.role == "folding-input"
        ),
        FOLDING_INPUT_CODEC.identity,
    )
    folding = bindings.data.get(folding_blob, FOLDING_INPUT_CODEC)
    chain = folding["A"].chains[0]
    assert chain.sequence == "GC"
    assert tuple(chain.ptms) == (("PHOS", 2),)
    assert chain.unpaired_msa
    assert chain.paired_msa

    # Candidate activation is scoped: pending mutation, scores, fixed state and
    # PTMs remain intact after preparation.
    assert global_config.segments["A_1"].sequence == "AC"
    assert global_config.segments["A_1"].mutations == [
        (0, "A", "G", 0.1, 0.2)
    ]
    assert global_config.segments["A_1"].probs is not None
    assert global_config.segments["A_1"].ptm_dict == {1: "PHOS"}
    assert protomer.locked_positions == frozenset({0})

    evidence = op.invoke(prepared)

    assert factory_calls == ["factory"]
    assert events == ["configure", "evaluate"]
    assert len(configure_calls) == 1
    assert configure_calls[0][1] is request
    assert configure_calls[0][0].evaluate_calls[0][1]["step"] == 3

    snapshots = tuple(
        artifact
        for artifact in evidence.artifacts
        if artifact.content.codec == NATIVE_SCORES_ENVELOPE_CODEC.identity
    )
    assert len(snapshots) == 1
    assert snapshots[0].sample == request.samples[0]
    restored = bindings.scores_for(evidence, tmp_path / "restored-scores")
    np.testing.assert_array_equal(
        restored.plddt["A"][0],
        np.asarray([0.4, 0.6], dtype=np.float32),
    )
    assert restored.complex_paths["A"].read_bytes() == (
        b"synthetic selected structure\n"
    )

    confidence = next(
        measurement
        for measurement in evidence.measurements
        if ".plddt.A.prediction_" in measurement.quantity
    )
    assert confidence.subjects[0].sample == request.samples[0]
    assert confidence.subjects[0].artifact is None
    op.shutdown()
    assert configure_calls[0][0].shutdown_calls == 1


def test_factory_loads_once_across_two_requests_then_shutdown_is_explicit(tmp_path):
    _, _, bindings = _setup(tmp_path)
    events = []
    instances = []

    def factory():
        evaluator = FakeEvaluator(tmp_path / "two.cif", events)
        instances.append(evaluator)
        return evaluator

    def configure(model, request):
        return {
            "step": numeric_step(request.samples[0]),
            "structure_path": str(tmp_path / "two.cif"),
        }

    op = _operation(
        bindings,
        factory,
        events,
        configure=configure,
        configure_identity="configure/v1",
    )
    _bootstrap(op)
    for index in ("0", "1"):
        request = PredictRequest(
            system=bindings.translation.system,
            candidate=bindings.translation.candidate,
            assemblies=("A",),
            conditioning=(),
            sources=(),
            samples=(_sample(index),),
        )
        prepared = op.prepare(request)
        evidence = op.invoke(prepared)
        assert evidence.evaluated_candidate == request.candidate.identity

    assert len(instances) == 1
    assert len(instances[0].evaluate_calls) == 2
    assert [call[1]["step"] for call in instances[0].evaluate_calls] == [0, 1]
    op.shutdown()
    assert instances[0].shutdown_calls == 1


def test_conditioning_capability_is_rejected_before_model_load(tmp_path):
    _, _, bindings = _setup(tmp_path)
    events = []
    factory_calls = []
    op = _operation(
        bindings,
        lambda: factory_calls.append("factory")
        or FakeEvaluator(tmp_path / "rejected.cif", events),
        events,
    )
    conditioning = Conditioning(
        identity=key_for("conditioning"),
        owner=key_for("owner"),
        intent=bindings.data.put({"intent": "test"}, JSONCodec()),
        mechanism="unsupported",
        effective_parameters=bindings.data.put({}, JSONCodec()),
        approximations=(),
        inputs=manifest(()),
    )
    request = PredictRequest(
        system=bindings.translation.system,
        candidate=bindings.translation.candidate,
        assemblies=("A",),
        conditioning=(conditioning,),
        sources=(),
        samples=(_sample("0"),),
    )

    with pytest.raises(UnsupportedNativePrediction, match="no owner handler"):
        op.prepare(request)
    assert factory_calls == []
    op.shutdown()


def test_runtime_replay_skips_native_call_and_inline_msa_change_conflicts(tmp_path):
    protomer, segment, bindings = _setup(tmp_path)
    events = []
    instances = []

    def factory():
        evaluator = FakeEvaluator(tmp_path / "replay.cif", events)
        instances.append(evaluator)
        return evaluator

    def configure(model, request):
        return {
            "step": numeric_step(request.samples[0]),
            "structure_path": str(tmp_path / "replay.cif"),
        }

    op = _operation(
        bindings,
        factory,
        events,
        configure=configure,
        configure_identity="configure/v1",
    )
    request = PredictRequest(
        system=bindings.translation.system,
        candidate=bindings.translation.candidate,
        assemblies=("A",),
        conditioning=(),
        sources=(),
        samples=(_sample("0"),),
    )
    history = History(key_for(()), ())
    with Runtime(
        (op, op.initialization),
        data=bindings.data,
        root=tmp_path / "runtime",
    ) as runtime:
        runtime.preflight(
            (
                Requirement("fold", INVOKE, "required"),
                Requirement("fold/initialize", INVOKE, "required"),
            )
        )
        with runtime.bind("work", history):
            runtime.call(
                "initialize",
                op.initialization,
                op.initialization_request,
            )
            first = runtime.call("predict/0", op, request)
            replayed = runtime.call("predict/0", op, request)
        assert replayed == first
        assert len(instances) == 1
        assert len(instances[0].evaluate_calls) == 1

        protomer._homology["unpairedMsa"] = ">query\nAC\n>new\nAC\n"
        if hasattr(protomer, "_homology_seq"):
            delattr(protomer, "_homology_seq")
        with runtime.bind("work", history):
            runtime.call(
                "initialize",
                op.initialization,
                op.initialization_request,
            )
            with pytest.raises(ContractConflict, match="Inputs or component changed"):
                runtime.call("predict/0", op, request)
        assert len(instances[0].evaluate_calls) == 1

    assert segment.sequence == "AC"
    assert segment.mutations == [(0, "A", "G", 0.1, 0.2)]
    assert segment.ptm_dict == {1: "PHOS"}
    assert segment.probs is not None
    assert protomer.locked_positions == frozenset({0})
    op.shutdown()


def test_cold_runtime_replays_init_and_prediction_without_loading_model(tmp_path):
    _, _, first_bindings = _setup(tmp_path)
    events = []
    instances = []

    def factory():
        evaluator = FakeEvaluator(tmp_path / "cold.cif", events)
        instances.append(evaluator)
        return evaluator

    def configure(model, request):
        return {
            "step": numeric_step(request.samples[0]),
            "structure_path": str(tmp_path / "cold.cif"),
        }

    first_op = _operation(
        first_bindings,
        factory,
        events,
        configure=configure,
        configure_identity="configure/v1",
    )
    request = PredictRequest(
        system=first_bindings.translation.system,
        candidate=first_bindings.translation.candidate,
        assemblies=("A",),
        conditioning=(),
        sources=(),
        samples=(_sample("0"),),
    )
    history = History(key_for(()), ())
    runtime_root = tmp_path / "cold-runtime"
    data_root = first_bindings.data.root
    with Runtime(
        (first_op, first_op.initialization),
        data=first_bindings.data,
        root=runtime_root,
    ) as runtime:
        runtime.preflight(
            (
                Requirement("fold", INVOKE, "required"),
                Requirement("fold/initialize", INVOKE, "required"),
            )
        )
        with runtime.bind("work", history):
            runtime.call(
                "initialize",
                first_op.initialization,
                first_op.initialization_request,
            )
            first = runtime.call("predict/0", first_op, request)
    assert len(instances) == 1
    first_op.shutdown()

    cold_data = DataStore(data_root)
    cold_context = LegacyContext()
    cold_translation = cold_context.capture_candidate(
        assemblies=("A",),
        data=cold_data,
    )
    cold_bindings = NativeBindings(
        context=cold_context,
        translation=cold_translation,
        data=cold_data,
        work_root=tmp_path / "cold-work",
    )
    second_op = _operation(
        cold_bindings,
        factory,
        events,
        configure=configure,
        configure_identity="configure/v1",
    )
    with Runtime(
        (second_op, second_op.initialization),
        data=cold_data,
        root=runtime_root,
    ) as runtime:
        runtime.preflight(
            (
                Requirement("fold", INVOKE, "required"),
                Requirement("fold/initialize", INVOKE, "required"),
            )
        )
        with runtime.bind("work", history):
            runtime.call(
                "initialize",
                second_op.initialization,
                second_op.initialization_request,
            )
            replayed = runtime.call("predict/0", second_op, request)
    assert replayed == first
    assert len(instances) == 1
    assert events == ["evaluate"]
    second_op.shutdown()


def test_recipe_supplemental_artifact_is_identity_bound_and_committed(tmp_path):
    _, _, bindings = _setup(tmp_path)
    events = []

    def factory():
        return FakeEvaluator(tmp_path / "sidecar.cif", events)

    def configure(model, request):
        return {
            "step": numeric_step(request.samples[0]),
            "structure_path": str(tmp_path / "sidecar.cif"),
        }

    def supplemental(scores, request):
        payload = bindings.data.put(
            {
                "schema": "recipe-sidecar/v1",
                "confidence": scores.plddt["A"][0].tolist(),
            },
            JSONCodec(),
        )
        sample = request.samples[0]
        inputs = manifest(
            (
                Dependency("candidate", request.candidate.identity),
                Dependency("sample", key_for(sample)),
                Dependency("content", payload.identity),
            )
        )
        return (
            Artifact(
                identity=key_for(
                    (
                        "artifact/recipe-sidecar/v1",
                        request.candidate.identity,
                        sample,
                        payload,
                    )
                ),
                candidate=request.candidate.identity,
                occurrences=request.system.occurrences,
                sample=sample,
                content=payload,
                derived_from=(),
                inputs=inputs,
            ),
        )

    identity = key_for("recipe-sidecar-owner/v1")
    op = _operation(
        bindings,
        factory,
        events,
        configure=configure,
        configure_identity="configure/v1",
        supplemental_artifacts=supplemental,
        supplemental_identity=identity,
    )
    _bootstrap(op)
    request = PredictRequest(
        system=bindings.translation.system,
        candidate=bindings.translation.candidate,
        assemblies=("A",),
        conditioning=(),
        sources=(),
        samples=(_sample("4"),),
    )
    evidence = op.invoke(op.prepare(request))
    sidecars = tuple(
        artifact
        for artifact in evidence.artifacts
        if artifact.content.codec == JSONCodec().identity
    )
    assert len(sidecars) == 1
    assert sidecars[0].sample == request.samples[0]
    sidecar = bindings.data.get(sidecars[0].content, JSONCodec())
    assert sidecar["schema"] == "recipe-sidecar/v1"
    np.testing.assert_array_equal(
        np.asarray(sidecar["confidence"], dtype=np.float32),
        np.asarray([0.4, 0.6], dtype=np.float32),
    )
    evidence_ref = bindings.data.put(evidence, design_codec)
    bindings.data.persist((evidence_ref,))
    cold = DataStore(bindings.data.root)
    cold_evidence = cold.get(evidence_ref, design_codec)
    cold_sidecar = next(
        artifact
        for artifact in cold_evidence.artifacts
        if artifact.content.codec == JSONCodec().identity
    )
    assert cold.get(cold_sidecar.content, JSONCodec())["schema"] == (
        "recipe-sidecar/v1"
    )

    other = _operation(
        bindings,
        factory,
        events,
        supplemental_artifacts=supplemental,
        supplemental_identity=key_for("different-sidecar-owner/v1"),
    )
    assert other.descriptor.identity != op.descriptor.identity
    other.shutdown()
    op.shutdown()


def test_configure_and_supplemental_callbacks_require_bound_identities(tmp_path):
    _, _, bindings = _setup(tmp_path)

    with pytest.raises(ValueError, match="configure_identity"):
        _operation(bindings, lambda: None, [], configure=lambda model, request: None)
    with pytest.raises(ValueError, match="supplemental_artifacts"):
        _operation(
            bindings,
            lambda: None,
            [],
            supplemental_identity=key_for("sidecar"),
        )


def test_folding_codec_round_trips_integer_template_map_msa_and_ccd(tmp_path):
    _, _, bindings = _setup(tmp_path)
    template = Template(
        mmcif="data_template\nloop_\n#\n",
        query_to_template_map={1: 7, 2: 9},
    )
    protein = ProteinChain(
        id="A",
        sequence="AC",
        ptms=[("PHOS", 2)],
        unpaired_msa=">query\nAC\n>homolog\nAC\n",
        paired_msa=">pair\nAC\n",
        templates=[template],
    )
    ligand = Ligand(id="B", ccd_ids=["ATP", "MG"])
    folding = {
        "AB": Input(
            name="AB",
            chains=[protein, ligand],
            rng_seeds=[41],
            bonded_atom_pairs=(
                (("A", 2, "OG"), ("B", 1, "O1")),
            ),
            bonded_atom_pair_meta=(
                {"order": "single", "target_distance": 1.4},
            ),
            cyclic_chain_ids=["A"],
            cyclic_relative_patches=(
                CyclicRelativePatch(
                    chain_id="A",
                    offsets=(
                        CyclicRelativeOffset(
                            residue_i=1,
                            residue_j=2,
                            offset=1,
                        ),
                    ),
                ),
            ),
            user_ccd="data_user_ccd\n",
        )
    }

    decoded = FOLDING_INPUT_CODEC.decode(
        FOLDING_INPUT_CODEC.encode(folding)
    )
    decoded_protein, decoded_ligand = decoded["AB"].chains
    assert decoded_protein.unpaired_msa == protein.unpaired_msa
    assert decoded_protein.paired_msa == protein.paired_msa
    assert decoded_protein.ptms == [("PHOS", 2)]
    assert decoded_protein.templates[0].mmcif == template.mmcif
    assert decoded_protein.templates[0].query_to_template_map == {1: 7, 2: 9}
    assert all(
        type(key) is int
        for key in decoded_protein.templates[0].query_to_template_map
    )
    assert decoded_ligand.ccd_ids == ["ATP", "MG"]
    assert decoded["AB"].bonded_atom_pairs == folding["AB"].bonded_atom_pairs
    assert decoded["AB"].cyclic_relative_patches[0].offsets[0] == (
        CyclicRelativeOffset(residue_i=1, residue_j=2, offset=1)
    )


def test_descriptor_hashes_factory_and_callback_sources(tmp_path):
    _, _, bindings = _setup(tmp_path)

    def factory_a():
        return FakeEvaluator(tmp_path / "a.cif", [])

    def factory_b():
        value = FakeEvaluator(tmp_path / "b.cif", [])
        return value

    def configure_a(model, request):
        return {"variant": "a"}

    def configure_b(model, request):
        value = {"variant": "b"}
        return value

    def supplemental_a(scores, request):
        return ()

    def supplemental_b(scores, request):
        value = ()
        return value

    identity = key_for("callback-identity")
    baseline = _operation(
        bindings,
        factory_a,
        [],
        configure=configure_a,
        configure_identity="configure/v1",
        supplemental_artifacts=supplemental_a,
        supplemental_identity=identity,
    )
    factory_changed = _operation(
        bindings,
        factory_b,
        [],
        configure=configure_a,
        configure_identity="configure/v1",
        supplemental_artifacts=supplemental_a,
        supplemental_identity=identity,
    )
    configure_changed = _operation(
        bindings,
        factory_a,
        [],
        configure=configure_b,
        configure_identity="configure/v1",
        supplemental_artifacts=supplemental_a,
        supplemental_identity=identity,
    )
    supplemental_changed = _operation(
        bindings,
        factory_a,
        [],
        configure=configure_a,
        configure_identity="configure/v1",
        supplemental_artifacts=supplemental_b,
        supplemental_identity=identity,
    )

    identities = {
        operation.descriptor.identity
        for operation in (
            baseline,
            factory_changed,
            configure_changed,
            supplemental_changed,
        )
    }
    owner_identities = {
        operation.owner.identity
        for operation in (
            baseline,
            factory_changed,
            configure_changed,
            supplemental_changed,
        )
    }
    assert len(identities) == 4
    assert len(owner_identities) == 4
    for operation in (
        baseline,
        factory_changed,
        configure_changed,
        supplemental_changed,
    ):
        operation.shutdown()


def test_explicit_prediction_invoker_replaces_evaluate_once(tmp_path):
    _, _, bindings = _setup(tmp_path)
    events = []
    output = tmp_path / "invoked.cif"

    class NeverEvaluate(FakeEvaluator):
        def evaluate(self, inputs, runtime_config=None):
            raise AssertionError("explicit invoker must replace evaluate()")

    def invoke(model, inputs, runtime_config):
        events.append("invoke")
        assert isinstance(model, NeverEvaluate)
        assert tuple(inputs) == ("A",)
        assert type(runtime_config["step"]) is int
        output.write_bytes(b"invoker selected structure\n")
        return StandardScores(
            names=["A"],
            plddt={
                "A": [np.asarray([0.5, 0.7], dtype=np.float32)]
            },
            complex_paths={"A": output},
        )

    operation = _operation(
        bindings,
        lambda: NeverEvaluate(output, events),
        events,
        invoke=invoke,
        invoke_identity="test/prediction-invoker/v1",
    )
    _bootstrap(operation)
    request = PredictRequest(
        system=bindings.translation.system,
        candidate=bindings.translation.candidate,
        assemblies=("A",),
        conditioning=(),
        sources=(),
        samples=(_sample("5"),),
    )

    evidence = operation.invoke(operation.prepare(request))

    assert events == ["invoke"]
    assert evidence.evaluated_candidate == request.candidate.identity
    assert "invoke" in {
        dependency.role for dependency in evidence.inputs.dependencies
    }


def test_default_prediction_path_still_calls_evaluate_once(tmp_path):
    _, _, bindings = _setup(tmp_path)
    events = []
    output = tmp_path / "default.cif"
    evaluator = FakeEvaluator(output, events)

    operation = _operation(
        bindings,
        lambda: evaluator,
        events,
    )
    assert operation.native_invoke is None
    _bootstrap(operation)
    request = PredictRequest(
        system=bindings.translation.system,
        candidate=bindings.translation.candidate,
        assemblies=("A",),
        conditioning=(),
        sources=(),
        samples=(_sample("0"),),
    )

    operation.invoke(operation.prepare(request))

    assert events == ["evaluate"]
    assert len(evaluator.evaluate_calls) == 1


def test_prediction_invoker_identity_and_source_change_descriptor(tmp_path):
    _, _, bindings = _setup(tmp_path)

    def invoke_a(model, inputs, runtime_config):
        return model.evaluate(inputs, runtime_config)

    def invoke_b(model, inputs, runtime_config):
        value = model.evaluate(inputs, runtime_config)
        return value

    first = _operation(
        bindings,
        lambda: FakeEvaluator(tmp_path / "a.cif", []),
        [],
        invoke=invoke_a,
        invoke_identity="test/prediction-invoker/v1",
    )
    changed_source = _operation(
        bindings,
        lambda: FakeEvaluator(tmp_path / "b.cif", []),
        [],
        invoke=invoke_b,
        invoke_identity="test/prediction-invoker/v1",
    )
    changed_identity = _operation(
        bindings,
        lambda: FakeEvaluator(tmp_path / "c.cif", []),
        [],
        invoke=invoke_a,
        invoke_identity="test/prediction-invoker/v2",
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
