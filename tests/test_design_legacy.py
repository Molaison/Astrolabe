from __future__ import annotations

import hashlib
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import Protomer, SegmentManager
from astrolabe.design import runtime as design_runtime
from astrolabe.design import legacy as design_legacy
from astrolabe.design.legacy import (
    LEGACY_REFERENCE_CODECS,
    LegacyContext,
    LegacyFileBytesCodec,
    UnsupportedLegacyState,
    invalidate_evidence_projection,
    project_evidence_dependencies,
    standard_scores_to_evidence,
    translate_legacy_state,
    validate_evidence_rebind,
)
from astrolabe.design.measurements import ImmutableMeasurementView
from astrolabe.design.store import ArrayCodec, DataStore, JSONCodec
from astrolabe.design.types import (
    Assignment,
    Axis,
    Evidence,
    Polymer,
    Reduction,
    Sample,
    key_for,
    manifest,
)
from astrolabe.evaluators.utils import StandardScores


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


def add_protomer(name: str, statement: str, kind: str = "p"):
    protomer = Protomer(name, statement, kind)
    global_config.protomers[name] = protomer
    return protomer


def sample(index: str = "0") -> Sample:
    return Sample(
        protocol=key_for("protocol"),
        replicate="replicate-1",
        seed=17,
        index=index,
    )


def test_shared_alias_is_one_variable_but_aab_has_three_occurrences():
    add_protomer("A", "AC")
    shared = global_config.segments["A_1"]
    global_config.segments["shared"] = shared
    add_protomer("B", "shared")

    translation = translate_legacy_state(
        assemblies=("AAB",),
        data=DataStore(),
    )

    assert translation.system.variables == ("A_1",)
    assert len(translation.system.occurrences) == 3
    assert [occurrence.chain for occurrence in translation.system.occurrences] == [
        "A",
        "B",
        "C",
    ]
    assert {item[0] for item in translation.assemblies[0].chains} == {"A", "B", "C"}
    assert translation.variables[0].aliases == ("A_1", "shared", "B_1")


def test_aab_keeps_two_copies_for_a_and_one_for_b():
    add_protomer("A", "AC")
    add_protomer("B", "GT")

    translation = translate_legacy_state(
        assemblies=("AAB",),
        data=DataStore(),
    )

    by_variable = {}
    for occurrence in translation.system.occurrences:
        by_variable.setdefault(occurrence.variable, []).append(occurrence.chain)
    assert by_variable == {"A_1": ["A", "B"], "B_1": ["C"]}
    assert translation.system.variables == ("A_1", "B_1")
    again = translate_legacy_state(
        assemblies=("AAB",),
        data=DataStore(),
    )
    assert again.system.identity == translation.system.identity
    assert again.candidate.identity == translation.candidate.identity


def test_molecule_ptm_fixed_locks_and_chemical_metadata_survive_round_trip():
    protein = add_protomer("A", "+AC")
    rna = add_protomer("R", "AU", "r")
    dna = add_protomer("D", "AC", "d")
    ligand = add_protomer("L", "CCO", "s")
    protein_segment = global_config.segments[protein.components[0]]
    protein_segment.ptm_dict = {1: "PHOS"}
    protein.add_locked_positions([0])
    data = DataStore()

    translation = translate_legacy_state(
        assemblies=("ARDL",),
        data=data,
    )

    assignments = {
        assignment.variable: assignment.value
        for assignment in translation.candidate.assignments
    }
    assert assignments["A_1"] == Polymer(
        "protein", ("A", "C"), ((1, "PHOS"),)
    )
    assert assignments["R_1"].kind == "rna"
    assert assignments["D_1"].kind == "dna"
    assert assignments["L_1"].__class__.__name__ == "Chemical"
    assert translation.system.domains[0].fixed_tokens == (0, 1)

    chemical_codec = JSONCodec("astrolabe.legacy.chemical/v1")
    payload = data.get(assignments["L_1"].graph, chemical_codec)
    assert payload["kind"] == "smiles"
    assert payload["value"] == "CCO"
    assert payload["is_fixed"] is True

    topology = data.get(
        translation.system.topology,
        JSONCodec("astrolabe.legacy.topology/v1"),
    )
    assert topology["protomers"]["A"]["locked_positions"] == [0]

    # The same Candidate assignments restore the mutable global state exactly.
    context = LegacyContext()
    context.apply_candidate(translation.candidate, translation=translation, data=data)
    assert global_config.segments[protein.components[0]].sequence == "AC"
    assert global_config.segments[protein.components[0]].ptm_dict == {1: "PHOS"}
    assert global_config.segments[rna.components[0]].type.name == "RNA"
    assert global_config.segments[dna.components[0]].type.name == "DNA"


def test_unsupported_chemical_shape_fails_instead_of_stringifying():
    add_protomer("L", "CCO", "s")
    segment = global_config.segments["L_1"]
    segment.type = SimpleNamespace(name="UNKNOWN")

    with pytest.raises(UnsupportedLegacyState, match="unsupported legacy segment type"):
        translate_legacy_state(assemblies=("L",), data=DataStore())


def test_context_snapshot_restores_mutations_probs_scores_caches_and_manager():
    protomer = add_protomer("A", "AC")
    segment = global_config.segments["A_1"]
    segment.probs = [np.asarray([[0.8, 0.2], [0.2, 0.8]])]
    segment.scores = [[0.9, 0.4]]
    segment.mutations = [(1, "C", "G", 0.25, 0.4)]
    protomer.add_locked_positions([0])
    _ = protomer.mutable_mask
    manager = global_config.segment_manager
    manager.begin_pre_update_step()

    data = DataStore()
    context = LegacyContext()
    translation = context.capture_candidate(assemblies=("A",), data=data)
    snapshot = context.capture_state()
    proposed = translation.system.edit(
        translation.candidate,
        (
            Assignment(
                "A_1",
                Polymer("protein", ("G", "G"), ()),
            ),
        ),
        data=data,
    )

    context.apply_candidate(proposed, translation=translation, data=data)
    assert segment.sequence == "GG"
    assert segment.mutations == []
    assert segment.probs is None
    assert segment.scores is None
    assert not hasattr(protomer, "_mutable_mask")
    assert manager._pre_update_snapshot is None

    context.restore_state(snapshot)
    assert segment.sequence == "AC"
    np.testing.assert_array_equal(
        segment.probs[0], np.asarray([[0.8, 0.2], [0.2, 0.8]])
    )
    assert segment.scores == [[0.9, 0.4]]
    assert segment.mutations == [(1, "C", "G", 0.25, 0.4)]
    assert protomer.locked_positions == frozenset({0})
    assert manager._pre_update_snapshot is not None


def test_activation_restores_constants_in_the_bound_runtime_store(tmp_path):
    add_protomer("A", "AC")
    global_config.init_const("out_dir", tmp_path / "original")
    bound = global_config.build_runtime_context()
    with LegacyContext().activate():
        global_config.update_const("out_dir", tmp_path / "temporary")
        global_config.init_const("per_residue_scores", {"A": [1.]})
    assert global_config.out_dir == tmp_path / "original"
    assert bound.get_const("out_dir") == tmp_path / "original"
    with pytest.raises(RuntimeError, match="not been initialized"):
        bound.get_const("per_residue_scores")


def test_activation_rolls_back_after_exception_and_serializes_global_owner():
    add_protomer("A", "AC")
    data = DataStore()
    context = LegacyContext()
    translation = context.capture_candidate(assemblies=("A",), data=data)
    proposed = translation.system.edit(
        translation.candidate,
        (Assignment("A_1", Polymer("protein", ("G", "G"), ())),),
        data=data,
    )

    with pytest.raises(RuntimeError, match="synthetic failure"):
        with context.activate(proposed, translation=translation, data=data):
            assert global_config.segments["A_1"].sequence == "GG"
            raise RuntimeError("synthetic failure")
    assert global_config.segments["A_1"].sequence == "AC"

    assert design_legacy._LEGACY_OWNER is design_runtime._SCIENTIFIC_OWNER
    entered = threading.Event()
    release = threading.Event()
    second_entered = threading.Event()
    errors = []
    first_context = LegacyContext()
    second_context = LegacyContext()

    def own_first():
        try:
            with first_context.activate():
                entered.set()
                assert release.wait(5)
        except BaseException as error:
            errors.append(error)

    def own_second():
        try:
            with second_context.activate():
                second_entered.set()
        except BaseException as error:
            errors.append(error)

    first = threading.Thread(target=own_first)
    second = threading.Thread(target=own_second)
    first.start()
    assert entered.wait(5)
    second.start()
    assert not second_entered.wait(0.1)
    release.set()
    first.join(5)
    second.join(5)
    assert not errors
    assert not first.is_alive() and not second.is_alive()
    assert second_entered.is_set()


def test_standard_scores_preserve_real_artifact_subjects_and_ambiguity(tmp_path):
    add_protomer("A", "AC")
    data = DataStore(tmp_path / "store")
    translation = translate_legacy_state(assemblies=("A",), data=data)
    sample_zero = sample()
    structure_bytes = b"data_test\n# synthetic immutable structure\n"
    structure_path = tmp_path / "prediction.cif"
    structure_path.write_bytes(structure_bytes)
    scores = StandardScores(
        names=["A"],
        plddt={"A": [np.asarray([0.4, 0.6], dtype=np.float32)]},
        complex_paths={"A": structure_path},
        per_proto_scores={
            "A": [np.asarray([0.11, 0.22], dtype=np.float64)]
        },
        pae={"A": [np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)]},
    )

    evidence = standard_scores_to_evidence(
        scores,
        translation.system,
        translation.candidate,
        data,
        samples={"A": (sample_zero,)},
        artifact_sample_by_assembly={"A": sample_zero},
    )

    assert len(evidence.artifacts) == 1
    artifact = evidence.artifacts[0]
    assert data.get(artifact.content, LegacyFileBytesCodec()) == structure_bytes
    assert artifact.content.identity.digest == hashlib.sha256(
        structure_bytes
    ).hexdigest()
    assert ("legacy-artifact-path" in {
        dependency.role for dependency in artifact.inputs.dependencies
    })

    plddt = next(
        measurement
        for measurement in evidence.measurements
        if ".plddt.A.prediction_" in measurement.quantity
    )
    subject = plddt.subjects[0]
    assert subject.assembly == "A"
    assert subject.occurrences == tuple(
        occurrence.identity for occurrence in translation.system.occurrences
    )
    assert subject.artifact == artifact.identity
    assert subject.sample == sample_zero
    assert len(plddt.axes) == 1
    assert plddt.axes[0].kind == "residue"
    residue_map = data.get(
        plddt.axes[0].index_map,
        JSONCodec("astrolabe.legacy.index-map/v1"),
    )
    assert [entry["token"] for entry in residue_map["entries"]] == ["A", "C"]

    plddt_values = data.get(plddt.values, ArrayCodec())
    assert plddt_values.dtype == np.float32
    assert plddt_values.shape == (2,)
    np.testing.assert_array_equal(plddt_values, np.asarray([0.4, 0.6], np.float32))

    aggregate = next(
        measurement
        for measurement in evidence.measurements
        if measurement.quantity.startswith("legacy.aggregate.per_proto_scores")
    )
    assert aggregate.subjects[0].assembly is None
    assert aggregate.subjects[0].occurrences == ()
    assert aggregate.subjects[0].sample is None
    assert aggregate.axes[0].kind == "legacy_position"


def test_missing_complex_path_is_rejected_not_claimed_as_artifact(tmp_path):
    add_protomer("A", "AC")
    data = DataStore()
    translation = translate_legacy_state(assemblies=("A",), data=data)
    scores = StandardScores(
        names=["A"],
        plddt={"A": [np.asarray([0.4, 0.6])]},
        complex_paths={"A": tmp_path / "missing.cif"},
    )

    with pytest.raises(UnsupportedLegacyState, match="is unavailable"):
        standard_scores_to_evidence(
            scores,
            translation.system,
            translation.candidate,
            data,
            samples=sample(),
        )


def test_ranked_structure_without_explicit_sample_binding_remains_unattached(
    tmp_path,
):
    add_protomer("A", "AC")
    data = DataStore(tmp_path / "store")
    translation = translate_legacy_state(assemblies=("A",), data=data)
    first = sample("0")
    second = sample("1")
    structure_bytes = b"synthetic-selected-structure"
    structure_path = tmp_path / "selected.cif"
    structure_path.write_bytes(structure_bytes)
    scores = StandardScores(
        names=["A"],
        plddt={
            "A": [
                np.asarray([0.4, 0.6], dtype=np.float32),
                np.asarray([0.5, 0.7], dtype=np.float32),
            ]
        },
        complex_paths={"A": structure_path},
    )

    evidence = standard_scores_to_evidence(
        scores,
        translation.system,
        translation.candidate,
        data,
        samples={"A": (first, second)},
    )

    assert evidence.artifacts == ()
    plddt_measurements = tuple(
        measurement
        for measurement in evidence.measurements
        if ".plddt.A.prediction_" in measurement.quantity
    )
    assert len(plddt_measurements) == 2
    assert [item.subjects[0].sample for item in plddt_measurements] == [
        first,
        second,
    ]
    assert all(item.subjects[0].artifact is None for item in plddt_measurements)
    selected = next(
        measurement
        for measurement in evidence.measurements
        if measurement.quantity == "legacy.selected_structure_bytes.A"
    )
    assert selected.subjects[0].sample is None
    assert selected.subjects[0].artifact is None
    assert data.get(selected.values, LegacyFileBytesCodec()) == structure_bytes


def test_artifact_attaches_only_to_explicitly_proven_same_sample(tmp_path):
    add_protomer("A", "AC")
    data = DataStore(tmp_path / "store")
    translation = translate_legacy_state(assemblies=("A",), data=data)
    selected = sample("selected")
    other = sample("other")
    structure_path = tmp_path / "selected.cif"
    structure_path.write_bytes(b"selected")
    scores = StandardScores(
        names=["A"],
        plddt={
            "A": [
                np.asarray([0.4, 0.6]),
                np.asarray([0.5, 0.7]),
            ]
        },
        complex_paths={"A": structure_path},
    )

    evidence = standard_scores_to_evidence(
        scores,
        translation.system,
        translation.candidate,
        data,
        samples={"A": (selected, other)},
        artifact_sample_by_assembly={"A": selected},
    )

    assert len(evidence.artifacts) == 1
    artifact = evidence.artifacts[0]
    plddt_measurements = tuple(
        measurement
        for measurement in evidence.measurements
        if ".plddt.A.prediction_" in measurement.quantity
    )
    assert plddt_measurements[0].subjects[0].artifact == artifact.identity
    assert plddt_measurements[1].subjects[0].artifact is None


def test_reference_codecs_cover_every_legacy_blob_for_cold_runtime_recovery(
    tmp_path,
):
    add_protomer("A", "AC")
    data = DataStore(tmp_path / "store")
    translation = translate_legacy_state(assemblies=("A",), data=data)
    identities = {codec.identity for codec in LEGACY_REFERENCE_CODECS}
    assert translation.system.topology.codec in identities
    assert all(
        occurrence.index_map.codec in identities
        for occurrence in translation.system.occurrences
    )
    assert {
        assignment.value.graph.codec
        for assignment in translation.candidate.assignments
        if hasattr(assignment.value, "graph")
    } <= identities
    assert LegacyFileBytesCodec().identity in identities


def test_variable_edit_invalidates_each_coupled_assembly_only():
    add_protomer("A", "AC")
    add_protomer("B", "GT")
    data = DataStore()
    translation = translate_legacy_state(
        assemblies=("A", "B", "AB"),
        data=data,
    )
    evidence = Evidence(
        identity=key_for("evidence"),
        evaluated_candidate=translation.candidate.identity,
        artifacts=(),
        measurements=(),
        inputs=manifest(()),
    )
    projection = project_evidence_dependencies(
        evidence,
        translation.system,
        translation.candidate,
        assemblies=("A", "B", "AB"),
    )
    edited = translation.system.edit(
        translation.candidate,
        (Assignment("A_1", Polymer("protein", ("G", "C"), ())),),
        data=data,
    )

    validity = invalidate_evidence_projection(
        projection,
        translation.system,
        edited,
    )
    assert {item.assembly: item.status for item in validity} == {
        "A": "invalid",
        "AB": "invalid",
        "B": "reference",
    }
    assert validate_evidence_rebind(
        projection, "B", translation.system, edited
    ).status == "reference"
    with pytest.raises(ValueError, match="changed variable dependencies"):
        validate_evidence_rebind(projection, "A", translation.system, edited)
    with pytest.raises(ValueError, match="changed variable dependencies"):
        validate_evidence_rebind(projection, "AB", translation.system, edited)


def test_nonfinite_scalar_preserves_raw_dtype_and_records_mask():
    add_protomer("A", "AC")
    data = DataStore()
    translation = translate_legacy_state(assemblies=("A",), data=data)
    raw = np.float32("nan")
    scores = StandardScores(
        names=["A"],
        plddt={"A": [np.asarray([0.4, 0.6], dtype=np.float32)]},
        complex_paths={},
        iptm={"A": [raw]},
    )

    evidence = standard_scores_to_evidence(
        scores,
        translation.system,
        translation.candidate,
        data,
    )
    measurement = next(
        item
        for item in evidence.measurements
        if ".iptm.A.prediction_0" in item.quantity
    )

    values = data.get(measurement.values, ArrayCodec())
    mask = data.get(measurement.valid_mask, ArrayCodec())
    assert values.dtype == np.float32
    assert values.shape == ()
    assert np.isnan(values.item())
    assert mask.dtype == np.bool_
    assert mask.shape == ()
    assert bool(mask.item()) is False
    assert measurement.missing == "partial"
    assert measurement.valid_mask is not None
    without_mask = key_for(
        (
            "measurement/v1",
            measurement.quantity,
            measurement.definition,
            measurement.units,
            measurement.producer,
            measurement.subjects,
            measurement.axes,
            measurement.values,
            measurement.missing,
            None,
            measurement.inputs,
        )
    )
    assert measurement.identity != without_mask
    assert np.isnan(scores.iptm["A"][0])


def test_mixed_nonfinite_array_keeps_repeated_chain_axes_and_raw_snapshot():
    add_protomer("A", "AC")
    shared = global_config.segments["A_1"]
    global_config.segments["shared"] = shared
    add_protomer("B", "shared")
    data = DataStore()
    translation = translate_legacy_state(assemblies=("AAB",), data=data)
    raw = np.asarray([0.91, np.nan, 0.73, 0.82, 0.67, 0.94], dtype=np.float32)
    original = raw.copy()
    scores = StandardScores(
        names=["AAB"],
        plddt={"AAB": [raw]},
        complex_paths={},
    )

    evidence = standard_scores_to_evidence(
        scores,
        translation.system,
        translation.candidate,
        data,
    )
    measurement = next(
        item
        for item in evidence.measurements
        if ".plddt.AAB.prediction_0" in item.quantity
    )
    values = data.get(measurement.values, ArrayCodec())
    mask = data.get(measurement.valid_mask, ArrayCodec())
    index_map = data.get(
        measurement.axes[0].index_map,
        JSONCodec("astrolabe.legacy.index-map/v1"),
    )

    assert measurement.missing == "partial"
    assert values.dtype == np.float32
    assert values.shape == (6,)
    np.testing.assert_array_equal(values, original)
    np.testing.assert_array_equal(
        mask,
        np.asarray([True, False, True, True, True, True]),
    )
    assert measurement.axes[0].kind == "residue"
    assert [entry["chain"] for entry in index_map["entries"]] == [
        "A",
        "A",
        "B",
        "B",
        "C",
        "C",
    ]
    assert {
        entry["occurrence"] for entry in index_map["entries"][:2]
    } == {index_map["entries"][0]["occurrence"]}
    first_copy, second_copy, third_copy = (
        index_map["entries"][index] for index in (0, 2, 4)
    )
    assert first_copy["occurrence"] != second_copy["occurrence"]
    assert first_copy["variable"] == second_copy["variable"]
    assert second_copy["variable"] == third_copy["variable"]
    assert len(
        {
            first_copy["occurrence"],
            second_copy["occurrence"],
            third_copy["occurrence"],
        }
    ) == 3
    np.testing.assert_array_equal(raw, original)


def test_all_nonfinite_values_are_partial_but_failed_stays_explicit():
    add_protomer("A", "AC")
    data = DataStore()
    translation = translate_legacy_state(assemblies=("A",), data=data)
    raw = np.asarray([np.nan, np.inf], dtype=np.float32)
    scores = StandardScores(
        names=["A"],
        plddt={"A": [np.asarray([0.4, 0.6])]},
        complex_paths={},
        pae={"A": [raw]},
    )
    evidence = standard_scores_to_evidence(
        scores,
        translation.system,
        translation.candidate,
        data,
    )
    partial_measurement = next(
        item
        for item in evidence.measurements
        if ".pae.A.prediction_0" in item.quantity
    )
    values = data.get(partial_measurement.values, ArrayCodec())
    mask = data.get(partial_measurement.valid_mask, ArrayCodec())
    assert partial_measurement.missing == "partial"
    assert values.dtype == np.float32
    np.testing.assert_array_equal(values, raw)
    np.testing.assert_array_equal(mask, np.asarray([False, False]))

    failed = design_legacy._make_measurement(
        data,
        system=translation.system,
        candidate=translation.candidate,
        state="legacy",
        quantity="synthetic.failed",
        units="unitless",
        value=raw,
        subject=partial_measurement.subjects[0],
        axes=partial_measurement.axes,
        provenance="explicit failed fixture",
        missing="failed",
    )
    assert failed.missing == "failed"
    assert failed.valid_mask is not None
    np.testing.assert_array_equal(
        data.get(failed.valid_mask, ArrayCodec()),
        np.asarray([False, False]),
    )


def test_none_and_unsupported_nonfinite_metadata_keep_existing_semantics():
    add_protomer("A", "AC")
    data = DataStore()
    translation = translate_legacy_state(assemblies=("A",), data=data)
    scores = StandardScores(
        names=["A"],
        plddt={"A": [np.asarray([0.4, 0.6])]},
        complex_paths={},
        iptm={"A": [None]},
    )
    evidence = standard_scores_to_evidence(
        scores,
        translation.system,
        translation.candidate,
        data,
    )
    absent = next(
        item
        for item in evidence.measurements
        if ".iptm.A.prediction_0" in item.quantity
    )
    assert absent.values is None
    assert absent.valid_mask is None
    assert absent.missing == "unavailable"

    with pytest.raises(UnsupportedLegacyState, match="non-finite metadata"):
        design_legacy._make_measurement(
            data,
            system=translation.system,
            candidate=translation.candidate,
            state="legacy",
            quantity="synthetic.metadata",
            units="unitless",
            value={"unknown": np.nan},
            subject=absent.subjects[0],
            axes=(),
            provenance="unsupported metadata fixture",
        )


def test_measurement_reader_fail_and_omit_use_nonfinite_valid_mask():
    add_protomer("A", "AC")
    data = DataStore()
    translation = translate_legacy_state(assemblies=("A",), data=data)
    scores = StandardScores(
        names=["A"],
        plddt={"A": [np.asarray([0.6, np.nan], dtype=np.float32)]},
        complex_paths={},
    )
    evidence = standard_scores_to_evidence(
        scores,
        translation.system,
        translation.candidate,
        data,
    )
    measurement = next(
        item
        for item in evidence.measurements
        if ".plddt.A.prediction_0" in item.quantity
    )
    measurement = design_legacy._make_measurement(
        data,
        system=translation.system,
        candidate=translation.candidate,
        state="legacy",
        quantity="synthetic.reader_nonfinite",
        units="unitless",
        value=np.asarray([0.6, np.nan], dtype=np.float32),
        subject=measurement.subjects[0],
        axes=(
            Axis(
                name="residue",
                kind="residue",
                index_map=data.put({"labels": ["0", "1"]}, JSONCodec()),
            ),
        ),
        provenance="synthetic reader fixture",
    )
    view = ImmutableMeasurementView(measurement, data)
    parameters = data.put(
        {"schema": "synthetic-reduction/v1"},
        JSONCodec("synthetic-reduction/v1"),
    )

    with pytest.raises(ValueError, match="missing='fail'"):
        view.reduce(
            Reduction(
                axes=("residue",),
                method="mean",
                parameters=parameters,
                missing="fail",
            )
        )

    omitted = view.reduce(
        Reduction(
            axes=("residue",),
            method="mean",
            parameters=parameters,
            missing="omit",
        )
    )
    reduced = data.get(omitted.values, ArrayCodec())
    assert omitted.missing == "present"
    assert reduced.shape == ()
    assert float(reduced) == pytest.approx(0.6)
