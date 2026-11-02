"""Reviewer repros use real legacy state, emitted measurements and shared owner."""
from dataclasses import replace
import threading

import numpy as np
import pytest

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import Protomer, SegmentManager
from astrolabe.design import legacy
from astrolabe.design.measurements import ImmutableMeasurementView
from astrolabe.design.owner import SCIENTIFIC_OWNER
from astrolabe.design.store import ArrayCodec, DataStore, JSONCodec
from astrolabe.design.types import Assignment, Polymer, Reduction, Selection, make_candidate, make_system
from astrolabe.evaluators.utils import StandardScores
from astrolabe.score_contributions import ScoreContribution


@pytest.fixture(autouse=True)
def state():
    global_config.reset()
    for name in ("segments", "protomers", "mutable_protomers", "named_files"):
        global_config.init_const(name, {})
    global_config.init_const("segment_manager", SegmentManager())
    yield
    global_config.reset()


def add(name="A", sequence="AC"):
    protomer = Protomer(name, sequence, "p")
    global_config.protomers[name] = protomer
    return protomer


def scores_for(translation, data, **values):
    scores = StandardScores(names=["A"], plddt={}, complex_paths={}, **values)
    return legacy.standard_scores_to_evidence(scores, translation.system, translation.candidate, data)


@pytest.mark.parametrize("drift", ["cyclization", "deleted_lock", "occurrence"])
def test_review01_activate_rejects_full_system_drift_before_entering(drift):
    a = add()
    a.add_locked_positions([0])
    global_config.init_const("cyclization", {"A": False})
    data = DataStore()
    context = legacy.LegacyContext()
    translation = context.capture_candidate(assemblies=("A",), data=data)
    if drift == "cyclization":
        global_config.cyclization["A"] = True
    elif drift == "deleted_lock":
        a._locked_positions.clear()
    else:
        a.components = ["A_1", "A_1"]
    with pytest.raises(legacy.UnsupportedLegacyState, match="[Ss]ystem|topology|lock"):
        with context.activate(translation.candidate, translation=translation, data=data):
            pytest.fail("stale scientific identity must not execute")
    assert not context.active
    assert global_config.segments["A_1"].sequence == "AC"


@pytest.mark.parametrize("drift", ["forged_target", "forged_source", "forged_source_system", "forged_occurrence", "topology", "occurrence"])
def test_review02_rebind_validates_both_systems_and_candidate_contents(drift):
    add()
    data = DataStore()
    original = legacy.translate_legacy_state(assemblies=("A",), data=data)
    evidence = scores_for(original, data, iptm={"A": [0.7]})
    projection = legacy.project_evidence_dependencies(evidence, original.system, original.candidate)
    system, candidate = original.system, original.candidate
    if drift in {"forged_target", "forged_source"}:
        forged = replace(candidate, assignments=(Assignment("A_1", Polymer("protein", tuple("GG"), ())),))
        if drift == "forged_source":
            projection = replace(projection, candidate=forged)
        else:
            candidate = forged
    elif drift == "forged_source_system":
        projection = replace(projection, system=replace(system, topology=data.put({"forged": True}, JSONCodec())))
    elif drift == "forged_occurrence":
        system = replace(system, occurrences=(replace(system.occurrences[0], chain="Z"),))
    else:
        topology = data.put({"different": "topology"}, JSONCodec()) if drift == "topology" else system.topology
        occurrences = (replace(system.occurrences[0], chain="Z"),) if drift == "occurrence" else system.occurrences
        system = make_system(system.variables, occurrences, topology, system.domains)
        candidate = make_candidate(system, candidate.assignments, data=data)
    with pytest.raises(ValueError, match="identity|[Ss]ystem|topology|occurrence|content"):
        legacy.validate_evidence_rebind(projection, "A", system, candidate)
    assert projection.evidence is evidence
    assert evidence.evaluated_candidate == original.candidate.identity


@pytest.mark.parametrize("value", [0.7, 2, np.float32(0.8)])
def test_review03_finite_scalar_measurements_are_actual_reader_inputs(value):
    add()
    data = DataStore()
    translation = legacy.translate_legacy_state(assemblies=("A",), data=data)
    measurement, = scores_for(translation, data, iptm={"A": [value]}).measurements
    view = ImmutableMeasurementView(measurement, data)
    assert view.measurement is measurement
    array = data.get(measurement.values, ArrayCodec())
    assert array.shape == ()
    assert array.dtype == np.asarray(value).dtype
    assert array.item() == pytest.approx(value)


@pytest.mark.parametrize("field,value", [
    ("plddt", np.array([0.6, np.nan], dtype=np.float32)),
    ("pae", np.array([[1., 2.], [3., 4.]], dtype=np.float32)),
    ("chain_pair_scores", np.arange(6.).reshape(2, 3)),
    ("per_proto_scores", np.array([0.4, 0.8])),
])
def test_review04_actual_array_axes_can_select_and_reduce(field, value):
    add()
    data = DataStore()
    translation = legacy.translate_legacy_state(assemblies=("A",), data=data)
    payload = {"A": {"test_pair": [value]}} if field == "chain_pair_scores" else {"A": [value]}
    scores = StandardScores(names=["A"], complex_paths={}, plddt={})
    setattr(scores, field, payload)
    evidence = legacy.standard_scores_to_evidence(scores, translation.system, translation.candidate, data)
    measurement, = evidence.measurements
    view = ImmutableMeasurementView(measurement, data)
    for axis, size in zip(measurement.axes, value.shape):
        axis_map = data.get(axis.index_map, JSONCodec())
        assert len(axis_map["labels"]) == len(set(axis_map["labels"])) == size
    axis = measurement.axes[0]
    selection = Selection(measurement.subjects[0], axis.name, axis.index_map, data.put(np.ones(value.shape[0], dtype=bool), ArrayCodec()))
    reduced = view.select(selection).reduce(Reduction(
        tuple(axis.name for axis in measurement.axes), "mean", data.put({}, JSONCodec()), "omit",
    ))
    assert float(data.get(reduced.values, ArrayCodec())) == pytest.approx(float(np.nanmean(value)))
    assert measurement.identity in [dep.identity for dep in reduced.inputs.dependencies if dep.role.endswith("source-measurement")]


def test_review05_exception_restores_complete_registrations_and_manager_owner():
    a = add()
    segment = global_config.segments["A_1"]
    global_config.segments["alias"] = segment
    segments, protomers, manager = global_config.segments, global_config.protomers, global_config.segment_manager
    managed, unique, primary = manager.segments, manager.unique_segments, manager.primary_names
    saved_managed = dict(managed)
    with pytest.raises(RuntimeError, match="evaluation failed"):
        with legacy.LegacyContext().activate():
            segment.sequence = "GG"
            a.components.clear()
            a._locked_positions.add(1)
            managed.clear()
            unique.clear()
            primary.clear()
            segments.clear()
            protomers.clear()
            global_config.update_const("segments", {})
            global_config.update_const("protomers", {})
            global_config.update_const("segment_manager", SegmentManager())
            raise RuntimeError("evaluation failed")
    assert global_config.segments is segments
    assert global_config.protomers is protomers
    assert global_config.protomers == {"A": a}
    assert global_config.segments == {"A_1": segment, "alias": segment}
    assert segment.sequence == "AC" and a.components == ["A_1"]
    assert a.locked_positions == frozenset()
    assert global_config.segment_manager is manager
    assert manager.segments is managed and managed == saved_managed
    assert manager.unique_segments is unique and unique == {id(segment): segment}
    assert manager.primary_names is primary and primary == {id(segment): "A_1"}


@pytest.mark.parametrize("check", ["references", "manager_aliases"])
def test_review06_empty_activation_keeps_references_and_each_registry_exact(check):
    a = add()
    segment = global_config.segments["A_1"]
    global_config.segments["alias"] = segment
    segment.probs = [np.array([0.25, 0.75])]
    locks, probs, array = a._locked_positions, segment.probs, segment.probs[0]
    binding_site = {"A": [0]}
    global_config.init_const("binding_site", binding_site)
    manager = global_config.segment_manager
    expected = dict(manager.segments)
    with legacy.LegacyContext().activate():
        pass
    if check == "references":
        assert a._locked_positions is locks
        assert segment.probs is probs and segment.probs[0] is array
        assert global_config.binding_site is binding_site
    else:
        assert manager.segments == expected
        assert "alias" not in manager.segments


@pytest.mark.parametrize("field,value", [("aggregation", "mean"), ("provenance", "different source record")])
def test_review07_contribution_definition_retains_aggregation_and_provenance(field, value):
    add()
    data = DataStore()
    translation = legacy.translate_legacy_state(assemblies=("A",), data=data)
    base = ScoreContribution("component", "metric", 0.7, aggregation="identity", provenance="original source record")
    before, = scores_for(translation, data, score_contributions=[base]).measurements
    after, = scores_for(translation, data, score_contributions=[replace(base, **{field: value})]).measurements
    assert before.definition != after.definition
    assert before.identity != after.identity


def test_review08_native_sequence_restore_obeys_shared_owner():
    a = add()
    began, completed = threading.Event(), threading.Event()
    errors = []

    def restore():
        began.set()
        try:
            legacy.restore_native_sequence("GG", a, global_config.segments)
        except BaseException as error:
            errors.append(error)
        finally:
            completed.set()

    with SCIENTIFIC_OWNER:
        thread = threading.Thread(target=restore)
        thread.start()
        assert began.wait(5)
        premature = completed.wait(0.1)
        protected_sequence = global_config.segments["A_1"].sequence
    thread.join(5)
    assert not thread.is_alive() and not errors
    assert not premature and protected_sequence == "AC"
    assert completed.is_set() and global_config.segments["A_1"].sequence == "GG"


def test_review01_activation_preflight_keeps_references_on_rejection():
    a = add()
    a.add_locked_positions([0])
    data = DataStore()
    translation = legacy.translate_legacy_state(assemblies=("A",), data=data)
    a._locked_positions.clear()
    locks = a._locked_positions
    manager_names = tuple(global_config.segment_manager.segments)
    with pytest.raises(legacy.UnsupportedLegacyState, match="System"):
        with legacy.LegacyContext().activate(translation.candidate, translation=translation, data=data):
            pytest.fail("must reject before entering")
    assert a._locked_positions is locks and locks == set()
    assert tuple(global_config.segment_manager.segments) == manager_names


def test_review02_real_scoped_reference_keeps_original_identity_and_aggregate_cannot_rebind():
    add("A", "AC")
    add("B", "GT")
    data = DataStore()
    translation = legacy.translate_legacy_state(assemblies=("A", "B"), data=data)
    edited = translation.system.edit(translation.candidate, (
        Assignment("A_1", Polymer("protein", tuple("GG"), ())),
    ), data=data)
    scores = StandardScores(names=["B"], complex_paths={}, plddt={"B": [np.array([0.7, 0.8])]})
    evidence = legacy.standard_scores_to_evidence(scores, translation.system, translation.candidate, data)
    projection = legacy.project_evidence_dependencies(evidence, translation.system, translation.candidate)
    assert legacy.validate_evidence_rebind(projection, "B", translation.system, edited).status == "reference"
    assert projection.evidence is evidence
    assert evidence.measurements[0].subjects[0].candidate == translation.candidate.identity
    with pytest.raises(ValueError, match="identity"):
        legacy.validate_evidence_rebind(
            replace(projection, evidence=replace(evidence, measurements=())), "B", translation.system, edited,
        )
    scores.per_proto_scores = {"B": [np.array([0.4, 0.8])]}
    aggregate_evidence = legacy.standard_scores_to_evidence(scores, translation.system, translation.candidate, data)
    aggregate = legacy.project_evidence_dependencies(aggregate_evidence, translation.system, translation.candidate)
    with pytest.raises(ValueError, match="changed variable"):
        legacy.validate_evidence_rebind(aggregate, "B", translation.system, edited)


def test_review05_mutable_protomer_registry_is_restored_as_the_same_owner():
    a = add()
    registry = global_config.mutable_protomers
    registry["A"] = a
    with pytest.raises(RuntimeError, match="failed"):
        with legacy.LegacyContext().activate():
            registry.clear()
            global_config.update_const("mutable_protomers", {})
            raise RuntimeError("failed")
    assert global_config.mutable_protomers is registry
    assert registry == {"A": a}


def test_review06_empty_activation_keeps_readonly_arrays_and_shared_caches():
    a = add()
    segment = global_config.segments["A_1"]
    array = np.array([0.25, 0.75])
    array.flags.writeable = False
    segment.probs = [array]
    cache = {"array": array}
    a._homology = cache
    with legacy.LegacyContext().activate():
        pass
    assert segment.probs[0] is array and not array.flags.writeable
    assert a._homology is cache and cache["array"] is array


def test_review07_source_record_preserves_substantive_contribution_fields():
    import json
    from astrolabe.design.types import Blob

    add()
    data = DataStore()
    translation = legacy.translate_legacy_state(assemblies=("A",), data=data)
    contribution = ScoreContribution("component", "metric", 0.7, aggregation="mean", provenance="measured on source structures")
    measurement, = scores_for(translation, data, score_contributions=[contribution]).measurements
    record, = [dep for dep in measurement.inputs.dependencies if dep.role == "legacy-provenance-record"]
    payload = data.get(Blob(record.identity, JSONCodec().identity), JSONCodec())
    source = json.loads(payload["description"])
    assert source["aggregation"] == "mean"
    assert source["provenance"] == "measured on source structures"
    assert source["shape"] == "scalar" and source["scope"] == "global"
    ImmutableMeasurementView(measurement, data)


def test_review04_unbound_structure_record_is_durable_and_numeric_view_is_explicitly_unavailable(tmp_path):
    from astrolabe.design.types import Blob

    add()
    data = DataStore(tmp_path / "store")
    translation = legacy.translate_legacy_state(assemblies=("A",), data=data)
    path = tmp_path / "selected.cif"
    content = b"data_source_structure\n# actual test source bytes\n"
    path.write_bytes(content)
    scores = StandardScores(names=["A"], complex_paths={"A": path}, plddt={})
    evidence = legacy.standard_scores_to_evidence(scores, translation.system, translation.candidate, data)
    measurement, = evidence.measurements
    ImmutableMeasurementView(measurement, data)
    assert measurement.values is None and measurement.missing == "unavailable"
    assert measurement.subjects[0].sample is None and not evidence.artifacts
    source, = [dep for dep in measurement.inputs.dependencies if dep.role.startswith("legacy-source-record/")]
    reference = Blob(source.identity, source.role.removeprefix("legacy-source-record/"))
    path.unlink()
    cold = DataStore(data.root)
    assert cold.get(reference, legacy.LegacyFileBytesCodec()) == content
    provenance, = [dep for dep in measurement.inputs.dependencies if dep.role == "legacy-provenance-record"]
    record = cold.get(Blob(provenance.identity, JSONCodec().identity), JSONCodec())
    assert str(path) in record["description"]


def test_review01_full_local_lock_union_does_not_hide_fixed_flag_drift():
    a = add()
    a.add_locked_positions([0, 1])
    data = DataStore()
    translation = legacy.translate_legacy_state(assemblies=("A",), data=data)
    global_config.segments["A_1"].is_fixed = True
    with pytest.raises(legacy.UnsupportedLegacyState, match="System"):
        with legacy.LegacyContext().activate(translation.candidate, translation=translation, data=data):
            pytest.fail("changed native fixed semantics must be recaptured")
    assert global_config.segments["A_1"].is_fixed is True


@pytest.mark.parametrize("value", [np.array([], dtype=np.float32), np.empty((0, 2)), np.zeros((2, 2, 2))])
def test_review04_empty_and_higher_rank_arrays_keep_explicit_reader_contract(value):
    add()
    data = DataStore()
    translation = legacy.translate_legacy_state(assemblies=("A",), data=data)
    scores = StandardScores(names=["A"], complex_paths={}, plddt={})
    scores.pae = {"A": [value]}
    evidence = legacy.standard_scores_to_evidence(scores, translation.system, translation.candidate, data)
    measurement, = evidence.measurements
    ImmutableMeasurementView(measurement, data)
    assert data.get(measurement.values, ArrayCodec()).shape == value.shape
    assert len(measurement.axes) == value.ndim
    if value.size == 0:
        assert measurement.missing == "partial"
        assert data.get(measurement.valid_mask, ArrayCodec()).size == 0


def test_review05_unowned_opaque_state_is_rejected_before_execution():
    add()
    segment = global_config.segments["A_1"]
    handle = object()
    segment.unowned_handle = handle
    context = legacy.LegacyContext()
    with pytest.raises(legacy.UnsupportedLegacyState, match="snapshot|owner"):
        with context.activate():
            pytest.fail("unrestorable state must fail before entry")
    assert not context.active and context.owner_thread is None
    assert segment.unowned_handle is handle
    assert segment.sequence == "AC"
