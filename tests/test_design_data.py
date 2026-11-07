from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from astrolabe.design.store import (
    ArrayCodec,
    DataStore,
    DataclassCodec,
    JSONCodec,
    design_codec,
)
from astrolabe.design.types import (
    Assignment,
    Blob,
    Chemical,
    Dependency,
    Domain,
    Evidence,
    EvidenceUse,
    Key,
    Measurement,
    Occurrence,
    Polymer,
    Subject,
    key_for,
    make_candidate,
    make_system,
    manifest,
)


@dataclass(frozen=True)
class NestedValue:
    label: str
    values: tuple[int, ...]


@dataclass(frozen=True)
class Payload:
    nested: NestedValue
    blob: Blob
    mapping: dict[str, tuple[str, ...]]


class GraphCodec:
    identity = "test/chemical-graph/v1"

    def encode(self, value):
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    def decode(self, data):
        return json.loads(data.decode("utf-8"))

    def references(self, value):
        return ()

    def compatibility_key(self):
        return (type(self).__name__, self.identity)

    def validate_edit(self, old, new, *, topology, occurrence_maps):
        # This test owner supports edits preserving atom IDs and attachments.
        return set(old["atoms"]) == set(new["atoms"]) and old.get("attachments") == new.get("attachments")

    def fixed_atom_records(self, value, fixed_atoms):
        records = {}
        for atom_id in fixed_atoms:
            if atom_id not in value["atoms"]:
                raise ValueError(f"fixed atom {atom_id!r} is absent")
            incident_bonds = [
                tuple(bond)
                for bond in value["bonds"]
                if atom_id in bond[:2]
            ]
            records[atom_id] = {
                "atom": value["atoms"][atom_id],
                "bonds": tuple(sorted(incident_bonds)),
                "attachments": tuple(
                    attachment
                    for attachment in value.get("attachments", ())
                    if atom_id in attachment
                ),
            }
        return records


def _topology(data: DataStore) -> Blob:
    return data.put({"cross_bonds": []}, JSONCodec())


def _index_map(data: DataStore, labels: tuple[str, ...]) -> Blob:
    return data.put({"labels": list(labels)}, JSONCodec())


def _polymer_system(data: DataStore):
    topology = _topology(data)
    occurrence = Occurrence(
        identity=key_for("occurrence/A/1"),
        state="state",
        assembly="assembly",
        chain="A",
        component="A_1",
        variable="A",
        index_map=_index_map(data, ("A", "B")),
    )
    allowed = data.put(
        {"allowed_tokens": {"1": ["C", "G"]}},
        JSONCodec(),
    )
    system = make_system(
        ("A", "B"),
        (occurrence,),
        topology,
        (Domain("A", (0,), (), allowed), Domain("B", (1,), (), None)),
    )
    candidate = make_candidate(
        system,
        (
            Assignment("B", Polymer("protein", ("T", "G"), ())),
            Assignment("A", Polymer("protein", ("A", "C"), ((1, "PTM1"),))),
        ),
    )
    return system, candidate, data


def test_keys_and_factories_are_content_based_and_assignment_order_independent():
    data = DataStore()
    system, candidate, _ = _polymer_system(data)
    reordered = make_candidate(
        system,
        (
            Assignment("A", Polymer("protein", ("A", "C"), ((1, "PTM1"),))),
            Assignment("B", Polymer("protein", ("T", "G"), ())),
        ),
    )

    assert candidate == reordered
    assert [item.variable for item in candidate.assignments] == ["A", "B"]
    assert key_for({"b": (2, 1), "a": [True]}) == key_for(
        {"a": [True], "b": (2, 1)}
    )
    assert key_for({"value": 1}) != key_for({"value": 2})

    first = manifest(
        (
            Dependency("source", key_for("second")),
            Dependency("model", key_for("first")),
        )
    )
    second = manifest(
        (
            Dependency("model", key_for("first")),
            Dependency("source", key_for("second")),
            Dependency("model", key_for("first")),
        )
    )
    assert first == second


def test_mutated_candidate_cannot_relabel_old_evidence_as_current():
    evidence = Evidence(key_for("evidence"), key_for("parent"), (), (), manifest(()))
    with pytest.raises(ValueError, match="Current evidence"):
        EvidenceUse(evidence, "current", key_for("child"))
    guidance = EvidenceUse(evidence, "base_guidance", key_for("child"))
    assert guidance.evidence.evaluated_candidate == key_for("parent")


def test_system_factory_rejects_unowned_occurrences_and_duplicate_domains():
    data = DataStore()
    topology = _topology(data)
    occurrence = Occurrence(
        identity=key_for("occurrence"),
        state="s",
        assembly="a",
        chain="A",
        component="A_1",
        variable="missing",
        index_map=_index_map(data, ("A",)),
    )
    with pytest.raises(ValueError, match="unknown variable"):
        make_system(("A",), (occurrence,), topology)

    domain = Domain("A", (), (), None)
    with pytest.raises(ValueError, match="multiple domains"):
        make_system(("A",), (), topology, (domain, domain))


def test_system_edit_changes_whole_assignment_and_preserves_base():
    system, base, data = _polymer_system(DataStore())
    edited = system.edit(
        base,
        (
            Assignment(
                "B",
                Polymer("protein", ("A", "G"), ()),
            ),
        ),
        data=data,
    )

    assert base.assignments[0].value.tokens == ("A", "C")
    assert edited.assignments[1].value.tokens == ("A", "G")
    assert edited.assignments[0].value.modifications == ((1, "PTM1"),)
    assert edited.identity != base.identity


def test_system_edit_rejects_length_fixed_token_ptm_and_restriction_changes():
    system, base, data = _polymer_system(DataStore())

    with pytest.raises(ValueError, match="token count"):
        system.edit(
            base,
            (Assignment("B", Polymer("protein", ("T",), ())),),
            data=data,
        )
    with pytest.raises(ValueError, match="fixed token"):
        system.edit(
            base,
            (Assignment("A", Polymer("protein", ("C", "C"), ((1, "PTM1"),))),),
            data=data,
        )
    with pytest.raises(ValueError, match="modification topology"):
        system.edit(
            base,
            (
                Assignment(
                    "A",
                    Polymer("protein", ("A", "G"), ((0, "PTM1"), (1, "PTM2"))),
                ),
            ),
            data=data,
        )
    unmodified = make_candidate(system, (Assignment("A", Polymer("protein", ("A", "C"), ())), base.assignments[1]))
    with pytest.raises(ValueError, match="not allowed"):
        system.edit(
            unmodified,
            (
                Assignment(
                    "A",
                    Polymer("protein", ("A", "T"), ()),
                ),
            ),
            data=data,
        )
    with pytest.raises(ValueError, match="DataStore"):
        system.edit(
            base,
            (
                Assignment(
                    "A",
                    Polymer("protein", ("A", "C"), ((1, "PTM1"),)),
                ),
            ),
        )


def test_restriction_decoder_rejects_unknown_json_schema():
    data = DataStore()
    topology = _topology(data)
    restriction = data.put({"condition": "do not ignore me"}, JSONCodec())
    system = make_system(
        ("A",),
        (),
        topology,
        (Domain("A", (), (), restriction),),
    )
    base = make_candidate(
        system,
        (Assignment("A", Polymer("protein", ("A",), ())),),
    )

    with pytest.raises(ValueError, match="exactly the allowed_tokens"):
        system.edit(
            base,
            (Assignment("A", Polymer("protein", ("G",), ())),),
            data=data,
        )


def test_system_edit_rejects_partial_or_forged_candidates():
    system, base, data = _polymer_system(DataStore())
    with pytest.raises(ValueError, match="missing assignments"):
        make_candidate(
            system,
            (Assignment("A", Polymer("protein", ("A", "C"), ())),),
        )

    forged = type(base)(
        identity=key_for("not-the-content"),
        system=base.system,
        assignments=base.assignments,
    )
    with pytest.raises(ValueError, match="identity"):
        system.edit(
            forged,
            (Assignment("A", Polymer("protein", ("A", "G"), ((1, "PTM2"),))),),
            data=data,
        )


def test_domain_absence_does_not_allow_unrepresented_topology_changes():
    data = DataStore()
    residue = make_system(
        ("A",),
        (),
        _topology(data),
    )
    base = make_candidate(
        residue,
        (Assignment("A", Polymer("protein", ("A",), ())),),
    )
    with pytest.raises(ValueError, match="modification topology"):
        residue.edit(
            base,
            (Assignment("A", Polymer("protein", ("G",), ((0, "PTM1"),))),),
        )

    graph_codec = GraphCodec()
    first_graph = data.put(
        {"atoms": {"C1": {"element": "C"}}, "bonds": ()},
        graph_codec,
    )
    second_graph = data.put(
        {"atoms": {"C1": {"element": "C"}}, "bonds": (("C1", "C1", 2),)},
        graph_codec,
    )
    chemical = make_system(("L",), (), _topology(data))
    chemical_base = make_candidate(
        chemical,
        (Assignment("L", Chemical(first_graph)),),
    )
    with pytest.raises(ValueError, match="chemical graph topology"):
        chemical.edit(
            chemical_base,
            (Assignment("L", Chemical(second_graph)),),
            data=data,
        )


def test_chemical_fixed_atoms_delegate_full_owner_graph_validation():
    data = DataStore()
    graph_codec = GraphCodec()
    base_graph = data.put(
        {
            "atoms": {
                "C1": {"element": "C", "charge": 0},
                "O1": {"element": "O", "charge": 0},
            },
            "bonds": [("C1", "O1", 1)],
            "attachments": [("C1", "protein")],
        },
        graph_codec,
    )
    system = make_system(
        ("L",),
        (),
        _topology(data),
        (Domain("L", (), ("C1",), None),),
    )
    base = make_candidate(
        system,
        (Assignment("L", Chemical(base_graph)),),
    )

    compatible_graph = data.put(
        {
            "atoms": {
                "C1": {"element": "C", "charge": 0},
                "O1": {"element": "O", "charge": -1},
            },
            "bonds": [("C1", "O1", 1)],
            "attachments": [("C1", "protein")],
        },
        graph_codec,
    )
    edited = system.edit(
        base,
        (Assignment("L", Chemical(compatible_graph)),),
        data=data,
    )
    assert edited != base

    boundary_changed = data.put(
        {
            "atoms": {
                "C1": {"element": "C", "charge": 0},
                "O1": {"element": "O", "charge": 0},
            },
            "bonds": [("C1", "O1", 2)],
            "attachments": [("C1", "protein")],
        },
        graph_codec,
    )
    with pytest.raises(ValueError, match="fixed chemical atoms"):
        system.edit(
            base,
            (Assignment("L", Chemical(boundary_changed)),),
            data=data,
        )

    json_graph = data.put({"atoms": ["C1"]}, JSONCodec())
    json_base = make_candidate(
        system,
        (Assignment("L", Chemical(json_graph)),),
    )
    with pytest.raises(ValueError, match="fixed_atom_records"):
        system.edit(
            json_base,
            (Assignment("L", Chemical(json_graph)),),
            data=data,
        )


def test_dataclass_codec_round_trips_registered_nested_types_without_imports():
    blob = Blob(Key("abc"), "json/v1")
    codec = DataclassCodec(
        Payload,
        "test/payload/v1",
        nested_types=(NestedValue,),
    )
    value = Payload(
        nested=NestedValue("nested", (2, 1)),
        blob=blob,
        mapping={"B": ("G", "C")},
    )

    decoded = codec.decode(codec.encode(value))
    assert decoded == value
    assert codec.references(value) == (blob,)

    with pytest.raises(TypeError, match="expects a Payload dataclass"):
        codec.encode(object())
    store = DataStore()
    with pytest.raises(ValueError, match="identity"):
        store.register(DataclassCodec(Payload, "same/identity"))
        store.register(DataclassCodec(NestedValue, "same/identity"))


def test_design_codec_supports_heterogeneous_design_roots():
    data = DataStore()
    value = Dependency("source", key_for("input"))
    reference = data.put(value, design_codec)
    assert data.get(reference, design_codec) == value

    checkpoint = {
        "history": (),
        "rng": {"before": (17, 23), "after": (31,)},
        "key": key_for("checkpoint"),
    }
    checkpoint_reference = data.put(checkpoint, design_codec)
    assert data.get(checkpoint_reference, design_codec) == checkpoint


@pytest.mark.parametrize("operation", ["encode", "references"])
@pytest.mark.parametrize(
    "bad_value, error, message",
    [
        ((object(),), TypeError, "unsupported"),
        ([np.array([1])], TypeError, "unsupported"),
        ({object()}, TypeError, "unsupported"),
        (frozenset({object()}), TypeError, "unsupported"),
        ({1: "bad key"}, TypeError, "keys"),
        (NestedValue("unregistered", (1,)), TypeError, "not registered"),
        (NestedValue, TypeError, "unsupported"),
        (float("nan"), ValueError, "non-finite"),
        (float("inf"), ValueError, "non-finite"),
        (float("-inf"), ValueError, "non-finite"),
    ],
)
def test_object_root_codec_rejects_malformed_nested_values(
    operation, bad_value, error, message
):
    codec = DataclassCodec(object, "test/structured/v1")
    with pytest.raises(error, match=message):
        getattr(codec, operation)({"outer": [{"inner": bad_value}]})


@pytest.mark.parametrize("operation", ["encode", "references"])
@pytest.mark.parametrize(
    "field, bad_value, error, message",
    [
        ("identity", Key({"nested": [float("nan")]}), ValueError, "non-finite"),
        ("identity", Key([object()]), TypeError, "unsupported"),
        ("identity", Key({1: "bad key"}), TypeError, "keys"),
        ("identity", NestedValue("unregistered", ()), TypeError, "not registered"),
        ("codec", {"nested": [np.array([1])]}, TypeError, "unsupported"),
        ("codec", {"nested": [float("inf")]}, ValueError, "non-finite"),
    ],
)
def test_object_root_codec_validates_nested_blob_fields(
    operation, field, bad_value, error, message
):
    codec = DataclassCodec(object, "test/structured-blob/v1")
    blob = Blob(**{"identity": Key("abc"), "codec": "json/v1", field: bad_value})
    with pytest.raises(error, match=message):
        getattr(codec, operation)({"outer": (blob,)})


@pytest.mark.parametrize("operation", ["encode", "references"])
def test_object_root_codec_checks_blob_subclass_allowlist_and_fields(operation):
    @dataclass(frozen=True)
    class ExtendedBlob(Blob):
        detail: object

    blob = ExtendedBlob(Key("abc"), "json/v1", float("nan"))
    codec = DataclassCodec(object, "test/unregistered-blob/v1")
    with pytest.raises(TypeError, match="not registered"):
        getattr(codec, operation)([blob])

    registered = DataclassCodec(
        object, "test/registered-blob/v1", nested_types=(ExtendedBlob,)
    )
    with pytest.raises(ValueError, match="non-finite"):
        getattr(registered, operation)([blob])


@pytest.mark.parametrize("number", [float("nan"), float("inf"), float("-inf")])
def test_typed_root_references_preserve_historic_nonfinite_float_support(number):
    blob = Blob(Key("abc"), "json/v1")
    value = Payload(NestedValue("nested", ()), blob, {"nested": (number,)})
    typed = DataclassCodec(Payload, "test/typed-float/v1", nested_types=(NestedValue,))
    structured = DataclassCodec(
        object, "test/structured-float/v1", nested_types=(Payload, NestedValue)
    )
    assert typed.references(value) == (blob,)
    with pytest.raises(ValueError, match="non-finite"):
        typed.encode(value)
    with pytest.raises(ValueError, match="non-finite"):
        structured.references(value)


def test_typed_root_references_preserve_blob_leaf_semantics():
    blob = Blob(Key({"nested": [object()]}), "json/v1")
    value = Payload(NestedValue("nested", ()), blob, {})
    typed = DataclassCodec(Payload, "test/typed-blob/v1", nested_types=(NestedValue,))
    assert typed.references(value) == (blob,)
    with pytest.raises(TypeError, match="unsupported"):
        typed.encode(value)


@pytest.mark.parametrize(
    "bad_value, error, message",
    [
        (object(), TypeError, "unsupported"),
        (float("nan"), ValueError, "non-finite"),
        ({1: "bad key"}, TypeError, "keys"),
        (NestedValue("unregistered", ()), TypeError, "not registered"),
        (Blob(Key([object()]), "json/v1"), TypeError, "unsupported"),
    ],
)
def test_object_root_decode_revalidates_values_changed_by_post_init(
    bad_value, error, message
):
    @dataclass
    class PostInitValue:
        value: object

        def __post_init__(self):
            if self.value == "replace":
                self.value = {"nested": [bad_value]}

    codec = DataclassCodec(
        object, "test/post-init/v1", nested_types=(PostInitValue,)
    )
    envelope = json.loads(codec.encode({"wrapped": PostInitValue("safe")}))
    envelope["$dict"]["wrapped"]["fields"]["value"] = "replace"
    with pytest.raises(error, match=message):
        codec.decode(json.dumps(envelope).encode("utf-8"))


def test_object_root_codec_preserves_canonical_bytes_identity_and_reference_order():
    codec = DataclassCodec(object, "test/canonical/v1")
    first = Blob(Key("first"), "json/v1")
    second = Blob(Key("second"), "json/v1")
    value = {
        "z": [second, first, second],
        "a": ("é", b"\x00\xff", Path("data/input"), {2, 1}, frozenset({4, 3}),
              None, True, -0.0, first),
    }
    first_json = (
        '{"$dataclass":"astrolabe.design.types.Blob","fields":{"codec":"json/v1",'
        '"identity":{"$dataclass":"astrolabe.design.types.Key",'
        '"fields":{"digest":"first"}}}}'
    )
    second_json = first_json.replace('"first"', '"second"')
    expected = (
        '{"$dict":{"a":{"$tuple":["é",{"$bytes":"AP8="},'
        '{"$path":"data/input"},{"$set":[1,2]},{"$set":[3,4]},null,true,-0.0,'
        + first_json + ']},"z":{"$list":['
        + ",".join((second_json, first_json, second_json)) + "]}}}"
    ).encode("utf-8")
    reordered = {"a": value["a"], "z": value["z"]}

    assert codec.encode(value) == codec.encode(reordered) == expected
    assert codec.references(value) == (second, first, second, first)
    assert codec.references(reordered) == (first, second, first, second)
    data = DataStore()
    reference = data.put(value, codec)
    assert reference == Blob(Key(hashlib.sha256(expected).hexdigest()), codec.identity)
    assert data.put(reordered, codec) == reference


def test_object_root_codec_rechecks_mutable_state_and_returns_fresh_decoded_values():
    @dataclass
    class MutableValue:
        data: dict[str, object]

    codec = DataclassCodec(object, "test/mutable/v1", nested_types=(MutableValue,))
    first = Blob(Key("first"), "json/v1")
    second = Blob(Key("second"), "json/v1")
    value = MutableValue({"numbers": [1], "blob": first})
    data = DataStore()
    original_bytes = codec.encode(value)
    original = data.put(value, codec)
    assert codec.references(value) == (first,)

    value.data["numbers"].append(2)
    value.data["blob"] = second
    assert codec.encode(value) != original_bytes
    assert codec.references(value) == (second,)
    assert data.put(value, codec) != original

    decoded = data.get(original, codec)
    decoded.data["numbers"].append(99)
    decoded.data["blob"] = second
    assert data.get(original, codec) == MutableValue({"numbers": [1], "blob": first})

    value.data["invalid"] = object()
    for operation in (codec.encode, codec.references):
        with pytest.raises(TypeError, match="unsupported"):
            operation(value)


def test_custom_dataclass_codec_seals_its_explicit_type_registry():
    codec = DataclassCodec(Payload, "test/sealed/v1")
    value = Payload(
        nested=NestedValue("unregistered", (1,)),
        blob=Blob(Key("abc"), "json/v1"),
        mapping={},
    )
    with pytest.raises(TypeError, match="not registered"):
        codec.encode(value)
    with pytest.raises(RuntimeError, match="after codec use"):
        codec.register_types(NestedValue)


def test_array_codec_returns_independent_read_only_arrays():
    codec = ArrayCodec()
    original = np.arange(6, dtype=np.int64).reshape(2, 3)
    decoded = codec.decode(codec.encode(original))

    np.testing.assert_array_equal(decoded, original)
    assert not np.shares_memory(decoded, original)
    with pytest.raises(ValueError, match="read-only"):
        decoded[0, 0] = 99

    with pytest.raises(TypeError, match="object dtype"):
        codec.encode(np.array([object()], dtype=object))


def test_store_persists_recursive_blob_closure_and_reloads(tmp_path: Path):
    data = DataStore(tmp_path / "data")
    array_codec = ArrayCodec()
    payload_codec = DataclassCodec(
        Payload,
        "test/payload/v1",
        nested_types=(NestedValue,),
    )
    array_blob = data.put(np.array([1.5, 2.5]), array_codec)
    payload = Payload(
        nested=NestedValue("persisted", (3, 4)),
        blob=array_blob,
        mapping={"state": ("A", "B")},
    )
    payload_blob = data.put(payload, payload_codec)
    data.persist((payload_blob,))

    reopened = DataStore(tmp_path / "data")
    reopened.register(payload_codec)
    reopened.register(array_codec)
    loaded = reopened.get(payload_blob, payload_codec)
    loaded_array = reopened.get(loaded.blob, array_codec)

    assert loaded == payload
    np.testing.assert_array_equal(loaded_array, np.array([1.5, 2.5]))
    assert not loaded_array.flags.writeable


def test_store_requires_known_codecs_and_rejects_digest_tampering(tmp_path: Path):
    root = tmp_path / "data"
    data = DataStore(root)
    codec = JSONCodec()
    reference = data.put({"value": [1, 2]}, codec)
    data.persist((reference,))

    unknown = DataStore(root)
    with pytest.raises(KeyError, match="unknown codec"):
        unknown.get(reference)

    tampered = DataStore(root)
    tampered.register(codec)
    tampered._path_for(reference).write_bytes(b'{"value":[9]}')
    with pytest.raises(ValueError, match="digest mismatch"):
        tampered.get(reference, codec)


def test_repersist_rechecks_existing_content_and_restores_missing_blob(tmp_path):
    data = DataStore(tmp_path / "data")
    reference = data.put({"value": 1}, JSONCodec())
    data.persist((reference,))
    target = data._path_for(reference)
    original = target.read_bytes()
    target.write_bytes(b'{"value":2}')
    with pytest.raises(ValueError, match="durable content differs"):
        data.persist((reference,))
    target.unlink()
    data.persist((reference,))
    assert target.read_bytes() == original


def test_repersist_existing_closure_retains_directory_sync_failure(tmp_path, monkeypatch):
    import os
    import stat
    import astrolabe.design.store as store_module

    data = DataStore(tmp_path / "data")
    reference = data.put({"value": 1}, JSONCodec())
    data.persist((reference,))
    real_fsync = os.fsync

    def fail_directory(descriptor):
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("directory sync failed")
        real_fsync(descriptor)

    monkeypatch.setattr(store_module.os, "fsync", fail_directory)
    with pytest.raises(OSError, match="directory sync failed"):
        data.persist((reference,))


def test_memory_store_refuses_persist_and_json_rejects_non_json_values():
    data = DataStore()
    reference = data.put({"value": 1}, JSONCodec())
    with pytest.raises(RuntimeError, match="memory-only"):
        data.persist((reference,))
    with pytest.raises(TypeError, match="not JSON-native"):
        JSONCodec().encode({"tuple": (1, 2)})


def test_evidence_with_measurements_appends_raw_values_and_deduplicates():
    candidate = key_for("candidate")
    subject = Subject(candidate, None, None, (), None, None)
    inputs = manifest(())
    first = Measurement(
        identity=key_for("measurement/1"),
        quantity="plddt",
        definition=key_for("definition/plddt"),
        units="fraction",
        producer=key_for("producer/model"),
        subjects=(subject,),
        axes=(),
        values=None,
        missing="unavailable",
        valid_mask=None,
        inputs=inputs,
    )
    second = Measurement(
        identity=key_for("measurement/2"),
        quantity="iptm",
        definition=key_for("definition/iptm"),
        units="fraction",
        producer=key_for("producer/model"),
        subjects=(subject,),
        axes=(),
        values=None,
        missing="unavailable",
        valid_mask=None,
        inputs=inputs,
    )
    evidence = Evidence(
        identity=key_for("evidence"),
        evaluated_candidate=candidate,
        artifacts=(),
        measurements=(first,),
        inputs=inputs,
    )

    merged = evidence.with_measurements((second, first))
    assert merged.measurements == (first, second)

    collision = Measurement(**{**second.__dict__, "units": "percent"})
    with pytest.raises(ValueError, match="identity collision"):
        merged.with_measurements((collision,))

    other_subject = Subject(
        key_for("other-candidate"),
        None,
        None,
        (),
        None,
        None,
    )
    wrong = Measurement(**{**second.__dict__, "subjects": (other_subject,)})
    with pytest.raises(ValueError, match="evaluated candidate"):
        merged.with_measurements((wrong,))
