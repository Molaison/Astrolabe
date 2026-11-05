from __future__ import annotations

from dataclasses import fields, replace
import math

import numpy as np
import pytest

from astrolabe.design.native_scores import (
    NATIVE_SCORES_ENVELOPE_CODEC,
    NATIVE_SCORES_FIELD_CODEC,
    NATIVE_SCORES_REFERENCE_CODECS,
    NativeScoreField,
    NativeScoresEnvelope,
    UnsupportedNativeScoresState,
    restore_scores,
    snapshot_scores,
)
from astrolabe.design.store import DataStore
from astrolabe.evaluators.utils import StandardScores
from astrolabe.score_contributions import ScoreContribution


def _equal_native(left, right) -> None:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        left_array = np.asarray(left)
        right_array = np.asarray(right)
        assert left_array.dtype == right_array.dtype
        assert left_array.shape == right_array.shape
        np.testing.assert_equal(left_array, right_array)
        return
    if isinstance(left, np.generic) or isinstance(right, np.generic):
        assert type(left) is type(right)
        assert left.dtype == right.dtype
        np.testing.assert_equal(left, right)
        return
    if isinstance(left, StandardScores):
        assert type(left) is type(right)
        _assert_scores_equal(left, right)
        return
    if isinstance(left, dict):
        assert type(left) is type(right)
        assert list(left) == list(right)
        for key in left:
            _equal_native(left[key], right[key])
        return
    if isinstance(left, (list, tuple)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _equal_native(left_item, right_item)
        return
    if isinstance(left, ScoreContribution):
        assert type(left) is type(right)
        for name in left.__dataclass_fields__:
            _equal_native(getattr(left, name), getattr(right, name))
        return
    if isinstance(left, float) and math.isnan(left):
        assert isinstance(right, float) and math.isnan(right)
        return
    assert type(left) is type(right)
    assert left == right


def _assert_scores_equal(left: StandardScores, right: StandardScores) -> None:
    assert list(left.__dataclass_fields__) == list(right.__dataclass_fields__)
    for field in left.__dataclass_fields__:
        if field == "complex_paths":
            assert list(left.complex_paths) == list(right.complex_paths)
            continue
        _equal_native(getattr(left, field), getattr(right, field))
    declared = set(left.__dataclass_fields__)
    left_extensions = sorted(set(vars(left)) - declared)
    right_extensions = sorted(set(vars(right)) - declared)
    assert left_extensions == right_extensions
    for name in left_extensions:
        _equal_native(getattr(left, name), getattr(right, name))


def test_snapshot_round_trip_preserves_fields_arrays_groups_and_cli_provenance(
    tmp_path,
):
    sources = tmp_path / "sources"
    sources.mkdir()
    first_path = sources / "first prediction.cif"
    second_path = sources / "second.pdb"
    first_bytes = b"data_first\n# exact structure bytes\n"
    second_bytes = b"ATOM      1  N   ALA A   1\nEND\n"
    first_path.write_bytes(first_bytes)
    second_path.write_bytes(second_bytes)

    scores = StandardScores(
        names=["A", "AB"],
        plddt={
            "A": [np.asarray([0.4, 0.6], dtype=np.float32)],
            "AB": [np.asarray([0.3, 0.7, 0.8], dtype=np.float64)],
        },
        complex_paths={"A": first_path, "AB": second_path},
        per_proto_scores={
            "A": [np.asarray([0.1, 1.2], dtype=np.float64)]
        },
        per_proto_score_terms={
            "A": {
                "plddt": [np.asarray([0.4, 0.6], dtype=np.float32)],
                "sap": [np.asarray([1, 2], dtype=np.int16)],
            }
        },
        ptm={"A": [0.75]},
        pae={"A": [np.asarray([[1, 2], [3, 4]], dtype=np.float32)]},
        pde={"A": [np.asarray([[0.5, 0.25]], dtype=np.float64)]},
        iptm={"AB": [0.55]},
        chain_pair_scores={
            "AB": {
                "chain_pair_iptm": [
                    np.asarray([[0.8, 0.2], [0.1, 0.9]], dtype=np.float32)
                ]
            }
        },
        complex_poses={},
        score_contributions=[
            ScoreContribution(
                source="test",
                metric_name="array_metric",
                value=np.asarray([1.5, 2.5], dtype=np.float32),
                scope="oligo",
                entity="A",
                shape="vector",
                provenance="native contribution",
            ),
            ScoreContribution(
                source="test",
                metric_name="nan_metric",
                value=math.nan,
            ),
        ],
    )
    # These values are deliberately changed AFTER native __post_init__.  Restore
    # must not interpret 25/75 as percentages and divide them again.
    scores.plddt["A"].append(25.0)
    scores.per_proto_scores["A"][0][1] = 75.0

    store = tmp_path / "store"
    data = DataStore(store)
    blob = snapshot_scores(scores, data)
    data.persist((blob,))

    original = data.get(blob, NATIVE_SCORES_ENVELOPE_CODEC)
    assert isinstance(original, NativeScoresEnvelope)
    original_paths = {
        structure.assembly: structure.original_path
        for structure in original.structures
    }
    assert original_paths == {
        "A": first_path.as_posix(),
        "AB": second_path.as_posix(),
    }

    # Prove restored content comes from the envelope, not mutable source files.
    first_path.write_bytes(b"changed after snapshot")
    second_path.unlink()
    cold = DataStore(store)
    for codec in NATIVE_SCORES_REFERENCE_CODECS:
        cold.register(codec)
    destination = tmp_path / "restored"
    restored = restore_scores(blob, cold, destination)

    assert restored.plddt["A"][1] == 25.0
    assert restored.per_proto_scores["A"][0][1] == 75.0
    assert restored.complex_paths["A"].parent == destination
    assert restored.complex_paths["A"].read_bytes() == first_bytes
    assert restored.complex_paths["AB"].read_bytes() == second_bytes
    assert (restored.complex_paths["A"].stat().st_mode & 0o777) == 0o600
    _assert_scores_equal(scores, restored)


def test_dynamic_extensions_and_nested_scores_survive_cold_restore(tmp_path):
    sources = tmp_path / "extension-sources"
    sources.mkdir()
    outer_path = sources / "outer.cif"
    first_candidate_path = sources / "candidate-one.cif"
    second_candidate_path = sources / "candidate-two.pdb"
    outer_bytes = b"outer structure bytes\n"
    first_candidate_bytes = b"first candidate structure bytes\n"
    second_candidate_bytes = b"ATOM      1  N   ALA A   1\nEND\n"
    outer_path.write_bytes(outer_bytes)
    first_candidate_path.write_bytes(first_candidate_bytes)
    second_candidate_path.write_bytes(second_candidate_bytes)

    first_candidate = StandardScores(
        names=["A"],
        plddt={
            "A": [
                np.asarray([0.25, 0.5], dtype=np.float32),
                np.asarray([0.75], dtype=np.float64),
            ]
        },
        complex_paths={"A": first_candidate_path},
        ptm={"A": [0.8]},
    )
    first_candidate.plddt["A"].append(np.asarray([25.0], dtype=np.float64))
    second_candidate = StandardScores(
        names=["AB"],
        plddt={
            "AB": [
                np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
            ]
        },
        complex_paths={"AB": second_candidate_path},
        iptm={"AB": [0.6]},
    )
    scores = StandardScores(
        names=["A", "AB"],
        plddt={
            "A": [np.asarray([0.4, 0.6], dtype=np.float32)],
            "AB": [np.asarray([0.5, 0.7], dtype=np.float64)],
        },
        complex_paths={"A": outer_path, "AB": second_candidate_path},
    )
    scores.pre_eval_hook_reports = {
        "A": [
            {
                "round": 1,
                "accepted": True,
                "arrays": [np.asarray([1, 2], dtype=np.int16)],
                "metadata": ("draft", {"finite": True}),
            }
        ]
    }
    scores.candidate_scores = [first_candidate, second_candidate]
    scores.post_eval_hook_reports = {
        "AB": {"ran": True, "energy": -12.5, "missing": None}
    }
    scores.rosetta_metrics = {"AB": {"score": np.float32(1.25)}}
    scores._interface_holes_details = {"A": {"count": 2, "raw": [1.0, 2.0]}}
    scores.report_path = tmp_path / "reports" / "native.json"

    store = tmp_path / "extension-store"
    data = DataStore(store)
    blob = snapshot_scores(scores, data)
    envelope = data.get(blob, NATIVE_SCORES_ENVELOPE_CODEC)
    declared = tuple(field.name for field in fields(StandardScores))
    extension_names = tuple(row.name for row in envelope.fields[len(declared) :])
    assert tuple(row.name for row in envelope.fields[: len(declared)]) == declared
    assert extension_names == tuple(sorted(extension_names))
    assert set(extension_names) == {
        "_interface_holes_details",
        "candidate_scores",
        "post_eval_hook_reports",
        "pre_eval_hook_reports",
        "report_path",
        "rosetta_metrics",
    }
    assert NATIVE_SCORES_ENVELOPE_CODEC.identity.endswith("/v2")
    assert NATIVE_SCORES_FIELD_CODEC.identity.endswith("/v2")
    data.persist((blob,))

    outer_path.write_bytes(b"changed outer")
    first_candidate_path.unlink()
    second_candidate_path.write_bytes(b"changed candidate")
    cold = DataStore(store)
    for codec in NATIVE_SCORES_REFERENCE_CODECS:
        cold.register(codec)
    restored = restore_scores(blob, cold, tmp_path / "extension-restored")

    _assert_scores_equal(scores, restored)
    assert restored.pre_eval_hook_reports["A"][0]["accepted"] is True
    assert type(restored.report_path) is type(scores.report_path)
    assert restored.report_path == scores.report_path
    assert restored.complex_paths["A"].read_bytes() == outer_bytes
    assert restored.complex_paths["AB"].read_bytes() == second_candidate_bytes
    assert restored.candidate_scores[0].plddt["A"][-1].tolist() == [25.0]
    assert (
        restored.candidate_scores[0].complex_paths["A"].read_bytes()
        == first_candidate_bytes
    )
    assert (
        restored.candidate_scores[1].complex_paths["AB"].read_bytes()
        == second_candidate_bytes
    )
    assert restored.candidate_scores[0].complex_paths["A"].parent.parent.name == (
        "_native_scores"
    )


def test_unsupported_dynamic_extension_is_not_silently_dropped(tmp_path):
    scores = StandardScores(
        names=["A"],
        plddt={"A": [np.asarray([0.5])]},
        complex_paths={},
    )
    scores.unsupported = object()

    with pytest.raises(
        UnsupportedNativeScoresState,
        match="builtins.object",
    ):
        snapshot_scores(scores, DataStore())


def test_extension_cycle_fails_explicitly(tmp_path):
    scores = StandardScores(
        names=["A"],
        plddt={"A": [np.asarray([0.5])]},
        complex_paths={},
    )
    cycle = []
    cycle.append(cycle)
    scores.extension_cycle = cycle

    with pytest.raises(UnsupportedNativeScoresState, match="reference cycle"):
        snapshot_scores(scores, DataStore())


def test_restore_rejects_missing_colliding_and_dunder_extension_rows(tmp_path):
    scores = StandardScores(
        names=["A"],
        plddt={"A": [np.asarray([0.5])]},
        complex_paths={},
    )
    data = DataStore()
    blob = snapshot_scores(scores, data)
    envelope = data.get(blob, NATIVE_SCORES_ENVELOPE_CODEC)

    missing = replace(envelope, fields=envelope.fields[:-1])
    missing_blob = data.put(missing, NATIVE_SCORES_ENVELOPE_CODEC)
    with pytest.raises(UnsupportedNativeScoresState, match="field names/order"):
        restore_scores(missing_blob, data, tmp_path / "missing")

    colliding = replace(
        envelope,
        fields=envelope.fields
        + (
            NativeScoreField(
                name="names",
                value=envelope.fields[0].value,
            ),
        ),
    )
    colliding_blob = data.put(colliding, NATIVE_SCORES_ENVELOPE_CODEC)
    with pytest.raises(UnsupportedNativeScoresState, match="colliding"):
        restore_scores(colliding_blob, data, tmp_path / "colliding")

    reserved = replace(
        envelope,
        fields=envelope.fields
        + (
            NativeScoreField(
                name="__unsafe__",
                value=envelope.fields[0].value,
            ),
        ),
    )
    reserved_blob = data.put(reserved, NATIVE_SCORES_ENVELOPE_CODEC)
    with pytest.raises(UnsupportedNativeScoresState, match="reserved"):
        restore_scores(reserved_blob, data, tmp_path / "reserved")


def test_round_trip_preserves_none_nesting_and_empty_mappings(tmp_path):
    scores = StandardScores(
        names=[],
        plddt={},
        complex_paths={},
        per_proto_scores=None,
        per_proto_score_terms=None,
        ptm=None,
        pae=None,
        pde=None,
        iptm=None,
        chain_pair_scores=None,
        complex_poses=None,
        score_contributions=None,
    )
    data = DataStore()
    blob = snapshot_scores(scores, data)
    restored = restore_scores(blob, data, tmp_path / "empty")

    assert restored.names == []
    assert restored.plddt == {}
    assert restored.complex_paths == {}
    assert restored.per_proto_scores is None
    assert restored.per_proto_score_terms is None
    assert restored.ptm is None
    assert restored.pae is None
    assert restored.pde is None
    assert restored.iptm is None
    assert restored.chain_pair_scores is None
    assert restored.complex_poses is None
    assert restored.score_contributions is None


def test_live_complex_poses_fail_explicitly(tmp_path):
    scores = StandardScores(
        names=["A"],
        plddt={"A": [np.asarray([0.5])]},
        complex_paths={},
        complex_poses={"A": object()},
    )

    with pytest.raises(
        UnsupportedNativeScoresState,
        match="real pose codec",
    ):
        snapshot_scores(scores, DataStore())


def test_missing_source_structure_fails_instead_of_fabricating_content(tmp_path):
    scores = StandardScores(
        names=["A"],
        plddt={"A": [np.asarray([0.5])]},
        complex_paths={"A": tmp_path / "missing.cif"},
    )

    with pytest.raises(UnsupportedNativeScoresState, match="is unavailable"):
        snapshot_scores(scores, DataStore())


def test_restore_refuses_to_overwrite_different_destination_bytes(tmp_path):
    source = tmp_path / "source.cif"
    source.write_bytes(b"immutable source")
    scores = StandardScores(
        names=["A"],
        plddt={"A": [np.asarray([0.5])]},
        complex_paths={"A": source},
    )
    data = DataStore()
    blob = snapshot_scores(scores, data)
    envelope = data.get(blob, NATIVE_SCORES_ENVELOPE_CODEC)
    target = tmp_path / "restored" / envelope.structures[0].restored_name
    target.parent.mkdir()
    target.write_bytes(b"different existing destination")

    with pytest.raises(FileExistsError, match="different data"):
        restore_scores(blob, data, target.parent)
    assert target.read_bytes() == b"different existing destination"


def test_restore_is_idempotent_for_identical_destination_bytes(tmp_path):
    source = tmp_path / "source.cif"
    source.write_bytes(b"same immutable bytes")
    scores = StandardScores(
        names=["A"],
        plddt={"A": [np.asarray([0.5])]},
        complex_paths={"A": source},
    )
    data = DataStore()
    blob = snapshot_scores(scores, data)
    destination = tmp_path / "restored"

    first = restore_scores(blob, data, destination)
    second = restore_scores(blob, data, destination)

    assert first.complex_paths == second.complex_paths
    assert second.complex_paths["A"].read_bytes() == b"same immutable bytes"
