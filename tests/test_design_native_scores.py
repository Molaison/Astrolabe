from __future__ import annotations

from dataclasses import fields, replace
import math
import os

import numpy as np
import pytest

from astrolabe.design import native_scores
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
        assert type(left) is type(right)
        assert not np.ma.isMaskedArray(left)
        assert not np.ma.isMaskedArray(right)
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


@pytest.mark.parametrize("location", ["pae", "extension", "contribution"])
@pytest.mark.parametrize("mask", [False, [False, True]])
def test_snapshot_rejects_masked_arrays_without_discarding_mask(tmp_path, location, mask):
    scores = StandardScores(names=["A"], plddt={"A": [0.5]}, complex_paths={})
    value = np.ma.array([1.0, 999.0], mask=mask)
    if location == "pae":
        scores.pae = {"A": [value]}
    elif location == "extension":
        scores.extension = {"nested": [value]}
    else:
        scores.score_contributions = [
            ScoreContribution(source="test", metric_name="masked", value=value)
        ]

    with pytest.raises(UnsupportedNativeScoresState, match="MaskedArray"):
        snapshot_scores(scores, DataStore())
    np.testing.assert_equal(value.data, [1.0, 999.0])
    np.testing.assert_equal(np.ma.getmaskarray(value), np.broadcast_to(mask, (2,)))


def _structure_snapshot(tmp_path):
    source = tmp_path / "source.cif"
    source.write_bytes(b"immutable structure\n")
    scores = StandardScores(
        names=["A"], plddt={"A": [0.5]}, complex_paths={"A": source}
    )
    data = DataStore()
    blob = snapshot_scores(scores, data)
    return scores, data, blob, data.get(blob, NATIVE_SCORES_ENVELOPE_CODEC)


@pytest.mark.parametrize(
    "name",
    ["../escaped.cif", "absolute", "", ".", "..", "sub/file.cif",
     "..\\escaped.cif", "C:\\escaped.cif", "bad\x00.cif"],
)
def test_restore_rejects_unsafe_names_before_writing_any_structures(tmp_path, name):
    _, data, _, envelope = _structure_snapshot(tmp_path)
    if name == "absolute":
        name = str(tmp_path / "escaped.cif")
    bad = replace(
        envelope,
        structures=envelope.structures
        + (replace(envelope.structures[0], assembly="B", restored_name=name),),
    )
    blob = data.put(bad, NATIVE_SCORES_ENVELOPE_CODEC)
    destination = tmp_path / "restored"

    with pytest.raises(ValueError, match="structure.*name"):
        restore_scores(blob, data, destination)
    assert not destination.exists()
    assert not (tmp_path / "escaped.cif").exists()


def test_restore_rejects_duplicate_structure_names_before_writing(tmp_path):
    _, data, _, envelope = _structure_snapshot(tmp_path)
    bad = replace(
        envelope,
        structures=envelope.structures
        + (replace(envelope.structures[0], assembly="B"),),
    )
    blob = data.put(bad, NATIVE_SCORES_ENVELOPE_CODEC)
    destination = tmp_path / "restored"

    with pytest.raises(ValueError, match="duplicate.*structure.*name"):
        restore_scores(blob, data, destination)
    assert not destination.exists()


@pytest.mark.parametrize("location", ["destination", "ancestor", "nested"])
def test_restore_rejects_symlink_directories(tmp_path, location):
    scores, data, blob, _ = _structure_snapshot(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    destination = tmp_path / "restored"
    if location == "destination":
        destination.symlink_to(outside, target_is_directory=True)
    elif location == "ancestor":
        parent = tmp_path / "linked-parent"
        parent.symlink_to(outside, target_is_directory=True)
        destination = parent / "restored"
    else:
        outer = StandardScores(names=[], plddt={}, complex_paths={})
        outer.candidate_scores = [scores]
        blob = snapshot_scores(outer, data)
        destination.mkdir()
        (destination / "_native_scores").symlink_to(outside, target_is_directory=True)

    with pytest.raises(OSError, match="symlink"):
        restore_scores(blob, data, destination)
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("existing", ["identical", "different", "dangling"])
def test_restore_rejects_final_file_symlinks(tmp_path, existing):
    _, data, blob, envelope = _structure_snapshot(tmp_path)
    outside = tmp_path / "outside.cif"
    if existing != "dangling":
        outside.write_bytes(
            b"immutable structure\n" if existing == "identical" else b"other content"
        )
    destination = tmp_path / "restored"
    destination.mkdir()
    target = destination / envelope.structures[0].restored_name
    target.symlink_to(outside)

    with pytest.raises(OSError, match="symlink"):
        restore_scores(blob, data, destination)
    assert target.is_symlink()
    if existing == "dangling":
        assert not outside.exists()
    else:
        assert outside.read_bytes() == (
            b"immutable structure\n" if existing == "identical" else b"other content"
        )


def test_restore_rejects_directory_replaced_with_symlink_during_creation(
    tmp_path, monkeypatch
):
    _, data, blob, _ = _structure_snapshot(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    destination = tmp_path / "restored"
    original_open = native_scores._open_directory_component
    swapped = False

    def swap_before_open(name, parent):
        nonlocal swapped
        if name == destination.name and destination.is_dir() and not swapped:
            destination.rmdir()
            destination.symlink_to(outside, target_is_directory=True)
            swapped = True
        return original_open(name, parent)

    monkeypatch.setattr(native_scores, "_open_directory_component", swap_before_open)
    with pytest.raises(OSError, match="symlink"):
        restore_scores(blob, data, destination)
    assert swapped
    assert list(outside.iterdir()) == []


def test_restore_rejects_final_symlink_installed_just_before_write(tmp_path, monkeypatch):
    _, data, blob, _ = _structure_snapshot(tmp_path)
    outside = tmp_path / "outside.cif"
    outside.write_bytes(b"immutable structure\n")
    original_write = native_scores._write_exact

    def link_before_write(path, content, directory_fd):
        path.symlink_to(outside)
        return original_write(path, content, directory_fd)

    monkeypatch.setattr(native_scores, "_write_exact", link_before_write)
    with pytest.raises(OSError, match="symlink"):
        restore_scores(blob, data, tmp_path / "restored")
    assert outside.read_bytes() == b"immutable structure\n"


@pytest.mark.parametrize("kind", ["directory", "fifo"])
def test_restore_rejects_nonregular_final_targets(tmp_path, kind):
    _, data, blob, envelope = _structure_snapshot(tmp_path)
    destination = tmp_path / "restored"
    destination.mkdir()
    target = destination / envelope.structures[0].restored_name
    if kind == "directory":
        target.mkdir()
    else:
        os.mkfifo(target)

    with pytest.raises(FileExistsError, match="different data"):
        restore_scores(blob, data, destination)


def test_restore_to_relative_directory_preserves_contents(tmp_path, monkeypatch):
    _, data, blob, _ = _structure_snapshot(tmp_path)
    monkeypatch.chdir(tmp_path)

    restored = restore_scores(blob, data, "restored/subdirectory")

    target = restored.complex_paths["A"]
    assert not target.is_absolute()
    assert target.read_bytes() == b"immutable structure\n"


def test_resnapshot_preserves_existing_path_provenance_semantics(tmp_path):
    _, data, blob, envelope = _structure_snapshot(tmp_path)
    restored = restore_scores(blob, data, tmp_path / "restored")

    resnapshot = snapshot_scores(restored, data)
    restored_envelope = data.get(resnapshot, NATIVE_SCORES_ENVELOPE_CODEC)

    # The envelope is path-sensitive, not a stable scientific identity across
    # materialization. Keep this boundary explicit rather than rewriting history.
    assert resnapshot != blob
    before = envelope.structures[0]
    after = restored_envelope.structures[0]
    assert after.content == before.content
    assert after.original_path == restored.complex_paths["A"].as_posix()
    assert after.restored_name == "A-" + before.restored_name


def test_cold_persist_has_complete_array_and_nested_score_reference_closure(tmp_path):
    source = tmp_path / "source.cif"
    source.write_bytes(b"immutable structure\n")
    child = StandardScores(
        names=["A"],
        plddt={"A": [np.asarray([0.5, math.nan], dtype=np.float32)]},
        complex_paths={"A": source},
    )
    scores = StandardScores(names=[], plddt={}, complex_paths={})
    scores.candidate_scores = [child]
    scores.extra = {"array": np.asarray([math.inf, -math.inf], dtype=np.float64)}
    root = tmp_path / "store"
    data = DataStore(root)
    blob = snapshot_scores(scores, data)
    data.persist((blob,))
    source.unlink()
    cold = DataStore(root)
    for codec in NATIVE_SCORES_REFERENCE_CODECS:
        cold.register(codec)

    # Persist BEFORE restore: restore must not accidentally supply missing codecs.
    cold.persist((blob,))
    restored = restore_scores(blob, cold, tmp_path / "restored")
    _assert_scores_equal(scores, restored)
    assert restored.candidate_scores[0].complex_paths["A"].read_bytes() == (
        b"immutable structure\n"
    )


@pytest.mark.parametrize("location", ["destination", "ancestor", "file"])
@pytest.mark.parametrize("replacement", ["symlink", "different", "identical"])
@pytest.mark.parametrize("existing", [False, True])
def test_restore_rejects_paths_replaced_after_verified_io(
    tmp_path, monkeypatch, location, replacement, existing
):
    _, data, blob, envelope = _structure_snapshot(tmp_path)
    parent = tmp_path / "parent"
    destination = parent / "restored"
    basename = envelope.structures[0].restored_name
    if existing:
        restore_scores(blob, data, destination)
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_content = b"external wrong content\n"
    external_file = outside / basename
    external_file.write_bytes(outside_content)
    (outside / "restored").mkdir()
    (outside / "restored" / basename).write_bytes(outside_content)
    original_write = native_scores._write_exact
    swapped = False

    def swap_after_io(path, content, directory_fd):
        nonlocal swapped
        result = original_write(path, content, directory_fd)
        if not swapped:
            target = {"destination": destination, "ancestor": parent, "file": path}[
                location
            ]
            target.rename(tmp_path / "saved")
            if replacement == "symlink":
                target.symlink_to(
                    external_file if location == "file" else outside,
                    target_is_directory=location != "file",
                )
            else:
                replacement_content = (
                    content if replacement == "identical" else outside_content
                )
                if location == "file":
                    target.write_bytes(replacement_content)
                else:
                    replacement_dir = (
                        target / "restored" if location == "ancestor" else target
                    )
                    replacement_dir.mkdir(parents=True)
                    (replacement_dir / basename).write_bytes(replacement_content)
            swapped = True
        return result

    monkeypatch.setattr(native_scores, "_write_exact", swap_after_io)
    with pytest.raises(OSError, match="changed|symlink"):
        restore_scores(blob, data, destination)
    assert swapped
    assert external_file.read_bytes() == outside_content
    assert (outside / "restored" / basename).read_bytes() == outside_content


def test_restore_rechecks_nested_paths_before_outer_return(tmp_path, monkeypatch):
    child, data, _, envelope = _structure_snapshot(tmp_path)
    outer = StandardScores(names=[], plddt={}, complex_paths={})
    outer.candidate_scores = [child]
    outer.z_last_field = "decode after nested restore"
    blob = snapshot_scores(outer, data)
    outer_envelope = data.get(blob, NATIVE_SCORES_ENVELOPE_CODEC)
    last_field = outer_envelope.fields[-1].value
    original_get = data.get
    destination = tmp_path / "restored"
    swapped = False

    def swap_on_later_field(reference, codec=None):
        nonlocal swapped
        value = original_get(reference, codec)
        if reference == last_field and not swapped:
            target, = destination.glob("_native_scores/*/" + envelope.structures[0].restored_name)
            target.rename(tmp_path / "saved.cif")
            target.write_bytes(b"late replacement\n")
            swapped = True
        return value

    monkeypatch.setattr(data, "get", swap_on_later_field)
    with pytest.raises(OSError, match="changed"):
        restore_scores(blob, data, destination)
    assert swapped


@pytest.mark.parametrize("replacement", ["regular", "symlink", "directory", "none"])
def test_fsync_failure_does_not_unlink_replacement_or_partial_files(
    tmp_path, monkeypatch, replacement
):
    _, data, blob, envelope = _structure_snapshot(tmp_path)
    destination = tmp_path / "restored"
    target = destination / envelope.structures[0].restored_name
    outside = tmp_path / "outside.cif"
    outside.write_bytes(b"external content\n")
    replacement_identity = None

    def fail_fsync(descriptor):
        nonlocal replacement_identity
        if replacement != "none":
            target.rename(tmp_path / "our-created-file.cif")
            if replacement == "regular":
                target.write_bytes(b"other owner's file\n")
            elif replacement == "symlink":
                target.symlink_to(outside)
            else:
                target.mkdir()
                (target / "keep").write_bytes(b"other owner's directory\n")
        state = target.lstat()
        replacement_identity = (state.st_dev, state.st_ino)
        raise OSError("injected fsync failure")

    monkeypatch.setattr(native_scores.os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="injected fsync failure"):
        restore_scores(blob, data, destination)
    state = target.lstat()
    assert (state.st_dev, state.st_ino) == replacement_identity
    assert outside.read_bytes() == b"external content\n"
    if replacement == "regular":
        assert target.read_bytes() == b"other owner's file\n"
    elif replacement == "symlink":
        assert target.is_symlink()
    elif replacement == "directory":
        assert (target / "keep").read_bytes() == b"other owner's directory\n"
    else:
        assert target.read_bytes() == b"immutable structure\n"


@pytest.mark.parametrize("location", ["directory", "file"])
def test_restore_rejects_replacement_while_file_is_still_open(
    tmp_path, monkeypatch, location
):
    _, data, blob, envelope = _structure_snapshot(tmp_path)
    destination = tmp_path / "restored"
    target = destination / envelope.structures[0].restored_name
    outside = tmp_path / "outside"
    outside.mkdir()
    external_file = outside / target.name
    external_file.write_bytes(b"external wrong content\n")
    original_fsync = os.fsync
    swapped = False

    def replace_during_fsync(descriptor):
        nonlocal swapped
        original_fsync(descriptor)
        if not swapped:
            victim = destination if location == "directory" else target
            victim.rename(tmp_path / "saved")
            victim.symlink_to(
                outside if location == "directory" else external_file,
                target_is_directory=location == "directory",
            )
            # The still-open descriptor contains our bytes, whereas the public
            # path already resolves to different external data.
            saved = tmp_path / "saved"
            saved_file = saved / target.name if location == "directory" else saved
            pinned = os.fstat(descriptor)
            moved = saved_file.stat()
            assert (pinned.st_dev, pinned.st_ino) == (moved.st_dev, moved.st_ino)
            assert saved_file.read_bytes() == b"immutable structure\n"
            assert target.read_bytes() == b"external wrong content\n"
            swapped = True

    monkeypatch.setattr(native_scores.os, "fsync", replace_during_fsync)
    with pytest.raises(OSError, match="changed"):
        restore_scores(blob, data, destination)
    assert swapped
    assert external_file.read_bytes() == b"external wrong content\n"


@pytest.mark.parametrize("existing", [False, True])
def test_restore_rechecks_contents_even_if_inode_is_unchanged(
    tmp_path, monkeypatch, existing
):
    _, data, blob, envelope = _structure_snapshot(tmp_path)
    destination = tmp_path / "restored"
    if existing:
        restore_scores(blob, data, destination)
    original_write = native_scores._write_exact

    def mutate_after_io(path, content, directory_fd):
        result = original_write(path, content, directory_fd)
        before = path.stat()
        path.write_bytes(b"changed in place\n")
        after = path.stat()
        assert (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino)
        return result

    monkeypatch.setattr(native_scores, "_write_exact", mutate_after_io)
    with pytest.raises(OSError, match="content changed"):
        restore_scores(blob, data, destination)


@pytest.mark.parametrize("bits", [0x7FF8000000000042, 0xFFF8000000000001])
def test_scalar_nan_round_trip_preserves_semantics_not_payload_contract(tmp_path, bits):
    scores = StandardScores(names=[], plddt={}, complex_paths={})
    nan_array = np.asarray([bits], dtype=np.uint64).view(np.float64)
    scores.numpy_nan = nan_array[0]
    scores.python_nan = float(nan_array[0])
    scores.nan_array = nan_array
    data = DataStore()
    blob = snapshot_scores(scores, data)

    restored = restore_scores(blob, data, tmp_path / "restored")

    assert type(restored.numpy_nan) is np.float64
    assert type(restored.python_nan) is float
    assert math.isnan(restored.numpy_nan) and math.isnan(restored.python_nan)
    # Array bytes have a different codec and retain these payloads; no scalar
    # sign/payload assertion is part of the scalar compatibility contract.
    np.testing.assert_equal(restored.nan_array.view(np.uint64), [bits])
