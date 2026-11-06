from __future__ import annotations

import hashlib
import random

import numpy as np
import pytest

from astrolabe.design.process_resources import (
    ProcessReference,
    ProcessResourceCodec,
    ProcessResources,
)
from astrolabe.design.store import DataStore
from astrolabe.design.types import Blob, Key
from astrolabe.evaluators.utils import StandardScores


class FakePose:
    def __init__(self, name: str) -> None:
        self.name = name
        self.serialization_calls = 0

    def __reduce__(self):
        self.serialization_calls += 1
        raise AssertionError("process resources must retain Python identity")

    def __getstate__(self):
        self.serialization_calls += 1
        raise AssertionError("process resources must not serialize pose state")

    def to_pdb(self) -> str:
        self.serialization_calls += 1
        raise AssertionError("process resources must not serialize pose output")


class _ContainerCodec:
    identity = "test.process-resource-container/v1"

    def __init__(self, child: Blob) -> None:
        self._child = child

    def encode(self, value: bytes) -> bytes:
        return value

    def decode(self, data: bytes) -> bytes:
        return data

    def references(self, value: bytes) -> tuple[Blob, ...]:
        return (self._child,)

    def compatibility_key(self) -> tuple[str, str]:
        return (type(self).__name__, self.identity)


def _assert_numpy_generator_state_equal(left, right) -> None:
    assert left[:1] == right[:1]
    np.testing.assert_array_equal(left[1], right[1])
    assert left[2:] == right[2:]


def test_requires_memory_store_and_exposes_non_persistent_codec() -> None:
    owner = ProcessResources(DataStore())

    assert type(owner.codec) is ProcessResourceCodec
    assert owner.codec.persistent is False
    assert owner.codec.identity == (
        f"astrolabe.process-resource/v1/{owner.nonce}"
    )
    assert owner.codec.references(
        ProcessReference(owner.nonce, 0, "unused")
    ) == ()

    with pytest.raises(TypeError, match="DataStore"):
        ProcessResources(object())


def test_requires_memory_only_store(tmp_path) -> None:
    root = tmp_path / "durable"
    with pytest.raises(ValueError, match="in-memory"):
        ProcessResources(DataStore(root))
    assert not root.exists()


def test_retains_arbitrary_object_by_exact_identity_without_serializing() -> None:
    data = DataStore()
    owner = ProcessResources(data)
    pose = FakePose("native")

    blob = owner.retain(pose, label="pose")
    reference = data.get(blob, owner.codec)

    assert reference == ProcessReference(owner.nonce, 0, "pose")
    assert owner.resolve(blob) is pose
    assert pose.serialization_calls == 0


def test_slots_are_session_local_ordinals() -> None:
    owner = ProcessResources(DataStore())
    first = owner.retain(object(), label="first")
    second = owner.retain(object(), label="second")

    first_reference = owner.data.get(first, owner.codec)
    second_reference = owner.data.get(second, owner.codec)

    assert first_reference == ProcessReference(owner.nonce, 0, "first")
    assert second_reference == ProcessReference(owner.nonce, 1, "second")


def test_candidate_and_revision_filters_are_exact() -> None:
    owner = ProcessResources(DataStore())
    candidate = Key("candidate-a")
    value = object()
    blob = owner.retain(
        value,
        label="scored pose",
        candidate=candidate,
        revision=3,
    )

    assert owner.resolve(blob) is value
    assert owner.resolve(blob, candidate=candidate, revision=3) is value
    with pytest.raises(ValueError, match="candidate"):
        owner.resolve(blob, candidate=Key("candidate-b"))
    with pytest.raises(ValueError, match="revision"):
        owner.resolve(blob, revision=4)
    with pytest.raises(TypeError, match="revision"):
        owner.resolve(blob, revision=True)


def test_validate_scores_checks_raw_identity_candidate_and_revision() -> None:
    owner = ProcessResources(DataStore())
    scores = StandardScores(
        names=["A"],
        plddt={"A": [0.8]},
        complex_paths={},
    )
    equal_but_distinct = StandardScores(
        names=["A"],
        plddt={"A": [0.8]},
        complex_paths={},
    )
    candidate = Key("candidate-a")
    blob = owner.retain(
        scores,
        label="native-scores",
        candidate=candidate,
        revision=7,
    )

    assert owner.validate_scores(blob, scores, candidate) == (blob, 7)
    assert owner.codec.validate_scores(blob, scores, candidate) == (blob, 7)
    assert (
        owner.validate_scores(blob, scores, candidate, revision=7)
        == (blob, 7)
    )
    with pytest.raises(ValueError, match="raw object"):
        owner.validate_scores(blob, equal_but_distinct, candidate)
    with pytest.raises(ValueError, match="candidate"):
        owner.validate_scores(blob, scores, Key("candidate-b"))
    with pytest.raises(ValueError, match="revision"):
        owner.validate_scores(blob, scores, candidate, revision=8)
    with pytest.raises(TypeError, match="candidate must be a Key"):
        owner.validate_scores(blob, scores, None)


def test_validate_scores_rejects_non_native_score_resources() -> None:
    owner = ProcessResources(DataStore())
    candidate = Key("candidate-a")
    result = {"operation": "complete"}
    blob = owner.retain(
        result,
        label="operation-result",
        candidate=candidate,
    )

    with pytest.raises(ValueError, match="native-scores"):
        owner.validate_scores(blob, result, candidate)


def test_expire_scores_rejects_raw_access_but_preserves_handle_metadata() -> None:
    data = DataStore()
    owner = ProcessResources(data)
    scores = StandardScores(
        names=["A"],
        plddt={"A": [0.8]},
        complex_paths={},
    )
    candidate = Key("candidate-a")
    scores_blob = owner.retain(
        scores,
        label="native-scores",
        candidate=candidate,
        revision=7,
    )
    pose = FakePose("native")
    pose_blob = owner.retain(pose, label="pose")

    owner.expire_scores()
    owner.expire_scores()

    with pytest.raises(RuntimeError, match="native-scores.*expired"):
        owner.resolve(scores_blob)
    with pytest.raises(RuntimeError, match="native-scores.*expired"):
        owner.validate_scores(scores_blob, scores, candidate)

    reference = data.get(scores_blob, owner.codec)
    assert reference == ProcessReference(owner.nonce, 0, "native-scores")
    assert owner.codec.references(reference) == ()
    assert owner.resolve(pose_blob) is pose


def test_cross_session_and_forged_slot_references_are_rejected() -> None:
    first = ProcessResources(DataStore())
    second = ProcessResources(DataStore())
    blob = first.retain(object(), label="session one")

    assert first.nonce != second.nonce
    assert first.codec.identity != second.codec.identity
    with pytest.raises(ValueError, match="exact session codec"):
        second.resolve(blob)

    forged = ProcessReference(first.nonce, 99, "forged")
    with pytest.raises(KeyError, match="slot"):
        first.codec.decode(first.codec._codec.encode(forged))


def test_forged_label_handle_is_rejected() -> None:
    owner = ProcessResources(DataStore())
    owner.retain(object(), label="pose")
    forged = ProcessReference(owner.nonce, 0, "native-scores")

    with pytest.raises(ValueError, match="label"):
        owner.codec.decode(owner.codec._codec.encode(forged))


def test_owner_close_is_terminal_and_clears_handles() -> None:
    owner = ProcessResources(DataStore())
    blob = owner.retain(object(), label="pose")

    owner.close()
    owner.close()

    assert owner._entries == {}
    with pytest.raises(RuntimeError, match="closed"):
        owner.resolve(blob)
    with pytest.raises(RuntimeError, match="closed"):
        owner.retain(object(), label="new")


def test_construction_does_not_consume_python_or_numpy_rng() -> None:
    random.seed(1729)
    python_state = random.getstate()
    np.random.seed(1729)
    numpy_state = np.random.get_state()

    owner = ProcessResources(DataStore())

    assert owner.nonce
    assert random.getstate() == python_state
    _assert_numpy_generator_state_equal(np.random.get_state(), numpy_state)


def test_rooted_put_rejects_process_codec_before_writing(tmp_path) -> None:
    owner = ProcessResources(DataStore())
    reference = ProcessReference(owner.nonce, 0, "unallocated")
    root = tmp_path / "durable"
    rooted = DataStore(root)

    with pytest.raises(ValueError, match="Process-local"):
        rooted.put(reference, owner.codec)

    assert rooted._memory == {}
    assert not root.exists()


def test_recursive_persist_rejects_indirect_process_reference(tmp_path) -> None:
    memory = DataStore()
    owner = ProcessResources(memory)
    resource_blob = owner.retain(object(), label="native pose")
    resource_bytes = memory._memory[resource_blob]

    root = tmp_path / "durable"
    rooted = DataStore(root)
    outer_codec = _ContainerCodec(resource_blob)
    outer_blob = rooted.put(b"outer", outer_codec)
    rooted.register(owner.codec)
    rooted._memory[resource_blob] = resource_bytes

    with pytest.raises(ValueError, match="including through"):
        rooted.persist((outer_blob,))

    assert not root.exists()


def test_tampered_content_digest_is_rejected() -> None:
    data = DataStore()
    owner = ProcessResources(data)
    blob = owner.retain(object(), label="pose")
    forged = Blob(
        identity=Key(hashlib.sha256(b"different bytes").hexdigest()),
        codec=blob.codec,
    )

    with pytest.raises(KeyError, match="not in memory"):
        owner.resolve(forged)
