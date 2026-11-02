from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
import multiprocessing
import os
import queue
import random
import sys
import threading

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


def _interleave(owner, method, statement, first, second):
    """Pause one operation at a statement and observe a competing lock attempt.

    A nonblocking acquire distinguishes actual contention from a thread that
    simply has not been scheduled. Events order the run; timeouts only bound
    failures and clean up threads.
    """
    events = queue.Queue()
    lock = getattr(owner, "_lock", threading.RLock())

    class ObservedLock:
        def __enter__(self):
            if not lock.acquire(blocking=False):
                events.put("blocked")
                lock.acquire()
            return self

        def __exit__(self, *args):
            lock.release()

    owner._lock = ObservedLock()
    source, start = inspect.getsourcelines(method)
    line = start + next(i for i, text in enumerate(source) if statement in text)
    entered, resume = threading.Event(), threading.Event()
    results, errors = {}, {}

    def trace(frame, event, arg):
        if (event == "line" and frame.f_code is method.__code__
                and frame.f_lineno == line):
            entered.set()
            if not resume.wait(5):
                raise TimeoutError("paused operation was not resumed")
        return trace

    def run(name, action):
        if name == "first":
            sys.settrace(trace)
        try:
            results[name] = action()
        except BaseException as exc:
            errors[name] = exc
        finally:
            sys.settrace(None)
            if name == "second":
                events.put("returned")

    threads = [threading.Thread(target=run, args=("first", first), daemon=True)]
    threads[0].start()
    try:
        assert entered.wait(5), "operation never reached the pause point"
        threads.append(threading.Thread(
            target=run, args=("second", second), daemon=True,
        ))
        threads[1].start()
        observed = events.get(timeout=5)
    finally:
        resume.set()
        for thread in threads:
            thread.join(5)
    assert all(not thread.is_alive() for thread in threads), "deadlocked operation"
    assert not errors, errors
    assert observed == "blocked", "competing operation bypassed the owner lock"
    return results["first"], results["second"]


def test_concurrent_retain_keeps_distinct_slots_and_exact_objects() -> None:
    owner = ProcessResources(DataStore())
    candidate = Key("same-candidate")
    first, second = object(), object()

    def retain(value):
        return owner.retain(
            value, label="native-scores", candidate=candidate, revision=7,
        )

    first_blob, second_blob = _interleave(
        owner, ProcessResources.retain, "self._next_slot += 1",
        lambda: retain(first), lambda: retain(second),
    )
    assert first_blob != second_blob
    for blob, value in ((first_blob, first), (second_blob, second)):
        assert owner.resolve(blob, candidate=candidate, revision=7) is value
        assert owner.validate_scores(blob, value, candidate, revision=7) == (blob, 7)


def test_close_waits_for_expire_and_cannot_repopulate_closed_owner() -> None:
    owner = ProcessResources(DataStore())
    owner.retain(object(), label="native-scores")
    _interleave(
        owner, ProcessResources.expire_scores,
        "self._entries[slot] = replace(entry, expired=True)",
        owner.expire_scores, owner.close,
    )
    assert not owner._alive
    assert owner._entries == {}


def test_expire_waits_for_close_then_rejects_closed_owner() -> None:
    owner = ProcessResources(DataStore())
    owner.retain(object(), label="native-scores")

    def expire_closed():
        with pytest.raises(RuntimeError, match="closed"):
            owner.expire_scores()

    _interleave(
        owner, ProcessResources.close, "self._entries.clear()",
        owner.close, expire_closed,
    )
    assert owner._entries == {}


def test_close_waits_for_retain_then_invalidates_its_handle() -> None:
    owner = ProcessResources(DataStore())
    blob, _ = _interleave(
        owner, ProcessResources.retain, "self._next_slot += 1",
        lambda: owner.retain(object(), label="pose"), owner.close,
    )
    assert owner._entries == {}
    with pytest.raises(RuntimeError, match="closed"):
        owner.resolve(blob)


def test_validate_scores_and_expiry_are_one_serialized_operation() -> None:
    owner = ProcessResources(DataStore())
    scores, candidate = object(), Key("candidate")
    blob = owner.retain(
        scores, label="native-scores", candidate=candidate, revision=7,
    )
    validated, _ = _interleave(
        owner, ProcessResources.validate_scores,
        'if entry.label != "native-scores":',
        lambda: owner.validate_scores(blob, scores, candidate, revision=7),
        owner.expire_scores,
    )
    assert validated == (blob, 7)
    with pytest.raises(RuntimeError, match="expired"):
        owner.validate_scores(blob, scores, candidate)


def test_resolve_and_expiry_are_one_serialized_operation() -> None:
    owner = ProcessResources(DataStore())
    scores = object()
    blob = owner.retain(scores, label="native-scores")
    resolved, _ = _interleave(
        owner, ProcessResources._resolve_entry, "if entry.expired:",
        lambda: owner.resolve(blob), owner.expire_scores,
    )
    assert resolved is scores
    with pytest.raises(RuntimeError, match="expired"):
        owner.resolve(blob)


@pytest.mark.parametrize("operation", ["encode", "decode"])
def test_codec_operations_and_close_are_serialized(operation) -> None:
    owner = ProcessResources(DataStore())
    blob = owner.retain(object(), label="pose")
    reference = owner.data.get(blob, owner.codec)
    encoded = owner.codec.encode(reference)
    method = getattr(ProcessResourceCodec, operation)
    argument = reference if operation == "encode" else encoded
    returned, _ = _interleave(
        owner, method, "self._owner._entry_for_reference(",
        lambda: getattr(owner.codec, operation)(argument), owner.close,
    )
    assert returned == (encoded if operation == "encode" else reference)
    with pytest.raises(RuntimeError, match="closed"):
        getattr(owner.codec, operation)(argument)


class _CallbackStore(DataStore):
    on_put = None

    def put(self, value, codec):
        blob = super().put(value, codec)
        callback, self.on_put = self.on_put, None
        if callback is not None:
            callback(blob)
        return blob


def _in_thread(action):
    results, errors = [], []

    def run():
        try:
            results.append(action())
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(5)
    assert not thread.is_alive(), "reentrant DataStore callback deadlocked"
    if errors:
        raise errors[0]
    return results[0]


def test_data_store_callback_can_reenter_resolve_and_retain() -> None:
    data = _CallbackStore()
    owner = ProcessResources(data)
    outer, inner = object(), object()
    nested = []

    def callback(blob):
        assert owner.resolve(blob) is outer
        nested.append(owner.retain(inner, label="pose"))

    data.on_put = callback
    blob = _in_thread(lambda: owner.retain(outer, label="pose"))
    assert blob != nested[0]
    assert owner.resolve(blob) is outer
    assert owner.resolve(nested[0]) is inner


@pytest.mark.parametrize("operation", ["close", "expire_scores"])
def test_retain_rechecks_liveness_after_reentrant_store_callback(operation) -> None:
    data = _CallbackStore()
    owner = ProcessResources(data)
    data.on_put = lambda blob: getattr(owner, operation)()

    def retain():
        with pytest.raises(RuntimeError, match="closed|expired"):
            owner.retain(object(), label="native-scores")

    _in_thread(retain)
    assert owner._entries == {}


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork")
@pytest.mark.parametrize("lock_held", [False, True])
def test_fork_invalidates_owner_and_codec_before_acquiring_inherited_lock(lock_held):
    owner = ProcessResources(DataStore())
    value, candidate = object(), Key("candidate")
    blob = owner.retain(
        value, label="native-scores", candidate=candidate, revision=7,
    )
    codec = owner.codec
    reference = owner.data.get(blob, codec)
    encoded = codec.encode(reference)
    ready, release = threading.Event(), threading.Event()
    lock = getattr(owner, "_lock", threading.RLock())

    def hold_lock():
        with lock:
            ready.set()
            release.wait()

    context = multiprocessing.get_context("fork")
    reader, writer = context.Pipe(duplex=False)

    def child():
        actions = {
            "resolve": lambda: owner.resolve(blob),
            "retain": lambda: owner.retain(object(), label="pose"),
            "validate": lambda: owner.validate_scores(blob, value, candidate),
            "expire": owner.expire_scores,
            "close": owner.close,
            "codec-property": lambda: owner.codec,
            "encode": lambda: codec.encode(reference),
            "decode": lambda: codec.decode(encoded),
            "references": lambda: codec.references(reference),
        }
        rejected, failures = [], []
        for name, action in actions.items():
            try:
                action()
            except RuntimeError as exc:
                if "another process" in str(exc):
                    rejected.append(name)
                else:
                    failures.append((name, str(exc)))
            else:
                failures.append((name, "inherited owner accepted"))
        fresh = ProcessResources(DataStore())
        fresh_blob = fresh.retain(value, label="pose")
        writer.send((rejected, failures, fresh.resolve(fresh_blob) is value))
        writer.close()

    holder = threading.Thread(target=hold_lock, daemon=True) if lock_held else None
    process = context.Process(target=child)
    try:
        if holder is not None:
            holder.start()
            assert ready.wait(5)
        process.start()
        writer.close()
        process.join(5)
        assert not process.is_alive(), "child waited on an inherited owner lock"
        assert process.exitcode == 0
        assert reader.poll(1)
        rejected, failures, fresh_valid = reader.recv()
        assert not failures, failures
        assert len(rejected) == 9
        assert fresh_valid
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)
        release.set()
        if holder is not None:
            holder.join(5)
        reader.close()
        writer.close()
    assert owner.resolve(blob, candidate=candidate, revision=7) is value


def test_expiry_and_close_do_not_revoke_already_returned_live_references() -> None:
    owner = ProcessResources(DataStore())
    value = {"score": 1}
    blob = owner.retain(value, label="native-scores")
    live = owner.resolve(blob)
    owner.expire_scores()
    with pytest.raises(RuntimeError, match="expired"):
        owner.resolve(blob)
    owner.close()
    live["score"] = 2
    assert live is value and value["score"] == 2


def test_full_resource_lifecycle_does_not_consume_python_or_numpy_rng() -> None:
    python_state, numpy_state = random.getstate(), np.random.get_state()
    owner = ProcessResources(DataStore())
    candidate, value = Key("candidate"), object()
    blob = owner.retain(value, label="native-scores", candidate=candidate, revision=7)
    assert owner.resolve(blob) is value
    assert owner.validate_scores(blob, value, candidate) == (blob, 7)
    owner.expire_scores()
    owner.close()
    assert random.getstate() == python_state
    _assert_numpy_generator_state_equal(np.random.get_state(), numpy_state)


def test_tampered_existing_blob_bytes_are_rejected() -> None:
    owner = ProcessResources(DataStore())
    blob = owner.retain(object(), label="pose")
    owner.data._memory[blob] = b"{}"
    with pytest.raises(ValueError, match="content digest mismatch"):
        owner.resolve(blob)


def _comparison_metadata(field, callback):
    class Text(str):
        def __eq__(self, other):
            callback()
            return True

        def __ne__(self, other):
            callback()
            return False

        __hash__ = str.__hash__

    class Candidate(Key):
        def __eq__(self, other):
            callback()
            return True

        def __ne__(self, other):
            callback()
            return False

        __hash__ = Key.__hash__

    class Revision(int):
        def __eq__(self, other):
            callback()
            return True

        def __ne__(self, other):
            callback()
            return False

        __hash__ = int.__hash__

    return {
        "candidate": lambda: Candidate("candidate"),
        "digest": lambda: Key(Text("candidate")),
        "revision": lambda: Revision(7),
        "label": lambda: Text("native-scores"),
        "session": lambda: Text("session"),
        "slot": lambda: Revision(0),
    }[field]()


@pytest.mark.parametrize("operation", ["close", "expire_scores"])
@pytest.mark.parametrize("field", ["candidate", "digest", "revision", "label"])
def test_retain_rejects_comparison_metadata_without_executing_it(operation, field):
    owner = ProcessResources(DataStore())
    calls = []

    def callback():
        calls.append(operation)
        getattr(owner, operation)()

    metadata = dict(label="native-scores", candidate=Key("candidate"), revision=7)
    metadata["candidate" if field == "digest" else field] = _comparison_metadata(
        field, callback,
    )
    with pytest.raises(TypeError, match="label|candidate|digest|revision"):
        owner.retain(object(), **metadata)
    assert calls == []
    assert owner._entries == {}
    value = object()
    assert owner.resolve(owner.retain(value, label="pose")) is value


@pytest.mark.parametrize("lookup", ["resolve", "validate_scores"])
@pytest.mark.parametrize("operation", ["close", "expire_scores"])
@pytest.mark.parametrize("field", ["candidate", "digest", "revision"])
def test_lookup_rejects_comparison_metadata_before_reentrant_invalidation(
    lookup, operation, field,
):
    owner = ProcessResources(DataStore())
    value, candidate = object(), Key("candidate")
    blob = owner.retain(
        value, label="native-scores", candidate=candidate, revision=7,
    )
    calls = []

    def callback():
        calls.append(operation)
        getattr(owner, operation)()

    metadata = dict(candidate=candidate, revision=7)
    metadata["candidate" if field == "digest" else field] = _comparison_metadata(
        field, callback,
    )
    with pytest.raises(TypeError, match="candidate|digest|revision"):
        if lookup == "resolve":
            owner.resolve(blob, **metadata)
        else:
            owner.validate_scores(blob, value, **metadata)
    assert calls == []
    assert owner.resolve(blob, candidate=candidate, revision=7) is value


@pytest.mark.parametrize("operation", ["encode", "references"])
@pytest.mark.parametrize("field", ["session", "slot", "label"])
def test_reference_metadata_requires_plain_types_before_comparison(operation, field):
    owner = ProcessResources(DataStore())
    blob = owner.retain(object(), label="native-scores")
    reference = owner.data.get(blob, owner.codec)
    calls = []

    def callback():
        calls.append(field)
        owner.close()

    reference = replace(reference, **{field: _comparison_metadata(field, callback)})
    with pytest.raises(TypeError, match="session|slot|label"):
        getattr(owner.codec, operation)(reference)
    assert calls == []
    owner.resolve(blob)


@pytest.mark.parametrize("operation", ["retain", "resolve", "validate_scores"])
def test_candidate_digest_must_be_a_plain_string(operation):
    owner = ProcessResources(DataStore())
    value = object()
    blob = owner.retain(value, label="native-scores", candidate=Key("candidate"))
    with pytest.raises(TypeError, match="digest"):
        if operation == "retain":
            owner.retain(value, label="native-scores", candidate=Key(123))
        elif operation == "resolve":
            owner.resolve(blob, candidate=Key(123))
        else:
            owner.validate_scores(blob, value, Key(123))


@pytest.mark.parametrize("lookup", ["resolve", "validate_scores"])
@pytest.mark.parametrize("field", ["candidate", "digest", "revision", "label"])
def test_stored_metadata_is_checked_before_comparison(lookup, field):
    owner = ProcessResources(DataStore())
    value, candidate = object(), Key("candidate")
    blob = owner.retain(value, label="native-scores", candidate=candidate, revision=7)
    original = owner._entries[0]
    calls = []

    def callback():
        calls.append(field)
        owner.close()

    owner._entries[0] = replace(original, **{
        "candidate" if field == "digest" else field: _comparison_metadata(field, callback),
    })
    with pytest.raises(TypeError, match="candidate|digest|revision|label"):
        if lookup == "resolve":
            owner.resolve(blob, candidate=candidate, revision=7)
        else:
            owner.validate_scores(blob, value, candidate, revision=7)
    assert calls == []
    owner._entries[0] = original
    assert owner.resolve(blob) is value


@pytest.mark.parametrize("lookup", ["resolve", "validate_scores"])
@pytest.mark.parametrize("field", ["codec", "identity", "digest"])
def test_blob_metadata_is_checked_before_hashing_or_comparison(lookup, field):
    owner = ProcessResources(DataStore())
    value, candidate = object(), Key("candidate")
    blob = owner.retain(value, label="native-scores", candidate=candidate)
    calls = []

    def callback():
        calls.append(field)
        owner.close()

    if field == "codec":
        forged = replace(blob, codec=_comparison_metadata("digest", callback).digest)
    elif field == "identity":
        forged = replace(blob, identity=_comparison_metadata("candidate", callback))
    else:
        forged = replace(blob, identity=_comparison_metadata("digest", callback))
    with pytest.raises(TypeError, match="blob"):
        if lookup == "resolve":
            owner.resolve(forged)
        else:
            owner.validate_scores(forged, value, candidate)
    assert calls == []
    assert owner.resolve(blob) is value


@pytest.mark.parametrize("slot", [False, 0.0])
def test_decoded_reference_slot_cannot_alias_an_integer_with_another_type(slot):
    owner = ProcessResources(DataStore())
    blob = owner.retain(object(), label="pose")
    reference = owner.data.get(blob, owner.codec)
    encoded = owner.codec._codec.encode(replace(reference, slot=slot))
    with pytest.raises(TypeError, match="slot"):
        owner.codec.decode(encoded)


@pytest.mark.parametrize("lookup", ["resolve", "validate_scores"])
@pytest.mark.parametrize("invalidation", ["close", "expire", "session", "entry", "codec"])
def test_lookup_rechecks_current_session_and_entry_immediately_before_return(
    monkeypatch, lookup, invalidation,
):
    owner = ProcessResources(DataStore())
    value, candidate = object(), Key("candidate")
    blob = owner.retain(value, label="native-scores", candidate=candidate, revision=7)
    resolve_entry = owner._resolve_entry

    def invalidate_after_lookup(*args, **kwargs):
        result = resolve_entry(*args, **kwargs)
        if invalidation == "close":
            owner.close()
        elif invalidation == "expire":
            owner.expire_scores()
        elif invalidation == "session":
            owner.nonce = "different-session"
        elif invalidation == "entry":
            owner._entries[0] = replace(owner._entries[0], revision=8)
        else:
            owner.data._codecs.pop(blob.codec)
        return result

    monkeypatch.setattr(owner, "_resolve_entry", invalidate_after_lookup)
    with pytest.raises((RuntimeError, ValueError), match="closed|expired|session|current"):
        if lookup == "resolve":
            owner.resolve(blob, candidate=candidate, revision=7)
        else:
            owner.validate_scores(blob, value, candidate, revision=7)
