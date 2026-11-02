"""Object-identity recovery and mutation commit ownership regressions."""
from contextlib import nullcontext
import threading

import numpy as np
import pytest

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import Protomer, SegmentManager
from astrolabe.design import legacy


@pytest.fixture(autouse=True)
def scientific_state():
    global_config.reset()
    for name in ("segments", "protomers", "mutable_protomers", "named_files"):
        global_config.init_const(name, {})
    global_config.init_const("segment_manager", SegmentManager())
    protomer = Protomer("A", "AC", "p")
    global_config.protomers["A"] = protomer
    yield protomer, global_config.segments["A_1"]
    global_config.reset()


@pytest.mark.parametrize("action", ["reverse", "clear"])
@pytest.mark.parametrize("raise_error", [False, True])
@pytest.mark.parametrize("readonly", [False, True])
def test_restore_original_arrays_by_identity_after_list_mutation(scientific_state, action, raise_error, readonly):
    _, segment = scientific_state
    a, b = np.array([1., 2.]), np.array([3., 4.])
    a.flags.writeable = b.flags.writeable = not readonly
    original = [a, b, a, b]
    segment.probs = original
    with pytest.raises(RuntimeError, match="evaluation failed") if raise_error else nullcontext():
        with legacy.LegacyContext().activate():
            getattr(original, action)()
            if action == "clear":
                # The externally held original arrays must still be restored.
                a.flags.writeable = b.flags.writeable = True
                a[:] = 11.
                b[:] = 22.
            if raise_error:
                raise RuntimeError("evaluation failed")
    np.testing.assert_array_equal(a, [1., 2.])
    np.testing.assert_array_equal(b, [3., 4.])
    assert segment.probs is original
    assert len(original) == 4 and original[0] is original[2] is a and original[1] is original[3] is b
    assert a.flags.writeable == b.flags.writeable == (not readonly)


@pytest.mark.parametrize("action", ["reverse", "clear"])
@pytest.mark.parametrize("raise_error", [False, True])
def test_nested_shared_graph_keeps_edges_across_owners(scientific_state, action, raise_error):
    protomer, segment = scientific_state
    a, b = np.array([1., 2.]), np.array([3., 4.])
    inner = [a, b, a, b]
    mapping = {"left": inner, "right": inner, "array": a}
    original = [inner, mapping, inner]
    segment.probs = original
    protomer._homology = mapping
    global_config.init_const("binding_site", {"cache": inner, "mapping": mapping})
    binding = global_config.binding_site
    with pytest.raises(RuntimeError, match="evaluation failed") if raise_error else nullcontext():
        with legacy.LegacyContext().activate():
            getattr(inner, action)()
            original.clear()
            mapping.clear()
            binding["cache"] = []
            if raise_error:
                raise RuntimeError("evaluation failed")
    assert segment.probs is original and original[0] is original[2] is inner
    assert original[1] is mapping and protomer._homology is mapping
    assert global_config.binding_site is binding
    assert binding["cache"] is mapping["left"] is mapping["right"] is inner
    assert binding["mapping"] is mapping and mapping["array"] is a
    assert inner[0] is inner[2] is a and inner[1] is inner[3] is b
    np.testing.assert_array_equal(a, [1., 2.])
    np.testing.assert_array_equal(b, [3., 4.])


@pytest.mark.parametrize("action", ["reverse", "clear"])
def test_immutable_buffer_arrays_keep_identity_without_forced_writes(scientific_state, action):
    _, segment = scientific_state
    a = np.frombuffer(b"\x01\x02", dtype=np.uint8)
    b = np.frombuffer(b"\x03\x04", dtype=np.uint8)
    original = [a, b, a, b]
    segment.probs = original
    with legacy.LegacyContext().activate():
        getattr(original, action)()
    assert segment.probs is original and original[0] is original[2] is a and original[1] is original[3] is b
    assert not a.flags.writeable and not b.flags.writeable
    np.testing.assert_array_equal(a, [1, 2])
    np.testing.assert_array_equal(b, [3, 4])


@pytest.mark.parametrize("explicit_manager", [False, True])
def test_finalize_waits_for_activation_rollback_then_preserves_its_commit(scientific_state, explicit_manager):
    _, segment = scientific_state
    manager = global_config.segment_manager
    segment.mutations = [(1, "C", "G", 0., 1.)]
    began, completed = threading.Event(), threading.Event()
    errors = []

    def finalize():
        began.set()
        try:
            legacy.finalize_pending_mutations(manager if explicit_manager else None)
        except BaseException as error:
            errors.append(error)
        finally:
            completed.set()

    with legacy.LegacyContext().activate():
        thread = threading.Thread(target=finalize)
        thread.start()
        assert began.wait(5)
        premature = completed.wait(0.1)
        protected_sequence = segment.sequence
    thread.join(5)
    assert not thread.is_alive() and not errors
    assert not premature and protected_sequence == "AC"
    assert completed.is_set() and segment.sequence == "AG"
    assert segment.mutations == []


def test_overlapping_readonly_views_restore_the_original_backing_array(scientific_state):
    _, segment = scientific_state
    base = np.arange(6., dtype=np.float64)
    a, b = base[:4], base[2:]
    base.flags.writeable = a.flags.writeable = b.flags.writeable = False
    original = [a, b, a, b]
    segment.probs = original
    with legacy.LegacyContext().activate():
        original.clear()
        base.flags.writeable = True
        base[:] = -1.
    assert segment.probs is original and original[0] is original[2] is a and original[1] is original[3] is b
    assert a.base is b.base is base and np.shares_memory(a, b)
    np.testing.assert_array_equal(base, np.arange(6.))
    np.testing.assert_array_equal(a, np.arange(4.))
    np.testing.assert_array_equal(b, np.arange(2., 6.))
    assert not a.flags.writeable and not b.flags.writeable and not base.flags.writeable


def test_cycle_and_shared_tuple_keep_their_original_edges(scientific_state):
    protomer, segment = scientific_state
    array = np.array([1., 2.])
    inner = [array]
    pair = (inner, array)
    inner.append(pair)
    root = {"pair": pair, "again": pair, "self": inner}
    segment.probs = [root]
    protomer._homology = root
    with legacy.LegacyContext().activate():
        inner.clear()
        root.clear()
        array[:] = -1.
    assert protomer._homology is root and segment.probs[0] is root
    assert root["pair"] is root["again"] is pair
    assert root["self"] is pair[0] is inner
    assert inner[0] is pair[1] is array and inner[1] is pair
    np.testing.assert_array_equal(array, [1., 2.])


@pytest.mark.parametrize("kind", ["object_array", "mutable_external_buffer", "array_subclass"])
def test_unowned_or_unsupported_array_graphs_fail_before_activation(scientific_state, kind):
    _, segment = scientific_state
    if kind == "object_array":
        array = np.array(["source", "record"], dtype=object)
    elif kind == "mutable_external_buffer":
        array = np.frombuffer(bytearray(b"\x01\x02"), dtype=np.uint8)
    else:
        class ExternalArray(np.ndarray):
            pass
        array = np.array([1., 2.]).view(ExternalArray)
    original = [array]
    segment.probs = original
    context = legacy.LegacyContext()
    with pytest.raises(legacy.UnsupportedLegacyState, match="snapshot|owner|storage"):
        with context.activate():
            pytest.fail("unsupported object graph must not execute")
    assert not context.active and context.owner_thread is None
    assert segment.probs is original and original[0] is array


@pytest.mark.parametrize("kind", ["signed_zero", "nan_payload"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("storage", ["owning", "readonly_overlap", "strided_overlap"])
@pytest.mark.parametrize("raise_error", [False, True])
def test_bitwise_array_rollback_preserves_arraycodec_identity(scientific_state, kind, dtype, storage, raise_error):
    from astrolabe.design.store import ArrayCodec, DataStore

    _, segment = scientific_state
    uint = np.uint32 if dtype is np.float32 else np.uint64
    if kind == "signed_zero":
        before_bits = 0x80000000 if dtype is np.float32 else 0x8000000000000000
        changed_bits = 0
    else:
        before_bits = 0x7FC00001 if dtype is np.float32 else 0x7FF8000000000001
        changed_bits = before_bits + 1
    # The captured base must itself be floating point. Registering an integer
    # backing array would let integer comparison mask the float comparison bug.
    base = np.empty(6, dtype=dtype)
    bits = base.view(uint)
    bits[:] = before_bits
    if storage == "owning":
        left = right = base
    elif storage == "readonly_overlap":
        left, right = base[:4], base[2:]
        base.flags.writeable = left.flags.writeable = right.flags.writeable = False
    else:
        left, right = base[::-1], base[::2]
    original = [left, right, left, right]
    segment.probs = original
    arrays = (base, left, right)
    raw_before = tuple(array.tobytes(order="C") for array in arrays)
    flags_before = tuple(array.flags.writeable for array in arrays)
    data, codec = DataStore(), ArrayCodec()
    references = tuple(data.put(array, codec) for array in arrays)
    with pytest.raises(RuntimeError, match="bit mutation failed") if raise_error else nullcontext():
        with legacy.LegacyContext().activate():
            original.reverse()
            # This external integer view was created before read-only flags
            # were set. It changes precisely the bits that float equality hides.
            bits[:] = changed_bits
            if raise_error:
                raise RuntimeError("bit mutation failed")
    assert tuple(array.tobytes(order="C") for array in arrays) == raw_before
    assert tuple(data.put(array, codec) for array in arrays) == references
    assert tuple(array.flags.writeable for array in arrays) == flags_before
    assert segment.probs is original and original[0] is original[2] is left
    assert original[1] is original[3] is right
    assert np.shares_memory(left, right)
    assert np.all(bits == before_bits)
