"""Protomer-local locks become shared-variable constraints before any write."""
import numpy as np
import pytest

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import Protomer, Segment, SegmentManager
from astrolabe.design.legacy import (
    LegacyContext, UnsupportedLegacyState, candidate_for_mutation, translate_legacy_state,
)
from astrolabe.design.store import DataStore
from astrolabe.design.types import Assignment, Polymer, make_candidate


@pytest.fixture(autouse=True)
def scientific_state():
    global_config.reset()
    for name in ("segments", "protomers", "mutable_protomers", "named_files"):
        global_config.init_const(name, {})
    global_config.init_const("segment_manager", SegmentManager())
    yield
    global_config.reset()


def add(name, sequence, kind="p"):
    protomer = Protomer(name, sequence, kind)
    global_config.protomers[name] = protomer
    return protomer


def shared_state():
    add("S", "ACDE@shared")
    a = add("A", "+TT shared shared GG")
    b = add("B", "GGG shared")
    unselected = add("U", "shared")
    a.add_locked_positions([2, 7, 11])  # shared[0], shared[1], tail[1]
    b.add_locked_positions([4])         # shared[1], despite a different offset
    unselected.add_locked_positions([2])
    return a, b, unselected


def test_locks_union_uses_component_order_repetitions_and_unselected_protomers():
    a, b, unselected = shared_state()
    data = DataStore()
    translation = translate_legacy_state(assemblies=("AAB",), data=data)
    domains = {domain.variable: domain.fixed_tokens for domain in translation.system.domains}
    assert domains == {"S_1": (0, 1, 2), "A_1": (0, 1), "A_4": (1,)}
    assert translation.assemblies[0].chains == (("A", "A"), ("B", "A"), ("C", "B"))
    shared = [item for item in translation.system.occurrences if item.variable == "S_1"]
    assert [item.chain for item in shared] == ["A", "A", "B", "B", "C"]
    assert a.locked_positions == frozenset({2, 7, 11})
    assert b.locked_positions == frozenset({4})
    assert unselected.locked_positions == frozenset({2})

    # Changing a lock in a protomer absent from every assembly still changes the System.
    unselected.add_locked_positions([3])
    changed = translate_legacy_state(assemblies=("AAB",), data=data)
    assert changed.system.identity != translation.system.identity
    assert next(d for d in changed.system.domains if d.variable == "S_1").fixed_tokens == (0, 1, 2, 3)


def test_unlocked_edit_preserves_aliases_local_locks_and_original_fixed_flags():
    protomers = shared_state()
    segment = global_config.segments["S_1"]
    lock_objects = [p._locked_positions for p in protomers]
    data = DataStore()
    translation = translate_legacy_state(assemblies=("AAB",), data=data)
    proposal = candidate_for_mutation(translation, (
        Assignment("S_1", Polymer("protein", tuple("ACDG"), ())),
        Assignment("A_4", Polymer("protein", tuple("AG"), ())),
    ), data=data)
    LegacyContext().apply_candidate(proposal, translation=translation, data=data)
    for alias in ("S_1", "shared", "A_2", "A_3", "B_2", "U_1"):
        assert global_config.segments[alias] is segment
        assert global_config.segments[alias].sequence == "ACDG"
    assert segment.is_fixed is False
    assert global_config.segments["A_1"].is_fixed is True
    assert global_config.segments["A_4"].is_fixed is False
    assert [p._locked_positions for p in protomers] == [{2, 7, 11}, {4}, {2}]
    assert all(p._locked_positions is locks for p, locks in zip(protomers, lock_objects))


@pytest.mark.parametrize("variable,sequence", [
    ("S_1", "GCDE"), ("S_1", "AGDE"), ("S_1", "ACGE"),
    ("A_1", "GT"), ("A_4", "GA"),
])
def test_locked_edit_and_direct_candidate_bypass_fail_before_apply(variable, sequence):
    protomers = shared_state()
    data = DataStore()
    context = LegacyContext()
    translation = context.capture_candidate(assemblies=("AAB",), data=data)
    replacement = Assignment(variable, Polymer("protein", tuple(sequence), ()))
    with pytest.raises(ValueError, match="fixed token"):
        translation.system.edit(translation.candidate, (replacement,), data=data)
    direct = make_candidate(translation.system, tuple(
        replacement if item.variable == variable else item
        for item in translation.candidate.assignments
    ), data=data)
    with pytest.raises(ValueError, match="fixed token"):
        context.apply_candidate(direct, translation=translation, data=data)
    assert global_config.segments["S_1"].sequence == "ACDE"
    assert global_config.segments["A_1"].sequence == "TT"
    assert global_config.segments["A_4"].sequence == "GG"
    assert [p._locked_positions for p in protomers] == [{2, 7, 11}, {4}, {2}]


def test_union_covering_entire_segment_does_not_promote_is_fixed():
    add("S", "ACDE@shared")
    for name, position in zip("ABCD", range(4)):
        add(name, "shared").add_locked_positions([position])
    data = DataStore()
    translation = translate_legacy_state(assemblies=("A",), data=data)
    assert translation.system.domains[0].fixed_tokens == (0, 1, 2, 3)
    segment = global_config.segments["S_1"]
    assert segment.is_fixed is False
    LegacyContext().apply_candidate(translation.candidate, translation=translation, data=data)
    assert segment.is_fixed is False
    assert [global_config.protomers[n].locked_positions for n in "ABCD"] == [
        frozenset({position}) for position in range(4)
    ]


@pytest.mark.parametrize("kind,sequence,replacement", [("r", "ACGU", "ACGA"), ("d", "ACGT", "ACGA")])
def test_local_locks_are_polymer_constraints_for_rna_and_dna(kind, sequence, replacement):
    protomer = add("A", sequence, kind)
    protomer.add_locked_positions([1])
    data = DataStore()
    translation = translate_legacy_state(assemblies=("A",), data=data)
    molecule_kind = translation.candidate.assignments[0].value.kind
    assert translation.system.domains[0].fixed_tokens == (1,)
    proposal = candidate_for_mutation(translation, (
        Assignment("A_1", Polymer(molecule_kind, tuple(replacement), ())),
    ), data=data)
    LegacyContext().apply_candidate(proposal, translation=translation, data=data)
    assert global_config.segments["A_1"].sequence == replacement
    assert global_config.segments["A_1"].is_fixed is False
    assert protomer.locked_positions == frozenset({1})


@pytest.mark.parametrize("locks", [{-1}, {2}, {"0"}, {0.5}, {True}, None, 1])
def test_unrepresentable_or_out_of_range_locks_fail_closed(locks):
    a = add("A", "AC")
    a._locked_positions = locks  # Deliberately corrupt native state, bypassing its setter.
    with pytest.raises(UnsupportedLegacyState, match="lock"):
        translate_legacy_state(assemblies=("A",), data=DataStore())
    assert global_config.segments["A_1"].sequence == "AC"


def test_chemical_locks_fail_without_guessing_atom_or_token_offsets():
    add("A", "AC")
    ligand = add("L", "CCO", "s")
    ligand._locked_positions = {0}
    with pytest.raises(UnsupportedLegacyState, match="chemical|polymer"):
        translate_legacy_state(assemblies=("A",), data=DataStore())
    ligand._locked_positions = set()
    a = global_config.protomers["A"]
    a.components.append("L_1")
    a._locked_positions = {2}
    with pytest.raises(UnsupportedLegacyState, match="chemical|polymer"):
        translate_legacy_state(assemblies=("A",), data=DataStore())


def test_partial_lock_edits_roll_back_after_mid_write_failure(monkeypatch):
    protomers = shared_state()
    segment = global_config.segments["S_1"]
    tail = global_config.segments["A_4"]
    segment.probs = [np.array([0.25, 0.75])]
    segment.mutations = [(3, "E", "G", 1.0, 0.8)]
    global_config.segment_manager.begin_pre_update_step()
    data = DataStore()
    context = LegacyContext()
    translation = context.capture_candidate(assemblies=("AAB",), data=data)
    proposal = candidate_for_mutation(translation, (
        Assignment("S_1", Polymer("protein", tuple("ACDG"), ())),
        Assignment("A_4", Polymer("protein", tuple("AG"), ())),
    ), data=data)
    original_setattr = Segment.__setattr__

    def fail_on_tail(self, name, value):
        if self is tail and name == "sequence" and value == "AG":
            raise RuntimeError("tail write failed")
        original_setattr(self, name, value)

    monkeypatch.setattr(Segment, "__setattr__", fail_on_tail)
    with pytest.raises(RuntimeError, match="tail write failed"):
        context.apply_candidate(proposal, translation=translation, data=data)
    assert segment.sequence == "ACDE" and tail.sequence == "GG"
    assert segment.is_fixed is False and tail.is_fixed is False
    assert global_config.segments["A_1"].is_fixed is True
    assert global_config.segments["A_2"] is global_config.segments["A_3"] is segment
    np.testing.assert_array_equal(segment.probs[0], [0.25, 0.75])
    assert segment.mutations == [(3, "E", "G", 1.0, 0.8)]
    assert global_config.segment_manager._pre_update_snapshot is not None
    assert [p._locked_positions for p in protomers] == [{2, 7, 11}, {4}, {2}]


def test_new_live_lock_invalidates_an_older_unrestricted_translation():
    a = add("A", "AC")
    data = DataStore()
    translation = translate_legacy_state(assemblies=("A",), data=data)
    proposal = candidate_for_mutation(translation, (
        Assignment("A_1", Polymer("protein", tuple("AG"), ())),
    ), data=data)
    a.add_locked_positions([1])
    with pytest.raises(UnsupportedLegacyState, match="lock"):
        LegacyContext().apply_candidate(proposal, translation=translation, data=data)
    assert global_config.segments["A_1"].sequence == "AC"
    assert a.locked_positions == frozenset({1})


def test_repeated_component_name_uses_each_physical_offset():
    a = add("A", "ACDE")
    a.components = ["A_1", "A_1"]
    a.add_locked_positions([5])  # Second physical copy, local index 1.
    data = DataStore()
    translation = translate_legacy_state(assemblies=("A",), data=data)
    assert translation.system.domains[0].fixed_tokens == (1,)
    occurrences = translation.system.occurrences
    assert len(occurrences) == 2
    assert occurrences[0].identity != occurrences[1].identity
    proposal = candidate_for_mutation(translation, (
        Assignment("A_1", Polymer("protein", tuple("ACGE"), ())),
    ), data=data)
    LegacyContext().apply_candidate(proposal, translation=translation, data=data)
    assert global_config.segments["A_1"].sequence == "ACGE"
    assert global_config.segments["A_1"].is_fixed is False
    assert a.locked_positions == frozenset({5})


def test_activation_exception_restores_shared_partial_lock_state():
    protomers = shared_state()
    segment = global_config.segments["S_1"]
    data = DataStore()
    context = LegacyContext()
    translation = context.capture_candidate(assemblies=("AAB",), data=data)
    proposal = candidate_for_mutation(translation, (
        Assignment("S_1", Polymer("protein", tuple("ACDG"), ())),
    ), data=data)
    with pytest.raises(RuntimeError, match="evaluation failed"):
        with context.activate(proposal, translation=translation, data=data):
            assert segment.sequence == "ACDG"
            assert segment.is_fixed is False
            assert [p._locked_positions for p in protomers] == [{2, 7, 11}, {4}, {2}]
            raise RuntimeError("evaluation failed")
    assert segment.sequence == "ACDE"
    assert segment.is_fixed is False
    assert global_config.segments["A_2"] is global_config.segments["A_3"] is segment
    assert [p._locked_positions for p in protomers] == [{2, 7, 11}, {4}, {2}]
    assert not context.active
