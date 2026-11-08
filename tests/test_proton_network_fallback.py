"""Best-effort proposal contracts with synthetic anchors, without Rosetta/model runs."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from astrolabe.mutators.proton_network_geometry import (
    NetworkProposalParameters,
    _best_effort_pair_preferences,
    build_network_pair_preferences,
)


_ONE_LETTER = {"HIS": "H", "LYS": "K", "ARG": "R"}


def _assert_legal_preferences(prefs, sequence, mutable_positions, pair):
    """Every role is legal, preserves fixed identities, and changes the sequence."""
    assert prefs
    free = set(mutable_positions)
    assert {frozenset((p["i"], p["j"])) for p in prefs} == {frozenset(pair)}
    for pref in prefs:
        assert pref["aa_i"] == "HIS"
        assert pref["aa_j"] in _ONE_LETTER
        assert pref["quality"] == 0.0
        edited = []
        for position, token in ((pref["i"], pref["aa_i"]),
                                (pref["j"], pref["aa_j"])):
            aa = _ONE_LETTER[token]
            if position not in free:
                assert aa == sequence[position]
            if aa != sequence[position]:
                edited.append(position)
        assert edited
        assert set(edited) <= free


def _anchors(**positions):
    return {int(i): np.asarray(xyz, dtype=float) for i, xyz in positions.items()}


def test_no_qualifying_geometry_still_returns_one_pair_with_all_legal_roles():
    parameters = NetworkProposalParameters()
    # These synthetic sites fail distance, sequence separation and depth filters.
    anchors = _anchors(**{"0": (0, 0, 0), "1": (20, 0, 0)})
    prefs, details = _best_effort_pair_preferences(
        "AA", [0, 1], anchors, {0: 0.1, 1: 0.1}, parameters
    )
    _assert_legal_preferences(prefs, "AA", [0, 1], (0, 1))
    assignments = {
        tuple(_ONE_LETTER[p["aa_i"]] if k == p["i"]
              else _ONE_LETTER[p["aa_j"]] for k in (0, 1))
        for p in prefs
    }
    assert assignments == {("H", "H"), ("H", "K"), ("H", "R"),
                           ("K", "H"), ("R", "H")}
    assert isinstance(details, dict)


def test_distance_violation_precedes_depth_and_sequence_separation():
    # Pair 0/1 is within the CB range but shallow and close in sequence;
    # pair 0/3 is deep and well separated but outside the distance range.
    anchors = _anchors(**{"0": (0, 0, 0), "1": (6, 0, 0), "3": (-10, 0, 0)})
    prefs, _ = _best_effort_pair_preferences(
        "AAAA", [0, 1, 3], anchors, {0: 0.1, 1: 0.1, 3: 100.0},
        NetworkProposalParameters(),
    )
    _assert_legal_preferences(prefs, "AAAA", [0, 1, 3], (0, 1))


@pytest.mark.parametrize("distance", [2.0, 10.0])
def test_smallest_distance_violation_wins_on_either_side_of_interval(distance):
    # 2 and 10 are each one angstrom outside the default [3, 9] interval.
    anchors = _anchors(**{"0": (0, 0, 0), "3": (distance, 0, 0),
                         "6": (-20, 0, 0)})
    prefs, _ = _best_effort_pair_preferences(
        "AAAAAAA", [0, 3, 6], anchors, {0: 1.0, 3: 1.0, 6: 100.0},
        NetworkProposalParameters(),
    )
    _assert_legal_preferences(prefs, "AAAAAAA", [0, 3, 6], (0, 3))


def test_sequence_separation_precedes_depth_when_distance_ties():
    anchors = _anchors(**{"0": (0, 0, 0), "1": (6, 0, 0), "3": (-6, 0, 0)})
    prefs, _ = _best_effort_pair_preferences(
        "AAAA", [0, 1, 3], anchors, {0: 1.0, 1: 100.0, 3: 1.0},
        NetworkProposalParameters(),
    )
    _assert_legal_preferences(prefs, "AAAA", [0, 1, 3], (0, 3))


def test_equal_distance_prefers_more_buried_pair():
    anchors = _anchors(**{"0": (0, 0, 0), "3": (6, 0, 0), "6": (-6, 0, 0)})
    prefs, _ = _best_effort_pair_preferences(
        "AAAAAAA", [0, 3, 6], anchors, {0: 1.0, 3: 2.0, 6: 5.0},
        NetworkProposalParameters(),
    )
    _assert_legal_preferences(prefs, "AAAAAAA", [0, 3, 6], (0, 6))


def test_interval_midpoint_is_derived_from_parameters():
    # The custom interval midpoint is 12; a hard-coded midpoint of 6 picks 0/3.
    anchors = _anchors(**{"0": (0, 0, 0), "3": (10, 0, 0), "6": (-12, 0, 0)})
    prefs, _ = _best_effort_pair_preferences(
        "AAAAAAA", [0, 3, 6], anchors, {0: 1.0, 3: 1.0, 6: 1.0},
        NetworkProposalParameters(min_cb_distance=9.0, max_cb_distance=15.0),
    )
    _assert_legal_preferences(prefs, "AAAAAAA", [0, 3, 6], (0, 6))


def test_final_position_tiebreak_is_independent_of_anchor_insertion_order():
    anchors = _anchors(**{"6": (-6, 0, 0), "3": (6, 0, 0), "0": (0, 0, 0)})
    prefs, _ = _best_effort_pair_preferences(
        "AAAAAAA", [6, 3, 0], anchors, {0: 1.0, 3: 1.0, 6: 1.0},
        NetworkProposalParameters(),
    )
    _assert_legal_preferences(prefs, "AAAAAAA", [6, 3, 0], (0, 3))


@pytest.mark.parametrize("fixed_aa", ["H", "K", "R"])
@pytest.mark.parametrize("fixed_position", [0, 1])
def test_fixed_endpoint_retains_its_current_canonical_identity(fixed_aa, fixed_position):
    sequence = list("AA")
    sequence[fixed_position] = fixed_aa
    sequence = "".join(sequence)
    mutable = [1 - fixed_position]
    prefs, _ = _best_effort_pair_preferences(
        sequence, mutable, _anchors(**{"0": (0, 0, 0), "1": (20, 0, 0)}),
        {0: 1.0, 1: 1.0}, NetworkProposalParameters(),
    )
    _assert_legal_preferences(prefs, sequence, mutable, (0, 1))


@pytest.mark.parametrize("sequence", ["HH", "HK", "HR", "KH", "RH"])
def test_existing_pair_never_supplies_a_zero_canonical_edit_preference(sequence):
    prefs, _ = _best_effort_pair_preferences(
        sequence, [0, 1], _anchors(**{"0": (0, 0, 0), "1": (6, 0, 0)}),
        {0: 4.0, 1: 4.0}, NetworkProposalParameters(),
    )
    _assert_legal_preferences(prefs, sequence, [0, 1], (0, 1))


def test_fixed_his_requires_mutable_his_to_change_to_k_or_r():
    prefs, _ = _best_effort_pair_preferences(
        "HH", [0], _anchors(**{"0": (0, 0, 0), "1": (6, 0, 0)}),
        {0: 1.0, 1: 1.0}, NetworkProposalParameters(),
    )
    _assert_legal_preferences(prefs, "HH", [0], (0, 1))
    assert {(p["i"], p["j"], p["aa_j"]) for p in prefs} == {
        (1, 0, "LYS"), (1, 0, "ARG")
    }


@pytest.mark.parametrize(
    ("sequence", "mutable", "anchors"),
    [
        pytest.param("HK", [], {0: (0, 0, 0), 1: (6, 0, 0)}, id="all-fixed"),
        pytest.param("AH", [1], {0: (0, 0, 0), 1: (6, 0, 0)}, id="fixed-alanine"),
        pytest.param("HA", [0], {0: (0, 0, 0), 1: (6, 0, 0)}, id="fixed-alanine-reversed"),
        pytest.param("AA", [0, 1], {0: (0, 0, 0)}, id="only-one-anchor"),
        pytest.param("AA", [0, 1], {}, id="no-anchors"),
    ],
)
def test_no_legal_pair_returns_empty_preferences(sequence, mutable, anchors):
    prefs, details = _best_effort_pair_preferences(
        sequence, mutable, {i: np.asarray(xyz, dtype=float) for i, xyz in anchors.items()},
        {i: 1.0 for i in anchors}, NetworkProposalParameters(),
    )
    assert prefs == []
    assert isinstance(details, dict)


def test_helper_accepts_supplied_ca_anchors_without_atom_or_pose_lookup():
    # The builder can supply CA coordinates when CB is absent. The pure helper
    # receives only those coordinates, with no atom labels or structure object.
    ca_anchors = {0: np.array([0.0, 0.0, 0.0]), 1: np.array([8.0, 0.0, 0.0])}
    original = {i: xyz.copy() for i, xyz in ca_anchors.items()}
    prefs, _ = _best_effort_pair_preferences(
        "GG", [0, 1], ca_anchors, {0: 1.0, 1: 1.0}, NetworkProposalParameters()
    )
    _assert_legal_preferences(prefs, "GG", [0, 1], (0, 1))
    for position in ca_anchors:
        np.testing.assert_array_equal(ca_anchors[position], original[position])


def test_geometry_builder_fallback_is_keyword_only_and_disabled_by_default():
    option = inspect.signature(build_network_pair_preferences).parameters["fallback_to_best"]
    assert option.kind is inspect.Parameter.KEYWORD_ONLY
    assert option.default is False


@pytest.mark.parametrize("target_internal_iptm", [0.0, 0.99])
def test_abcd_score_uses_a_to_bcd_interfaces_and_a_chain_ptm(target_internal_iptm):
    from astrolabe.losses.base import LossEngine
    from astrolabe.losses.confidence import binder_ptm, chain_pair_iptm

    # Synthetic confidence values: target-only interfaces must not affect iPTM_A_BCD.
    matrix = np.array([
        [0.0, 0.1, 0.2, 0.3],
        [0.1, 0.0, target_internal_iptm, target_internal_iptm],
        [0.2, target_internal_iptm, 0.0, target_internal_iptm],
        [0.3, target_internal_iptm, target_internal_iptm, 0.0],
    ])
    evaluation = SimpleNamespace(names=["ABCD"], chain_pair_scores={"ABCD": {
        "chain_pair_iptm": [matrix],
        "chain_ptm": [np.array([0.65, 0.95, 0.95, 0.95])],
    }})
    interface = chain_pair_iptm(evaluation, chain_a="A", chain_b=["B", "C", "D"])["ABCD"]
    ptm_a = binder_ptm(evaluation, binder_chain="A")["ABCD"]
    assert interface == pytest.approx(0.2)
    assert ptm_a == pytest.approx(0.65)

    expression = "iptm_A_BCD + 0.4 * ptm_A + R"
    engine = LossEngine({"loss_terms": {}}, validate_config=False)
    engine.setup(weighted_score_expression=expression, early_stopping=None,
                 acceptance_threshold=None, validate_config=False)
    # R is an arbitrary synthetic scalar here; its scientific definition is unchanged.
    scores = {"iptm_A_BCD": interface, "ptm_A": ptm_a, "R": 0.125}
    assert engine.weighted_score(scores, expression) == pytest.approx(0.2 + 0.4 * 0.65 + 0.125)
