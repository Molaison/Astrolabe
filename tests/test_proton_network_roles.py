"""Producer role contracts with synthetic genotypes; no model or pose loading."""

from types import SimpleNamespace

import numpy as np
import pytest

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import ProtomerTypes
from astrolabe.mutators import proton_network_geometry as geometry
from astrolabe.mutators import proton_pottsmpnn as adapter


def _pair(i, j, aa_j="LYS", quality=1.0):
    return dict(i=i, j=j, aa_i="HIS", aa_j=aa_j, quality=quality)


@pytest.fixture
def producer(monkeypatch):
    global_config.reset()
    proto = SimpleNamespace(
        name="P1", sequence="AAAA", type=ProtomerTypes.PROTEIN,
        mutable_mask=np.array([True, True, True, False]),
        tied_positions=[], ptm_dict={}, components=["S1"],
        components_slices=[slice(0, 4)],
    )
    queued = []

    def add_mutations(records, count):
        assert count == len(records)
        queued.extend(records)

    global_config.init_const("protomers", {"P1": proto})
    global_config.init_const("segments", {"S1": SimpleNamespace(mutations=[])})
    global_config.init_const("segment_manager", SimpleNamespace(add_mutations=add_mutations))
    global_config.init_const("exclude_AA", [])
    mutator = adapter.ProtonPottsMPNNMutator(
        sequence_model_runner=SimpleNamespace(close=lambda: None), network_only=True,
    )
    monkeypatch.setattr(mutator, "_mutation_budget", lambda: 2)
    try:
        yield mutator, proto, queued
    finally:
        mutator.shutdown()
        global_config.reset()


def _emit(mutator, proto, current, designed, preferences, runtime):
    proto.sequence = current
    tokens = {"A": "ALA", "H": "HIS-S", "K": "LYS", "R": "ARG"}
    output = {("P1",): {
        "feature_dict": dict(
            binder_protomer="P1", binder_chain="A", binder_sequence=current,
            fixed_positions=[3], criteria={**mutator.criteria,
                                          "network_pair_preferences": preferences},
        ),
        "selected": dict(canonical_sequence=designed,
                         extended_tokens=[tokens[aa] for aa in designed]),
    }}
    return mutator.prepare_mutation_output(output, {"P1": proto}, None, runtime)


@pytest.mark.parametrize("current,designed,partner,expected,changed", [
    ("AAAA", "HKAA", "LYS", {0: "H", 1: "K"}, [0, 1]),
    ("AKAA", "HKAA", "LYS", {0: "H", 1: "K"}, [0]),
    ("HAAA", "HKAA", "LYS", {0: "H", 1: "K"}, [1]),
    ("AHAA", "HHAA", "HIS", {0: "H", 1: "H"}, [0]),
    ("ARAA", "HRAA", "ARG", {0: "H", 1: "R"}, [0]),
])
def test_emits_both_roles_including_unchanged_partner(
    producer, current, designed, partner, expected, changed,
):
    mutator, proto, queued = producer
    runtime = {"unrelated": "keep"}
    result = _emit(mutator, proto, current, designed, [_pair(0, 1, partner)], runtime)
    telemetry = result["proton_pottsmpnn"]
    assert runtime == {"unrelated": "keep", "network_pair_roles": {"P1": expected}}
    assert telemetry["network_pair_roles"] == {"P1": expected}
    assert telemetry["oligos"][("P1",)]["declared_network_pair_roles"] == expected
    assert "not an engine pair-id claim" in telemetry["oligos"][("P1",)]["pair_identity_source"]
    assert [record[1][0] for record in queued] == changed
    assert proto.mutable_mask.tolist() == [True, True, True, False]


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("qualities,expected_partner", [((1.0, 2.0), 2), ((2.0, 2.0), 1)])
def test_declared_canonical_match_is_deterministic(producer, reverse, qualities, expected_partner):
    mutator, proto, _ = producer
    preferences = [_pair(0, 1, quality=qualities[0]), _pair(0, 2, quality=qualities[1])]
    if reverse:
        preferences.reverse()
    runtime = {}
    _emit(mutator, proto, "AKKA", "HKKA", preferences, runtime)
    assert runtime["network_pair_roles"] == {"P1": {0: "H", expected_partner: "K"}}


def test_next_prepare_clears_roles_before_selection_without_permanent_locks(producer, monkeypatch):
    mutator, proto, queued = producer
    runtime = {"unrelated": "keep"}
    _emit(mutator, proto, "AKAA", "HKAA", [_pair(0, 1)], runtime)
    assert runtime["network_pair_roles"] == {"P1": {0: "H", 1: "K"}}
    queued.clear()  # Simulate accepting the previous proposal, then starting a new step.
    proto.sequence = "HKAA"
    criteria_before = dict(mutator.criteria)

    def select(*args, **kwargs):
        assert "network_pair_roles" not in kwargs["runtime_config"]
        return {}

    def build(path, chain, sequence, mutable_positions, parameters, **kwargs):
        assert sequence == "HKAA"
        assert mutable_positions == [0, 1, 2]
        assert kwargs["fallback_to_best"] is True
        return [_pair(0, 2)], {}

    monkeypatch.setattr(adapter, "prepare_exclude_sites", select)
    monkeypatch.setattr(geometry, "build_network_pair_preferences", build)
    scores = SimpleNamespace(names=[("P1",)], complex_paths={("P1",): "/synthetic/context.pdb"})
    for _ in range(2):
        runtime["network_pair_roles"] = {"P1": {0: "H", 1: "K"}}
        features = mutator.prepare_mutation_input({"P1": proto}, scores, runtime)[("P1",)]
        assert features["fixed_positions"] == [3]
        assert runtime == {"unrelated": "keep"}
        assert mutator.criteria == criteria_before
        assert proto.mutable_mask.tolist() == [True, True, True, False]


def test_noop_output_replaces_stale_roles_with_empty_map(producer):
    mutator, proto, queued = producer
    runtime = {"network_pair_roles": {"P1": {0: "H", 1: "K"}}}
    _emit(mutator, proto, "HKAA", "HKAA", [_pair(0, 1)], runtime)
    assert runtime["network_pair_roles"] == {}
    assert mutator.last_mutation_telemetry["network_pair_roles"] == {}
    assert queued == []
