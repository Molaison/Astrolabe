from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import Protomer, SegmentManager
from astrolabe.mutation_budget import count_mutable_sites, current_mutation_count, resolve_mutation_num
from astrolabe.mutators.base import Mutator
from astrolabe.stage_manager import StageManager


@pytest.mark.parametrize("value", [True, False, -1, -0.1, 1.6, "1.6", "-10%", "101%", "NaN", "inf%", "", {}, []])
def test_invalid_budget_fails_instead_of_rounding_or_mutating_one_site(value):
    with pytest.raises((TypeError, ValueError), match="mutation_num"):
        resolve_mutation_num(value, protomer_pool={})


@pytest.mark.parametrize("value, expected", [(0, 0), (4, 4), (4.0, 4), ("4", 4), ("4.0", 4), ("0%", 0), ("50%", 4), ("1%", 1), ("all", 8)])
def test_count_and_percentage_contract(value, expected):
    assert resolve_mutation_num(value, protomer_pool={"A": SimpleNamespace(mutable_length=8)}) == expected


@pytest.mark.parametrize("value", ["0%", "1%", "50%", "100%", "all"])
def test_empty_pool_has_zero_percentage_budget(value):
    assert resolve_mutation_num(value, protomer_pool={}) == 0
    assert resolve_mutation_num(value, protomer_pool={"A": SimpleNamespace(mutable_length=0)}) == 0


@pytest.fixture
def shared_runtime():
    global_config.reset()
    for name, value in (
        ("segments", {}), ("named_files", {}), ("predefined_statement", {}),
        ("exclude_AA", []), ("segment_manager", SegmentManager()),
    ):
        global_config.init_const(name, value)
    first = Protomer("A", "ACDEFGHI", "p")
    second = Protomer("B", "A_1", "p")
    protomers = {"A": first, "B": second}
    global_config.init_const("protomers", protomers)
    global_config.init_const("mutable_protomers", protomers)
    loss = SimpleNamespace(setup=lambda *args, **kwargs: None)
    manager = StageManager({
        "default": {"oligo": ["AB"], "mutation_num": "50%", "mutators": ["MCMC"],
                    "evaluator": "sequence", "weighted_score_expression": "0"},
        "later": {"_entrance": "True", "mutation_num": 3},
    }, loss_computer=loss)
    global_config.init_const("stage_manager", manager)
    yield first, second, manager
    global_config.reset()


def test_shared_segment_budget_counts_independent_sites(shared_runtime):
    first, second, _ = shared_runtime
    assert first.component_ids == second.component_ids
    assert count_mutable_sites(global_config.protomers, segments=global_config.segments) == 8
    assert current_mutation_count() == 4


def test_lock_and_length_changes_refresh_without_stage_transition(shared_runtime):
    first, second, manager = shared_runtime
    assert global_config.mutation_num == 4
    first.add_locked_positions([0, 1, 2, 3])
    assert Mutator().n_mutations() == 2
    assert global_config.mutation_num == 2
    global_config.segments["A_1"].sequence = "ACDEFGHIKLMN"
    global_config.segment_manager.refresh_protomers()
    assert first.length == second.length == 12
    assert current_mutation_count() == 4
    assert manager.get_current_stage() == "default"


def test_ptm_and_fixed_sites_do_not_inflate_percentage(shared_runtime):
    first, _, _ = shared_runtime
    global_config.segments["A_1"].ptm_dict = {2: "MSE"}
    global_config.segment_manager.refresh_protomers()
    first.add_locked_positions([0])
    assert count_mutable_sites(global_config.protomers, segments=global_config.segments) == 6
    global_config.segments["A_1"].is_fixed = True
    global_config.segment_manager.refresh_protomers()
    assert current_mutation_count() == 0


def test_stage_switch_weight_and_selected_site_cap_share_resolver(shared_runtime):
    _, _, manager = shared_runtime
    mutator = Mutator()
    mutator.mutation_weight = 0.5
    assert mutator.n_mutations() == 2
    assert mutator.n_mutations(maximum=1) == 1
    manager.set_stage("later")
    assert mutator.n_mutations() == 2
    manager.set_stage("default")
    assert mutator.n_mutations() == 2


def test_budget_resolution_does_not_consume_rng(shared_runtime):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    assert current_mutation_count() == 4
    assert random.getstate() == python_state
    after = np.random.get_state()
    assert after[0] == numpy_state[0]
    np.testing.assert_array_equal(after[1], numpy_state[1])
    assert after[2:] == numpy_state[2:]
