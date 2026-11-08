from __future__ import annotations

import ast
from dataclasses import dataclass, field
import math
from pathlib import Path
import random
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pytest

torch = pytest.importorskip("torch")


TOKENS = ("ALA", "HIS", "LYS", "ARG", "GLY")
ONE_TO_TOKEN = {"A": 0, "H": 1, "K": 2, "R": 3, "G": 4}


@pytest.fixture
def native():
    path = (
        Path(__file__).resolve().parents[1]
        / "vendor/proton_pottsmpnn/foundry/models/mpnn/src/mpnn/"
        "inference_engines/potts_mpnn_ph.py"
    )
    tree = ast.parse(path.read_text())
    names = {
        "PHDesignCriteria",
        "_greedy_energy_block",
        "_greedy_energy_block_with_network_pairs",
        "_aa_bias_value",
        "_aa_bias_block",
        "_mutation_budget_block",
        "_parent_vocab_mask",
        "_network_pair_specs",
        "_network_pair_match_value",
        "_network_pair_bias_block",
    }
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        and node.name in names
    ]
    assert len(nodes) == len(names)
    env = dict(
        torch=torch,
        np=np,
        F=torch.nn.functional,
        dataclass=dataclass,
        field=field,
        Dict=Dict,
        List=List,
        Optional=Optional,
        Any=Any,
        Sequence=Sequence,
        math=math,
        DEFAULT_DEP_MAP={"HIS-S": ["HIS-P"]},
        DEFAULT_FORBIDDEN_TOKENS=[],
        ALL_METHODS=("greedy_energy_block", "block_descent"),
        WHOLE_CHAIN_METHODS=(),
        PLACEMENT_METHODS=("greedy_energy_block", "block_descent"),
        DICT_THREE_TO_ONE={
            "ALA": "A",
            "HIS": "H",
            "LYS": "K",
            "ARG": "R",
            "GLY": "G",
        },
        __name__=__name__,
    )
    env["_record"] = lambda *args, **kwargs: None
    env["_block_mutation_tables"] = (
        lambda ctx, valid, S, neigh, *args: (
            torch.zeros(len(neigh), ctx.V),
            None,
            None,
        )
    )
    env["_zscale"] = lambda *args: 1.0
    exec(
        compile(
            ast.Module(body=nodes, type_ignores=[]),
            str(path),
            "exec",
            dont_inherit=True,
        ),
        env,
    )
    return SimpleNamespace(**{name: env[name] for name in names})


class ToyScorer:
    def __init__(self, costs: Sequence[float] = (0.0, 1.0, 1.0, 1.0, 0.0)):
        self.costs = torch.tensor(costs, dtype=torch.float32)

    def H_of(self, S):
        return float(self.costs[S.long()].sum())

    def cond_energy_at(self, S, _position):
        return self.costs.clone()

    def block_stability_potentials(self, S, block):
        return torch.stack([self.costs.clone() for _ in block]), []


def make_context(
    *,
    chain_a: Sequence[bool],
    sequence: str,
    costs: Sequence[float] = (0.0, 1.0, 1.0, 1.0, 0.0),
    include_chain_a_t: bool = True,
):
    chain_a_t = torch.tensor(chain_a, dtype=torch.bool)
    binder_positions = chain_a_t.nonzero(as_tuple=False).reshape(-1).tolist()
    assert len(sequence) == len(binder_positions)
    tokens = [ONE_TO_TOKEN["G"]] * len(chain_a)
    for position, aa in zip(binder_positions, sequence):
        tokens[int(position)] = ONE_TO_TOKEN[aa]
    S_native = torch.tensor(tokens, dtype=torch.long)
    scorer = ToyScorer(costs)

    def encode_initial_sequence(value: str):
        result = S_native.clone()
        if len(value) != len(binder_positions):
            raise ValueError("seed length mismatch")
        for position, aa in zip(binder_positions, value):
            result[int(position)] = ONE_TO_TOKEN[aa]
        return result

    values = dict(
        V=len(TOKENS),
        device="cpu",
        canonical_map=torch.arange(len(TOKENS)),
        encoding=SimpleNamespace(
            idx_to_token={index: token for index, token in enumerate(TOKENS)}
        ),
        token_aa=SimpleNamespace(res_id=np.arange(1, len(chain_a) + 1)),
        res_id_to_pos={
            index + 1: index for index in range(len(chain_a))
        },
        stability_scorer=lambda *args: scorer,
        region_mask=lambda _names: chain_a_t.clone(),
        encode_initial_sequence=encode_initial_sequence,
        S_native=S_native,
    )
    if include_chain_a_t:
        values["chainA_t"] = chain_a_t
    return SimpleNamespace(**values)


def preference(i: int, j: int, *, aa_j: str = "LYS", quality: float = 1.0):
    return {
        "i": i,
        "j": j,
        "aa_i": "HIS",
        "aa_j": aa_j,
        "quality": quality,
    }


def criteria(native, preferences, *, bias: float = 3.0, **kwargs):
    temperature = kwargs.pop("temperature", 0.0)
    return native.PHDesignCriteria(
        method="greedy_energy_block",
        center_count=0,
        infill_scope="chain",
        temperature=temperature,
        seed_source="native",
        block_size=2,
        max_mutations=4,
        cv_patience=1,
        cv_max=8,
        network_pair_bias=bias,
        network_pair_preferences=list(preferences),
        **kwargs,
    )


def solve(native, ctx, preferences, *, valid=None, **kwargs):
    chain_a_t = getattr(ctx, "chainA_t", torch.zeros(3, dtype=torch.bool))
    neigh = chain_a_t.nonzero(as_tuple=False).reshape(-1).tolist()
    if valid is None:
        valid = torch.ones(ctx.V, dtype=torch.bool)
    return native._greedy_energy_block(
        ctx,
        criteria(native, preferences, **kwargs),
        valid,
        [],
        neigh,
        ctx.S_native,
        random.Random(0),
    )


def test_zero_his_joint_pair_beats_two_adverse_singles_without_pins(native):
    ctx = make_context(
        chain_a=[False, True, True, False],
        sequence="AA",
    )
    prefs = [preference(0, 1)]
    start_objective = ToyScorer().H_of(ctx.S_native)
    h_only = ToyScorer().H_of(torch.tensor([4, 1, 0, 4]))
    k_only = ToyScorer().H_of(torch.tensor([4, 0, 2, 4]))
    joint_raw = ToyScorer().H_of(torch.tensor([4, 1, 2, 4]))
    assert h_only > start_objective
    assert k_only > start_objective
    assert joint_raw - 3.0 < start_objective

    result = solve(native, ctx, prefs)

    assert result.tolist() == [4, 1, 2, 4]
    assert ToyScorer().H_of(result) == pytest.approx(2.0)  # raw H excludes bias

    retained = solve(
        native,
        make_context(chain_a=[False, True, True, False], sequence="HK"),
        prefs,
    )
    assert retained.tolist() == [4, 1, 2, 4]


def test_empty_preferences_degrade_without_reading_chain_mapping(native):
    ctx = make_context(
        chain_a=[False, True, True, False],
        sequence="AA",
        include_chain_a_t=False,
    )
    original = ctx.S_native.clone()
    result = native._greedy_energy_block(
        ctx,
        criteria(native, [], bias=2.0),
        torch.ones(ctx.V, dtype=torch.bool),
        [],
        [1, 2],
        ctx.S_native,
        random.Random(0),
    )
    assert result.tolist() == original.tolist()


def test_pair_tensor_uses_global_max_and_exact_outside_partner(native):
    ctx = make_context(
        chain_a=[False, False, True, True, False],
        sequence="KA",
    )
    prefs = [preference(1, 0, quality=0.8)]  # reversed roles: i=H, j=K
    conf = criteria(native, prefs, bias=1.0)
    specs = native._network_pair_specs(ctx, conf, [2, 3])
    assert specs[0][:2] == (3, 2)

    S = ctx.S_native.clone()
    tensor = native._network_pair_bias_block(ctx, S, [3], specs, 1.0)
    assert tensor[1].item() == pytest.approx(-0.8)
    assert tensor[0].item() == pytest.approx(0.0)

    S[3] = 1
    tensor = native._network_pair_bias_block(ctx, S, [2], specs, 1.0)
    assert tensor[2].item() == pytest.approx(-0.8)

    global_ctx = make_context(
        chain_a=[False, True, True, False],
        sequence="HK",
    )
    global_prefs = [
        preference(0, 1, quality=0.4),
        preference(0, 1, quality=0.9),
    ]
    global_conf = criteria(native, global_prefs, bias=2.0)
    global_specs = native._network_pair_specs(
        global_ctx, global_conf, [1, 2]
    )
    tensor = native._network_pair_bias_block(
        global_ctx, global_ctx.S_native, [1, 2], global_specs, 2.0
    )
    assert tensor[1, 2].item() == pytest.approx(-1.8)


def test_fixed_endpoint_can_be_rewarded_only_when_it_already_has_role(native):
    mutable_ctx = make_context(
        chain_a=[False, True, True, False],
        sequence="AK",
    )
    result = solve(native, mutable_ctx, [preference(0, 1)])
    assert result.tolist() == [4, 1, 2, 4]

    mismatched = make_context(
        chain_a=[False, True, True, False],
        sequence="AR",
    )
    with pytest.raises(ValueError, match="fixed endpoint"):
        native._greedy_energy_block(
            mismatched,
            criteria(native, [preference(0, 1)]),
            torch.ones(mismatched.V, dtype=torch.bool),
            [],
            [1],
            mismatched.S_native,
            random.Random(0),
        )

    fixed_only = make_context(
        chain_a=[False, True, True, False],
        sequence="AK",
    )
    neigh = []
    with pytest.raises(ValueError, match="no mutable endpoint"):
        native._network_pair_specs(
            fixed_only, criteria(native, [preference(0, 1)]), neigh
        )


def test_forbidden_role_and_budgets_are_enforced(native):
    ctx = make_context(
        chain_a=[False, True, True, False],
        sequence="AA",
        costs=(0.0, 0.0, 0.0, 0.0, 0.0),
    )
    forbidden_valid = torch.tensor([True, False, True, True, True])
    result = solve(
        native,
        ctx,
        [preference(0, 1)],
        valid=forbidden_valid,
    )
    assert result.tolist() == ctx.S_native.tolist()

    budgeted = solve(
        native,
        make_context(chain_a=[False, True, True, False], sequence="AA"),
        [preference(0, 1)],
        mutation_reference_sequence="AA",
        max_total_mutations=1,
    )
    assert budgeted.tolist() == [4, 0, 0, 4]


def test_pair_step_budget_limits_disjoint_preferences(native):
    ctx = make_context(
        chain_a=[False, True, True, True, True, False],
        sequence="AAAA",
    )
    result = solve(
        native,
        ctx,
        [preference(0, 1), preference(2, 3)],
        network_pair_max_steps=1,
    )
    binder = result[[1, 2, 3, 4]].tolist()
    matched = sum(
        binder[start : start + 2] == [1, 2]
        for start in (0, 2)
    )
    assert matched == 1


def test_mapping_rejects_out_of_range_preference(native):
    ctx = make_context(
        chain_a=[False, True, True, False],
        sequence="AA",
    )
    with pytest.raises(ValueError, match="outside binder length"):
        solve(native, ctx, [preference(0, 2)])


@pytest.mark.parametrize(
    "preference_override",
    [
        {"i": 0, "j": 1, "aa_i": "HIS", "aa_j": "ASP", "quality": 1.0},
        {"i": 0, "j": 1, "aa_i": "HIS", "aa_j": "LYS", "quality": float("nan")},
        {"i": 0, "j": 1, "aa_i": "HIS", "aa_j": "LYS", "quality": 1.1},
        {"i": 0, "j": 0, "aa_i": "HIS", "aa_j": "LYS", "quality": 1.0},
    ],
)
def test_invalid_preferences_fail_in_criteria(native, preference_override):
    with pytest.raises(ValueError):
        criteria(native, [preference_override])


def test_pair_settings_require_centre_free_and_positive_step_budget(native):
    with pytest.raises(ValueError, match="network_pair_max_steps"):
        criteria(native, [preference(0, 1)], network_pair_max_steps=0)
    with pytest.raises(ValueError, match="centre-free"):
        native.PHDesignCriteria(
            method="block_descent",
            center_count=1,
            network_pair_bias=1.0,
            network_pair_preferences=[preference(0, 1)],
        )


def test_temperature_pair_sampling_masks_non_improving_assignments(
    native, monkeypatch
):
    ctx = make_context(
        chain_a=[False, True, True, False],
        sequence="AA",
    )
    seen = []

    def choose(probabilities, _draws):
        seen.append(probabilities.clone())
        return int(torch.argmax(probabilities))

    monkeypatch.setattr(torch, "multinomial", choose)
    result = solve(
        native,
        ctx,
        [preference(0, 1)],
        temperature=0.5,
    )
    assert result.tolist() == [4, 1, 2, 4]
    assert seen
    assert float(seen[0].sum()) == pytest.approx(1.0)
    assert int((seen[0] == 0).sum()) > 0
