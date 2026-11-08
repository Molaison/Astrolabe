"""Exercise the actual vendored solver on an independently enumerable Potts model."""
from __future__ import annotations

import ast
import itertools
import random
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def native():
    path = Path(__file__).resolve().parents[1] / "vendor/proton_pottsmpnn/foundry/models/mpnn/src/mpnn/inference_engines/potts_mpnn_ph.py"
    tree = ast.parse(path.read_text())
    names = {"PHDesignCriteria", "_greedy_energy_block",
             "_greedy_energy_block_with_network_pairs",
             "_aa_bias_value", "_aa_bias_block", "_mutation_budget_block",
             "_parent_vocab_mask", "_network_pair_specs",
             "_network_pair_match_value", "_network_pair_bias_block"}
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert len(nodes) == len(names)
    env = dict(torch=torch, np=np, F=torch.nn.functional, dataclass=dataclass, field=field,
               Dict=Dict, List=List, Optional=Optional, Any=Any, Sequence=Sequence,
               DEFAULT_DEP_MAP={"HIS-S": ["HIS-P"]}, DEFAULT_FORBIDDEN_TOKENS=[],
               ALL_METHODS=("greedy_energy_block", "block_descent"),
               WHOLE_CHAIN_METHODS=(), PLACEMENT_METHODS=("greedy_energy_block", "block_descent"),
               DICT_THREE_TO_ONE={"HIS": "H", "ALA": "A"}, __name__=__name__)
    env["_record"] = lambda *a, **k: None
    env["_block_mutation_tables"] = lambda ctx, valid, S, neigh, *a: (torch.zeros(len(neigh), ctx.V), None, None)
    env["_zscale"] = lambda *a: 1.0
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec", dont_inherit=True), env)
    return SimpleNamespace(**{name: env[name] for name in names})


class ToyScorer:
    @staticmethod
    def H_of(S):
        return float(0.3 * (S == 1).sum() + 0.5 * S[2] - 0.1 * S[0] * S[1])

    def cond_energy_at(self, S, i):
        return torch.tensor([self.H_of(torch.tensor([a if j == i else int(S[j]) for j in range(3)]))
                             for a in range(2)])

    @staticmethod
    def block_stability_potentials(S, block):
        unary = torch.zeros(len(block), 2)
        for bi, p in enumerate(block):
            unary[bi, 1] = 0.3 + (0.5 if p == 2 else 0.0)
            partner = 1 - p
            if p in (0, 1) and partner not in block:
                unary[bi, 1] -= 0.1 * S[partner]
        edges = []
        if 0 in block and 1 in block:
            m = torch.zeros(2, 2)
            m[1, 1] = -0.1
            edges.append((block.index(0), block.index(1), m))
        return unary, edges


def context():
    return SimpleNamespace(
        V=2, device="cpu", canonical_map=torch.arange(2),
        encoding=SimpleNamespace(idx_to_token={0: "ALA", 1: "HIS"}),
        token_aa=SimpleNamespace(res_id=np.arange(1, 4)), res_id_to_pos={1: 0, 2: 1, 3: 2},
        stability_scorer=lambda *a: ToyScorer(),
        region_mask=lambda names: torch.tensor([False, True, False]) if "core" in names else torch.ones(3, dtype=torch.bool),
        encode_initial_sequence=lambda s: torch.tensor([int(x == "H") for x in s]),
    )


def criteria(native, **kwargs):
    return native.PHDesignCriteria(method="greedy_energy_block", center_count=0,
                                   infill_scope="chain", temperature=0.0,
                                   seed_source="native", block_size=2, **kwargs)


def solve(native, start="AAA", positions=(0, 1, 2), **kwargs):
    return native._greedy_energy_block(
        context(), criteria(native, **kwargs), torch.ones(2, dtype=torch.bool), [], list(positions),
        torch.tensor([int(x == "H") for x in start]), random.Random(0))


def test_bias_introduces_histidines_without_seeds_or_pins(native):
    result = solve(native, aa_bias={"HIS": 0.4}, aa_bias_caps={"HIS": 2}, max_mutations=2)
    # Compare with an exhaustive independent oracle, not the solver's bias helper.
    states = [torch.tensor(s) for s in itertools.product(range(2), repeat=3) if sum(s) <= 2]
    expected = min(states, key=lambda s: ToyScorer.H_of(s) - 0.4 * min(int(s.sum()), 2))
    assert result.tolist() == expected.tolist() == [1, 1, 0]


def test_no_bias_preserves_the_unmodified_parent(native):
    assert solve(native).tolist() == [0, 0, 0]
    assert solve(native, aa_bias={"HIS": 0.0}, aa_bias_caps={"HIS": 2}).tolist() == [0, 0, 0]


def test_his_reward_saturates_and_does_not_pin_existing_histidines(native):
    assert solve(native, aa_bias={"HIS": 0.4}, aa_bias_caps={"HIS": 1}).sum() == 1
    assert solve(native, start="HHA", aa_bias={"HIS": -0.4}).tolist() == [0, 0, 0]


def test_joint_budget_is_enforced_inside_the_solver(native):
    assert solve(native, aa_bias={"HIS": 1.0}, max_mutations=1).sum() == 1
    assert solve(native, aa_bias={"HIS": 1.0}, max_mutations=2,
                 mutation_reference_sequence="AAA", max_total_mutations=1).sum() == 1


def test_fixed_sites_and_parent_reference_remain_protected(native):
    assert solve(native, positions=(0,), aa_bias={"HIS": 1.0}).tolist() == [1, 0, 0]
    with pytest.raises(ValueError, match="reference budget"):
        solve(native, start="HHA", mutation_reference_sequence="AAA", max_total_mutations=1)


def test_bias_only_applies_in_its_configured_region(native):
    assert solve(native, aa_bias={"HIS": 0.4}, aa_bias_region=["core"]).tolist() == [0, 1, 0]


@pytest.mark.parametrize("kwargs", [
    {"aa_bias": {"XXX": 1.0}}, {"aa_bias": {"HIS": float("nan")}},
    {"aa_bias": {"HIS": 1.0}, "aa_bias_caps": {"HIS": 0}},
    {"max_total_mutations": 2},
])
def test_invalid_bias_and_reference_fail_explicitly(native, kwargs):
    with pytest.raises(ValueError):
        criteria(native, **kwargs)
