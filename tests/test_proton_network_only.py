"""Network-only contracts, using AST-extracted solvers and inert runners.

The reused solver vocabulary is deliberately a toy: its HIS token does not
represent native HIS-S. Native microstate restrictions are checked separately
on the adapter criteria, without loading a checkpoint or geometry backend.
"""

from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import random
import sys

import pytest


_fixture_name = "_proton_network_only_pair_fixture"
_spec = importlib.util.spec_from_file_location(
    _fixture_name, Path(__file__).with_name("test_proton_network_pair_bias.py")
)
assert _spec is not None and _spec.loader is not None
_pair = importlib.util.module_from_spec(_spec)
sys.modules[_fixture_name] = _pair
_spec.loader.exec_module(_pair)

native = _pair.native
torch = _pair.torch
make_context = _pair.make_context
preference = _pair.preference
criteria = _pair.criteria
solve = _pair.solve


class _PositionScorer:
    """Independent site energies, allowing a stronger off-network decoy."""

    def __init__(self, rows):
        self.rows = torch.tensor(rows, dtype=torch.float32)

    def H_of(self, sequence):
        return float(self.rows[torch.arange(len(sequence)), sequence.long()].sum())

    def cond_energy_at(self, _sequence, position):
        return self.rows[position].clone()

    def block_stability_potentials(self, _sequence, block):
        return self.rows[list(block)].clone(), []


@pytest.mark.parametrize("partner", ["HIS", "ARG", "LYS"])
def test_zero_his_changes_only_the_geometry_pair(native, partner):
    ctx = make_context(chain_a=[False, True, True, True, False], sequence="AAA")
    scorer = _PositionScorer(
        [[0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0],
         [0, 1, 1, 1, -100], [0, 0, 0, 0, 0]]
    )
    ctx.stability_scorer = lambda *_: scorer
    original = ctx.S_native.clone()

    result = solve(
        native, ctx, [preference(0, 1, aa_j=partner)], bias=0, network_pair_only=True
    )

    assert result.tolist() == [4, 1, _pair.TOKENS.index(partner), 0, 4]
    assert ctx.S_native.tolist() == original.tolist()


@pytest.mark.parametrize("partner", ["HIS", "ARG", "LYS"])
@pytest.mark.parametrize("temperature", [0.0, 0.5])
def test_quality_zero_fallback_still_proposes_supported_pair(native, partner, temperature):
    ctx = make_context(
        chain_a=[False, True, True, True, False], sequence="AAA",
        costs=(0, 20, 20, 20, -100),
    )
    original = ctx.S_native.clone()
    prefs = [preference(0, 1, aa_j=partner, quality=0.0)]
    result = solve(
        native, ctx, prefs, bias=0, network_pair_only=True, temperature=temperature
    )
    assert result.tolist() == [4, 1, _pair.TOKENS.index(partner), 0, 4]
    assert ctx.S_native.tolist() == original.tolist()
    # The unsupported Gly decoy is energetically better; the legal fallback
    # still changes exactly the supplied pair despite its adverse raw energy.
    scorer = ctx.stability_scorer()
    assert scorer.H_of(result) - scorer.H_of(original) == pytest.approx(40.0)
    conf = criteria(native, prefs, bias=0, network_pair_only=True)
    specs = native._network_pair_specs(ctx, conf, [1, 2, 3])
    assert native._network_pair_match_value(result, specs) == 0.0
    reward = native._network_pair_bias_block(ctx, result, [1, 2], specs, 0.0)
    assert torch.count_nonzero(reward).item() == 0


def test_empty_preferences_never_fall_back_to_stability(native):
    ctx = make_context(
        chain_a=[False, True, True, False], sequence="AA",
        costs=(0, 1, 1, 1, -100), include_chain_a_t=False,
    )
    result = native._greedy_energy_block(
        ctx, criteria(native, [], bias=0, network_pair_only=True),
        torch.ones(ctx.V, dtype=torch.bool), [], [1, 2], ctx.S_native,
        random.Random(0),
    )
    assert result.tolist() == ctx.S_native.tolist()


@pytest.mark.parametrize("reason", ["forbidden_his", "forbidden_partner"])
def test_no_eligible_pair_is_noop_despite_stability_gain(native, reason):
    ctx = make_context(
        chain_a=[False, True, True, True, False], sequence="AAA",
        costs=(0, 20, 20, 20, -100),
    )
    valid = torch.ones(ctx.V, dtype=torch.bool)
    valid[1 if reason == "forbidden_his" else 2] = False
    result = solve(
        native, ctx, [preference(0, 1)], valid=valid, bias=0, network_pair_only=True
    )
    assert result.tolist() == ctx.S_native.tolist()


def test_worse_raw_potts_energy_still_proposes_geometric_hk_without_bias(native):
    # Toy HIS denotes the role only; this does not test native microstates.
    ctx = make_context(
        chain_a=[False, True, True, False], sequence="AA", costs=(0, 20, 20, 20, 0)
    )
    scorer = ctx.stability_scorer()
    parent_energy = scorer.H_of(ctx.S_native)
    result = solve(native, ctx, [preference(0, 1)], bias=0, network_pair_only=True)
    assert result.tolist() == [4, 1, 2, 4]
    assert scorer.H_of(result) - parent_energy == pytest.approx(40.0)
    # A proposal is not AF3 acceptance: the outer neutral-preference gate is separate.


@pytest.mark.parametrize("lower_energy_quality", [0.1, 0.0])
def test_raw_delta_h_selects_pair_without_geometry_quality_reward(native, lower_energy_quality):
    ctx = make_context(chain_a=[True, True, True, True], sequence="AAAA")
    scorer = _PositionScorer([
        [0, 3, 3, 3, -100], [0, 3, 3, 3, -100],
        [0, 1, 1, 1, -100], [0, 1, 1, 1, -100],
    ])
    ctx.stability_scorer = lambda *_: scorer
    result = solve(
        native, ctx,
        [preference(0, 1, quality=1.0),
         preference(2, 3, aa_j="ARG", quality=lower_energy_quality)],
        bias=0, network_pair_only=True, temperature=0,
    )
    assert result.tolist() == [0, 0, 1, 3]
    assert scorer.H_of(result) - scorer.H_of(ctx.S_native) == pytest.approx(2.0)


def test_one_joint_pair_only_even_when_both_pairs_lower_energy(native):
    ctx = make_context(
        chain_a=[False, True, True, True, True, False], sequence="AAAA",
        costs=(0, -1, -1, -1, 0),
    )
    result = solve(
        native, ctx, [preference(0, 1), preference(2, 3)],
        bias=0, network_pair_only=True, network_pair_max_steps=12,
    )
    # Either geometry candidate is valid, but their edits cannot accumulate.
    assert result.tolist() in ([4, 1, 2, 0, 0, 4], [4, 0, 0, 1, 2, 4])


@pytest.mark.parametrize("temperature", [0.0, 0.5])
def test_pair_endpoints_cannot_take_unrelated_low_energy_gly(native, temperature):
    ctx = make_context(
        chain_a=[False, True, True, False], sequence="AA",
        costs=(0, 1, 1, 1, -100),
    )
    result = solve(
        native, ctx, [preference(0, 1)], bias=0, network_pair_only=True,
        temperature=temperature,
    )
    assert result.tolist() == [4, 1, 2, 4]


def test_partner_must_match_an_actual_preference_not_just_any_cation(native):
    ctx = make_context(
        chain_a=[False, True, True, False], sequence="AA",
        costs=(0, 1, 1, -100, 0),
    )
    result = solve(native, ctx, [preference(0, 1)], bias=0, network_pair_only=True)
    assert result.tolist() == [4, 1, 2, 4]


def test_roles_cannot_be_mixed_between_different_geometry_preferences(native):
    ctx = make_context(chain_a=[True, True, True], sequence="AAA")
    scorer = _PositionScorer([[0, 1, 1, 1, 0], [0, 1, -20, 1, 0],
                             [0, 1, 1, -20, 0]])
    ctx.stability_scorer = lambda *_: scorer
    result = solve(
        native, ctx, [preference(0, 1), preference(0, 2, aa_j="ARG")],
        bias=0, network_pair_only=True,
    )
    assert result.tolist() in ([1, 2, 0], [1, 0, 3])


@pytest.mark.parametrize("quality", [1.0, 0.0])
def test_fixed_matching_partner_is_retained_and_only_mutable_endpoint_changes(native, quality):
    ctx = make_context(
        chain_a=[False, True, True, False], sequence="AK",
        costs=(0, 1, 1, 1, -100),
    )
    conf = replace(
        criteria(native, [preference(0, 1, quality=quality)], bias=0, network_pair_only=True),
        max_mutations=1,
    )
    result = native._greedy_energy_block(
        ctx, conf, torch.ones(ctx.V, dtype=torch.bool), [], [1],
        ctx.S_native, random.Random(0),
    )
    assert result.tolist() == [4, 1, 2, 4]


@pytest.mark.parametrize("budget_kind", ["step", "total"])
@pytest.mark.parametrize("quality", [1.0, 0.0])
def test_budget_masks_whole_pair_instead_of_applying_one_edit(native, budget_kind, quality):
    ctx = make_context(chain_a=[False, True, True, False], sequence="AA")
    conf = criteria(native, [preference(0, 1, quality=quality)], bias=0, network_pair_only=True)
    if budget_kind == "step":
        conf = replace(conf, max_mutations=1)
    else:
        conf = replace(conf, mutation_reference_sequence="AA", max_total_mutations=1)
    result = native._greedy_energy_block(
        ctx, conf, torch.ones(ctx.V, dtype=torch.bool), [], [1, 2],
        ctx.S_native, random.Random(0),
    )
    assert result.tolist() == ctx.S_native.tolist()


def test_his_position_can_change_in_next_round(native):
    first_ctx = make_context(chain_a=[False, True, True, False], sequence="AA")
    first = solve(native, first_ctx, [preference(0, 1)], bias=0, network_pair_only=True)
    assert first.tolist() == [4, 1, 2, 4]

    # The next round's geometry reverses the roles; the old His is not pinned.
    second_ctx = make_context(chain_a=[False, True, True, False], sequence="HK")
    scorer = _PositionScorer([[0, 0, 0, 0, 0], [0, 1, 0, 1, 0],
                             [0, 0, 1, 1, 0], [0, 0, 0, 0, 0]])
    second_ctx.stability_scorer = lambda *_: scorer
    second = solve(
        native, second_ctx, [preference(1, 0)], bias=0, network_pair_only=True
    )
    assert second.tolist() == [4, 2, 1, 4]


def test_default_false_preserves_ordinary_stability_path(native):
    assert criteria(native, []).network_pair_only is False
    ctx = make_context(
        chain_a=[False, True, True, False], sequence="AA", costs=(0, 1, 1, 1, -10)
    )
    result = solve(native, ctx, [])
    assert result.tolist() == [4, 4, 4, 4]


def test_pair_only_criteria_accept_zero_bias(native):
    conf = criteria(native, [preference(0, 1)], bias=0, network_pair_only=True)
    assert conf.network_pair_only is True
    assert conf.network_pair_bias == 0
    assert not conf.aa_bias


def test_pair_only_criteria_reject_aa_reward(native):
    with pytest.raises(ValueError):
        criteria(
            native, [preference(0, 1)], bias=0, network_pair_only=True,
            aa_bias={"HIS": 1.0},
        )


def test_pair_only_criteria_require_center_free_even_with_zero_bias(native):
    with pytest.raises(ValueError):
        native.PHDesignCriteria(
            method="greedy_energy_block", center_count=1, infill_scope="chain",
            network_pair_only=True, network_pair_bias=0,
            network_pair_preferences=[preference(0, 1)],
        )


class _CloseableRunner:
    def __init__(self, config=None):
        self.config = config
        self.closed = False

    def close(self):
        self.closed = True


def _assert_network_presets(mutator):
    conf = mutator.criteria
    assert conf["method"] == "greedy_energy_block"
    assert conf["center_count"] == 0
    assert not conf.get("center_types")
    assert not conf.get("explicit_centers")
    assert conf["infill_scope"] == "chain"
    assert conf["seed_source"] == "native"
    assert conf["backend"] == "potts"
    assert conf["selective"] is False
    assert conf["combined_lambda"] == 0
    assert conf["global_weight"] == 0
    assert not conf.get("aa_bias")
    assert not conf.get("aa_bias_caps")
    assert conf["network_pair_only"] is True
    assert conf["network_pair_bias"] == 0
    assert conf["temperature"] == pytest.approx(0.1)
    assert mutator.network_proposal.enabled
    forbidden = set(conf["forbidden_tokens"])
    assert "HIS-P" in forbidden
    assert "HIS-S" not in forbidden
    assert {"HIS", "HIS-A", "HIS-D"} <= forbidden


def test_adapter_network_only_sets_complete_presets_without_user_details():
    from astrolabe.mutators.proton_pottsmpnn import ProtonPottsMPNNMutator

    runner = _CloseableRunner()
    mutator = ProtonPottsMPNNMutator(sequence_model_runner=runner, network_only=True)
    _assert_network_presets(mutator)
    assert mutator.runner is runner


def test_network_only_clears_both_biases_and_fixes_temperature():
    from astrolabe.mutators.proton_pottsmpnn import ProtonPottsMPNNMutator

    mutator = ProtonPottsMPNNMutator(
        sequence_model_runner=_CloseableRunner(), network_only=True,
        criteria={"aa_bias": {"HIS": 99}, "aa_bias_caps": {"HIS": 4},
                  "network_pair_bias": 8, "temperature": 0.7},
    )
    _assert_network_presets(mutator)


@pytest.mark.parametrize(
    ("current", "designed"),
    [
        pytest.param("AAAA", "HKRA", id="more-than-two-edits"),
        pytest.param("HKAA", "HKGA", id="edit-outside-matching-pair"),
        pytest.param("AAAA", "HRAA", id="wrong-cation-role"),
        pytest.param("AAAA", "HGAA", id="unrelated-aa-at-pair-endpoint"),
    ],
)
def test_output_guard_rejects_edits_outside_one_exact_geometry_pair(
    monkeypatch, current, designed
):
    from types import SimpleNamespace

    from astrolabe.core import global_config
    from astrolabe.mutators.proton_pottsmpnn import ProtonPottsMPNNMutator

    mutator = ProtonPottsMPNNMutator(
        sequence_model_runner=_CloseableRunner(), network_only=True
    )
    # Keep the ordinary budget permissive so the independent pair guard rejects.
    monkeypatch.setattr(mutator, "_mutation_budget", lambda: 8)
    tokens = {"A": "ALA", "H": "HIS-S", "K": "LYS", "R": "ARG", "G": "GLY"}
    features = {
        "binder_protomer": "P1", "binder_chain": "A",
        "binder_sequence": current, "fixed_positions": [],
        "criteria": {**mutator.criteria,
                     "network_pair_preferences": [preference(0, 1)]},
    }
    selected = {
        "canonical_sequence": designed,
        "extended_tokens": [tokens[aa] for aa in designed],
        "final_potts_energy": -1.0,
    }
    output = {("P1",): {"feature_dict": features, "selected": selected}}
    global_config.reset()
    global_config.init_const("protomers", {"P1": SimpleNamespace(name="P1")})
    try:
        with pytest.raises(ValueError, match="network_only.*supported.*pair"):
            mutator.prepare_mutation_output(
                output, global_config.protomers, evaluation_scores=None,
                runtime_config=None,
            )
        assert mutator.last_mutation_telemetry["oligos"] == {}
        assert selected["canonical_sequence"] == designed
    finally:
        global_config.reset()


@pytest.mark.parametrize("location", ["default", "term"])
def test_factory_routes_network_only_to_mutator_not_runner(monkeypatch, tmp_path, location):
    from omegaconf import OmegaConf
    from astrolabe.mutators.base import build_mutators
    from astrolabe.mutators.wrappers import proton_pottsmpnn_runner

    calls = []

    class Runner(_CloseableRunner):
        def __init__(self, config):
            super().__init__(config)
            calls.append(config)
            assert "network_only" not in config
            assert "network_proposal" not in config

    monkeypatch.setattr(proton_pottsmpnn_runner, "ProtonPottsMPNNRunner", Runner)
    config = OmegaConf.create({
        "ProtonPottsMPNN": {"python_bin": "/not/started/python", "device": "cpu"},
        "mutator_terms": {"network": {"_type": "proton_pottsmpnn"}},
    })
    target = config.ProtonPottsMPNN if location == "default" else config.mutator_terms.network
    target.network_only = True
    mutator = build_mutators(config, tmp_path / "raw")["network"]
    assert len(calls) == 1
    _assert_network_presets(mutator)


@pytest.mark.parametrize(
    ("gap", "unpaired", "accepted"),
    [(1.0, 0, True), (-1.0, 0, False), (0.0, 0, False), (1.0, 1, False)],
)
def test_neutral_preference_gate_and_score_direction_without_model(gap, unpaired, accepted):
    from astrolabe.losses.base import LossEngine

    engine = LossEngine({"loss_terms": {}}, validate_config=False)
    expression = "ptm + motif + neutral_preference"
    engine.setup(
        weighted_score_expression=expression, early_stopping=None,
        acceptance_threshold=None,
        acceptance_gate="neutral_preference > 0 and unpaired_his == 0",
        validate_config=False,
    )
    scores = {"ptm": 0.5, "motif": 0.2, "neutral_preference": gap,
              "unpaired_his": unpaired}
    assert engine.passes_acceptance_gate(scores) is accepted
    assert engine.weighted_score(scores, expression) == pytest.approx(0.7 + gap)
