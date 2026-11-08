"""Toy-only fixed-context audits using the vendored Hamiltonian and selectivity.

Extract native definitions via AST to avoid importing the Python 3.12 model
stack into the licensed Python 3.11 unit harness. No definitions are rewritten.
"""
from __future__ import annotations

import ast
import copy
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List, Sequence

import numpy as np
import pytest
import torch

from astrolabe.mutators.wrappers import proton_pottsmpnn_daemon as daemon
from astrolabe.mutators.wrappers.proton_pottsmpnn_runner import ProtonPottsMPNNRunner


@pytest.fixture(scope="module")
def native():
    root = Path(__file__).resolve().parents[1] / "vendor/proton_pottsmpnn/foundry/models/mpnn/src/mpnn"
    env = dict(torch=torch, List=List, Sequence=Sequence, copy=copy)
    path = root / "model/pottsmpnn.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PottsMPNN")
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)
               and n.name in {"calc_potts_eners", "_directed_pair_edges"}]
    assert len(methods) == 2
    model = ast.ClassDef(name="PottsMPNN", bases=[], keywords=[], body=methods, decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[model], type_ignores=[])), str(path), "exec"), env)
    path = root / "inference_engines/potts_mpnn_ph.py"
    wanted = {"_PottsScorer", "_selective_energy_of", "_selective_energy_sum"}
    nodes = [n for n in ast.parse(path.read_text()).body
             if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in wanted]
    assert len(nodes) == len(wanted)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), env)
    return SimpleNamespace(**{name: env[name] for name in wanted})


@pytest.fixture
def toy(native, tmp_path, monkeypatch):
    vocab = {token: i for i, token in enumerate(sorted(daemon.V6_BINDER_TOKENS | {"UNK"}))}
    length, size = 6, len(vocab)
    # B1, A1, A2, A3, A4, B2; the two audited centers are context indices 2,3.
    # E(SS)=0, E(SP)=2, E(PS)=1, E(PP)=8: +5 interacting P/P term.
    table = torch.zeros(1, length, 2, size, size, dtype=torch.float64)
    p = vocab["HIS-P"]
    table[0, 2, 0, p, p] = 1
    table[0, 3, 0, p, p] = 2
    table[0, 2, 1, p, p] = 5
    neighbors = torch.tensor([[[0, 1], [1, 0], [2, 3], [3, 2], [4, 5], [5, 4]]])

    class Scorer(native._PottsScorer):
        def __init__(self):
            super().__init__(table, neighbors)
            self.calls = []

        def H_of(self, sequence):
            self.calls.append(tuple(int(x) for x in sequence))
            return super().H_of(sequence)

    token_aa = SimpleNamespace(
        chain_id=np.array(["B", "A", "A", "A", "A", "B"]),
        res_id=np.array([8, 10, 11, 12, 13, 9]),
        res_name=np.array(["GLU", "ALA", "HIS", "HIS", "ALA", "HIS"]),
    )
    ctx = SimpleNamespace(
        S_native=torch.tensor([vocab[t] for t in ["GLU-P", "ALA", "HIS-S", "HIS-S", "ALA", "HIS-P"]]),
        chainA_all_idx=np.array([1, 2, 3, 4]), token_aa=token_aa,
        encoding=SimpleNamespace(token_to_idx=vocab), scorer=Scorer(), L=length,
    )

    class Engine:
        calls = 0

        def _build_context(self, atom_array, binder_chain):
            assert atom_array is token_aa and binder_chain == "A"
            self.calls += 1
            return ctx

    engine = Engine()
    observed = {"A": "AHHA", "B": "EH"}
    monkeypatch.setattr(daemon, "_load_atom_array", lambda path: token_aa)
    monkeypatch.setattr(daemon, "_token_chain_sequences", lambda array: (array, observed))
    pdb = tmp_path / "input.pdb"
    pdb.write_text("toy-only; no structure parser or model is invoked\n")
    message = dict(id="toy", type="score_states", pdb_path=str(pdb), binder_chain="A",
                   binder_sequence="AHHA", chain_sequences=dict(observed), histidine_positions=[1, 2])

    def run(**kwargs):
        return daemon._score_states((engine, None), {**message, **kwargs}, tmp_path / "out")

    return SimpleNamespace(ctx=ctx, engine=engine, run=run, message=message,
                           observed=observed, token_aa=token_aa, vocab=vocab, tmp_path=tmp_path)


def test_native_mixed_patterns_complete_tokens_sign_and_raw(toy):
    patterns = ["SS", "SP", "PS", "PP"]
    result = toy.run(state_patterns=patterns)
    assert toy.engine.calls == 1
    assert len(toy.ctx.scorer.calls) == 4
    assert result["state_names"] == patterns
    assert [s["energy"] for s in result["states"]] == [0, 2, 1, 8]
    assert (result["neutral_energy"], result["protonated_energy"], result["gap"]) == (0, 8, 8)
    energies = {s["name"]: s["energy"] for s in result["states"]}
    assert energies["SS"] - energies["PP"] == -8  # acid-off S target: negative desired
    assert result["histidine_positions"] == [1, 2]
    assert result["context_histidine_positions"] == [2, 3]
    assert result["outside_centers_unchanged"] is result["same_context"] is True
    assert result["audit_source"] == "fixed_geometry_potts_context"
    for state, pattern in zip(result["states"], patterns):
        assert len(state["tokens"]) == len(state["encoded_sequence"]) == 6
        assert state["center_tokens"] == [f"HIS-{s}" for s in pattern]
        assert [state["tokens"][p] for p in [0, 1, 4, 5]] == ["GLU-P", "ALA", "ALA", "HIS-P"]
    raw = json.loads(Path(result["raw_output_path"]).read_text())
    assert raw["input_state_patterns"] == patterns
    assert raw["input_request"] == {**toy.message, "state_patterns": patterns}
    assert raw["states"] == result["states"]
    assert raw["native_tokens"] == result["native_tokens"]


def test_native_selectivity_sum_differs_from_simultaneous_flip(native, toy):
    ctx, vocab = toy.ctx, toy.vocab
    s, p = vocab["HIS-S"], vocab["HIS-P"]
    pins = [SimpleNamespace(position=i, prot_idx=s, dep_idxs=[p]) for i in [2, 3]]
    # Native prot_idx means the desired state; acid-off deliberately pins S.
    each = [native._selective_energy_of(ctx, ctx.S_native, pin.position, s, [p]) for pin in pins]
    locked_sum = native._selective_energy_sum(ctx, ctx.S_native, pins)
    result = toy.run(state_patterns=["SS", "SP", "PS", "PP"])
    simultaneous = result["neutral_energy"] - result["protonated_energy"]
    assert each == [-1, -2]
    assert locked_sum == sum(each) == -3
    assert simultaneous == -8
    assert locked_sum != simultaneous
    reverse_pins = [SimpleNamespace(position=i, prot_idx=p, dep_idxs=[s]) for i in [2, 3]]
    assert native._selective_energy_sum(ctx, ctx.S_native, reverse_pins) == 13
    assert result["gap"] == 8


@pytest.mark.parametrize("patterns", [
    [["S", "S"], ["S", "P"], ["P", "S"], ["P", "P"]],
    {"target": "SS", "mixed": ["S", "P"], "comparison": "PP"},
])
def test_list_tokens_or_named_strings(toy, patterns):
    result = toy.run(state_patterns=patterns)
    expected = list(patterns) if isinstance(patterns, dict) else ["SS", "SP", "PS", "PP"]
    assert result["state_names"] == expected


def test_subset_keeps_legacy_endpoints_and_unsorted_center_order(toy):
    result = toy.run(state_patterns={"mixed": "SP"}, histidine_positions=[2, 1])
    assert result["states"][0]["energy"] == 1
    assert result["context_histidine_positions"] == [3, 2]
    assert result["gap"] == 8
    assert len(toy.ctx.scorer.calls) == 3


def test_omitted_options_exact_legacy_response(toy):
    result = toy.run()
    assert len(toy.ctx.scorer.calls) == 2
    assert set(result) == {"neutral_energy", "protonated_energy", "gap", "n_centers",
                           "histidine_positions", "binder_sequence", "outside_centers_unchanged", "raw_output_path"}
    assert result["gap"] == 8
    raw = json.loads(Path(result["raw_output_path"]).read_text())
    assert "states" not in raw and "input_request" not in raw


@pytest.mark.parametrize("patterns", [
    [], {}, "SS", ("SS", "PP"), [""], ["S"], ["SSS"], ["SX"], ["sp"],
    ["S P"], ["SS", "SS"], [["HIS-S", "HIS-P"]], [["S", 1]], [None],
    [{"name": "target", "pattern": "SS"}], {"a": "SS", "b": "SS"},
    {"": "SS"}, {"bad name": "SS"}, {1: "SS"}, {"x" * 65: "SS"},
    ["SS"] * 17,
])
def test_reject_pattern_schema_before_context(toy, patterns):
    with pytest.raises(ValueError, match="state_patterns"):
        toy.run(state_patterns=patterns)
    assert toy.engine.calls == 0
    assert not toy.ctx.scorer.calls


def test_pattern_limits_and_no_legacy_center_cap():
    all4 = ["".join(p) for p in itertools.product("SP", repeat=4)]
    assert len(daemon._validated_state_patterns(all4, 4)) == 16
    with pytest.raises(ValueError, match="4 His"):
        daemon._validated_state_patterns(["SSSSS"], 5)
    assert daemon._validated_state_patterns(None, 5) is None


@pytest.mark.parametrize("positions", [[], [1, 1], [0], [4], [-1], [True], [1.0], ["1"]])
def test_reject_centers(toy, positions):
    with pytest.raises(ValueError):
        toy.run(histidine_positions=positions, state_patterns=["SS"])
    assert toy.engine.calls == 0


def test_encoded_genotype_preserves_structure_identity_and_noncenters(toy):
    tokens = ["GLY", "HIS-P", "HIS-S", "ASP-D"]
    native_before = toy.ctx.S_native.clone()
    result = toy.run(encoded_binder_tokens=tokens, state_patterns=["SS", "SP", "PS", "PP"])
    assert result["binder_sequence"] == "AHHA"
    assert result["scored_binder_sequence"] == "GHHD"
    assert result["scored_binder_tokens"] == tokens
    assert result["scored_tokens"] == ["GLU-P", *tokens, "HIS-P"]
    assert torch.equal(toy.ctx.S_native, native_before)
    assert toy.engine.calls == 1
    for state in result["states"]:
        assert [state["tokens"][p] for p in [0, 1, 4, 5]] == ["GLU-P", "GLY", "ASP-D", "HIS-P"]
    raw = json.loads(Path(result["raw_output_path"]).read_text())
    assert raw["input_encoded_binder_tokens"] == tokens


def test_encoded_new_his_uses_original_geometry_without_structure_edit(toy):
    toy.token_aa.res_name[2:4] = "ALA"
    toy.observed["A"] = "AAAA"
    toy.ctx.S_native[2:4] = toy.vocab["ALA"]
    result = toy.run(binder_sequence="AAAA", chain_sequences={"A": "AAAA", "B": "EH"},
                     encoded_binder_tokens=["ALA", "HIS-S", "HIS-P", "ALA"],
                     state_patterns=["SS", "SP", "PS", "PP"])
    assert result["binder_sequence"] == "AAAA"
    assert result["scored_binder_sequence"] == "AHHA"
    assert toy.token_aa.res_name.tolist() == ["GLU", "ALA", "ALA", "ALA", "ALA", "HIS"]
    assert toy.engine.calls == 1


def test_encoded_only_retains_legacy_energy_pair(toy):
    result = toy.run(encoded_binder_tokens=["ALA", "HIS-P", "HIS-S", "ALA"])
    assert result["state_names"] == ["SS", "PP"]
    assert len(toy.ctx.scorer.calls) == 2
    assert result["gap"] == 8


@pytest.mark.parametrize("tokens", [
    "ALA HIS-S HIS-S ALA", [], ["ALA"] * 3, ["ALA"] * 5,
    ["ALA", "HIS-D", "HIS-S", "ALA"], ["ALA", "HID", "HIS-S", "ALA"],
    ["ALA", "HIS-S", "HIS-P", "UNK"], ["ALA", "HIS-S", "HIS-P", "gly"],
    ["ALA", "HIS-S", "HIS-P", None], ["ALA", "HIS-S", "HIS-P", {}],
    ["ALA", "ALA", "HIS-P", "ALA"],
])
def test_reject_bad_encoded_genotype(toy, tokens):
    with pytest.raises(ValueError):
        toy.run(encoded_binder_tokens=tokens, state_patterns=["SS"])
    assert toy.engine.calls == 0


@pytest.mark.parametrize("bad", ["chain", "sequence", "mapping", "vocab"])
def test_reject_different_context(toy, bad):
    if bad == "chain":
        toy.token_aa.chain_id[2] = "B"
    elif bad == "sequence":
        toy.token_aa.res_name[1] = "GLY"
    elif bad == "mapping":
        toy.ctx.chainA_all_idx = np.array([1, 2, 2, 4])
    else:
        del toy.ctx.encoding.token_to_idx["HIS-S"]
    with pytest.raises((RuntimeError, KeyError)):
        toy.run(state_patterns=["SS"])
    assert not toy.ctx.scorer.calls


def test_reject_chain_sequence_mismatch(toy):
    with pytest.raises(ValueError, match="does not match"):
        toy.run(chain_sequences={"A": "AAAA"}, state_patterns=["SS"])


@pytest.mark.parametrize("bad", ["nan_mixed", "mutate_noncenter", "mutate_native"])
def test_reject_broken_scorer(toy, bad):
    original = toy.ctx.scorer.H_of

    def broken(seq):
        value = original(seq)
        if bad == "nan_mixed" and int(seq[2]) != int(seq[3]):
            return float("nan")
        if bad == "mutate_noncenter":
            seq[0] = toy.vocab["ALA"]
        if bad == "mutate_native":
            toy.ctx.S_native[0] = toy.vocab["ALA"]
        return value

    toy.ctx.scorer.H_of = broken
    with pytest.raises(RuntimeError, match="non-finite|changed"):
        toy.run(state_patterns=["SS", "SP", "PS", "PP"])


def _runner(toy, monkeypatch, damage=None):
    # Exercise the real runner protocol/validation without a pretrained process.
    import threading
    runner = object.__new__(ProtonPottsMPNNRunner)
    runner._lock = threading.RLock()
    runner.repo_path = toy.tmp_path
    runner.checkpoint_path = toy.tmp_path / "unused.ckpt"
    runner.extended_vocab, runner.device, runner.request_timeout = "v6", "cpu", 10
    messages = []
    monkeypatch.setattr(runner, "_ensure_daemon", lambda: SimpleNamespace(pid=123))
    monkeypatch.setattr(runner, "_write_request", lambda proc, message: messages.append(message))

    def respond(proc, timeout):
        message = messages[-1]
        response = {"type": "result", "id": message["id"],
                    **daemon._score_states((toy.engine, None), message, toy.tmp_path / "runner")}
        if damage:
            damage(response)
        return json.loads(json.dumps(response))  # exact JSON transport types

    monkeypatch.setattr(runner, "_read_response", respond)
    return runner, messages


def _kwargs(toy):
    return {k: v for k, v in toy.message.items() if k not in {"id", "type"}}


def test_runner_native_path_and_parameter_record(toy, monkeypatch):
    runner, messages = _runner(toy, monkeypatch)
    result = runner.score_states(**_kwargs(toy),
        state_patterns={"target": "SS", "mixed1": "SP", "mixed2": "PS", "comparison": "PP"},
        encoded_binder_tokens=["GLY", "HIS-S", "HIS-P", "ASP-D"])
    assert result["scored_binder_sequence"] == "GHHD"
    assert result["binder_sequence"] == "AHHA"
    assert [s["energy"] for s in result["states"]] == [0, 2, 1, 8]
    raw = json.loads(Path(result["raw_output_path"]).read_text())
    assert raw["input_request"] == messages[0]
    assert raw["input_request"]["audit_parameters"]["checkpoint_path"] == str(runner.checkpoint_path)


def test_runner_legacy_payload_unchanged(toy, monkeypatch):
    runner, messages = _runner(toy, monkeypatch)
    result = runner.score_states(**_kwargs(toy))
    assert "states" not in result
    assert "state_patterns" not in messages[0]
    assert "encoded_binder_tokens" not in messages[0]
    assert "audit_parameters" not in messages[0]


@pytest.mark.parametrize("damage", [
    lambda r: r.update(same_context=False),
    lambda r: r.update(scored_binder_sequence="AAAA"),
    lambda r: r.update(state_names=["wrong"]),
    lambda r: r["states"].pop(),
    lambda r: r["states"][0].update(energy=float("nan")),
    lambda r: r["states"][0].update(energy=100),
    lambda r: r["states"][1]["encoded_sequence"].__setitem__(0, -1),
    lambda r: r["states"][1]["tokens"].__setitem__(0, "ALA"),
    lambda r: r.update(context_histidine_positions=[3, 2]),
])
def test_runner_rejects_invalid_audit_response(toy, monkeypatch, damage):
    runner, _ = _runner(toy, monkeypatch, damage)
    with pytest.raises(RuntimeError, match="invalid state audit"):
        runner.score_states(**_kwargs(toy), state_patterns=["SS", "SP", "PS", "PP"])


def test_runner_rejects_invalid_patterns_before_start(toy, monkeypatch):
    runner, messages = _runner(toy, monkeypatch)
    with pytest.raises(ValueError, match="duplicate"):
        runner.score_states(**_kwargs(toy), state_patterns=["SS", "SS"])
    assert not messages and toy.engine.calls == 0
