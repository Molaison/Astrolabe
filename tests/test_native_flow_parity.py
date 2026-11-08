"""CPU parity checks for the default coordinator-backed optimization flow.

These tests intentionally state the expected scientific trajectory directly
instead of carrying a second copy of the old optimization loop in the test
suite. They drive the production root/advance callbacks through the real
coordinator, Decisions owner, SegmentManager, StandardScores, and staged
mutation semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest

import astrolabe.decision_maker as decision_module
import run as run_module
from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import Protomer, SegmentManager
from astrolabe.design import coordinator
from astrolabe.design.native_flow import NativeFlow
from astrolabe.evaluators.utils import StandardScores
from astrolabe.losses.base import LossEngine
from astrolabe.stage_manager import StageManager


@dataclass(frozen=True)
class _ScriptedStep:
    sequence: str
    weighted: float
    early_stop: bool = False
    soft_accepted: bool = False
    gate_passed: bool = True


@pytest.fixture(autouse=True)
def _clean_global_config():
    global_config.reset()
    yield
    global_config.reset()


def _scores(
    tmp_path: Path,
    step: int,
    scripted: _ScriptedStep,
) -> StandardScores:
    path = tmp_path / f"step_{step:02d}.cif"
    path.write_text(
        f"data_step_{step}\n# sequence={scripted.sequence} synthetic CPU fixture\n",
        encoding="utf-8",
    )
    scores = StandardScores(
        names=["A"],
        plddt={"A": [np.asarray([1.0], dtype=np.float32)]},
        complex_paths={"A": path},
    )
    scores.weighted = float(scripted.weighted)
    scores.early_stop = bool(scripted.early_stop)
    scores.soft_accepted = bool(scripted.soft_accepted)
    scores.gate_passed = bool(scripted.gate_passed)
    scores.sequence = scripted.sequence
    return scores


class _ScriptedLoss(LossEngine):
    """Loss boundary returning the decision fields Decisions consumes."""

    def __init__(self) -> None:
        super().__init__(
            {"loss_terms": {}},
            runtime_context=global_config.build_runtime_context(),
            validate_config=False,
        )

    def compute_loss(self, eval_result, runtime_config):
        detail = {
            "weighted": float(eval_result.weighted),
            "early_stop": 1.0 if eval_result.early_stop else 0.0,
            "soft_accepted": 1.0 if eval_result.soft_accepted else 0.0,
            "gate_passed": 1.0 if eval_result.gate_passed else 0.0,
        }
        return detail["weighted"], detail

    def is_early_stopping(self, scores_detail):
        return bool(scores_detail.get("early_stop"))

    def is_soft_accepted(self, scores_detail):
        return bool(scores_detail.get("soft_accepted"))

    def passes_acceptance_gate(self, scores_detail):
        return bool(scores_detail.get("gate_passed", 1.0))


class _StagedMutator:
    name = "ParityMCMC"
    alias = "ParityMCMC"

    def __init__(self, script: dict[int, _ScriptedStep]) -> None:
        self._script = script
        self.last_mutation_telemetry = {"metrics": {}}
        self.observed_pending: list[tuple[int, str]] = []

    def mutate(self, mutable_protomers, evaluation_result, runtime_config):
        step = int(runtime_config["step"])
        expected = self._script[step]
        protomer = mutable_protomers["A"]
        segment = global_config.segments[protomer.components[0]]
        before = protomer.mutated_sequence
        differences = [
            (index, current, target)
            for index, (current, target) in enumerate(
                zip(before, expected.sequence, strict=True)
            )
            if current != target
        ]
        assert len(differences) == 1
        if segment.mutations:
            raise AssertionError("mutation step started with residual pending mutations")
        index, current, target = differences[0]
        assert current == segment.sequence[index]
        segment.add_mutations((index, current, target, 0.0, 0.0))
        self.observed_pending.append((step, protomer.mutated_sequence))


class _SequenceObserverEvaluator:
    alias = "ParityEvaluator"

    def __init__(
        self,
        tmp_path: Path,
        script: dict[int, _ScriptedStep],
    ) -> None:
        self._tmp_path = tmp_path
        self._script = script
        self.observed_inputs: list[tuple[int, str]] = []

    def evaluate(self, inputs, runtime_config):
        step = int(runtime_config["step"])
        expected = self._script[step]
        sequence = inputs["A"].chains[0].sequence
        assert sequence == expected.sequence
        assert global_config.protomers["A"].mutated_sequence == expected.sequence
        self.observed_inputs.append((step, sequence))
        return _scores(self._tmp_path, step, expected)


def _stage_manager(
    *,
    loss: LossEngine,
    mutator: _StagedMutator,
    evaluator: _SequenceObserverEvaluator,
) -> StageManager:
    stages = OmegaConf.create(
        {
            "default": {
                "oligo": ["A"],
                "mutation_num": 1,
                "weighted_score_expression": "weighted",
                "early_stopping": "early_stop > 0.0",
                "acceptance_threshold": "soft_accepted > 0.0",
                "evaluator": "ParityEvaluator",
                "mutators": ["ParityMCMC"],
            }
        }
    )
    stage_manager = StageManager(
        stages,
        loss_computer=loss,
        runtime_context=global_config.build_runtime_context(),
    )
    global_config.init_const("stage_manager", stage_manager)
    global_config.init_const(
        "mutators",
        {"ParityMCMC": mutator},
    )
    global_config.init_const(
        "evaluators",
        {"ParityEvaluator": evaluator},
    )
    return stage_manager


def _run_default_flow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    steps: int,
    seed: int,
    metropolis: bool,
    initial_temperature: float,
    script: dict[int, _ScriptedStep],
):
    global_config.init_const("segments", {})
    global_config.init_const("protomers", {})
    global_config.init_const("mutable_protomers", {})
    global_config.init_const("named_files", {})
    global_config.init_const("segment_manager", SegmentManager())

    protomer = Protomer("A", "AAAA", "p")
    global_config.protomers["A"] = protomer
    global_config.mutable_protomers["A"] = protomer

    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("initial_temperature", initial_temperature)
    global_config.init_const("use_metropolis_acceptance", metropolis)

    loss = _ScriptedLoss()
    global_config.init_const("loss_computer", loss)
    mutator = _StagedMutator(script)
    evaluator = _SequenceObserverEvaluator(tmp_path, script)
    stage_manager = _stage_manager(
        loss=loss,
        mutator=mutator,
        evaluator=evaluator,
    )

    initial = _scores(
        tmp_path,
        0,
        _ScriptedStep(sequence="AAAA", weighted=1.0),
    )
    args = SimpleNamespace(
        steps=steps,
        seed=seed,
        out_dir=tmp_path,
        task_name="native-flow-parity",
        post_eval_hooks=None,
        enable_pyrosetta=False,
        skip_complex_poses=True,
        per_residue_scores=None,
        evaluator=None,
    )
    observed: dict[str, object] = {}
    coordinator_calls = []
    original_coordinator_run = coordinator.run

    def coordinator_spy(system, root, trial, policy, runtime, *, capacity=1):
        coordinator_calls.append(
            {
                "system": system,
                "root": root,
                "trial": trial,
                "policy": policy,
                "runtime": runtime,
            }
        )
        run_result = original_coordinator_run(
            system,
            root,
            trial,
            policy,
            runtime,
            capacity=capacity,
        )
        checkpoint = runtime.load("run/state")
        observed["policy_state"] = runtime.data.get(checkpoint["state"], policy.codec)
        observed["attempts"] = runtime.attempts()
        observed["run_result"] = run_result
        return run_result

    monkeypatch.setattr(coordinator, "run", coordinator_spy)
    runtime_config = {"mutator_stack_pre_update": True}
    result = run_module.run_optimization_loop(
        args,
        loss,
        initial,
        stage_manager,
        runtime_config,
    )
    return {
        "result": result,
        "decisions": global_config.decisions,
        "mutator": mutator,
        "evaluator": evaluator,
        "stage_manager": stage_manager,
        "segment_manager": global_config.segment_manager,
        "observed": observed,
        "coordinator_calls": coordinator_calls,
    }


def _assert_numpy_state_equal(actual, expected) -> None:
    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2:] == expected[2:]


def test_default_flow_matches_explicit_decision_trajectory_and_early_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    script = {
        1: _ScriptedStep("GAAA", 1.4),  # better than 1.0
        2: _ScriptedStep("CAAA", 1.3),  # worse than accepted 1.4
        3: _ScriptedStep("GGAA", 1.4),  # tie with accepted 1.4
        4: _ScriptedStep("GAGA", 1.3, soft_accepted=True),  # threshold
        5: _ScriptedStep("GAAA", 1.5, gate_passed=False),  # hard-gate reject
        6: _ScriptedStep(
            "GAGG",
            1.2,
            early_stop=True,
            soft_accepted=True,
        ),
        7: _ScriptedStep("TAAA", 0.1),  # must not run after early stop
    }

    def unexpected_uniform(_low, _high):
        raise AssertionError("Metropolis RNG must not be consumed when disabled")

    monkeypatch.setattr(decision_module.np.random, "uniform", unexpected_uniform)
    np.random.seed(91)
    rng_before = np.random.get_state()

    with caplog.at_level(logging.INFO):
        run = _run_default_flow(
            tmp_path,
            monkeypatch,
            steps=7,
            seed=23,
            metropolis=False,
            initial_temperature=1.0,
            script=script,
        )

    decisions = run["decisions"]
    assert decisions.scores == {
        0: 1.0,
        1: 1.4,
        2: 1.3,
        3: 1.4,
        4: 1.3,
        5: 1.5,
        6: 1.2,
    }
    assert decisions.accepted_steps == [0, 1, 4, 6]
    assert decisions.last_accepted == 6
    assert decisions.stage_history == {step: "default" for step in range(7)}
    assert decisions.sequences == {
        0: {"pA": "AAAA"},
        1: {"pA": "GAAA"},
        2: {"pA": "CAAA"},
        3: {"pA": "GGAA"},
        4: {"pA": "GAGA"},
        5: {"pA": "GAAA"},
        6: {"pA": "GAGG"},
    }
    assert run["evaluator"].observed_inputs == [
        (step, script[step].sequence) for step in range(1, 7)
    ]
    assert run["mutator"].observed_pending == [
        (step, script[step].sequence) for step in range(1, 7)
    ]
    assert global_config.protomers["A"].sequence == "GAGG"
    assert global_config.protomers["A"].mutated_sequence == "GAGG"
    assert run["segment_manager"]._pre_update_snapshot is None
    assert run["result"].weighted == 1.2
    assert run["observed"]["policy_state"].stop_reason == "scientific_early_stop"
    _assert_numpy_state_equal(np.random.get_state(), rng_before)

    assert len(run["coordinator_calls"]) == 1
    call = run["coordinator_calls"][0]
    assert isinstance(call["trial"], NativeFlow)
    assert call["system"].variables == ("A_1",)
    assert call["root"] == call["trial"].initial
    assert all(attempt.status == "completed" for attempt in run["observed"]["attempts"])
    assert (
        len(
            [
                attempt
                for attempt in run["observed"]["attempts"]
                if attempt.key.named_round.startswith("prediction/")
            ]
        )
        == 6
    )

    messages = [record.getMessage() for record in caplog.records]
    assert any("decision=accept reason=better" in message for message in messages)
    assert any("decision=reject reason=worse" in message for message in messages)
    assert any("decision=accept reason=threshold" in message for message in messages)
    assert any(
        "decision=reject reason=acceptance_gate" in message for message in messages
    )
    assert any("Early stopping triggered" in message for message in messages)

    execution = (tmp_path / "execution.json").read_text(encoding="utf-8")
    assert '"runner": "astrolabe.design.coordinator"' in execution
    assert '"recovery": "process_lifetime"' in execution


def test_default_flow_metropolis_rng_stream_matches_independent_mt19937(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = 0
    temperature = 1.0
    script = {
        1: _ScriptedStep("GAAA", 0.6),  # worse than 1.0; first draw accepts
        2: _ScriptedStep("CAAA", 0.2),  # worse than 0.6; second draw rejects
    }

    rng = np.random.RandomState(seed)
    draws = rng.uniform(0.0, 1.0, size=2)
    expected_rng_state = rng.get_state()
    assert draws.tolist() == [0.5488135039273248, 0.7151893663724195]

    # The production schedule is exactly 1.0 for an initial temperature of 1.0.
    step1_temperature = temperature
    step1_probability = np.exp(-(1.0 - 0.6) / step1_temperature)
    assert draws[0] < step1_probability

    step2_temperature = temperature
    step2_probability = np.exp(-(0.6 - 0.2) / step2_temperature)
    assert draws[1] >= step2_probability

    uniform_calls = []
    original_uniform = decision_module.np.random.uniform

    def record_uniform(low, high):
        uniform_calls.append((low, high))
        return original_uniform(low, high)

    monkeypatch.setattr(decision_module.np.random, "uniform", record_uniform)
    np.random.seed(seed)
    run = _run_default_flow(
        tmp_path,
        monkeypatch,
        steps=3,
        seed=seed,
        metropolis=True,
        initial_temperature=temperature,
        script=script,
    )

    decisions = run["decisions"]
    assert decisions.scores == {0: 1.0, 1: 0.6, 2: 0.2}
    assert decisions.accepted_steps == [0, 1]
    assert decisions.last_accepted == 1
    assert decisions.sequences == {
        0: {"pA": "AAAA"},
        1: {"pA": "GAAA"},
        2: {"pA": "CAAA"},
    }
    assert global_config.protomers["A"].sequence == "GAAA"
    assert run["result"].weighted == 0.6
    assert run["observed"]["policy_state"].stop_reason is None
    assert uniform_calls == [(0, 1), (0, 1)]
    _assert_numpy_state_equal(np.random.get_state(), expected_rng_state)
