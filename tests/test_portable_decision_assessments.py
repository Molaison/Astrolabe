"""Cold portable recovery must not replay or reconstruct decision observations."""

from dataclasses import fields, replace
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest

from astrolabe.core import global_config
import astrolabe.decision_maker as decision_module
from astrolabe.decision_maker import DecisionAssessment, Decisions
from astrolabe.design.portable_native_state import (
    DECISION_STATE_CODEC,
    NUMERIC_TREE_CODEC,
    PORTABLE_NATIVE_REFERENCE_CODECS,
    restore_decision_state,
    snapshot_decision_state,
)
from astrolabe.design.store import DataStore
from astrolabe.evaluators.evidence import EvaluationRecord, InputObservation
from astrolabe.evaluators.utils import StandardScores


class Loss:
    def __init__(self, score):
        self.score = score

    def compute_loss(self, result, runtime):
        return self.score, {"metric": {"value": self.score}}

    def is_early_stopping(self, detail):
        return False

    def is_soft_accepted(self, detail):
        return False


@pytest.fixture
def native_runtime(tmp_path):
    global_config.reset()
    stage = SimpleNamespace(current_stage="one")
    for name, value in {
        "stage_manager": stage,
        "segment_manager": SimpleNamespace(
            apply_mutations=lambda: None, clear_mutations=lambda: None,
        ),
        "out_dir": tmp_path,
        "protomers": {
            "A": SimpleNamespace(
                type=SimpleNamespace(value="p"), mutated_sequence="BBB",
            ),
        },
        "use_metropolis_acceptance": False,
    }.items():
        global_config.init_const(name, value)
    try:
        yield stage
    finally:
        global_config.reset()


def _scores(tmp_path, label):
    source = tmp_path / f"{label}.cif"
    source.write_bytes(f"data_{label}\n# original prediction\n".encode())
    return StandardScores(
        names=["A"],
        plddt={"A": [np.array([0.7, 0.8], dtype=np.float32)]},
        complex_paths={"A": source},
        evaluation_records=(EvaluationRecord(
            invocation_id=label, step=0, committed_stage="one",
            execution_stage="one", backend_alias="fixture",
            input=InputObservation("A", (("A", "ProteinChain", "AAA"),), (17,)),
            original_complex_path=source,
        ),),
    )


def _initial(tmp_path):
    return Decisions(
        {0: {"pA": "AAA"}}, {0: 1.0}, {0: _scores(tmp_path, "initial")},
        {0: {"array": np.array([1, 2], dtype=np.int16)}}, [0], 0, {0: "one"},
    )


def _details(assessment):
    return {
        item.name: getattr(assessment, item.name)
        for item in fields(DecisionAssessment)
        if item.name != "evaluation_result"
    }


def _persist(state, data):
    for codec in PORTABLE_NATIVE_REFERENCE_CODECS:
        data.register(codec)
    reference = data.put(state, DECISION_STATE_CODEC)
    data.persist((reference,))
    return reference


def _restore_cold(reference, store, destination):
    cold = DataStore(store)
    for codec in PORTABLE_NATIVE_REFERENCE_CODECS:
        cold.register(codec)
    return restore_decision_state(
        cold.get(reference, DECISION_STATE_CODEC), cold, destination,
    )


def _no_replay(*args, **kwargs):
    pytest.fail("snapshot/restore replayed a decision or observation")


def test_cold_history_preserves_rejections_reassessments_and_next_identity(
    tmp_path, native_runtime, monkeypatch,
):
    decisions = _initial(tmp_path)
    decisions.dump_results(tmp_path)
    assert not decisions.make_mutation_decision(
        1, _scores(tmp_path, "rejected"), Loss(0), {},
    )[2]
    assert decisions.make_mutation_decision(
        2, _scores(tmp_path, "accepted"), Loss(2), {},
    )[2]
    for stage, score in (("two", 3), ("three", 4)):
        native_runtime.current_stage = stage
        decisions.recompute_last_score(Loss(score), {}, _scores(tmp_path, stage))
    assert [item.kind for item in decisions.assessments] == [
        "initial", "mutation", "mutation", "stage_reassessment", "stage_reassessment",
    ]
    expected = [NUMERIC_TREE_CODEC.encode(_details(a)) for a in decisions.assessments]
    expected_rows = [dict(row) for row in decisions.row_history]
    expected_bytes = [
        a.evaluation_result.complex_paths["A"].read_bytes()
        for a in decisions.assessments
    ]
    store = tmp_path / "store"
    data = DataStore(store)
    numpy_rng, python_rng = np.random.get_state(), random.getstate()
    with monkeypatch.context() as guard:
        for name in (
            "__post_init__", "_observe_assessment", "make_mutation_decision",
            "recompute_last_score",
        ):
            guard.setattr(Decisions, name, _no_replay)
        reference = _persist(snapshot_decision_state(decisions, data), data)
        # Neither original source files nor the live history may supply recovery.
        for source in tmp_path.glob("*.cif"):
            source.unlink()
        decisions.assessments.clear()
        restored = _restore_cold(reference, store, tmp_path / "restored")
    np.testing.assert_equal(np.random.get_state(), numpy_rng)
    assert random.getstate() == python_rng
    assert [NUMERIC_TREE_CODEC.encode(_details(a)) for a in restored.assessments] == expected
    assert restored.row_history == expected_rows
    assert restored.accepted_steps == [0, 2] and restored.last_accepted == 2
    assert restored._attempt_count == 2 and restored._accepted_attempt == 2
    assert restored._revision_known is True
    assert [a.assessment_id for a in restored.assessments] == list(range(5))
    assert [a.attempt_id for a in restored.assessments] == [None, 1, 2, 2, 2]
    assert [a.accepted_revision for a in restored.assessments] == [0, 0, 1, 1, 1]
    assert [a.weighted_score for a in restored.assessments] == [1, 0, 2, 3, 4]
    assert [
        a.evaluation_result.evaluation_records[0].invocation_id
        for a in restored.assessments
    ] == ["initial", "rejected", "accepted", "two", "three"]
    assert [
        a.evaluation_result.complex_paths["A"].read_bytes()
        for a in restored.assessments
    ] == expected_bytes
    assert isinstance(restored.assessments[0].evidence_snapshot["evaluation_records"], tuple)
    np.testing.assert_array_equal(restored.assessments[0].score_details["array"], [1, 2])
    assert restored.assessments[0].score_details["array"].dtype == np.dtype("int16")
    assert restored.make_mutation_decision(
        3, _scores(tmp_path, "next"), Loss(5), {},
    )[2]
    next_assessment = restored.assessments[-1]
    assert (next_assessment.assessment_id, next_assessment.attempt_id,
            next_assessment.accepted_revision) == (5, 3, 2)


@pytest.mark.parametrize("operation", ["initial", "mutation", "stage_reassessment"])
def test_cold_observation_errors_and_unknown_revision_are_not_repaired(
    tmp_path, native_runtime, monkeypatch, operation,
):
    decisions = None if operation == "initial" else _initial(tmp_path)
    with monkeypatch.context() as failure:
        def bad_snapshot(result):
            raise RuntimeError("historical evidence unavailable")

        failure.setattr(decision_module, "_snapshot_result", bad_snapshot)
        if operation == "initial":
            decisions = _initial(tmp_path)
        elif operation == "mutation":
            assert decisions.make_mutation_decision(
                1, _scores(tmp_path, "failed_observation"), Loss(2), {},
            )[2]
        else:
            native_runtime.current_stage = "two"
            decisions.recompute_last_score(Loss(2), {})
    assert decisions.observation_errors[0]["operation"] == operation
    expected_errors = [dict(error) for error in decisions.observation_errors]
    expected = [NUMERIC_TREE_CODEC.encode(_details(a)) for a in decisions.assessments]
    expected_counters = (
        decisions._attempt_count, decisions._accepted_attempt, decisions._revision_known,
    )
    store = tmp_path / "store"
    data = DataStore(store)
    reference = _persist(snapshot_decision_state(decisions, data), data)
    with monkeypatch.context() as guard:
        guard.setattr(Decisions, "__post_init__", _no_replay)
        guard.setattr(decision_module, "_snapshot_result", _no_replay)
        restored = _restore_cold(reference, store, tmp_path / "restored")
    assert restored.observation_errors == expected_errors
    assert [NUMERIC_TREE_CODEC.encode(_details(a)) for a in restored.assessments] == expected
    assert (restored._attempt_count, restored._accepted_attempt,
            restored._revision_known) == expected_counters
    assert restored.make_mutation_decision(
        max(restored.scores) + 1, _scores(tmp_path, "next"), Loss(3), {},
    )[2]
    assert restored.assessments[-1].attempt_id == expected_counters[0] + 1
    assert restored.assessments[-1].accepted_revision == (
        1 if operation == "stage_reassessment" else None
    )
    assert restored.observation_errors == expected_errors


def test_cold_initial_rejection_stays_rejected(tmp_path, native_runtime):
    decisions = _initial(tmp_path)
    decisions.reject_initial_assessment()
    store = tmp_path / "store"
    data = DataStore(store)
    reference = _persist(snapshot_decision_state(decisions, data), data)
    restored = _restore_cold(reference, store, tmp_path / "restored")
    assert len(restored.assessments) == 1
    assert restored.assessments[0].accepted is False
    assert restored.assessments[0].accepted_revision is None
    assert restored._revision_known is False
    assert restored.accepted_steps == [0] and restored.last_accepted == 0


def test_legacy_wire_record_loads_without_fabricating_assessments(
    tmp_path, native_runtime, monkeypatch,
):
    decisions = _initial(tmp_path)
    assert decisions.make_mutation_decision(
        1, _scores(tmp_path, "accepted"), Loss(2), {},
    )[2]
    store = tmp_path / "store"
    data = DataStore(store)
    legacy = replace(
        snapshot_decision_state(decisions, data), assessments=None, observation_state=None,
    )
    wire_fields = json.loads(DECISION_STATE_CODEC.encode(legacy))["fields"]
    assert "assessments" not in wire_fields and "observation_state" not in wire_fields
    reference = _persist(legacy, data)
    with monkeypatch.context() as guard:
        guard.setattr(Decisions, "__post_init__", _no_replay)
        restored = _restore_cold(reference, store, tmp_path / "restored")
    assert restored.scores == decisions.scores
    assert restored.accepted_steps == [0, 1] and restored.last_accepted == 1
    assert restored.assessments == [] and restored.observation_errors == []
    assert restored._attempt_count == 0 and restored._accepted_attempt is None
    assert restored._revision_known is False
    assert restored.make_mutation_decision(
        2, _scores(tmp_path, "next"), Loss(3), {},
    )[2]
    assert len(restored.assessments) == 1
    assert restored.assessments[0].kind == "mutation"
    assert restored.assessments[0].accepted_revision is None
