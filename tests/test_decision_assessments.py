from types import SimpleNamespace
import csv

import pytest

from astrolabe.core import global_config
from astrolabe.decision_maker import Decisions


class Loss:
    def __init__(self, score):
        self.score = score

    def compute_loss(self, result, runtime):
        return self.score, {"metric": {"value": self.score}}

    def is_early_stopping(self, detail):
        return False

    def is_soft_accepted(self, detail):
        return False


def test_attempts_and_stage_reassessment_preserve_evidence_and_csv(tmp_path):
    global_config.reset()
    calls = []
    stage = SimpleNamespace(current_stage="one")
    for key, value in {
        "stage_manager": stage,
        "segment_manager": SimpleNamespace(
            apply_mutations=lambda: calls.append("apply"),
            clear_mutations=lambda: calls.append("clear"),
        ),
        "out_dir": tmp_path,
        "protomers": {"A": SimpleNamespace(type=SimpleNamespace(value="p"), mutated_sequence="BBB")},
        "use_metropolis_acceptance": False,
    }.items():
        global_config.init_const(key, value)
    initial = object()
    decisions = Decisions({0: {"pA": "AAA"}}, {0: 1.0}, {0: initial},
                          {0: {"metric": {"value": 1.0}}}, [0], 0, {0: "one"})
    decisions.dump_results(tmp_path)
    rejected, accepted = object(), object()
    assert decisions.make_mutation_decision(1, rejected, Loss(0), {})[2] is False
    assert decisions.make_mutation_decision(2, accepted, Loss(2), {})[2] is True
    stage.current_stage = "two"
    reevaluated = object()
    decisions.recompute_last_score(Loss(3), {}, reevaluated)
    stage.current_stage = "three"
    decisions.recompute_last_score(Loss(4), {}, object())
    items = decisions.assessments
    assert [(x.kind, x.step, x.attempt_id, x.accepted_revision, x.accepted) for x in items] == [
        ("initial", 0, None, 0, True), ("mutation", 1, 1, 0, False),
        ("mutation", 2, 2, 1, True), ("stage_reassessment", 2, 2, 1, True),
        ("stage_reassessment", 2, 2, 1, True),
    ]
    assert [x.assessment_id for x in items] == list(range(5))
    assert [x.weighted_score for x in items] == [1, 0, 2, 3, 4]
    assert items[1].evaluation_result is rejected
    assert items[2].evaluation_result is accepted
    assert items[3].evaluation_result is reevaluated
    decisions.scores_detail[2]["metric"]["value"] = 99
    assert items[2].score_details["metric"]["value"] == 2
    assert calls == ["clear", "apply"]
    assert decisions.last_accepted == 2
    with (tmp_path / "scores.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [(r["step"], r["accepted"], r["stage"]) for r in rows] == [
        ("0", "True", "one"), ("1", "False", "one"), ("2", "True", "one"),
        ("2", "True", "two"), ("2", "True", "three"),
    ]
    assert not {"attempt_id", "assessment_id", "accepted_revision"} & rows[0].keys()
    global_config.reset()


def test_failed_decision_does_not_record_completed_assessment(tmp_path):
    global_config.reset()
    global_config.init_const("stage_manager", SimpleNamespace(current_stage="one"))
    decisions = Decisions({0: {}}, {0: 1}, {0: object()}, {0: {}}, [0], 0, {0: "one"})
    class Broken:
        def compute_loss(self, result, runtime):
            raise RuntimeError("failure")
    with pytest.raises(RuntimeError, match="failure"):
        decisions.make_mutation_decision(1, object(), Broken(), {})
    assert len(decisions.assessments) == 1
    global_config.reset()


@pytest.mark.parametrize("failure", ["apply", "clear", "dump"])
def test_commit_failure_keeps_existing_mutation_boundary(tmp_path, monkeypatch, failure):
    global_config.reset()
    events = []

    def action(name):
        events.append(name)
        if name == failure:
            raise RuntimeError(name)

    for key, value in {
        "stage_manager": SimpleNamespace(current_stage="one"),
        "segment_manager": SimpleNamespace(
            apply_mutations=lambda: action("apply"),
            clear_mutations=lambda: action("clear"),
        ),
        "out_dir": tmp_path,
        "protomers": {},
        "use_metropolis_acceptance": False,
    }.items():
        global_config.init_const(key, value)
    decisions = Decisions({0: {}}, {0: 1}, {0: "initial"}, {0: {}}, [0], 0, {0: "one"})
    if failure == "dump":
        monkeypatch.setattr(decisions, "dump_results", lambda *args, **kwargs: action("dump"))
    score = 0 if failure == "clear" else 2
    with pytest.raises(RuntimeError, match=failure):
        decisions.make_mutation_decision(1, "candidate", Loss(score), {})
    assert decisions.last_accepted == (0 if failure == "clear" else 1)
    assert decisions.accepted_steps == ([0] if failure == "clear" else [0, 1])
    assert events == (["apply", "dump"] if failure == "dump" else [failure])
    assert len(decisions.row_history) == (1 if failure == "dump" else 0)
    assert len(decisions.assessments) == 1
    global_config.reset()


def test_observation_failure_does_not_undo_decision_and_marks_revision_unknown(tmp_path, monkeypatch):
    import json
    import astrolabe.decision_maker as module

    global_config.reset()
    events = []
    for key, value in {
        "stage_manager": SimpleNamespace(current_stage="one"),
        "segment_manager": SimpleNamespace(apply_mutations=lambda: events.append("apply"),
                                           clear_mutations=lambda: events.append("clear")),
        "out_dir": tmp_path, "protomers": {}, "use_metropolis_acceptance": False,
    }.items():
        global_config.init_const(key, value)
    decisions = Decisions({0: {}}, {0: 1}, {0: object()}, {0: {}}, [0], 0, {0: "one"})
    original_snapshot = module._snapshot_result
    monkeypatch.setattr(module, "_snapshot_result", lambda _: (_ for _ in ()).throw(RuntimeError("snapshot failed")))
    result = decisions.make_mutation_decision(1, object(), Loss(2), {})
    assert result[2] is True
    assert events == ["apply"]
    assert decisions.last_accepted == 1
    assert decisions.accepted_steps == [0, 1]
    assert decisions.row_history[-1]["accepted"] is True
    assert (tmp_path / "scores.csv").exists()
    assert len(decisions.assessments) == 1
    assert decisions._attempt_count == 0
    assert json.loads(json.dumps(decisions.observation_errors))[0]["error_type"] == "RuntimeError"
    monkeypatch.setattr(module, "_snapshot_result", original_snapshot)
    assert decisions.make_mutation_decision(2, object(), Loss(3), {})[2] is True
    assert decisions.assessments[-1].attempt_id == 1
    assert decisions.assessments[-1].accepted_revision is None
    assert decisions._attempt_count == 1
    global_config.reset()


def test_evidence_snapshot_detaches_result_and_numpy_values():
    import numpy as np
    from pathlib import Path
    from astrolabe.evaluators.evidence import CandidateGroup, DerivedStructure, EvaluationRecord, InputObservation
    from astrolabe.evaluators.utils import StandardScores

    array = np.array([1.0, 2.0])
    record = EvaluationRecord(1, 0, "one", "one", "eval", InputObservation("A", (), (1,)), Path("old.cif"))
    derived = {"A": [DerivedStructure(Path("old.cif"), Path("new.cif"), "min")]}
    groups = (CandidateGroup(((("A", Path("candidate.cif")),),), (1,)),)
    result = StandardScores(names=["A"], plddt={"A": [0.5]}, complex_paths={"A": Path("old.cif")},
                            evaluation_records=(record,), derived_structures=derived, candidate_groups=groups)
    decisions = Decisions({0: {}}, {0: 1}, {0: result}, {0: {"array": array}}, [0], 0, {0: "one"})
    assessment = decisions.assessments[0]
    result.complex_paths["A"] = Path("changed.cif")
    result.evaluation_records = ()
    result.derived_structures["A"].clear()
    result.candidate_groups = ()
    array[0] = 99
    assert assessment.score_details["array"].tolist() == [1., 2.]
    snapshot = assessment.evidence_snapshot
    assert snapshot["complex_paths"]["A"] == Path("old.cif")
    assert snapshot["evaluation_records"][0]["original_complex_path"] == Path("old.cif")
    assert snapshot["derived_structures"]["A"][0]["output_path"] == Path("new.cif")
    assert snapshot["candidate_groups"][0]["candidate_paths"][0][0][1] == Path("candidate.cif")


def test_initial_rejection_corrects_observation_only():
    decisions = Decisions({0: {}}, {0: 1}, {0: object()}, {0: {}}, [0], 0, {0: "one"})
    decisions.reject_initial_assessment()
    assert decisions.assessments[0].accepted is False
    assert decisions.assessments[0].accepted_revision is None
    assert decisions.last_accepted == 0 and decisions.accepted_steps == [0]
    assert decisions.observation_errors == []


def test_initial_and_stage_observation_failures_are_nonfatal(tmp_path, monkeypatch):
    import astrolabe.decision_maker as module
    global_config.reset()
    global_config.init_const("stage_manager", SimpleNamespace(current_stage="one"))
    global_config.init_const("out_dir", tmp_path)
    snapshot = module._snapshot_result
    monkeypatch.setattr(module, "_snapshot_result", lambda _: (_ for _ in ()).throw(ValueError("bad evidence")))
    decisions = Decisions({0: {}}, {0: 1}, {0: object()}, {0: {}}, [0], 0, {0: "one"})
    assert decisions.assessments == []
    decisions.reject_initial_assessment()
    assert [error["operation"] for error in decisions.observation_errors] == ["initial", "reject_initial_assessment"]
    monkeypatch.setattr(module, "_snapshot_result", snapshot)
    decisions.recompute_last_score(Loss(3), {})
    assert decisions.scores[0] == 3
    assert decisions.assessments[0].accepted_revision is None
    monkeypatch.setattr(module, "_snapshot_result", lambda _: (_ for _ in ()).throw(ValueError("bad evidence")))
    decisions.recompute_last_score(Loss(4), {})
    assert decisions.scores[0] == 4
    assert len(decisions.assessments) == 1
    assert decisions.observation_errors[-1]["operation"] == "stage_reassessment"
    global_config.reset()


def test_observation_failure_preserves_rng_and_policy_outcome(tmp_path, monkeypatch):
    import numpy as np
    import astrolabe.decision_maker as module

    monkeypatch.setattr(module, "calculate_temperature", lambda step: 1.)
    original_snapshot = module._snapshot_result

    def run_case(fail_observation):
        global_config.reset()
        operations = []
        for key, value in {
            "stage_manager": SimpleNamespace(current_stage="one"),
            "segment_manager": SimpleNamespace(apply_mutations=lambda: operations.append("apply"),
                                               clear_mutations=lambda: operations.append("clear")),
            "out_dir": tmp_path, "protomers": {}, "use_metropolis_acceptance": True,
        }.items():
            global_config.init_const(key, value)
        monkeypatch.setattr(module, "_snapshot_result", original_snapshot)
        decisions = Decisions({0: {}}, {0: 1}, {0: object()}, {0: {}}, [0], 0, {0: "one"})
        if fail_observation:
            monkeypatch.setattr(module, "_snapshot_result", lambda _: (_ for _ in ()).throw(RuntimeError("observer")))
        np.random.seed(12)
        early, _, accepted = decisions.make_mutation_decision(1, object(), Loss(0), {})
        return early, accepted, decisions.last_accepted, decisions.row_history, operations, float(np.random.uniform())

    assert run_case(False) == run_case(True)
    global_config.reset()


def test_initial_rejection_is_visible_to_existing_audit_builder():
    from astrolabe.execution_audit import build_execution_audit
    decisions = Decisions({0: {}}, {0: 1}, {0: object()}, {0: {}}, [0], 0, {0: "one"})
    decisions.reject_initial_assessment()
    audit = build_execution_audit(run_id="test", status="failed", decisions=decisions)
    assert audit["assessments"][0]["accepted"] is False
    assert audit["assessments"][0]["accepted_revision"] is None
