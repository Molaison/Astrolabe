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
