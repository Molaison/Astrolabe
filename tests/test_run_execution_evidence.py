from __future__ import annotations

import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest

import run
from astrolabe.core import global_config
from astrolabe.core.folding_input import Input, ProteinChain
from astrolabe.evaluators.evidence import attach_evaluation_records, observe_evaluation_inputs
from astrolabe.evaluators.utils import StandardScores
from astrolabe.execution_audit import EvaluationJournal, write_execution_audit


@pytest.fixture(autouse=True)
def reset_state():
    global_config.reset()
    yield
    global_config.reset()


def _input(name="A"):
    return Input(name=name, chains=[ProteinChain(id="A", sequence="AAA", ptms=[])], rng_seeds=[17])


def _scores(name="A"):
    return StandardScores(names=[name], plddt={name: [0.75]}, complex_paths={name: Path(f"{name}.pdb")})


def _journal(tmp_path):
    journal = EvaluationJournal(str(tmp_path))
    global_config.init_const("evaluation_journal", journal)
    global_config.init_const("stage_manager", SimpleNamespace(
        current_stage="source", transition_consistency_gate_context=SimpleNamespace(target_stage="target")
    ))
    return journal


def test_dispatch_preserves_caller_inputs_rng_and_requested_stage(tmp_path):
    journal = _journal(tmp_path)
    inputs = {"A": _input()}
    calls = []

    class Backend:
        alias = "ObservedBackend"

        def evaluate(self, outgoing, runtime):
            calls.append(outgoing)
            outgoing["A"].rng_seeds.append(99)
            outgoing["A"].chains[0].ptms.append(("MOD", 1))
            return _scores()

    random.seed(53)
    np.random.seed(53)
    py_state, np_state = random.getstate(), np.random.get_state()
    result = run._evaluate_with_observability(
        Backend(), inputs, {}, step=1, oligo_names=["A"], detail="test"
    )
    assert len(calls) == 1
    assert inputs["A"].rng_seeds == [17]
    assert inputs["A"].chains[0].ptms == []
    assert random.getstate() == py_state
    assert np.random.get_state()[0] == np_state[0]
    assert np.array_equal(np.random.get_state()[1], np_state[1])
    assert np.random.get_state()[2:] == np_state[2:]
    record = result.evaluation_records[0]
    assert record.input.rng_seeds == (17,)
    assert record.committed_stage == "source"
    assert record.execution_stage == "target"
    assert record.sample_identity is None
    assert journal.invocations[0].status == "completed"
    assert result.plddt == {"A": [0.75]}


def test_failed_dispatch_records_failure_without_success_evidence(tmp_path):
    journal = _journal(tmp_path)

    class Broken:
        def evaluate(self, inputs, runtime):
            raise ValueError("native failure")

    with pytest.raises(ValueError, match="native failure"):
        run._evaluate_with_observability(Broken(), {"A": _input()}, {}, step=2, oligo_names=["A"], detail="test")
    assert journal.invocations[0].status == "failed"
    assert journal.invocations[0].error_type == "ValueError"
    assert journal.invocations[0].records == ()


def test_real_score_merge_keeps_request_order_without_inventing_candidates():
    results = []
    for name, invocation_id in (("A", "run:1"), ("B", "run:2")):
        score = _scores(name)
        score.candidate_scores = [_scores(name), _scores(name)]
        attach_evaluation_records(score, observe_evaluation_inputs({name: _input(name)}),
            invocation_id=invocation_id, step=1, committed_stage="default",
            execution_stage="default", backend_alias="backend")
        results.append(score)
    merged = run._merge_standard_scores(results)
    assert merged.names == ["A", "B"]
    assert merged.plddt == {"A": [0.75], "B": [0.75]}
    assert merged.candidate_scores is None
    assert [record.invocation_id for record in merged.evaluation_records] == ["run:1", "run:2"]
    assert len(merged.candidate_groups) == 2
    assert all(group.alignment == "unknown" for group in merged.candidate_groups)
    assert merged.filter_by_oligomer(["B"]).evaluation_records[0].input.oligomer == "B"


def test_component_cleanup_is_identity_deduplicated_and_failure_tolerant():
    calls = []

    class Closer:
        def shutdown(self):
            calls.append("shared")

    class Broken:
        def close(self):
            calls.append("broken")
            raise ValueError("cleanup error")

    class Last:
        def close(self):
            calls.append("last")

    shared = Closer()
    global_config.init_const("mutators", {"first": shared, "alias": shared})
    global_config.init_const("evaluators", {"shared": shared, "broken": Broken()})
    global_config.init_const("predockers", {"last": Last()})
    run._shutdown_pipeline_resources()
    assert calls == ["shared", "broken", "last"]


def test_audit_is_additive_and_explicitly_not_a_checkpoint(tmp_path):
    journal = _journal(tmp_path)
    invocation = journal.begin(step=0, committed_stage="source", execution_stage="target", backend_alias="backend", inputs=())
    journal.finish(invocation, error=RuntimeError("failed"))
    legacy = tmp_path / "scores.csv"
    legacy.write_text("step,accepted\n0,True\n")
    path = write_execution_audit(tmp_path, status="failed", journal=journal)
    payload = json.loads(path.read_text())
    assert payload["recovery"]["exact_resume"] is False
    assert payload["invocations"][0]["status"] == "failed"
    assert legacy.read_text() == "step,accepted\n0,True\n"
    assert not list(tmp_path.glob(".*.tmp"))


def test_audit_write_failure_does_not_replace_old_evidence_or_raise(tmp_path, monkeypatch):
    path = tmp_path / "execution_audit.json"
    path.write_text("old")

    def fail_dump(*args, **kwargs):
        args[1].write("partial")
        raise OSError("disk error")

    monkeypatch.setattr("astrolabe.execution_audit.json.dump", fail_dump)
    assert write_execution_audit(tmp_path, status="completed") is None
    assert path.read_text() == "old"
    assert not list(tmp_path.glob(".*.tmp"))


def test_stage_entry_observation_does_not_claim_an_acceptance(tmp_path, monkeypatch):
    journal = _journal(tmp_path)
    manager = global_config.stage_manager
    decisions = SimpleNamespace(last_accepted=4, assessments=[])
    monkeypatch.setattr(run, "begin_stage_entry_transaction", lambda *args, **kwargs: None)
    result = run._handle_stage_entry(args=None, step=4, evaluation_result=_scores(),
        decisions=decisions, loss_computer=None, runtime_config={}, stage_manager=manager)
    assert result is None
    assert decisions.last_accepted == 4
    assert journal.transitions == [{
        "step": 4, "source_stage": "source", "target_stage": None,
        "committed_stage_after": "source", "gate_active_after": True,
        "accepted_step": 4, "accepted_revision": None, "error_type": None,
    }]


def test_stage_observation_scope_never_leaks_into_backend_runtime(tmp_path, monkeypatch):
    journal = _journal(tmp_path)
    target = object()
    manager = SimpleNamespace(stage_args={"target": target}, resolve_stage_evaluator_instance=lambda unit: "backend")
    runtime = {"existing": 1}

    def evaluate(args, backend, observed_runtime, step, oligo, **kwargs):
        assert observed_runtime is runtime
        assert observed_runtime == {"existing": 1}
        assert run._execution_stage_name() == "target"
        raise ValueError("target failure")

    monkeypatch.setattr(run, "perform_evaluation", evaluate)
    with pytest.raises(ValueError, match="target failure"):
        run._perform_stage_reevaluation(args=None, stage_manager=manager,
            stage_unit=target, runtime_config=runtime, step=3, oligo=["A"])
    assert not journal.has_execution_stage
    assert runtime == {"existing": 1}
