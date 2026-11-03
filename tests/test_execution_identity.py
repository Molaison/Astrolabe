import random
from types import SimpleNamespace

import numpy as np
import pytest

from astrolabe.core import global_config
from astrolabe.execution_identity import (current_execution_identity, input_digest,
                                          scientific_operation, stage_transaction_identity)


@pytest.fixture
def owner(monkeypatch):
    global_config.reset()
    segment = SimpleNamespace()
    protomer = SimpleNamespace(sequence="AAAA", mutated_sequence="AAAW", components=["shared"])
    accepted = SimpleNamespace(assessment_id="accepted-7")
    decisions = SimpleNamespace(last_accepted=7, evaluation_results={7: accepted})
    for key, value in {"out_dir": "/run/trajectory", "segments": {"shared": segment, "alias": segment},
                       "protomers": {"A": protomer}, "decisions": decisions,
                       "stage_manager": SimpleNamespace(current_stage="design")}.items():
        global_config.init_const(key, value)
    global_config.set_variable("step", 9)
    monkeypatch.setenv("ASTROLABE_EXECUTION_ENDPOINT", "@test-only-context")
    yield protomer
    global_config.reset()


def test_owner_request_binds_revision_and_reference_without_consuming_rng(owner):
    random.seed(17)
    np.random.seed(17)
    py, numpy = random.getstate(), np.random.get_state()
    with scientific_operation("mutator", "MPNN", inputs={"coordinates": np.zeros((4, 3))},
                              reference=SimpleNamespace(assessment_id="reference-3"),
                              runtime_config={"transition_consistency_gate_metadata": {
                                  "source_stage": "design", "target_stage": "refine", "attempt": 2}}) as request:
        assert request["accepted_revision"] == 7 and request["step"] == 9
        assert request["reference_assessment"] == "reference-3"
        assert request["stage_transaction"]["target_stage"] == "refine"
        copied = current_execution_identity()
        copied["accepted_revision"] = -1
        assert current_execution_identity()["accepted_revision"] == 7
    assert current_execution_identity() is None
    assert random.getstate() == py
    after = np.random.get_state()
    assert np.array_equal(numpy[1], after[1]) and numpy[2:] == after[2:]


def test_late_owner_change_is_fenced_and_context_is_discarded(owner):
    with pytest.raises(RuntimeError, match="stale scientific result"):
        with scientific_operation("evaluator", "AF3", inputs={"sequence": "AAAA"}):
            owner.mutated_sequence = "WWWW"
    assert current_execution_identity() is None


def test_snapshot_digest_tracks_bits_order_and_nested_mutation():
    first = {"chain_order": ["A", "B"], "features": np.array([0.0, -0.0], dtype=np.float32)}
    before = input_digest(first)
    first["features"][1] = 0.0
    assert input_digest(first) != before
    first["features"][1] = -0.0
    first["chain_order"].reverse()
    assert input_digest(first) != before
    with pytest.raises(TypeError, match="unsupported live request"):
        input_digest(SimpleNamespace(sequence="AAAA"))


def test_seed_gate_identity_exists_before_gate_result_and_cannot_leak(owner):
    manager = global_config.get_const("stage_manager")
    plan = SimpleNamespace(source_stage_name="design", target_stage_name="refine",
                           source_evaluator_alias="AF3", target_evaluator_alias="AF2",
                           gate_evaluator_alias="AF3")
    with stage_transaction_identity(plan, manager):
        with scientific_operation("evaluator", "AF3", inputs="seed") as request:
            assert request["committed_stage"] == "design"
            assert request["stage_transaction"]["target_stage"] == "refine"
            assert request["stage_transaction"]["attempt"] == 1
    with scientific_operation("evaluator", "AF3", inputs="ordinary") as request:
        assert request["stage_transaction"] == {}


def test_active_gate_attempt_comes_from_live_owner_and_changes_are_fenced(owner):
    manager = global_config.get_const("stage_manager")
    manager.transition_consistency_gate_context = SimpleNamespace(source_stage="design", target_stage="refine")
    manager.transition_consistency_gate_attempts = 3
    with pytest.raises(RuntimeError, match="stale scientific result"):
        with scientific_operation("mutator", "MPNN", inputs="proposal",
                                  runtime_config={"transition_consistency_gate_metadata": {"attempt": 2}}) as request:
            assert request["stage_transaction"]["attempt"] == 4
            manager.transition_consistency_gate_attempts = 4


def test_file_content_and_symlink_target_are_scientific_dependencies(owner, tmp_path):
    source = tmp_path / "reference.pdb"
    source.write_text("old")
    first = input_digest({"pdb_path": str(source)})
    source.write_text("new")
    assert input_digest({"pdb_path": str(source)}) != first
    assert input_digest({"pdb_path": str(source)}) != input_digest({"pdb_path": source})
    with pytest.raises(RuntimeError, match="input file changed"):
        with scientific_operation("mutator", "MPNN", inputs={"pdb_path": str(source)}):
            source.write_text("after request")
    sequence = str(source)
    before = input_digest({"sequence": sequence})
    source.write_text("another")
    assert input_digest({"sequence": sequence}) == before
    alias = tmp_path / "alias.pdb"
    other = tmp_path / "other.pdb"
    other.write_text(source.read_text())
    alias.symlink_to(source)
    with pytest.raises(RuntimeError, match="input file changed"):
        with scientific_operation("mutator", "MPNN", inputs={"pdb_path": alias}):
            alias.unlink()
            alias.symlink_to(other)
