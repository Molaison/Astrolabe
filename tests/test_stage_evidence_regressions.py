"""Behavioral regressions independently reproduced during cross-review."""
from pathlib import Path
from types import SimpleNamespace
import pytest
import run
from astrolabe.core import global_config as cfg
from astrolabe.core.folding_input import Input, ProteinChain
from astrolabe.evaluators.utils import StandardScores
from astrolabe.execution_audit import EvaluationJournal, build_execution_audit
from astrolabe.decision_maker import Decisions
from astrolabe.structure_start import ReferenceGateResult


def scores(path='a.cif'):
    return StandardScores(names=['A'], plddt={'A': [0.8]}, complex_paths={'A': Path(path)})

def input_data():
    return {'A': Input(name='x', chains=[ProteinChain(id='A',sequence='AA',ptms=[])],rng_seeds=[17])}

def setup_journal(tmp_path):
    cfg.reset()
    cfg.init_const('out_dir', tmp_path)
    cfg.init_const('protomers', {})
    cfg.init_const('stage_manager', SimpleNamespace(current_stage='source', transition_consistency_gate_context=None))
    journal = EvaluationJournal('review')
    cfg.init_const('evaluation_journal', journal)
    return journal


def test_failed_step0_reference_gate_audit_is_rejected(tmp_path, monkeypatch):
    journal = setup_journal(tmp_path)
    def prepare(result, *, timing, loss_computer, runtime_config):
        runtime_config['reference_gate_result'] = ReferenceGateResult(gate_active=True,score=0.1,threshold=0.8,passed=False,comparisons={})
        return result
    monkeypatch.setattr(run, '_prepare_structure_start_runtime', prepare)
    loss = SimpleNamespace(compute_loss=lambda result, runtime: (1., {}))
    with pytest.raises(ValueError, match='step0 reference gate failed'):
        run._initialize_optimization_state(SimpleNamespace(steps=1), loss, scores(), cfg.stage_manager, {})
    decisions = cfg.get_const('decisions')
    assert decisions.row_history[0]['accepted'] is False
    report = build_execution_audit(run_id='review',status='failed',decisions=decisions,journal=journal)
    assert report['assessments'][0]['accepted'] is False


def test_seed_gate_invocation_reports_target_stage(tmp_path, monkeypatch):
    journal = setup_journal(tmp_path)
    manager = cfg.stage_manager
    manager.resolve_transition_consistency_gate_evaluator_instance = lambda target: Backend()
    plan = SimpleNamespace(target_stage_name='target', needs_stage_reevaluation=True,target_oligo=['A'])
    entry = SimpleNamespace(plan=plan,previous_stage_result=scores(),decision_runtime=lambda runtime: runtime)
    monkeypatch.setattr(run, 'preserve_seed_transition_source', lambda *a, **k: SimpleNamespace(previous_stage_result=scores(),artifact_root=None))
    class Stop(Exception): pass
    monkeypatch.setattr(run, 'begin_seed_transition_gate_attempt', lambda *a, **k: (_ for _ in ()).throw(Stop()))
    monkeypatch.setattr(run, 'perform_evaluation', lambda args, backend, runtime, step, oligos: run._evaluate_with_observability(backend,input_data(),runtime,step=step,oligo_names=oligos,detail='seed'))
    with pytest.raises(Stop):
        run._handle_transition_gate_stage_entry(args=None,step=4,decisions=None,loss_computer=None,runtime_config={},stage_manager=manager,stage_entry=entry)
    assert journal.invocations[0].committed_stage == 'source'
    assert journal.invocations[0].execution_stage == 'target'


class Backend:
    def evaluate(self, inputs, runtime):
        return scores()


def test_accepted_preupdate_survives_later_stage_entry_failure(tmp_path, monkeypatch):
    from astrolabe.core.protomer_config_parser import Segment, SegmentManager

    setup_journal(tmp_path)
    segments = SegmentManager()
    segment = Segment("AAA", ptm_dict={})
    segments.add_segment("A", segment)
    cfg.init_const("segment_manager", segments)
    manager = cfg.stage_manager
    manager.current_mutators = []
    manager.current_evaluator = Backend()
    manager.current_oligo = ["A"]
    manager.current_transition_input_scores = None
    decisions = SimpleNamespace(scores_detail={}, last_accepted=0)

    def mutate(*args):
        segments.begin_pre_update_step()
        segment.add_mutations((0, "A", "G", 0.1, 0.0))
        segments.preapply_pending_mutations()

    def accept(*args):
        decisions.last_accepted = 1
        return False, scores(), True

    def fail_stage(**kwargs):
        raise ValueError("stage observation failed")

    monkeypatch.setattr(run, "execute_mutation_step", mutate)
    monkeypatch.setattr(run, "perform_evaluation", lambda *a, **k: scores())
    monkeypatch.setattr(run, "_prepare_structure_start_runtime", lambda value, **k: value)
    monkeypatch.setattr(run, "_make_native_decision", accept)
    monkeypatch.setattr(run, "_handle_stage_entry", fail_stage)
    with pytest.raises(ValueError, match="stage observation failed"):
        run._advance_optimization_state(
            SimpleNamespace(steps=2), None, scores(), manager,
            {"mutator_stack_pre_update": True}, decisions, 1,
        )
    assert decisions.last_accepted == 1
    assert segment.sequence == "GAA"
    assert segments._pre_update_snapshot is None


def test_single_dispatch_candidates_appear_in_assessment_audit(tmp_path):
    journal=setup_journal(tmp_path)
    class Multi:
        def evaluate(self, inputs, runtime):
            result=scores('canonical.cif')
            result.candidate_scores=[scores('candidate0.cif'), scores('candidate1.cif')]
            return result
    result = run._evaluate_with_observability(Multi(),input_data(),{},step=0,oligo_names=['A'],detail='initial')
    decisions=Decisions({0:{}},{0:1.},{0:result},{0:{}},[0],0,{0:'source'})
    audit=build_execution_audit(run_id='review',status='completed',decisions=decisions,journal=journal)
    text=repr(audit)
    assert 'candidate0.cif' in text and 'candidate1.cif' in text




@pytest.fixture(autouse=True)
def reset_review_state():
    cfg.reset()
    yield
    cfg.reset()
