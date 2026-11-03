from pathlib import Path
from types import SimpleNamespace
import pytest

from astrolabe.core import global_config
from astrolabe.evaluators.utils import StandardScores
from astrolabe.losses.base import LossEngine
from astrolabe.mutators.dplm import DPLMMutator
from astrolabe.mutators.utils import build_mutator_runtime_config
from astrolabe.score_contributions import strict_score_contributions_from_runtime_config
from astrolabe.score_evidence import ScoreQuery, query_score, records_from_contributions


def test_real_dplm_quality_telemetry_enters_loss_query_without_relabeling(monkeypatch):
    global_config.reset()
    global_config.set_variable('step', 1)
    global_config.init_const('segments', {})
    model = DPLMMutator.__new__(DPLMMutator)
    model.runner = SimpleNamespace(num_samples=1, design_sequences=lambda tasks: {'AB': ['ACDE']})
    model._prefetch = {}
    model._prefetch_save = False
    model._quality_filter_enable = True
    model._quality_filter_fallback_to_raw = False
    model._quality_filter_attempt_num_samples = 1
    model._quality_filter_max_rounds = 1
    model._quality_cfg = object()
    monkeypatch.setattr('astrolabe.mutators.dplm.filter_dplm_sequences',
                        lambda sequences, **kw: (sequences, {'raw': 1, 'kept': 1, 'entropy_norm_mean': .7}))
    features = {'AB': {'combined_sequence': 'AAAA', 'masked_sequence': '<mask>AAA',
                       'mask_positions': [0], 'mutation_quota': 1}}
    catalog = [{'source': 'mutator.dplm', 'name': 'entropy_norm', 'entity': 'AB', 'entity_scope': 'oligo'}]
    accepted = StandardScores(names=['AB'], plddt={'AB': [.8]}, complex_paths={'AB': Path('ab.pdb')})
    proposal = StandardScores(names=['AB'], plddt={'AB': [.9]}, complex_paths={'AB': Path('next.pdb')})
    runtime = {'score_catalog': catalog, 'score_capture_snapshot': proposal.assessment_id,
               'score_snapshots': {'last_accepted': accepted.assessment_id, 'current_proposal': proposal.assessment_id}}
    monkeypatch.setattr(model, 'prepare_mutation_input', lambda *args: features)
    monkeypatch.setattr(model, 'prepare_mutation_output', lambda output, *args: output)
    model.mutate({}, accepted, runtime)
    measurement = runtime['score_snapshots']['producer:DPLM']
    assert measurement not in {accepted.assessment_id, proposal.assessment_id}
    record, = records_from_contributions(strict_score_contributions_from_runtime_config(runtime), catalog, proposal.assessment_id)
    assert record.input_snapshot == accepted.assessment_id and record.snapshot == measurement
    query = {'name': 'entropy_norm', 'source': 'mutator.dplm', 'entity': 'AB',
             'snapshot': 'producer:DPLM', 'subject': 'proposal', 'relation': 'proposal_on_reference'}
    engine = LossEngine({'loss_terms': {}, 'score_catalog': catalog, 'score_queries': {'entropy': query}}, validate_config=False)
    engine.setup('entropy', None, None, validate_config=False)
    value, detail = engine.compute_loss(proposal, runtime)
    assert value == pytest.approx(.7) and detail['entropy'] == pytest.approx(.7)
    with pytest.raises(ValueError, match='selector must use'):
        query_score([record], ScoreQuery(**query), snapshots=runtime['score_snapshots'], generating_selector=True)
    assert not runtime.get('score_contributions')  # no implicit legacy flattening
    build_mutator_runtime_config(2, former_config=runtime)
    assert not runtime.get('strict_score_contributions')
    assert 'producer:DPLM' not in runtime['score_snapshots']
    global_config.reset()
