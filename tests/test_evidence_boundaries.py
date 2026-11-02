from pathlib import Path

import pytest

from astrolabe.core.folding_input import Input, ProteinChain, Template, snapshot_folding_inputs
from astrolabe.evaluators.evidence import (
    DerivedStructure, attach_evaluation_records, merge_evidence_metadata,
    observe_evaluation_inputs,
)
from astrolabe.evaluators.utils import StandardScores
from astrolabe.post_eval_hooks import rosetta_minimize


def scores(*names):
    return StandardScores(names=list(names), plddt={n: [0.8] for n in names},
                          complex_paths={n: Path(n + '.cif') for n in names})


def test_clone_filter_and_merge_declared_metadata():
    original = scores('AB', 'AC')
    original.candidate_scores = [scores('AB', 'AC'), scores('AB', 'AC')]
    record = DerivedStructure(Path('AB.cif'), Path('AB.pdb'), 'repack')
    original.derived_structures = {'AB': [record], 'AC': []}
    original.post_eval_hook_reports = {'AB': {'ran': True}, 'AC': {'ran': False}}
    clone = original.clone()
    assert clone.candidate_scores is original.candidate_scores
    filtered = clone.filter_by_oligomer(['AB'])
    assert filtered.derived_structures == {'AB': [record]}
    assert filtered.post_eval_hook_reports == {'AB': {'ran': True}}
    assert [c.names for c in filtered.candidate_scores] == [['AB'], ['AB']]
    assert filtered.candidate_scores[0].complex_paths == {'AB': Path('AB.cif')}
    other = scores('DE')
    other.post_eval_hook_reports = {'DE': {'ran': False}}
    merged = merge_evidence_metadata(scores('AB', 'DE'), [filtered, other])
    assert list(merged.post_eval_hook_reports) == ['AB', 'DE']
    assert merged.derived_structures == {'AB': [record]}
    assert merged.candidate_scores is None
    assert merged.candidate_groups[0].alignment == 'unknown'
    other.candidate_scores = [scores('DE')]
    merge_evidence_metadata(merged, [filtered, other])
    assert merged.candidate_scores is None
    assert len(merged.candidate_groups) == 2
    with pytest.raises(ValueError, match='candidate identity'):
        merge_evidence_metadata(merged, [filtered, other], strict=True)
    assert len(merged.candidate_groups) == 2


def test_snapshot_isolates_nested_lists_and_preserves_backend_types():
    template = Template(mmcif='data', query_to_template_map={0: 1})
    chain = ProteinChain(id='A', sequence='AA', ptms=[('p', 1)], templates=[template])
    original = {'AB': Input(name='job', chains=[chain], rng_seeds=[1],
                            bonded_atom_pairs=[[['A', 1, 'C'], ['B', 2, 'N']]])}
    snapshot = snapshot_folding_inputs(original)
    assert isinstance(snapshot['AB'], Input)
    assert isinstance(snapshot['AB'].chains, list)
    assert isinstance(snapshot['AB'].chains[0].ptms, list)
    snapshot['AB'].chains[0].ptms.append(('x', 2))
    snapshot['AB'].chains[0].templates[0].query_to_template_map[0] = 9
    snapshot['AB'].bonded_atom_pairs[0][0][1] = 8
    original['AB'].rng_seeds.append(2)
    assert chain.ptms == [('p', 1)]
    assert template.query_to_template_map == {0: 1}
    assert original['AB'].bonded_atom_pairs[0][0][1] == 1
    assert snapshot['AB'].rng_seeds == [1]


def test_real_hook_tracks_only_changed_structures_and_keeps_scores(monkeypatch, tmp_path):
    result = scores('AB', 'AC')
    def minimize(source, output, **kwargs):
        ran = source.name == 'AB.cif'
        return (output if ran else source), {'ran': ran, 'mode': kwargs['mode'],
                'energy_before': 1., 'energy_after': 0. if ran else 1.}
    monkeypatch.setattr(rosetta_minimize, '_minimize_path', minimize)
    out = rosetta_minimize.apply_rosetta_minimize_hook(result, {}, out_dir=tmp_path, step=2)
    assert out.plddt == {'AB': [0.8], 'AC': [0.8]}
    assert out.complex_paths['AB'] == tmp_path / 'step_002_AB_repack.pdb'
    assert out.derived_structures == {'AB': [DerivedStructure(Path('AB.cif'),
        tmp_path / 'step_002_AB_repack.pdb', 'rosetta_minimize:repack')]}
    assert out.post_eval_hook_reports['AC']['ran'] is False


def test_metadata_is_not_implicitly_merged_as_score_fields():
    # run.py merges dataclass fields generically; metadata requires explicit wiring.
    assert not {'candidate_scores', 'post_eval_hook_reports', 'derived_structures'} & set(
        StandardScores.__dataclass_fields__
    )


def test_hook_failure_does_not_publish_partial_metadata(monkeypatch, tmp_path):
    result = scores('AB', 'AC')
    previous = {'AB': [DerivedStructure(Path('old'), Path('AB.cif'), 'previous')]}
    result.derived_structures = previous
    def minimize(source, output, **kwargs):
        if source.name == 'AC.cif':
            raise RuntimeError('failed')
        return output, {'ran': True, 'mode': kwargs['mode'],
                        'energy_before': 1., 'energy_after': 0.}
    monkeypatch.setattr(rosetta_minimize, '_minimize_path', minimize)
    with pytest.raises(RuntimeError, match='failed'):
        rosetta_minimize.apply_rosetta_minimize_hook(result, {}, out_dir=tmp_path, step=1)
    assert result.complex_paths == {'AB': Path('AB.cif'), 'AC': Path('AC.cif')}
    assert result.derived_structures is previous


def test_dispatch_observations_survive_candidate_selection_clone_filter_and_hook(monkeypatch, tmp_path):
    chain_a = ProteinChain(id='A', sequence='ACT', ptms=[])
    chain_b = ProteinChain(id='B', sequence='GG', ptms=[])
    inputs = {'AB': Input(name='job', chains=[chain_a, chain_b], rng_seeds=[7, 9])}
    observed = observe_evaluation_inputs(inputs)

    # A backend may mutate the dispatched containers during evaluation.
    def evaluate(dispatched):
        dispatched['AB'].rng_seeds.append(99)
        result = scores('AB')
        result.candidate_scores = [scores('AB')]
        result.candidate_scores[0].complex_paths = {'AB': Path('candidate.cif')}
        return result
    result = evaluate(inputs)
    attach_evaluation_records(result, observed, invocation_id=12, step=3,
                              committed_stage='old', execution_stage='new',
                              backend_alias='af3')
    record = result.evaluation_records[0]
    assert (record.invocation_id, record.step, record.committed_stage,
            record.execution_stage, record.backend_alias) == (12, 3, 'old', 'new', 'af3')
    assert record.input.chains == (('A', 'ProteinChain', 'ACT'), ('B', 'ProteinChain', 'GG'))
    assert record.input.rng_seeds == (7, 9)
    assert record.seed_semantics == 'dispatched_input_rng_seeds'
    assert record.original_complex_path == Path('AB.cif')
    assert record.sample_identity is None
    with pytest.raises((AttributeError, TypeError)):
        record.step = 5

    # Selecting a candidate retains invocation context without claiming its path
    # is the canonical output or that either path has a known sample identity.
    selected = result.candidate_scores[0].clone().filter_by_oligomer(['AB'])
    assert selected.evaluation_records == (record,)
    def minimize(source, output, **kwargs):
        return output, {'ran': True, 'mode': kwargs['mode'],
                        'energy_before': 1., 'energy_after': 0.}
    monkeypatch.setattr(rosetta_minimize, '_minimize_path', minimize)
    rosetta_minimize.apply_rosetta_minimize_hook(selected, {}, out_dir=tmp_path, step=3)
    assert selected.evaluation_records == (record,)
    assert selected.derived_structures['AB'][0].source_path == Path('candidate.cif')
    assert selected.complex_paths['AB'] != record.original_complex_path


def test_records_and_candidate_groups_merge_in_request_order_without_pairing():
    results = []
    for invocation, oligo in [(2, 'AB'), (3, 'AC')]:
        inputs = {oligo: Input(name=oligo, chains=[ProteinChain(id='A', sequence=oligo, ptms=[])],
                               rng_seeds=[invocation])}
        result = scores(oligo)
        result.candidate_scores = [scores(oligo), scores(oligo)]
        attach_evaluation_records(result, observe_evaluation_inputs(inputs), invocation_id=invocation,
                                  step=1, committed_stage=None, execution_stage=None, backend_alias='dummy')
        results.append(result)
    # Mirrors the old multi-result canonical output: no candidate list.
    merged = merge_evidence_metadata(scores('AB', 'AC'), results)
    assert merged.candidate_scores is None
    assert [r.invocation_id for r in merged.evaluation_records] == [2, 3]
    assert [g.invocation_ids for g in merged.candidate_groups] == [(2,), (3,)]
    assert all(g.alignment == 'unknown' and len(g.candidate_paths) == 2
               for g in merged.candidate_groups)
    filtered = merged.clone().filter_by_oligomer(['AC'])
    assert [r.invocation_id for r in filtered.evaluation_records] == [3]
    assert [g.invocation_ids for g in filtered.candidate_groups] == [(3,)]
    assert len(filtered.candidate_groups[0].candidate_paths) == 2
    # Single result remains the same canonical object/list as the old merge.
    candidate_list = results[0].candidate_scores
    assert merge_evidence_metadata(results[0], [results[0]]).candidate_scores is candidate_list


def test_actual_candidate_selection_retains_invocation_observation(monkeypatch):
    from types import SimpleNamespace
    from astrolabe import structure_start
    from astrolabe.core import global_config

    wrapper = scores('AB')
    wrapper.candidate_scores = [scores('AB'), scores('AB')]
    wrapper.candidate_scores[0].complex_paths = {'AB': Path('first.cif')}
    wrapper.candidate_scores[1].complex_paths = {'AB': Path('second.cif')}
    inputs = {'AB': Input(name='AB', chains=[ProteinChain(id='A', sequence='AA', ptms=[])],
                          rng_seeds=[5])}
    attach_evaluation_records(wrapper, observe_evaluation_inputs(inputs), invocation_id=1,
                              step=0, committed_stage='s', execution_stage='s', backend_alias='dummy')
    def gate(candidate, **kwargs):
        return structure_start.ReferenceGateResult(
            gate_active=True, score=1.0 if candidate is wrapper.candidate_scores[1] else 0.0,
            threshold=0.0, passed=True, comparisons={})
    monkeypatch.setattr(structure_start, 'evaluate_structure_start_gate', gate)
    global_config.reset()
    global_config.init_const('structure_start', SimpleNamespace(config=SimpleNamespace(mode='soft_select')))
    try:
        selected, decision = structure_start.select_structure_start_candidate(wrapper, timing='step0')
    finally:
        global_config.reset()
    assert selected is wrapper.candidate_scores[1]
    assert decision.selected_candidate_index == 1
    assert selected.evaluation_records == wrapper.evaluation_records
    assert selected.evaluation_records[0].original_complex_path == Path('AB.cif')
    assert selected.evaluation_records[0].sample_identity is None
