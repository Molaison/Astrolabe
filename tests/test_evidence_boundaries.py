from pathlib import Path

import pytest

from astrolabe.core.folding_input import Input, ProteinChain, Template, snapshot_folding_inputs
from astrolabe.evaluators.evidence import DerivedStructure, merge_evidence_metadata
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
    assert len(merged.candidate_scores) == 2
    other.candidate_scores = [scores('DE')]
    with pytest.raises(ValueError, match='candidate identity'):
        merge_evidence_metadata(merged, [filtered, other])
    assert len(merged.candidate_scores) == 2


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
