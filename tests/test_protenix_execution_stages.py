"""Exercise the real Protenix evaluator stages with a CPU mock runner."""
from types import SimpleNamespace
from unittest import mock

import torch


def test_forward_updates_length_and_finalize_retains_owner_atom_array():
    from astrolabe.evaluators.protenix import ProtenixEvaluator
    runner = SimpleNamespace(configs=SimpleNamespace(seed=11), dumper=SimpleNamespace(dump=mock.Mock()))
    runner.update_model_configs = mock.Mock()
    runner.predict = mock.Mock(return_value={'prediction': torch.tensor([42])})
    data = {'N_token': torch.tensor(9), 'entity_poly_type': {'1': 'protein'},
            'input_feature_dict': {'atom_to_token_idx': torch.tensor([0])}}
    atom_array = object()
    with mock.patch('astrolabe.evaluators.protenix.update_inference_configs', return_value={'length': 9}) as update:
        result = ProtenixEvaluator.forward_prepared(runner, data)
        update.assert_called_once_with(runner.configs, 9)
        runner.update_model_configs.assert_called_once_with({'length': 9})
        runner.predict.assert_called_once_with(data)
    out = ProtenixEvaluator.finalize_prediction(runner, 'step_001', 'A', data, atom_array, result)
    assert out is result
    assert out['atom_to_residue'] is data['input_feature_dict']['atom_to_token_idx']
    assert runner.dumper.dump.call_args.kwargs['atom_array'] is atom_array
    assert runner.dumper.dump.call_args.kwargs['seed'] == 11


def test_codec_result_preserves_legacy_scores_and_dump_path(tmp_path, monkeypatch):
    import json
    import numpy as np
    from pathlib import Path
    from astrolabe.core import global_config
    from astrolabe.core.folding_input import ProteinChain
    from astrolabe.evaluators.protenix import ProtenixEvaluator
    from astrolabe.evaluators.tree_codec import dump_tree, load_tree
    from astrolabe.evaluators.wrappers import protenix as wrapper

    monkeypatch.setattr(wrapper, '_lazy_import_protenix', lambda: None)
    monkeypatch.setattr(wrapper, 'save_structure_cif', lambda **kw: Path(kw['output_fpath']).write_text(str(kw['pred_coordinate'].tolist())), raising=False)
    monkeypatch.setattr(wrapper, 'save_json', lambda data, path, **kw: Path(path).write_text(json.dumps(data)), raising=False)
    runner = SimpleNamespace(configs=SimpleNamespace(seed=11, dump_dir=str(tmp_path / 'raw')),
                             dumper=wrapper.DataDumper(str(tmp_path / 'raw')))
    evaluator = ProtenixEvaluator(runner)
    monkeypatch.setattr(evaluator, '_structure_start_soft_select_enabled', lambda: False)
    global_config.reset(); global_config.init_const('out_dir', tmp_path); global_config.set_variable('step', 1)
    data = {'entity_poly_type': {'1': 'protein'}, 'input_feature_dict': {'atom_to_token_idx': torch.tensor([0, 1])}}
    atom_array = SimpleNamespace(set_annotation=lambda *args: None)
    original = {'coordinate': torch.tensor([[[1., 2., 3.], [4., 5., 6.]]]),
                'summary_confidence': [{'ranking_score': 0.9, 'ptm': 0.8, 'iptm': 0.7}],
                'full_data': [{'atom_plddt': torch.tensor([0.6, 0.9]),
                               'token_pair_pae': torch.tensor([[1., 2.], [2., 1.]]),
                               'token_pair_pde': torch.tensor([[3., 4.], [4., 3.]])}]}
    dump_tree(tmp_path / 'codec', original)
    restored = load_tree(tmp_path / 'codec')
    inputs = {'A': SimpleNamespace(name='A', chains=[ProteinChain(id='A', sequence='AC', ptms=[])])}
    def run(prediction):
        out = evaluator.finalize_prediction(runner, 'step_001', 'A', data, atom_array, prediction)
        return evaluator.prepare_output({'A': out}, inputs, None)
    legacy = run(original); reconstructed = run(restored)
    assert legacy.complex_paths == reconstructed.complex_paths
    assert legacy.complex_paths['A'].name == 'step_001_A_model.cif'
    assert legacy.complex_paths['A'].read_text() == '[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]'
    np.testing.assert_array_equal(legacy.plddt['A'], reconstructed.plddt['A'])
    np.testing.assert_array_equal(legacy.pae['A'], reconstructed.pae['A'])
    assert legacy.ptm == reconstructed.ptm and legacy.iptm == reconstructed.iptm
