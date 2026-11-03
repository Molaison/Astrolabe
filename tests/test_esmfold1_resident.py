"""Exercise the real ESMFold1 factory and daemon with fake local weights."""
from types import SimpleNamespace

import pytest
import torch


def test_esmfold1_factory_resident_reuses_model(tmp_path, monkeypatch):
    from astrolabe.core import global_config
    from astrolabe.evaluators.utils import _build_esmfold_evaluator
    fake = tmp_path / 'fake_esm.py'
    fake.write_text('''\
import os
import torch
class Model:
    def __init__(self, esmfold_config):
        with open(os.environ['ESM1_TEST_LOADS'], 'a') as f: f.write('load\\n')
    def state_dict(self): return {}
    def load_state_dict(self, state, strict=False): pass
    def eval(self): return self
    def to(self, device): return self
    def infer(self, sequences):
        n = len(sequences[0])
        return {'sequence': sequences[0], 'chain_index': torch.zeros((1,n),dtype=torch.int64), 'plddt': torch.ones((1,n,2))*80, 'mean_plddt': torch.tensor([80.]), 'ptm': torch.tensor([.7]), 'predicted_aligned_error': torch.ones((1,n,n))}
    def output_to_pdb(self, output): return [output['sequence']]
''')
    (tmp_path / 'sitecustomize.py').write_text('''\
import astrolabe.evaluators.esmfold as module
from fake_esm import Model
module._load_esmfold_model_class = lambda: Model
''')
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv('PYTHONPATH', str(tmp_path) + ':' + __import__('os').environ.get('PYTHONPATH', ''))
    monkeypatch.setenv('ESM1_TEST_LOADS', str(tmp_path / 'loads'))
    monkeypatch.setenv('TORCH_HOME', str(tmp_path / 'torch-home'))
    monkeypatch.delenv('ASTROLABE_EXECUTION_ENDPOINT', raising=False)
    monkeypatch.delenv('ASTROLABE_EXECUTION_TOKEN', raising=False)
    weights = tmp_path / 'weights.pt'
    torch.save({'cfg': {'model': {}}, 'model': {}}, weights)
    global_config.reset()
    global_config.init_const('out_dir', tmp_path)
    global_config.init_const('protomers', {'A': SimpleNamespace(length=3)})
    global_config.set_variable('step', 1)
    (tmp_path / 'structures').mkdir()
    evaluator = _build_esmfold_evaluator(
        {'model_name': str(weights), 'device': 'cpu', 'execution': 'daemon'},
        alias='esmfold', raw_model_results_dir=tmp_path, oligo_names=['A'], evaluator_conf={})
    try:
        first = evaluator.run_function({'A': 'AAA'})
        second = evaluator.run_function({'A': 'CCC'})
        assert first.plddt['A'] == pytest.approx([.8])
        assert second.plddt['A'] == pytest.approx([.8])
        assert second.complex_paths['A'].read_text() == 'CCC\n'
        assert (tmp_path / 'loads').read_text().splitlines() == ['load']
    finally:
        evaluator.close()
        global_config.reset()
