"""AF2 factory to CPU preparation and real daemon transport with fake runner."""
import numpy as np


def test_af2_factory_cpu_prepare_resident_forward(tmp_path, monkeypatch):
    from astrolabe.evaluators.utils import _build_alphafold2_evaluator
    import astrolabe.evaluators.wrappers.alphafold as wrapper
    fake = tmp_path / 'fake_af2_runner.py'
    fake.write_text('''\
import os
import numpy as np
class Runner:
    multimer_mode = False
    def __init__(self, cfg, prepare_only):
        self._gh_initial_guess_cfg = cfg.get('initial_guess')
        if not prepare_only:
            with open(os.environ['AF2_TEST_LOADS'], 'a') as f: f.write('load\\n')
    def process_features(self, data, random_seed):
        return {'value': data['value'] + random_seed}
    def predict(self, data, random_seed):
        return {'value': np.asarray(data['value'] + random_seed)}
def setup(oligo_names, prepare_only=False, **cfg):
    runner = Runner(cfg, prepare_only)
    return {name: runner for name in oligo_names}
''')
    (tmp_path / 'sitecustomize.py').write_text('''\
import astrolabe.evaluators.wrappers.alphafold as wrapper
from fake_af2_runner import setup
wrapper.alphafold_setup = setup
''')
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv('PYTHONPATH', str(tmp_path) + ':' + __import__('os').environ.get('PYTHONPATH', ''))
    monkeypatch.setenv('AF2_TEST_LOADS', str(tmp_path / 'loads'))
    monkeypatch.delenv('ASTROLABE_EXECUTION_ENDPOINT', raising=False)
    monkeypatch.delenv('ASTROLABE_EXECUTION_TOKEN', raising=False)
    (tmp_path / 'params').mkdir()
    (tmp_path / 'params' / 'params_model_3_ptm.npz').write_bytes(b'fake')
    from fake_af2_runner import setup
    monkeypatch.setattr(wrapper, 'alphafold_setup', setup)
    evaluator = _build_alphafold2_evaluator(
        {'data_dir': str(tmp_path), 'suffix': 'ptm', 'execution': 'daemon',
         'initial_guess': {'enabled': True, 'path': 'original.pdb'}},
        alias='alphafold2', raw_model_results_dir=tmp_path, oligo_names=['A'], evaluator_conf={})
    runner = evaluator.model_runners['A']
    try:
        assert runner._gh_initial_guess_cfg['path'] == 'original.pdb'
        for value, seed in ((3, 11), (5, 17)):
            prepared = runner.process_features({'value': np.asarray(value)}, random_seed=seed)
            prediction = runner.predict(prepared, random_seed=seed)
            assert int(prediction['value']) == value + 2 * seed
        assert (tmp_path / 'loads').read_text().splitlines() == ['load']
    finally:
        runner.close()
