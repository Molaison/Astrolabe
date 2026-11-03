"""Mol-1 production JSONL handler, with fake pinned upstream model APIs."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


def test_mol1_end_to_end_resident_scores_and_input_guards(tmp_path, monkeypatch):
    from astrolabe.core import global_config
    from astrolabe.core.folding_input import ProteinChain, DnaChain
    from astrolabe.evaluators.mol1 import Mol1Evaluator
    from astrolabe.evaluators.mol1_resident import Mol1Runner
    source = tmp_path / 'source'; package = source / 'mol'
    (package / 'config').mkdir(parents=True); (package / 'runner').mkdir()
    for p in (package / '__init__.py', package / 'config' / '__init__.py', package / 'runner' / '__init__.py'):
        p.write_text('')
    root = tmp_path / 'assets'; root.mkdir()
    for name in ('model.pt', 'ccd.cif', 'ccd.pkl', 'clusters.txt', 'obsolete.csv'):
        (root / name).write_text(name)
    (package / 'config' / 'base.py').write_text('configs = {}\n')
    (package / 'config' / 'data.py').write_text('data_configs = {}\n')
    (package / 'config' / 'inference.py').write_text('inference_configs = {}\n')
    (package / 'config' / 'parser.py').write_text('''\
import os
from pathlib import Path
from types import SimpleNamespace
def parse_configs(**kwargs):
    root = Path(os.environ['MOL_ROOT_DIR'])
    return SimpleNamespace(data=SimpleNamespace(ccd_components_file=str(root / 'ccd.cif'), ccd_components_rdkit_mol_file=str(root / 'ccd.pkl'), pdb_cluster_file=str(root / 'clusters.txt'), obsolete_release_data_csv=str(root / 'obsolete.csv')))
''')
    (package / 'runner' / 'batch_inference.py').write_text('''\
import os
from types import SimpleNamespace
import torch
from mol.config.parser import parse_configs
inference_configs = {}
def download_inference_cache(cfg): raise AssertionError('must not download')
def get_default_runner(**kwargs):
    assert kwargs['use_seeds_in_json'] and kwargs['model_name'] == 'mol-1-base'
    assert kwargs['n_cycle'] == 20 and kwargs['n_sample'] == 1
    data = parse_configs().data
    cfg = SimpleNamespace(data=data, model_name=kwargs['model_name'], load_checkpoint_path=kwargs['load_checkpoint_path'], dump_dir=inference_configs['dump_dir'])
    download_inference_cache(cfg)
    with open(os.environ['FAKE_MOL1_LOADS'], 'a') as f: f.write('load\\n')
    from mol.runner.inference import InferenceRunner
    runner = InferenceRunner()
    runner.configs, runner.device, runner.dumper = cfg, torch.device('cpu'), SimpleNamespace(base_dir=cfg.dump_dir)
    runner.load_checkpoint()
    return runner
def preprocess_input(input_json, **kwargs): return input_json
''')
    (package / 'runner' / 'inference.py').write_text('''\
import json
from pathlib import Path
class InferenceRunner:
    def load_checkpoint(self):
        import torch
        self.model = torch.nn.Linear(1, 1)
def infer_predict(runner, cfg):
    job = json.loads(Path(cfg.input_json_path).read_text())[0]
    if job['sequences'][0]['proteinChain']['sequence'] == 'GG': return
    for seed in job['modelSeeds']:
        out = Path(cfg.dump_dir) / job['name'] / ('seed_' + str(seed)) / 'predictions'
        out.mkdir(parents=True)
        name = job['name']
        (out / (name + '_summary_confidence_sample_0.json')).write_text(json.dumps({'plddt': 90.0, 'ptm': .8, 'iptm': .7, 'chain_plddt': [.6] * len(job['sequences'])}))
        (out / (name + '_sample_0.cif')).write_text(job['sequences'][0]['proteinChain']['sequence'])
        lengths = [len(item['proteinChain']['sequence']) for item in job['sequences']]
        n = sum(lengths)
        full = {'atom_plddt': [0.4, 0.8] * n, 'atom_to_token_idx': [i for i in range(n) for _ in range(2)], 'token_asym_id': [i+1 for i,length in enumerate(lengths) for _ in range(length)]}
        (out / (name + '_full_data_sample_0.json')).write_text(json.dumps(full))
''')
    monkeypatch.setenv('FAKE_MOL1_LOADS', str(tmp_path / 'loads'))
    monkeypatch.delenv('ASTROLABE_EXECUTION_ENDPOINT', raising=False)
    monkeypatch.delenv('ASTROLABE_EXECUTION_TOKEN', raising=False)
    global_config.reset(); global_config.init_const('out_dir', tmp_path); global_config.set_variable('step', 1)
    runner = Mol1Runner(str(source), str(root), str(root / 'model.pt'), python_bin=sys.executable, num_samples=1, need_atom_confidence=True)
    evaluator = Mol1Evaluator(runner, tmp_path / 'raw')
    try:
        for seed, seq in ((11, 'AC'), (22, 'CA')):
            fold = SimpleNamespace(chains=[ProteinChain(id='A', sequence=seq, ptms=[])], rng_seeds=[seed])
            inputs = {'A': fold}
            output = evaluator.run_function(evaluator.prepare_input(inputs, None))
            scores = evaluator.prepare_output(output, inputs, None)
            assert scores.plddt['A'] == [0.9]
            np.testing.assert_allclose(scores.per_proto_scores['A'], [[0.6, 0.6]])
            assert scores.complex_paths['A'].name == 'step_001_A_model.cif'
            assert scores.complex_paths['A'].read_text() == seq
            runner.need_atom_confidence = False
            summary_only = evaluator.prepare_output(output, inputs, None)
            assert summary_only.per_proto_scores is None
            runner.need_atom_confidence = True
        fold = SimpleNamespace(chains=[ProteinChain(id='A', sequence='AC', ptms=[]), ProteinChain(id='B', sequence='AC', ptms=[])], rng_seeds=[33])
        inputs = {'AA': fold}
        scores = evaluator.prepare_output(evaluator.run_function(evaluator.prepare_input(inputs, None)), inputs, None)
        assert set(scores.per_proto_scores) == {'A'}
        assert len(scores.per_proto_scores['A']) == 2
        assert (tmp_path / 'loads').read_text().splitlines() == ['load']
        assert runner._resident._client.hello['load_count'] == 1
        runner.dtype = 'fp32'
        with pytest.raises(ValueError, match='configuration changed'):
            evaluator.run_function(evaluator.prepare_input(inputs, None))
        runner.dtype = 'bf16'
        from alphafold3.common.folding_input import ProteinChain as AF3ProteinChain
        af3_chain = AF3ProteinChain(id='A', sequence='AC', ptms=[], unpaired_msa='', paired_msa='', templates=[])
        converted = evaluator.prepare_input({'A': SimpleNamespace(chains=[af3_chain], rng_seeds=[33])}, None)
        assert json.loads(converted['A'].json_path.read_text())[0]['sequences'][0]['proteinChain']['sequence'] == 'AC'
        from astrolabe.runtime.client import WorkerError
        bad_output = SimpleNamespace(chains=[ProteinChain(id='A', sequence='GG', ptms=[])], rng_seeds=[44])
        with pytest.raises(WorkerError, match='incomplete outputs'):
            evaluator.run_function(evaluator.prepare_input({'A': bad_output}, None))
        assert (tmp_path / 'loads').read_text().splitlines() == ['load']
        bad = SimpleNamespace(chains=[DnaChain(id='A', sequence='AC', modifications=[])], rng_seeds=[1])
        with pytest.raises(ValueError, match='protein chains only'):
            evaluator.prepare_input({'A': bad}, None)
        bad = SimpleNamespace(chains=[ProteinChain(id='A', sequence='AC', ptms=[])], rng_seeds=[1], cyclic_chain_ids=['A'])
        with pytest.raises(ValueError, match='topology constraints'):
            evaluator.prepare_input({'A': bad}, None)
    finally:
        runner.close()


def test_mol1_registry_uses_own_model(monkeypatch, tmp_path):
    from omegaconf import OmegaConf
    from astrolabe.evaluators.utils import construct_evaluators
    conf = OmegaConf.create({'evaluator_terms': {'m': {'_type': 'mol1', 'repo_path': str(tmp_path), 'mol_root_dir': str(tmp_path), 'load_checkpoint_path': str(tmp_path / 'model.pt')}}})
    result = construct_evaluators(conf, raw_model_results_dir=tmp_path)
    assert result['m'].runner.model_name == 'mol-1-base'
