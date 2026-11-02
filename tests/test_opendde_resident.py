"""Real JSONL worker and pinned OpenDDE API shape with a tiny fake model."""
import json
from pathlib import Path
import sys

import pytest


def test_opendde_resident_reuses_model_and_isolates_requests(tmp_path, monkeypatch):
    from astrolabe.evaluators.wrappers.opendde import OpenDDERunner, OpenDDEPreparedInput, collect_opendde_outputs
    root = tmp_path / 'source'; (root / 'opendde' / 'config').mkdir(parents=True); (root / 'runner').mkdir()
    for p in (root / 'opendde' / '__init__.py', root / 'opendde' / 'config' / '__init__.py', root / 'runner' / '__init__.py'):
        p.write_text('')
    assets = tmp_path / 'assets'; assets.mkdir()
    for name in ('model.pt', 'ccd.cif', 'ccd.pkl'):
        (assets / name).write_text(name)
    (root / 'opendde' / 'config' / 'inference.py').write_text('''\
import os
from types import SimpleNamespace
from pathlib import Path
def build_inference_config(model_name, fill_required_with_null=True):
    root = Path(os.environ['OPENDDE_ROOT_DIR'])
    return SimpleNamespace(data=SimpleNamespace(ccd_components_file=str(root / 'ccd.cif'), ccd_components_rdkit_mol_file=str(root / 'ccd.pkl')))
''')
    (root / 'runner' / 'batch_inference.py').write_text('''\
import os
from pathlib import Path
from types import SimpleNamespace
import torch
from opendde.config.inference import build_inference_config
def download_inference_cache(configs): raise AssertionError('must not download')
def get_default_runner(**kwargs):
    base = build_inference_config(kwargs['model_name'])
    cfg = SimpleNamespace(data=base.data, model_name=kwargs['model_name'], load_checkpoint_path=kwargs['load_checkpoint_path'], use_template=kwargs['use_template'], seeds=None, dump_dir=kwargs['dump_dir'])
    assert kwargs['n_cycle'] == 10 and kwargs['n_step'] == 200 and kwargs['n_sample'] == 1
    download_inference_cache(cfg)
    with open(os.environ['FAKE_OPENDDE_LOADS'], 'a') as f: f.write('load\\n')
    return SimpleNamespace(configs=cfg, device=torch.device('cpu'), dumper=SimpleNamespace(base_dir=cfg.dump_dir))
def preprocess_input(input_json, **kwargs): return input_json
''')
    (root / 'runner' / 'inference.py').write_text('''\
import json
from pathlib import Path
def infer_predict(runner, cfg):
    job = json.loads(Path(cfg.input_json_path).read_text())[0]
    for seed in job['modelSeeds']:
        out = Path(cfg.dump_dir) / job['name'] / ('seed_' + str(seed)) / 'predictions'
        out.mkdir(parents=True)
        name = job['name']
        (out / (name + '_summary_confidence_sample_0.json')).write_text(json.dumps({'plddt': seed / 100.0}))
        (out / (name + '_sample_0.cif')).write_text(job['sequence'])
''')
    monkeypatch.setenv('FAKE_OPENDDE_LOADS', str(tmp_path / 'loads'))
    monkeypatch.delenv('ASTROLABE_EXECUTION_ENDPOINT', raising=False)
    monkeypatch.delenv('ASTROLABE_EXECUTION_TOKEN', raising=False)
    runner = OpenDDERunner(python_bin=sys.executable, repo_path=root, opendde_root_dir=assets,
                           load_checkpoint_path=str(assets / 'model.pt'), num_samples=1, execution='daemon')
    try:
        results = []
        for seed, sequence in ((11, 'AAA'), (22, 'CCC')):
            input_json = tmp_path / f'{seed}.json'
            input_json.write_text(json.dumps([{'name': 'job', 'modelSeeds': [seed], 'sequence': sequence}]))
            dest = runner.run_inference(input_json=input_json, out_dir=tmp_path / 'output')
            path = dest / 'job' / f'seed_{seed}' / 'predictions'
            assert (path / 'job_sample_0.cif').read_text() == sequence
            assert json.loads((path / 'job_summary_confidence_sample_0.json').read_text())['plddt'] == seed / 100.0
            prepared = OpenDDEPreparedInput(json_path=input_json, output_dir=dest, job_name='job', chain_order=[], protein_lengths={})
            scores = collect_opendde_outputs(prepared, structures_dir=tmp_path / ('structures-' + str(seed)))
            assert scores['plddt'] == [seed / 100.0]
            assert scores['complex_path'].name == 'job_model.cif'
            assert scores['complex_path'].read_text() == sequence
            results.append(dest)
        assert results[0] != results[1]
        assert (tmp_path / 'loads').read_text().splitlines() == ['load']
        assert runner._resident._client.hello['load_count'] == 1
    finally:
        if hasattr(runner, '_resident'): runner._resident.close()


def test_opendde_resident_rejects_distributed_and_broker(monkeypatch, tmp_path):
    from astrolabe.evaluators.opendde_resident import OpenDDEResident
    from astrolabe.evaluators.wrappers.opendde import OpenDDERunner
    runner = OpenDDERunner(python_bin=sys.executable, foldcp_mode='distributed')
    with pytest.raises(ValueError, match='single-device'):
        OpenDDEResident(runner)
    monkeypatch.setenv('ASTROLABE_EXECUTION_ENDPOINT', 'broker')
    monkeypatch.setenv('ASTROLABE_EXECUTION_TOKEN', 'token')
    with pytest.raises(ValueError, match='trajectory-exclusive'):
        OpenDDEResident(runner)
