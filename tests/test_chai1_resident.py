"""CPU-only Chai1 resident contract through the production evaluator and daemon."""
import json
from pathlib import Path
import pytest


def test_chai1_resident_component_cache_requests_and_fencing(tmp_path, monkeypatch):
    fake = tmp_path / 'chai_fake.py'
    fake.write_text('''\
import json, os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

def install(chai):
    class Component:
        def to(self, device): return self
    def load(key, device):
        with open(os.environ['CHAI_TEST_LOADS'], 'a') as f: f.write(key+'\\n')
        return SimpleNamespace(jit_module=Component())
    def infer(fasta_file, *, output_dir, constraint_path=None, msa_directory=None,
              cyclic_subchains=None, cyclic_relative_patches=None, use_esm_embeddings=True,
              num_trunk_recycles=3, num_diffn_timesteps=200, seed=None, device=None):
        with chai._component_moved_to('trunk.pt', torch.device(device)):
            text = Path(fasta_file).read_text()
            if 'FAIL' in text: raise RuntimeError('fixture inference failed')
            payload = {'fasta':text, 'constraint':Path(constraint_path).read_text() if constraint_path else None,
                       'cycles':cyclic_subchains, 'patches':cyclic_relative_patches,
                       'recycles':num_trunk_recycles, 'steps':num_diffn_timesteps, 'seed':seed}
            output_dir.mkdir(parents=True)
            cif=output_dir/'model.cif'; cif.write_text(json.dumps(payload))
            ptm=SimpleNamespace(complex_ptm=.7,interface_ptm=.6,per_chain_ptm=np.array([.7]),
                                per_chain_pair_iptm=np.array([[.6]]))
            rank=SimpleNamespace(aggregate_score=.5,ptm_scores=ptm,
                                 plddt_scores=SimpleNamespace(per_chain_plddt=np.array([.8])))
            return SimpleNamespace(cif_paths=[cif],ranking_data=[rank],pae=np.ones((1,1,1)),
                                   pde=np.ones((1,1,1)),plddt=np.array([[.8]]))
    chai.load_exported = load
    chai.run_inference = infer
''')
    (tmp_path / 'sitecustomize.py').write_text('''\
from chai_fake import install
import astrolabe.evaluators.chai1 as module
original = module._load_chai1_module
def patched():
    model = original()
    install(model)
    return model
module._load_chai1_module = patched
''')
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv('PYTHONPATH', str(tmp_path) + ':' + __import__('os').environ.get('PYTHONPATH', ''))
    monkeypatch.setenv('CHAI_TEST_LOADS', str(tmp_path/'loads'))
    monkeypatch.setenv('CHAI_DOWNLOADS_DIR', str(tmp_path/'downloads'))
    monkeypatch.delenv('ASTROLABE_EXECUTION_ENDPOINT', raising=False)
    monkeypatch.delenv('ASTROLABE_EXECUTION_TOKEN', raising=False)
    from astrolabe.evaluators.chai1_resident import COMPONENTS
    cache = tmp_path/'downloads'; (cache/'models_v2').mkdir(parents=True)
    (cache/'conformers_v1.apkl').write_bytes(b'conformers')
    for name in COMPONENTS: (cache/'models_v2'/name).write_bytes(name.encode())
    from astrolabe.evaluators.chai1 import Chai1Evaluator
    from astrolabe.runtime.client import WorkerError
    evaluator = Chai1Evaluator({'use_esm_embeddings':False, 'num_trunk_recycles':4,
                                'num_diffn_timesteps':17, 'seed':23, 'device':'cpu'}, tmp_path, execution='daemon')
    try:
        for i in range(2):
            fasta=tmp_path/f'{i}.fasta'; fasta.write_text(f'>protein|A\\n{"A"*(i+1)}\\n')
            constraint=tmp_path/f'{i}.restraints'; constraint.write_text(f'contact-{i}')
            output=tmp_path/f'output-{i}'; output.mkdir()
            result=evaluator.run_function({'A': {'fasta_file':fasta,'output_dir':output,
                'constraint_path':constraint, 'msa_directory':None, 'cyclic_subchains':['A'],
                'cyclic_relative_patches':None}})['A']
            record=json.loads(result.cif_paths[0].read_text())
            assert record['constraint'] == f'contact-{i}' and record['cycles'] == ['A']
            assert record['recycles'] == 4 and record['steps'] == 17 and record['seed'] == 23
            assert result.plddt.tolist() == [[.8]]
        assert evaluator._resident._client.hello['load_count'] == 1
        assert (tmp_path/'loads').read_text().splitlines() == ['trunk.pt']
        fasta=tmp_path/'fail.fasta'; fasta.write_text('FAIL')
        with pytest.raises(WorkerError):
            evaluator.run_function({'A': {'fasta_file':fasta, 'output_dir':tmp_path/'fail-out',
                                           'constraint_path':None, 'msa_directory':None}})
        with pytest.raises(WorkerError):
            evaluator.run_function({'A': {'fasta_file':fasta, 'output_dir':tmp_path/'retry-out',
                                           'constraint_path':None, 'msa_directory':None}})
        assert (tmp_path/'loads').read_text().splitlines() == ['trunk.pt']
    finally:
        evaluator.close()
